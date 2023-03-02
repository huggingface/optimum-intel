#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import importlib
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import numpy as np
from transformers import CLIPFeatureExtractor, CLIPTokenizer

import openvino
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME
from huggingface_hub import snapshot_download
from openvino._offline_transformations import compress_model_transformation
from openvino.offline_transformations import compress_model_transformation
from openvino.runtime import Core
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export_models, get_stable_diffusion_models_for_export
from optimum.pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from optimum.utils import (
    CONFIG_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)

from .modeling_base import OVBaseModel
from .utils import ONNX_WEIGHTS_NAME, OV_TO_NP_TYPE, OV_XML_FILE_NAME


core = Core()

logger = logging.getLogger(__name__)


class OVStableDiffusionPipeline(OVBaseModel, StableDiffusionPipelineMixin):

    auto_model_class = StableDiffusionPipeline
    config_name = "model_index.json"
    export_feature = "stable-diffusion"

    def __init__(
        self,
        vae_decoder: openvino.runtime.Model,
        text_encoder: openvino.runtime.Model,
        unet: openvino.runtime.Model,
        config: Dict[str, Any],
        tokenizer: "CLIPTokenizer",
        scheduler: Union["DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        compile: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self._internal_dict = config
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = ov_config if ov_config is not None else {}
        model_save_dir = model_save_dir.name if isinstance(model_save_dir, TemporaryDirectory) else model_save_dir

        self.vae_decoder = OVModelVaeDecoder(
            vae_decoder,
            self,
            {**self.ov_config, "CACHE_DIR": os.path.join(model_save_dir, DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER)},
        )
        self.text_encoder = OVModelTextEncoder(
            text_encoder,
            self,
            {**self.ov_config, "CACHE_DIR": os.path.join(model_save_dir, DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER)},
        )
        self.unet = OVModelUnet(
            unet,
            self,
            {**self.ov_config, "CACHE_DIR": os.path.join(model_save_dir, DIFFUSION_MODEL_UNET_SUBFOLDER)},
        )
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.vae_decoder_request = None
        self.text_encoder_request = None
        self.unet_request = None
        self.safety_checker = None
        self.preprocessors = []

        if self.is_dynamic:
            self.reshape(batch_size=-1, height=-1, width=-1, num_images_per_prompt=-1)

        if compile:
            self.compile()

        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
        }
        for name in sub_models.keys():
            self._internal_dict[name] = ("optimum", sub_models[name].__class__.__name__)
        self._internal_dict.pop("vae", None)

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        vae_decoder_file_name: str = OV_XML_FILE_NAME,
        text_encoder_file_name: str = OV_XML_FILE_NAME,
        unet_file_name: str = OV_XML_FILE_NAME,
        **kwargs,
    ):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            vae_decoder_file_name (`str`, defaults to `optimum.intel.utils.ONNX_WEIGHTS_NAME`):
                The VAE decoder model file name. Overwrites the default file name and allows one to save the VAE decoder model
                with a different name.
            text_encoder_file_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_WEIGHTS_NAME`):
                The text encoder model file name. Overwrites the default file name and allows one to save the text encoder model
                with a different name.
            unet_file_name (`str`, defaults to `optimum.onnxruntime.ONNX_WEIGHTS_NAME`):
                The U-NET model file name. Overwrites the default file name and allows one to save the U-NET model
                with a different name.
        """
        save_directory = Path(save_directory)
        src_to_dst_file = {
            self.vae_decoder.model: save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            self.text_encoder.model: save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            self.unet.model: save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
        }
        for src_file, dst_path in src_to_dst_file.items():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            openvino.runtime.serialize(src_file, str(dst_path), str(dst_path.with_suffix(".bin")))

        self.tokenizer.save_pretrained(save_directory.joinpath("tokenizer"))
        self.scheduler.save_pretrained(save_directory.joinpath("scheduler"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory.joinpath("feature_extractor"))

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        vae_decoder_file_name: Optional[str] = None,
        text_encoder_file_name: Optional[str] = None,
        unet_file_name: Optional[str] = None,
        local_files_only: bool = False,
        from_onnx: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        vae_decoder_file_name = vae_decoder_file_name or default_file_name
        text_encoder_file_name = text_encoder_file_name or default_file_name
        unet_file_name = unet_file_name or default_file_name
        model_id = str(model_id)
        sub_models_to_load, _, _ = cls.extract_init_dict(config)
        sub_models_names = set(sub_models_to_load.keys()).intersection({"feature_extractor", "tokenizer", "scheduler"})
        sub_models = {}

        if not os.path.isdir(model_id):
            allow_patterns = [os.path.join(k, "*") for k in config.keys() if not k.startswith("_")]
            allow_patterns += list(
                {
                    vae_decoder_file_name,
                    text_encoder_file_name,
                    unet_file_name,
                    vae_decoder_file_name.replace(".xml", ".bin"),
                    vae_decoder_file_name.replace(".xml", ".bin"),
                    unet_file_name.replace(".xml", ".bin"),
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                    cls.config_name,
                }
            )
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*pytorch_model.bin"],
            )
        new_model_save_dir = Path(model_id)

        for name in sub_models_names:
            library_name, library_classes = sub_models_to_load[name]
            if library_classes is not None:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (new_model_save_dir / name).is_dir():
                    sub_models[name] = load_method(new_model_save_dir / name)
                else:
                    sub_models[name] = load_method(new_model_save_dir)

        vae_decoder = cls.load_model(
            new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name
        )
        text_encoder = cls.load_model(
            new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name
        )
        unet = cls.load_model(new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name)

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            unet=unet,
            config=config,
            tokenizer=sub_models["tokenizer"],
            scheduler=sub_models["scheduler"],
            feature_extractor=sub_models.pop("feature_extractor", None),
            model_save_dir=model_save_dir,
            **kwargs,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        task: Optional[str] = None,
        **kwargs,
    ):
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        model = TasksManager.get_model_from_task(
            task,
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            config=config,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
        )

        output_names = [
            os.path.join(DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
            os.path.join(DIFFUSION_MODEL_UNET_SUBFOLDER, ONNX_WEIGHTS_NAME),
            os.path.join(DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
            os.path.join(DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
        ]
        models_and_onnx_configs = get_stable_diffusion_models_for_export(model)

        model.save_config(save_dir_path)
        model.tokenizer.save_pretrained(save_dir_path.joinpath("tokenizer"))
        model.scheduler.save_pretrained(save_dir_path.joinpath("scheduler"))
        if model.feature_extractor is not None:
            model.feature_extractor.save_pretrained(save_dir_path.joinpath("feature_extractor"))

        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=save_dir_path,
            output_names=output_names,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=True,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            model_save_dir=save_dir,  # important
            **kwargs,
        )

    def to(self, device: str):
        self._device = device.upper()
        self.clear_requests()
        return self

    @property
    def device(self) -> str:
        return self._device.lower()

    def _reshape_unet(
        self,
        model: openvino.runtime.Model,
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
    ):
        if batch_size == -1 or num_images_per_prompt == -1:
            batch_size = -1
        else:
            # The factor of 2 comes from the guidance scale > 1
            batch_size = 2 * batch_size * num_images_per_prompt

        height = height // 8 if height > 0 else height
        width = width // 8 if width > 0 else width
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if inputs.get_any_name().startswith("timestep"):
                shapes[inputs][0] = 1
            elif inputs.get_any_name().startswith("sample"):
                shapes[inputs] = [batch_size, 4, height, width]
            else:
                shapes[inputs][0] = batch_size
                shapes[inputs][1] = self.tokenizer.model_max_length
        model.reshape(shapes)
        return model

    def _reshape_text_encoder(self, model: openvino.runtime.Model, batch_size: int = -1):
        if batch_size != -1:
            shapes = {model.inputs[0]: [batch_size, self.tokenizer.model_max_length]}
            model.reshape(shapes)
        return model

    def _reshape_vae_decoder(self, model: openvino.runtime.Model, height: int = -1, width: int = -1):
        height = height // 8 if height > -1 else height
        width = width // 8 if width > -1 else width
        shapes = {model.inputs[0]: [1, 4, height, width]}
        model.reshape(shapes)
        return model

    def reshape(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_images_per_prompt: int = -1,
    ):
        self.is_dynamic = -1 in {batch_size, height, width, num_images_per_prompt}
        self.text_encoder.model = self._reshape_text_encoder(self.text_encoder.model, batch_size)
        self.vae_decoder.model = self._reshape_vae_decoder(self.vae_decoder.model, height, width)
        self.unet.model = self._reshape_unet(self.unet.model, batch_size, height, width, num_images_per_prompt)
        self.clear_requests()
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        compress_model_transformation(self.vae_decoder.model)
        compress_model_transformation(self.text_encoder.model)
        compress_model_transformation(self.unet.model)
        self.clear_requests()
        return self

    def clear_requests(self):
        self.text_encoder.request = None
        self.vae_decoder.request = None
        self.unet.request = None

    def compile(self):
        self.text_encoder._create_inference_request()
        self.vae_decoder._create_inference_request()
        self.unet._create_inference_request()

    def __call__(self, *args, **kwargs):
        guidance_scale = kwargs.get("guidance_scale", None)
        if guidance_scale is not None and guidance_scale <= 1 and not self.is_dynamic:
            raise ValueError(
                f"`guidance_scale` was set to {guidance_scale}, static shapes are only supported for `guidance_scale` > 1, "
                "please set `dynamic_shapes` to `True` when loading the model."
            )
        return StableDiffusionPipelineMixin.__call__(self, *args, **kwargs)

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)


class OVModelPart:
    def __init__(
        self, model: openvino.runtime.Model, parent_model: OVBaseModel, ov_config: Optional[Dict[str, str]] = None
    ):
        self.model = model
        self.parent_model = parent_model
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtype = {
            inputs.get_any_name(): OV_TO_NP_TYPE[inputs.get_element_type().get_type_name()]
            for inputs in self.model.inputs
        }
        self.ov_config = ov_config or self.parent_model.ov_config
        self.request = None

    def _create_inference_request(self):
        if self.request is None:
            logger.info("Compiling the encoder and creating the inference request ...")
            compiled_model = core.compile_model(self.model, self.device, self.ov_config)
            self.request = compiled_model.create_infer_request()

    @property
    def device(self):
        return self.parent_model._device


class OVModelTextEncoder(OVModelPart):
    def __call__(self, input_ids: np.ndarray):

        self._create_inference_request()

        inputs = {
            "input_ids": input_ids,
        }
        outputs = self.request.infer(inputs)
        return list(outputs.values())


class OVModelUnet(OVModelPart):
    def __call__(self, sample: np.ndarray, timestep: np.ndarray, encoder_hidden_states: np.ndarray):

        self._create_inference_request()

        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        outputs = self.request.infer(inputs)
        return list(outputs.values())


class OVModelVaeDecoder(OVModelPart):
    def __call__(self, latent_sample: np.ndarray):

        self._create_inference_request()

        inputs = {
            "latent_sample": latent_sample,
        }
        outputs = self.request.infer(inputs)
        return list(outputs.values())
