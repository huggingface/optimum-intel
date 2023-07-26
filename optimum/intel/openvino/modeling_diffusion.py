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
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openvino
from diffusers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME
from huggingface_hub import snapshot_download
from openvino._offline_transformations import compress_model_transformation
from openvino.runtime import Core
from transformers import CLIPFeatureExtractor, CLIPTokenizer

from optimum.exporters.onnx import main_export
from optimum.pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)

from .modeling_base import OVBaseModel
from .utils import ONNX_WEIGHTS_NAME, OV_TO_NP_TYPE, OV_XML_FILE_NAME


core = Core()

logger = logging.getLogger(__name__)


class OVStableDiffusionPipelineBase(OVBaseModel):
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
        vae_encoder: Optional[openvino.runtime.Model] = None,
        text_encoder_2: Optional[openvino.runtime.Model] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
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
        self._model_save_dir = (
            Path(model_save_dir.name) if isinstance(model_save_dir, TemporaryDirectory) else model_save_dir
        )
        self.vae_decoder = OVModelVaeDecoder(vae_decoder, self)
        self.unet = OVModelUnet(unet, self)
        self.text_encoder = OVModelTextEncoder(text_encoder, self) if text_encoder is not None else None
        self.text_encoder_2 = (
            OVModelTextEncoder(text_encoder_2, self, model_name=DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER)
            if text_encoder_2 is not None
            else None
        )
        self.vae_encoder = OVModelVaeEncoder(vae_encoder, self) if vae_encoder is not None else None

        if "block_out_channels" in self.vae_decoder.config:
            self.vae_scale_factor = 2 ** (len(self.vae_decoder.config["block_out_channels"]) - 1)
        else:
            self.vae_scale_factor = 8

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
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
            DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER: self.vae_encoder,
            DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER: self.text_encoder_2,
        }
        for name in sub_models.keys():
            self._internal_dict[name] = (
                ("optimum", sub_models[name].__class__.__name__) if sub_models[name] is not None else (None, None)
            )

        self._internal_dict.pop("vae", None)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files
        """
        save_directory = Path(save_directory)

        sub_models_to_save = {
            self.unet: DIFFUSION_MODEL_UNET_SUBFOLDER,
            self.vae_decoder: DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            self.vae_encoder: DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            self.text_encoder: DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
            self.text_encoder_2: DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
        }

        for ov_model, dst_path in sub_models_to_save.items():
            if ov_model is not None:
                dst_path = save_directory / dst_path / OV_XML_FILE_NAME
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                openvino.runtime.serialize(ov_model.model, dst_path)
                model_dir = ov_model.config.get("_name_or_path", None) or ov_model._model_dir / ov_model._model_name
                config_path = Path(model_dir) / ov_model.CONFIG_NAME
                if config_path.is_file():
                    shutil.copyfile(config_path, dst_path.parent / ov_model.CONFIG_NAME)

        self.scheduler.save_pretrained(save_directory / "scheduler")
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory / "feature_extractor")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory / "tokenizer_2")

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
        vae_encoder_file_name: Optional[str] = None,
        text_encoder_2_file_name: Optional[str] = None,
        local_files_only: bool = False,
        from_onnx: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        vae_decoder_file_name = vae_decoder_file_name or default_file_name
        text_encoder_file_name = text_encoder_file_name or default_file_name
        text_encoder_2_file_name = text_encoder_2_file_name or default_file_name
        unet_file_name = unet_file_name or default_file_name
        vae_encoder_file_name = vae_encoder_file_name or default_file_name
        model_id = str(model_id)
        patterns = set(config.keys())
        sub_models_names = patterns.intersection({"feature_extractor", "tokenizer", "tokenizer_2", "scheduler"})

        if not os.path.isdir(model_id):
            patterns.update({"vae_encoder", "vae_decoder"})
            allow_patterns = {os.path.join(k, "*") for k in patterns if not k.startswith("_")}
            allow_patterns.update(
                {
                    vae_decoder_file_name,
                    text_encoder_file_name,
                    text_encoder_2_file_name,
                    unet_file_name,
                    vae_encoder_file_name,
                    vae_decoder_file_name.replace(".xml", ".bin"),
                    text_encoder_file_name.replace(".xml", ".bin"),
                    text_encoder_2_file_name.replace(".xml", ".bin"),
                    unet_file_name.replace(".xml", ".bin"),
                    vae_encoder_file_name.replace(".xml", ".bin"),
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
            # Check if the subcomponent needs to be loaded
            if kwargs.get(name, None) is not None:
                continue
            library_name, library_classes = config[name]
            if library_classes is not None:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (new_model_save_dir / name).is_dir():
                    kwargs[name] = load_method(new_model_save_dir / name)
                else:
                    kwargs[name] = load_method(new_model_save_dir)

        unet = cls.load_model(new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name)

        components = {
            "vae_encoder": new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            "vae_decoder": new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "text_encoder": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "text_encoder_2": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name,
        }

        for key, value in components.items():
            components[key] = cls.load_model(value) if value.is_file() else None

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            vae_decoder=components["vae_decoder"],
            text_encoder=components["text_encoder"],
            unet=unet,
            config=config,
            tokenizer=kwargs.pop("tokenizer", None),
            scheduler=kwargs.pop("scheduler"),
            feature_extractor=kwargs.pop("feature_extractor", None),
            vae_encoder=components["vae_encoder"],
            text_encoder_2=components["text_encoder_2"],
            tokenizer_2=kwargs.pop("tokenizer_2", None),
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
        tokenizer: "CLIPTokenizer" = None,
        scheduler: Union["DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        **kwargs,
    ):
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=cls.export_feature,
            do_validation=False,
            no_post_process=True,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
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
            model_save_dir=save_dir,
            tokenizer=tokenizer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            **kwargs,
        )

    def to(self, device: str):
        self._device = device.upper()
        self.clear_requests()
        return self

    @property
    def device(self) -> str:
        return self._device.lower()

    @property
    def height(self) -> int:
        height = self.unet.model.inputs[0].get_partial_shape()[2]
        if height.is_dynamic:
            return -1
        return height.get_length() * self.vae_scale_factor

    @property
    def width(self) -> int:
        width = self.unet.model.inputs[0].get_partial_shape()[3]
        if width.is_dynamic:
            return -1
        return width.get_length() * self.vae_scale_factor

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

        requires_aesthetics_score = getattr(self.config, "requires_aesthetics_score", False)
        height = height // self.vae_scale_factor if height > 0 else height
        width = width // self.vae_scale_factor if width > 0 else width
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if inputs.get_any_name() == "timestep":
                shapes[inputs][0] = 1
            elif inputs.get_any_name() == "sample":
                in_channels = self.unet.config.get("in_channels", None)
                if in_channels is None:
                    in_channels = shapes[inputs][1]
                    if in_channels.is_dynamic:
                        logger.warning(
                            "Could not identify `in_channels` from the unet configuration, to statically reshape the unet please provide a configuration."
                        )
                        self.is_dynamic = True

                shapes[inputs] = [batch_size, in_channels, height, width]
            elif inputs.get_any_name() == "text_embeds":
                shapes[inputs] = [batch_size, self.text_encoder_2.config["projection_dim"]]
            elif inputs.get_any_name() == "time_ids":
                shapes[inputs] = [batch_size, 5 if requires_aesthetics_score else 6]
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
        height = height // self.vae_scale_factor if height > -1 else height
        width = width // self.vae_scale_factor if width > -1 else width
        latent_channels = self.vae_decoder.config.get("latent_channels", None)
        if latent_channels is None:
            latent_channels = model.inputs[0].get_partial_shape()[1]
            if latent_channels.is_dynamic:
                logger.warning(
                    "Could not identify `latent_channels` from the VAE decoder configuration, to statically reshape the VAE decoder please provide a configuration."
                )
                self.is_dynamic = True
        shapes = {model.inputs[0]: [1, latent_channels, height, width]}
        model.reshape(shapes)
        return model

    def _reshape_vae_encoder(
        self, model: openvino.runtime.Model, batch_size: int = -1, height: int = -1, width: int = -1
    ):
        in_channels = self.vae_encoder.config.get("in_channels", None)
        if in_channels is None:
            in_channels = model.inputs[0].get_partial_shape()[1]
            if in_channels.is_dynamic:
                logger.warning(
                    "Could not identify `in_channels` from the VAE encoder configuration, to statically reshape the VAE encoder please provide a configuration."
                )
                self.is_dynamic = True
        shapes = {model.inputs[0]: [batch_size, in_channels, height, width]}
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
        self.vae_decoder.model = self._reshape_vae_decoder(self.vae_decoder.model, height, width)
        self.unet.model = self._reshape_unet(self.unet.model, batch_size, height, width, num_images_per_prompt)

        if self.text_encoder is not None:
            self.text_encoder.model = self._reshape_text_encoder(self.text_encoder.model, batch_size)

        if self.text_encoder_2 is not None:
            self.text_encoder_2.model = self._reshape_text_encoder(self.text_encoder_2.model, batch_size)

        if self.vae_encoder is not None:
            self.vae_encoder.model = self._reshape_vae_encoder(self.vae_encoder.model, batch_size, height, width)

        self.clear_requests()
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        compress_model_transformation(self.vae_decoder.model)
        compress_model_transformation(self.unet.model)
        for component in {self.text_encoder, self.text_encoder_2, self.vae_encoder}:
            if component is not None:
                compress_model_transformation(component.model)
        self.clear_requests()
        return self

    def clear_requests(self):
        self.vae_decoder.request = None
        self.unet.request = None
        for component in {self.text_encoder, self.text_encoder_2, self.vae_encoder}:
            if component is not None:
                component.request = None

    def compile(self):
        self.vae_decoder._compile()
        self.unet._compile()
        for component in {self.text_encoder, self.text_encoder_2, self.vae_encoder}:
            if component is not None:
                component._compile()

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)


class OVModelPart:
    CONFIG_NAME = "config.json"

    def __init__(
        self,
        model: openvino.runtime.Model,
        parent_model: OVBaseModel,
        ov_config: Optional[Dict[str, str]] = None,
        model_name: str = "encoder",
        model_dir: str = None,
    ):
        self.model = model
        self.parent_model = parent_model
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtype = {
            inputs.get_any_name(): OV_TO_NP_TYPE[inputs.get_element_type().get_type_name()]
            for inputs in self.model.inputs
        }
        self.ov_config = ov_config or {**self.parent_model.ov_config}
        self.request = None
        self._model_name = model_name
        self._model_dir = Path(model_dir or parent_model._model_save_dir)
        config_path = self._model_dir / model_name / self.CONFIG_NAME
        self.config = self.parent_model._dict_from_json_file(config_path) if config_path.is_file() else {}

        # TODO : disable if self._model_dir tmp directory
        if "CACHE_DIR" not in self.ov_config:
            self.ov_config["CACHE_DIR"] = os.path.join(self._model_dir, self._model_name)

    def _compile(self):
        if self.request is None:
            logger.info(f"Compiling the {self._model_name}...")
            self.request = core.compile_model(self.model, self.device, self.ov_config)

    @property
    def device(self):
        return self.parent_model._device


class OVModelTextEncoder(OVModelPart):
    def __init__(
        self,
        model: openvino.runtime.Model,
        parent_model: OVBaseModel,
        ov_config: Optional[Dict[str, str]] = None,
        model_name: str = "text_encoder",
    ):
        super().__init__(model, parent_model, ov_config, model_name)

    def __call__(self, input_ids: np.ndarray):
        self._compile()

        inputs = {
            "input_ids": input_ids,
        }
        outputs = self.request(inputs, shared_memory=True)
        return list(outputs.values())


class OVModelUnet(OVModelPart):
    def __init__(
        self, model: openvino.runtime.Model, parent_model: OVBaseModel, ov_config: Optional[Dict[str, str]] = None
    ):
        super().__init__(model, parent_model, ov_config, "unet")

    def __call__(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        text_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
    ):
        self._compile()

        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if text_embeds is not None:
            inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            inputs["time_ids"] = time_ids

        outputs = self.request(inputs, shared_memory=True)
        return list(outputs.values())


class OVModelVaeDecoder(OVModelPart):
    def __init__(
        self, model: openvino.runtime.Model, parent_model: OVBaseModel, ov_config: Optional[Dict[str, str]] = None
    ):
        super().__init__(model, parent_model, ov_config, "vae_decoder")

    def __call__(self, latent_sample: np.ndarray):
        self._compile()

        inputs = {
            "latent_sample": latent_sample,
        }
        outputs = self.request(inputs, shared_memory=True)
        return list(outputs.values())


class OVModelVaeEncoder(OVModelPart):
    def __init__(
        self, model: openvino.runtime.Model, parent_model: OVBaseModel, ov_config: Optional[Dict[str, str]] = None
    ):
        super().__init__(model, parent_model, ov_config, "vae_encoder")

    def __call__(self, sample: np.ndarray):
        self._compile()

        inputs = {
            "sample": sample,
        }
        outputs = self.request(inputs, shared_memory=True)
        return list(outputs.values())


class OVStableDiffusionPipeline(OVStableDiffusionPipelineBase, StableDiffusionPipelineMixin):
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        height = height or self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        width = width or self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        _height = self.height
        _width = self.width

        if _height != -1 and height != _height:
            logger.warning(
                f"`height` was set to {height} but the static model will output images of height {_height}."
                "To fix the height, please reshape your model accordingly using the `.reshape()` method."
            )
            height = _height

        if _width != -1 and width != _width:
            logger.warning(
                f"`width` was set to {width} but the static model will output images of width {_width}."
                "To fix the width, please reshape your model accordingly using the `.reshape()` method."
            )
            width = _width

        if guidance_scale is not None and guidance_scale <= 1 and not self.is_dynamic:
            raise ValueError(
                f"`guidance_scale` was set to {guidance_scale}, static shapes are only supported for `guidance_scale` > 1, "
                "please set `dynamic_shapes` to `True` when loading the model."
            )

        return StableDiffusionPipelineMixin.__call__(
            self,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )


class OVStableDiffusionImg2ImgPipeline(OVStableDiffusionPipelineBase, StableDiffusionImg2ImgPipelineMixin):
    def __call__(self, *args, **kwargs):
        # TODO : add default height and width if model statically reshaped
        # resize image if doesn't match height and width given during reshaping
        return StableDiffusionImg2ImgPipelineMixin.__call__(self, *args, **kwargs)


class OVStableDiffusionInpaintPipeline(OVStableDiffusionPipelineBase, StableDiffusionInpaintPipelineMixin):
    def __call__(self, *args, **kwargs):
        # TODO : add default height and width if model statically reshaped
        return StableDiffusionInpaintPipelineMixin.__call__(self, *args, **kwargs)


class OVStableDiffusionXLPipelineBase(OVStableDiffusionPipelineBase):
    auto_model_class = StableDiffusionXLPipeline
    export_feature = "stable-diffusion-xl"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # additional invisible-watermark dependency for SD XL
        from optimum.pipelines.diffusers.watermark import StableDiffusionXLWatermarker

        self.watermark = StableDiffusionXLWatermarker()


class OVStableDiffusionXLPipeline(OVStableDiffusionXLPipelineBase, StableDiffusionXLPipelineMixin):
    def __call__(self, *args, **kwargs):
        return StableDiffusionXLPipelineMixin.__call__(self, *args, **kwargs)


class OVStableDiffusionXLImg2ImgPipeline(OVStableDiffusionXLPipelineBase, StableDiffusionXLImg2ImgPipelineMixin):
    def __call__(self, *args, **kwargs):
        return StableDiffusionXLImg2ImgPipelineMixin.__call__(self, *args, **kwargs)
