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
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openvino
import PIL
from diffusers import (
    ConfigMixin,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, is_invisible_watermark_available
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino._offline_transformations import compress_model_transformation
from openvino.runtime import Core
import openvino.runtime
from transformers import CLIPFeatureExtractor, CLIPTokenizer

from optimum.pipelines.diffusers.pipeline_latent_consistency import LatentConsistencyPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin
from optimum.pipelines.diffusers.pipeline_utils import VaeImageProcessor
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)

from ...exporters.openvino import main_export
from .configuration import OVConfig, OVQuantizationMethod, OVWeightQuantizationConfig
from .loaders import OVTextualInversionLoaderMixin
from .modeling_base import OVBaseModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    OV_TO_NP_TYPE,
    OV_XML_FILE_NAME,
    _print_compiled_model_properties,
)

import torch
from typing import Union, List, Optional, Tuple
from tqdm.auto import tqdm
from diffusers.utils import numpy_to_pil

core = Core()

logger = logging.getLogger(__name__)


class OVStableDiffusionPipelineBase(OVBaseModel, OVTextualInversionLoaderMixin):
    auto_model_class = StableDiffusionPipeline
    config_name = "model_index.json"
    export_feature = "stable-diffusion"

    def __init__(
        self,
        unet: openvino.runtime.Model,
        config: Dict[str, Any],
        scheduler: Union["DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
        vae_decoder: Optional[openvino.runtime.Model] = None,
        vae_encoder: Optional[openvino.runtime.Model] = None,
        text_encoder: Optional[openvino.runtime.Model] = None,
        text_encoder_2: Optional[openvino.runtime.Model] = None,
        tokenizer: Optional["CLIPTokenizer"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        compile: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        self._internal_dict = config
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting
        # would end-up removing the directory containing the underlying OpenVINO model
        self._model_save_dir_tempdirectory_instance = None
        if isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self._model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self._model_save_dir = Path(model_save_dir)
        else:
            self._model_save_dir = model_save_dir

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

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        self.preprocessors = []

        if self.is_dynamic:
            self.reshape(batch_size=-1, height=-1, width=-1, num_images_per_prompt=-1)

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

        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()

        if compile:
            self.compile()

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
                openvino.save_model(ov_model.model, dst_path, compress_to_fp16=False)
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

        self._save_openvino_config(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        vae_decoder_file_name: Optional[str] = None,
        text_encoder_file_name: Optional[str] = None,
        unet_file_name: Optional[str] = None,
        vae_encoder_file_name: Optional[str] = None,
        text_encoder_2_file_name: Optional[str] = None,
        local_files_only: bool = False,
        from_onnx: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

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
            ignore_patterns = ["*.msgpack", "*.safetensors", "*pytorch_model.bin"]
            if not from_onnx:
                ignore_patterns.extend(["*.onnx", "*.onnx_data"])
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
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

        unet_path = new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name
        components = {
            "vae_encoder": new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            "vae_decoder": new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "text_encoder": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "text_encoder_2": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name,
        }

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)
        if quantization_config is None or quantization_config.dataset is None:
            unet = cls.load_model(unet_path, quantization_config)
            for key, value in components.items():
                components[key] = cls.load_model(value, quantization_config) if value.is_file() else None
        else:
            # Load uncompressed models to apply hybrid quantization further
            unet = cls.load_model(unet_path)
            for key, value in components.items():
                components[key] = cls.load_model(value) if value.is_file() else None
            sd_model = cls(unet=unet, config=config, model_save_dir=model_save_dir, **components, **kwargs)

            supported_pipelines = (
                OVStableDiffusionPipeline,
                OVStableDiffusionXLPipeline,
                OVLatentConsistencyModelPipeline,
            )
            if not isinstance(sd_model, supported_pipelines):
                raise NotImplementedError(f"Quantization in hybrid mode is not supported for {cls.__name__}")

            from optimum.intel import OVQuantizer

            hybrid_quantization_config = deepcopy(quantization_config)
            hybrid_quantization_config.quant_method = OVQuantizationMethod.HYBRID
            quantizer = OVQuantizer(sd_model)
            quantizer.quantize(ov_config=OVConfig(quantization_config=hybrid_quantization_config))

            return sd_model

        return cls(
            unet=unet,
            config=config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **components,
            **kwargs,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        tokenizer: Optional["CLIPTokenizer"] = None,
        scheduler: Union["DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")
        
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=cls.export_feature,
            do_validation=False,
            no_post_process=True,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            ov_config=ov_config,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=False,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            model_save_dir=save_dir,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()
        else:
            logger.debug(f"device must be of type {str} but got {type(device)} instead")

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

    @property
    def _batch_size(self) -> int:
        batch_size = self.unet.model.inputs[0].get_partial_shape()[0]
        if batch_size.is_dynamic:
            return -1
        return batch_size.get_length()

    def _reshape_unet(
        self,
        model: openvino.runtime.Model,
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
        tokenizer_max_length: int = -1,
    ):
        if batch_size == -1 or num_images_per_prompt == -1:
            batch_size = -1
        else:
            batch_size *= num_images_per_prompt
            # The factor of 2 comes from the guidance scale > 1
            if "timestep_cond" not in {inputs.get_any_name() for inputs in model.inputs}:
                batch_size *= 2

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
                shapes[inputs] = [batch_size, inputs.get_partial_shape()[1]]
            elif inputs.get_any_name() == "timestep_cond":
                shapes[inputs] = [batch_size, self.unet.config["time_cond_proj_dim"]]
            else:
                shapes[inputs][0] = batch_size
                shapes[inputs][1] = tokenizer_max_length
        model.reshape(shapes)
        return model

    def _reshape_text_encoder(
        self, model: openvino.runtime.Model, batch_size: int = -1, tokenizer_max_length: int = -1
    ):
        if batch_size != -1:
            shapes = {model.inputs[0]: [batch_size, tokenizer_max_length]}
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
        if self.tokenizer is None and self.tokenizer_2 is None:
            tokenizer_max_len = -1
        else:
            tokenizer_max_len = (
                self.tokenizer.model_max_length if self.tokenizer is not None else self.tokenizer_2.model_max_length
            )
        self.unet.model = self._reshape_unet(
            self.unet.model, batch_size, height, width, num_images_per_prompt, tokenizer_max_len
        )

        if self.text_encoder is not None:
            self.text_encoder.model = self._reshape_text_encoder(
                self.text_encoder.model, batch_size, self.tokenizer.model_max_length
            )

        if self.text_encoder_2 is not None:
            self.text_encoder_2.model = self._reshape_text_encoder(
                self.text_encoder_2.model, batch_size, self.tokenizer_2.model_max_length
            )

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

    def _compile(self):
        if self.request is None:
            if (
                "CACHE_DIR" not in self.ov_config.keys()
                and not str(self._model_dir).startswith(gettempdir())
                and "gpu" in self.device.lower()
            ):
                self.ov_config["CACHE_DIR"] = os.path.join(self._model_dir, self._model_name, "model_cache")

            logger.info(f"Compiling the {self._model_name} to {self.device} ...")
            self.request = core.compile_model(self.model, self.device, self.ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                logger.info(f"{self.device} SUPPORTED_PROPERTIES:")
                _print_compiled_model_properties(self.request)

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
        outputs = self.request(inputs, share_inputs=True)
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
        timestep_cond: Optional[np.ndarray] = None,
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
        if timestep_cond is not None:
            inputs["timestep_cond"] = timestep_cond

        outputs = self.request(inputs, share_inputs=True)
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
        outputs = self.request(inputs, share_inputs=True)
        return list(outputs.values())

    def _compile(self):
        if "GPU" in self.device:
            self.ov_config.update({"INFERENCE_PRECISION_HINT": "f32"})
        super()._compile()


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
        outputs = self.request(inputs, share_inputs=True)
        return list(outputs.values())

    def _compile(self):
        if "GPU" in self.device:
            self.ov_config.update({"INFERENCE_PRECISION_HINT": "f32"})
        super()._compile()


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
        expected_batch_size = self._batch_size

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

        if expected_batch_size != -1:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = kwargs.get("prompt_embeds").shape[0]

            _raise_invalid_batch_size(expected_batch_size, batch_size, num_images_per_prompt, guidance_scale)

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
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Union[np.ndarray, PIL.Image.Image] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        _height = self.height
        _width = self.width
        expected_batch_size = self._batch_size

        if _height != -1 and _width != -1:
            image = self.image_processor.preprocess(image, height=_height, width=_width).transpose(0, 2, 3, 1)

        if expected_batch_size != -1:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = kwargs.get("prompt_embeds").shape[0]

            _raise_invalid_batch_size(expected_batch_size, batch_size, num_images_per_prompt, guidance_scale)

        return StableDiffusionImg2ImgPipelineMixin.__call__(
            self,
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )


class OVStableDiffusionInpaintPipeline(OVStableDiffusionPipelineBase, StableDiffusionInpaintPipelineMixin):
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]],
        image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
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
        expected_batch_size = self._batch_size

        if _height != -1 and _width != -1:
            if height != _height:
                logger.warning(
                    f"`height` was set to {height} but the static model will output images of height {_height}."
                    "To fix the height, please reshape your model accordingly using the `.reshape()` method."
                )
                height = _height

            if width != _width:
                logger.warning(
                    f"`width` was set to {width} but the static model will output images of width {_width}."
                    "To fix the width, please reshape your model accordingly using the `.reshape()` method."
                )
                width = _width

            if isinstance(image, list):
                image = [self.image_processor.resize(i, _height, _width) for i in image]
            else:
                image = self.image_processor.resize(image, _height, _width)

            if isinstance(mask_image, list):
                mask_image = [self.image_processor.resize(i, _height, _width) for i in mask_image]
            else:
                mask_image = self.image_processor.resize(mask_image, _height, _width)

        if expected_batch_size != -1:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = kwargs.get("prompt_embeds").shape[0]

            _raise_invalid_batch_size(expected_batch_size, batch_size, num_images_per_prompt, guidance_scale)

        return StableDiffusionInpaintPipelineMixin.__call__(
            self,
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )


class OVStableDiffusionXLPipelineBase(OVStableDiffusionPipelineBase):
    auto_model_class = StableDiffusionXLPipeline
    export_feature = "stable-diffusion-xl"

    def __init__(self, *args, add_watermarker: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            if not is_invisible_watermark_available():
                raise ImportError(
                    "`add_watermarker` requires invisible-watermark to be installed, which can be installed with `pip install invisible-watermark`."
                )
            from optimum.pipelines.diffusers.watermark import StableDiffusionXLWatermarker

            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None


class OVStableDiffusionXLPipeline(OVStableDiffusionXLPipelineBase, StableDiffusionXLPipelineMixin):
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        height = height or self.unet.config["sample_size"] * self.vae_scale_factor
        width = width or self.unet.config["sample_size"] * self.vae_scale_factor
        _height = self.height
        _width = self.width
        expected_batch_size = self._batch_size

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

        if expected_batch_size != -1:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = kwargs.get("prompt_embeds").shape[0]

            _raise_invalid_batch_size(expected_batch_size, batch_size, num_images_per_prompt, guidance_scale)

        return StableDiffusionXLPipelineMixin.__call__(
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


class OVStableDiffusionXLImg2ImgPipeline(OVStableDiffusionXLPipelineBase, StableDiffusionXLImg2ImgPipelineMixin):
    auto_model_class = StableDiffusionXLImg2ImgPipeline

    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Union[np.ndarray, PIL.Image.Image] = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        _height = self.height
        _width = self.width
        expected_batch_size = self._batch_size

        if _height != -1 and _width != -1:
            image = self.image_processor.preprocess(image, height=_height, width=_width).transpose(0, 2, 3, 1)

        if expected_batch_size != -1:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = kwargs.get("prompt_embeds").shape[0]

            _raise_invalid_batch_size(expected_batch_size, batch_size, num_images_per_prompt, guidance_scale)

        return StableDiffusionXLImg2ImgPipelineMixin.__call__(
            self,
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )


class OVLatentConsistencyModelPipeline(OVStableDiffusionPipelineBase, LatentConsistencyPipelineMixin):
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        original_inference_steps: int = None,
        guidance_scale: float = 8.5,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        height = height or self.unet.config["sample_size"] * self.vae_scale_factor
        width = width or self.unet.config["sample_size"] * self.vae_scale_factor
        _height = self.height
        _width = self.width
        expected_batch_size = self._batch_size

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

        if expected_batch_size != -1:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = kwargs.get("prompt_embeds").shape[0]

            _raise_invalid_batch_size(expected_batch_size, batch_size, num_images_per_prompt, guidance_scale=0.0)

        return LatentConsistencyPipelineMixin.__call__(
            self,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            original_inference_steps=original_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )


def _raise_invalid_batch_size(
    expected_batch_size: int, batch_size: int, num_images_per_prompt: int, guidance_scale: float
):
    current_batch_size = batch_size * num_images_per_prompt * (1 if guidance_scale <= 1 else 2)

    if expected_batch_size != current_batch_size:
        msg = ""
        if guidance_scale is not None and guidance_scale <= 1:
            msg = f"`guidance_scale` was set to {guidance_scale}, static shapes are currently only supported for `guidance_scale` > 1 "

        raise ValueError(
            "The model was statically reshaped and the pipeline inputs do not match the expected shapes. "
            f"The `batch_size`, `num_images_per_prompt` and `guidance_scale` were respectively set to {batch_size}, {num_images_per_prompt} and {guidance_scale}. "
            f"The static model expects an input of size equal to {expected_batch_size} and got the following value instead : {current_batch_size}. "
            f"To fix this, please either provide a different inputs to your model so that `batch_size` * `num_images_per_prompt` * 2 is equal to {expected_batch_size} "
            "or reshape it again accordingly using the `.reshape()` method by setting `batch_size` to -1. " + msg
        )

class OVModelControlNet(OVModelPart):
    def __init__(
        self, model: openvino.runtime.Model, parent_model: OVBaseModel, ov_config: Optional[Dict[str, str]] = None
    ):
        super().__init__(model, parent_model, ov_config, "unet")

    def __call__(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        controlnet_cond: np.ndarray,
        text_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
        timestep_cond: Optional[np.ndarray] = None,
    ):
        self._compile()

        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
        }

        if text_embeds is not None:
            inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            inputs["time_ids"] = time_ids
        if timestep_cond is not None:
            inputs["timestep_cond"] = timestep_cond

        outputs = self.request(inputs, share_inputs=True)
        return list(outputs.values())
    
class OVModelUnetControlNet(OVModelPart):
    def __init__(
        self, model: openvino.runtime.Model, parent_model: OVBaseModel, ov_config: Optional[Dict[str, str]] = None
    ):
        super().__init__(model, parent_model, ov_config, "unet")

    def __call__(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        down_and_mid_block_samples: Tuple[np.ndarray],
        text_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
        timestep_cond: Optional[np.ndarray] = None,
    ):
        self._compile()

        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "mid_block_additional_residual": down_and_mid_block_samples[-1],
        }
        a = 1
        for block in down_and_mid_block_samples:
            if a == 23:
                inputs[f"down_block_additional_residual"] = block
                break
            else:
                inputs[f"down_block_additional_residual.{a}"] = block
            a += 2

        if text_embeds is not None:
            inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            inputs["time_ids"] = time_ids
        if timestep_cond is not None:
            inputs["timestep_cond"] = timestep_cond

        outputs = self.request(inputs, share_inputs=True)
        return list(outputs.values())


# class OVControlNetModel(OVBaseModel):
#         export_feature = "stable-diffusion-controlnet"

#         @classmethod
#         def _from_transformers(        
#             cls,
#             model_id: str,
#             config: Dict[str, Any],
#             token: Optional[Union[bool, str]] = None,
#             revision: Optional[str] = None,
#             force_download: bool = False,
#             cache_dir: str = HUGGINGFACE_HUB_CACHE,
#             local_files_only: bool = False,
#             **kwargs,
#         ):
#             ov_config = OVConfig(dtype="fp32")
#             save_dir = kwargs.get("save_dir")
#             save_dir_path = Path(save_dir)
#             main_export(
#                 model_name_or_path=model_id,
#                 output=save_dir_path,
#                 task=cls.export_feature,
#                 do_validation=False,
#                 no_post_process=True,
#                 revision=revision,
#                 cache_dir=cache_dir,
#                 token=token,
#                 local_files_only=local_files_only,
#                 force_download=force_download,
#                 ov_config=ov_config,
#             )


class OVStableDiffusionControlNetPipelineBase(OVStableDiffusionPipelineBase):
    """
    OpenVINO inference pipeline for Stable Diffusion with ControlNet guidence
    """
    auto_model_class = StableDiffusionControlNetPipeline
    export_feature = "stable-diffusion-controlnet"
    
    def __init__(
        self,
        unet: openvino.runtime.Model,
        controlnet: openvino.runtime.Model,
        scheduler: Union[None, "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
        text_encoder: Optional[openvino.runtime.Model] = None,
        vae_decoder: Optional[openvino.runtime.Model] = None,
        tokenizer: Optional["CLIPTokenizer"] = None,
        device: str = "CPU",
        dynamic_shapes: bool = False,
        compile: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        # super().__init__()
        self._device = device.upper()
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.is_dynamic = dynamic_shapes

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting
        # would end-up removing the directory containing the underlying OpenVINO model
        self._model_save_dir_tempdirectory_instance = None
        if isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self._model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self._model_save_dir = Path(model_save_dir)
        else:
            self._model_save_dir = model_save_dir

        self.vae_decoder = OVModelVaeDecoder(vae_decoder, self)
        self.unet = OVModelUnetControlNet(unet, self)
        self.controlnet = OVModelControlNet(controlnet, self)
        self.text_encoder = OVModelTextEncoder(text_encoder, self) if text_encoder is not None else None
        
        self.vae_scale_factor = 8

        self.scheduler = scheduler


        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer = tokenizer
    
    def export_controlnet(
        model_id: str,
        save_dir_path: Optional[Union[str, Path]] = None,
    ):
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float32)
        controlnet.eval()
        dummy_inputs = {
            "sample": torch.randn((2, 4, 64, 64)),
            "timestep": torch.tensor(1),
            "encoder_hidden_states": torch.randn((2, 77, 768)),
            "controlnet_cond": torch.randn((2, 3, 512, 512)),
        }
        input_info = []
        for name, inp in dummy_inputs.items():
            shape = openvino.PartialShape(inp.shape)
            # element_type = dtype_mapping[input_tensor.dtype]
            if len(shape) == 4:
                shape[0] = -1
                shape[2] = -1
                shape[3] = -1
            elif len(shape) == 3:
                shape[0] = -1
            input_info.append((shape))

        CONTROLNET_OV_PATH = save_dir_path / "controlnet/openvino_model.xml"
        with torch.no_grad():
            from functools import partial
            controlnet.forward = partial(controlnet.forward, return_dict=False)
            ov_model = openvino.convert_model(controlnet, example_input=dummy_inputs, input=input_info)
            openvino.save_model(ov_model, CONTROLNET_OV_PATH)
            del ov_model
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()
        print("ControlNet successfully converted to IR")

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        vae_decoder_file_name: Optional[str] = None,
        text_encoder_file_name: Optional[str] = None,
        unet_file_name: Optional[str] = None,
        controlnet_file_name: Optional[str] = None,
        vae_encoder_file_name: Optional[str] = None,
        text_encoder_2_file_name: Optional[str] = None,
        local_files_only: bool = False,
        from_onnx: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        vae_decoder_file_name = vae_decoder_file_name or default_file_name
        text_encoder_file_name = text_encoder_file_name or default_file_name
        text_encoder_2_file_name = text_encoder_2_file_name or default_file_name
        unet_file_name = unet_file_name or default_file_name
        controlnet_file_name = controlnet_file_name or default_file_name
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
                    controlnet_file_name,
                    vae_encoder_file_name,
                    vae_decoder_file_name.replace(".xml", ".bin"),
                    text_encoder_file_name.replace(".xml", ".bin"),
                    text_encoder_2_file_name.replace(".xml", ".bin"),
                    unet_file_name.replace(".xml", ".bin"),
                    controlnet_file_name.replace(".xml", ".bin"),
                    vae_encoder_file_name.replace(".xml", ".bin"),
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                    cls.config_name,
                }
            )
            ignore_patterns = ["*.msgpack", "*.safetensors", "*pytorch_model.bin"]
            if not from_onnx:
                ignore_patterns.extend(["*.onnx", "*.onnx_data"])
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
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

        unet_path = new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name
        controlnet_path = new_model_save_dir / "controlnet" /controlnet_file_name
        components = {
            "vae_encoder": new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            "vae_decoder": new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "text_encoder": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "text_encoder_2": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name,
        }

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)
        if quantization_config is None or quantization_config.dataset is None:
            unet = cls.load_model(unet_path, quantization_config)
            controlnet = cls.load_model(controlnet_path)
            for key, value in components.items():
                components[key] = cls.load_model(value, quantization_config) if value.is_file() else None
        else:
            # Load uncompressed models to apply hybrid quantization further
            raise NotImplementedError(f"Quantization in hybrid mode is not supported for {cls.__name__}")

        return cls(
            unet=unet,
            controlnet=controlnet,
            config=config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **components,
            **kwargs,
        )
    
    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        tokenizer: Optional["CLIPTokenizer"] = None,
        scheduler: Union["DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        if "controlnet_model_id" not in kwargs.keys():
            raise ValueError("You must give controlnet id with controlnet_model_id=controlnet_model_id.")
        else:
            cls.export_controlnet(model_id=kwargs["controlnet_model_id"], save_dir_path=save_dir_path)
                
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=cls.export_feature,
            do_validation=False,
            no_post_process=True,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            ov_config=ov_config,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=False,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            model_save_dir=save_dir,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )
class StableDiffusionContrlNetPipelineMixin(ConfigMixin):
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: np.dtype = np.float32,
        latents: np.ndarray = None,
    ):
        """
        Preparing noise to image generation. If initial latents are not provided, they will be generated randomly,
        then prepared latents scaled by the standard deviation required by the scheduler

        Parameters:
           batch_size (int): input batch size
           num_channels_latents (int): number of channels for noise generation
           height (int): image height
           width (int): image width
           dtype (np.dtype, *optional*, np.float32): dtype for latents generation
           latents (np.ndarray, *optional*, None): initial latent noise tensor, if not provided will be generated
        Returns:
           latents (np.ndarray): scaled initial noise for diffusion
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = self.randn_tensor(shape, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: np.array, pad: Tuple[int]):
        """
        Decode predicted image from latent space using VAE Decoder and unpad image result

        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
           pad (Tuple[int]): each side padding sizes obtained on preprocessing step
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[0]
        (_, end_h), (_, end_w) = pad[1:3]
        h, w = image.shape[2:]
        unpad_h = h - end_h
        unpad_w = w - end_w
        image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        return image
    
    def scale_fit_to_window(self, dst_width: int, dst_height: int, image_width: int, image_height: int):
        """
        Preprocessing helper function for calculating image size for resize with peserving original aspect ratio
        and fitting image to specific window size

        Parameters:
        dst_width (int): destination window width
        dst_height (int): destination window height
        image_width (int): source image width
        image_height (int): source image height
        Returns:
        result_width (int): calculated width for resize
        result_height (int): calculated height for resize
        """
        im_scale = min(dst_height / image_height, dst_width / image_width)
        return int(im_scale * image_width), int(im_scale * image_height)

    def preprocess(self, image: PIL.Image.Image):
        """
        Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
        then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
        converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
        The function returns preprocessed input tensor and padding size, which can be used in postprocessing.

        Parameters:
        image (PIL.Image.Image): input image
        Returns:
        image (np.ndarray): preprocessed image tensor
        pad (Tuple[int]): pading size for each dimension for restoring image size in postprocessing
        """
        src_width, src_height = image.size
        dst_width, dst_height = self.scale_fit_to_window(512, 512, src_width, src_height)
        image = np.array(image.resize((dst_width, dst_height), resample=PIL.Image.Resampling.LANCZOS))[None, :]
        pad_width = 512 - dst_width
        pad_height = 512 - dst_height
        pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        image = np.pad(image, pad, mode="constant")
        image = image.astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        return image, pad

    def randn_tensor(self,
        shape: Union[Tuple, List],
        dtype: Optional[np.dtype] = np.float32,
    ):
        """
        Helper function for generation random values tensor with given shape and data type

        Parameters:
        shape (Union[Tuple, List]): shape for filling random values
        dtype (np.dtype, *optiona*, np.float32): data type for result
        Returns:
        latents (np.ndarray): tensor with random values with given data type and shape (usually represents noise in latent space)
        """
        latents = np.random.randn(*shape).astype(dtype)

        return latents
    
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs
        
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image,
        num_inference_steps: int = 10,
        negative_prompt: Union[str, List[str]] = None,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        eta: float = 0.0,
        latents: Optional[np.array] = None,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Parameters:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`PIL.Image.Image`):
                `PIL.Image`, or tensor representing an image batch which will be repainted according to `prompt`.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            negative_prompt (`str` or `List[str]`):
                negative prompt or prompts for generation
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. This pipeline requires a value of at least `1`.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
        Returns:
            image ([List[Union[np.ndarray, PIL.Image.Image]]): generaited images

        """

        # 1. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)

        # 3. Preprocess image
        orig_width, orig_height = image.size
        image, pad = self.preprocess(image)
        height, width = image.shape[-2:]
        if do_classifier_free_guidance:
            image = np.concatenate(([image] * 2))

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.set_progress_bar_config(disable=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                result = self.controlnet(sample=latent_model_input, timestep=t, encoder_hidden_states=text_embeddings, controlnet_cond=image)
                
                down_and_mid_block_samples = [sample * controlnet_conditioning_scale for sample in result]
                down_and_mid_block_samples = tuple(down_and_mid_block_samples)
                # predict the noise residual
                noise_pred = self.unet(sample=latent_model_input, timestep=t, encoder_hidden_states=text_embeddings, down_and_mid_block_samples=down_and_mid_block_samples)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents, pad)

        # 9. Convert to PIL
        image = numpy_to_pil(image)
        image = [img.resize((orig_width, orig_height), PIL.Image.Resampling.LANCZOS) for img in image]

        return image
    
class OVStableDiffusionControlNetPipeline(OVStableDiffusionControlNetPipelineBase, StableDiffusionContrlNetPipelineMixin):
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[PIL.Image.Image] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        eta: float = 0.0,
        latents: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        height = height or self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        width = width or self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        _height = self.height
        _width = self.width
        expected_batch_size = self._batch_size

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

        if expected_batch_size != -1:
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = kwargs.get("prompt_embeds").shape[0]

            _raise_invalid_batch_size(expected_batch_size, batch_size, num_images_per_prompt, guidance_scale)

        return StableDiffusionContrlNetPipelineMixin.__call__(
            self,
            prompt = prompt,
            image = image,
            num_inference_steps = num_inference_steps,
            negative_prompt = negative_prompt,
            guidance_scale = guidance_scale,
            controlnet_conditioning_scale = controlnet_conditioning_scale,
            eta = eta,
            latents = latents,
        )
    

    

