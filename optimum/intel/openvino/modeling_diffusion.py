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
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Dict, Optional, Union

import numpy as np
import openvino
import torch
from diffusers.configuration_utils import ConfigMixin, FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    LatentConsistencyModelImg2ImgPipeline,
    LatentConsistencyModelPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.schedulers import SchedulerMixin
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, is_invisible_watermark_available
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import validate_hf_hub_args
from openvino._offline_transformations import compress_model_transformation
from openvino.runtime import Core
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.modeling_outputs import ModelOutput

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
    OV_TO_PT_TYPE,
    OV_XML_FILE_NAME,
    _print_compiled_model_properties,
    np_to_pt_generators,
)


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker


core = Core()

logger = logging.getLogger(__name__)


class OVPipeline(OVBaseModel):
    auto_model_class = None

    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    _library_name = "diffusers"

    def __init__(
        self,
        config: Dict[str, Any],
        scheduler: SchedulerMixin,
        unet: openvino.runtime.Model,
        vae_encoder: Optional[openvino.runtime.Model] = None,
        vae_decoder: Optional[openvino.runtime.Model] = None,
        text_encoder: Optional[openvino.runtime.Model] = None,
        text_encoder_2: Optional[openvino.runtime.Model] = None,
        tokenizer: Optional["CLIPTokenizer"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        device: str = "CPU",
        compile: bool = True,
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.model_save_dir = model_save_dir
        self.ov_config = {} if ov_config is None else {**ov_config}
        self._openvino_config = OVConfig(quantization_config=quantization_config) if quantization_config else None
        self._compile_only = kwargs.get("compile_only", False)

        self.unet = OVModelUnet(unet, self)
        self.vae_encoder = OVModelVaeEncoder(vae_encoder, self) if vae_encoder is not None else None
        self.vae_decoder = OVModelVaeDecoder(vae_decoder, self) if vae_decoder is not None else None
        self.text_encoder = OVModelTextEncoder(text_encoder, self) if text_encoder is not None else None
        self.text_encoder_2 = OVModelTextEncoder(text_encoder_2, self) if text_encoder_2 is not None else None

        # We create VAE encoder & decoder and wrap them in one object to
        # be used by the pipeline mixins with minimal code changes (simulating the diffusers API)
        self.vae = OVModelVae(self.vae_encoder, self.vae_decoder, self)

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.feature_extractor = feature_extractor
        self.safety_checker = kwargs.get("safety_checker", None)

        if hasattr(self.vae.config, "block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        sub_models = {
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER: self.vae_encoder,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER: self.text_encoder_2,
        }
        for name in sub_models.keys():
            config[name] = (
                ("optimum", sub_models[name].__class__.__name__) if sub_models[name] is not None else (None, None)
            )

        self._internal_dict = FrozenDict(config)

        self._set_ov_config_parameters()

        if self.is_dynamic and not self._compile_only:
            self.reshape(batch_size=-1, height=-1, width=-1, num_images_per_prompt=-1)

        if compile and not self._compile_only:
            self.compile()

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files
        """
        if self._compile_only:
            raise ValueError(
                "`save_pretrained()` is not supported with `compile_only` mode, please intialize model without this option"
            )

        save_directory = Path(save_directory)

        sub_models_to_save = {
            self.unet: DIFFUSION_MODEL_UNET_SUBFOLDER,
            self.vae_encoder: DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            self.vae_decoder: DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            self.text_encoder: DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
            self.text_encoder_2: DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
        }

        for ov_model, dst_path in sub_models_to_save.items():
            if ov_model is not None:
                dst_path = save_directory / dst_path / OV_XML_FILE_NAME
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                openvino.save_model(ov_model.model, dst_path, compress_to_fp16=False)
                model_dir = ov_model.config.get("_name_or_path", None) or ov_model._model_dir / ov_model._model_name
                config_path = Path(model_dir) / ov_model.config_name
                if config_path.is_file():
                    shutil.copyfile(config_path, dst_path.parent / ov_model.config_name)

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
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME

        unet_file_name = unet_file_name or default_file_name
        vae_encoder_file_name = vae_encoder_file_name or default_file_name
        vae_decoder_file_name = vae_decoder_file_name or default_file_name
        text_encoder_file_name = text_encoder_file_name or default_file_name
        text_encoder_2_file_name = text_encoder_2_file_name or default_file_name

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

        compile_only = kwargs.get("compile_only", False)

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)
        if (quantization_config is None or quantization_config.dataset is None) and not compile_only:
            unet = cls.load_model(unet_path, quantization_config)
            for key, value in components.items():
                components[key] = cls.load_model(value, quantization_config) if value.is_file() else None
        elif compile_only:
            ov_config = kwargs.get("ov_config", {})
            device = kwargs.get("device", "CPU")
            vae_ov_conifg = {**ov_config}
            if "GPU" in device.upper() and "INFERENCE_PRECISION_HINT" not in vae_ov_conifg:
                vae_ov_conifg["INFERENCE_PRECISION_HINT"] = "f32"
            unet = cls._compile_model(unet_path, device, ov_config, Path(model_save_dir) / "unet")
            for key, value in components.items():
                components[key] = (
                    cls._compile_model(
                        value, device, ov_config if "vae" not in key else vae_ov_conifg, Path(model_save_dir) / key
                    )
                    if value.is_file()
                    else None
                )
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
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        tokenizer: Optional["CLIPTokenizer"] = None,
        scheduler: Optional[SchedulerMixin] = None,
        feature_extractor: Optional["CLIPFeatureExtractor"] = None,
        tokenizer_2: Optional["CLIPTokenizer"] = None,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        compile_only = kwargs.pop("compile_only", False)

        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            do_validation=False,
            no_post_process=True,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            ov_config=ov_config,
            library_name=cls._library_name,
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
            compile_only=compile_only,
            **kwargs,
        )

    def to(self, device: str):
        if self._compile_only and not isinstance(device, str):
            raise ValueError(
                "`to()` is not supported with `compile_only` mode, please intialize model without this option"
            )

        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()
        else:
            raise ValueError(
                "The `device` argument should be a string representing the device on which the model should be loaded."
            )

        return self

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

    def _reshape_vae_decoder(
        self, model: openvino.runtime.Model, height: int = -1, width: int = -1, num_images_per_prompt: int = -1
    ):
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
        shapes = {model.inputs[0]: [num_images_per_prompt, latent_channels, height, width]}
        model.reshape(shapes)
        return model

    def reshape(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_images_per_prompt: int = -1,
    ):
        if self._compile_only:
            raise ValueError(
                "`reshape()` is not supported with `compile_only` mode, please intialize model without this option"
            )

        self.is_dynamic = -1 in {batch_size, height, width, num_images_per_prompt}

        if self.tokenizer is None and self.tokenizer_2 is None:
            tokenizer_max_len = -1
        else:
            tokenizer_max_len = (
                self.tokenizer.model_max_length if self.tokenizer is not None else self.tokenizer_2.model_max_length
            )

        self.unet.model = self._reshape_unet(
            self.unet.model, batch_size, height, width, num_images_per_prompt, tokenizer_max_len
        )

        if self.vae_encoder is not None:
            self.vae_encoder.model = self._reshape_vae_encoder(self.vae_encoder.model, batch_size, height, width)

        if self.vae_decoder is not None:
            self.vae_decoder.model = self._reshape_vae_decoder(
                self.vae_decoder.model, height, width, num_images_per_prompt
            )

        if self.text_encoder is not None:
            self.text_encoder.model = self._reshape_text_encoder(
                self.text_encoder.model, batch_size, self.tokenizer.model_max_length
            )

        if self.text_encoder_2 is not None:
            self.text_encoder_2.model = self._reshape_text_encoder(
                self.text_encoder_2.model, batch_size, self.tokenizer_2.model_max_length
            )

        self.clear_requests()
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        if self._compile_only:
            raise ValueError(
                "`half()` is not supported with `compile_only` mode, please intialize model without this option"
            )

        components = {self.unet, self.vae_encoder, self.vae_decoder, self.text_encoder, self.text_encoder_2}
        for component in components:
            if component is not None:
                compress_model_transformation(component.model)

        self.clear_requests()

        return self

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please intialize model without this option"
            )

        components = {self.unet, self.vae_encoder, self.vae_decoder, self.text_encoder, self.text_encoder_2}
        for component in components:
            if component is not None:
                component.request = None

    def compile(self):
        components = {self.unet, self.vae_encoder, self.vae_decoder, self.text_encoder, self.text_encoder_2}
        for component in components:
            if component is not None:
                component._compile()

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)

    @property
    def _execution_device(self):
        return self.device

    def __call__(self, *args, **kwargs):
        device = self._execution_device

        for i in range(len(args)):
            args[i] = np_to_pt_generators(args[i], device)

        for k, v in kwargs.items():
            kwargs[k] = np_to_pt_generators(v, device)

        return self.auto_model_class.__call__(self, *args, **kwargs)


class OVModelPart:
    model_part_folder: str = None
    config_name: str = "config.json"

    def __init__(
        self, model: openvino.runtime.Model, parent_model: OVBaseModel, ov_config: Optional[Dict[str, str]] = None
    ):
        self.model = model
        self.parent_model = parent_model

        config_path = self.model_save_dir / self.config_name
        self.config = FrozenDict(parent_model._dict_from_json_file(config_path) if config_path.is_file() else {})

        self.ov_config = ov_config or {**self.parent_model.ov_config}
        self._compile_only = parent_model._compile_only
        self.request = None if not self._compile_only else self.model

    @property
    def _device(self) -> str:
        return self.parent_model._device

    @property
    def device(self) -> torch.device:
        return self.parent_model.device

    @property
    def dtype(self) -> torch.dtype:
        return OV_TO_PT_TYPE[self.ov_config.get("dtype", "f32")]

    @property
    def model_save_dir(self) -> Optional[Union[str, Path, TemporaryDirectory]]:
        if isinstance(self.parent_model.model_save_dir, TemporaryDirectory):
            return Path(self.parent_model.model_save_dir.name) / self.model_part_folder
        else:
            return Path(self.parent_model.model_save_dir) / self.model_part_folder

    def _compile(self):
        if self.request is None:
            if (
                "CACHE_DIR" not in self.ov_config.keys()
                and not str(self.model_save_dir).startswith(gettempdir())
                and "GPU" in self._device
            ):
                self.ov_config["CACHE_DIR"] = os.path.join(self.model_save_dir, self.model_part_folder, "model_cache")

            logger.info(f"Compiling the {self.model_save_dir} to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, self.ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                logger.info(f"{self._device} SUPPORTED_PROPERTIES:")
                _print_compiled_model_properties(self.request)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, *args, device: Optional[Union[str]] = None, dtype: Optional[torch.dtype] = None, **kwargs):
        for arg in args:
            if isinstance(arg, str):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        if device is not None and isinstance(device, str):
            self._device = device.upper()
            self.request = None
        elif device is not None:
            raise ValueError(
                "The `device` argument should be a string representing the device on which the model should be loaded."
            )

        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                f"Cannot change the dtype of the model from {self.dtype} to {dtype}. "
                f"Please export the model with the desired dtype."
            )


class OVModelTextEncoder(OVModelPart):
    model_part_folder: str = "text_encoder"

    def forward(
        self,
        input_ids: Union[np.ndarray, torch.Tensor],
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = False,
    ):
        self._compile()

        model_inputs = {"input_ids": input_ids}

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if output_hidden_states:
            model_outputs["hidden_states"] = []
            for i in range(self.config.num_hidden_layers):
                model_outputs["hidden_states"].append(model_outputs.pop(f"hidden_states.{i}"))
            model_outputs["hidden_states"].append(model_outputs.get("last_hidden_state"))

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class OVModelUnet(OVModelPart):
    model_part_folder: str = "unet"

    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        timestep: Union[np.ndarray, torch.Tensor],
        encoder_hidden_states: Union[np.ndarray, torch.Tensor],
        text_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        time_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        timestep_cond: Optional[Union[np.ndarray, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        self._compile()

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if text_embeds is not None:
            model_inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            model_inputs["time_ids"] = time_ids
        if timestep_cond is not None:
            model_inputs["timestep_cond"] = timestep_cond
        if cross_attention_kwargs is not None:
            model_inputs.update(cross_attention_kwargs)
        if added_cond_kwargs is not None:
            model_inputs.update(added_cond_kwargs)

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class OVModelVaeEncoder(OVModelPart):
    model_part_folder: str = "vae_encoder"

    def forward(self, sample: Union[np.ndarray, torch.Tensor], return_dict: bool = False):
        self._compile()

        model_inputs = {"sample": sample}

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if "latent_parameters" in model_outputs:
            model_outputs["latent_dist"] = DiagonalGaussianDistribution(
                parameters=model_outputs.pop("latent_parameters")
            )

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)

    def _compile(self):
        if "GPU" in self._device and "INFERENCE_PRECISION_HINT" not in self.ov_config:
            self.ov_config.update({"INFERENCE_PRECISION_HINT": "f32"})
        super()._compile()


class OVModelVaeDecoder(OVModelPart):
    model_part_folder: str = "vae_decoder"

    def forward(
        self,
        latent_sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        self._compile()

        model_inputs = {"latent_sample": latent_sample}

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)

    def _compile(self):
        if "GPU" in self._device and "INFERENCE_PRECISION_HINT" not in self.ov_config:
            self.ov_config.update({"INFERENCE_PRECISION_HINT": "f32"})
        super()._compile()


class OVModelVae(OVModelPart):
    model_part_folder: str = "vae_decoder"

    def __init__(self, vae_encoder: OVModelVaeEncoder, vae_decoder: OVModelVaeDecoder, parent_model: OVPipeline):
        super().__init__(vae_decoder.model, parent_model)

        self.encode = vae_encoder.forward
        self.decode = vae_decoder.forward


class OVStableDiffusionPipeline(OVPipeline, OVTextualInversionLoaderMixin, StableDiffusionPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion#diffusers.StableDiffusionPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionPipeline


class OVStableDiffusionImg2ImgPipeline(OVPipeline, OVTextualInversionLoaderMixin, StableDiffusionImg2ImgPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    main_input_name = "image"
    auto_model_class = StableDiffusionImg2ImgPipeline


class OVStableDiffusionInpaintPipeline(OVPipeline, OVTextualInversionLoaderMixin, StableDiffusionInpaintPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    main_input_name = "image"
    auto_model_class = StableDiffusionInpaintPipeline


class OVStableDiffusionXLPipeline(OVPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLPipeline

    def __init__(self, *args, add_watermarker: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        requires_aesthetics_score = kwargs.get("requires_aesthetics_score", False)
        force_zeros_for_empty_prompt = kwargs.get("force_zeros_for_empty_prompt", True)
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)

        return add_time_ids


class OVStableDiffusionXLImg2ImgPipeline(OVPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLImg2ImgPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    main_input_name = "image"
    auto_model_class = StableDiffusionXLImg2ImgPipeline

    def __init__(self, *args, add_watermarker: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        requires_aesthetics_score = kwargs.get("requires_aesthetics_score", False)
        force_zeros_for_empty_prompt = kwargs.get("force_zeros_for_empty_prompt", True)
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids


class OVStableDiffusionXLInpaintPipeline(OVPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLInpaintPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline).
    """

    main_input_name = "image"
    auto_model_class = StableDiffusionXLInpaintPipeline

    def __init__(self, *args, add_watermarker: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        requires_aesthetics_score = kwargs.get("requires_aesthetics_score", False)
        force_zeros_for_empty_prompt = kwargs.get("force_zeros_for_empty_prompt", True)
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint.StableDiffusionXLInpaintPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids


class OVLatentConsistencyModelPipeline(OVPipeline, OVTextualInversionLoaderMixin, LatentConsistencyModelPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    main_input_name = "prompt"
    auto_model_class = LatentConsistencyModelPipeline


class OVLatentConsistencyModelImg2ImgPipeline(
    OVPipeline, OVTextualInversionLoaderMixin, LatentConsistencyModelImg2ImgPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency_img2img#diffusers.LatentConsistencyModelImg2ImgPipeline).
    """

    main_input_name = "image"
    auto_model_class = LatentConsistencyModelImg2ImgPipeline


SUPPORTED_OV_PIPELINES = [
    OVStableDiffusionPipeline,
    OVStableDiffusionImg2ImgPipeline,
    OVStableDiffusionInpaintPipeline,
    OVStableDiffusionXLPipeline,
    OVStableDiffusionXLImg2ImgPipeline,
    OVStableDiffusionXLInpaintPipeline,
    OVLatentConsistencyModelPipeline,
    OVLatentConsistencyModelImg2ImgPipeline,
]


def _get_pipeline_class(pipeline_class_name: str, throw_error_if_not_exist: bool = True):
    for ort_pipeline_class in SUPPORTED_OV_PIPELINES:
        if (
            ort_pipeline_class.__name__ == pipeline_class_name
            or ort_pipeline_class.auto_model_class.__name__ == pipeline_class_name
        ):
            return ort_pipeline_class

    if throw_error_if_not_exist:
        raise ValueError(f"OVDiffusionPipeline can't find a pipeline linked to {pipeline_class_name}")


class OVDiffusionPipeline(ConfigMixin):
    config_name = "model_index.json"

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        load_config_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download", None),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir", None),
            "revision": kwargs.get("revision", None),
            "proxies": kwargs.get("proxies", None),
            "token": kwargs.get("token", None),
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        config = config[0] if isinstance(config, tuple) else config
        class_name = config["_class_name"]

        ort_pipeline_class = _get_pipeline_class(class_name)

        return ort_pipeline_class.from_pretrained(pretrained_model_or_path, **kwargs)


OV_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", OVStableDiffusionPipeline),
        ("stable-diffusion-xl", OVStableDiffusionXLPipeline),
        ("latent-consistency", OVLatentConsistencyModelPipeline),
    ]
)

OV_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", OVStableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", OVStableDiffusionXLImg2ImgPipeline),
        ("latent-consistency", OVLatentConsistencyModelImg2ImgPipeline),
    ]
)

OV_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", OVStableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", OVStableDiffusionXLInpaintPipeline),
    ]
)

SUPPORTED_OV_PIPELINES_MAPPINGS = [
    OV_TEXT2IMAGE_PIPELINES_MAPPING,
    OV_IMAGE2IMAGE_PIPELINES_MAPPING,
    OV_INPAINT_PIPELINES_MAPPING,
]


def _get_task_class(mapping, pipeline_class_name):
    def _get_model_name(pipeline_class_name):
        for ort_pipelines_mapping in SUPPORTED_OV_PIPELINES_MAPPINGS:
            for model_name, ort_pipeline_class in ort_pipelines_mapping.items():
                if (
                    ort_pipeline_class.__name__ == pipeline_class_name
                    or ort_pipeline_class.auto_model_class.__name__ == pipeline_class_name
                ):
                    return model_name

    model_name = _get_model_name(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    raise ValueError(f"OVPipelineForTask can't find a pipeline linked to {pipeline_class_name} for {model_name}")


class OVPipelineForTask(ConfigMixin):
    config_name = "model_index.json"

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        load_config_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download", None),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir", None),
            "revision": kwargs.get("revision", None),
            "proxies": kwargs.get("proxies", None),
            "token": kwargs.get("token", None),
        }
        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        config = config[0] if isinstance(config, tuple) else config
        class_name = config["_class_name"]

        ort_pipeline_class = _get_task_class(cls.ort_pipelines_mapping, class_name)

        return ort_pipeline_class.from_pretrained(pretrained_model_or_path, **kwargs)


class OVPipelineForText2Image(OVPipelineForTask):
    auto_model_class = AutoPipelineForText2Image
    ort_pipelines_mapping = OV_TEXT2IMAGE_PIPELINES_MAPPING


class OVPipelineForImage2Image(OVPipelineForTask):
    auto_model_class = AutoPipelineForImage2Image
    ort_pipelines_mapping = OV_IMAGE2IMAGE_PIPELINES_MAPPING


class OVPipelineForInpainting(OVPipelineForTask):
    auto_model_class = AutoPipelineForInpainting
    ort_pipelines_mapping = OV_INPAINT_PIPELINES_MAPPING
