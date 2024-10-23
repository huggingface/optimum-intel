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
import inspect
import logging
import os
import shutil
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Optional, Union

import numpy as np
import openvino
import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipelines import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DiffusionPipeline,
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
from diffusers.utils.constants import CONFIG_NAME
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
from ..utils.import_utils import is_diffusers_version
from .configuration import OVConfig, OVQuantizationMethod, OVWeightQuantizationConfig
from .loaders import OVTextualInversionLoaderMixin
from .modeling_base import OVBaseModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    OV_TO_PT_TYPE,
    OV_XML_FILE_NAME,
    TemporaryDirectory,
    _print_compiled_model_properties,
    model_has_dynamic_inputs,
    np_to_pt_generators,
)


if is_diffusers_version(">=", "0.25.0"):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
else:
    from diffusers.models.vae import DiagonalGaussianDistribution


core = Core()

logger = logging.getLogger(__name__)


# TODO: support DiffusionPipeline.from_pipe()
# TODO: makes more sense to have a compositional OVMixin class
# TODO: instead of one bloated __init__, we should consider an __init__ per pipeline
class OVDiffusionPipeline(OVBaseModel, DiffusionPipeline):
    auto_model_class = DiffusionPipeline
    config_name = "model_index.json"
    _library_name = "diffusers"

    def __init__(
        self,
        scheduler: SchedulerMixin,
        unet: openvino.runtime.Model,
        vae_decoder: openvino.runtime.Model,
        # optional pipeline models
        vae_encoder: Optional[openvino.runtime.Model] = None,
        text_encoder: Optional[openvino.runtime.Model] = None,
        text_encoder_2: Optional[openvino.runtime.Model] = None,
        # optional pipeline submodels
        tokenizer: Optional[CLIPTokenizer] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        # stable diffusion xl specific arguments
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetics_score: bool = False,
        add_watermarker: Optional[bool] = None,
        # openvino specific arguments
        device: str = "CPU",
        compile: bool = True,
        compile_only: bool = False,
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self._compile_only = compile_only
        self.model_save_dir = model_save_dir
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])

        if self._compile_only:
            if not compile:
                raise ValueError(
                    "`compile_only` mode does not support disabling compilation."
                    "Please provide `compile=True` if you want to use `compile_only=True` or set `compile_only=False`"
                )

            if not isinstance(unet, openvino.runtime.CompiledModel):
                raise ValueError("`compile_only` expect that already compiled model will be provided")

            model_is_dynamic = model_has_dynamic_inputs(unet)
            if dynamic_shapes ^ model_is_dynamic:
                requested_shapes = "dynamic" if dynamic_shapes else "static"
                compiled_shapes = "dynamic" if model_is_dynamic else "static"
                raise ValueError(
                    f"Provided compiled model with {compiled_shapes} shapes but requested to use {requested_shapes}. "
                    f"Please set `compile_only=False` or `dynamic_shapes={model_is_dynamic}`"
                )

        self.unet = OVModelUnet(unet, self, DIFFUSION_MODEL_UNET_SUBFOLDER)
        self.vae_decoder = OVModelVaeDecoder(vae_decoder, self, DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER)
        self.vae_encoder = (
            OVModelVaeEncoder(vae_encoder, self, DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER)
            if vae_encoder is not None
            else None
        )
        self.text_encoder = (
            OVModelTextEncoder(text_encoder, self, DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER)
            if text_encoder is not None
            else None
        )
        self.text_encoder_2 = (
            OVModelTextEncoder(text_encoder_2, self, DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER)
            if text_encoder_2 is not None
            else None
        )
        # We wrap the VAE Decoder & Encoder in a single object to simulate diffusers API
        self.vae = OVModelVae(decoder=self.vae_decoder, encoder=self.vae_encoder)

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.feature_extractor = feature_extractor

        # we allow passing these as torch models for now
        self.image_encoder = kwargs.pop("image_encoder", None)  # TODO: maybe mplement OVModelImageEncoder
        self.safety_checker = kwargs.pop("safety_checker", None)  # TODO: maybe mplement OVModelSafetyChecker

        all_pipeline_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }

        diffusers_pipeline_args = {}
        for key in inspect.signature(self.auto_model_class).parameters.keys():
            if key in all_pipeline_init_args:
                diffusers_pipeline_args[key] = all_pipeline_init_args[key]
        # inits diffusers pipeline specific attributes (registers modules and config)
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)
        # we use auto_model_class.__init__ here because we can't call super().__init__
        # as OptimizedModel already defines an __init__ which is the first in the MRO

        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
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

        models_to_save_paths = {
            (self.unet, save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER),
            (self.vae_decoder, save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER),
            (self.vae_encoder, save_directory / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER),
            (self.text_encoder, save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER),
            (self.text_encoder_2, save_directory / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER),
        }
        for model, save_path in models_to_save_paths:
            if model is not None:
                dst_path = save_path / OV_XML_FILE_NAME
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                openvino.save_model(model.model, dst_path, compress_to_fp16=False)
                model_dir = model.config.get("_name_or_path", None) or model.model_save_dir
                config_path = Path(model_dir) / CONFIG_NAME
                if config_path.is_file():
                    config_save_path = save_path / CONFIG_NAME
                    shutil.copyfile(config_path, config_save_path)

        self.scheduler.save_pretrained(save_directory / "scheduler")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory / "tokenizer_2")
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory / "feature_extractor")

        self._save_openvino_config(save_directory)

    def _save_config(self, save_directory):
        """
        Saves a model configuration into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        model_dir = (
            self.model_save_dir
            if not isinstance(self.model_save_dir, TemporaryDirectory)
            else self.model_save_dir.name
        )
        save_dir = Path(save_directory)
        original_config = Path(model_dir) / self.config_name
        if original_config.exists():
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            shutil.copy(original_config, save_dir)
        else:
            self.config.save_pretrained(save_dir)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        unet_file_name: Optional[str] = None,
        vae_decoder_file_name: Optional[str] = None,
        vae_encoder_file_name: Optional[str] = None,
        text_encoder_file_name: Optional[str] = None,
        text_encoder_2_file_name: Optional[str] = None,
        from_onnx: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        # same as DiffusionPipeline.from_pretraoned, if called directly, it loads the class in the config
        if cls.__name__ == "OVDiffusionPipeline":
            class_name = config["_class_name"]
            ov_pipeline_class = _get_ov_class(class_name)
        else:
            ov_pipeline_class = cls

        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME

        unet_file_name = unet_file_name or default_file_name
        vae_encoder_file_name = vae_encoder_file_name or default_file_name
        vae_decoder_file_name = vae_decoder_file_name or default_file_name
        text_encoder_file_name = text_encoder_file_name or default_file_name
        text_encoder_2_file_name = text_encoder_2_file_name or default_file_name

        if not os.path.isdir(str(model_id)):
            all_components = {key for key in config.keys() if not key.startswith("_")} | {"vae_encoder", "vae_decoder"}
            allow_patterns = {os.path.join(component, "*") for component in all_components}
            allow_patterns.update(
                {
                    unet_file_name,
                    vae_encoder_file_name,
                    vae_decoder_file_name,
                    text_encoder_file_name,
                    text_encoder_2_file_name,
                    unet_file_name.replace(".xml", ".bin"),
                    vae_encoder_file_name.replace(".xml", ".bin"),
                    vae_decoder_file_name.replace(".xml", ".bin"),
                    text_encoder_file_name.replace(".xml", ".bin"),
                    text_encoder_2_file_name.replace(".xml", ".bin"),
                    SCHEDULER_CONFIG_NAME,
                    cls.config_name,
                    CONFIG_NAME,
                }
            )
            ignore_patterns = ["*.msgpack", "*.safetensors", "*pytorch_model.bin"]
            if not from_onnx:
                ignore_patterns.extend(["*.onnx", "*.onnx_data"])

            model_save_folder = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                revision=revision,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        else:
            model_save_folder = str(model_id)

        model_save_path = Path(model_save_folder)

        if model_save_dir is None:
            model_save_dir = model_save_path

        submodels = {"scheduler": None, "tokenizer": None, "tokenizer_2": None, "feature_extractor": None}
        for name in submodels.keys():
            if kwargs.get(name, None) is not None:
                submodels[name] = kwargs.pop(name)
            elif config.get(name, (None, None))[0] is not None:
                library_name, library_classes = config.get(name)
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (model_save_path / name).is_dir():
                    submodels[name] = load_method(model_save_path / name)
                else:
                    submodels[name] = load_method(model_save_path)

        models = {
            "unet": model_save_path / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
            "vae_decoder": model_save_path / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "vae_encoder": model_save_path / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            "text_encoder": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "text_encoder_2": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name,
        }

        compile_only = kwargs.get("compile_only", False)
        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)
        if (quantization_config is None or quantization_config.dataset is None) and not compile_only:
            for name, path in models.items():
                if kwargs.get(name, None) is not None:
                    models[name] = kwargs.pop(name)
                else:
                    models[name] = cls.load_model(path, quantization_config) if path.is_file() else None
        elif compile_only:
            ov_config = kwargs.get("ov_config", {})
            device = kwargs.get("device", "CPU")
            vae_ov_conifg = {**ov_config}
            if "GPU" in device.upper() and "INFERENCE_PRECISION_HINT" not in vae_ov_conifg:
                vae_ov_conifg["INFERENCE_PRECISION_HINT"] = "f32"
            for name, path in models.items():
                if kwargs.get(name, None) is not None:
                    models[name] = kwargs.pop(name)
                else:
                    models[name] = (
                        cls._compile_model(
                            path,
                            device,
                            ov_config if "vae" not in name else vae_ov_conifg,
                            Path(model_save_dir) / name,
                        )
                        if path.is_file()
                        else None
                    )
        else:
            # why is this quantization not performed in __init__?
            if ov_pipeline_class.export_feature != "text-to-image":
                raise NotImplementedError(f"Quantization in hybrid mode is not supported for {cls.__name__}")

            from optimum.intel import OVQuantizer

            for name, path in models.items():
                if kwargs.get(name, None) is not None:
                    models[name] = kwargs.pop(name)
                else:
                    models[name] = cls.load_model(path) if path.is_file() else None

            ov_pipeline = ov_pipeline_class(**models, **submodels, model_save_dir=model_save_dir, **kwargs)
            # same as in DiffusionPipeline.from_pretrained, we save where the model was instantiated from
            ov_pipeline.register_to_config(_name_or_path=config.get("_name_or_path", str(model_id)))

            hybrid_quantization_config = deepcopy(quantization_config)
            hybrid_quantization_config.quant_method = OVQuantizationMethod.HYBRID
            quantizer = OVQuantizer(ov_pipeline)
            quantizer.quantize(ov_config=OVConfig(quantization_config=hybrid_quantization_config))

            return ov_pipeline

        ov_pipeline = ov_pipeline_class(
            **models,
            **submodels,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )
        # same as in DiffusionPipeline.from_pretrained, we save where the model was instantiated from
        ov_pipeline.register_to_config(_name_or_path=config.get("_name_or_path", str(model_id)))

        return ov_pipeline

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
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        compile_only: bool = False,
        **kwargs,
    ):
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False

        # If load_in_8bit and quantization_config not specified then ov_config is set
        # to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        model_save_dir = TemporaryDirectory()
        model_save_path = Path(model_save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=model_save_path,
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
            model_id=model_save_path,
            config=config,
            from_onnx=False,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            load_in_8bit=load_in_8bit,
            compile_only=compile_only,
            **kwargs,
        )

    def to(self, *args, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        for arg in args:
            if isinstance(arg, str):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        if isinstance(device, str):
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
    def batch_size(self) -> int:
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
        self.vae_decoder.model = self._reshape_vae_decoder(
            self.vae_decoder.model, height, width, num_images_per_prompt
        )

        if self.vae_encoder is not None:
            self.vae_encoder.model = self._reshape_vae_encoder(self.vae_encoder.model, batch_size, height, width)

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

        for component in {self.unet, self.vae_encoder, self.vae_decoder, self.text_encoder, self.text_encoder_2}:
            if component is not None:
                compress_model_transformation(component.model)

        self.clear_requests()

        return self

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please intialize model without this option"
            )

        for component in {self.unet, self.vae_encoder, self.vae_decoder, self.text_encoder, self.text_encoder_2}:
            if component is not None:
                component.request = None

    def compile(self):
        for component in {self.unet, self.vae_encoder, self.vae_decoder, self.text_encoder, self.text_encoder_2}:
            if component is not None:
                component._compile()

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)

    @property
    def components(self) -> Dict[str, Any]:
        components = {
            "vae": self.vae,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
        }
        components = {k: v for k, v in components.items() if v is not None}
        return components

    def __call__(self, *args, **kwargs):
        # we do this to keep numpy random states support for now
        # TODO: deprecate and add warnings when a random state is passed

        args = list(args)
        for i in range(len(args)):
            args[i] = np_to_pt_generators(args[i], self.device)

        for k, v in kwargs.items():
            kwargs[k] = np_to_pt_generators(v, self.device)

        # we use auto_model_class.__call__ here because we can't call super().__call__
        # as OptimizedModel already defines a __call__ which is the first in the MRO
        return self.auto_model_class.__call__(self, *args, **kwargs)


class OVPipelinePart(ConfigMixin):
    config_name: str = CONFIG_NAME

    def __init__(
        self,
        model: openvino.runtime.Model,
        parent_pipeline: OVDiffusionPipeline,
        model_name: str = "",
    ):
        self.model = model
        self.model_name = model_name
        self.parent_pipeline = parent_pipeline
        self.request = None if not parent_pipeline._compile_only else self.model
        self.ov_config = parent_pipeline.ov_config

        if isinstance(parent_pipeline.model_save_dir, TemporaryDirectory):
            self.model_save_dir = Path(parent_pipeline.model_save_dir.name) / self.model_name
        else:
            self.model_save_dir = Path(parent_pipeline.model_save_dir) / self.model_name

        config_file_path = self.model_save_dir / self.config_name

        if not config_file_path.is_file():
            # config is mandatory for the model part to be used for inference
            raise ValueError(f"Configuration file for {self.__class__.__name__} not found at {config_file_path}")

        config_dict = self._dict_from_json_file(config_file_path)
        self.register_to_config(**config_dict)

    @property
    def _device(self) -> str:
        return self.parent_pipeline._device

    @property
    def device(self) -> torch.device:
        return self.parent_pipeline.device

    @property
    def dtype(self) -> torch.dtype:
        return OV_TO_PT_TYPE[self.ov_config.get("dtype", "f32")]

    def _compile(self):
        if self.request is None:
            if (
                "CACHE_DIR" not in self.ov_config.keys()
                and not str(self.model_save_dir).startswith(gettempdir())
                and "GPU" in self._device
            ):
                self.ov_config["CACHE_DIR"] = os.path.join(self.model_save_dir, "model_cache")

            logger.info(f"Compiling the {self.model_name} to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, self.ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                logger.info(f"{self._device} SUPPORTED_PROPERTIES:")
                _print_compiled_model_properties(self.request)

    def to(self, *args, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        for arg in args:
            if isinstance(arg, str):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        if isinstance(device, str):
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

        return self

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def modules(self):
        return []


class OVModelTextEncoder(OVPipelinePart):
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


class OVModelUnet(OVPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self.config, "time_cond_proj_dim"):
            logger.warning(
                "The `time_cond_proj_dim` attribute is missing from the UNet configuration. "
                "Please re-export the model with newer version of optimum and diffusers."
            )
            self.register_to_config(time_cond_proj_dim=None)

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


class OVModelVaeEncoder(OVPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self.config, "scaling_factor"):
            logger.warning(
                "The `scaling_factor` attribute is missing from the VAE encoder configuration. "
                "Please re-export the model with newer version of optimum and diffusers."
            )
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
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


class OVModelVaeDecoder(OVPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # can be missing from models exported long ago
        if not hasattr(self.config, "scaling_factor"):
            logger.warning(
                "The `scaling_factor` attribute is missing from the VAE decoder configuration. "
                "Please re-export the model with newer version of optimum and diffusers."
            )
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

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


class OVModelVae:
    def __init__(self, decoder: OVModelVaeDecoder, encoder: OVModelVaeEncoder):
        self.decoder = decoder
        self.encoder = encoder

    @property
    def config(self):
        return self.decoder.config

    @property
    def dtype(self):
        return self.decoder.dtype

    @property
    def device(self):
        return self.decoder.device

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.decoder.to(*args, **kwargs)
        if self.encoder is not None:
            self.encoder.to(*args, **kwargs)


class OVStableDiffusionPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion#diffusers.StableDiffusionPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusionPipeline


class OVStableDiffusionImg2ImgPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionImg2ImgPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = StableDiffusionImg2ImgPipeline


class OVStableDiffusionInpaintPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionInpaintPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = StableDiffusionInpaintPipeline


class OVStableDiffusionXLPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusionXLPipeline

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


class OVStableDiffusionXLImg2ImgPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLImg2ImgPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = StableDiffusionXLImg2ImgPipeline

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


class OVStableDiffusionXLInpaintPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLInpaintPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline).
    """

    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = StableDiffusionXLInpaintPipeline

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


class OVLatentConsistencyModelPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, LatentConsistencyModelPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = LatentConsistencyModelPipeline


class OVLatentConsistencyModelImg2ImgPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, LatentConsistencyModelImg2ImgPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency_img2img#diffusers.LatentConsistencyModelImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
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


def _get_ov_class(pipeline_class_name: str, throw_error_if_not_exist: bool = True):
    for ov_pipeline_class in SUPPORTED_OV_PIPELINES:
        if (
            ov_pipeline_class.__name__ == pipeline_class_name
            or ov_pipeline_class.auto_model_class.__name__ == pipeline_class_name
        ):
            return ov_pipeline_class

    if throw_error_if_not_exist:
        raise ValueError(f"OVDiffusionPipeline can't find a pipeline linked to {pipeline_class_name}")


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


def _get_task_ov_class(mapping, pipeline_class_name):
    def _get_model_name(pipeline_class_name):
        for ov_pipelines_mapping in SUPPORTED_OV_PIPELINES_MAPPINGS:
            for model_name, ov_pipeline_class in ov_pipelines_mapping.items():
                if (
                    ov_pipeline_class.__name__ == pipeline_class_name
                    or ov_pipeline_class.auto_model_class.__name__ == pipeline_class_name
                ):
                    return model_name

    model_name = _get_model_name(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    raise ValueError(f"OVPipelineForTask can't find a pipeline linked to {pipeline_class_name} for {model_name}")


class OVPipelineForTask(ConfigMixin):
    auto_model_class = DiffusionPipeline
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

        ov_pipeline_class = _get_task_ov_class(cls.ov_pipelines_mapping, class_name)

        return ov_pipeline_class.from_pretrained(pretrained_model_or_path, **kwargs)


class OVPipelineForText2Image(OVPipelineForTask):
    auto_model_class = AutoPipelineForText2Image
    ov_pipelines_mapping = OV_TEXT2IMAGE_PIPELINES_MAPPING


class OVPipelineForImage2Image(OVPipelineForTask):
    auto_model_class = AutoPipelineForImage2Image
    ov_pipelines_mapping = OV_IMAGE2IMAGE_PIPELINES_MAPPING


class OVPipelineForInpainting(OVPipelineForTask):
    auto_model_class = AutoPipelineForInpainting
    ov_pipelines_mapping = OV_INPAINT_PIPELINES_MAPPING
