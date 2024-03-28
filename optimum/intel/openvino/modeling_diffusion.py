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
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openvino
import PIL
from diffusers import (
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
from openvino._offline_transformations import compress_model_transformation
from openvino.runtime import Core
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
from .configuration import OVConfig, OVWeightQuantizationConfig
from .loaders import OVTextualInversionLoaderMixin
from .modeling_base import OVBaseModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    OV_TO_NP_TYPE,
    OV_XML_FILE_NAME,
    PREDEFINED_SD_DATASETS,
    _print_compiled_model_properties,
)


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
        self.ov_config = ov_config if ov_config is not None else {}
        if self.ov_config.get("PERFORMANCE_HINT") is None:
            self.ov_config["PERFORMANCE_HINT"] = "LATENCY"

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

        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)

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
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
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
            ignore_patterns = ["*.msgpack", "*.safetensors", "*pytorch_model.bin"]
            if not from_onnx:
                ignore_patterns.extend(["*.onnx", "*.onnx_data"])
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
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

        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)

        unet_path = new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name
        if quantization_config is not None and quantization_config.dataset is not None:
            # load the UNet model uncompressed to apply hybrid quantization further
            unet = cls.load_model(unet_path)
            # Apply weights compression to other `components` without dataset
            weight_quantization_params = {
                param: value for param, value in quantization_config.__dict__.items() if param != "dataset"
            }
            weight_quantization_config = OVWeightQuantizationConfig.from_dict(weight_quantization_params)
        else:
            weight_quantization_config = quantization_config
            unet = cls.load_model(unet_path, weight_quantization_config)

        components = {
            "vae_encoder": new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            "vae_decoder": new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "text_encoder": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "text_encoder_2": new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name,
        }

        for key, value in components.items():
            components[key] = cls.load_model(value, weight_quantization_config) if value.is_file() else None

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        if quantization_config is not None and quantization_config.dataset is not None:
            sd_model = cls(unet=unet, config=config, model_save_dir=model_save_dir, **components, **kwargs)

            supported_pipelines = (
                OVStableDiffusionPipeline,
                OVStableDiffusionXLPipeline,
                OVLatentConsistencyModelPipeline,
            )
            if not isinstance(sd_model, supported_pipelines):
                raise NotImplementedError(f"Quantization in hybrid mode is not supported for {cls.__name__}")

            nsamples = quantization_config.num_samples if quantization_config.num_samples else 200
            unet_inputs = sd_model._prepare_unet_inputs(quantization_config.dataset, nsamples)

            from .quantization import _hybrid_quantization

            unet = _hybrid_quantization(sd_model.unet.model, weight_quantization_config, dataset=unet_inputs)

        return cls(
            unet=unet,
            config=config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **components,
            **kwargs,
        )

    def _prepare_unet_inputs(
        self,
        dataset: Union[str, List[Any]],
        num_samples: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        self.compile()

        size = self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        height = height or min(size, 512)
        width = width or min(size, 512)

        if isinstance(dataset, str):
            dataset = deepcopy(dataset)
            available_datasets = PREDEFINED_SD_DATASETS.keys()
            if dataset not in available_datasets:
                raise ValueError(
                    f"""You have entered a string value for dataset. You can only choose between
                    {list(available_datasets)}, but the {dataset} was found"""
                )

            from datasets import load_dataset

            dataset_metadata = PREDEFINED_SD_DATASETS[dataset]
            dataset = load_dataset(dataset, split=dataset_metadata["split"], streaming=True).shuffle(seed=seed)
            input_names = dataset_metadata["inputs"]
            dataset = dataset.select_columns(list(input_names.values()))

            def transform_fn(data_item):
                return {inp_name: data_item[column] for inp_name, column in input_names.items()}

        else:

            def transform_fn(data_item):
                return data_item if isinstance(data_item, (list, dict)) else [data_item]

        from .quantization import InferRequestWrapper

        calibration_data = []
        self.unet.request = InferRequestWrapper(self.unet.request, calibration_data)

        for inputs in dataset:
            inputs = transform_fn(inputs)
            if isinstance(inputs, dict):
                self.__call__(**inputs, height=height, width=width)
            else:
                self.__call__(*inputs, height=height, width=width)
            if len(calibration_data) > num_samples:
                break

        self.unet.request = self.unet.request.request
        return calibration_data[:num_samples]

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
        tokenizer: Optional["CLIPTokenizer"] = None,
        scheduler: Union["DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"] = None,
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
            ov_config=ov_config,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=False,
            use_auth_token=use_auth_token,
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
                and self.device.lower().split(":")[0] == "gpu"
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
