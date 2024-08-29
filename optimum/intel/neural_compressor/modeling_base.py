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

import logging
import os
import types
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import EntryNotFoundError
from neural_compressor.transformers import GPTQConfig, RtnConfig
from neural_compressor.transformers.quantization import convert_to_quantized_model, save_low_bit
from neural_compressor.utils.pytorch import load
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.utils.generic import ContextManagers

from optimum.intel.generation import BaseModelForCausalLM

from ...modeling_base import OptimizedModel
from ..utils.import_utils import (
    _torch_version,
    is_ipex_version,
    is_neural_compressor_version,
    is_torch_version,
)
from .configuration import INCConfig
from .utils import QUANTIZATION_CONFIG_NAME


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.propagate = False
logger.addHandler(handler)
logger.setLevel(logging.INFO)

MODEL_START_DOCSTRING = r"""
    This model check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        model (`PyTorch model`): is the main class used to run inference.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is the Model configuration class with all the parameters of the model.
        device (`str`, defaults to `"cpu"`):
            The device type for which the model will be optimized for. The resulting compiled model will contains nodes specific to this device.
"""


class INCModel(OptimizedModel):
    auto_model_class = AutoModel
    export_feature = "feature-extraction"
    base_model_prefix = "inc_model"
    _supports_cache_class = False

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        q_config: Dict = None,
        inc_config: Dict = None,
        **kwargs,
    ):
        super().__init__(model=model, config=config, **kwargs)
        self.inc_config = inc_config
        self._q_config = q_config
        self.model_save_dir = model_save_dir
        self._device = getattr(self.model, "device", None) or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.generation_config = GenerationConfig.from_model_config(config)

        # Registers the INCModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: str = WEIGHTS_NAME,
        local_files_only: bool = False,
        subfolder: str = "",
        trust_remote_code: bool = False,
        *model_args,
        **kwargs,
    ):
        device_map = kwargs.get("device_map", "cpu")
        use_xpu = True if device_map == torch.device("xpu") or device_map == "xpu" else False

        quantization_config = kwargs.pop("quantization_config", None)
        if not isinstance(config, PretrainedConfig):
            config, _ = AutoConfig.from_pretrained(
                model_id,
                return_unused_kwargs=True,
                **kwargs,
            )
        if hasattr(config, "quantization_config"):
            if config.quantization_config is None:
                logger.warning(
                    "Quantization_config loading failed. If you want to load saved "
                    "low bit model, please check your quantizate_config.json."
                )
            else:
                logger.info("quantization_config: {}".format(config.quantization_config))

                try:
                    from neural_compressor.transformers.models.modeling_auto import (
                        _BaseINCAutoModelClass,
                    )

                    _BaseINCAutoModelClass.ORIG_MODEL = cls.auto_model_class
                    model = _BaseINCAutoModelClass.load_low_bit(
                        model_id,
                        subfolder=subfolder,
                        revision=revision,
                        cache_dir=cache_dir,
                        token=token,
                        local_files_only=local_files_only,
                        force_download=force_download,
                        trust_remote_code=trust_remote_code,
                        config=config,
                        **kwargs,
                    )
                    logger.info("Saved low bit model loading successfully. Other input args " "will be ignored.")
                    return model
                except Exception as e:
                    logger.error(e)
                    logger.error("Saved low bit model loading failed, please check your model.")
                    exit(0)
        if isinstance(quantization_config, (RtnConfig, GPTQConfig)):
            logger.info("Applying Weight Only Quantization.")
            warnings.warn(
                "Weight only quantization provided by intel_extension_for_transformers is deprecated and it is provided by INC now.",
                DeprecationWarning,
            )
            if is_neural_compressor_version("<=", "3.0"):
                raise AssertionError("Please use neural_compressor version > 3.0.")
            if is_ipex_version("<", "2.3.1") and use_xpu:
                raise AssertionError("Please use intel_extension_for_pytorch version >= 2.3.1.")

            if use_xpu:
                # TODO: if low_cpu_mem_uasge is True, gptj will have accuracy issue on CPU device.
                kwargs["low_cpu_mem_usage"] = True
                kwargs["device_map"] = "cpu"
                try:
                    model = cls.auto_model_class.from_pretrained(
                        model_id,
                        *model_args,
                        config=config,
                        **kwargs,
                    )
                    model.config.update({"low_cpu_mem_usage": True})
                except NotImplementedError:
                    logger.info(
                        "Failed to load models with `low_cpu_mem_usage` specified, "
                        "will fall to traditional load method with higher memory consumption."
                    )
                    kwargs["low_cpu_mem_usage"] = False
                    model = cls.auto_model_class.from_pretrained(
                        model_id,
                        *model_args,
                        config=config,
                        **kwargs,
                    )
                    model.config.update({"low_cpu_mem_usage": False})
                    quantization_config.post_init_xpu()
            else:
                kwargs["low_cpu_mem_usage"] = True
                model = cls.auto_model_class.from_pretrained(
                    model_id,
                    *model_args,
                    config=config,
                    **kwargs,
                )
                model.config.update({"low_cpu_mem_usage": True})
                quantization_config.post_init_cpu()
            model.eval()

            if use_xpu:
                assert hasattr(torch, "xpu") and torch.xpu.is_available(), "There is no xpu device in this system!"
                quantization_config.update(**{"device": "xpu"})
                quantization_config.post_init_xpu()
            if (
                not torch.cuda.is_available() or device_map == "cpu" or device_map == torch.device("cpu")
            ) and model.config.model_type == "chatglm":
                model = model.float()
            model = convert_to_quantized_model(model, quantization_config, device=device_map)
            quantization_config.remove_redundant_parameters()
            model.config.quantization_config = quantization_config
            # add quantization_config and save_low_bit to pretrained model dynamically
            model.device_map = device_map
            model.quantization_config = quantization_config
            model.save_pretrained = types.MethodType(save_low_bit, model)
            logger.info("WeightOnlyQuant done.")
            return model
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        model_path = Path(model_id)
        is_local = model_path.is_dir()
        model_cache_path = None
        inc_config = None
        msg = None
        if is_local:
            if (model_path / subfolder / SAFE_WEIGHTS_NAME).is_file():
                file_name = SAFE_WEIGHTS_NAME
            elif not (model_path / subfolder / file_name).is_file():
                raise EnvironmentError(
                    f"Error no file named {SAFE_WEIGHTS_NAME} or {file_name} found in directory {model_path / subfolder}"
                )
            model_cache_path = model_path / subfolder / file_name
        else:
            # Try download safetensors if exist
            try:
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFE_WEIGHTS_NAME,
                    subfolder=subfolder,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except EntryNotFoundError:
                pass

            if model_cache_path is None:
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    subfolder=subfolder,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )

        model_save_dir = Path(model_cache_path).parent

        try:
            inc_config = INCConfig.from_pretrained(model_id, subfolder=subfolder, revision=revision)
            if not is_torch_version("==", inc_config.torch_version):
                msg = f"Quantized model was obtained with torch version {inc_config.torch_version} but {_torch_version} was found."
                logger.warning(f"{msg}")
        except EnvironmentError:
            msg = (
                f"Please check if torch quantization the model was obtained with is compatible with {_torch_version}."
            )

        if getattr(config, "backend", None) == "ipex" or getattr(config, "torchscript", False):
            logger.warning(
                f"Using `{cls.__name__}` to load a TorchScript model will be deprecated in v1.15.0, to load your model please use `{cls.__name__.replace('INC', 'IPEX')}` instead."
            )
            model = torch.jit.load(model_cache_path)
            model = torch.jit.freeze(model.eval())
            return cls(model, config=config, model_save_dir=model_save_dir, inc_config=inc_config, **kwargs)

        model_class = _get_model_class(config, cls.auto_model_class._model_mapping)
        # Load the state dictionary of the model to verify whether the model to get the quantization config
        state_dict = torch.load(model_cache_path, map_location="cpu")

        q_config = state_dict.get("best_configure", None)
        if q_config is None:
            model = model_class.from_pretrained(model_save_dir)
        else:
            init_contexts = [no_init_weights(_enable=False)]
            with ContextManagers(init_contexts):
                model = model_class(config)
            try:
                model = load(model_cache_path, model)
            except Exception as e:
                # For incompatible torch version check
                if msg is not None:
                    e.args += (msg,)
                raise

        return cls(
            model, config=config, model_save_dir=model_save_dir, q_config=q_config, inc_config=inc_config, **kwargs
        )

    def _save_pretrained(self, save_directory: Union[str, Path]):
        if isinstance(self.model, torch.nn.Module):
            # For INC weight only model
            if isinstance(self._q_config, PretrainedConfig):
                self._q_config.to_json_file(os.path.join(save_directory, QUANTIZATION_CONFIG_NAME))
                self.model.save_pretrained(save_directory)
            # For INC model the state dictionary needs to be modified to include the quantization parameters
            else:
                state_dict = self.model.state_dict()
                if isinstance(self._q_config, dict):
                    state_dict["best_configure"] = self._q_config
                torch.save(state_dict, os.path.join(save_directory, WEIGHTS_NAME))
        else:
            torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))

        if self.inc_config:
            self.inc_config.save_pretrained(save_directory)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def can_generate(self):
        return isinstance(self.model, GenerationMixin)

    def generate(self, *args, **kwargs):
        if not self.can_generate():
            raise TypeError(
                f"The current model class {self.model.__class__} is not compatible with `.generate()`, as it doesn't have a language model head."
            )
        return self.model.generate(*args, **kwargs)


class INCModelForQuestionAnswering(INCModel):
    auto_model_class = AutoModelForQuestionAnswering
    export_feature = "question-answering"


class INCModelForSequenceClassification(INCModel):
    auto_model_class = AutoModelForSequenceClassification
    export_feature = "text-classification"


class INCModelForTokenClassification(INCModel):
    auto_model_class = AutoModelForTokenClassification
    export_feature = "token-classification"


class INCModelForMultipleChoice(INCModel):
    auto_model_class = AutoModelForMultipleChoice
    export_feature = "multiple-choice"


class INCModelForSeq2SeqLM(INCModel):
    auto_model_class = AutoModelForSeq2SeqLM
    export_feature = "text2text-generation"


class INCModelForMaskedLM(INCModel):
    auto_model_class = AutoModelForMaskedLM
    export_feature = "fill-mask"


class INCModelForVision2Seq(INCModel):
    auto_model_class = AutoModelForVision2Seq
    export_feature = "image-to-text"


class INCModelForCausalLM(INCModel, BaseModelForCausalLM):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    forward = BaseModelForCausalLM.forward
    generate = BaseModelForCausalLM.generate
    can_generate = BaseModelForCausalLM.can_generate

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        q_config: Dict = None,
        inc_config: Dict = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super(INCModelForCausalLM, self).__init__(
            model=model,
            config=config,
            model_save_dir=model_save_dir,
            q_config=q_config,
            inc_config=inc_config,
            use_cache=use_cache,
            **kwargs,
        )
