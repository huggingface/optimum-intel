#  Copyright 2024 The HuggingFace Team. All rights reserved.
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


import inspect
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Union

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.candidate_generator import _crop_past_key_values
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.auto_factory import _get_model_class as get_model_class

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel
from optimum.utils import NormalizedConfigManager

from ...exporters.ipex.cache_utils import IPEXPagedCache
from ...exporters.ipex.model_patcher import (
    _IPEX_EXPORTED_GENERATION_TASKS,
    _IPEX_MINIMUM_VERSION_FOR_PATCHING,
    _patch_model,
)
from ..utils.constant import _TASK_ALIASES
from ..utils.import_utils import is_ipex_version, is_torch_version, is_transformers_version
from ..utils.modeling_utils import recursive_to_device


logger = logging.getLogger(__name__)


_IPEX_SUPPORT_MODEL_TYPES = ("llama", "bert", "vit", "falcon", "gpt2", "qwen2", "mistral")
_IPEX_EXPORTED_GENERATION_METHODS = ("sample", "greedy_search", "beam_sample", "beam_search", "assisted_generation")
_IPEX_MINIMUM_VERSION_FOR_COMPILE = "2.5.0"
# Page attention model cannot use torch.compile for now.
if is_torch_version("<", "2.6"):
    _COMPILE_NOT_READY_MODEL_TYPES = ("electra", "roformer", "gpt_neox", "beit", "llama", "falcon", "gpt2", "qwen2")
elif is_torch_version("<", "2.7"):
    _COMPILE_NOT_READY_MODEL_TYPES = ("llama", "falcon", "gpt2", "qwen2", "mistral")
else:
    _COMPILE_NOT_READY_MODEL_TYPES = ("mistral",)


try:
    import intel_extension_for_pytorch as ipex

    if hasattr(torch, "xpu") and torch.xpu.is_available() and not ipex._C._has_xpu():
        logger.warning(
            "Detect you have XPU device but the ipex do not support XPU, please install a xpu version ipex by checking https://pytorch-extension.intel.com/installation?platform=gpu"
        )
except ImportError:
    logger.warning("No intel_extension_for_pytorch found, please `pip install intel_extension_for_pytorch`")


def _is_patched_with_ipex(model, task, use_cache: bool = True):
    if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
        return False
    if not use_cache and task in _IPEX_EXPORTED_GENERATION_TASKS:
        return False
    return model.config.model_type in _IPEX_SUPPORT_MODEL_TYPES


def get_float_type(model_dtype: torch.dtype):
    if model_dtype == torch.bfloat16:
        return "bf16"
    elif model_dtype == torch.float16:
        return "fp16"
    else:
        return "fp32"


def prepare_jit_inputs(model: PreTrainedModel, task: str, use_cache: bool = False):
    task = _TASK_ALIASES.get(task, task)
    signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.__call__)
    onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
    float_dtype = get_float_type(model.dtype)
    if "text-generation" in task:
        onnx_config = onnx_config_class(
            model.config, use_past=use_cache, use_past_in_inputs=use_cache, float_dtype=float_dtype
        )
    else:
        onnx_config = onnx_config_class(model.config)

    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")

    return {
        key: recursive_to_device(dummy_inputs[key], model.device)
        for key in signature.parameters
        if dummy_inputs.get(key, None) is not None
    }


class IPEXModel(OptimizedModel):
    auto_model_class = AutoModel
    export_feature = "feature-extraction"
    base_model_prefix = "ipex_model"
    main_input_name = "input_ids"
    output_name = "last_hidden_state"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        config = config or model.config
        OptimizedModel.__init__(self, model=model, config=config)

        self._supports_cache_class = getattr(model, "_supports_cache_class", None)
        self._supports_sdpa = getattr(model, "_supports_sdpa", None)
        self._supports_quantized_cache = getattr(model, "_supports_quantized_cache", None)
        self._supports_static_cache = getattr(model, "_supports_static_cache", None)
        self._dtype = self.model.dtype if self.model.dtype is not None else torch.float32
        self.use_cache = kwargs.get("use_cache", False)
        self.model_save_dir = model_save_dir
        self._add_patch = _is_patched_with_ipex(model, self.export_feature, self.use_cache)
        self.model.config.compile = self.can_compile()

        self.input_names = set(inspect.signature(model.forward).parameters)

        if self._add_patch:
            model = _patch_model(model)
        # Registers the IPEXModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

        if getattr(self.model.config, "compile", False):
            self.apply_torch_compile()

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
        """

        model = cls.auto_model_class.from_pretrained(model_id, **kwargs)
        if getattr(model.config, "torchscript", False):
            raise ValueError("IPEXModel is no longer support torchscript models.")
        return cls(model, config=kwargs.pop("config", model.config), **kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        self.model.save_pretrained(save_directory, safe_serialization=False)

    def push_to_hub(self, *args, **kwargs):
        kwargs["safe_serialization"] = False
        return self.model.push_to_hub(*args, **kwargs)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def model_dtype(self):
        logger.warning(
            "access to the `model_dtype` attribute is deprecated and will be removed after v1.18.0, please use `_dtype` instead."
        )
        return self._dtype

    @property
    def add_patch(self) -> bool:
        return self._add_patch

    def to(self, device: Union[torch.device, str]):
        self.model.to(device)
        return self

    def can_generate(self):
        return isinstance(self, GenerationMixin)

    def can_compile(self):
        if (
            self.model.device.type != "cpu"
            or self.model.config.model_type in _COMPILE_NOT_READY_MODEL_TYPES
            or is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_COMPILE)
            or getattr(self.model.config, "quantization_config", None)
        ):
            return False

        if self.use_cache and not self.model._supports_cache_class and not self._add_patch:
            return False

        return True

    def apply_torch_compile(self):
        from torch._inductor import config as inductor_config

        # System level optimization
        # Patched model can disable cpp wrapper to get better performance.
        inductor_config.cpp_wrapper = False if self._add_patch and self.export_feature == "text-generation" else True
        os.environ["TORCHINDUCTOR_FREEZING"] = "1"
        logger.info("Enable torch.compile optimization")
        self.model.forward = torch.compile(self.model.forward)


class IPEXModelForSequenceClassification(IPEXModel):
    auto_model_class = AutoModelForSequenceClassification
    export_feature = "text-classification"
    output_name = "logits"


class IPEXModelForTokenClassification(IPEXModel):
    auto_model_class = AutoModelForTokenClassification
    export_feature = "token-classification"
    output_name = "logits"


class IPEXModelForMaskedLM(IPEXModel):
    auto_model_class = AutoModelForMaskedLM
    export_feature = "fill-mask"
    output_name = "logits"


class IPEXModelForImageClassification(IPEXModel):
    auto_model_class = AutoModelForImageClassification
    export_feature = "image-classification"


class IPEXModelForAudioClassification(IPEXModel):
    auto_model_class = AutoModelForAudioClassification
    export_feature = "audio-classification"


class IPEXModelForQuestionAnswering(IPEXModel):
    auto_model_class = AutoModelForQuestionAnswering
    export_feature = "question-answering"


class IPEXModelForCausalLM(IPEXModel, GenerationMixin):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(model, config, model_save_dir=model_save_dir, use_cache=use_cache)
        if self._add_patch:
            self._supports_cache_class = True
        GenerationMixin.__init__(self)

        model_type = self.config.model_type.replace("_", "-")
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(model_type)(self.config)

        self.config.is_decoder = True
        self.config.is_encoder_decoder = False

        self.generation_config = GenerationConfig.from_model_config(self.config)
        try:
            self.model_cls = get_class_from_dynamic_module(
                self.config.auto_map["AutoModelForCausalLM"], model_save_dir
            )
        except AttributeError:
            self.model_cls = get_model_class(self.config, AutoModelForCausalLM._model_mapping)

        if hasattr(self.model_cls, "_convert_to_standard_cache"):
            self._convert_to_standard_cache = self.model_cls._convert_to_standard_cache
        if hasattr(self.model_cls, "_convert_to_bloom_cache"):
            self._convert_to_bloom_cache = self.model_cls._convert_to_bloom_cache

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.add_patch:
            if input_ids is not None and attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            kwargs = self.prepare_page_attn_inputs(input_ids, attention_mask, **kwargs)

        results = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if self.add_patch and self.use_cache and results.get("past_key_values", None) is not None:
            self.postprocess_ipex_paged_cache(results["past_key_values"], kwargs["input_lens"])

        return results

    def prepare_page_attn_inputs(self, input_ids, attention_mask, **kwargs):
        if not hasattr(self, "batch_size") or (input_ids.shape[0] != getattr(self, "batch_size", 0)):
            self.batch_size = input_ids.shape[0]
            self.decode_index = torch.arange(self.batch_size, dtype=torch.int).to(input_ids.device)
            self.decode_query_len_tensor = torch.arange(self.batch_size + 1, dtype=torch.int).to(input_ids.device)

        kwargs["input_lens"] = attention_mask.sum(-1).to(torch.int32)
        kwargs["seq_len_tensor"] = torch.cat(
            (kwargs["input_lens"].new_tensor([0]), kwargs["input_lens"].cumsum(-1).int())
        )
        kwargs["query_len_tensor"] = (
            kwargs["seq_len_tensor"].clone() if input_ids.shape[-1] != 1 else self.decode_query_len_tensor
        )
        if self.use_cache and kwargs.get("past_key_values", None) is not None:
            self.preprocess_ipex_paged_cache(kwargs["past_key_values"], kwargs["input_lens"])

        kwargs["index"] = (
            attention_mask.view(-1).nonzero().squeeze().int() if input_ids.shape[-1] != 1 else self.decode_index
        )
        # The int value will be recognized as constant if we pass it in the forward.
        # To avoid recompile, we pass the int value through config so the torch.compile will not recognize it as constant.
        self.model.config.max_input_lens = kwargs["input_lens"].max().item()

        return kwargs

    def preprocess_ipex_paged_cache(self, past_key_values, input_lens):
        batch_size = input_lens.shape[0]
        past_key_values_length = past_key_values.get_seq_length()
        if past_key_values_length == 0:
            past_key_values.alloc_slot_for_prefill(input_lens, batch_size)
        else:
            past_key_values.alloc_slot_for_decode(batch_size)

    def postprocess_ipex_paged_cache(self, past_key_values, input_lens):
        past_key_values_length = past_key_values.get_seq_length()
        # Use inplace op to keep the same memory address, avoid recompile
        if past_key_values_length == 0:
            past_key_values._seen_tokens = past_key_values._seen_tokens.add_(input_lens)
        else:
            past_key_values._seen_tokens = past_key_values._seen_tokens.add_(1)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        kwargs["use_cache"] = self.use_cache
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_method = generation_config.get_generation_mode().value
        if (
            getattr(self.model.config, "compile", False)
            and generation_config.cache_implementation != "ipex_paged"
            and self._supports_static_cache
        ):
            # Use static cache for torch compile
            generation_config.cache_implementation = "static"
        if generation_method not in _IPEX_EXPORTED_GENERATION_METHODS:
            raise ValueError(
                f"The generation method {generation_method} is not supported for IPEXModelForCausalLM for now, support methods are {_IPEX_EXPORTED_GENERATION_METHODS}"
            )

        return generation_config, model_kwargs

    def _reorder_cache(self, *args, **kwargs):
        return self.model._reorder_cache(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _supports_logits_to_keep(self) -> bool:
        """
        Return True if the current model supports the keyword argument `logits_to_keep` in forward()
        to save memory. Checking it in this way allows to avoid using a new model attribute.
        """
        return "logits_to_keep" in set(inspect.signature(self.model.forward).parameters.keys())

    def _supports_num_logits_to_keep(self) -> bool:
        """
        Will be deprecated after we no longer support transformers < 4.49

        Return True if the current model supports the keyword argument `num_logits_to_keep` in forward()
        to save memory. Checking it in this way allows to avoid using a new model attribute.
        """
        return "num_logits_to_keep" in set(inspect.signature(self.model.forward).parameters.keys())

    def generate(self, *args, **kwargs):
        if self._add_patch and kwargs.get("assistant_model", None):
            raise ValueError(
                f"Assisted decoding is not supported for patched models for now, support methods are {_IPEX_EXPORTED_GENERATION_METHODS}"
            )
        # Patch functions to support ipex_paged cache
        if self._add_patch:
            transformers.generation.utils.NEED_SETUP_CACHE_CLASSES_MAPPING["ipex_paged"] = IPEXPagedCache
            self.generation_config.cache_implementation = "ipex_paged"
            if is_transformers_version(">=", "4.45.0"):
                if "ipex_paged" not in transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS:
                    transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS.append("ipex_paged")
            if kwargs.get("generation_config", None):
                # Change cache implementation temporarily
                orig_cache_implementation = kwargs["generation_config"].cache_implementation
                kwargs["generation_config"].cache_implementation = "ipex_paged"

        if self._add_patch and kwargs.get("assistant_model", None):
            transformers.generation.utils._crop_past_key_values = _ipex_crop_past_key_values
        elif self._add_patch:
            transformers.generation.candidate_generator._crop_past_key_values = _ipex_crop_past_key_values

        try:
            result = super().generate(*args, **kwargs)
        except Exception as e:
            transformers.generation.utils._crop_past_key_values = _crop_past_key_values
            transformers.generation.candidate_generator._crop_past_key_values = _crop_past_key_values
            raise e

        if self._add_patch and kwargs.get("assistant_model", None):
            transformers.generation.utils._crop_past_key_values = _crop_past_key_values
            transformers.generation.candidate_generator._crop_past_key_values = _crop_past_key_values

        # change back cache_implementation
        if self._add_patch and kwargs.get("generation_config", None):
            kwargs["generation_config"].cache_implementation = orig_cache_implementation

        return result


class IPEXModelForSeq2SeqLM(IPEXModel, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    export_feature = "text2text-generation"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(model, config, model_save_dir=model_save_dir, use_cache=use_cache)
        GenerationMixin.__init__(self)

        model_type = self.config.model_type.replace("_", "-")
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(model_type)(self.config)

        self.config.is_decoder = False
        self.config.is_encoder_decoder = True

        self.generation_config = GenerationConfig.from_model_config(self.config)
        try:
            self.model_cls = get_class_from_dynamic_module(
                self.config.auto_map["AutoModelForSeq2SeqLM"], model_save_dir
            )
        except AttributeError:
            self.model_cls = get_model_class(self.config, AutoModelForSeq2SeqLM._model_mapping)

        if hasattr(self.model_cls, "_convert_to_standard_cache"):
            self._convert_to_standard_cache = self.model_cls._convert_to_standard_cache

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        # Use static cache for torch.compile
        if getattr(self.model.config, "compile", False):
            generation_config.cache_implementation = "static"

        return generation_config, model_kwargs

    def _reorder_cache(self, *args, **kwargs):
        return self.model._reorder_cache(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def get_encoder(self, *args, **kwargs):
        return self.model.get_encoder(*args, **kwargs)

    def _supports_logits_to_keep(self) -> bool:
        """
        Return True if the current model supports the keyword argument `logits_to_keep` in forward()
        to save memory. Checking it in this way allows to avoid using a new model attribute.
        """
        return "logits_to_keep" in set(inspect.signature(self.model.forward).parameters.keys())

    def _supports_num_logits_to_keep(self) -> bool:
        """
        Will be deprecated after we no longer support transformers < 4.49

        Return True if the current model supports the keyword argument `num_logits_to_keep` in forward()
        to save memory. Checking it in this way allows to avoid using a new model attribute.
        """
        return "num_logits_to_keep" in set(inspect.signature(self.model.forward).parameters.keys())


def _ipex_crop_past_key_values(model, past_key_values, max_length):
    if isinstance(model, IPEXModel) and _is_patched_with_ipex(model, "text-generation"):
        if isinstance(past_key_values, IPEXPagedCache):
            # .crop is an inplace op, returns None
            past_key_values = past_key_values.crop(max_length)
            return past_key_values
        else:
            raise ValueError("only support IPEXPagedCache input now")
    return _crop_past_key_values(model, past_key_values, max_length)
