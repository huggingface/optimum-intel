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
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.candidate_generator import _crop_past_key_values
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.auto_factory import _get_model_class as get_model_class

from optimum.modeling_base import OptimizedModel
from optimum.utils import NormalizedConfigManager

from ...exporters.ipex.cache_utils import IPEXPagedCache
from ...exporters.ipex.model_patcher import (
    _IPEX_EXPORTED_GENERATION_TASKS,
    _IPEX_MINIMUM_VERSION_FOR_PATCHING,
    _patch_model,
)
from ..generation.modeling import prepare_jit_inputs
from ..utils.import_utils import is_ipex_version, is_transformers_version


logger = logging.getLogger(__name__)


_IPEX_SUPPORT_MODEL_TYPES = ("llama", "bert", "vit", "falcon", "gpt2")
_IPEX_EXPORTED_GENERATION_METHODS = ("sample", "greedy_search", "beam_sample", "beam_search", "assisted_generation")
_IPEX_MINIMUM_VERSION_FOR_COMPILE = "2.5.0"
# TODO: Some models are already fixed in torch 2.6, will enable them when torch upgrading to 2.6
_COMPILE_NOT_READY_MODEL_TYPES = ("electra", "roformer", "gpt_neox", "beit", "llama", "falcon", "gpt2")


def _is_patched_with_ipex(model, task, use_cache: bool = True):
    if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
        return False
    if not use_cache and task in _IPEX_EXPORTED_GENERATION_TASKS:
        return False
    return model.config.model_type in _IPEX_SUPPORT_MODEL_TYPES


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
        warmup: Optional[bool] = True,
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
        self.compiled = False

        self.input_names = set(inspect.signature(model.forward).parameters)

        if self._add_patch:
            model = _patch_model(model)
        # Registers the IPEXModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

        self.maybe_apply_torch_compile()

        if warmup:
            self._init_warmup()

    @classmethod
    def _from_transformers(cls, *args, **kwargs):
        return cls._from_pretrained(*args, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
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
        if getattr(config, "torchscript", False):
            raise ValueError("IPEXModel is no longer support torchscript models.")

        model = cls.auto_model_class.from_pretrained(model_id, **kwargs)
        return cls(model, config=model.config, **kwargs)

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

    def maybe_apply_torch_compile(self):
        if (
            self.model.device.type != "cpu"
            or self.config.model_type in _COMPILE_NOT_READY_MODEL_TYPES
            or is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_COMPILE)
        ):
            return
        if self.use_cache and not self._supports_static_cache:
            return
        from torch._inductor import config as inductor_config

        # System level optimization
        inductor_config.cpp_wrapper = True
        os.environ["TORCHINDUCTOR_FREEZING"] = "1"
        logger.info("Enable torch.compile optimization")
        self.model.forward = torch.compile(self.model.forward)
        self.compiled = True

    def _init_warmup(self):
        inputs = prepare_jit_inputs(self.model, self.export_feature, False)
        with torch.no_grad():
            self.model(**inputs)
            self.model(**inputs)
        logger.info("Warm up end")


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
        warmup: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(model, config, model_save_dir=model_save_dir, warmup=False, use_cache=use_cache)
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

        if warmup:
            self._init_warmup()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.add_patch and input_ids is not None and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        generation_method = generation_config.get_generation_mode().value
        if self.compiled and generation_config.cache_implementation != "ipex_paged" and self._supports_static_cache:
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

    def _init_warmup(self):
        inputs = prepare_jit_inputs(self.model, self.export_feature, False)
        self.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=4)
        self.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=4)
        logger.info("Warm up end")


class IPEXModelForSeq2SeqLM(IPEXModel, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    export_feature = "text2text-generation"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        warmup: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(model, config, model_save_dir=model_save_dir, warmup=False, use_cache=use_cache)
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

        if warmup:
            self._init_warmup()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        # Use static cache for torch.compile
        if self.compiled:
            generation_config.cache_implementation = "static"

        return generation_config, model_kwargs

    def _reorder_cache(self, *args, **kwargs):
        return self.model._reorder_cache(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def get_encoder(self, *args, **kwargs):
        return self.model.get_encoder(*args, **kwargs)

    def _init_warmup(self):
        inputs = prepare_jit_inputs(self.model, self.export_feature, False)
        self.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=4)
        self.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=4)
        logger.info("Warm up end")


def _ipex_crop_past_key_values(model, past_key_values, max_length):
    if isinstance(model, IPEXModel) and _is_patched_with_ipex(model, "text-generation"):
        if isinstance(past_key_values, IPEXPagedCache):
            # .crop is an inplace op, returns None
            past_key_values = past_key_values.crop(max_length)
            return past_key_values
        else:
            raise ValueError("only support IPEXPagedCache input now")
    return _crop_past_key_values(model, past_key_values, max_length)
