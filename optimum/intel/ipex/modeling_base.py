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
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Union

import intel_extension_for_pytorch as ipex
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from intel_extension_for_pytorch.cpu._auto_kernel_selection import _enable_tpp
from intel_extension_for_pytorch.transformers.optimize import get_dummy_input
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
    is_torch_xpu_available,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.auto.auto_factory import _get_model_class as get_model_class
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel
from optimum.utils import NormalizedConfigManager

from ...exporters.ipex.model_patcher import _IPEX_EXPORTED_TASK, _IPEX_MINIMUM_VERSION_FOR_PATCHING, _patch_model
from ..generation.modeling import prepare_jit_inputs
from ..utils.import_utils import is_ipex_version, is_torch_version, is_transformers_version
from ..utils.modeling_utils import MULTI_QUERY_ATTN_MODELS, patch_decoder_attention_mask, recursive_to_device


logger = logging.getLogger(__name__)


_IPEX_SUPPORT_MODEL_TYPES = ("llama",)
_IPEX_EXPORTED_GENERATION_METHODS = ("sample", "greedy_search", "beam_sample", "beam_search", "assisted_generation")


def _is_patched_with_ipex(model, task):
    if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
        return False

    if isinstance(model, torch.jit.ScriptModule):
        for node in model.graph.nodes():
            # Jit will record the codes position so we can check if the node use ipex exporter.
            if "torch_ipex::rotary_position_embedding" in node.__str__():
                return True
        return False
    else:
        # The ipex IAKV op in patched model requires the hidden size at least 64
        return (
            model.config.model_type in _IPEX_SUPPORT_MODEL_TYPES
            and task in _IPEX_EXPORTED_TASK
            and model.config.hidden_size >= 64
        )


def ipex_jit_trace(model, task, use_cache):
    # Only support torch version >= 2.1.0 to support example_kwarg_inputs in jit.trace
    if is_torch_version("<", "2.1.0"):
        raise ImportError("`torch>=2.1.0` is needed to trace your model")

    if _is_patched_with_ipex(model, task):
        model = _patch_model(model)
        # TODO: integerate in prepare_jit_inputs.
        sample_inputs = get_dummy_input(model, return_dict=True)
        # Use Tensor Processing Primitives to accelerate linear, see https://arxiv.org/abs/2104.05755.
        _enable_tpp()
    else:
        model = patch_decoder_attention_mask(model)
        sample_inputs = prepare_jit_inputs(model, task, use_cache)

    model.config.return_dict = False

    if "past_key_values" in sample_inputs:
        model.config.use_cache = use_cache
        if not use_cache:
            sample_inputs.pop("past_key_values")

    model = ipex.optimize(model.eval(), dtype=model.dtype, inplace=True)
    # Disable repack while jit tracing to reduce the memory
    ipex._C.disable_jit_linear_repack()
    with torch.no_grad():
        trace_model = torch.jit.trace(
            model,
            example_kwarg_inputs=sample_inputs,
            strict=False,
            check_trace=False,
        )
        trace_model = torch.jit.freeze(trace_model)
        trace_model(**sample_inputs)
        trace_model(**sample_inputs)

    return trace_model


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
        export: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        warmup: bool = True,
        **kwargs,
    ):
        if is_torch_xpu_available(check_device=True):
            self._device = torch.device("xpu:0")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        # CPU only support jit model for now.
        if export:
            if isinstance(model, torch.jit.RecursiveScriptModule):
                logger.warning("The model has been exported already.")
            else:
                config = model.config if config is None else config
                use_cache = kwargs.get("use_cache", True)
                model = ipex_jit_trace(model, self.export_feature, use_cache)
                config.torchscript = True

        OptimizedModel.__init__(self, model=model, config=config)

        self.model.to(self._device)
        self._dtype = self.config.torch_dtype if self.config.torch_dtype is not None else torch.float32
        self.model_save_dir = model_save_dir
        self._is_ipex_exported = _is_patched_with_ipex(model, self.export_feature)

        if isinstance(model, torch.jit.RecursiveScriptModule):
            self.input_names = {
                inputs.debugName().split(".")[0] for inputs in model.graph.inputs() if inputs.debugName() != "self"
            }
        else:
            self.input_names = set(inspect.signature(model.forward).parameters)

        # Registers the IPEXModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)
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
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        trust_remote_code: bool = False,
        file_name: Optional[str] = WEIGHTS_NAME,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "Both the arguments `use_auth_token` and `token` were specified, which is not supported. Please specify only `token`."
                )
            token = use_auth_token

        commit_hash = kwargs.pop("_commit_hash", None)

        model_kwargs = {
            "revision": revision,
            "token": token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }

        if not getattr(config, "torchscript", False):
            logger.warning("Detect torchscript is false. Convert to torchscript model!")

            if is_torch_version("<", "2.1.0"):
                raise ImportError("`torch>=2.0.0` is needed to trace your model")

            task = cls.export_feature
            config.torch_dtype = torch_dtype
            model = TasksManager.get_model_from_task(
                task,
                model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                _commit_hash=commit_hash,
                **model_kwargs,
            )

            return cls(model, config=config, export=True, **kwargs)

        # Load the model from local directory
        if os.path.isdir(model_id):
            model_cache_path = os.path.join(model_id, file_name)
            model_save_dir = model_id
        # Download the model from the hub
        else:
            model_cache_path = hf_hub_download(repo_id=model_id, filename=file_name, **model_kwargs)
            model_save_dir = Path(model_cache_path).parent

        model = torch.jit.load(model_cache_path)
        torch.jit.freeze(model.eval())

        return cls(model, config=config, model_save_dir=model_save_dir, **kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        output_path = os.path.join(save_directory, WEIGHTS_NAME)
        if getattr(self.config, "torchscript", None):
            torch.jit.save(self.model, output_path)
        else:
            logger.warning("The module is not a torchscript model, will be treated as a transformers model.")
            self.model.save_pretrained(output_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids

        outputs = self._call_model(**inputs)
        if isinstance(outputs, dict):
            model_output = ModelOutput(**outputs)
        else:
            model_output = ModelOutput()
            model_output[self.output_name] = outputs[0]
        return model_output

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def model_dtype(self):
        logger.warning(
            "access to the `model_dtype` attribute is deprecated and will be removed after v1.18.0, please use `_dtype` instead."
        )
        return self._dtype

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def can_generate(self):
        return isinstance(self, GenerationMixin)

    def _call_model(self, *args, **kwargs):
        try:
            with torch.autocast(self.device.type, self.dtype), torch.no_grad():
                out = self.model(*args, **kwargs)
        except RuntimeError:
            out = self.model(*args, **kwargs)
        return out

    def _init_warmup(self):
        # warmup, the first 2 forwards of an IPEX model include some preprocessing steps and
        # the results of the compute are unpredictable
        # TODO : add warmup for IPEX exported model
        if not self._is_ipex_exported:
            use_cache = "past_key_values" in self.input_names
            dummy_inputs = prepare_jit_inputs(self, self.export_feature, use_cache)
            if self._device.type != "cpu":
                dummy_inputs = recursive_to_device(value=dummy_inputs, device=self._device)
            for _ in range(2):
                self(**dummy_inputs)


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

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        inputs = {
            "pixel_values": pixel_values,
        }

        outputs = self._call_model(**inputs)
        return ModelOutput(**outputs) if isinstance(outputs, dict) else ModelOutput(logits=outputs[0])


class IPEXModelForAudioClassification(IPEXModel):
    auto_model_class = AutoModelForAudioClassification
    export_feature = "audio-classification"

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        inputs = {
            "input_values": input_values,
        }

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        outputs = self._call_model(**inputs)
        return ModelOutput(**outputs) if isinstance(outputs, dict) else ModelOutput(logits=outputs[0])


class IPEXModelForQuestionAnswering(IPEXModel):
    auto_model_class = AutoModelForQuestionAnswering
    export_feature = "question-answering"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids

        outputs = self._call_model(**inputs)
        start_logits = outputs["start_logits"] if isinstance(outputs, dict) else outputs[0]
        end_logits = outputs["end_logits"] if isinstance(outputs, dict) else outputs[1]
        return ModelOutput(start_logits=start_logits, end_logits=end_logits)


class IPEXModelForCausalLM(IPEXModel, GenerationMixin):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        export: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        warmup: bool = True,
        **kwargs,
    ):
        # Perform the initial warmup at the end of __init__
        super().__init__(
            model, config, export=export, model_save_dir=model_save_dir, warmup=False, use_cache=use_cache
        )
        GenerationMixin.__init__(self)

        model_type = self.config.model_type.replace("_", "-")
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(model_type)(self.config)
        self.use_cache = "past_key_values" in self.input_names

        if isinstance(model, torch.jit.RecursiveScriptModule) and use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False

        self.generation_config = GenerationConfig.from_model_config(self.config)
        try:
            self.model_cls = get_class_from_dynamic_module(
                self.config.auto_map["AutoModelForCausalLM"], model_save_dir
            )
        except AttributeError:
            self.model_cls = get_model_class(self.config, AutoModelForCausalLM._model_mapping)

        if self._is_ipex_exported:
            self._reorder_cache = _ipex_reorder_cache
        else:
            # Check if _reorder_cache is a static method
            if isinstance(self.model_cls.__dict__["_reorder_cache"], staticmethod):
                self._reorder_cache = self.model_cls._reorder_cache
            else:
                self._reorder_cache = self.model_cls._reorder_cache.__get__(self)

        if is_transformers_version(">=", "4.38.0") and model_type in {"llama", "phi", "persimmon"}:
            self.prepare_inputs_for_generation = _prepare_inputs_for_generation_for_llama
        else:
            self.prepare_inputs_for_generation = self.model_cls.prepare_inputs_for_generation.__get__(self)

        if hasattr(self.model_cls, "_convert_to_standard_cache"):
            self._convert_to_standard_cache = self.model_cls._convert_to_standard_cache
        if hasattr(self.model_cls, "_convert_to_bloom_cache"):
            self._convert_to_bloom_cache = self.model_cls._convert_to_bloom_cache
        if warmup:
            self._init_warmup()

    def _prepare_past_key_values(self, input_ids):
        model_type = self.config.model_type.replace("_", "-")
        nb_pkv = 2
        num_layers = self.normalized_config.num_layers
        d_k = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        batch_size = input_ids.shape[0]

        if model_type in {"mistral", "llama"}:
            num_attention_heads = self.normalized_config.num_key_value_heads
        else:
            num_attention_heads = self.normalized_config.num_attention_heads

        if self._is_ipex_exported:
            # Indirect access kv cache has a different data layout compared with most transformers model,
            # see https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/llm.html#indirect-access-kv-cache
            beam_idx_tmp = torch.zeros(
                (self.config.max_position_embeddings, input_ids.shape[0]), dtype=torch.long
            ).contiguous()
            past_key_values = tuple(
                [
                    (
                        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        beam_idx_tmp,
                    )
                    for i in range(num_layers)
                ]
            )
            return past_key_values
        elif model_type == "bloom":
            shape_key = (batch_size * num_attention_heads, d_k, 0)
            shape_value = (batch_size * num_attention_heads, 0, d_k)
            key = torch.empty(size=shape_key, dtype=self.model_dtype, device=self._device)
            value = torch.empty(size=shape_value, dtype=self.model_dtype, device=self._device)
            past_key_values = tuple(
                tuple(key if idx % 2 == 0 else value for idx in range(nb_pkv)) for _ in range(num_layers)
            )
        elif model_type.replace("-", "_") in MULTI_QUERY_ATTN_MODELS:
            shape = (batch_size, 0, d_k * 2)
            pkv = torch.empty(size=shape, dtype=self.model_dtype, device=self._device)
            past_key_values = tuple(pkv for _ in range(num_layers))
        else:
            shape = (batch_size, num_attention_heads, 0, d_k)
            pkv = torch.empty(size=shape, dtype=self.model_dtype, device=self._device)
            past_key_values = tuple(tuple(pkv for _ in range(nb_pkv)) for _ in range(num_layers))

        return past_key_values

    # Temporary fix, will delete when https://github.com/huggingface/transformers/pull/31226 release.
    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        if not model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = None
            return model_kwargs

        past_length = 0
        if "past_key_values" in model_kwargs:
            past_length = model_kwargs["past_key_values"][0][0].shape[-2]
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        else:
            cur_len = input_ids.shape[-1]
        model_kwargs["cache_position"] = torch.arange(past_length, cur_len, device=input_ids.device)
        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1. Prepare model inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "position_ids" in self.input_names or not self.input_names:
            inputs["position_ids"] = position_ids

        if self.use_cache:
            if past_key_values is None:
                past_key_values = self._prepare_past_key_values(input_ids)

            inputs["past_key_values"] = past_key_values

        # 2. Model forward
        outputs = self._call_model(**inputs)

        # 3. Process model outputs
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            past_key_values = outputs[1] if self.use_cache else None
        else:
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"] if self.use_cache else None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        generation_method = generation_config.get_generation_mode().value
        if generation_method not in _IPEX_EXPORTED_GENERATION_METHODS:
            raise ValueError(
                f"The generation method {generation_method} is not supported for IPEXModelForCausalLM for now, support methods are {_IPEX_EXPORTED_GENERATION_METHODS}"
            )

        return generation_config, model_kwargs

    def generate(self, *args, **kwargs):
        if self._is_ipex_exported and kwargs.get("assistant_model", None):
            raise ValueError(
                f"Assisted decoding is not supported for patched models for now, support methods are {_IPEX_EXPORTED_GENERATION_METHODS}"
            )
        return super().generate(*args, **kwargs)


def _prepare_inputs_for_generation_for_llama(
    input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    from transformers.cache_utils import Cache

    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def _ipex_reorder_cache(
    past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
) -> Tuple[Tuple[torch.Tensor]]:
    # Ipex patched model uses indirect access kv cache which has a different shape with other transformers models
    if len(past_key_values[0]) == 4 and past_key_values[0][0].shape[-1] == 1:
        for layer_past in past_key_values:
            layer_past[3][layer_past[0].size(-2) - 1] = beam_idx
        return past_key_values
    elif len(past_key_values[0]) == 8:
        for layer_past in past_key_values:
            layer_past[3][layer_past[0].size(-2) - 1] = beam_idx
            layer_past[7][layer_past[0].size(-2) - 1] = beam_idx
        return past_key_values
    else:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
