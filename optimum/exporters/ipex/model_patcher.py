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

from transformers.models.bert.modeling_bert import BertIntermediate
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaRMSNorm,
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralModel, MistralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2RMSNorm,
)
from transformers.models.vit.modeling_vit import ViTIntermediate

from optimum.intel.utils.import_utils import is_ipex_version, is_transformers_version
from optimum.intel.utils.modeling_utils import replace_customized_linear_with_linear

from .modeling_utils import (
    _IPEX_MINIMUM_VERSION_FOR_PATCHING,
    _falcon_for_causal_lm_forward,
    _falcon_model_forward,
    _gpt2_lm_head_model_forward,
    _gpt2_model_forward,
    _ipex_rms_layer_norm_forward,
    _IPEXFalconDecoderLayer,
    _IPEXGPT2Block,
    _IPEXIntermediate,
    _IPEXLlamaDecoderLayer,
    _IPEXMistralDecoderLayer,
    _IPEXQwen2DecoderLayer,
    _llama_model_forward,
    _mistral_model_forward,
    _qwen2_model_forward,
)


# Please also update in the setup.py and .github/workflows/test_ipex.yml if you change the transformers version
_TRANSFORMERS_MIN_VERSION = "4.50.0"
_TRANSFORMERS_MAX_VERSION = "4.51.3"

_IPEX_EXPORTED_GENERATION_TASKS = ("text-generation",)


def convert_func(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)


def convert_functions(m, target_m, new_function_name, new_function):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            convert_func(sub_m, new_function_name, new_function)
        convert_functions(sub_m, target_m, new_function_name, new_function)


def convert_class(m, target_m, new_class, device, config):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, device, config)
            setattr(m, name, new_m)
        convert_class(sub_m, target_m, new_class, device, config)


def patch_op(m, target_m, new_op_name, new_op):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            setattr(sub_m, new_op_name, new_op)
        patch_op(sub_m, target_m, new_op_name, new_op)


def _patch_llama_model(model):
    """
    Patch llama model:
        1. Use IPEX rope and paged cache
        2. Linear fusion with (2 Linears + Silu + Mul) and (Linear + Add)
    """
    convert_functions(model, LlamaModel, "forward", _llama_model_forward)
    convert_functions(model, LlamaRMSNorm, "forward", _ipex_rms_layer_norm_forward)
    convert_class(model, LlamaDecoderLayer, _IPEXLlamaDecoderLayer, model.device, model.config)
    return model


def _patch_falcon_model(model):
    """
    Patch falcon model:
        1. Use IPEX rope and paged cache
        2. Linear fusion with (Linear + Gelu) and (Linear + Add + Add)
    """
    num_key_value_heads = (
        model.config.num_kv_heads if (model.config.new_decoder_architecture or not model.config.multi_query) else 1
    )
    setattr(model.config, "num_key_value_heads", num_key_value_heads)
    convert_func(model, "forward", _falcon_for_causal_lm_forward)
    convert_functions(model, FalconModel, "forward", _falcon_model_forward)
    replace_customized_linear_with_linear(model)
    convert_class(model, FalconDecoderLayer, _IPEXFalconDecoderLayer, model.device, model.config)
    return model


def _patch_gpt2_model(model):
    """
    Patch gpt2 model:
        1. Use IPEX paged attention
        2. Linear fusion with (Linear + Add)
    """
    num_key_value_heads = model.config.num_attention_heads
    setattr(model.config, "num_key_value_heads", num_key_value_heads)
    convert_func(model, "forward", _gpt2_lm_head_model_forward)
    convert_functions(model, GPT2Model, "forward", _gpt2_model_forward)
    convert_class(model, GPT2Block, _IPEXGPT2Block, model.device, model.config)
    return model


def _patch_qwen2_model(model):
    """
    Patch qwen2 model:
        1. Use IPEX rope and paged cache
        2. Linear fusion with (2 Linears + Silu + Mul) and (Linear + Add)
    """
    # To avoid call _ignore_causal_mask_sdpa which will cause recompile
    model.config._attn_implementation = "ipex_paged"
    convert_functions(model, Qwen2Model, "forward", _qwen2_model_forward)
    convert_functions(model, Qwen2RMSNorm, "forward", _ipex_rms_layer_norm_forward)
    convert_class(model, Qwen2DecoderLayer, _IPEXQwen2DecoderLayer, model.device, model.config)
    return model


def _patch_mistral_model(model):
    """
    Patch mistral model:
        1. Use IPEX rope and paged cache
        2. Linear fusion with (Linear + Add)
    """
    convert_functions(model, MistralModel, "forward", _mistral_model_forward)
    convert_functions(model, MistralRMSNorm, "forward", _ipex_rms_layer_norm_forward)
    convert_class(model, MistralDecoderLayer, _IPEXMistralDecoderLayer, model.device, model.config)
    return model


def _patch_bert_model(model):
    """
    Patch bert model:
        1. Linear fusion with Linear + Gelu
    """
    convert_class(model, BertIntermediate, _IPEXIntermediate, model.device, model.config)
    return model


def _patch_vit_model(model):
    """
    Patch vit model:
        1. Linear fusion with Linear + Gelu
    """
    convert_class(model, ViTIntermediate, _IPEXIntermediate, model.device, model.config)
    return model


def _patch_model(model):
    if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
        raise ImportError(f"Only ipex version >= {_IPEX_MINIMUM_VERSION_FOR_PATCHING} supports llama model patching")
    if is_transformers_version("<", _TRANSFORMERS_MIN_VERSION) or is_transformers_version(
        ">", _TRANSFORMERS_MAX_VERSION
    ):
        raise ImportError(
            f"Only transformers versions {_TRANSFORMERS_MIN_VERSION} ~ {_TRANSFORMERS_MAX_VERSION} are verified."
        )
    if model.config.model_type == "llama":
        model = _patch_llama_model(model)
    elif model.config.model_type == "falcon":
        model = _patch_falcon_model(model)
    elif model.config.model_type == "gpt2":
        model = _patch_gpt2_model(model)
    elif model.config.model_type == "qwen2":
        model = _patch_qwen2_model(model)
    elif model.config.model_type == "mistral":
        model = _patch_mistral_model(model)
    elif model.config.model_type == "bert":
        model = _patch_bert_model(model)
    elif model.config.model_type == "vit":
        model = _patch_vit_model(model)
    return model
