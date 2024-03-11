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

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)

from optimum.intel.utils.import_utils import is_ipex_version

from .modeling_utils import (
    _IPEXLlamaDecoderLayerRef,
    _llama_attn_forward,
    _llama_layer_norm_forward,
    _llama_model_forward,
)


_IPEX_EXPORTED_ARCH = ("LlamaForCausalLM",)
_IPEX_EXPORTED_TASK = ("text-generation",)


def convert_func(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)


def convert_functions(m, target_m, new_function_name, new_function):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            convert_func(sub_m, new_function_name, new_function)
        convert_functions(sub_m, target_m, new_function_name, new_function)


def convert_class(m, target_m, new_class, config, distributed=False):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, config, distributed)
            setattr(m, name, new_m)
        convert_class(sub_m, target_m, new_class, config, distributed)


def patch_op(m, target_m, new_op_name, new_op):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            setattr(sub_m, new_op_name, new_op)
        patch_op(sub_m, target_m, new_op_name, new_op)


def _patch_llama_model(model):
    if is_ipex_version("<", "2.5.0"):
        raise ImportError("Only ipex version > 2.3.0 supports RotaryEmbedding and IndirectAccessKVCache")

    from intel_extension_for_pytorch.llm.modules import IndirectAccessKVCache, RotaryEmbedding

    ipex_rope = RotaryEmbedding(
        model.config.max_position_embeddings,
        model.config.hidden_size // model.config.num_attention_heads,
        model.config.rope_theta,
        model.config.architectures[0],
    )
    ipex_scale_dot_product = IndirectAccessKVCache(text_max_length=model.config.max_position_embeddings)
    patch_op(model, LlamaAttention, "ipex_rope", ipex_rope)
    patch_op(model, LlamaAttention, "ipex_scale_dot_product", ipex_scale_dot_product)

    convert_functions(model, LlamaModel, "forward", _llama_model_forward)
    convert_functions(model, LlamaAttention, "forward", _llama_attn_forward)
    convert_functions(model, LlamaRMSNorm, "forward", _llama_layer_norm_forward)

    convert_class(model, LlamaDecoderLayer, _IPEXLlamaDecoderLayerRef, model.config)
    return model


def _patch_model(model):
    if isinstance(model, LlamaForCausalLM):
        model = _patch_llama_model(model)
    return model
