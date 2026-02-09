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

import functools
import inspect
import logging
import logging as log
import math
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput

from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.model_patcher import (
    UNSUPPORTED_OPS_PATCHING_SPEC,
    ModelPatcher,
    gpt_oss_forward,
    override_arguments,
    sdpa_mask_without_vmap,
)
from optimum.intel.utils.import_utils import is_diffusers_version, is_torch_version, is_transformers_version


if is_transformers_version(">=", "4.53"):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask, sdpa_mask
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
if is_transformers_version(">=", "4.54"):
    from transformers.masking_utils import create_causal_mask
if is_transformers_version(">=", "4.56"):
    import transformers.masking_utils

if TYPE_CHECKING:
    from transformers.cache_utils import Cache
    from transformers.modeling_utils import PreTrainedModel

    from optimum.exporters.onnx.config import OnnxConfig

if is_transformers_version(">=", "4.54"):
    from transformers.utils import TransformersKwargs
else:
    TransformersKwargs = object


logger = logging.getLogger(__name__)


for idx, spec in enumerate(UNSUPPORTED_OPS_PATCHING_SPEC):
    if spec.name in {
        # onnx-exporter-specific fixes
        "triu",
        "tril",
        "norm",
        "unfold",
        "movedim",
        "rms_norm",
        "repeat_interleave",
        "scaled_dot_product_attention",
    }:
        UNSUPPORTED_OPS_PATCHING_SPEC.pop(idx)


def patch_update_causal_mask(
    model, transformers_version, inner_model_name="model", patch_fn=None, patch_extrnal_model=False
):
    if is_transformers_version(">=", transformers_version):
        inner_model = getattr(model, inner_model_name, None) if not patch_extrnal_model else model
        if inner_model is not None:
            if hasattr(inner_model, "_update_causal_mask"):
                inner_model._orig_update_causal_mask = inner_model._update_causal_mask
            patch_fn = patch_fn or _update_causal_mask_patched
            inner_model._update_causal_mask = types.MethodType(patch_fn, inner_model)


# adopted from https://github.com/huggingface/transformers/blob/f4014e75db0190792b3feeccfc5dc5b5f9f0ce7b/src/transformers/models/llama/modeling_llama.py#L1036
def _update_causal_mask_patched(
    self,
    attention_mask,
    input_tensor,
    cache_position,
    past_key_values,
    output_attentions,
):
    from transformers.cache_utils import StaticCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    # difference with original modeling
    # using minimum from dtype with larger bandwidth (float32) may lead to overflow
    # during execution on platforms with default lower precision (bfloat16, float16)
    min_dtype = torch.finfo(torch.float16).min

    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        # difference with original modeling
        causal_mask = (
            torch.full((sequence_length, target_length), fill_value=1, dtype=dtype, device=device) * min_dtype
        )

        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


# TODO: this is a numerical issue in OpenVINO, let's open an issue to track it
# initialization of sin/cos cached in bf16/fp16 leads to accuracy loss
# reinitialize them to save in float32 before export
def patch_cos_sin_cached_fp32(model):
    if (
        hasattr(model, "layers")
        and hasattr(model.layers[0], "self_attn")
        and hasattr(model.layers[0].self_attn, "rotary_emb")
        and hasattr(model.layers[0].self_attn.rotary_emb, "dtype")
        and hasattr(model.layers[0].self_attn.rotary_emb, "inv_freq")
        and hasattr(model.layers[0].self_attn.rotary_emb, "max_position_embeddings")
        and hasattr(model.layers[0].self_attn.rotary_emb, "_set_cos_sin_cache")
    ):
        for layer in model.layers:
            if layer.self_attn.rotary_emb.dtype != torch.float32:
                layer.self_attn.rotary_emb._set_cos_sin_cache(
                    seq_len=layer.self_attn.rotary_emb.max_position_embeddings,
                    device=layer.self_attn.rotary_emb.inv_freq.device,
                    dtype=torch.float32,
                )


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
# Specifically for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
def eager_mask_without_vmap(*args, **kwargs) -> Optional[torch.Tensor]:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    # we use torch.finfo(torch.float16).min instead torch.finfo(dtype).min to avoid an overflow but not
    # sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
    mask = torch.where(
        mask,
        torch.tensor(0.0, device=mask.device, dtype=dtype),
        torch.tensor(torch.finfo(torch.float16).min, device=mask.device, dtype=dtype),
    )
    return mask


class OVDecoderModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()

        patch_cos_sin_cached_fp32(self._model)
        if hasattr(self._model, "model"):
            patch_cos_sin_cached_fp32(self._model.model)

        if is_transformers_version("<", "4.53.0") and hasattr(self._model, "_update_causal_mask"):
            self._model._update_causal_mask_original = self._model._update_causal_mask
            self._model._update_causal_mask = types.MethodType(_update_causal_mask_patched, self._model)

        if is_transformers_version(">=", "4.53.0"):
            # for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
            # Although I'm not sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

            # for decoder models, we use eager mask without vmap for sdpa as well
            # to avoid a nan output issue in OpenVINO that only happens in case of:
            # non-stateful models on cpu and stateful models on npu
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", eager_mask_without_vmap)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version("<", "4.53") and hasattr(self._model, "_update_causal_mask_original"):
            self._model._update_causal_mask = self._model._update_causal_mask_original
            del self._model._update_causal_mask_original

        if is_transformers_version(">=", "4.53"):
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask)


def _mixtral_sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class MixtralModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        for layer in self._model.model.layers:
            layer.block_sparse_moe._unpatched_forward = layer.block_sparse_moe.forward
            layer.block_sparse_moe.forward = types.MethodType(
                _mixtral_sparse_moe_block_forward, layer.block_sparse_moe
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        for layer in self._model.model.layers:
            layer.block_sparse_moe.forward = layer.block_sparse_moe._unpatched_forward


class ArcticModelPatcher(MixtralModelPatcher):
    def __enter__(self):
        # model initialize some weights for matrix multiplication in bfloat16, that lead to inconsistency of dtype
        try:
            self._model.to(torch.float32)
        except Exception:
            pass

        super().__enter__()


def _chatglm_transformer_forward(
    self,
    input_ids,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.BoolTensor] = None,
    full_attention_mask: Optional[torch.BoolTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size, seq_length = input_ids.shape

    if inputs_embeds is None:
        inputs_embeds = self.embedding(input_ids)

    if getattr(self, "pre_seq_len", None) is not None:
        if past_key_values is None:
            past_key_values = self.get_prompt(
                batch_size=batch_size,
                device=input_ids.device,
                dtype=inputs_embeds.dtype,
            )
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask.new_ones((batch_size, self.pre_seq_len)),
                    attention_mask,
                ],
                dim=-1,
            )

    if full_attention_mask is None:
        if past_key_values is not None:
            full_attention_mask = torch.ones(
                batch_size,
                seq_length,
                seq_length,
                device=input_ids.device,
                dtype=torch.float,
            ) * float("-inf")
            full_attention_mask.triu_(diagonal=1)
            past_length = 0
            if past_key_values:
                past_length = past_key_values[0][0].shape[0]
            if past_length:
                full_attention_mask = torch.cat(
                    (
                        torch.zeros(batch_size, seq_length, past_length, device=input_ids.device),
                        full_attention_mask,
                    ),
                    dim=-1,
                )
            full_attention_mask.unsqueeze_(1)

    # Rotary positional embeddings
    rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
    if position_ids is not None:
        rotary_pos_emb = rotary_pos_emb[position_ids]
    else:
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]
    rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

    # Run encoder.
    hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
        inputs_embeds,
        full_attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        kv_caches=past_key_values,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
    )

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def _chatglm2_get_context_layer(query_layer: torch.Tensor, key_layer: torch.Tensor, value_layer: torch.Tensor):
    mask = torch.zeros((query_layer.shape[-2], key_layer.shape[-2]), dtype=query_layer.dtype)
    if query_layer.shape[2] == key_layer.shape[2]:
        tmp_mask = torch.ones((query_layer.shape[-2], key_layer.shape[-2]), dtype=torch.bool).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))

    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer, key_layer, value_layer, attn_mask=mask
    )
    return context_layer


def _chatglm2_core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
    if attention_mask is None:
        context_layer = _chatglm2_get_context_layer(query_layer, key_layer, value_layer)
    else:
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


def _glm4_core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    causal_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    causal_mask.masked_fill_(attention_mask, float("-inf"))
    context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, causal_mask)
    context_layer = context_layer.transpose(1, 2).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)
    return context_layer


class ChatGLMModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)
        self.is_v4 = hasattr(self._model.config, "rope_ratio")

    def __enter__(self):
        super().__enter__()

        if not self.is_v4:
            self._model.transformer._orig_forward = self._model.transformer.forward
            self._model.transformer.forward = types.MethodType(_chatglm_transformer_forward, self._model.transformer)
        for block in self._model.transformer.encoder.layers:
            block.self_attention.core_attention._orig_forward = block.self_attention.core_attention.forward
            block.self_attention.core_attention.forward = types.MethodType(
                _chatglm2_core_attention_forward if not self.is_v4 else _glm4_core_attention_forward,
                block.self_attention.core_attention,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model.transformer, "_orig_forward"):
            self._model.transformer.forward = self._model.transformer._orig_forward
        for block in self._model.transformer.encoder.layers:
            block.self_attention.core_attention.forward = block.self_attention.core_attention._orig_forward


# what does this patch exactly ?
def llama_gemma_rotary_emb_forward(self, x, position_ids, seq_len=None):
    # adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py#L104
    _seq_len = torch.max(position_ids) + 1 if seq_len is None else seq_len
    if _seq_len > self.embed_positions.shape[0]:
        if seq_len is None:
            return self._orig_forward(x, position_ids)
        else:
            return self._orig_forward(x, position_ids, seq_len)
    sincos = self.embed_positions[position_ids]
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    return cos, sin


def create_sinusoidal_positions(num_pos: int, dim: int, base: int = 10000, inv_freq=None) -> torch.Tensor:
    # adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L101
    if inv_freq is None:
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim))

    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.int64).float(), inv_freq).float()
    emb = torch.cat((sinusoid_inp, sinusoid_inp), dim=-1)
    return torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)


# cos/sin for rotary position embeddings also having issues with bf16 and efficiency due to calculation on each step, use precomputed
def create_embed_positions_buffer(rotary_emb, max_position_embeddings: int = None):
    inv_freq = getattr(rotary_emb, "inv_freq", None)

    dim, base = None, None
    if inv_freq is None:
        base = rotary_emb.base
        dim = rotary_emb.dim

    return create_sinusoidal_positions(max_position_embeddings, dim, base, inv_freq)


# copied from https://github.com/huggingface/transformers/commit/57d7594a79a9f5d835abf2d4d384db0e4818e548 to unblock export with transformers 4.42
def _mistral_update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: "Cache",
    use_cache: bool,
    output_attentions: bool,
):
    from transformers.cache_utils import SlidingWindowCache, StaticCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if self._attn_implementation == "flash_attention_2":
        if attention_mask is not None and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.

    # cache_position must be valid here no matter which cache we use
    past_seen_tokens = cache_position[0] if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)
    using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

    if (
        self.config._attn_implementation == "sdpa"
        and not (using_static_cache or using_sliding_window_cache)
        and not output_attentions
    ):
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            sliding_window=self.config.sliding_window,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(torch.float16).min
    sequence_length = input_tensor.shape[1]
    # SlidingWindowCache
    if using_sliding_window_cache:
        target_length = max(sequence_length, self.config.sliding_window)
    # StaticCache
    elif using_static_cache:
        target_length = past_key_values.get_max_length()
    # DynamicCache or no cache
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        exclude_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        if self.config.sliding_window is not None:
            if not using_sliding_window_cache or sequence_length > self.config.sliding_window:
                exclude_mask = exclude_mask.bitwise_or(
                    torch.arange(target_length, device=device)
                    <= (cache_position.reshape(-1, 1) - self.config.sliding_window)
                )
        causal_mask *= exclude_mask
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


class MistralModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.48.0"):
            # apply fix https://github.com/huggingface/transformers/commit/57d7594a79a9f5d835abf2d4d384db0e4818e548
            self._model.model._orig_update_causal_mask = self._model.model._update_causal_mask
            self._model.model._update_causal_mask = types.MethodType(_mistral_update_causal_mask, self._model.model)

        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            for layer in self._model.model.layers:
                if hasattr(layer.self_attn, "rotary_emb"):
                    embed_positions = create_embed_positions_buffer(
                        rotary_emb=layer.self_attn.rotary_emb,
                        max_position_embeddings=self._model.config.max_position_embeddings,
                    )
                    layer.self_attn.rotary_emb.register_buffer("embed_positions", embed_positions)
                    layer.self_attn.rotary_emb._orig_forward = layer.self_attn.rotary_emb.forward
                    layer.self_attn.rotary_emb.forward = types.MethodType(
                        llama_gemma_rotary_emb_forward, layer.self_attn.rotary_emb
                    )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version("<", "4.48.0"):
            self._model.model._update_causal_mask = self._model.model._orig_update_causal_mask
            del self._model.model._orig_update_causal_mask

        if hasattr(self._model.model, "model") and hasattr(self._model.model.model, "layers"):
            for layer in self._model.model.layers:
                if hasattr(layer.self_attn, "rotary_emb"):
                    layer.self_attn.rotary_emb.forward = layer.self_attn.rotary_emb._orig_forward
                    del layer.self_attn.rotary_emb._orig_forward


SUPPORT_SDPA = is_torch_version(">", "2.1.0")


# TODO: why
def _qwen_rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


# TODO: why
def _qwen_apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = (t_ * cos) + (_qwen_rotate_half(t_) * sin)
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


# TODO: why
def _qwen_quantize_cache_v(fdata, bits, qmax, qmin):
    # b, s, head, h-dim->b, head, s, h-dim
    qtype = torch.uint8
    device = fdata.device
    shape = fdata.shape

    fdata_cal = torch.flatten(fdata, 2)
    fmax = torch.amax(fdata_cal, dim=-1, keepdim=True)
    fmin = torch.amin(fdata_cal, dim=-1, keepdim=True)
    # Compute params
    if qmax.device != fmax.device:
        qmax = qmax.to(device)
        qmin = qmin.to(device)
    scale = (fmax - fmin) / (qmax - qmin)
    zero = qmin - fmin / scale
    scale = scale.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    zero = zero.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    # Quantize
    res_data = fdata / scale + zero
    qdata = torch.clamp(res_data, qmin, qmax).to(qtype)
    return qdata.contiguous(), scale, zero


# TODO: why
def _qwen_attention_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
):
    mixed_x_layer = self.c_attn(hidden_states)

    query, key, value = mixed_x_layer.split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    if rotary_pos_emb_list is not None:
        cur_len = query.shape[1]
        if len(rotary_pos_emb_list) == 1:
            rotary_pos_emb = rotary_pos_emb_list[0]
            rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
            rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # Slice the pos emb for current inference
            query = _qwen_apply_rotary_pos_emb(query, q_pos_emb)
            key = _qwen_apply_rotary_pos_emb(key, k_pos_emb)
        else:
            query_list = []
            key_list = []
            for i, rotary_pos_emb in enumerate(rotary_pos_emb_list):
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # Slice the pos emb for current inference
                query_list += [_qwen_apply_rotary_pos_emb(query[i : i + 1, :, :], q_pos_emb)]
                key_list += [_qwen_apply_rotary_pos_emb(key[i : i + 1, :, :], k_pos_emb)]
            query = torch.cat(query_list, dim=0)
            key = torch.cat(key_list, dim=0)

    if self.use_cache_quantization:
        key = _qwen_quantize_cache_v(key.permute(0, 2, 1, 3), bits=8, qmin=self.cache_qmin, qmax=self.cache_qmax)
        value = _qwen_quantize_cache_v(value.permute(0, 2, 1, 3), bits=8, qmin=self.cache_qmin, qmax=self.cache_qmax)

    if layer_past is not None:
        past_key, past_value = layer_past[0], layer_past[1]
        if self.use_cache_quantization:
            # use_cache_quantization:
            # present=((q_key,key_scale,key_zero_point),
            #          (q_value,value_scale,value_zero_point))
            key = (
                torch.cat((past_key[0], key[0]), dim=2),
                torch.cat((past_key[1], key[1]), dim=2),
                torch.cat((past_key[2], key[2]), dim=2),
            )
            value = (
                torch.cat((past_value[0], value[0]), dim=2),
                torch.cat((past_value[1], value[1]), dim=2),
                torch.cat((past_value[2], value[2]), dim=2),
            )
        else:
            # not use_cache_quantization:
            # present=(key,value)
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

    if use_cache:
        present = (key, value)
    else:
        present = None

    if self.use_logn_attn and not self.training:
        if self.use_cache_quantization:
            seq_start = key[0].size(2) - query.size(1)
            seq_end = key[0].size(2)
        else:
            seq_start = key.size(1) - query.size(1)
            seq_end = key.size(1)
        logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
        query = query * logn_tensor.expand_as(query)

    if self.use_flash_attn and not self.is_fp32 and query.is_cuda:
        q, k, v = query, key, value
        attn_output = self.core_attention_flash(q, k, v, attention_mask=attention_mask)
    else:
        registered_causal_mask = torch.tril(
            torch.ones((key.size(1), key.size(1)), dtype=torch.bool, device=key.device)
        ).view(1, 1, key.size(1), key.size(1))
        query = query.permute(0, 2, 1, 3)
        if not self.use_cache_quantization:
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

        if not self.use_cache_quantization and SUPPORT_SDPA:
            # For performance, using constant tril to generate causal_mask
            causal_mask = self.bias[:, :, key.size(-2) - query.size(-2) : key.size(-2), : key.size(-2)]
            if attention_mask is not None:
                attention_mask = attention_mask.expand(-1, -1, query.size(2), -1).masked_fill(
                    ~causal_mask, torch.finfo(query.dtype).min
                )
            else:
                attention_mask = causal_mask
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask).transpose(1, 2)
            attn_weight = None
        else:
            attn_output, attn_weight = self._attn(query, key, value, registered_causal_mask, attention_mask, head_mask)
    context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)

    attn_output = self.c_proj(context_layer)

    outputs = (attn_output, present)
    if output_attentions:
        if self.use_flash_attn and not self.is_fp32:
            raise ValueError("Cannot output attentions while using flash-attn")
        else:
            outputs += (attn_weight,)
    return outputs


class QwenModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        self.original_fp16 = model.config.fp16
        self.original_bf16 = model.config.bf16
        model.config.bf16 = False
        model.config.fp16 = False
        if self.original_fp16 or self.original_bf16:
            # GPTQ models does to support casting to dtype
            try:
                model.to(torch.float32)
            except Exception:
                pass
        model.transformer.rotary_emb(2048)

    def __enter__(self):
        super().__enter__()
        max_positions = self._model.config.seq_length
        for block in self._model.transformer.h:
            block.attn._orig_forward = block.attn.forward
            # For performance, using constant tril to generate causal_mask
            block.attn.register_buffer(
                "bias",
                torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                    1, 1, max_positions, max_positions
                ),
                persistent=False,
            )
            block.attn.forward = types.MethodType(_qwen_attention_forward, block.attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for block in self._model.transformer.h:
            block.attn.forward = block.attn._orig_forward
        self._model.config.bf16 = self.original_bf16
        self._model.config.fp16 = self.original_fp16


def _baichuan13b_atten_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if past_key_value is not None:
        # reuse k, v, self_attention
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, -key_states.shape[-2] :, :]
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if not output_attentions:
        past_key_value = (key_states, value_states) if use_cache else None
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask)
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1:  # inference with cache
                if len(attention_mask.size()) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


# Adapted from https://huggingface.co/baichuan-inc/Baichuan-7B/blob/262c8cb58b6d3615c208d9230baa869fddee2adb/modeling_baichuan.py#L181
def _baichuan7b_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    bsz, q_len, _ = hidden_states.size()

    proj = self.W_pack(hidden_states)
    proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
    query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    if not output_attentions:
        attn_weights = None
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, scale=1 / math.sqrt(self.head_dim)
        )
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


class BaichuanModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)
        # model has first inference buffers initialization
        if hasattr(self._model.lm_head, "first_flag"):
            self._model(torch.ones((1, 10), dtype=torch.int64), torch.ones((1, 10), dtype=torch.int64))

    def __enter__(self):
        super().__enter__()
        # override signature to have position_ids
        if "position_ids" not in inspect.signature(self._model.forward).parameters:
            self._model._orig_forward = self._model.forward

            def forward(
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
                position_ids: Optional[torch.LongTensor] = None,
            ):
                return self._orig_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=past_key_values is not None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=self.config.return_dict,
                )

            self._model.forward = types.MethodType(forward, self._model)
            for layer in self._model.model.layers:
                layer.self_attn._orig_forward = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_baichuan13b_atten_forward, layer.self_attn)
        else:
            for layer in self._model.model.layers:
                layer.self_attn._orig_forward = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_baichuan7b_attn_forward, layer.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model, "_orig_forward"):
            self._model.forward = self._model._orig_forward

        for layer in self._model.model.layers:
            if hasattr(layer.self_attn, "_orig_forward"):
                layer.self_attn.forward = layer.self_attn._orig_forward


# Modified from https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/mpt/modeling_mpt.py#L90
def _mpt_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
):
    batch_size, seq_length = hidden_states.shape[:2]

    mixed_qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
    query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

    if past_key_value is not None:
        # starting from v4.54 https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/mpt/modeling_mpt.py#L362
        if is_transformers_version(">=", "4.54"):
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            pkv_seq_length = past_key_value.get_seq_length()

        else:
            if len(past_key_value) != 0:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states)
            pkv_seq_length = past_key_value[0].shape[2]

    key_length = key_states.shape[-2]
    query_length = seq_length if past_key_value is None else seq_length + pkv_seq_length
    attention_mask_sdpa = torch.ones(
        (query_states.shape[0], query_states.shape[1], query_states.shape[2], key_states.shape[2]),
        dtype=query_states.dtype,
    )
    if position_bias is not None:
        position_bias_query_index = max(0, position_bias.size(1) - query_length)
        position_bias_key_index = max(0, position_bias.size(2) - key_length)

        position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]
        attention_mask_sdpa += position_bias
    attention_mask_sdpa.masked_fill_(attention_mask, torch.finfo(query_states.dtype).min)
    context_states = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask_sdpa,
        dropout_p=self.attn_dropout_p,
        scale=self.softmax_scale,
    )

    context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
    attn_output = self.out_proj(context_states)

    outputs = (attn_output, None)

    if is_transformers_version("<", "4.54"):
        outputs += (past_key_value,)

    return outputs


# Mofied from https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/mpt/modeling_mpt.py#L188
def _mpt_block_forward(
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    cache_position: Optional[torch.Tensor] = None,
):
    # hidden_states: [batch_size, seq_length, hidden_size]
    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.norm_1(hidden_states)

    residual = hidden_states

    if not output_attentions:
        # Self attention.
        attn_out = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
            cache_position=cache_position,
        )
    else:
        attn_out = self.attn._orig_forward(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
            cache_position=cache_position,
        )

    hidden_states = self.resid_attn_dropout(attn_out[0]) + residual

    layernorm_output = self.norm_2(hidden_states)

    # Get residual
    residual = hidden_states

    # MLP.
    output = self.ffn(layernorm_output, residual)
    outputs = (output,)

    if use_cache and is_transformers_version("<", "4.54"):
        outputs += (attn_out[2],)

    if output_attentions:
        outputs += (attn_out[1],)

    return outputs


class MPTModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_torch_version(">=", "2.1.0"):
            for block in self._model.transformer.blocks:
                block._orig_forward = block.forward
                block.forward = types.MethodType(_mpt_block_forward, block)
                block.attn._orig_forward = block.attn.forward
                block.attn.forward = types.MethodType(_mpt_sdpa_attention_forward, block.attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for block in self._model.transformer.blocks:
            if hasattr(block, "_orig_forward"):
                block.forward = block._orig_forward
            if hasattr(block.attn, "_orig_forward"):
                block.attn.forward = block.attn._orig_forward


def _internlm2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    from einops import rearrange

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors."""
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    is_legacy = not hasattr(self, "layer_idx")

    if is_legacy:
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if not output_attentions:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask, scale=(1 / math.sqrt(self.head_dim))
        )
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.wo(attn_output)

    return attn_output, attn_weights, past_key_value


class InternLM2Patcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_torch_version(">=", "2.1.0"):
            for block in self._model.model.layers:
                block.attention._orig_forward = block.attention.forward
                block.attention.forward = types.MethodType(_internlm2_attention_forward, block.attention)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for block in self._model.model.layers:
            if hasattr(block.attention, "_orig_forward"):
                block.attention.forward = block.attention._orig_forward


def phi3_442_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    from transformers.cache_utils import Cache
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# Adapted from https://github.com/huggingface/transformers/blob/ccdabc5642bf84849af93f591e207dc625c8e1e1/src/transformers/models/phi3/modeling_phi3.py#L729
def _phi3_self_attn_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        return self._orig_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


class Phi3ModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # currently, long RoPE can not be traced for long context support, disable it for avoid potential accuracy issues
        if self._model.config.max_position_embeddings != getattr(
            self._model.config, "original_max_position_embeddings", self._model.config.max_position_embeddings
        ):
            self._model.config.max_position_embeddings = self._model.config.original_max_position_embeddings

        if is_transformers_version("<", "4.48.0"):
            self._model.model._orig_forward = self._model.model.forward
            self._model.model.forward = types.MethodType(phi3_442_forward, self._model.model)

        # https://github.com/huggingface/transformers/blob/30ee508c6c92a1c0aa0281d193c7c0fb815b8d2f/src/transformers/models/phi3/modeling_phi3.py#L113
        # init inv_freq for torchscript tracing
        # 4.48 transformers version phi3 fixed, but issue still visible with trust_remote_true=True (trust_remote_code has _support_sdpa = False)
        for layer in self._model.model.layers:
            if (
                is_torch_version(">=", "2.1.0")
                and is_transformers_version("<", "4.48.0")
                or not getattr(self._model, "_supports_sdpa", False)
            ):
                orig_self_attn_fwd = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_phi3_self_attn_sdpa_forward, layer.self_attn)
                layer.self_attn._orig_forward = orig_self_attn_fwd

            if (
                hasattr(layer.self_attn, "rotary_emb")
                and getattr(layer.self_attn.rotary_emb, "inv_freq", None) is None
            ):
                rotary_emb = layer.self_attn.rotary_emb
                layer.self_attn.rotary_emb.inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim)
                )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model.model, "_orig_forward"):
            self._model.model.forward = self._model.model._orig_forward
        if hasattr(self._model.model, "_orig_update_causal_mask"):
            self._model.model._update_causal_mask = self._model.model._orig_update_causal_mask
        for layer in self._model.model.layers:
            if hasattr(layer.self_attn, "_orig_forward"):
                layer.self_attn.forward = layer.self_attn._orig_forward


# Modified from https://github.com/huggingface/transformers/blob/v4.50.2/src/transformers/models/phimoe/modeling_phimoe.py#L756
# removed usage nonfriendly for tracing operation continue
def _phi_moe_sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    from transformers.models.phimoe.modeling_phimoe import sparsemixer

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.input_jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(
            1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
        )
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    routing_weights, selected_experts = sparsemixer(
        router_logits,
        jitter_eps=self.router_jitter_noise,
        training=self.training,
    )

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # if top_x.shape[0] == 0:
        #     continue

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class PhiMoEModelPatcher(Phi3ModelPatcher):
    def __enter__(self):
        super().__enter__()
        for layer in self._model.model.layers:
            layer.block_sparse_moe._orig_forward = layer.block_sparse_moe.forward
            layer.block_sparse_moe.forward = types.MethodType(
                _phi_moe_sparse_moe_block_forward, layer.block_sparse_moe
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for layer in self._model.model.layers:
            layer.block_sparse_moe.forward = layer.block_sparse_moe._orig_forward


def _aquila_self_attn_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    if output_attentions:
        return self._orig_forward(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache
        )
    bsz, q_len, _ = hidden_states.size()

    if hasattr(self.config, "pretraining_tp") and self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, getattr(self, "num_key_value_heads", self.num_heads), self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, getattr(self, "num_key_value_heads", self.num_heads), self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if hasattr(self, "num_key_value_groups"):
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, attention_mask, scale=(1 / math.sqrt(self.head_dim))
    )
    attn_weights = None

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if hasattr(self.config, "pretraining_tp") and self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


class AquilaModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for layer in self._model.model.layers:
            if is_torch_version(">=", "2.1.0"):
                orig_self_attn_fwd = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_aquila_self_attn_sdpa_forward, layer.self_attn)
                layer.self_attn._orig_forward = orig_self_attn_fwd

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for layer in self._model.model.layers:
            if hasattr(layer.self_attn, "_orig_forward"):
                layer.self_attn.forward = layer.self_attn._orig_forward


def _xverse_self_attn_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    if output_attentions:
        return self._orig_forward(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, attention_mask, scale=(1 / math.sqrt(self.head_dim))
    )
    attn_weights = None

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def _internlm_self_attn_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    if output_attentions:
        return self._orig_forward(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache
        )

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, attention_mask, scale=(1 / math.sqrt(self.head_dim))
    )
    attn_weights = None

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


class XverseModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for layer in self._model.model.layers:
            if is_torch_version(">=", "2.1.0"):
                orig_self_attn_fwd = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_xverse_self_attn_sdpa_forward, layer.self_attn)
                layer.self_attn._orig_forward = orig_self_attn_fwd

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for layer in self._model.model.layers:
            if hasattr(layer.self_attn, "_orig_forward"):
                layer.self_attn.forward = layer.self_attn._orig_forward


class InternLMModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for layer in self._model.model.layers:
            if is_torch_version(">=", "2.1.0"):
                orig_self_attn_fwd = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_internlm_self_attn_sdpa_forward, layer.self_attn)
                layer.self_attn._orig_forward = orig_self_attn_fwd

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for layer in self._model.model.layers:
            if hasattr(layer.self_attn, "_orig_forward"):
                layer.self_attn.forward = layer.self_attn._orig_forward


# Adapted from https://github.com/huggingface/optimum/blob/3adbe7c75e3c41c1a3b945cf085e74ece7f8e192/optimum/bettertransformer/models/attention.py#L234
def codegen_wrapped_scaled_dot_product(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    # in codegen the query and key are always in fp32 regardless of the dtype of the model
    # https://github.com/huggingface/transformers/blob/5b28b7833297adf65c5160a685425ddb1eee5ce2/src/transformers/models/codegen/modeling_codegen.py#L226
    query = query.to(value.dtype)
    key = key.to(value.dtype)

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            # first step of the decoding
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # in this case, which is the later decoding steps, the `causal_mask` in
            # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/gpt2/modeling_gpt2.py#L195
            # is [True, ..., True] so actually not causal
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length = query.size(-2)
        # causal_mask is always [True, ..., True] otherwise, so executing this
        # is unnecessary
        if query_length > 1:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    return sdpa_result, None


# copied from  https://github.com/huggingface/optimum/blob/2112e99122d7f23a1da1a9d263fef64301050ea7/optimum/bettertransformer/models/attention.py#L168
# for preserving backward compatibility between outdated codegen remote code and new transformers
def _codegen_wrapped_scaled_dot_product_legacy(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    if head_mask is not None:
        raise ValueError("`head_mask` input argument is not supported")
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, -1, -1] < -1:
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")

    # in codegen the query and key are always in fp32 regardless of the dtype of the model
    # https://github.com/huggingface/transformers/blob/5b28b7833297adf65c5160a685425ddb1eee5ce2/src/transformers/models/codegen/modeling_codegen.py#L226
    query = query.to(value.dtype)
    key = key.to(value.dtype)

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            # first step of the decoding
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            # in this case, which is the later decoding steps, the `causal_mask`` in
            # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/gpt2/modeling_gpt2.py#L195
            # is [True, ..., True] so actually not causal
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        # causal_mask is always [True, ..., True] otherwise, so executing this is unnecessary
        if query_length > 1:
            causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)

            causal_mask = torch.where(causal_mask, 0, mask_value)

            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

            # we use torch.min to avoid having tensor(-inf)
            attention_mask = torch.min(causal_mask, attention_mask)

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    return sdpa_result, None


class CodeGenModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # TODO: why is this needed again ? it's too convoluted
        attn_fn = codegen_wrapped_scaled_dot_product
        if is_torch_version(">=", "2.1.0"):
            # in transformers 4.45 causal_mask const buffer was removed from the model
            # if it still exists, it means legacy remote code was loaded
            if hasattr(self._model.transformer.h[0].attn, "causal_mask"):
                attn_fn = _codegen_wrapped_scaled_dot_product_legacy

        for layer in self._model.transformer.h:
            if is_torch_version(">=", "2.1.0") and not self._model.config.output_attentions:
                orig_self_attn_fwd = layer.attn._attn
                layer.attn._attn = types.MethodType(attn_fn, layer.attn)
                layer.attn._orig_attn = orig_self_attn_fwd

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version("<", "4.53") and hasattr(self._model.transformer, "_update_causal_mask_original"):
            self._model.transformer._update_causal_mask = self._model.transformer._update_causal_mask_original
            del self._model.transformer._update_causal_mask_original

        for layer in self._model.transformer.h:
            if hasattr(layer.attn, "_orig_attn"):
                layer.attn._attn = layer.attn._orig_attn


# Adapted from https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/dbrx/modeling_dbrx.py#L763
def _dbrx_experts_forward(
    self, x: torch.Tensor, weights: torch.Tensor, top_weights: torch.Tensor, top_experts: torch.LongTensor
):
    bsz, q_len, hidden_size = x.shape
    x = x.view(-1, hidden_size)
    out = torch.zeros_like(x)

    expert_mask = torch.nn.functional.one_hot(top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
    # Chunk experts at once to avoid storing full parameter multiple times in autograd
    w1_chunked = self.mlp.w1.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
        self.moe_num_experts, dim=0
    )
    v1_chunked = self.mlp.v1.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
        self.moe_num_experts, dim=0
    )
    w2_chunked = self.mlp.w2.view(self.mlp.moe_num_experts, self.mlp.ffn_hidden_size, self.mlp.hidden_size).chunk(
        self.moe_num_experts, dim=0
    )
    w1_chunked = [w1.squeeze(dim=0) for w1 in w1_chunked]
    v1_chunked = [v1.squeeze(dim=0) for v1 in v1_chunked]
    w2_chunked = [w2.squeeze(dim=0) for w2 in w2_chunked]
    for expert_idx in range(0, self.moe_num_experts):
        topk_idx, token_idx = torch.where(expert_mask[expert_idx])

        # Difference with original: removal
        # if token_idx.shape[0] == 0:
        #     continue
        # loop interruption depends on input data and may affect torchscript tracing

        token_list = token_idx
        topk_list = topk_idx

        expert_tokens = x[None, token_list].reshape(-1, hidden_size)
        expert_out = (
            self.mlp(expert_tokens, w1_chunked[expert_idx], v1_chunked[expert_idx], w2_chunked[expert_idx])
            * top_weights[token_list, topk_list, None]
        )

        out.index_add_(0, token_idx, expert_out)

    out = out.reshape(bsz, q_len, hidden_size)
    return out


# adopted from https://github.com/huggingface/transformers/blob/1b3dba9417eebe16b7c206d1dfca6a4c7f11dbec/src/transformers/models/dbrx/modeling_dbrx.py#L1204
def _dbrx_update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: "Cache",
    output_attentions: bool,
):
    from transformers.cache_utils import StaticCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    # difference with original modeling
    # using minimum from dtype with larger bandwidth (float32) may lead to overflow
    # during execution on platforms with default lower precision (bfloat16, float16)
    min_dtype = torch.finfo(torch.float16).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    if attention_mask is not None and attention_mask.dim() == 4:
        # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
        if attention_mask.max() != 0:
            raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
        causal_mask = attention_mask
    else:
        # difference with original modeling
        causal_mask = (
            torch.full((sequence_length, target_length), fill_value=1, dtype=dtype, device=device) * min_dtype
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


class DBRXModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        # dbrx has some accuracy issues with bf16 with transformers >= 4.40
        # fill causal mask in slightly different way for avoid overflow on some platforms
        self._model.transformer._orig_update_causal_mask = self._model.transformer._update_causal_mask
        self._model.transformer._update_causal_mask = types.MethodType(
            _dbrx_update_causal_mask, self._model.transformer
        )

        inv_freq = getattr(self._model.transformer.blocks[0].norm_attn_norm.attn.rotary_emb, "inv_freq")
        dim, base = None, None
        if inv_freq is None:
            dim = self._model.transformer.blocks[0].norm_attn_norm.attn.rotary_emb.dim
            base = self._model.transformer.blocks[0].norm_attn_norm.attn.rotary_emb.base
        max_positions = self._model.config.max_seq_len
        embed_positions = create_sinusoidal_positions(max_positions, dim, base, inv_freq)

        for block in self._model.transformer.blocks:
            rotary_emb = block.norm_attn_norm.attn.rotary_emb
            # initialize inv_freq for torchscript tracing
            if rotary_emb.inv_freq is None:
                inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim)
                )
                rotary_emb.inv_freq = inv_freq

            rotary_emb.register_buffer("embed_positions", embed_positions)
            rotary_emb._orig_forward = rotary_emb.forward
            rotary_emb.forward = types.MethodType(llama_gemma_rotary_emb_forward, rotary_emb)

            # remove continue-operator from iteration loop over experts
            block.ffn.experts._orig_forward = block.ffn.experts.forward
            block.ffn.experts.forward = types.MethodType(_dbrx_experts_forward, block.ffn.experts)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.transformer._update_causal_mask = self._model.transformer._orig_update_causal_mask
        for block in self._model.transformer.blocks:
            block.ffn.experts.forward = block.ffn.experts._orig_forward

            if hasattr(block.norm_attn_norm.attn.rotary_emb, "_orig_forward"):
                block.norm_attn_norm.attn.rotary_emb.forward = block.norm_attn_norm.attn.rotary_emb._orig_forward


# Adapted from https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/models/persimmon/modeling_persimmon.py#L264
def _persimmon_self_attn_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    from transformers.models.persimmon.modeling_persimmon import apply_rotary_pos_emb

    if output_attentions:
        return self._orig_forward(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache
        )

    bsz, q_len, _ = hidden_states.size()

    # [batch_size, seq_length, 3 x hidden_size]
    fused_qkv = self.query_key_value(hidden_states)

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_states, key_states, value_states) = self._split_heads(fused_qkv)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
    query_states = query_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    if position_embeddings is None:
        log.warning(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    rotary_ndims = self.rotary_ndims
    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., :rotary_ndims],
        query_states[..., rotary_ndims:],
    )
    key_rot, key_pass = (
        key_states[..., :rotary_ndims],
        key_states[..., rotary_ndims:],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    if past_key_value is not None:
        # Specific to RoPE models with partial rotation
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        causal_mask,
        scale=1 / math.sqrt(self.head_dim),
        dropout_p=self.attention_dropout.p,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.dense(attn_output)

    outputs = (attn_output, None)

    if is_transformers_version("<", "4.54"):
        outputs += (past_key_value,)

    return outputs


class PersimmonModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_transformers_version("<", "4.56"):
            for layer in self._model.model.layers:
                if is_torch_version(">=", "2.1.0"):
                    orig_self_attn_fwd = layer.self_attn.forward
                    layer.self_attn.forward = types.MethodType(_persimmon_self_attn_sdpa_forward, layer.self_attn)
                    layer.self_attn._orig_forward = orig_self_attn_fwd

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version("<", "4.56"):
            for layer in self._model.model.layers:
                if hasattr(layer.self_attn, "_orig_forward"):
                    layer.self_attn.forward = layer.self_attn._orig_forward


def _jais_attn_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    position_bias: Optional[torch.FloatTensor] = None,
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn"):
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `JAISAttention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask
    else:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    if layer_past is not None:
        past_key, past_value = layer_past
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    if self.reorder_and_upcast_attn:
        attn_output, attn_weights = self._upcast_and_reordered_attn(
            query, key, value, attention_mask, head_mask, position_bias
        )
    else:
        # Difference with original: override attn realization with sdpa if not output_attentions
        if not output_attentions:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, position_bias)
        else:
            attn_output, attn_weights = self._orig_attn(query, key, value, attention_mask, head_mask, position_bias)

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def _jais_attn(self, query, key, value, attention_mask=None, head_mask=None, position_bias=None):
    scale = 1.0
    if self.scale_attn_weights:
        scale = 1 / self.head_dim**self.attn_scale_power

    # Layer-wise attention scaling
    if self.scale_attn_by_inverse_layer_idx:
        scale = scale / float(self.layer_idx + 1)

    query_length = query.size(-2)
    attention_mask_sdpa = torch.ones(
        (query.shape[0], query.shape[1], query.shape[2], key.shape[2]),
        dtype=query.dtype,
    )

    if not self.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(torch.float16).min
        attention_mask_sdpa.masked_fill_(~causal_mask, mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attention_mask_sdpa = attention_mask_sdpa + attention_mask

    if position_bias is not None:
        attention_mask_sdpa += position_bias.type_as(attention_mask_sdpa).unsqueeze(0)

    # Mask heads if we want to
    if head_mask is not None:
        attention_mask_sdpa = attention_mask_sdpa * head_mask

    attn_output = F.scaled_dot_product_attention(
        query, key, value, attention_mask_sdpa, dropout_p=self.attn_dropout.p, scale=scale
    )

    return attn_output, None


class JaisModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        for layer in self._model.transformer.h:
            if is_torch_version(">=", "2.1.0"):
                orig_self_attn_fwd = layer.attn._attn
                layer.attn._attn = types.MethodType(_jais_attn, layer.attn)
                layer.attn._orig_attn = orig_self_attn_fwd
                layer.attn._orig_forward = layer.attn.forward
                layer.attn.forward = types.MethodType(_jais_attn_forward, layer.attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for layer in self._model.transformer.h:
            if hasattr(layer.attn, "_orig_attn"):
                layer.attn._attn = layer.attn._orig_attn
                layer.attn.forward = layer.attn._orig_forward


# Adapted from https://github.com/huggingface/transformers/blob/31f9a289a6207be6cae746e009d8e0db523be203/src/transformers/models/falcon/modeling_falcon.py#L1138
def _falcon_prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_position: torch.Tensor,
    batch_size: int,
    **kwargs,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        # different from original: allow to provide min_dtype as parameter
        min_dtype = torch.finfo(dtype).min if "min_dtype" not in kwargs else kwargs["min_dtype"]
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


def _falcon_update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: "Cache",
    output_attentions: bool,
    head_mask: torch.Tensor,
    alibi: torch.Tensor,
):
    # copied from  https://github.com/huggingface/transformers/blob/a30c865f991dfec9452cc64bd9a97bfbb96be036/src/transformers/models/falcon/modeling_falcon.py#L1130
    from transformers.cache_utils import StaticCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if (
        self.config._attn_implementation == "sdpa"
        and not using_static_cache
        and not output_attentions
        and head_mask is None
        and alibi is None
    ):
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device
    # difference from original, replace torch.finfo(dtype).min to float16 for prevent overflow for fp16/bf16 execution
    min_dtype = torch.finfo(torch.float16).min
    batch_size, sequence_length, _ = input_tensor.shape
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = _falcon_prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )

    # We take care to integrate alibi bias in the causal_mask here
    if head_mask is None and alibi is not None:
        alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
        causal_mask = torch.masked_fill(
            alibi / math.sqrt(self.config.hidden_size // self.num_heads),
            causal_mask < -1,
            min_dtype,
        )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


class FalconModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        patch_cos_sin_cached_fp32(self._model.transformer)

        if is_transformers_version("<", "4.53") and hasattr(self._model.transformer, "_update_causal_mask"):
            self._model.transformer._update_causal_mask_original = self._model.transformer._update_causal_mask
            self._model.transformer._update_causal_mask = types.MethodType(
                _falcon_update_causal_mask, self._model.transformer
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version("<", "4.53") and hasattr(self._model.transformer, "_update_causal_mask_original"):
            self._model.transformer._update_causal_mask = self._model.transformer._update_causal_mask_original
            del self._model.transformer._update_causal_mask_original


class GptNeoxModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        if (
            is_transformers_version("<", "4.53")
            and hasattr(self._model, "transformer")
            and hasattr(self._model.transformer, "_update_causal_mask")
        ):
            self._model.transformer._update_causal_mask_original = self._model.transformer._update_causal_mask
            self._model.transformer._update_causal_mask = types.MethodType(
                _falcon_update_causal_mask, self._model.transformer
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if (
            is_transformers_version("<", "4.53")
            and hasattr(self._model, "transformer")
            and hasattr(self._model.transformer, "_update_causal_mask_original")
        ):
            self._model.transformer._update_causal_mask = self._model.transformer._update_causal_mask_original
            del self._model.transformer._update_causal_mask_original


# Adopted from https://github.com/huggingface/optimum/blob/v1.24.0/optimum/bettertransformer/models/attention.py#L96
def _gptj_attn(self, query, key, value, attention_mask=None, head_mask=None):
    if head_mask is not None:
        return self._orig_attn(query, key, value, attention_mask, head_mask)

    batch_size = query.shape[0]

    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    if getattr(self, "downcast_qk", False):
        query = query.to(value.dtype)
        key = key.to(value.dtype)

    if batch_size == 1 and attention_mask is not None and attention_mask[0, 0, -1, -1] < -1:
        return self._orig_attn(query, key, value, attention_mask, head_mask)

    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
    else:
        query_length = query.size(-2)
        # causal_mask is always [True, ..., True] otherwise, so executing this
        # is unnecessary
        if query_length > 1:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False
        )

    # in gpt-neo-x and gpt-j the query and keys are always in fp32
    # thus we need to cast them to the value dtype
    if getattr(self, "downcast_qk", False):
        sdpa_result = sdpa_result.to(value.dtype)

    return sdpa_result, None


def gptj_attn_forward(
    self,
    hidden_states: torch.FloatTensor,
    layer_past: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    if output_attentions:
        self._attn = self._orig_attn

    return self._orig_forward(
        hidden_states,
        layer_past,
        attention_mask,
        position_ids,
        head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
    )


class GptJModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        self._model.config._orig_attn_implementation = self._model.config._attn_implementation
        self._model.config._attn_implementation = "sdpa"
        for block in self._model.transformer.h:
            block.attn._orig_forward = block.attn.forward
            block.attn.forward = types.MethodType(gptj_attn_forward, block.attn)
            block.attn._orig_attn = block.attn._attn
            block.attn._attn = types.MethodType(_gptj_attn, block.attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.config._attn_implementation = self._model.config._orig_attn_implementation
        for block in self._model.transformer.h:
            block.attn.forward = block.attn._orig_forward
            block.attn._attn = block.attn._orig_attn


# Adopted from https://github.com/huggingface/optimum/blob/main/optimum/bettertransformer/models/attention.py#L721
def _bloom_attn_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past=None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    from transformers.models.bloom.modeling_bloom import dropout_add

    if head_mask is not None or output_attentions:
        return self._orig_forward(
            hidden_states,
            residual,
            alibi,
            attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
    batch_size, q_length, _ = hidden_states.shape
    # [batch_size, seq_length, 3 x hidden_size]
    fused_qkv = self.query_key_value(hidden_states)
    # 3 x [batch_size, num_heads, seq_length, head_dim]
    query_layer, key_layer, value_layer = self._reshape(fused_qkv)

    if layer_past is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

    alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])

    if attention_mask is not None:  # no matter the length, we just slice it
        kv_length = cache_position[-1] + 1  # cache position is 0-indexed while length should start from 1
        causal_mask = attention_mask[:, :, :, :kv_length]
        alibi = torch.masked_fill(alibi, causal_mask.bool(), torch.finfo(alibi.dtype).min)

    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer,
        key_layer,
        value_layer,
        attn_mask=alibi,
        dropout_p=self.dropout_prob_attn if self.training else 0.0,
    )

    # Transform [batch_size, num_heads, seq_length, head_dim] to [batch_size, seq_length, num_heads * head_dim]
    context_layer = context_layer.transpose(1, 2)
    context_layer = context_layer.reshape(batch_size, q_length, self.hidden_size)

    # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + F.linear(
                context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
            )
    else:
        output_tensor = self.dense(context_layer)

    output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

    outputs = (output_tensor, layer_past)

    return outputs


class BloomModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        self._model.config._orig_attn_implementation = self._model.config._attn_implementation
        self._model.config._attn_implementation = "sdpa"
        for block in self._model.transformer.h:
            block.self_attention._orig_forward = block.self_attention.forward
            block.self_attention.forward = types.MethodType(_bloom_attn_forward, block.self_attention)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.config._attn_implementation = self._model.config._orig_attn_implementation
        for block in self._model.transformer.h:
            block.self_attention.forward = block.self_attention._orig_forward


def _gpt_neo_attn_forward(
    self,
    hidden_states,
    attention_mask=None,
    layer_past=None,
    head_mask=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    if output_attentions:
        self._attn = self._orig_attn

    return self._orig_forward(
        hidden_states,
        attention_mask=attention_mask,
        layer_past=layer_past,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
    )


# Adopted from https://github.com/huggingface/optimum/blob/main/optimum/bettertransformer/models/attention.py#L185
def _gpt_neo_attn_sdpa(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
):
    batch_size = query.shape[0]

    mask_value = torch.finfo(torch.float16).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)

    dropout_p = float(self.config.attention_dropout) if self.training else 0.0
    if (batch_size == 1 or self.training) and self.attention_type == "global":
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False, scale=1.0
            )
    else:
        query_length, key_length = query.size(-2), key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        causal_mask = torch.where(causal_mask, 0, mask_value)
        if batch_size > 1:
            # torch.Tensor.expand does no memory copy
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)

        if attention_mask is not None:
            attention_mask = causal_mask + attention_mask

        sdpa_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False, scale=1.0
        )

    return sdpa_result, None


class GptNeoModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_torch_version(">=", "2.1.0"):
            self._model.config._orig_attn_implementation = self._model.config._attn_implementation
            self._model.config._attn_implementation = "sdpa"
            for layer in self._model.transformer.h:
                self_attn = layer.attn.attention
                self_attn._orig_attn = self_attn._attn
                self_attn._attn = types.MethodType(_gpt_neo_attn_sdpa, self_attn)
                self_attn._orig_forward = types.MethodType(_gpt_neo_attn_forward, self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model.config, "_orig_attn_implementation"):
            self._model.config._attn_implementation = self._model.config._orig_attn_implementation
            for layer in self._model.transformer.h:
                for layer in self._model.transformer.h:
                    layer.attn.attention.forward = layer.attn.attention._orig_forward
                    layer.attn.attention._attn = layer.attn.attention._orig_attn


class Gemma2ModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)
            return_legacy_cache = False
            pkv_in_args = False
            legacy_pkv = None
            if "past_key_values" in kwargs:
                legacy_pkv = kwargs.pop("past_key_values", None)
            sign_names = list(signature.parameters.keys())
            pkv_argument_index = sign_names.index("past_key_values")
            cache_position_index = sign_names.index("cache_position") if "cache_position" in sign_names else -1
            input_ids_index = sign_names.index("input_ids" if "input_ids" in sign_names else "inputs_embeds")
            if legacy_pkv is None and len(args) > pkv_argument_index:
                legacy_pkv = args[pkv_argument_index]
                pkv_in_args = True
            if legacy_pkv is not None:
                pkv = DynamicCache.from_legacy_cache(legacy_pkv)
                return_legacy_cache = True
                if not pkv_in_args:
                    kwargs["past_key_values"] = pkv
                else:
                    args[pkv_argument_index] = pkv

            if (
                return_legacy_cache
                and cache_position_index != -1
                and (cache_position_index > len(args) and "cache_position" not in kwargs)
            ):
                past_seen_tokens = legacy_pkv[0][0].shape[-2]
                input_ids = args[input_ids_index] if "input_ids" not in kwargs else kwargs["input_ids"]
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
                )
                kwargs["cache_position"] = cache_position

            outputs = self.orig_forward(*args, **kwargs)
            if return_legacy_cache:
                outputs.past_key_values = outputs.past_key_values.to_legacy_cache()

            return outputs

        self.patched_forward = patched_forward


def _decilm_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # decilm contains bug in attention calculation for case if past key values is not None
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    bsz, q_len, _ = hidden_states.size()
    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=attention_mask is None, attn_mask=attention_mask
    )

    # modified, in original implementation .transpose(1, 2) missed
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    attn_weights = None

    return attn_output, attn_weights, past_key_value


class DeciLMModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        for layer in self._model.model.layers:
            layer.self_attn._orig_forward = layer.self_attn.forward
            layer.self_attn.forward = types.MethodType(_decilm_attn_forward, layer.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        for layer in self._model.model.layers:
            layer.self_attn.forward = layer.self_attn._orig_forward


class IBertModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        if getattr(self._model, "ibert"):
            embeddings = self._model.ibert.embeddings
        else:
            embeddings = self._model.embeddings
        # model has first inference buffers initialization, it may breaks tracing
        if getattr(embeddings.LayerNorm, "dim_sqrt") is None:
            self._model(torch.ones([1, 1], dtype=torch.long))


class InternVLChatImageEmbeddingModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = model.extract_feature

        if model.vision_model.encoder.layers[0].attn.use_flash_attn:
            for layer in model.vision_model.encoder.layers:
                layer.attn._orig_use_flash_attn = layer.attn.use_flash_attn
                layer.attn.use_flash_attn = False

        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward
        if hasattr(self._model.vision_model.encoder.layers[0].attn, "_orig_use_flash_attn"):
            for layer in self._model.vision_model.encoder.layers:
                layer.attn.use_flash_attn = layer.attn._orig_use_flash_attn


class InternVL2ChatLangModelPatcher(OVDecoderModelPatcher):
    def __init__(self, config: "OnnxConfig", model: "PreTrainedModel", model_kwargs: Dict[str, Any]):
        model_type = model.config.model_type
        patcher_for_model_type = {
            "llama": OVDecoderModelPatcher,
            "qwen2": OVDecoderModelPatcher,
            "internlm2": InternLM2Patcher,
            "phi3": Phi3ModelPatcher,
        }
        self._internal_patcher = None
        self._patched_forward = None
        internal_patcher_cls = patcher_for_model_type.get(model_type)
        if internal_patcher_cls is not None:
            self._internal_patcher = internal_patcher_cls(config, model, model_kwargs)
            self._patched_forward = self._internal_patcher.patched_forward
        super().__init__(config, model, model_kwargs)

    @property
    def patched_forward(self):
        if self._internal_patcher is not None:
            return self._internal_patcher.patched_forward
        return self._patched_forward

    @patched_forward.setter
    def patched_forward(self, fn):
        self._patched_forward = fn
        if self._internal_patcher is not None:
            self._internal_patcher.patched_forward = fn

    def __enter__(self):
        if is_torch_version(">=", "2.1.0"):
            if (
                self._model.config.model_type in ["qwen2", "llama"]
                and self._model.config._attn_implementation != "sdpa"
            ):
                self._model.config._orig_attn_implementation = self._model.config._attn_implementation
                self._model.config._attn_implementation = "sdpa"
                if self._model.config.model_type == "qwen2" and is_transformers_version("<", "4.48"):
                    from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES

                    sdpa_attn = QWEN2_ATTENTION_CLASSES["sdpa"]

                    for layer in self._model.model.layers:
                        layer.self_attn._orig_forward = layer.self_attn.forward
                        layer.self_attn.forward = types.MethodType(sdpa_attn.forward, layer.self_attn)

                if self._model.config.model_type == "llama" and is_transformers_version("<", "4.47"):
                    from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES

                    sdpa_attn = LLAMA_ATTENTION_CLASSES["sdpa"]
                    for layer in self._model.model.layers:
                        layer.self_attn._orig_forward = layer.self_attn.forward
                        layer.self_attn.forward = types.MethodType(sdpa_attn.forward, layer.self_attn)

        if self._internal_patcher is not None:
            return self._internal_patcher.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._internal_patcher:
            self._internal_patcher.__exit__(exc_type, exc_value, traceback)
        else:
            super().__exit__(exc_type, exc_value, traceback)

        if hasattr(self._model.config, "_orig_attn_implementation"):
            self._model.config._attn_implementation = self._model.config._orig_attn_implementation
            for layer in self._model.model.layers:
                if hasattr(layer.self_attn, "_orig_forward"):
                    layer.self_attn.forward = layer.self_attn._orig_forward


def llava_vision_embed_forward(self, pixel_values):
    # copied from https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llava/modeling_llava.py#L428-L441
    # these changes does not bring any difference from original, it only packs model subcomponent inference together
    # that allow us avoid memory overheads and their inference results handling on code-level
    image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
    # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
    selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]

    if self.config.vision_feature_select_strategy == "default":
        selected_image_feature = selected_image_feature[:, 1:]
    elif self.config.vision_feature_select_strategy == "full":
        selected_image_feature = selected_image_feature
    else:
        raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

    image_features = self.multi_modal_projector(selected_image_feature)
    return image_features


def llava_next_video_vision_embed_forward(self, pixel_values):
    # copied from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L519
    # these changes does not bring any difference from original, it only packs model subcomponent inference together
    # that allow us avoid memory overheads and their inference results handling on code-level
    image_features = self.vision_tower(pixel_values, output_hidden_states=True)
    vision_feature_layer = self.config.vision_feature_layer
    if isinstance(vision_feature_layer, int):
        selected_image_feature = image_features.hidden_states[vision_feature_layer]
    else:
        hs_pool = [image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
        selected_image_feature = torch.cat(hs_pool, dim=-1)

    if self.config.vision_feature_select_strategy == "default":
        selected_image_feature = selected_image_feature[:, 1:]
    elif self.config.vision_feature_select_strategy == "full":
        selected_image_feature = selected_image_feature
    else:
        raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
    return selected_image_feature


# Modified from https://huggingface.co/microsoft/maira-2/blob/main/modeling_maira2.py#L68
def maira_vision_embed_forward(self, pixel_values):
    vision_feature_select_strategy = self.config.vision_feature_select_strategy
    vision_feature_layer = self.config.vision_feature_layer
    return self.get_image_features(pixel_values, vision_feature_layer, vision_feature_select_strategy)


class LlavaImageEmbeddingModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = types.MethodType(llava_vision_embed_forward, model)

        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class MairaImageEmbeddingModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = types.MethodType(maira_vision_embed_forward, model)

        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class LlavaNextVideoImageEmbeddingModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = types.MethodType(llava_next_video_vision_embed_forward, model)

        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


def _embednb_forward(self, ids: torch.Tensor) -> torch.Tensor:
    def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        assert dim % 2 == 0, "The dimension must be even."

        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
        omega = 1.0 / (theta**scale)

        batch_size, seq_length = pos.shape
        out = pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
        cos_out = torch.cos(out)
        sin_out = torch.sin(out)

        stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
        out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
        return out.float()

    n_axes = ids.shape[-1]
    emb = torch.cat(
        [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
        dim=-3,
    )
    return emb.unsqueeze(1)


class FluxTransfromerModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_diffusers_version("<", "0.31.0"):
            self._model.pos_embed._orig_forward = self._model.pos_embed.forward
            self._model.pos_embed.forward = types.MethodType(_embednb_forward, self._model.pos_embed)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model.pos_embed, "_orig_forward"):
            self._model.pos_embed.forward = self._model.pos_embed._orig_forward


def _minicpmv_resampler_forward(self, image_feature, pos_embed, key_padding_mask):
    bs = image_feature.shape[0]
    image_feature = self.kv_proj(image_feature)  # B * L * D
    image_feature = self.ln_kv(image_feature).permute(1, 0, 2)  # L * B * D

    q = self.ln_q(self.query)  # Q * D

    q_bs = q.unsqueeze(1).repeat(1, bs, 1)

    out = self.attn(q_bs, image_feature + pos_embed, image_feature, key_padding_mask=key_padding_mask)[
        0
    ]  # Q * B * D  # L * B * D +  L * B * D
    #  out: Q * B * D
    x = out.permute(1, 0, 2)  # B * Q * D

    x = self.ln_post(x)
    x = x @ self.proj
    return x


def _minicpmv_siglip_vis_embed_forward(
    self,
    pixel_values: torch.FloatTensor,
    patch_attention_mask: torch.BoolTensor,
    tgt_sizes: Optional[torch.IntTensor] = None,
    position_ids: Optional[torch.FloatTensor] = None,
) -> torch.Tensor:
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)

    if position_ids is None:
        batch_size = pixel_values.size(0)
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

    position_ids = position_ids.to(self.position_embedding.weight.device)

    embeddings = embeddings + self.position_embedding(position_ids)
    return embeddings


def _minicpmv_siglip_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    batch_size, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, attention_mask, is_causal=attention_mask is None
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None


def _minicpmv_siglip_transformer_forward(
    self,
    pixel_values,
    patch_attention_mask: Optional[torch.BoolTensor] = None,
    tgt_sizes: Optional[torch.IntTensor] = None,
    position_ids: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size = pixel_values.size(0)
    if patch_attention_mask is None:
        patch_attention_mask = torch.ones(
            size=(
                batch_size,
                pixel_values.size(2) // self.config.patch_size,
                pixel_values.size(3) // self.config.patch_size,
            ),
            dtype=torch.bool,
            device=pixel_values.device,
        )

    hidden_states = self.embeddings(
        pixel_values=pixel_values,
        patch_attention_mask=patch_attention_mask,
        tgt_sizes=tgt_sizes,
        position_ids=position_ids,
    )

    patch_attention_mask = patch_attention_mask.view(batch_size, -1)
    attention_mask = (
        _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
        if not getattr(self, "_use_flash_attention_2", False)
        else patch_attention_mask
    )

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.post_layernorm(last_hidden_state)

    if not return_dict:
        return (last_hidden_state, None) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=None,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


class MiniCPMVResamplerModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = types.MethodType(_minicpmv_resampler_forward, model)

        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class MiniCPMVImageEmbeddingsModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = types.MethodType(_minicpmv_siglip_transformer_forward, model)

        super().__init__(config, model, model_kwargs)

    def __enter__(self):
        super().__enter__()
        self._model.embeddings._orig_forward = self._model.embeddings.forward
        self._model.embeddings.forward = types.MethodType(_minicpmv_siglip_vis_embed_forward, self._model.embeddings)

        if is_torch_version(">=", "2.0.0"):
            for layer in self._model.encoder.layers:
                layer.self_attn._orig_forward = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_minicpmv_siglip_attn_forward, layer.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward
        self._model.embeddings.forward = self._model.embeddings._orig_forward
        if is_torch_version(">=", "2.0.0"):
            for layer in self._model.encoder.layers:
                layer.self_attn.forward = layer.self_attn._orig_forward


class LlavaQwen2ImageEmbeddingsModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = model.encode_images
        super().__init__(config, model, model_kwargs)
        if not self._model.get_vision_tower().is_loaded:
            self._model.get_vision_tower().load_model()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class InputEmbeddingPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward

        def forward(self, input):
            return self.__orig_forward(input)

        model.forward = types.MethodType(forward, model)

        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


def phi3_vision_embeddings_forward(self, pixel_values: torch.FloatTensor):
    return self.get_img_features(pixel_values)


class Phi3VisionImageEmbeddingsPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = types.MethodType(phi3_vision_embeddings_forward, model)
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


def minicpm3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.
        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        orig_dtype = k.dtype
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
        q_fp32 = q.to(dtype=torch.float32, device=q.device)
        k_fp32 = k.to(dtype=torch.float32, device=k.device)
        q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
        k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
        return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)

    if output_attentions:
        return self._orig_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.shape

    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(hidden_states.shape[0], hidden_states.shape[1], 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    # Difference with original code, k_pe.new_empty create constant tensor in torchscript
    query_states = torch.concat([q_nope, q_pe], dim=-1)
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = torch.concat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)
    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


class MiniCPM3Patcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for block in self._model.model.layers:
            block.self_attn._orig_forward = block.self_attn.forward
            block.self_attn.forward = types.MethodType(minicpm3_attn_forward, block.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for block in self._model.model.layers:
            block.self_attn.forward = block.self_attn._orig_forward


class DeepseekPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        self_attn = {
            "deepseek_v3": deepseek_v3_attn_forward,
            "deepseek_v2": deepseek_v2_attn_forward,
            "deepseek": minicpm3_attn_forward,
        }

        self_attn_fwd = self_attn.get(self._model.config.model_type)
        for block in self._model.model.layers:
            if self_attn_fwd is not None:
                block.self_attn._orig_forward = block.self_attn.forward
                block.self_attn.forward = types.MethodType(self_attn_fwd, block.self_attn)
            if hasattr(block.mlp, "moe_infer"):
                block.mlp._org_moe_infer = block.mlp.moe_infer
                block.mlp.moe_infer = types.MethodType(deepseek_moe_infer, block.mlp)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for block in self._model.model.layers:
            block.self_attn.forward = block.self_attn._orig_forward
            if hasattr(block.mlp, "_orig_moe_infer"):
                block.mlp.moe_infer = block.mlp._orig_moe_infer


def deepseek_v3_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # modified from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L751
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        orig_dtype = k.dtype
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
        q_fp32 = q.to(dtype=torch.float32, device=q.device)
        k_fp32 = k.to(dtype=torch.float32, device=k.device)
        q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
        k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
        return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)

    if output_attentions:
        return self._orig_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    # Difference with original code, k_pe.new_empty create constant tensor in torchscript
    query_states = torch.concat([q_nope, q_pe], dim=-1)
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = torch.concat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)
    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def deepseek_v2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # modified from https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L806
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)

        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    if output_attentions:
        return self._orig_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.shape

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    # Difference with original code, k_pe.new_empty create constant tensor in torchscript
    query_states = torch.concat([q_nope, q_pe], dim=-1)
    # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = torch.concat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)
    # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def deepseek_moe_infer(self, x, topk_ids, topk_weight):
    cnts = torch.zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0).to(torch.long)
    idxs = torch.argsort(topk_ids.view(-1))
    sorted_tokens = x[idxs // topk_ids.shape[1]]

    outputs = []
    start_idx = torch.tensor(0, dtype=torch.long)
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        # difference with original: removed skiping expert if empty num_tokens
        expert_id = i + self.ep_rank * self.experts_per_rank
        expert = self.experts[expert_id]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    # difference with original: removed usage torch.new_empty if outputs empty
    outs = torch.cat(outputs, dim=0)

    new_x = torch.zeros_like(outs)
    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .to(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .to(new_x.dtype)
    )
    return final_out


class Qwen2VLLanguageModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any] = None,
    ):
        model.__orig_forward = model.forward

        def forward_wrap(
            self,
            attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            input_ids=None,
            use_cache=True,
        ):
            new_past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            result = self.__orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=new_past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )
            if past_key_values is not None:
                result["past_key_values"] = result["past_key_values"].to_legacy_cache()
            return result

        model.forward = types.MethodType(forward_wrap, model)
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class Qwen3VLLanguageModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L2156-L2178
        # moved audio and vision features processing outside model
        # This method in original model: https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1344-L1362
        def lm_forward(
            self,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
            use_cache=True,
        ):
            from transformers.cache_utils import DynamicCache

            pkv = DynamicCache.from_legacy_cache(past_key_values)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                past_key_values=pkv,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
            hidden_states = outputs[0]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states)
            return (logits, outputs.past_key_values.to_legacy_cache())

        model.__orig_forward = model.forward
        model.forward = types.MethodType(lm_forward, model)
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


def patch_qwen2vl_vision_blocks(model, force_new_behaviour=False):
    if not force_new_behaviour and is_transformers_version("<=", "4.48.99"):
        # Modified from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L390
        # added attention_mask input instead of internal calculation (unsupported by tracing due to cycle with dynamic len)
        def sdpa_attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> torch.Tensor:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

            if is_transformers_version(">=", "4.49"):
                if position_embeddings is None:
                    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                    cos = emb.cos().float()
                    sin = emb.sin().float()
                else:
                    cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            else:
                q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
                k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output

        # Modified from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L430
        # added attention_mask input propagation to self.attn
        def block_forward(self, hidden_states, attention_mask, rotary_pos_emb) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states), attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
            )
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
            return hidden_states

    else:
        # Modified from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L391
        # added attention_mask input instead of internal calculation (unsupported by tracing due to cycle with dynamic len)
        def sdpa_attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ):
            def rotate_half(x):
                """Rotates half the hidden dims of the input."""
                x1 = x[..., : x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2 :]
                return torch.cat((-x2, x1), dim=-1)

            def apply_rotary_pos_emb_vision(
                q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                orig_q_dtype = q.dtype
                orig_k_dtype = k.dtype
                q, k = q.float(), k.float()
                cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
                q_embed = (q * cos) + (rotate_half(q) * sin)
                k_embed = (k * cos) + (rotate_half(k) * sin)
                q_embed = q_embed.to(orig_q_dtype)
                k_embed = k_embed.to(orig_k_dtype)
                return q_embed, k_embed

            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            if position_embeddings is None:
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos = emb.cos().float()
                sin = emb.sin().float()
            else:
                cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output

        # Modified from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L446
        # added attention_mask input propagation to self.attn
        def block_forward(
            self,
            hidden_states,
            attention_mask,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
            )
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
            return hidden_states

    for block in model.blocks:
        block._orig_forward = block.forward
        block.forward = types.MethodType(block_forward, block)
        block.attn._orig_forward = block.attn.forward
        block.attn.forward = types.MethodType(sdpa_attn_forward, block.attn)


class Qwen2VLVisionEmbMergerPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any] = None,
    ):
        model.__orig_forward = model.forward

        # Modified from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1118
        # added attention_mask input instead cu_lens for its internal calculation model (unsupported by tracing due to cycle with dynamic len)
        # separated patch_embed and rot_pos_emb calls for performing as part of another model
        def image_embed_forward(
            self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, rotary_pos_emb: torch.Tensor
        ) -> torch.Tensor:
            for blk in self.blocks:
                hidden_states = blk(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
            return self.merger(hidden_states)

        model.forward = types.MethodType(image_embed_forward, model)
        super().__init__(config, model, model_kwargs)

    def __enter__(self):
        patch_qwen2vl_vision_blocks(self._model)
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward
        for block in self._model.blocks:
            block.forward = block._orig_forward
            block.attn.forward = block.attn._orig_forward


class Qwen2_5_VLVisionEmbMergerPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(config, model, model_kwargs)

        model.__orig_forward = model.forward

        # Modified from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L405
        # added attention_mask and window_attention_mask inputs instead cu_lens and window_cu_lens processing for its internal calculation model
        # (unsupported by tracing due to cycle with dynamic len)
        # separated patch_embed and rot_pos_emb calls for performing as part of another model
        def image_embed_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            window_attention_mask: torch.Tensor,
            window_index: torch.Tensor,
            rotary_pos_emb: torch.Tensor,
        ) -> torch.Tensor:
            seq_len = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    attention_mask_now = attention_mask
                else:
                    attention_mask_now = window_attention_mask
                hidden_states = blk(
                    hidden_states, attention_mask=attention_mask_now, position_embeddings=position_embeddings
                )

            hidden_states = self.merger(hidden_states)
            reverse_indices = torch.argsort(window_index)
            hidden_states = hidden_states[reverse_indices, :]

            return hidden_states

        model.forward = types.MethodType(image_embed_forward, model)
        super().__init__(config, model, model_kwargs)

    def __enter__(self):
        patch_qwen2vl_vision_blocks(self._model, force_new_behaviour=True)
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward
        for block in self._model.blocks:
            block.forward = block._orig_forward
            block.attn.forward = block.attn._orig_forward


class Qwen3VLVisionEmbMergerPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel"],
        model_kwargs: Dict[str, Any] = None,
    ):
        model.__orig_forward = model.forward

        # Modified from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1118
        # added attention_mask input instead cu_lens for its internal calculation model (unsupported by tracing due to cycle with dynamic len)
        # separated patch_embed and rot_pos_emb calls for performing as part of another model
        # This code part in original model: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L794-L808
        def image_embed_forward(
            self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, rotary_pos_emb: torch.Tensor
        ) -> torch.Tensor:
            deepstack_feature_lists = []
            for layer_num, blk in enumerate(self.blocks):
                hidden_states = blk(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
                if layer_num in self.deepstack_visual_indexes:
                    deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                        hidden_states
                    )
                    deepstack_feature_lists.append(deepstack_feature)
            last_hidden_state = self.merger(hidden_states)
            return last_hidden_state, torch.stack(deepstack_feature_lists, dim=0)

        model.forward = types.MethodType(image_embed_forward, model)
        super().__init__(config, model, model_kwargs)

    def __enter__(self):
        patch_qwen2vl_vision_blocks(self._model)
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward
        for block in self._model.blocks:
            block.forward = block._orig_forward
            block.attn.forward = block.attn._orig_forward


# copied from https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/granitemoe/modeling_granitemoe.py#L321
def _granite_moe_topk_gating_forward(self, hidden_states):
    # compute the top_k routing decision
    logits = self.layer(hidden_states).float()  # [batch_size x seq_len, num_experts]
    top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)  # [num_tokens, top_k]
    top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)  # [num_tokens, top_k]

    # compute number of input given to each expert
    zeros = torch.zeros(
        [top_k_gates.size(0), self.num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device
    )  # [num_tokens, num_experts]
    gates = zeros.scatter(1, top_k_indices, 1)  # [num_tokens, num_experts]
    expert_size = gates.long().sum(0)  # [num_experts,]
    # difference with original, removed expert_size = expert_size.tolist() due to incorrect tracing

    # sort and group input tokens according to expert assignment
    top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
    _, index_sorted_experts = top_k_experts.sort(0)  # [num_tokens * top_k]
    batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")  # [num_tokens * top_k]

    # gather the gate values for grouped input tokens
    top_k_gates = top_k_gates.flatten()  # [num_tokens * top_k]
    batch_gates = top_k_gates[index_sorted_experts]  # [num_tokens * top_k]

    return index_sorted_experts, batch_index, batch_gates, expert_size, logits


# copied from https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/granitemoe/modeling_granitemoe.py#L281
def _granite_moe_parallel_experts_forward(self, inputs, expert_size):
    output_list = []
    # difference with original
    # 1) expert_size is tensor instead of list of ints after gating patching, that does not allow use original inputs.split(expert_size)
    # 2) use index_start:next_index for obtaining expert inputs splits one by one instead of precomputed splits once before cycle
    index_start = torch.tensor(0, dtype=torch.int64)
    for i in range(self.num_experts):
        next_index = index_start + expert_size[i]
        output_list.append(F.linear(inputs[index_start:next_index], self.weight[i]))
        index_start = next_index
    results = torch.cat(output_list, dim=0)
    return results


class GraniteMoEModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for layer in self._model.model.layers:
            block_sparse_moe = layer.block_sparse_moe
            block_sparse_moe.router._orig_forward = block_sparse_moe.router.forward
            block_sparse_moe.router.forward = types.MethodType(
                _granite_moe_topk_gating_forward, block_sparse_moe.router
            )
            block_sparse_moe.input_linear._orig_forward = block_sparse_moe.input_linear.forward
            block_sparse_moe.input_linear.forward = types.MethodType(
                _granite_moe_parallel_experts_forward, block_sparse_moe.input_linear
            )
            block_sparse_moe.output_linear._orig_forward = block_sparse_moe.output_linear.forward
            block_sparse_moe.output_linear.forward = types.MethodType(
                _granite_moe_parallel_experts_forward, block_sparse_moe.output_linear
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for layer in self._model.model.layers:
            block_sparse_moe = layer.block_sparse_moe
            block_sparse_moe.router.forward = block_sparse_moe.router._orig_forward
            block_sparse_moe.input_linear.forward = block_sparse_moe.input_linear._orig_forward
            block_sparse_moe.output_linear.forward = block_sparse_moe.output_linear._orig_forward


class OVSeq2SeqModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        # re-use the patched forward method from the parent class
        self.super_patched_forward = self.patched_forward

        @functools.wraps(self.super_patched_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.super_patched_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            # with statful decoder, we always return the self attn only, cross attn is part of the state
            if (
                getattr(self.real_config, "stateful", False)
                and self.real_config._behavior == "decoder"
                and "past_key_values" in signature.parameters
            ):
                pkv = None
                pkv_arg_index = list(signature.parameters.keys()).index("past_key_values")

                if "past_key_values" in kwargs:
                    pkv = kwargs["past_key_values"]
                elif len(args) > pkv_arg_index:
                    pkv = args[pkv_arg_index]

                if pkv is not None:
                    if isinstance(pkv, EncoderDecoderCache):
                        pkv = pkv.self_attention_cache.to_legacy_cache()
                    else:
                        pkv = [pkv_item[:2] for pkv_item in pkv]
                    pkv = EncoderDecoderCache.from_legacy_cache(pkv)

                    if "past_key_values" in kwargs:
                        kwargs["past_key_values"] = pkv
                    elif len(args) > pkv_arg_index:
                        args[pkv_arg_index] = pkv

            outputs = self.super_patched_forward(*args, **kwargs)

            # the optimum-onnx seq2seq model patcher only converts to tuple starting from 4.48
            if isinstance(outputs.get("past_key_values"), (DynamicCache, EncoderDecoderCache)):
                outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

            # we still need to filter out cross attention in the case of non-stateful decoder
            filtered_outputs = {}
            for name, value in outputs.items():
                if (
                    self.real_config._behavior == "decoder"
                    and self.real_config.use_past_in_inputs
                    and name.startswith("past_key_values")
                ):
                    filtered_outputs[name] = tuple([v[:2] for v in value])
                else:
                    filtered_outputs[name] = value
            return filtered_outputs

        self.patched_forward = patched_forward

    def __enter__(self):
        super().__enter__()

        if is_transformers_version(">=", "4.53.0"):
            # for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
            # to avoid overflow issues on some hardware (e.g. Intel NPU)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

            # for decoder models, we use eager mask without vmap for sdpa as well
            # to avoid a nan output issue in OpenVINO that only happens in case of:
            # non-stateful models on cpu and stateful models on npu
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", eager_mask_without_vmap)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version(">=", "4.53"):
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask)


class SanaTextEncoderModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_transformers_version("<", "4.47.0"):
            from transformers.models.gemma2.modeling_gemma2 import GEMMA2_ATTENTION_CLASSES

            sdpa_attn = GEMMA2_ATTENTION_CLASSES["sdpa"]
            for layer in self._model.layers:
                layer.self_attn._orig_forward = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(sdpa_attn.forward, layer.self_attn)
        else:
            self._model.config._orig_attn_implementation = self._model.config._attn_implementation
            self._model.config._attn_implementation = "sdpa"

        if is_transformers_version(">=", "4.53"):
            # starting from 4.53, we get unmatching outputs if we use the boolean mask
            # TODO: This is an openvino issue (inconsistency between boolean and float masks)
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", eager_mask_without_vmap)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version("<", "4.47.0"):
            for layer in self._model.layers:
                layer.self_attn.forward = layer.self_attn._orig_forward
                del layer.self_attn._orig_forward
        else:
            self._model.config._attn_implementation = self._model.config._orig_attn_implementation
            del self._model.config._orig_attn_implementation

        if is_transformers_version(">=", "4.53"):
            # remove the eager_mask_without_vmap from the ALL_MASK_ATTENTION_FUNCTIONS
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask)


class MiniCPMModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for layer in model.model.layers:
            if hasattr(layer, "scale_depth"):
                layer.self_attn.o_proj.to(torch.float32)
                layer.mlp.down_proj.to(torch.float32)

        super().__init__(config, model, model_kwargs)


class CommonImageEmbeddingsModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L835
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-Gemma-3/src/transformers/models/gemma3/modeling_gemma3.py#L1321
        if hasattr(model, "model") and hasattr(model.model, "get_image_features"):
            model.forward = model.model.get_image_features
        else:
            model.forward = model.get_image_features
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


# Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-Gemma-3/src/transformers/models/gemma3/modeling_gemma3.py#L1147
def _gemma3_mm_update_causal_mask(
    self, attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training: bool = False
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted
        # form and requires no inversion or slicing.
        return attention_mask

    min_dtype = torch.finfo(torch.float16).min
    inputs_lead_dim, sequence_length = input_tensor.shape[:2]
    target_length = (
        attention_mask.shape[-1]
        if isinstance(attention_mask, torch.Tensor)
        else cache_position[0] + sequence_length + 1
    )

    causal_mask = torch.full(
        (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
    )

    # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

    # Apply bidirectional mask on images if token type ids are provided
    if token_type_ids is not None and sequence_length != 1:
        token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
        token_type_mask[token_type_ids == 0] = False  # if text token do not change anything
        token_type_mask = token_type_mask.unsqueeze(1).to(causal_mask.device, dtype=torch.bool)
        causal_mask = causal_mask.clone()
        causal_mask[:, :, :, :sequence_length] = causal_mask[:, :, :, :sequence_length].masked_fill(
            token_type_mask, 0.0
        )

    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]

        # Then apply padding mask (will mask pad tokens)
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

    return causal_mask


class Gemma3LMModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Difference from original:
        # uses Dynamic cache from legacy cache instead of HybridCache
        # calculate causal mask from multimodal

        def forward(
            self, attention_mask, position_ids, past_key_values, token_type_ids, inputs_embeds, use_cache=True
        ):
            pkv = DynamicCache.from_legacy_cache(past_key_values)

            past_seen_tokens = past_key_values[0][0].shape[-2]
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            forward_kwargs = {}

            if is_transformers_version("<", "4.52"):
                attention_mask = self._update_causal_mask_mm(
                    attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds
                )
            else:
                forward_kwargs["token_type_ids"] = token_type_ids

            result = self.__orig_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=pkv,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **forward_kwargs,
            )
            upd_pkv = result["past_key_values"]
            result["past_key_values"] = upd_pkv.to_legacy_cache()
            return result

        if is_transformers_version("<", "4.53.0"):
            model.__orig_forward = model.forward
            model.forward = types.MethodType(forward, model)

        super().__init__(config, model, model_kwargs)

    def __enter__(self):
        super().__enter__()

        if is_transformers_version("<", "4.52.0"):
            self._model._update_causal_mask_mm = types.MethodType(_gemma3_mm_update_causal_mask, self._model)
        elif (
            is_transformers_version("<", "4.53.0")
            and hasattr(self._model, "model")
            and hasattr(self._model.model, "_update_causal_mask")
        ):
            self._model.model._orig_update_causual_mask = self._model.model._update_causal_mask
            self._model.model._update_causal_mask = types.MethodType(_gemma3_mm_update_causal_mask, self._model.model)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version("<", "4.53.0"):
            self._model.forward = self._model.__orig_forward

        if is_transformers_version("<", "4.52"):
            del self._update_causal_mask_mm
        elif (
            is_transformers_version("<", "4.53.0")
            and hasattr(self._model, "model")
            and hasattr(self._model.model, "_orig_update_causual_mask")
        ):
            self._model.model._update_causal_mask = self._model.model._orig_update_causual_mask
            del self._model.model._orig_update_causual_mask


class Idefics3ImageEmbeddingsModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-SmolVLM-2/src/transformers/models/idefics3/modeling_idefics3.py#L999-L1005
        def get_image_features(self, pixel_values, patch_attention_mask, patch_position_ids):
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
                patch_position_ids=patch_position_ids,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)
            return image_hidden_states

        model.__orig_forward = model.forward
        model.forward = types.MethodType(get_image_features, model)
        super().__init__(config, model, model_kwargs)

    def __enter__(self):
        super().__enter__()

        # The difference from original code is only in getting patch_position_ids as input and propogation it into embeddings instead of calculation inside based on patch_attention_mask
        # method for calculation position_ids is not pytorch tracing friendly due to cycle over batch size.
        def transformer_forward(
            self,
            pixel_values,
            patch_attention_mask: Optional[torch.BoolTensor] = None,
            patch_position_ids: Optional[torch.IntTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            batch_size = pixel_values.size(0)
            if patch_attention_mask is None:
                patch_size = self.patch_size
                patch_attention_mask = torch.ones(
                    (
                        batch_size,
                        pixel_values.size(2) // patch_size,
                        pixel_values.size(3) // patch_size,
                    )
                )
                patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

            hidden_states = self.embeddings(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
                patch_position_ids=patch_position_ids,
            )

            patch_attention_mask = patch_attention_mask.view(batch_size, -1)
            # The call to `_upad_input` in `_flash_attention_forward` is expensive
            # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
            # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
            if not torch.any(~patch_attention_mask):
                patch_attention_mask = None
            elif not getattr(self, "_use_flash_attention_2", False):
                patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=patch_attention_mask,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            if not return_dict:
                return (last_hidden_state,) + encoder_outputs[1:]

            return BaseModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        def embeddings_forward(
            self,
            pixel_values: torch.FloatTensor,
            patch_attention_mask: torch.BoolTensor,
            patch_position_ids: Optional[torch.IntTensor] = None,
        ) -> torch.Tensor:
            batch_size, _, max_im_h, max_im_w = pixel_values.shape

            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if patch_position_ids is None:
                max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
                boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
                position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

                for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                    nb_patches_h = p_attn_mask[:, 0].sum()
                    nb_patches_w = p_attn_mask[0].sum()

                    if is_transformers_version("<", "4.55"):
                        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)
                    else:
                        h_indices = torch.arange(nb_patches_h, device=pixel_values.device, dtype=pixel_values.dtype)
                        w_indices = torch.arange(nb_patches_w, device=pixel_values.device, dtype=pixel_values.dtype)
                        fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
                        fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

                    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                    pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
                    position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
            else:
                position_ids = patch_position_ids

            position_ids = position_ids.to(self.position_embedding.weight.device)
            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

        def attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            if output_attentions:
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            batch_size, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

            attn_output = self.out_proj(attn_output)

            return attn_output, None

        self._model.vision_model._orig_forward = self._model.vision_model.forward
        self._model.vision_model.forward = types.MethodType(transformer_forward, self._model.vision_model)
        self._model.vision_model.embeddings._orig_forward = self._model.vision_model.embeddings.forward
        self._model.vision_model.embeddings.forward = types.MethodType(
            embeddings_forward, self._model.vision_model.embeddings
        )

        for layer in self._model.vision_model.encoder.layers:
            layer.self_attn._orig_forward = layer.self_attn.forward
            layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        self._model.forward = self._model.__orig_forward
        self._model.vision_model.forward = self._model.vision_model._orig_forward
        self._model.vision_model.embeddings.forward = self._model.vision_model.embeddings._orig_forward
        for layer in self._model.vision_model.encoder.layers:
            layer.self_attn.forward = layer.self_attn._orig_forward


# Adopted from https://github.com/huggingface/optimum/blob/main/optimum/bettertransformer/models/decoder_models.py#L367
def _blenderbot_attn_forward_legacy(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions or layer_head_mask is not None:
        return self._orig_forward(
            hidden_states, key_value_states, past_key_value, attention_mask, layer_head_mask, output_attentions
        )
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states)
    # get key, value proj
    # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `key_value_states` to support prefix tuning
    if is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    query_states = self._shape(query_states, tgt_len, bsz)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )

    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value


# Adopted from https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/blenderbot/modeling_blenderbot.py#L156
def _blenderbot_attn_forward_new(
    self,
    hidden_states: torch.Tensor,
    key_value_states=None,
    past_key_value=None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    from transformers.cache_utils import EncoderDecoderCache

    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    if output_attentions or layer_head_mask is not None:
        return self._orig_forward(
            hidden_states,
            key_value_states,
            past_key_value,
            attention_mask,
            layer_head_mask,
            output_attentions,
            cache_position,
        )
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = query_states

    if past_key_value is not None:
        if isinstance(past_key_value, EncoderDecoderCache):
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache
        else:
            curr_past_key_value = past_key_value

    current_states = key_value_states if is_cross_attention else hidden_states
    if is_cross_attention and past_key_value is not None and is_updated:
        # reuse k,v, cross_attentions
        key_states = curr_past_key_value.key_cache[self.layer_idx]
        value_states = curr_past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self.k_proj(current_states)
        value_states = self.v_proj(current_states)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True

    proj_shape = (bsz, self.num_heads, -1, self.head_dim)
    # difference with original, removed query_states = query_states.reshape(*proj_shape) * self.scale as scale is part of SDPA
    query_states = query_states.reshape(*proj_shape)
    key_states = key_states.reshape(*proj_shape)
    value_states = value_states.reshape(*proj_shape)

    # Difference with original, use SDPA instead of eager attention

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False,
    )

    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    outputs = (attn_output, None)

    if is_transformers_version("<", "4.54"):
        outputs += (past_key_value,)

    return outputs


if is_transformers_version(">=", "4.52"):
    _blenderbot_attn_forward = _blenderbot_attn_forward_new
else:
    _blenderbot_attn_forward = _blenderbot_attn_forward_legacy


def modulewise_patch(model, module_cls, patch_forward):
    for _, module in model.named_children():
        if isinstance(module, module_cls):
            module._orig_forward = module.forward
            module.forward = types.MethodType(patch_forward, module)
            return
        else:
            if len(list(module.children())) > 0:
                modulewise_patch(module, module_cls, patch_forward)


def modulewise_unpatch(model, module_cls):
    for _, module in model.named_children():
        if isinstance(module, module_cls):
            if hasattr(module, "_orig_forward"):
                module.forward = module._orig_forward
        else:
            if len(list(module.children())) > 0:
                modulewise_unpatch(module, module_cls)


class BlenderbotModelPatcher(OVSeq2SeqModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.56"):
            from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention

            modulewise_patch(self._model, BlenderbotAttention, _blenderbot_attn_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if is_transformers_version("<", "4.56"):
            from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention

            modulewise_unpatch(self._model, BlenderbotAttention)


class BlenderbotSmallModelPatcher(OVSeq2SeqModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.56"):
            from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallAttention

            modulewise_patch(self._model, BlenderbotSmallAttention, _blenderbot_attn_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if is_transformers_version("<", "4.56"):
            from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallAttention

            modulewise_unpatch(self._model, BlenderbotSmallAttention)


class PegasusModelPatcher(OVSeq2SeqModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.56"):
            from transformers.models.pegasus.modeling_pegasus import PegasusAttention

            modulewise_patch(self._model, PegasusAttention, _blenderbot_attn_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if is_transformers_version("<", "4.56"):
            from transformers.models.pegasus.modeling_pegasus import PegasusAttention

            modulewise_unpatch(self._model, PegasusAttention)


# Copied from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L596
# In 4.52.0, the loop is only over hitted experts, but we need to loop over all experts for tracing
def _qwen2moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class Qwen2MoEPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version(">=", "4.52.0"):
            from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

            modulewise_patch(self._model, Qwen2MoeSparseMoeBlock, _qwen2moe_sparse_block_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if is_transformers_version(">=", "4.52.0"):
            from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

            modulewise_unpatch(self._model, Qwen2MoeSparseMoeBlock)


class MarianModelPatcher(OVSeq2SeqModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version(">=", "4.49.0") and is_transformers_version("<", "4.56"):
            from transformers.models.marian.modeling_marian import MarianAttention

            modulewise_patch(self._model, MarianAttention, _blenderbot_attn_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if is_transformers_version(">=", "4.49.0") and is_transformers_version("<", "4.56"):
            from transformers.models.marian.modeling_marian import MarianAttention

            modulewise_unpatch(self._model, MarianAttention)


# Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/speecht5/modeling_speecht5.py#L698
# this is a patch to avoid PyTorch FE issue
# with the same tensor names on input and intermediate tensor for speaker_embeddings
def speecht5_decoder_prenet_forward(
    self,
    input_values: torch.Tensor,
    speaker_embeddings: Optional[torch.Tensor] = None,
):
    inputs_embeds = input_values
    for layer in self.layers:
        inputs_embeds = torch.nn.functional.relu(layer(inputs_embeds))
        inputs_embeds = self._consistent_dropout(inputs_embeds, self.config.speech_decoder_prenet_dropout)

    inputs_embeds = self.final_layer(inputs_embeds)
    inputs_embeds = self.encode_positions(inputs_embeds)

    if speaker_embeddings is not None:
        # this is a patch to avoid for PyTorch FE issue!!!
        # with the same tensor names on input and intermediate tensor in a model
        speaker_embeddings_norm = torch.nn.functional.normalize(speaker_embeddings)
        speaker_embeddings_unsqueeze = speaker_embeddings_norm.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)
        inputs_embeds = torch.cat([inputs_embeds, speaker_embeddings_unsqueeze], dim=-1)
        inputs_embeds = torch.nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))

    return inputs_embeds


# Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/speecht5/modeling_speecht5.py#L889
# this is a patch to avoid CPU plugin issue that is happened on 16-th iteration of token generation
# values computed by self-attention attn_output = torch.bmm(attn_probs, value_states) in a decoder gets incorrect
def speecht5_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    position_bias: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    serialize: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    # relative attention bias
    if position_bias is not None:
        reshape_q = query_states.contiguous().view(bsz * self.num_heads, -1, self.head_dim).transpose(0, 1)
        rel_pos_bias = torch.matmul(reshape_q, position_bias.transpose(-2, -1))
        rel_pos_bias = rel_pos_bias.transpose(0, 1).view(
            bsz * self.num_heads, position_bias.size(0), position_bias.size(1)
        )
        attn_weights += rel_pos_bias

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    # this is a patch to avoid CPU plugin issue!!!
    # issue is happened on 16-th iteration of token generation
    # since 16-th iteration of token generation, values computed by self-attention in a decoder gets incorrect
    eps = 1e-30
    attn_output = torch.bmm(attn_probs + eps, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, past_key_value


# Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/speecht5/modeling_speecht5.py#L1121
# this is a patch for a model to avoid incorrect tracing
# cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple are computed using encoder_hidden_states
def speecht5_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = True,
    serialize: bool = False,
):
    residual = hidden_states

    # Self Attention
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    # add present self-attn cache to positions 1,2 of present_key_value tuple
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        serialize=serialize,
    )

    hidden_states = self.dropout(hidden_states)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states

        # this is a patch for a model to avoid incorrect tracing!!!
        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        # are computed using encoder_hidden_states
        if past_key_value is not None and len(past_key_value) > 3:
            cross_attn_past_key_value = past_key_value[-2:]
        else:
            cross_attn_past_key_value = None
        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

    # Fully Connected
    hidden_states = hidden_states + self.feed_forward(hidden_states)
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


class OVSpeechT5ModelPatcher(ModelPatcher):
    def __enter__(self):
        if self.real_config._behavior != "vocoder":
            super().__enter__()
        if self.real_config._behavior == "decoder":
            self._model.speecht5.decoder.prenet.__orig_forward = self._model.speecht5.decoder.prenet.forward
            self._model.speecht5.decoder.prenet.forward = types.MethodType(
                speecht5_decoder_prenet_forward, self._model.speecht5.decoder.prenet
            )
            if is_transformers_version("<", "4.54"):
                for layer in self._model.speecht5.decoder.wrapped_decoder.layers:
                    layer.__orig_forward = layer.forward
                    layer.forward = types.MethodType(speecht5_decoder_layer_forward, layer)
                    layer.self_attn.__orig_forward = layer.self_attn.forward
                    layer.self_attn.forward = types.MethodType(speecht5_attention_forward, layer.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.real_config._behavior != "vocoder":
            super().__exit__(exc_type, exc_value, traceback)
        if self.real_config._behavior == "decoder":
            self._model.speecht5.decoder.prenet.forward = types.MethodType(
                self._model.speecht5.decoder.prenet.__orig_forward, self._model.speecht5.decoder.prenet
            )
            if is_transformers_version("<", "4.54"):
                for layer in self._model.speecht5.decoder.wrapped_decoder.layers:
                    layer.forward = types.MethodType(layer.__orig_forward, layer)
                    layer.self_attn.forward = types.MethodType(layer.self_attn.__orig_forward, layer.self_attn)

    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        def patched_encoder_forward(
            input_ids: torch.FloatTensor = None,
        ):
            encoder_attention_mask = torch.ones_like(input_ids)

            hidden_states = self._model.prenet(input_ids)

            encoder_out = self._model.wrapped_encoder(
                hidden_states=hidden_states,
                attention_mask=encoder_attention_mask,
                return_dict=True,
            )
            # downsample encoder attention mask
            if isinstance(model, SpeechT5EncoderWithSpeechPrenet):
                encoder_attention_mask = model.prenet._get_feature_vector_attention_mask(
                    encoder_out[0].shape[1], encoder_attention_mask
                )

            result = {
                "encoder_outputs": encoder_out.last_hidden_state,
                "encoder_attention_mask": encoder_attention_mask,
            }
            return result

        def patched_decoder_forward(
            inputs_embeds=None,
            speaker_embeddings=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
        ):
            if past_key_values is not None:
                past_key_values = [cache_item[:2] for cache_item in past_key_values]
                if is_transformers_version(">=", "4.56"):
                    past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

            output_sequence = inputs_embeds
            output_cross_attentions = False
            bsz = output_sequence.size(0)

            # Run the decoder prenet on the entire output sequence.
            decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)
            # Run the decoder layers on the last element of the prenet output.
            decoder_out = model.speecht5.decoder.wrapped_decoder(
                hidden_states=decoder_hidden_states[:, -1:],
                encoder_hidden_states=encoder_hidden_states[0],
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=output_cross_attentions,
                return_dict=True,
            )
            last_decoder_output = decoder_out.last_hidden_state.squeeze(1)

            # Predict the new mel spectrum for this step in the sequence.
            spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
            spectrum = spectrum.view(bsz, model.config.reduction_factor, model.config.num_mel_bins)

            # Extend the output sequence with the new mel spectrum.
            new_spectrogram = spectrum[:, -1, :].view(bsz, 1, model.config.num_mel_bins)
            output_sequence_out = torch.cat((output_sequence, new_spectrogram), dim=1)
            # Predict the probability that this is the stop token.
            prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

            past_key_values = decoder_out.past_key_values
            if past_key_values is not None:
                if isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values = past_key_values.self_attention_cache.to_legacy_cache()
                else:
                    past_key_values = [cache_item[:2] for cache_item in past_key_values]

            result = {
                "output_sequence_out": output_sequence_out,
                "spectrum": spectrum,
                "prob": prob,
                "past_key_values": past_key_values,
            }
            return result

        def patched_postnet_forward(raw_spectrogram: torch.FloatTensor):
            raw_spectrogram = raw_spectrogram.transpose(0, 1).flatten(1, 2)
            spectrogram = model.speech_decoder_postnet.postnet(raw_spectrogram)
            result = {"postnet_spectrogram": spectrogram}
            return result

        def patched_vocoder_forward(spectrogram: torch.FloatTensor):
            waveform = model(spectrogram)
            result = {"waveform": waveform}
            return result

        if self.real_config._behavior == "encoder":
            self.patched_forward = patched_encoder_forward
        elif self.real_config._behavior == "decoder":
            self.patched_forward = patched_decoder_forward
        elif self.real_config._behavior == "postnet":
            self.patched_forward = patched_postnet_forward
        elif self.real_config._behavior == "vocoder":
            self.patched_forward = patched_vocoder_forward
        else:
            raise ValueError("Unknown ")
        self.orig_forward = self.patched_forward


class Phi4MMLanguageModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if hasattr(model.config, "vision_lora") and model.config.vision_lora is not None:
            model.set_lora_adapter("vision")
        if hasattr(model.config, "speech_lora") and model.config.speech_lora is not None:
            model.set_lora_adapter("speech")

        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L2156-L2178
        # moved audio and vision features processing outside model
        def lm_forward(self, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache=True):
            pkv = DynamicCache.from_legacy_cache(past_key_values)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                past_key_values=pkv,
            )
            hidden_states = outputs[0]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states)
            return (logits, outputs.past_key_values.to_legacy_cache())

        model.__orig_forward = model.forward
        model.forward = types.MethodType(lm_forward, model)
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class Phi4MMAudioForwardEmbeddingsPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1121
        def forward(self, audio_input):
            if hasattr(self, "_forward_embeddings_code"):
                audio_input, masks = self._forward_embeddings_core(audio_input, None)
            else:
                audio_input, masks = self.embed(audio_input, None)
            return audio_input

        model.__orig_forward = model.forward
        model.forward = types.MethodType(forward, model)
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class Phi4MMAudioEncoderPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1201-L1212
        def forward(self, audio_feature, audio_mask):
            if hasattr(self, "init_relative_attention_bias"):
                relative_attention_bias = self.init_relative_attention_bias(audio_feature)

                _simplified_path = self.extra_layer_output_idx == -1 and relative_attention_bias is None

                if _simplified_path:
                    audio_feature, *_ = self.encoders(audio_feature, None, None, audio_mask)
                else:
                    for layer in self.encoders:
                        audio_feature, _, _, _ = layer(
                            audio_feature,
                            None,
                            None,
                            audio_mask,
                            relative_attention_bias=relative_attention_bias,
                        )
            else:
                relative_attention_bias = self.relative_attention_bias_layer(audio_feature)
                attention_mask = audio_mask.unsqueeze(1) + relative_attention_bias
                for layer in self.encoders:
                    audio_feature = layer(audio_feature, attention_mask)
            return audio_feature

        model.__orig_forward = model.forward
        model.forward = types.MethodType(forward, model)
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward


class Phi4MMVisionEmbeddingsPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        def get_img_features_legacy(
            self, pixel_values: torch.FloatTensor, patch_attention_mask=None, patch_position_ids=None
        ) -> torch.FloatTensor:
            LAYER_IDX = self.layer_idx
            TYPE_FEATURE = self.type_feature

            if self.freeze_img_processor:
                with torch.no_grad():
                    if patch_attention_mask is not None:
                        img_processor_output = self.img_processor(
                            pixel_values,
                            output_hidden_states=True,
                            patch_attention_mask=patch_attention_mask,
                            position_ids=patch_position_ids,
                        )
                    else:
                        img_processor_output = self.img_processor(
                            pixel_values, output_hidden_states=True, position_ids=patch_position_ids
                        )
                    img_feature = img_processor_output.hidden_states[LAYER_IDX]
            else:
                if patch_attention_mask is not None:
                    img_processor_output = self.img_processor(
                        pixel_values,
                        output_hidden_states=True,
                        patch_attention_mask=patch_attention_mask,
                        position_ids=patch_position_ids,
                    )
                else:
                    img_processor_output = self.img_processor(
                        pixel_values, output_hidden_states=True, position_ids=patch_position_ids
                    )
                img_feature = img_processor_output.hidden_states[LAYER_IDX]

            if TYPE_FEATURE == "patch":
                patch_feature = img_feature
                if self.image_token_compression is not None:
                    # reshape to 2D tensor
                    width = int(math.sqrt(patch_feature.size(1)))
                    patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                    # convert to NCHW
                    patch_feature = patch_feature.permute(0, 3, 1, 2)
                    if getattr(self, "img_processor_padding", None) is not None:
                        patch_feature = self.img_processor_padding(patch_feature)
                    patch_feature = self.image_token_compression(patch_feature)
                    # convert to NHWC
                    patch_feature = patch_feature.permute(0, 2, 3, 1)
                    patch_feature = patch_feature.view(
                        -1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1)
                    )
                elif getattr(self, "img_processor_padding", None) is not None:
                    width = int(math.sqrt(patch_feature.size(1)))
                    patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                    # convert to NCHW
                    patch_feature = patch_feature.permute(0, 3, 1, 2)
                    patch_feature = self.img_processor_padding(patch_feature)
                    # convert to NHWC
                    patch_feature = patch_feature.permute(0, 2, 3, 1)
                    patch_feature = patch_feature.view(
                        -1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1)
                    )
                return patch_feature

            if TYPE_FEATURE == "cls_patch":
                if self.image_token_compression is not None:
                    # reshape to 2D tensor
                    patch_feature = img_feature[:, 1:]
                    cls_feature = img_feature[:, 0]
                    width = math.sqrt(patch_feature.size(1))
                    patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
                    patch_feature = self.image_token_compression(patch_feature)
                    patch_feature = patch_feature.view(-1, patch_feature.size(-2) * patch_feature.size(-1))
                    img_feature = torch.cat([cls_feature, patch_feature], dim=1)
                return img_feature

        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L649
        # added possibility to provide patch_position_ids
        def get_img_features(
            self, pixel_values: torch.FloatTensor, patch_attention_mask=None, patch_position_ids=None
        ):
            img_processor_output = self.img_processor(
                pixel_values,
                patch_attention_mask=patch_attention_mask,
                output_hidden_states=True,
                position_ids=patch_position_ids,
            )
            img_feature = img_processor_output.hidden_states[self.layer_idx]

            patch_feature = img_feature
            # reshape to 2D tensor
            width = int(math.sqrt(patch_feature.size(1)))
            patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
            # convert to NCHW
            patch_feature = patch_feature.permute(0, 3, 1, 2)
            if getattr(self, "img_processor_padding", None) is not None:
                patch_feature = self.img_processor_padding(patch_feature)
            patch_feature = self.image_token_compression(patch_feature)
            # convert to NHWC
            patch_feature = patch_feature.permute(0, 2, 3, 1)
            patch_feature = patch_feature.view(
                -1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1)
            )
            return patch_feature

        model.__orig_forward = model.forward
        model.forward = types.MethodType(
            get_img_features_legacy if hasattr(model, "type_feature") else get_img_features, model
        )
        super().__init__(config, model, model_kwargs)

    def __enter__(self):
        super().__enter__()

        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L563
        # added possibility calculate position_ids outside
        def transformer_fwd(
            self,
            pixel_values,
            patch_attention_mask: Optional[torch.BoolTensor] = None,
            position_ids: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            batch_size = pixel_values.size(0)
            if patch_attention_mask is None:
                patch_attention_mask = torch.ones(
                    size=(
                        batch_size,
                        pixel_values.size(2) // self.config.patch_size,
                        pixel_values.size(3) // self.config.patch_size,
                    ),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )

            hidden_states = self.embeddings(
                pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, position_ids=position_ids
            )

            patch_attention_mask = patch_attention_mask.view(batch_size, -1)
            # The call to `_upad_input` in `_flash_attention_forward` is expensive
            # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
            # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
            if not torch.any(~patch_attention_mask):
                attention_mask = None
            else:
                attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            pooled_output = self.head(
                hidden_state=last_hidden_state,
                attention_mask=patch_attention_mask,
            )

            if not return_dict:
                return (last_hidden_state, pooled_output) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L76
        # used SDPA instead of eager attention
        def attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            if output_attentions:
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            batch_size, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

            attn_output = self.out_proj(attn_output)

            return attn_output, None

        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L488
        # moved position_ids calculation outside of model
        def embd_forward(
            self,
            pixel_values: torch.FloatTensor,
            patch_attention_mask: torch.BoolTensor,
            position_ids: torch.FloatTensor = None,
        ) -> torch.Tensor:
            batch_size = pixel_values.size(0)

            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if position_ids is None:
                max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
                max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
                boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
                position_ids = torch.full(
                    size=(
                        batch_size,
                        max_nb_patches_h * max_nb_patches_w,
                    ),
                    fill_value=0,
                )

                for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                    nb_patches_h = p_attn_mask[:, 0].sum()
                    nb_patches_w = p_attn_mask[0].sum()

                    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

                    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                    pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
                    position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

            position_ids = position_ids.to(self.position_embedding.weight.device)

            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

        if (
            getattr(self._model.img_processor.encoder.layers[0].self_attn.config, "_attn_implementation", "eager")
            != "sdpa"
        ):
            for layer in self._model.img_processor.encoder.layers:
                layer.self_attn._orig_forward = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
        self._model.img_processor._orig_forward = self._model.img_processor.forward
        self._model.img_processor.forward = types.MethodType(transformer_fwd, self._model.img_processor)
        self._model.img_processor.embeddings._orig_forward = self._model.img_processor.embeddings.forward
        self._model.img_processor.embeddings.forward = types.MethodType(
            embd_forward, self._model.img_processor.embeddings
        )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward
        for layer in self._model.img_processor.encoder.layers:
            if hasattr(layer.self_attn, "_orig_frward"):
                layer.self_attn.forward = layer.self_attn._orig_forward
        self._model.img_processor.forward = self._model.img_processor._orig_forward
        self._model.img_processor.embeddings.forward = self._model.img_processor.embeddings._orig_forward


class Llama4ImageEmbeddingsModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward

        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py#L1732-L1741
        def get_image_embeddings(self, pixel_values):
            if is_transformers_version("<", "4.57"):
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=self.config.vision_config.vision_feature_layer,
                    vision_feature_select_strategy=self.config.vision_config.vision_feature_select_strategy,
                )
            else:
                image_features = self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_select_strategy=self.config.vision_config.vision_feature_select_strategy,
                )
            vision_flat = image_features.view(-1, image_features.size(-1))
            projected_vision_flat = self.multi_modal_projector(vision_flat)
            return projected_vision_flat

        model.forward = types.MethodType(get_image_embeddings, model)
        super().__init__(config, model, model_kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        self._model.forward = self._model.__orig_forward


# modified from https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py#L229
# use real cos / sin instead of complex
def llama4_rope_forward(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)
    # Core RoPE block
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# use real cos / sin instead of complex
# Modified from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/llama4/modeling_llama4.py#L247
# Based on https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L292
# Native DeepSeek apply rotary emb works in the same way like llama4 apply rotary emb
def llama4_apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from transformers.models.llama.modeling_llama import rotate_half

    xq_ = xq.float()
    xk_ = xk.float()
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    b, h, s, d = xq_.shape
    xq_ = xq_.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = xk_.shape
    xk_ = xk_.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (xq_ * cos) + (rotate_half(xq_) * sin)
    k_embed = (xk_ * cos) + (rotate_half(xk_) * sin)
    return q_embed.type_as(xq), k_embed.type_as(xk)


# https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py#L329
# use real cos / sin instead of complex
def llama4_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[tuple[tuple[torch.Tensor]]] = None,
    past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    from transformers.models.llama4.modeling_llama4 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward

    past_key_value = past_key_value or past_key_values

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if self.use_rope:  # the 16E model skips rope for long context on certain layers
        cos, sin = position_embeddings[0], position_embeddings[1]
        query_states, key_states = llama4_apply_rotary_emb(
            query_states, key_states, cos.to(query_states.device), sin.to(query_states.device)
        )

    if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
        query_states = self.qk_norm(query_states)
        key_states = self.qk_norm(key_states)

    # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
    if self.attn_temperature_tuning and not self.use_rope:
        attn_scales = (
            torch.log1p(torch.floor((cache_position.float() + 1.0) / self.floor_scale)) * self.attn_scale + 1.0
        )
        attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand((*input_shape, 1, 1))  # batch size > 1
        query_states = (query_states * attn_scales).to(query_states.dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# modified from https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py#L157
# due to openvino transformations issue removed routed_out.view(-1, hidden_dim) in scatter_add_
def llama4_moe_forward(self, hidden_states):
    batch, seq_len, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, self.hidden_dim)
    router_logits = self.router(hidden_states).transpose(0, 1)
    tokens_per_expert = batch * seq_len

    router_top_value, router_indices = torch.topk(router_logits.transpose(0, 1), self.top_k, dim=1)
    router_scores = (
        torch.full_like(router_logits.transpose(0, 1), float("-inf"))
        .scatter_(1, router_indices, router_top_value)
        .transpose(0, 1)
    )
    # We do this to make sure we have -inf for non topK tokens before going through the !
    # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
    router_indices = (
        torch.arange(tokens_per_expert, device=hidden_states.device).view(1, -1).expand(router_scores.size(0), -1)
    )
    router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

    router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
    routed_in = torch.gather(
        input=hidden_states,
        dim=0,
        index=router_indices,
    ).to(hidden_states.device)
    # we gather inputs corresponding to each expert based on the router indices
    routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
    routed_out = self.experts(routed_in)
    out = self.shared_expert(hidden_states)
    # now that we finished expert computation -> we scatter add because we gathered previously
    # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
    # this scales a lot better if you do EP!
    out.scatter_add_(dim=0, index=router_indices, src=routed_out)
    return out, router_scores


class Llama4TextModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()

        self._model.model.rotary_emb._orig_forward = self._model.model.rotary_emb.forward
        self._model.model.rotary_emb.forward = types.MethodType(llama4_rope_forward, self._model.model.rotary_emb)
        for layer in self._model.model.layers[: self._model.model.config.num_hidden_layers]:
            if layer.is_moe_layer and is_transformers_version("<", "4.54"):
                layer.feed_forward._orig_forward = layer.feed_forward.forward
                layer.feed_forward.forward = types.MethodType(llama4_moe_forward, layer.feed_forward)
            layer.self_attn._orig_forward = layer.self_attn.forward
            layer.self_attn.forward = types.MethodType(llama4_attn_forward, layer.self_attn)

        if is_transformers_version(">=", "4.56"):
            # openvino is not able to trace through the new chunked_overlay with left_padding
            self.original_chunked_overlay = transformers.masking_utils.chunked_overlay
            transformers.masking_utils.chunked_overlay = (
                lambda chunk_size, left_padding: transformers.masking_utils._legacy_chunked_overlay(chunk_size)
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        self._model.model.rotary_emb.forward = self._model.model.rotary_emb._orig_forward
        for layer in self._model.model.layers[: self._model.model.config.num_hidden_layers]:
            if layer.is_moe_layer and is_transformers_version("<", "4.54"):
                layer.feed_forward.forward = layer.feed_forward._orig_forward
            layer.self_attn.forward = layer.self_attn._orig_forward

        if is_transformers_version(">=", "4.56"):
            transformers.masking_utils.chunked_overlay = self.original_chunked_overlay


# Vectorized implementation of ConvSequenceTransform to avoid if-else branching
class ConvSequenceTransform(torch.nn.Module):
    def __init__(self, conv_kernel_size, use_conv_bias, conv1, act, conv_bias):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.use_conv_bias = use_conv_bias
        self.conv1d = conv1
        self.act = act
        self.conv_bias = conv_bias

    def forward(self, hidden_states, cache_position, conv_state):
        # Pad the input
        is_prefill = cache_position.shape[0] == self.conv_kernel_size
        pad_value = (self.conv_kernel_size - hidden_states.shape[-1]) * (is_prefill)
        new_conv_state = torch.nn.functional.pad(hidden_states, (pad_value, 0))

        # Update convolutional state
        upd_cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)
        upd_conv_state = conv_state.roll(shifts=-1, dims=-1)
        upd_conv_state[:, :, upd_cache_position] = new_conv_state

        # compute both versions of hidden_states
        # 1. cache_position.shape[0] == self.conv_kernel_size, prefill step
        prefill_kernel_applied = self.conv1d(hidden_states)[:, :, : hidden_states.shape[-1]]

        # the second version
        # 2. re-compute conv_states for decoding step
        new_upd_conv_state = upd_conv_state * self.conv1d.weight[:, 0, :]
        decoder_kernel_applied = new_upd_conv_state.sum(dim=-1, keepdim=True)
        decoder_kernel_applied = decoder_kernel_applied.squeeze(2)
        if self.use_conv_bias:
            decoder_kernel_applied += self.conv_bias
        decoder_kernel_applied = decoder_kernel_applied.unsqueeze(-1)

        # Select the correct result
        is_prefill = torch.tensor(is_prefill, dtype=hidden_states.dtype)
        hidden_states = is_prefill * prefill_kernel_applied + (1 - is_prefill) * decoder_kernel_applied

        # Apply activation
        hidden_states = self.act(hidden_states)
        return hidden_states, upd_conv_state


# Vectorized implementation of the selective scan to compute SSM states and scan outputs at each time step
class SelectiveScan(torch.nn.Module):
    def forward(self, ssm, u, dt, A, B, C, D):
        # dt - [batch, seq_len, intermediate_size]
        # A - [intermediate_size, ssm_state_size]
        # B [batch, seq_len, ssm_state_size]
        # u, hidden states - [batch, seq_len, intermediate_size]
        dA = dt.unsqueeze(-1) * A
        dB_u = dt * u  # shape: (b, l, d)
        dB_u = dB_u.unsqueeze(-1) * B.unsqueeze(2)  # (b, l, d, 1) * (b, l, 1, n) => (b, l, d, n)
        dA_cumsum = torch.nn.functional.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
        x = dB_u * dA_cumsum + (ssm.unsqueeze(1) * dA[:, :1].exp())
        x = x.cumsum(1) / (dA_cumsum + 1e-12)
        y = (x * C.unsqueeze(2)).sum(dim=-1)
        return y + u * D, x[:, -1, :, :]


# The original implementation of this forward method can be found at:
# https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/mamba/modeling_mamba.py#L233
# This patch modifies the method to vectorize the selective scan procedure, enabling correct graph tracing
def mamba_mixer_forward(
    self,
    input_states,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        ssm_state = ssm_state.to(hidden_states.device)
        # use `cache_position.shape[0]` to check whether we are in prefill
        # stage, it's equivalent to check `cache_position[0] == 0`, which
        # breaks dynamo fullgraph constraints
        hidden_states, conv_state = self.conv_sequence_transform(
            hidden_states, cache_position, cache_params.conv_states[self.layer_idx]
        )
        cache_params.conv_states[self.layer_idx] = conv_state
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
    )

    if self.ssm_rms_normalization:
        B = self.ssm_rms_normalization(B, variance_epsilon=self.rms_eps)
        C = self.ssm_rms_normalization(C, variance_epsilon=self.rms_eps)
        time_step = self.ssm_rms_normalization(time_step, variance_epsilon=self.rms_eps)

    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    discrete_time_step = torch.nn.functional.softplus(discrete_time_step)  # [batch, intermediate_size, seq_len]

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())
    B = B.float()
    D = self.D.float()

    scan_output, ssm_state = self.selective_scan(
        ssm_state, hidden_states.float().transpose(1, 2), discrete_time_step, A, B, C, D
    )
    scan_output = scan_output.transpose(1, 2)
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
    return contextualized_states


# This patcher class serves the following purposes:
# 1. Inject a MambaCache structure into the original model to simplify input and output handling related to SSM states
# 2. Patch ConvSequenceTransform module to avoid if-else branching
# 3. Vectorize the selective scan operation to ensure correct behavior during JIT tracing
class MambaPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from transformers.models.mamba.modeling_mamba import MambaCache

        super().__init__(config, model, model_kwargs)

        class MambaCacheWrap(MambaCache):
            # This class serves to create MambaCache object for additional input data format
            # when lists of tensors representing convolution and SSM states are separated
            def __init__(
                self,
                config: "PretrainedConfig",
                batch_size: int = None,
                dtype: torch.dtype = torch.float32,
                device: Optional[Union[torch.device, str]] = None,
                max_batch_size: Optional[int] = None,
                conv_states: Optional[List[torch.Tensor]] = None,
                ssm_states: Optional[List[torch.Tensor]] = None,
            ):
                self.dtype = dtype
                self.max_batch_size = batch_size or max_batch_size
                self.intermediate_size = config.intermediate_size
                self.ssm_state_size = config.state_size
                self.conv_kernel_size = config.conv_kernel
                self.device = torch.device(device) if device is not None else torch.device("cpu")

                if conv_states is not None:
                    self.conv_states = conv_states
                else:
                    self.conv_states = []
                    for _ in range(config.num_hidden_layers):
                        conv_state: torch.Tensor = torch.zeros(
                            self.max_batch_size,
                            self.intermediate_size,
                            self.conv_kernel_size,
                            device=self.device,
                            dtype=dtype,
                        )
                        self.conv_states.append(conv_state)

                if ssm_states is not None:
                    self.ssm_states = ssm_states
                else:
                    self.ssm_states: List[torch.Tensor] = []
                    for _ in range(config.num_hidden_layers):
                        ssm_state: torch.Tensor = torch.zeros(
                            self.max_batch_size,
                            self.intermediate_size,
                            self.ssm_state_size,
                            device=self.device,
                            dtype=dtype,
                        )

                        self.ssm_states.append(ssm_state)

        def patched_forward(
            input_ids,
            attention_mask=None,
            cache_params=None,
            cache_position=None,
        ):
            use_cache = False
            wrapped_cache_params = None
            if cache_params is not None:
                use_cache = True
                # past_ssm_conv_states is a list of tuples of (ssm_state, conv_state) for each Mamba block
                ssm_states = [ssm_conv_state[0] for ssm_conv_state in cache_params]
                conv_states = [ssm_conv_state[1] for ssm_conv_state in cache_params]
                wrapped_cache_params = MambaCacheWrap(
                    self.real_config._config,
                    input_ids.shape[0],
                    conv_states=conv_states,
                    ssm_states=ssm_states,
                )

            result = self.orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                cache_params=wrapped_cache_params,
                use_cache=use_cache,
            )

            if use_cache:
                present_ssm_conv_states = []
                for ssm_state, conv_state in zip(result.cache_params.ssm_states, result.cache_params.conv_states):
                    present_ssm_conv_states.append(
                        (
                            ssm_state,
                            conv_state,
                        )
                    )
                result = {
                    "logits": result.logits,
                    "cache_params_present": present_ssm_conv_states,
                }

            return result

        self.patched_forward = patched_forward

        model_type = getattr(self.real_config._config, "model_type", None)
        self.ssm_rms_normalization = None

        # falcon-mamba model has only difference from mamba that is RMS normalization for B, C, and time-step coefficients
        if model_type == "falcon_mamba":
            from transformers.models.falcon_mamba.modeling_falcon_mamba import rms_forward

            self.ssm_rms_normalization = rms_forward

    def __enter__(self):
        super().__enter__()
        setattr(self._model, self.orig_forward_name, self.patched_forward)
        selective_scan = SelectiveScan()

        for layer in self._model.backbone.layers:
            layer.mixer.selective_scan = selective_scan
            layer.mixer._orig_forward = layer.mixer.forward

            layer.mixer.ssm_rms_normalization = self.ssm_rms_normalization

            layer.mixer.forward = types.MethodType(mamba_mixer_forward, layer.mixer)
            conv_transform = ConvSequenceTransform(
                layer.mixer.conv_kernel_size,
                layer.mixer.use_conv_bias,
                layer.mixer.conv1d,
                layer.mixer.act,
                layer.mixer.conv1d.bias,
            )
            layer.mixer.conv_sequence_transform = conv_transform

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        setattr(self._model, self.orig_forward_name, self.orig_forward)
        for layer in self._model.backbone.layers:
            layer.mixer.forward = layer.mixer._orig_forward


# https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L228
def qwen3_moe_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # TODO: we loop over all possible experts instead of hitted ones to avoid issues in graph execution.
    # expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class Qwen3MoeModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_transformers_version(">=", "4.53"):
            self.original_moe_forward = Qwen3MoeSparseMoeBlock.forward
            Qwen3MoeSparseMoeBlock.forward = qwen3_moe_forward_patched

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version(">=", "4.53"):
            Qwen3MoeSparseMoeBlock.forward = self.original_moe_forward


# The original implementation of this forward method can be found at:
# https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/zamba2/modeling_zamba2.py#L729
# This patch modifies `forward()` so that when traced by torch.jit, it works correctly
# during both the prefill and decoding steps.
# The patched version differs from the original in that it executes both the prefill
# and decoding branches every time to compute and store the values of B, C, hidden states,
# conv_state, and ssm_state in the cache.
# The distinction between prefill and decoding modes is determined by the sequence length
# (seq_len): 1. seq_len > 1 indicates the prefill phase;
# seq_len = 1 indicates the decoding phase.
def zamba2_mamba_mixer(
    self,
    hidden_states,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
        return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)

    def reshape_into_chunks(input_tensor, pad_size, chunk_size):
        # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
        input_tensor = pad_tensor_by_size(input_tensor, pad_size)
        if len(input_tensor.shape) == 3:
            # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
            return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
        else:
            # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
            return input_tensor.reshape(
                input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
            )

    def segment_sum(input_tensor):
        chunk_size = input_tensor.size(-1)
        # 1. expand input tensor to have an additional dimension and repeat along that dimension
        # [..., chunk_size] -> [..., chunk_size, chunk_size]
        input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
        # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
        mask = torch.tril(
            torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1
        )
        input_tensor = input_tensor.masked_fill(~mask, 0)
        # 3. compute actual cumsum
        tensor_segsum = torch.cumsum(input_tensor, dim=-2)
        # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
        tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
        return tensor_segsum

    input_states = hidden_states
    layer_idx = self.layer_idx
    if cache_params is not None and hasattr(cache_params, "mamba_layer_idx_mapping"):
        layer_idx = cache_params.mamba_layer_idx_mapping[layer_idx]

    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # distinguish prefill and decoding stage
    is_decoding = torch.tensor(seq_len == 1).to(dtype)

    # Gated MLP's linear projection
    input_states_prefill = (input_states * attention_mask[:, :seq_len, None]).to(dtype)
    input_states_dec = input_states
    input_states = input_states_dec * is_decoding + input_states_prefill * (1.0 - is_decoding)
    projected_states = self.in_proj(input_states)

    d_mlp = (
        projected_states.shape[-1]
        - 2 * self.intermediate_size
        - 2 * self.n_groups * self.ssm_state_size
        - self.num_heads
    ) // 2
    _, _, gate, hidden_states, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
    )

    # 1. Convolution sequence transformation
    # 1.1 Convolution sequence transformation for decoding step
    if cache_params is not None:
        conv_state_dec = cache_params.conv_states[layer_idx]
        conv_state_dec = torch.roll(conv_state_dec, shifts=-1, dims=-1)
        conv_state_dec[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states

        hidden_states_dec = torch.sum(conv_state_dec.to(projected_states.device) * self.conv1d.weight[:, 0, :], dim=-1)
        if self.use_conv_bias:
            hidden_states_dec += self.conv1d.bias
        hidden_states_dec = self.act(hidden_states_dec).to(dtype)[
            :, None, ...
        ]  # [batch, 1, intermediate_size] : decoding

        # 1.2 Convolution sequence transformation for prefill step
        hidden_states_prefill = hidden_states.transpose(1, 2)
        conv_state_prefill = torch.nn.functional.pad(
            hidden_states_prefill, (self.conv_kernel_size - hidden_states_prefill.shape[-1], 0)
        )

        hidden_states_prefill = self.act(self.conv1d(hidden_states_prefill).transpose(1, 2))[
            :, :seq_len, :
        ]  # [batch, intermediate_size, seq_len]
        if attention_mask is not None:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states_prefill = (hidden_states_prefill * attention_mask[:, :seq_len, None]).to(dtype)

        # Compute final conv state and set into the cache
        conv_state = conv_state_prefill * (1.0 - is_decoding) + conv_state_dec * is_decoding
        cache_params.conv_states[layer_idx].copy_(conv_state)
    else:
        hidden_states_prefill = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states_dec = hidden_states_prefill[:, :1]

    hidden_states_prefill, B_prefill, C_prefill = torch.split(
        hidden_states_prefill,
        [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
        dim=-1,
    )
    hidden_states_dec, B_dec, C_dec = torch.split(
        hidden_states_dec,
        [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
        dim=-1,
    )

    A = -torch.exp(self.A_log.float())  # [num_heads]

    # 2. SSM state
    # 2.1 Compute SSM state for decoding step
    if cache_params is not None:
        dt_dec = dt
        dt_dec = dt_dec.reshape(dt_dec.shape[0], -1, dt_dec.shape[-1])[:, :1, :]
        # dt - [B, 1, H], H - num_heads
        dt_dec = dt_dec.transpose(1, 2).expand(
            batch_size, dt_dec.shape[-1], self.head_dim
        )  # dt - [B, num_heads, head_dim]
        dt_bias_dec = self.dt_bias
        dt_bias_dec = dt_bias_dec.reshape(dt_bias_dec.shape[0], -1).expand(
            dt_bias_dec.shape[0], self.head_dim
        )  # dt_bias [num_heads, head_dim]
        dt_dec = torch.nn.functional.softplus(dt_dec + dt_bias_dec)
        dt_dec = torch.clamp(dt_dec, self.time_step_min)  # dt - [B, num_heads, head_dim]

        A_dec = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
        dA = torch.exp(dt_dec[..., None] * A_dec)

        # Discretize B
        # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
        # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
        B_dec = B_dec.reshape(batch_size, -1)[:, : self.n_groups * self.ssm_state_size]
        B_dec = B_dec.reshape(batch_size, self.n_groups, -1)[..., None, :]
        B_dec = B_dec.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B_dec.shape[-1]).contiguous()
        B_dec = B_dec.reshape(batch_size, -1, B_dec.shape[-1])
        dB = dt_dec[..., None] * B_dec[..., None, :]

        # Discretize x into dB
        # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
        hidden_states_dec = hidden_states_dec.reshape(batch_size, -1, self.head_dim)
        dBx = dB * hidden_states_dec[..., None]

        # State calculation
        new_ssm_state_dec = cache_params.ssm_states[layer_idx] * dA + dBx

        # Subsequent output
        # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
        C_dec = C_dec.reshape(batch_size, -1)[:, : self.n_groups * self.ssm_state_size]
        C_dec = C_dec.reshape(batch_size, self.n_groups, -1)[..., None, :]
        C_dec = C_dec.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C_dec.shape[-1]).contiguous()
        C_dec = C_dec.reshape(batch_size, -1, C_dec.shape[-1])

        ssm_states_dec = new_ssm_state_dec.to(C_dec.dtype)  # Shape: [b, h, d, n]

        # Reshape ssm_states to merge the first two dimensions
        ssm_states_reshaped = ssm_states_dec.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
        C_reshaped = C_dec.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
        y_dec = torch.bmm(ssm_states_reshaped, C_reshaped)
        y_dec = y_dec.view(batch_size, self.num_heads, self.head_dim)

        # D skip connection
        # [num_heads] -> [num_heads, head_dim]
        D_dec = self.D
        D_dec = D_dec[..., None].expand(D_dec.shape[0], self.head_dim)
        y_dec = (y_dec + hidden_states_dec * D_dec).to(y_dec.dtype)

        # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
        y_dec = y_dec.reshape(batch_size, -1)[:, None, ...]

    # 2.2 Compute SSM state for prefill step
    dt = torch.nn.functional.softplus(dt + self.dt_bias)
    dt = torch.clamp(dt, self.time_step_min)

    hidden_states_prefill = hidden_states_prefill.reshape(batch_size, seq_len, -1, self.head_dim).float()
    B_prefill = B_prefill.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
    C_prefill = C_prefill.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
    B_prefill = B_prefill.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
    C_prefill = C_prefill.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
    pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

    D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states_prefill, pad_size)

    # Discretize x and A
    hidden_states_prefill = hidden_states_prefill * dt[..., None]
    A = A.to(hidden_states_prefill.dtype) * dt

    # Rearrange into blocks/chunks
    hidden_states_prefill, A, B_prefill, C_prefill = [
        reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states_prefill, A, B_prefill, C_prefill)
    ]

    # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # Compute the output for each intra-chunk (diagonal blocks)
    # This is the analog of a causal mask
    L = torch.exp(segment_sum(A))

    # First, contraction of C and B to get G (attention-weights like)
    G_intermediate = C_prefill[:, :, :, None, :, :] * B_prefill[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
    G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

    # Compute M, equivalent to applying attention mask to weights
    M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_intermediate.sum(dim=-1)

    # Compute Y_diag (apply to values)
    Y_diag = (M[..., None] * hidden_states_prefill[:, :, None]).sum(3)

    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay_contraction = B_prefill * decay_states.permute(0, 2, 3, 1)[..., None]

    # permute back B * decay states
    states = (
        (
            B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]
            * hidden_states_prefill.permute(0, 1, 3, 2, 4)[..., None, :]
        )
        .sum(dim=3)
        .permute(0, 1, 2, 4, 3)
    )
    previous_states = torch.zeros_like(states[:, :1])

    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

    states_permuted = states.permute(0, 2, 1, 3, 4)
    result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
    new_states = result.permute(0, 2, 1, 3, 4)
    states, new_ssm_state_prefill = new_states[:, :-1], new_states[:, -1]

    # Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)

    # compute Yoff
    C_times_states = C_prefill[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    y = Y_diag + Y_off

    # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
    y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

    y = y + D_residual

    # Cutting off padded chunks
    pad_mask = torch.tensor(pad_size > 0).to(torch.long)
    y_new_len = y.size(1) * (1 - pad_mask) + seq_len * pad_mask
    y = y[:, :y_new_len]
    y_prefill = y.reshape(batch_size, seq_len, -1)

    if cache_params is not None:
        y = y_prefill[:, :seq_len] * (1.0 - is_decoding) + y_dec * is_decoding
        ssm_state = new_ssm_state_prefill * (1.0 - is_decoding) + new_ssm_state_dec * is_decoding

        # Set final ssm state into the cache
        cache_params.ssm_states[layer_idx].copy_(ssm_state)
    else:
        y = y_prefill

    scan_output = self.norm(y, gate)

    # Final linear projection
    contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]

    return contextualized_states


# This patcher class serves the following purposes:
# 1. Packs the KV-cache, conv_state, and ssm_state tensors into a Zamba2HybridDynamicCache structure
#    for subsequent invocation of the model's `forward` method.
# 2. Patches the Zamba2MambaMixer so that the traced `forward` function works correctly
#    during both the prefill and decoding steps.
class Zamba2ModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from transformers.models.zamba2.modeling_zamba2 import Zamba2HybridDynamicCache

        super().__init__(config, model, model_kwargs)

        class Zamba2HybridDynamicCacheWrap(Zamba2HybridDynamicCache):
            def __init__(self, config, batch_size: int, conv_states, ssm_states, key_cache, value_cache):
                # Call parent constructor with all required arguments
                super().__init__(config=config, batch_size=batch_size)
                self.conv_states = conv_states
                self.ssm_states = ssm_states
                self.key_cache = key_cache
                self.value_cache = value_cache
                self.layer_idx_mapping = {v: i for i, v in enumerate(config.hybrid_layer_ids)}

            def update(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: Optional[dict[str, Any]] = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # map layer_idx to key_cache (value_cache) idx
                layer_idx = self.layer_idx_mapping[layer_idx]
                # Update the cache
                if self.key_cache[layer_idx].shape[-1] == 0:
                    self.key_cache[layer_idx] = key_states
                    self.value_cache[layer_idx] = value_states
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
                layer_idx = self.layer_idx_mapping[layer_idx]
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
                # take any layer that contains cache and not empty tensor
                layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
                layer_idx = self.layer_idx_mapping[layer_idx]
                if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
                    return 0
                return self.key_cache[layer_idx].shape[-2]

        # the patch is needed to include KV-cache, Conv, and SSM states in the inputs and outputs.
        def patched_forward(
            input_ids,
            attention_mask=None,
            cache_params=None,
        ):
            num_hidden_layers = self.real_config._config.num_hidden_layers
            num_hybrid_layers = len(self.real_config._config.hybrid_layer_ids)
            use_cache = False
            wrapped_cache_params = None
            if cache_params is not None:
                use_cache = True
                conv_states = []
                ssm_states = []
                key_cache = []
                value_cache = []

                # decouple ssm_states, conv_states, keys and values from cache_params
                batch_size = cache_params[0].size(0)
                for idx in range(num_hidden_layers):
                    conv_states.append(cache_params[2 * idx])
                    ssm_states.append(cache_params[2 * idx + 1])

                for idx in range(num_hybrid_layers):
                    key_cache.append(cache_params[2 * num_hidden_layers + 2 * idx])
                    value_cache.append(cache_params[2 * num_hidden_layers + 2 * idx + 1])

                wrapped_cache_params = Zamba2HybridDynamicCacheWrap(
                    self.real_config._config, batch_size, conv_states, ssm_states, key_cache, value_cache
                )

            causal_lm_output = self.model_orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=wrapped_cache_params,
                use_cache=use_cache,
            )
            outputs = {
                "logits": causal_lm_output.logits,
            }

            if use_cache:
                past_key_values = causal_lm_output.past_key_values
                # unwrap Zamba2HybridDynamicCache object
                present_key_values = []
                for idx in range(num_hidden_layers):
                    present_key_values.append(past_key_values.conv_states[idx])
                    present_key_values.append(past_key_values.ssm_states[idx])

                for idx in range(num_hybrid_layers):
                    present_key_values.append(past_key_values.key_cache[idx])
                    present_key_values.append(past_key_values.value_cache[idx])

                outputs["present_key_values"] = present_key_values

            return outputs

        self.patched_forward = patched_forward
        self.model_orig_forward = self.orig_forward
        self.orig_forward = patched_forward

    def __enter__(self):
        from transformers.models.zamba2.modeling_zamba2 import Zamba2HybridLayer, Zamba2MambaDecoderLayer

        super().__enter__()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

        for layer in self._model.model.layers:
            if isinstance(layer, Zamba2MambaDecoderLayer):
                mamba_layer = layer.mamba
            elif isinstance(layer, Zamba2HybridLayer):
                mamba_layer = layer.mamba_decoder.mamba
            else:
                continue
            mamba_layer._orig_forward = mamba_layer.forward
            mamba_layer.forward = types.MethodType(zamba2_mamba_mixer, mamba_layer)

    def __exit__(self, exc_type, exc_value, traceback):
        from transformers.models.zamba2.modeling_zamba2 import Zamba2HybridLayer, Zamba2MambaDecoderLayer

        super().__exit__(exc_type, exc_value, traceback)
        setattr(self._model, self.orig_forward_name, self.model_orig_forward)
        for layer in self._model.model.layers:
            if isinstance(layer, Zamba2MambaDecoderLayer):
                mamba_layer = layer.mamba
            elif isinstance(layer, Zamba2HybridLayer):
                mamba_layer = layer.mamba_decoder.mamba
            else:
                continue
            mamba_layer.forward = mamba_layer._orig_forward


# The original implementation of this method can be found at:
# https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/lfm2/modeling_lfm2.py#L476
# This patch modifies `slow_forward()` so that when traced by torch.jit, it works correctly
# during both the prefill and decoding steps.
# The patched version differs from the original in that it executes both the prefill
# and decoding branches every time to compute and store the correct value of conv_state in the cache.
# The distinction between prefill and decoding modes is determined by the sequence length
# (seq_len): 1. seq_len > 1 indicates the prefill phase;
# seq_len = 1 indicates the decoding phase.
def lfm2_short_conv_forward_patched(
    self,
    x: torch.Tensor,
    past_key_values=None,
    cache_position=None,
    attention_mask=None,
):
    from transformers.models.lfm2.modeling_lfm2 import apply_mask_to_padding_states

    seqlen = x.shape[1]

    x = apply_mask_to_padding_states(x, attention_mask)
    BCx = self.in_proj(x).transpose(-1, -2)
    B, C, x = BCx.chunk(3, dim=-2)

    Bx = B * x

    if past_key_values is not None:
        layer_idx = past_key_values.conv_layer_idx_mapping[self.layer_idx]
        conv_state_dec = past_key_values.conv_cache[layer_idx]
        cache_position = cache_position.clamp(0, self.L_cache - 1)
        conv_state_dec = conv_state_dec.roll(shifts=-1, dims=-1)
        conv_state_dec[:, :, cache_position] = Bx.to(device=conv_state_dec.device, dtype=conv_state_dec.dtype)

        conv_out_dec = torch.sum(conv_state_dec.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
        if self.bias:
            conv_out_dec += self.conv.bias
        conv_out_dec = conv_out_dec.unsqueeze(-1)

        conv_state_prefill = torch.nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
        conv_out_prefill = self.conv(Bx)[..., :seqlen]

        is_decoding = torch.tensor(seqlen == 1).to(x.dtype)
        conv_out = conv_out_dec * is_decoding + conv_out_prefill * (1.0 - is_decoding)
        new_conv_state = conv_state_dec * is_decoding + conv_state_prefill * (1.0 - is_decoding)
        past_key_values.conv_cache[layer_idx].copy_(new_conv_state)
    else:
        conv_out = self.conv(Bx)[..., :seqlen]

    y = C * conv_out
    y = y.transpose(-1, -2).contiguous()
    y = self.out_proj(y)
    return y


# This patcher class serves the following purposes:
# 1. Packs the KV-cache and conv_state tensors into a Lfm2HybridConvCacheWrap structure
#    for subsequent invocation of the model's `forward` method.
# 2. Patches the Lfm2ShortConv so that the traced `slow_forward` function works correctly
#    during both the prefill and decoding steps.
class Lfm2ModelPatcher(OVDecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache

        super().__init__(config, model, model_kwargs)

        # This cache wrapper class serves for following purposes:
        # 1. Wraps KV-cache and conv_state to allow model instantiation from tensor lists.
        # 2. Removes the unused cache items that the source model contains.
        # For this reason cache items re-indexing is required.
        class Lfm2HybridConvCacheWrap(Lfm2HybridConvCache):
            def __init__(self, config, max_batch_size: int, conv_cache, key_cache, value_cache):
                # Call parent constructor with all required arguments
                super().__init__(config=config, max_batch_size=max_batch_size)
                self.key_cache = key_cache
                self.value_cache = value_cache
                self.conv_cache = conv_cache
                self.conv_layer_idx_mapping = {}
                self.attention_layer_idx_mapping = {}
                conv_layer_idx = 0
                attention_layer_idx = 0
                for i in range(config.num_hidden_layers):
                    if config.layer_types[i] == "full_attention":
                        self.attention_layer_idx_mapping[i] = attention_layer_idx
                        attention_layer_idx += 1
                    elif config.layer_types[i] == "conv":
                        self.conv_layer_idx_mapping[i] = conv_layer_idx
                        conv_layer_idx += 1

            def update(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: Optional[dict[str, Any]] = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """
                Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
                Patching is required for calculating of the correct cache index.
                """
                # Update the cache
                layer_idx = self.attention_layer_idx_mapping[layer_idx]
                if self.key_cache[layer_idx].numel() == 0:
                    self.key_cache[layer_idx] = key_states
                    self.value_cache[layer_idx] = value_states
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
                """
                Returns the sequence length of the cached states. A layer index can be optionally passed.
                Patching is required for calculating of the correct cache index.
                """
                # take any layer that contains cache and not empty tensor
                layer_idx = (
                    self.first_attention_layer if self.layer_types[layer_idx] != "full_attention" else layer_idx
                )
                layer_idx = self.attention_layer_idx_mapping[layer_idx]
                if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
                    return 0
                return self.key_cache[layer_idx].shape[-2]

        def patched_forward(
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cache_params=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position=None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
        ):
            use_cache = False
            wrapped_cache_params = None
            num_conv_layers = self.real_config._config.layer_types.count("conv")
            num_atten_layers = self.real_config._config.layer_types.count("full_attention")
            if cache_params is not None:
                use_cache = True
                conv_cache = []
                key_cache = []
                value_cache = []

                # decouple ssm_states, conv_states, keys and values from cache_params
                batch_size = cache_params[0].size(0)

                for idx in range(num_conv_layers):
                    conv_cache.append(cache_params[idx])

                for idx in range(num_atten_layers):
                    key_cache.append(cache_params[num_conv_layers + idx * 2])
                    value_cache.append(cache_params[num_conv_layers + idx * 2 + 1])

                wrapped_cache_params = Lfm2HybridConvCacheWrap(
                    self.real_config._config, batch_size, conv_cache, key_cache, value_cache
                )

            causal_lm_output = self.model_orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=wrapped_cache_params,
                use_cache=use_cache,
            )
            outputs = {
                "logits": causal_lm_output.logits,
            }

            if use_cache:
                key_values = causal_lm_output.past_key_values
                # unwrap Lfm2HybridDynamicCache object
                present_key_values = []
                for idx in range(num_conv_layers):
                    present_key_values.append(key_values.conv_cache[idx])

                for idx in range(num_atten_layers):
                    present_key_values.append(key_values.key_cache[idx])
                    present_key_values.append(key_values.value_cache[idx])

                outputs["present_key_values"] = present_key_values

            return outputs

        self.patched_forward = patched_forward
        self.model_orig_forward = self.orig_forward
        self.orig_forward = patched_forward

    def __enter__(self):
        from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

        super().__enter__()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

        for layer in self._model.model.layers:
            if hasattr(layer, "conv") and isinstance(layer.conv, Lfm2ShortConv):
                conv_layer = layer.conv
            else:
                continue
            conv_layer._orig_forward = conv_layer.slow_forward
            conv_layer.slow_forward = types.MethodType(lfm2_short_conv_forward_patched, conv_layer)

    def __exit__(self, exc_type, exc_value, traceback):
        from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

        super().__exit__(exc_type, exc_value, traceback)
        setattr(self._model, self.orig_forward_name, self.model_orig_forward)

        for layer in self._model.model.layers:
            if hasattr(layer, "conv") and isinstance(layer.conv, Lfm2ShortConv):
                conv_layer = layer.conv
            else:
                continue
            conv_layer.slow_forward = conv_layer._orig_forward


class GptOssModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_transformers_version(">=", "4.55.0"):
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

            self.original_gpt_oss_forward = GptOssExperts.forward
            GptOssExperts.forward = gpt_oss_forward

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version(">=", "4.55.0"):
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

            GptOssExperts.forward = self.original_gpt_oss_forward


# This patch overrides the following line in Transformers:
# https://github.com/huggingface/transformers/blob/v4.55-release/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py#L1553
# It is required to work around an OpenVINO issue:
# [CPU] Broadcast node '__module.model/aten::copy_/Broadcast' failed the check
# 'arg_shape[i - start_axis].is_dynamic()...' in src/core/shape_inference/include/broadcast_shape_inference.hpp:89
def granite_moe_hybrid_update_causal_mask(
    self,
    attention_mask,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    output_attentions: bool = False,
):
    dtype = input_tensor.dtype
    batch_size = input_tensor.shape[0]
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1]

    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            new_causal_mask = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            causal_mask = new_causal_mask

    return causal_mask


class GraniteMoeHybridModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: "PreTrainedModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache

        super().__init__(config, model, model_kwargs)

        class GraniteMoeHybridDynamicCacheWrap(HybridMambaAttentionDynamicCache):
            def __init__(self, config, batch_size: int, conv_states, ssm_states, key_cache, value_cache):
                # Call parent constructor with all required arguments
                super().__init__(config=config, batch_size=batch_size)
                self.conv_states = conv_states
                self.ssm_states = ssm_states
                self.key_cache = key_cache
                self.value_cache = value_cache
                self.attention_layer_idx_mapping = {}
                self.mamba_layer_idx_mapping = {}
                attention_layer_idx = 0
                mamba_layer_idx = 0
                for i in range(config.num_hidden_layers):
                    if self.layers_block_type[i] == "attention":
                        self.attention_layer_idx_mapping[i] = attention_layer_idx
                        attention_layer_idx += 1
                    elif self.layers_block_type[i] == "mamba":
                        self.mamba_layer_idx_mapping[i] = mamba_layer_idx
                        mamba_layer_idx += 1

            def update(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                layer_idx: int,
                cache_kwargs: Optional[dict[str, Any]] = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # map layer_idx to key_cache (value_cache) idx
                layer_idx = self.attention_layer_idx_mapping[layer_idx]
                # Update the cache
                if self.key_cache[layer_idx].shape[-1] == 0:
                    self.key_cache[layer_idx] = key_states
                    self.value_cache[layer_idx] = value_states
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
                layer_idx = self.attention_layer_idx_mapping[layer_idx]
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
                # take any layer that contains cache and not empty tensor
                layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
                layer_idx = self.attention_layer_idx_mapping[layer_idx]
                # if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
                #    return 0
                return self.key_cache[layer_idx].shape[-2]

        # the patch is needed to include KV-cache, Conv, and SSM states in the inputs and outputs.
        def patched_forward(
            input_ids,
            attention_mask=None,
            cache_params=None,
        ):
            num_mamba_layers = self.real_config._config.layer_types.count("mamba")
            num_attention_layers = self.real_config._config.layer_types.count("attention")
            use_cache = False
            wrapped_cache_params = None
            if cache_params is not None:
                use_cache = True
                conv_states = []
                ssm_states = []
                key_cache = []
                value_cache = []

                # decouple ssm_states, conv_states, keys and values from cache_params
                batch_size = cache_params[0].size(0)
                for idx in range(num_mamba_layers):
                    conv_states.append(cache_params[2 * idx])
                    ssm_states.append(cache_params[2 * idx + 1])

                for idx in range(num_attention_layers):
                    key_cache.append(cache_params[2 * num_mamba_layers + 2 * idx])
                    value_cache.append(cache_params[2 * num_mamba_layers + 2 * idx + 1])

                wrapped_cache_params = GraniteMoeHybridDynamicCacheWrap(
                    self.real_config._config, batch_size, conv_states, ssm_states, key_cache, value_cache
                )

            causal_lm_output = self.model_orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=wrapped_cache_params,
                use_cache=use_cache,
            )
            outputs = {
                "logits": causal_lm_output.logits,
            }

            if use_cache:
                past_key_values = causal_lm_output.past_key_values
                # unwrap GraniteMoeHybridDynamicCacheWrap object
                present_key_values = []
                for idx in range(num_mamba_layers):
                    present_key_values.append(past_key_values.conv_states[idx])
                    present_key_values.append(past_key_values.ssm_states[idx])

                for idx in range(num_attention_layers):
                    present_key_values.append(past_key_values.key_cache[idx])
                    present_key_values.append(past_key_values.value_cache[idx])

                outputs["present_key_values"] = present_key_values

            return outputs

        self.patched_forward = patched_forward
        self.model_orig_forward = self.orig_forward
        self.orig_forward = patched_forward

    def __enter__(self):
        def patch_sparse_moe(sparse_moe_layer):
            sparse_moe_layer.router._orig_forward = sparse_moe_layer.router.forward
            sparse_moe_layer.router.forward = types.MethodType(
                _granite_moe_topk_gating_forward, sparse_moe_layer.router
            )
            sparse_moe_layer.input_linear._orig_forward = sparse_moe_layer.input_linear.forward
            sparse_moe_layer.input_linear.forward = types.MethodType(
                _granite_moe_parallel_experts_forward, sparse_moe_layer.input_linear
            )
            sparse_moe_layer.output_linear._orig_forward = sparse_moe_layer.output_linear.forward
            sparse_moe_layer.output_linear.forward = types.MethodType(
                _granite_moe_parallel_experts_forward, sparse_moe_layer.output_linear
            )

        super().__enter__()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

        self._model.model._orig_update_causal_mask = self._model.model._update_causal_mask
        self._model.model._update_causal_mask = types.MethodType(
            granite_moe_hybrid_update_causal_mask, self._model.model
        )
        for idx, layer in enumerate(self._model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                patch_sparse_moe(layer.block_sparse_moe)
            if self.real_config._config.layers_block_type[idx] == "mamba":
                mamba_layer = layer.mamba
            else:
                continue
            mamba_layer._orig_forward = mamba_layer.forward
            mamba_layer.forward = types.MethodType(zamba2_mamba_mixer, mamba_layer)

    def __exit__(self, exc_type, exc_value, traceback):
        def unpatch_sparse_moe(sparse_moe_layer):
            sparse_moe_layer.router.forward = sparse_moe_layer.router._orig_forward
            sparse_moe_layer.input_linear.forward = sparse_moe_layer.input_linear._orig_forward
            sparse_moe_layer.output_linear.forward = sparse_moe_layer.output_linear._orig_forward

        super().__exit__(exc_type, exc_value, traceback)
        setattr(self._model, self.orig_forward_name, self.model_orig_forward)

        self._model.model._update_causal_mask = self._model.model._orig_update_causal_mask
        for idx, layer in enumerate(self._model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                unpatch_sparse_moe(layer.block_sparse_moe)
            if self.real_config._config.layers_block_type[idx] == "mamba":
                mamba_layer = layer.mamba
            else:
                continue
            mamba_layer.forward = mamba_layer._orig_forward


class BigBirdPegasusModelPatcher(OVSeq2SeqModelPatcher):
    def __enter__(self):
        super().__enter__()

        if self.real_config._behavior == "encoder" and self._model.config.attention_type == "block_sparse":
            logger.warning(
                "BigBirdPegasus model is using block sparse attention, which is not supported in ONNX export. "
                "The model will be exported with original full attention."
            )
            self._model.set_attention_type("original_full")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if self.real_config._behavior == "encoder" and self._model.config.attention_type == "block_sparse":
            self._model.set_attention_type("block_sparse")


# Patch MoE implementation to enable correct Torch tracing:
# https://huggingface.co/arcee-ai/Trinity-Nano-Preview/blob/main/modeling_afmoe.py#L265
#
# The original code contains a conditional branch inside a Python for-loop.
# For certain example inputs, this branch may be skipped during tracing,
# resulting in an incorrect or incomplete final graph.
#
# Additionally, the non-vectorized implementation produces a very large
# OpenVINO graph with excessive nodes, which is expensive for graph
# transformations and significantly increases model conversion time.
# So the patch provides a vectorized form of MoE.
def afmoe_moe_forward_patched(self, hidden_states):
    num_experts = self.config.num_experts
    batch_size, seq_len, hidden_dim = hidden_states.shape
    routing_weights, selected_experts = self.router(hidden_states, self.expert_bias)
    new_routing_weights = torch.zeros(batch_size * seq_len, self.config.num_experts, dtype=routing_weights.dtype)
    new_routing_weights.scatter_(dim=1, index=selected_experts, src=routing_weights)
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Process through shared experts
    if self.shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
    else:
        shared_output = torch.zeros_like(hidden_states)

    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_dim)
    act_fn = self.experts[0].act_fn

    # compute experts outputs in a vectorized form
    gate = torch.bmm(hidden_states, self.gate_projs)
    up = torch.bmm(hidden_states, self.up_projs)
    gate_up = act_fn(gate) * up
    next_states = torch.bmm(gate_up, self.down_projs)
    next_states = next_states.view(num_experts, batch_size, -1, hidden_dim)
    next_states = next_states * new_routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    next_states = next_states.sum(dim=0)

    shared_output = shared_output.view(batch_size, -1, hidden_dim)
    output = shared_output + next_states
    return output.view(batch_size, seq_len, hidden_dim)


class AfmoeModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for idx, layer in enumerate(self._model.model.layers):
            if layer.moe_enabled:
                afmoe_moe = layer.mlp
                num_experts = afmoe_moe.config.num_experts
                afmoe_moe._orig_forward = afmoe_moe.forward
                afmoe_moe.forward = types.MethodType(afmoe_moe_forward_patched, afmoe_moe)

                # prepare weigths to combine them from all experts to get the common gate, up and down weights
                # this is required for vectorized batched matmul representation of MoE
                # Fix CVS-180119: currently OpenVINO PyTorch Frontend incorrectly patching torch.bmm operation
                # with bf16 weights that leads to operands types mismatch in torch.bmm during TorchScript tracing
                # Now we align with hidden_states (that will be always fp32 due to patching
                # above for embedding layer during tracing)
                afmoe_moe.down_projs = (
                    torch.concat(
                        tuple(afmoe_moe.experts[i].down_proj.weight.unsqueeze(0) for i in range(num_experts)),
                        dim=0,
                    )
                    .transpose(1, 2)
                    .float()
                )
                afmoe_moe.gate_projs = (
                    torch.concat(
                        tuple(afmoe_moe.experts[i].gate_proj.weight.unsqueeze(0) for i in range(num_experts)),
                        dim=0,
                    )
                    .transpose(1, 2)
                    .float()
                )
                afmoe_moe.up_projs = (
                    torch.concat(
                        tuple(afmoe_moe.experts[i].up_proj.weight.unsqueeze(0) for i in range(num_experts)), dim=0
                    )
                    .transpose(1, 2)
                    .float()
                )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for idx, layer in enumerate(self._model.model.layers):
            if layer.moe_enabled:
                afmoe_moe = layer.mlp
                afmoe_moe.forward = afmoe_moe._orig_forward
                del afmoe_moe.down_projs, afmoe_moe.gate_projs, afmoe_moe.up_projs


# adopted from https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py#L197
class LlamaEagle3Attention(LlamaAttention):
    """
    LLaMA-style multi-headed self-attention adapted for Eagle-3 speculative decoding.

    This attention module extends the standard LLaMA attention mechanism to
    support Eagle-3 draft models, where the attention input is formed by
    concatenating the main model hidden states with the corresponding input
    embeddings.
    """

    def __init__(self, config):
        super().__init__(config, 0)
        self.q_proj = nn.Linear(config.hidden_size * 2, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=False)


# adopted from https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py#L268
class LlamaEagle3DecoderLayer(nn.Module):
    """
    Eagle-3 decoder layer that consumes main-model hidden states and input embeddings.

    This decoder layer is designed for the Eagle-3 draft model used in
    speculative decoding. Unlike a standard LLaMA decoder layer, it accepts
    two separate inputs:
        - `hidden_states`: hidden states produced by the main (target) model
        - `input_emb`: input token embeddings corresponding to the same positions

    This layer is used exclusively within the Eagle-3 draft model and is not
    compatible with token-only decoding pipelines.
    """

    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaEagle3Attention(config=config)
        self.mlp = LlamaMLP(config)
        self.last = last
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# adopted from https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py#L334
class LlamaEagle3Model(LlamaPreTrainedModel):
    """
    Eagle-3 draft model built on a LLaMA backbone for speculative decoding.

    This model extends the standard LLaMA architecture to support Eagle-3’s
    draft model workflow, where the model operates on hidden states produced
    by a main (target) model rather than directly on token IDs.

    **Triple Hidden-State Concatenation**: The model accepts three
    hidden-state tensors from the main model. These are concatenated along
    the hidden dimension and then projected via a linear layer (`fc`) to
    match `hidden_size`. This forms the effective input to the decoder.
    """

    def __init__(self, config: LlamaConfig):
        config.tie_word_embeddings = False
        super().__init__(config)
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.midlayer = LlamaEagle3DecoderLayer(config)
        self.target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.fc = nn.Linear(self.target_hidden_size * 3, self.hidden_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        batch_size, seq_length, _ = hidden_states.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        if hidden_states is None:
            hidden_states = torch.zeros(
                [batch_size, seq_length, self.embed_dim],
                self.embed_dim,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        # hidden_states = inputs_embeds
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        hidden_states = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@dataclass
class Eagle3Output(ModelOutput):
    """
    Output container for the Eagle-3 draft model.

    This class extends `ModelOutput` to hold the outputs of the Eagle-3
    speculative decoding model. It contains `d2t`,
    a mapping from generated draft tokens to the corresponding main
    model tokens. This is used to reconcile the draft model outputs
    with the main model vocabulary during speculative decoding.
    """

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    d2t: Optional[torch.LongTensor] = None


# adopted from https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py#L413
class LlamaEagle3ForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaEagle3Model(config)
        self.vocab_size = config.vocab_size
        self.identity = torch.nn.Identity()
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Eagle3Output:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.model.lm_head(hidden_states[:, slice_indices, :])

        d2t_out = self.identity(self.model.d2t)
        return Eagle3Output(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            d2t=d2t_out,
        )
