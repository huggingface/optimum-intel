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
import logging as log
import math
import types
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, TFPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.utils import is_tf_available

from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.model_patcher import (
    DecoderModelPatcher,
    ModelPatcher,
    Seq2SeqModelPatcher,
    override_arguments,
)
from optimum.intel.utils.import_utils import (
    _openvino_version,
    _torch_version,
    _transformers_version,
    is_diffusers_version,
    is_openvino_version,
    is_torch_version,
    is_transformers_version,
)


if TYPE_CHECKING:
    from transformers.cache_utils import Cache
    from transformers.modeling_utils import PreTrainedModel

    from optimum.exporters.onnx.config import OnnxConfig

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel


BETTERTRANSFORMER_IGNORE = [
    "codegen",
]

# in transformers 4.45 gpt_neo has SDPA
if is_transformers_version(">=", "4.44.99"):
    BETTERTRANSFORMER_IGNORE.append("gpt_neo")


def patch_model_with_bettertransformer(model):
    COLOR_RED = "\033[1;31m"
    COLOR_RESET = "\033[0m"

    # check that the model has not yet been pathced
    if hasattr(model, "use_bettertransformer") and model.use_bettertransformer is True:
        return model

    if is_transformers_version("<", "4.36") or is_torch_version("<", "2.1.1"):
        log.warning(
            COLOR_RED
            + "[WARNING] For good performance with stateful models, transformers>=4.36.2 and PyTorch>=2.1.1 are required. "
            f"This Python environment has Transformers {_transformers_version} and PyTorch {_torch_version}. "
            "Consider upgrading PyTorch and Transformers, for example by running "
            "`pip install --upgrade --upgrade-strategy eager optimum[openvino]`, and export the model again"
            + COLOR_RESET
        )

    if (
        getattr(model.config, "model_type") in {"gpt_bigcode", "llama", "gemma"}
        and is_transformers_version(">=", "4.38")
        and is_openvino_version("<", "2024.1.0-14612")
    ):
        # display commit-id only when a nightly/prerelease of OpenVINO is installed.
        display_version = (
            _openvino_version.split("-")[0] if is_openvino_version("<=", "2024.0.0-14509") else _openvino_version
        )
        log.warning(
            COLOR_RED
            + f"[WARNING] Stateful models are not supported for Llama, Gemma and GPTBigCode with Transformers "
            f"{_transformers_version} and OpenVINO {display_version}. For good performance, consider using a nightly OpenVINO build: "
            "https://docs.openvino.ai/2024/get-started/install-openvino.html. For gpt-bigcode and llama models, "
            "it is also an option to downgrade transformers: `pip install transformers==4.37.2`" + COLOR_RESET
        )

    # model already has required SDPA implementation
    if getattr(model, "_supports_sdpa", False) and getattr(model.config, "_attn_implementation", "eager") == "sdpa":
        return model

    if model.config.model_type in BETTERTRANSFORMER_IGNORE:
        return model

    try:
        model = model.to_bettertransformer()
    except Exception as e:
        log.warning(
            f"Cannot apply model.to_bettertransformer because of the exception:\n{e}."
            " Usage model with stateful=True may be non-effective if model does not contain torch.functional.scaled_dot_product_attention"
        )
        return model

    return model


def patch_update_causal_mask(
    model, transformers_version, inner_model_name="model", patch_fn=None, patch_extrnal_model=False
):
    if is_transformers_version(">=", transformers_version):
        inner_model = getattr(model, inner_model_name, None) if not patch_extrnal_model else model
        if inner_model is not None:
            if hasattr(inner_model, "_update_causal_mask"):
                inner_model._orig_update_causal_mask = inner_model._update_causal_mask
            patch_fn = patch_fn or _llama_gemma_update_causal_mask
            inner_model._update_causal_mask = types.MethodType(patch_fn, inner_model)


def unpatch_update_causal_mask(model, inner_model_name="model", patch_extrnal_model=False):
    inner_model = getattr(model, inner_model_name, None) if not patch_extrnal_model else model
    if inner_model is not None and hasattr(inner_model, "._orig_update_causal_mask"):
        inner_model._update_causal_mask = inner_model._orig_update_causal_mask


# initialization of sin/cos cached in bf16/fp16 leads to accuracy loss
# reinitialize them to save in float32 before export
def _reinitialize_cos_sin_cached_fp32(rotary_emb):
    if rotary_emb.cos_cached.dtype != torch.float32:
        rotary_emb._set_cos_sin_cache(
            seq_len=rotary_emb.max_position_embeddings, device=rotary_emb.inv_freq.device, dtype=torch.float32
        )


def _mixtral_sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """ """
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
        if is_transformers_version("<", "4.37.0"):
            current_hidden_states = expert_layer(current_state, routing_weights[top_x, idx, None])
        else:
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class MixtralModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        patch_update_causal_mask(self._model, "4.42.0")

        for layer in self._model.model.layers:
            layer.block_sparse_moe._unpatched_forward = layer.block_sparse_moe.forward
            layer.block_sparse_moe.forward = types.MethodType(
                _mixtral_sparse_moe_block_forward, layer.block_sparse_moe
            )
            if is_transformers_version("<", "4.44.99"):
                _reinitialize_cos_sin_cached_fp32(layer.self_attn.rotary_emb)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model.model, "_orig_update_causal_mask"):
            self._model.model._update_causal_mask = self._model.model._orig_update_causal_mask

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


class ChatGLMModelPatcher(DecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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


# adopted from
# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/gemma/modeling_gemma.py#L965
# https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L1058
def _llama_gemma_update_causal_mask_legacy(self, attention_mask, input_tensor, cache_position, past_seen_tokens=None):
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    if self.config._attn_implementation == "sdpa" and past_seen_tokens is not None:
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
        # in order to dispatch on Flash Attention 2.
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
        ):
            return None

    dtype, device = input_tensor.dtype, input_tensor.device

    # difference with original modeling
    # using minimum from dtype with larger bandwith (floa32) may lead to overflow
    # during execution on platforms with default lower precision (bfloat16, float16)
    min_dtype = torch.finfo(torch.float16).min
    sequence_length = input_tensor.shape[1]
    # difference with original modeling
    if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
        target_length = self.config.max_position_embeddings
    else:  # dynamic cache
        if past_seen_tokens is not None:
            current_length = past_seen_tokens + sequence_length + 1
        # TODO : remove after support of transformers >= v4.40.0
        else:
            current_length = cache_position[-1] + 1

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else current_length

    # difference with original modeling
    causal_mask = torch.full((sequence_length, target_length), fill_value=1, dtype=dtype, device=device) * min_dtype

    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
        elif attention_mask.dim() == 4:
            # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
            # cache. In that case, the 4D attention mask attends to the newest tokens only.
            if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                offset = cache_position[0]
            else:
                offset = 0
            mask_shape = attention_mask.shape
            mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
            causal_mask[
                : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
            ] = mask_slice

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


# adopted from https://github.com/huggingface/transformers/blob/f4014e75db0190792b3feeccfc5dc5b5f9f0ce7b/src/transformers/models/llama/modeling_llama.py#L1036
def _llama_gemma_update_causal_mask_latest(
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
    # using minimum from dtype with larger bandwith (floa32) may lead to overflow
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


# TODO : deprecate _llama_gemma_update_causal_mask_legacy when transformers>=4.41.0
if is_transformers_version(">", "4.40.2"):
    _llama_gemma_update_causal_mask = _llama_gemma_update_causal_mask_latest
else:
    _llama_gemma_update_causal_mask = _llama_gemma_update_causal_mask_legacy


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


def register_sin_cos_buffer(model):
    max_positions = model.config.max_position_embeddings

    # cos/sin for rotary position embeddings also having issues with bf16 and efficiency due to calculation on each step
    # use precomputed

    rotary_emb = model.model.layers[0].self_attn.rotary_emb
    dim, base = None, None
    inv_freq = getattr(rotary_emb, "inv_freq", None)
    if inv_freq is None:
        base = rotary_emb.base
        dim = rotary_emb.dim
    embed_positions = create_sinusoidal_positions(max_positions, dim, base, inv_freq)

    for layer in model.model.layers:
        layer.self_attn.rotary_emb.register_buffer("embed_positions", embed_positions)
        layer.self_attn.rotary_emb._orig_forward = layer.self_attn.rotary_emb.forward

        layer.self_attn.rotary_emb.forward = types.MethodType(
            llama_gemma_rotary_emb_forward, layer.self_attn.rotary_emb
        )


class LlamaModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # llama/gemma has some accuracy issues with bf16 with transformers >= 4.39
        # fill causal mask in slightly different way for avoid overflow on some platforms
        patch_update_causal_mask(self._model, "4.39.0", "model" if hasattr(self._model, "model") else "transformer")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model, "model" if hasattr(self._model, "model") else "transformer")


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


class MistralModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version(">=", "4.42.0") and is_transformers_version("<", "4.48.0"):
            # apply fix https://github.com/huggingface/transformers/commit/57d7594a79a9f5d835abf2d4d384db0e4818e548
            self._model.model._orig_update_causal_mask = self._model.model._update_causal_mask
            self._model.model._update_causal_mask = types.MethodType(_mistral_update_causal_mask, self._model.model)

        else:
            for layer in self._model.model.layers:
                if hasattr(layer.self_attn, "rotary_emb"):
                    _reinitialize_cos_sin_cached_fp32(layer.self_attn.rotary_emb)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if hasattr(self._model.model, "_orig_update_causal_mask"):
            self._model.model._update_causal_mask = self._model.model._orig_update_causal_mask

        for layer in self._model.model.layers:
            if hasattr(layer.self_attn, "rotary_emb") and hasattr(layer.self_attn.rotary_emb, "_orig_forward"):
                layer.self_attn.rotary_emb.forward = layer.self_attn.rotary_emb._orig_forward


SUPPORT_SDPA = is_torch_version(">", "2.1.0")


def _qwen_rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def _qwen_apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.float()
    t_pass_ = t_pass_.float()
    t_ = (t_ * cos) + (_qwen_rotate_half(t_) * sin)
    return torch.cat((t_, t_pass_), dim=-1).type_as(t)


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


class QwenModelPatcher(DecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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


class BaichuanModelPatcher(DecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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


def _mpt_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    batch_size, seq_length = hidden_states.shape[:2]

    mixed_qkv = self.Wqkv(hidden_states)
    query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
    query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states)
    else:
        past_key_value = (key_states, value_states)

    key_length = key_states.shape[-2]
    query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]
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

    return attn_output, None, past_key_value


def _mpt_block_forward(
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    # hidden_states: [batch_size, seq_length, hidden_size]
    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.norm_1(hidden_states)

    residual = hidden_states

    if not output_attentions:
        # Self attention.
        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
        )
    else:
        attn_outputs, attn_weights, past_key_value = self.attn._orig_forward(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
        )

    hidden_states = self.resid_attn_dropout(attn_outputs) + residual

    layernorm_output = self.norm_2(hidden_states)

    # Get residual
    residual = hidden_states

    # MLP.
    output = self.ffn(layernorm_output, residual)
    outputs = (output,)

    if use_cache:
        outputs += (past_key_value,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


class MPTModelPatcher(DecoderModelPatcher):
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


class InternLM2Patcher(DecoderModelPatcher):
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
    from transformers.cache_utils import Cache, DynamicCache
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

    if is_transformers_version(">=", "4.41.0"):
        from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

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


class Phi3ModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # currently, long RoPE can not be traced for long context support, disable it for avoid potential accuracy issues
        if self._model.config.max_position_embeddings != getattr(
            self._model.config, "original_max_position_embeddings", self._model.config.max_position_embeddings
        ):
            self._model.config.max_position_embeddings = self._model.config.original_max_position_embeddings

        if is_transformers_version(">=", "4.42.0") and is_transformers_version("<", "4.48.0"):
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

            if hasattr(layer.self_attn, "rotary_emb") and layer.self_attn.rotary_emb.inv_freq is None:
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


class AquilaModelPatcher(DecoderModelPatcher):
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


class XverseModelPatcher(DecoderModelPatcher):
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


class InternLMModelPatcher(DecoderModelPatcher):
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
    from optimum.bettertransformer.models.attention import raise_on_head_mask

    raise_on_head_mask(head_mask)
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


class CodeGenModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # whole codegen bettertransformer patch include attn.forward and does not cover codegen2.
        # For avoiding breaking model on tracing stage, we reduce area of bettertransformer patch only for _attn.
        from optimum.bettertransformer.models.attention import codegen_wrapped_scaled_dot_product

        attn_fn = codegen_wrapped_scaled_dot_product
        if is_torch_version(">=", "2.1.0") and is_transformers_version(">=", "4.45"):
            # in transformers 4.45 causal_mask const buffer was removed from the model
            # if it still exists, it means legacy remote code was loaded
            if hasattr(self._model.transformer.h[0].attn, "causal_mask"):
                attn_fn = _codegen_wrapped_scaled_dot_product_legacy

        for layer in self._model.transformer.h:
            if is_torch_version(">=", "2.1.0") and not self._model.config.output_attentions:
                orig_self_attn_fwd = layer.attn._attn
                layer.attn._attn = types.MethodType(attn_fn, layer.attn)
                layer.attn._orig_attn = orig_self_attn_fwd
        patch_update_causal_mask(self._model, "4.45.0", "transformer")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model, "transformer")
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


# Adapted from https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/dbrx/modeling_dbrx.py#L1228
def _dbrx_update_causal_mask_legacy(
    self, attention_mask: Optional[torch.Tensor], input_tensor: torch.Tensor, cache_position: torch.Tensor
) -> Optional[torch.Tensor]:
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None

    dtype, device = input_tensor.dtype, input_tensor.device
    # difference with original modeling
    # using minimum from dtype with larger bandwith (floa32) may lead to overflow
    # during execution on platforms with default lower precision (bfloat16, float16)
    min_dtype = torch.finfo(torch.float16).min
    sequence_length = input_tensor.shape[1]
    if hasattr(self.blocks[0].norm_attn_norm.attn, "past_key_value"):  # static cache
        target_length = self.config.max_position_embeddings
    else:  # dynamic cache
        target_length = (
            attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
        )
    # difference with original modeling
    # removed target_length = int(target_length).
    # Casting to int leads to constant folding during tracing that makes impossible to use model for sequence of different length
    causal_mask = torch.full((sequence_length, target_length), fill_value=1, dtype=dtype, device=device) * min_dtype
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
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
        elif attention_mask.dim() == 4:
            # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
            # cache. In that case, the 4D attention mask attends to the newest tokens only.
            if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                offset = cache_position[0]
            else:
                offset = 0
            mask_shape = attention_mask.shape
            mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
            causal_mask[
                : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
            ] = mask_slice

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
    ):
        # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
        is_tracing = (
            torch.jit.is_tracing()
            or isinstance(input_tensor, torch.fx.Proxy)
            or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
        )
        if not is_tracing and torch.any(attention_mask != 1):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


# adopted from https://github.com/huggingface/transformers/blob/1b3dba9417eebe16b7c206d1dfca6a4c7f11dbec/src/transformers/models/dbrx/modeling_dbrx.py#L1204
def _dbrx_update_causal_mask_latest(
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
    # using minimum from dtype with larger bandwith (floa32) may lead to overflow
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


if is_transformers_version(">", "4.40.2"):
    _dbrx_update_causal_mask = _dbrx_update_causal_mask_latest
else:
    _dbrx_update_causal_mask = _dbrx_update_causal_mask_legacy


class DBRXModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        # dbrx has some accuracy issues with bf16 with transformers >= 4.40
        # fill causal mask in slightly different way for avoid overflow on some platforms
        self._model.transformer._orig_update_causal_mask = self._model.transformer._update_causal_mask
        self._model.transformer._update_causal_mask = types.MethodType(
            _dbrx_update_causal_mask, self._model.transformer
        )

        # starting from transformers 4.41 issue also observable for calculation sin/cos for rotary_emb
        patch_rope_sin_cos = is_transformers_version(">=", "4.41.0")

        inv_freq = getattr(self._model.transformer.blocks[0].norm_attn_norm.attn.rotary_emb, "inv_freq")
        dim, base = None, None
        if inv_freq is None:
            dim = self._model.transformer.blocks[0].norm_attn_norm.attn.rotary_emb.dim
            base = self._model.transformer.blocks[0].norm_attn_norm.attn.rotary_emb.base
        max_positions = self._model.config.max_seq_len
        if patch_rope_sin_cos:
            embed_positions = create_sinusoidal_positions(max_positions, dim, base, inv_freq)

        for block in self._model.transformer.blocks:
            rotary_emb = block.norm_attn_norm.attn.rotary_emb
            # initialize inv_freq for torchscript tracing
            if rotary_emb.inv_freq is None:
                inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim)
                )
                rotary_emb.inv_freq = inv_freq

            if patch_rope_sin_cos:
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

    if is_transformers_version("<", "4.44.99"):
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    else:
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

    if is_transformers_version("<", "4.44.99"):
        rotary_ndims = self.rotary_emb.dim
    else:
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

    return attn_output, None, past_key_value


class PersimmonModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        patch_update_causal_mask(self._model, "4.42.0")

        for layer in self._model.model.layers:
            if is_torch_version(">=", "2.1.0"):
                orig_self_attn_fwd = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_persimmon_self_attn_sdpa_forward, layer.self_attn)
                layer.self_attn._orig_forward = orig_self_attn_fwd
            if is_transformers_version("<", "4.44.99"):
                _reinitialize_cos_sin_cached_fp32(layer.self_attn.rotary_emb)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model)
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


class JaisModelPatcher(DecoderModelPatcher):
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


class UpdateCausalMaskModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        patch_update_causal_mask(self._model, "4.42.0")
        if hasattr(self._model.model.layers[0].self_attn, "rotary_emb") and hasattr(
            self._model.model.layers[0].self_attn.rotary_emb, "_set_cos_sin_cache"
        ):
            for layer in self._model.model.layers:
                _reinitialize_cos_sin_cached_fp32(layer.self_attn.rotary_emb)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model)


class RotaryEmbPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.44.99"):
            for layer in self._model.model.layers:
                _reinitialize_cos_sin_cached_fp32(layer.self_attn.rotary_emb)


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


class FalconModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.44.99"):
            for layer in self._model.transformer.h:
                _reinitialize_cos_sin_cached_fp32(layer.self_attention.rotary_emb)
        else:
            patch_update_causal_mask(self._model, "4.45.0", "transformer", _falcon_update_causal_mask)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model, "transformer")


class GptNeoxModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.44.99"):
            for layer in self._model.gpt_neox.layers:
                _reinitialize_cos_sin_cached_fp32(layer.attention.rotary_emb)
        else:
            patch_update_causal_mask(self._model, "4.45.0", "gpt_neox")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model, "gpt_neox")


class GptJModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        patch_update_causal_mask(self._model, "4.45.0", "transformer")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model, "transformer")


class GptNeoxJapaneseModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version("<", "4.44.99"):
            for layer in self._model.gpt_neox_japanese.layers:
                _reinitialize_cos_sin_cached_fp32(layer.attention.rotary_emb)
        else:
            patch_update_causal_mask(self._model, "4.45.0", "gpt_neox_japanese")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model, "gpt_neox_japanese")


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


class GptNeoModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if is_transformers_version(">=", "4.45.0") and is_torch_version(">=", "2.1.0"):
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


class Gemma2ModelPatcher(LlamaModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, model, model_kwargs)

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            from transformers.cache_utils import DynamicCache

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
                input_ids = args[input_ids_index]
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


class DeciLMModelPatcher(DecoderModelPatcher):
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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


class InternVL2ChatLangModelPatcher(DecoderModelPatcher):
    def __init__(
        self, config: "OnnxConfig", model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Dict[str, Any]
    ):
        model_type = model.config.model_type
        patcher_for_model_type = {
            "llama": LlamaModelPatcher,
            "qwen2": UpdateCausalMaskModelPatcher,
            "phi3": Phi3ModelPatcher,
            "internlm2": InternLM2Patcher,
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
            if self._model.config.model_type == "qwen2" and self._model.config._attn_implementation != "sdpa":
                if is_transformers_version("<", "4.48"):
                    from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES

                    sdpa_attn = QWEN2_ATTENTION_CLASSES["sdpa"]
                    self._model.config._orig_attn_implementation = self._model.config._attn_implementation
                    self._model.config._attn_implementation = "sdpa"

                    for layer in self._model.model.layers:
                        layer.self_attn._orig_forward = layer.self_attn.forward
                        layer.self_attn.forward = types.MethodType(sdpa_attn.forward, layer.self_attn)

            if self._model.config.model_type == "llama" and self._model.config._attn_implementation != "sdpa":
                self._model.config._orig_attn_implementation = self._model.config._attn_implementation
                self._model.config._attn_implementation = "sdpa"
                if is_transformers_version("<", "4.47"):
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


class LlavaImageEmbeddingModelPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Dict[str, Any],
    ):
        model.__orig_forward = model.forward
        model.forward = types.MethodType(llava_vision_embed_forward, model)

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
        if not self._use_flash_attention_2
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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


class MiniCPM3Patcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for block in self._model.model.layers:
            block.self_attn._orig_forward = block.self_attn.forward
            block.self_attn.forward = types.MethodType(minicpm3_attn_forward, block.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for block in self._model.model.layers:
            block.self_attn.forward = block.self_attn._orig_forward


class DeepseekPatcher(DecoderModelPatcher):
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


class Qwen2VLLanguageModelPatcher(DecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
        ):
            from transformers.cache_utils import DynamicCache

            new_past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            result = self.__orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=new_past_key_values,
                inputs_embeds=inputs_embeds,
            )
            if past_key_values is not None:
                result["past_key_values"] = result["past_key_values"].to_legacy_cache()
            return result

        model.forward = types.MethodType(forward_wrap, model)
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
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


class GraniteMoEModelPatcher(LlamaModelPatcher):
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


# copied from https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L401
def gpt_bigcode_attn(self, query, key, value, attention_mask=None, head_mask=None):
    if head_mask is not None:
        # The super dispatch is done in the forward.
        raise ValueError("PyTorch SDPA does not support head_mask. Please open an issue in Transformers repository.")

    scale = None
    if not self.scale_attn_weights:
        scale = 1

    # MQA models: (batch_size, query_length, num_heads * head_dim)
    # MHA models: (batch_size, num_heads, query_length, head_dim)
    query_shape = query.shape
    batch_size = query_shape[0]
    key.shape[-2]

    if self.multi_query:
        query_length = query_shape[1]

        # SDPA requires the dimension [..., sequence_length, head_dim].
        query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Without these unsqueeze, SDPA complains as the query and key/value have a different number of dimensions.
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        # Although these expand are not numerically useful, PyTorch can not dispatch to memory-efficient backend
        # and flash attention backend (No available kernel.  Aborting execution.) from the shapes
        # query = [batch_size, num_heads, query_length, head_dim]
        # key = [batch_size, 1, past_length, head_dim]
        # value = [batch_size, 1, past_length, head_dim]
        #
        # torch==2.1.2 is bugged with non-contiguous inputs with custom attn_mask (https://github.com/pytorch/pytorch/issues/112577), hence the check.
        if is_torch_version(">=", "2.2.0"):
            key = key.expand(-1, self.num_heads, -1, -1)
            value = value.expand(-1, self.num_heads, -1, -1)
    else:
        query_length = query_shape[-1]

        # See the comment above.
        if query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not
    # create a causal mask in case query_length == 1.
    is_causal = True if self.is_causal and attention_mask is None and query_length > 1 else False
    # different from original, due to loading model weights in original format transformer.wte dtype may be different from query dtype
    if attention_mask is not None:
        attention_mask = attention_mask.to(query.dtype)
    sdpa_result = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=self.attn_pdrop if self.training else 0.0,
        is_causal=is_causal,
        scale=scale,
    )

    if self.multi_query:
        # (batch_size, num_heads, seq_len, head_dim) --> (batch_size, seq_len, num_heads, head_dim)
        sdpa_result = sdpa_result.transpose(1, 2)

        # Reshape is kind of expensive here, as it does a memory copy,
        # but I did not manage to make away without it (logits do not match when using view)
        # (batch_size, seq_len, num_heads, head_dim) --> (batch_size, seq_len, num_heads * head_dim)
        sdpa_result = sdpa_result.reshape(query_shape)

    return sdpa_result, None


class GptBigCodeModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        if getattr(self._model.config, "_attn_implementation", "eager") == "sdpa":
            for layer in self._model.transformer.h:
                layer.attn._orig_attn = layer.attn._attn
                layer.attn._attn = types.MethodType(gpt_bigcode_attn, layer.attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if getattr(self._model.config, "_attn_implementation", "eager") == "sdpa":
            for layer in self._model.transformer.h:
                layer.attn._attn = layer.attn._orig_attn


class StatefulSeq2SeqDecoderPatcher(Seq2SeqModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        model.__orig_forward = model.forward

        @functools.wraps(model.__orig_forward)
        def patched_forward(*args, **kwargs):
            from transformers.cache_utils import EncoderDecoderCache

            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            return_legacy_cache = False
            pkv_in_args = False
            legacy_pkv = None
            if "past_key_values" in kwargs:
                legacy_pkv = kwargs.pop("past_key_values", None)
            sign_names = list(signature.parameters.keys())
            pkv_argument_index = sign_names.index("past_key_values")
            if legacy_pkv is None and len(args) > pkv_argument_index:
                legacy_pkv = args[pkv_argument_index]
                pkv_in_args = True
            if legacy_pkv is not None:
                only_self_cache = [cache_item[:2] for cache_item in legacy_pkv]
                pkv = EncoderDecoderCache.from_legacy_cache(only_self_cache)
                return_legacy_cache = True
                if not pkv_in_args:
                    kwargs["past_key_values"] = pkv
                else:
                    args[pkv_argument_index] = pkv

            outputs = model.__orig_forward(*args, **kwargs)
            if return_legacy_cache:
                outputs.past_key_values = outputs.past_key_values.to_legacy_cache()

            return outputs

        model.forward = patched_forward

        super().__init__(config, model, model_kwargs)


class SanaTextEncoderModelPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()
        patch_update_causal_mask(self._model, "4.39.0", None, patch_extrnal_model=True)

        if self._model.config._attn_implementation != "sdpa":
            self._model.config._orig_attn_implementation = self._model.config._attn_implementation
            self._model.config._attn_implementation = "sdpa"
            if is_transformers_version("<", "4.47.0"):
                from transformers.models.gemma2.modeling_gemma2 import GEMMA2_ATTENTION_CLASSES

                sdpa_attn = GEMMA2_ATTENTION_CLASSES["sdpa"]
                for layer in self._model.layers:
                    layer.self_attn._orig_forward = layer.self_attn.forward
                    layer.self_attn.forward = types.MethodType(sdpa_attn.forward, layer.self_attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        unpatch_update_causal_mask(self._model, None, True)
        if hasattr(self._model.config, "_orig_attn_implementation"):
            self._model.config._attn_implementation = self._model.config._orig_attn_implementation
            for layer in self._model.layers:
                if hasattr(layer.self_attn, "_orig_forward"):
                    layer.self_attn.forward = layer.self_attn._orig_forward


class MiniCPMModelPatcher(DecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for layer in model.model.layers:
            if hasattr(layer, "scale_depth"):
                layer.self_attn.o_proj.to(torch.float32)
                layer.mlp.down_proj.to(torch.float32)

        super().__init__(config, model, model_kwargs)


class ConvSequenceTransform(torch.nn.Module):
    def __init__(self, conv_kernel_size, use_conv_bias, conv1, act, conv_bias):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.use_conv_bias = use_conv_bias
        self.conv1d = conv1
        self.act = act
        self.conv_bias = conv_bias

    def update_conv_state(
        self, conv_state, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state_1 = conv_state.roll(shifts=-1, dims=-1)
        upd_conv_state = conv_state_1.scatter(2, cache_position, new_conv_state)
        return upd_conv_state

    def get_positions(self, conv_state, cache_position):
        cache_position_clamped = cache_position.clamp(0, self.conv_kernel_size - 1)
        positions = cache_position_clamped.expand(conv_state.shape[0], conv_state.shape[1], -1)
        return positions

    def forward(self, hidden_states, cache_position, conv_state):
        pad_value = (self.conv_kernel_size - hidden_states.shape[-1]) * (
            cache_position.shape[0] == self.conv_kernel_size
        )
        new_conv_state = torch.nn.functional.pad(hidden_states, (pad_value, 0))
        upd_cache_position = self.get_positions(conv_state, cache_position)
        upd_conv_state = self.update_conv_state(conv_state, new_conv_state, upd_cache_position)
        if cache_position.shape[0] == self.conv_kernel_size:
            hidden_states = self.conv1d(hidden_states)[:, :, : hidden_states.shape[-1]]
        else:
            hidden_states = torch.sum(upd_conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            hidden_states += self.conv_bias
            hidden_states = hidden_states.unsqueeze(-1)
        hidden_states = self.act(hidden_states)  # [batch, intermediate_size, seq_len]
        return hidden_states, upd_conv_state


class SelectiveScan(torch.nn.Module):
    def forward(self, ssm, u, dt, A, B, C, D):
        dA = torch.einsum("bld,dn->bldn", dt, A)
        dB_u = torch.einsum("bld,bld,bln->bldn", dt, u, B)
        dA_cumsum = torch.nn.functional.pad(dA[:, 1:], (0, 0, 0, 0, 0, 1)).flip(1).cumsum(1).exp().flip(1)
        x = dB_u * dA_cumsum + (ssm.unsqueeze(1) * dA[:, :1].exp())
        x = x.cumsum(1) / (dA_cumsum + 1e-12)
        y = torch.einsum("bldn,bln->bld", x, C)
        return y + u * D, x[:, -1, :, :]


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
    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]

    # DIFF
    # discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

    # # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    # A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
    # discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
    # discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
    # deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    # if self.use_mambapy and self.training and cache_params is None:
    #     hs = pscan(discrete_A.transpose(1, 2), deltaB_u.transpose(1, 2)) # [batch, seq_len, intermediate_size, ssm_state_size]

    #     scan_output = (hs @ C.unsqueeze(-1)).squeeze(3).transpose(1, 2) # [batch, intermediate_size, seq_len]
    #     scan_output = scan_output + hidden_states * self.D[None, :, None]
    #     scan_output = scan_output * self.act(gate)
    # else:
    #     scan_outputs = []
    #     for i in range(seq_len):
    #         ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
    #         scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
    #         scan_outputs.append(scan_output[:, :, 0])
    #     scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
    #     scan_output = scan_output + (hidden_states * self.D[None, :, None])
    #     scan_output = (scan_output * self.act(gate))

    #     if cache_params is not None:
    #         cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    discrete_time_step = torch.nn.functional.softplus(discrete_time_step)  # [batch, intermediate_size, seq_len]
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


def falcon_mamba_mixer_forward(
    self,
    input_states,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
):
    from transformers.models.falcon_mamba.modeling_falcon_mamba import rms_forward

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

    B = rms_forward(B, variance_epsilon=self.rms_eps)
    C = rms_forward(C, variance_epsilon=self.rms_eps)
    time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)
    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]

    discrete_time_step = torch.nn.functional.softplus(discrete_time_step)  # [batch, intermediate_size, seq_len]
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


class MambaPatcher(ModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._patching_specs = []
        from transformers.cache_utils import MambaCache

        class MambaCacheWrap(MambaCache):
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
                # print(config.num_hidden_layers)

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

            def update_conv_state(
                self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
            ) -> torch.Tensor:
                conv_state = self.conv_states[layer_idx]
                cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

                conv_state = conv_state.roll(shifts=-1, dims=-1)
                conv_state = conv_state.scatter(
                    2, cache_position.expand(conv_state.shape[0], conv_state.shape[1], -1), new_conv_state
                )
                self.conv_states[layer_idx] = conv_state
                return self.conv_states[layer_idx]

        def forward_wrap(
            self, input_ids, attention_mask=None, cache_position=None, past_ssm_states=None, past_conv_states=None
        ):
            use_cache = False
            cache_params = None
            if past_ssm_states is not None and past_conv_states is not None:
                use_cache = True
                cache_params = MambaCacheWrap(
                    self.config,
                    input_ids.shape[0],
                    conv_states=list(past_conv_states),
                    ssm_states=list(past_ssm_states),
                )
            result = self.__orig_forward(
                input_ids=input_ids, cache_position=cache_position, cache_params=cache_params, use_cache=use_cache
            )
            if use_cache:
                return {
                    "logits": result.logits,
                    "ssm_states": result.cache_params.ssm_states,
                    "conv_states": result.cache_params.conv_states,
                }
            return result

        model.__orig_forward = model.forward
        model.forward = types.MethodType(forward_wrap, model)
        self._model = model

        self.orig_forward_name = "forward" if hasattr(self._model, "forward") else "call"
        self.orig_forward = getattr(self._model, self.orig_forward_name)

        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.real_config = config

        allow_past_in_outputs = hasattr(self.real_config, "use_past") and self.real_config.use_past

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            outputs = self.orig_forward(*args, **kwargs)

            # This code block handles different cases of the filterd_outputs input to align it with the expected
            # format of outputs. It is common for the output type of a model to vary, such as tensor, list,
            # tuple, etc. For Transformers models, the output is encapsulated in a ModelOutput object that
            # contains the output names of the model. In the case of Timm classification models, the output
            # is of type tensor. By default, it is assumed that the output names mentioned in the ONNX config
            # match the outputs in order.
            filterd_outputs = {}
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    output_name = config.torch_to_onnx_output_map.get(name, name)
                    if (
                        output_name in config.outputs
                        or (
                            allow_past_in_outputs and (name.startswith("ssm_states") or name.startswith("conv_states"))
                        )
                        or any(key.startswith(output_name) for key in config.outputs.keys())
                    ):
                        filterd_outputs[name] = value
            elif isinstance(outputs, (list, tuple)):
                outputs_list = list(config.outputs.keys())
                filterd_outputs = dict(zip(outputs_list, outputs))
            else:
                if len(config.outputs) > 1:
                    num_outputs = len(config.outputs)
                    outputs_str = ", ".join(config.outputs.keys())
                    raise ValueError(
                        f"config.outputs should have only one outputs, but it has {num_outputs} keys: {outputs_str}"
                    )
                else:
                    name = list(config.outputs.keys())[0]
                    filterd_outputs[name] = outputs
                name = list(config.outputs.keys())[0]
                filterd_outputs[name] = outputs

            return filterd_outputs

        self.patched_forward = patched_forward

    def __enter__(self):
        super().__enter__()
        selective_scan = SelectiveScan()

        for layer in self._model.backbone.layers:
            layer.mixer.selective_scan = selective_scan
            layer.mixer._orig_forward = layer.mixer.forward
            layer.mixer.forward = types.MethodType(mamba_mixer_forward, layer.mixer)
            conv_transform = ConvSequenceTransform(
                layer.mixer.conv_kernel_size,
                layer.mixer.use_conv_bias,
                layer.mixer.conv1d,
                layer.mixer.act,
                layer.mixer.conv1d.bias,
            )
            layer.mixer.conv_sequence_transform = torch.jit.script(conv_transform)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.forward = self._model.__orig_forward
        for layer in self._model.backbone.layers:
            layer.mixer.forward = layer.mixer._orig_forward


class FalconMambaPatcher(MambaPatcher):
    def __enter__(self):
        super().__enter__()
        selective_scan = SelectiveScan()

        for layer in self._model.backbone.layers:
            layer.mixer.selective_scan = selective_scan
            layer.mixer._orig_forward = layer.mixer.forward
            layer.mixer.forward = types.MethodType(falcon_mamba_mixer_forward, layer.mixer)
            conv_transform = ConvSequenceTransform(
                layer.mixer.conv_kernel_size,
                layer.mixer.use_conv_bias,
                layer.mixer.conv1d,
                layer.mixer.act,
                layer.mixer.conv1d.bias,
            )
            layer.mixer.conv_sequence_transform = torch.jit.script(conv_transform)
