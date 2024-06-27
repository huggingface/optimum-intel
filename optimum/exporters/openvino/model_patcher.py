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
import logging as log
import math
import types
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_tf_available

from optimum.exporters.onnx.model_patcher import DecoderModelPatcher
from optimum.intel.utils.import_utils import (
    _openvino_version,
    _torch_version,
    _transformers_version,
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


BETTERTRANSFORMER_IGNORE = ("codegen",)


def patch_model_with_bettertransformer(model):
    COLOR_RED = "\033[1;31m"
    COLOR_RESET = "\033[0m"

    # check that the model has not yet been pathced
    if hasattr(model, "use_bettertransformer") and model.use_bettertransformer is True:
        return model

    if is_transformers_version("<", "4.36") or is_torch_version("<", "2.1.1"):
        log.warn(
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
        log.warn(
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
        log.warn(
            f"Cannot apply model.to_bettertransformer because of the exception:\n{e}."
            " Usage model with stateful=True may be non-effective if model does not contain torch.functional.scaled_dot_product_attention"
        )
        return model

    return model


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
    attention_mask = ~attention_mask
    context_layer = torch.nn.functional.scaled_dot_product_attention(
        query_layer, key_layer, value_layer, attention_mask.to(torch.float32)
    )
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


class GemmaModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # gemma has some accuracy issues with bf16 with transformers >= 4.39
        # fill causal mask in slightly different way for avoid overflow on some platforms
        if is_transformers_version(">=", "4.39.0"):
            self._model.model._orig_update_causal_mask = self._model.model._update_causal_mask
            self._model.model._update_causal_mask = types.MethodType(
                _llama_gemma_update_causal_mask, self._model.model
            )

        # init inv_freq for torchscript tracing
        # https://github.com/huggingface/transformers/blob/ed74d97871468f3a4695ede50abdc0b55717a84d/src/transformers/models/gemma/modeling_gemma.py#L108
        for layer in self._model.model.layers:
            if layer.self_attn.rotary_emb.inv_freq is None:
                rotary_emb = layer.self_attn.rotary_emb
                layer.self_attn.rotary_emb.inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim)
                )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model.model, "_orig_update_causal_mask"):
            self._model.model._update_causal_mask = self._model.model._orig_update_causal_mask


class LlamaModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # llama has some accuracy issues with bf16 with transformers >= 4.39
        # fill causal mask in slightly different way for avoid overflow on some platforms
        if is_transformers_version(">=", "4.39.0"):
            self._model.model._orig_update_causal_mask = self._model.model._update_causal_mask
            self._model.model._update_causal_mask = types.MethodType(
                _llama_gemma_update_causal_mask, self._model.model
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        if hasattr(self._model.model, "_orig_update_causal_mask"):
            self._model.model._update_causal_mask = self._model.model._orig_update_causal_mask


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

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors."""
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
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

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


# Adapted from https://github.com/huggingface/transformers/blob/ccdabc5642bf84849af93f591e207dc625c8e1e1/src/transformers/models/phi3/modeling_phi3.py#L426
def _phi3_self_attn_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
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

    # TO DO: remove llama imports when transformers with phi3 support will be released
    try:
        from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv
    except ImportError:
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
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


class Phi3ModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        # https://github.com/huggingface/transformers/blob/30ee508c6c92a1c0aa0281d193c7c0fb815b8d2f/src/transformers/models/phi3/modeling_phi3.py#L113
        # init inv_freq for torchscript tracing
        for layer in self._model.model.layers:
            if is_torch_version(">=", "2.1.0"):
                orig_self_attn_fwd = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_phi3_self_attn_sdpa_forward, layer.self_attn)
                layer.self_attn._orig_forward = orig_self_attn_fwd

            if layer.self_attn.rotary_emb.inv_freq is None:
                rotary_emb = layer.self_attn.rotary_emb
                layer.self_attn.rotary_emb.inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim)
                )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
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


class CodeGenModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # whole codegen bettertransformer patch include attn.forward and does not cover codegen2.
        # For avoiding breaking model on tracing stage, we reduce area of bettertransformer patch only for _attn.
        from optimum.bettertransformer.models.attention import codegen_wrapped_scaled_dot_product

        for layer in self._model.transformer.h:
            if is_torch_version(">=", "2.1.0") and not self._model.config.output_attentions:
                orig_self_attn_fwd = layer.attn._attn
                layer.attn._attn = types.MethodType(codegen_wrapped_scaled_dot_product, layer.attn)
                layer.attn._orig_attn = orig_self_attn_fwd

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for layer in self._model.transformer.h:
            if hasattr(layer.attn, "_orig_attn"):
                layer.attn._attn = layer.attn._orig_attn


# adapted from https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/dbrx/modeling_dbrx.py#L763
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


# adapted from https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/dbrx/modeling_dbrx.py#L1228
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

        for block in self._model.transformer.blocks:
            rotary_emb = block.norm_attn_norm.attn.rotary_emb
            # initialize inv_freq for torchscript tracing
            if rotary_emb.inv_freq is None:
                inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim)
                )
                rotary_emb.inv_freq = inv_freq
            # remove continue-operator from iteration loop over experts
            block.ffn.experts._orig_forward = block.ffn.experts.forward
            block.ffn.experts.forward = types.MethodType(_dbrx_experts_forward, block.ffn.experts)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.transformer._update_causal_mask = self._model.transformer._orig_update_causal_mask
        for block in self._model.transformer.blocks:
            block.ffn.experts.forward = block.ffn.experts._orig_forward


def _persimmon_self_attn_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional["Cache"] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
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

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim :],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim :],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    if past_key_value is not None:
        # Specific to RoPE models with partial rotation
        cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
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
        for layer in self._model.model.layers:
            if is_torch_version(">=", "2.1.0"):
                orig_self_attn_fwd = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_persimmon_self_attn_sdpa_forward, layer.self_attn)
                layer.self_attn._orig_forward = orig_self_attn_fwd

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
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
