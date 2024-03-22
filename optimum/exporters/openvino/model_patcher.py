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

import logging as log
import math
import types
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union

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
    from transformers.modeling_utils import PreTrainedModel

    from optimum.exporters.onnx.config import OnnxConfig

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel


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

    if self.pre_seq_len is not None:
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


class ChatGLMModelPatcher(DecoderModelPatcher):
    def __init__(
        self,
        config: "OnnxConfig",
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        model_kwargs: Dict[str, Any],
    ):
        super().__init__(config, model, model_kwargs)

        self.original_chatglm_transformer_forward = model.transformer.forward

    def __enter__(self):
        super().__enter__()
        self._model.transformer.forward = types.MethodType(_chatglm_transformer_forward, self._model.transformer)
        for block in self._model.transformer.encoder.layers:
            block.self_attention.core_attention._orig_forward = block.self_attention.core_attention.forward
            block.self_attention.core_attention.forward = types.MethodType(
                _chatglm2_core_attention_forward, block.self_attention.core_attention
            )

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.transformer.forward = self.original_chatglm_transformer_forward
        for block in self._model.transformer.encoder.layers:
            block.self_attention.core_attention.forward = block.self_attention.core_attention._orig_forward


class GemmaModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()

        # init inv_freq for torchscript tracing
        # https://github.com/huggingface/transformers/blob/ed74d97871468f3a4695ede50abdc0b55717a84d/src/transformers/models/gemma/modeling_gemma.py#L108
        for layer in self._model.model.layers:
            if layer.self_attn.rotary_emb.inv_freq is None:
                rotary_emb = layer.self_attn.rotary_emb
                layer.self_attn.rotary_emb.inv_freq = 1.0 / (
                    rotary_emb.base ** (torch.arange(0, rotary_emb.dim, 2, dtype=torch.int64).float() / rotary_emb.dim)
                )


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
            causal_mask = registered_causal_mask[:, :, key.size(-2) - query.size(-2) : key.size(-2), : key.size(-2)]
            if attention_mask is not None:
                attention_mask = attention_mask.expand(-1, -1, causal_mask.size(2), -1).masked_fill(
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
        for block in self._model.transformer.h:
            block.attn._orig_forward = block.attn.forward
            block.attn.forward = types.MethodType(_qwen_attention_forward, block.attn)

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for block in self._model.transformer.h:
            block.attn.forward = block.attn._orig_forward
        self._model.config.bf16 = self.original_bf16
        self._model.config.fp16 = self.original_fp16


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


class OlmoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def _olmo_model_forward(
    self,
    input_ids: torch.LongTensor,
    input_embeddings: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
    use_cache: bool = False,
    last_logits_only: bool = False,
    output_hidden_states: Optional[bool] = None,
):
    output_hidden_states = output_hidden_states if output_hidden_states is not None else False

    if past_key_values:
        assert len(past_key_values) == self.config.n_layers

    batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
    if past_key_values is None:
        past_length = 0
    else:
        past_length = past_key_values[0][0].size(-2)

    # Get embeddings of input.
    # shape: (batch_size, seq_len, d_model)
    x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

    if not (self.config.alibi or self.config.rope):
        # Get positional embeddings.
        # shape: (1, seq_len)
        pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        # shape: (1, seq_len, d_model)
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = pos_emb + x

    # Add input + positional embeddings and apply dropout.
    # shape: (batch_size, seq_len, d_model)
    x = self.transformer.emb_drop(x)  # type: ignore

    # Transform the attention mask into what the blocks expect.
    if attention_mask is not None:
        # shape: (batch_size, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

    # Merge attention mask with attention bias.
    if attention_bias is not None or attention_mask is not None or self.config.alibi or past_key_values is not None:
        if attention_bias is None and self.config.alibi:
            attention_bias = self.get_causal_attention_bias(
                past_length + seq_len, x.device
            ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
        elif attention_bias is None:
            attention_bias = self.get_causal_attention_bias(past_length + seq_len, x.device)
        elif attention_bias.dtype in (torch.int8, torch.bool):
            attention_bias = attention_bias.to(dtype=torch.float)
            attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

        # Transform to the right shape and data type.
        mask_len = seq_len
        if attention_mask is not None:
            mask_len = attention_mask.shape[-1]
        elif past_key_values is not None:
            mask_len = past_key_values[0][0].shape[-2] + seq_len
        attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

        # Add in the masking bias.
        if attention_mask is not None:
            attention_bias = attention_bias + attention_mask
            # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
            # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
            # it can produce NaNs.
            ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

    # decoder layers
    all_hidden_states = []

    # Apply blocks one-by-one.
    if self.config.block_group_size == 1:
        for block_idx, block in enumerate(self.transformer.blocks):
            if output_hidden_states:
                # add hidden states
                all_hidden_states.append(x)

            layer_past = None if past_key_values is None else past_key_values[block_idx]
            # shape: (batch_size, seq_len, d_model)
            x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
    else:
        for group_idx, block_group in enumerate(self.transformer.block_groups):
            if output_hidden_states:
                # add hidden states
                all_hidden_states.append(x)

            layers_past = (
                None
                if past_key_values is None
                else past_key_values[
                    group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                ]
            )
            x, cache = block_group(x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.extend(cache)

    if last_logits_only:
        # shape: (batch_size, 1, d_model)
        x = x[:, -1, :].unsqueeze(1)

    # Apply final layer norm.
    # shape: (batch_size, seq_len or 1, d_model)
    x = self.transformer.ln_f(x)  # type: ignore
    if output_hidden_states:
        # add final hidden state post-final-layernorm, following HuggingFace's convention
        all_hidden_states.append(x)

    # Get logits.
    # shape: (batch_size, seq_len or 1, vocab_size)
    if self.config.weight_tying:
        logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
    else:
        logits = self.transformer.ff_out(x)  # type: ignore
    if self.config.scale_logits:
        logits.mul_(1 / math.sqrt(self.config.d_model))

    return OlmoOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]


def _olmo_causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def _olmo_get_causal_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
    if hasattr(self, "causal_bias") and self.causal_bias.shape[-1] >= seq_len:
        return self.causal_bias.to(device)
    with torch.autocast(device.type, enabled=False):
        causal_bias = _olmo_causal_attention_bias(seq_len, device)
        self.register_buffer("causal_bias", causal_bias)
    return causal_bias


def _olmo_alibi_attention_bias(seq_len: int, config, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """
    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


def _olmo_get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
    alibi_bias = getattr(self, "alibi_attention_bias", None)
    if alibi_bias is not None and alibi_bias.shape[-1] >= seq_len:
        if alibi_bias.device != device:
            alibi_bias = alibi_bias.to(device)
        return alibi_bias
    with torch.autocast(device.type, enabled=False):
        alibi_bias = _olmo_alibi_attention_bias(seq_len, self.config, device)
        self.register_buffer("alibi_attention_bias", alibi_bias)
    return alibi_bias


def _olmo_get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        hasattr(self, "rope_pos_sin")
        and hasattr(self, "rope_pos_cos")
        and self.rope_pos_sin.shape[-2] >= seq_len
        and self.rope_pos_cos.shape[-2] >= seq_len
    ):
        return self.rope_pos_sin.to(device)[:, :, :seq_len, :], self.rope_pos_sin.to(device)[:, :, :seq_len, :]

    with torch.autocast(device.type, enabled=False):
        dim = self.config.d_model // self.config.n_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
        seq = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        positions = torch.cat((freqs, freqs), dim=-1)
        pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]

    self.register_buffer("rope_pos_sin", pos_sin)
    self.register_buffer("rope_pos_cos", pos_cos)
    return pos_sin, pos_cos


class OLMoModelPatcher(DecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        # model uses custom cache buffers for storing rotary_embeddings and attention biases.
        # these objects are nontracable, replace them with standard torch tensors during export
        self._model.model._orig_forward = self._model.model.forward
        self._model.model._orig_get_alibi_attention_bias = self._model.model.get_alibi_attention_bias
        self._model.model.forward = types.MethodType(_olmo_model_forward, self._model.model)
        self._model.model.get_alibi_attention_bias = types.MethodType(
            _olmo_get_alibi_attention_bias, self._model.model
        )
        self._model.model.get_alibi_attention_bias(self._model.config.max_sequence_length, torch.device("cpu"))
        self._model.model.get_causal_attention_bias = types.MethodType(
            _olmo_get_causal_attention_bias, self._model.model
        )
        self._model.model.get_causal_attention_bias(self._model.config.max_sequence_length, torch.device("cpu"))
        for block in self._model.model.transformer.blocks:
            block.rotary_emb._orig_get_rotary_embedding = block.rotary_emb.get_rotary_embedding
            block.rotary_emb.get_rotary_embedding = types.MethodType(_olmo_get_rotary_embedding, block.rotary_emb)
            block.rotary_emb.get_rotary_embedding(self._model.config.max_sequence_length, torch.device("cpu"))

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._model.model.forward = self._model.model._orig_forward
        self._model.model.get_alibi_attention_bias = self._model.model._orig_get_alibi_attention_bias
        for block in self._model.model.transformer.blocks:
            block.rotary_emb.get_rotary_embedding = block.rotary_emb._orig_get_rotary_embedding
