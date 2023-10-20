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

import types
from typing import Tuple

import torch
from transformers.modeling_utils import PreTrainedModel


# Modified from transformers.models.bloom.modeling_bloom._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    device: torch.device,
    past_key_values_length: int,
    dtype: torch.dtype = torch.bool,
) -> torch.BoolTensor:
    """
    Make causal mask used for bi-directional self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.zeros((target_length, target_length + past_key_values_length), dtype=dtype, device=device)
    seq_ids = torch.arange(target_length, device=device)

    mask[:, past_key_values_length:] = (
        (seq_ids[:, None] < seq_ids[None, :]) * torch.finfo(dtype).min
        if torch.is_floating_point(mask)
        else seq_ids[:, None] < seq_ids[None, :]
    )

    return mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)


# Modified from transformers.models..bloom.modeling_bloom._prepare_attn_mask
def _prepare_attn_mask(
    attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
) -> torch.BoolTensor:
    from transformers.models.bloom.modeling_bloom import _expand_mask

    # create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape

    combined_attention_mask = _make_causal_mask(
        input_shape, device=device, past_key_values_length=past_key_values_length
    )
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]_prepare_decoder_attention_mask
    expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask


# Modified from transformers.models.llama.modeling_llama._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    from transformers.models.llama.modeling_llama import _expand_mask

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    combined_attention_mask = _make_causal_mask(
        input_shape,
        device=inputs_embeds.device,
        past_key_values_length=past_key_values_length,
        dtype=inputs_embeds.dtype,
    )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


@torch.jit.script_if_tracing
def _chatglm2_get_context_layer(query_layer: torch.Tensor, key_layer: torch.Tensor, value_layer: torch.Tensor):
    mask = torch.zeros((query_layer.shape[-2], key_layer.shape[-2]), dtype=query_layer.dtype)
    if query_layer.shape[2] == key_layer.shape[2]:
        tmp_mask = torch.ones((query_layer.shape[-2], key_layer.shape[-2]), dtype=torch.bool).triu(diagonal=1)
        mask.masked_fill_(tmp_mask, float("-inf"))

    context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=mask)
    return context_layer


def _core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
    query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
    if attention_mask is None:
        context_layer = _chatglm2_get_context_layer(query_layer, key_layer, value_layer)
    else:
        attention_mask = ~attention_mask
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
    context_layer = context_layer.permute(2, 0, 1, 3)
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.reshape(*new_context_layer_shape)

    return context_layer


def _patch_chatglm_core_attention_forward(model: "PreTrainedModel"):
    for block in model.transformer.encoder.layers:
        block.self_attention.core_attention.forward = types.MethodType(
            _core_attention_forward, block.self_attention.core_attention
        )


def patch_decoder_attention_mask(model: "PreTrainedModel"):
    """
    Apply patch on decoder with past model forward to resolve first inference based on model architecture

    Args:
        model (PretrainedModel): The model to patch.

    Returns:
        model with applied patch
    """
    if model.config.model_type in {"bloom", "mpt"}:
        model.transformer._prepare_attn_mask = _prepare_attn_mask
    elif model.config.model_type == "llama":
        model.model._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    elif model.config.model_type in {"blenderbot-small", "blenderbot", "opt", "pegasus", "bart"}:
        model.model.decoder._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    elif model.config.model_type == "chatglm":
        _patch_chatglm_core_attention_forward(model)

    return model
