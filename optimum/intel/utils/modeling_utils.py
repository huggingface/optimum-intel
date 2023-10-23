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

from typing import Tuple

import torch
from transformers.modeling_utils import PreTrainedModel


# from ...utils.modeling_utils import _prepare_decoder_sliding_window_attention_mask


MULTI_QUERY_ATTN_MODELS = {"falcon", "gpt_bigcode"}


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
    # elif model.config.model_type == "mistral":
    #     model.model._prepare_decoder_attention_mask = _prepare_decoder_sliding_window_attention_mask
    elif model.config.model_type in {"blenderbot-small", "blenderbot", "opt", "pegasus", "bart"}:
        model.model.decoder._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    return model
