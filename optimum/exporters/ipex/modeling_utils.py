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

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions

from optimum.intel.utils.import_utils import is_ipex_version, is_torch_version
from optimum.intel.utils.modeling_utils import _setattr_from_module

from .cache_utils import IPEXPagedCache


logger = logging.getLogger(__name__)

_IPEX_MINIMUM_VERSION_FOR_PATCHING = "2.4.0"


if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
    logger.warning(
        f"Please upgrade the IPEX version to at least {_IPEX_MINIMUM_VERSION_FOR_PATCHING} if you want to patch the model."
    )
else:
    from intel_extension_for_pytorch.llm.functional import rms_norm, rotary_embedding, varlen_attention
    from intel_extension_for_pytorch.llm.modules import (
        Linear2SiluMul,
        LinearAdd,
        LinearAddAdd,
        LinearGelu,
        LinearNewGelu,
        PagedAttention,
    )


# TODO: Following XPULinearXXX op classes will be put into ipex after 2.6.0 version
class XPULinear2SiluMul(torch.nn.Module):
    def __init__(
        self,
        gate_proj: torch.nn.Module,
        up_proj: torch.nn.Module,
    ):
        super().__init__()
        self.gate_proj_weight = gate_proj.weight.transpose(0, 1).contiguous()
        self.up_proj_weight = up_proj.weight.transpose(0, 1).contiguous()
        self.gate_proj_bias = gate_proj.bias
        self.up_proj_bias = up_proj.bias

    def forward(
        self,
        hidden_states,
    ):
        up = torch.ops.torch_ipex.mm_silu(hidden_states, self.gate_proj_weight)
        if self.gate_proj_bias is not None:
            up += self.gate_proj_bias
        hidden_states = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_proj_weight, up)
        if self.up_proj_bias is not None:
            hidden_states += self.up_proj_bias
        return hidden_states


class XPULinearGelu(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x):
        return torch.ops.torch_ipex.matmul_gelu(x, self.weight, self.bias, 1.0, "tanh")


class XPULinearAdd(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super().__init__()
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(
        self,
        hidden_states,
        residual,
    ):
        token_len, _ = hidden_states.size()
        if residual is None:
            hidden_states = torch.matmul(hidden_states, self.weight)
            if self.bias is not None:
                hidden_states += self.bias
        else:
            if self.bias is not None:
                hidden_states = torch.ops.torch_ipex.mm_bias_resadd(
                    hidden_states, self.weight, self.bias, 1.0, residual, 1.0
                )
            else:
                hidden_states = torch.addmm(
                    residual.flatten(0, -2),
                    hidden_states.flatten(0, -2),
                    self.weight,
                    beta=1.0,
                )
        hidden_states = hidden_states.view(token_len, -1)
        return hidden_states


class XPUlinearAddAdd(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(self, x, y, z):
        if self.bias is not None:
            x = torch.ops.torch_ipex.mm_bias_resadd(x, self.weight, self.bias, 1.0, y, 1.0)
            x += z
        else:
            x = torch.ops.torch_ipex.mm_bias_resadd(x, self.weight, z, 1.0, y, 1.0)
        return x


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L83
def _ipex_rms_layer_norm_forward(self, hidden_states):
    return rms_norm(hidden_states, self.weight, self.variance_epsilon)


# Adapted from https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L918
def _llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
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

    if past_key_values is not None and not isinstance(past_key_values, IPEXPagedCache):
        raise ValueError("only support IPEXPagedCache input now")

    past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    input_lens = attention_mask.cumsum(-1)[:, -1].to(torch.int32)

    if past_key_values_length == 0 and past_key_values is not None:
        # first token, remove the padding from hidden_states, varlen do not accept attention mask
        hidden_states_copy = hidden_states
        index = attention_mask.view(-1) != 0
        hidden_states = (hidden_states.view(-1, hidden_states.shape[-1]))[index]
        cos = position_embeddings[0]
        sin = position_embeddings[1]
        cos = (cos.reshape(-1, cos.shape[-1]))[index]
        sin = (sin.reshape(-1, sin.shape[-1]))[index]
        position_embeddings = (cos.unsqueeze(1), sin.unsqueeze(1))
    else:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    if past_key_values is None:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(input_ids.shape[0], input_ids.shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            input_lens=input_lens,
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

    next_cache = next_decoder_cache if use_cache else None
    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy
    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/models/falcon/modeling_falcon.py#L945
def _falcon_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    batch_size, seq_length, _ = inputs_embeds.shape

    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    input_lens = attention_mask.cumsum(-1)[:, -1].to(torch.int32)

    if past_key_values_length == 0 and past_key_values is not None:
        # first token, remove the padding from hidden_states, varlen do not accept attention mask
        hidden_states_copy = hidden_states
        index = attention_mask.view(-1) != 0
        hidden_states = (hidden_states.view(-1, hidden_states.shape[-1]))[index]
        cos = position_embeddings[0]
        sin = position_embeddings[1]
        cos = (cos.reshape(-1, cos.shape[-1]))[index]
        sin = (sin.reshape(-1, sin.shape[-1]))[index]
        position_embeddings = (cos.unsqueeze(1), sin.unsqueeze(1))
    else:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    if past_key_values is None:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(input_ids.shape[0], input_ids.shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

    next_decoder_cache = None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    for i, block in enumerate(self.h):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = block(
            hidden_states,
            layer_past=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask[i],
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            input_lens=input_lens,
        )

        hidden_states = outputs[0]
        if use_cache is True:
            next_decoder_cache = outputs[1]

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy

    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def _gpt2_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    batch_size, seq_length, _ = inputs_embeds.shape
    position_embeddings = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeddings

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    input_lens = attention_mask.cumsum(-1)[:, -1].to(torch.int32)

    if past_length == 0 and past_key_values is not None:
        # first token, remove the padding from hidden_states, varlen do not accept attention mask
        hidden_states_copy = hidden_states
        index = attention_mask.view(-1) != 0
        hidden_states = (hidden_states.view(-1, hidden_states.shape[-1]))[index]
    else:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    if past_key_values is None:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(input_ids.shape[0], input_ids.shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )

    presents = None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, block in enumerate(self.h):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = block(
            hidden_states,
            layer_past=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask[i],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            input_lens=input_lens,
        )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = outputs[1]

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

    hidden_states = self.ln_f(hidden_states)
    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy

    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


# To pass input_lens, adapted from https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/gpt2/modeling_gpt2.py#L602
def _gpt2_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        **kwargs,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    if hasattr(self.attn, "linear_add"):
        hidden_states = self.attn.linear_add(attn_output, residual)
    else:
        hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    if hasattr(self.mlp, "linear_add"):
        hidden_states = self.mlp.linear_add(feed_forward_hidden_states, residual)
    else:
        hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


class _IPEXAttention(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = next(module.parameters()).device
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=self.module_device
        ).repeat_interleave(self.num_groups)
        self.use_sdpa = False

    def qkv_gemm(self, hidden_states):
        raise NotImplementedError("Need to implement in specific model class")

    def rope(self, *args, **kwargs):
        raise NotImplementedError("Need to implement in specific model class")

    def postprocess_attention_output(self, attn_output):
        if self.use_sdpa:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(-1, attn_output.shape[-2] * attn_output.shape[-1])
        return attn_output

    # Maybe removed after torch 2.6 released
    def has_flash_attn(self, query):
        if query.device.type == "cpu":
            return is_torch_version(">", "2.4.99")
        elif query.device.type == "xpu":
            return is_torch_version(">", "2.5.99")

    def attention_interface(
        self, query, key_cache, value_cache, key, value, past_key_value, attention_mask, input_lens, past_len
    ):
        if past_key_value is None:
            n_rep = query.shape[1] // key.shape[1]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query.reshape(input_lens.shape[0], input_lens.max().item(), -1, query.shape[-1]).transpose(1, 2),
                key.reshape(input_lens.shape[0], input_lens.max().item(), -1, key.shape[-1])
                .transpose(1, 2)
                .repeat_interleave(n_rep, 1),
                value.reshape(input_lens.shape[0], input_lens.max().item(), -1, value.shape[-1])
                .transpose(1, 2)
                .repeat_interleave(n_rep, 1),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=True,
            )
            self.use_sdpa = True
        elif self.has_flash_attn(query):
            attn_output = torch.empty_like(query)
            seq_len_tensor = torch.cat((input_lens.new_tensor([0]), input_lens.cumsum(-1).int()))
            query_len_tensor = seq_len_tensor if past_len == 0 else torch.arange(seq_len_tensor.shape[0]).int()
            query_max_len = input_lens.max() if past_len == 0 else 1
            PagedAttention.flash_attn_varlen_func(
                attn_output,
                query.contiguous() if query.device.type == "xpu" else query,
                key_cache.contiguous() if key_cache.device.type == "xpu" else key_cache,
                value_cache.contiguous() if value_cache.device.type == "xpu" else value_cache,
                query_len_tensor,
                seq_len_tensor,
                query_max_len,
                input_lens.max(),
                1.0 / math.sqrt(self.head_dim),
                True,
                past_key_value.block_tables,
                None,
            )
        elif past_len == 0:
            # prefill, remove padding
            attn_output = torch.empty_like(query)
            seq_len_tensor = torch.cat((input_lens.new_tensor([0]), input_lens.cumsum(-1).int()))
            varlen_attention(
                query.contiguous() if query.device.type == "xpu" else query,
                key.contiguous() if key.device.type == "xpu" else key,
                value.contiguous() if value.device.type == "xpu" else value,
                attn_output,
                seq_len_tensor,
                seq_len_tensor,
                input_lens.max(),
                input_lens.max(),
                0.0,
                1.0 / math.sqrt(self.head_dim),
                False,
                True,
                False,
                None,
            )
        else:
            # decode
            attn_output = torch.empty_like(query)
            PagedAttention.single_query_cached_kv_attention(
                attn_output,
                query,
                key_cache,
                value_cache,
                self.kv_head_mapping,
                1.0 / math.sqrt(self.head_dim),
                past_key_value.block_tables,
                input_lens,
                past_key_value.block_size,
                input_lens.max(),
                None,
            )

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[IPEXPagedCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None and kwargs.get("layer_past", None) is not None:
            past_key_value = kwargs.pop("layer_past", None)
        input_lens = kwargs.pop("input_lens", None)
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value.get_seq_length()
        query, key, value = self.qkv_gemm(hidden_states)
        query, key = self.rope(query, key, **kwargs)

        key_cache, value_cache = None, None
        if past_key_value is not None:
            key_cache, value_cache = past_key_value.update(key, value, self.layer_idx, attention_mask, input_lens)

        attn_output = self.attention_interface(
            query, key_cache, value_cache, key, value, past_key_value, attention_mask, input_lens, past_len
        )

        attn_output = self.postprocess_attention_output(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value, attn_weights


class _IPEXLlamaAttention(_IPEXAttention):
    def __init__(self, module, config) -> None:
        super().__init__(module, config)
        concat_weight = torch.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]).contiguous()
        bias_list = [bias for bias in [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias] if bias]
        use_bias = bias_list != []
        self.concat_qkv = nn.Linear(concat_weight.shape[1], concat_weight.shape[0], bias=use_bias)
        self.concat_qkv.weight = nn.Parameter(concat_weight)
        if use_bias:
            concat_bias = torch.concat(bias_list, 0).contiguous()
            self.concat_linear.bias = nn.Parameter(concat_bias)
        self.q_slice = self.q_proj.weight.shape[0]
        self.k_slice = self.q_slice + self.k_proj.weight.shape[0]
        self.v_slice = self.k_slice + self.v_proj.weight.shape[0]
        if self.module_device.type == "cpu":
            if module.o_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mha_linear_add = LinearAdd(module.o_proj)

        elif self.module_device.type == "xpu":
            if module.o_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mha_linear_add = XPULinearAdd(module.o_proj)

    def qkv_gemm(self, hidden_states):
        qkv_out = self.concat_qkv(hidden_states)
        query = qkv_out[:, : self.q_slice].view(-1, self.num_heads, self.head_dim)
        key = qkv_out[:, self.q_slice : self.k_slice].view(-1, self.num_key_value_heads, self.head_dim)
        value = qkv_out[:, self.k_slice :].view(-1, self.num_key_value_heads, self.head_dim)

        return query, key, value

    def rope(self, query, key, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)
        rotary_embedding(query, key, position_embeddings[1], position_embeddings[0], query.size(-1), True)
        return query, key


class _IPEXFalconAttention(_IPEXAttention):
    def __init__(self, module, config):
        self.num_key_value_heads = config.num_key_value_heads
        super().__init__(module, config)
        self.q_slice = self.head_dim * config.num_kv_heads
        self.k_slice = self.q_slice + self.head_dim
        self.v_slice = self.k_slice + self.head_dim

    def qkv_gemm(self, hidden_states):
        qkv_out = self.query_key_value(hidden_states)
        if self.new_decoder_architecture:
            qkv_out = qkv_out.view(qkv_out.shape[0], -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv_out[:, :, :-2, :].flatten(1, 2)
            key = qkv_out[:, :, [-2], :].flatten(1, 2)
            value = qkv_out[:, :, [-1], :].flatten(1, 2)
        else:
            query = qkv_out[:, : self.q_slice].view(-1, self.num_heads, self.head_dim)
            key = qkv_out[:, self.q_slice : self.k_slice].view(-1, self.num_key_value_heads, self.head_dim)
            value = qkv_out[:, self.k_slice :].view(-1, self.num_key_value_heads, self.head_dim)
        return query, key, value

    def rope(self, query, key, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)
        rotary_embedding(query, key, position_embeddings[1], position_embeddings[0], query.size(-1), True)
        return query, key


class _IPEXGPT2Attention(_IPEXAttention):
    def __init__(self, module, config) -> None:
        self.num_key_value_heads = config.num_key_value_heads
        super().__init__(module, config)
        _setattr_from_module(self, module)
        self.c_attn_linear = nn.Linear(self.c_attn.weight.shape[0], self.c_attn.weight.shape[1])
        self.c_attn_linear.weight = nn.Parameter(self.c_attn.weight.t())
        self.c_attn_linear.bias = self.c_attn.bias
        self.c_proj_linear = nn.Linear(self.c_proj.weight.shape[0], self.c_proj.weight.shape[1])
        self.c_proj_linear.weight = nn.Parameter(self.c_proj.weight.t())
        self.c_proj_linear.bias = self.c_proj.bias
        if self.module_device.type == "cpu":
            if self.c_proj_linear not in ["LinearAllreduce"]:
                self.linear_add = LinearAdd(self.c_proj_linear)

        elif self.module_device.type == "xpu":
            if self.c_proj_linear not in ["LinearAllreduce"]:
                self.linear_add = XPULinearAdd(self.c_proj_linear)

    def qkv_gemm(self, hidden_states):
        query, key, value = self.c_attn_linear(hidden_states).split(self.split_size, dim=-1)
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_heads, self.head_dim)
        value = value.view(-1, self.num_heads, self.head_dim)
        return query, key, value

    def rope(self, query, key, *args, **kwargs):
        return query, key

    def postprocess_attention_output(self, attn_output):
        if self.use_sdpa:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(-1, attn_output.shape[-2] * attn_output.shape[-1])
        if not hasattr(self, "linear_add"):
            attn_output = self.c_proj(attn_output)
        return attn_output


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L186
class _IPEXLlamaMLP(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = next(module.parameters()).device
        if self.module_device.type == "cpu":
            # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
            if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mlp_linear_add = LinearAdd(module.down_proj)
            self.linear_silu_mul = Linear2SiluMul(module.gate_proj, module.up_proj)
        elif self.module_device.type == "xpu":
            # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
            if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mlp_linear_add = XPULinearAdd(module.down_proj)
            self.linear_silu_mul = XPULinear2SiluMul(module.gate_proj, module.up_proj)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None, **kwargs):
        if hasattr(self, "linear_silu_mul"):
            mlp_gate = self.linear_silu_mul(hidden_states)
            if hasattr(self, "mlp_linear_add"):
                hidden_states = self.mlp_linear_add(mlp_gate, residual)
            else:
                hidden_states = self.down_proj(mlp_gate)
                hidden_states = residual + hidden_states
        else:
            hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
            hidden_states = residual + hidden_states

        return hidden_states


class _IPEXFalconMLP(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
        self.module_device = next(module.parameters()).device
        if self.module_device.type == "cpu":
            self.linear_gelu = LinearGelu(module.dense_h_to_4h)
        elif self.module_device.type == "xpu":
            self.linear_gelu = XPULinearGelu(module.dense_h_to_4h)
        if module.dense_4h_to_h.__class__.__name__ not in ["LinearAllreduce"]:
            if self.module_device.type == "cpu":
                self.linear_add_add = LinearAddAdd(module.dense_4h_to_h)
            elif self.module_device.type == "xpu":
                self.linear_add_add = XPUlinearAddAdd(module.dense_4h_to_h)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor = None,
        residual: torch.Tensor = None,
        **kwargs,
    ):
        mlp_hidden_states = self.linear_gelu(hidden_states)
        if hasattr(self, "linear_add_add"):
            output = self.linear_add_add(mlp_hidden_states, attention_output, residual)
        else:
            mlp_output = self.mlp.dense_4h_to_h(mlp_hidden_states)
            output = mlp_output + attention_output + residual

        return output


class _IPEXGPT2MLP(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = next(module.parameters()).device
        self.c_fc_linear = nn.Linear(self.c_fc.weight.shape[0], self.c_fc.weight.shape[1])
        self.c_fc_linear.weight = nn.Parameter(self.c_fc.weight.t())
        self.c_fc_linear.bias = self.c_fc.bias
        self.c_proj_linear = nn.Linear(self.c_proj.weight.shape[0], self.c_proj.weight.shape[1])
        self.c_proj_linear.weight = nn.Parameter(self.c_proj.weight.t())
        self.c_proj_linear.bias = self.c_proj.bias
        if self.module_device.type == "cpu":
            self.linear_new_gelu = LinearNewGelu(self.c_fc_linear)

        if self.module_device.type == "cpu":
            if self.c_proj_linear not in ["LinearAllreduce"]:
                self.linear_add = LinearAdd(self.c_proj_linear)

        elif self.module_device.type == "xpu":
            if self.c_proj_linear not in ["LinearAllreduce"]:
                self.linear_add = XPULinearAdd(self.c_proj_linear)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        if hasattr(self, "linear_new_gelu"):
            hidden_states = self.linear_new_gelu(hidden_states)
        else:
            hidden_states = self.c_fc(hidden_states)
            hidden_states = self.act(hidden_states)
        if not hasattr(self, "linear_add"):
            hidden_states = self.c_proj(hidden_states)
        return hidden_states


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L694
class _IPEXLlamaDecoderLayer(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.self_attn = _IPEXLlamaAttention(module.self_attn, config)
        self.mlp = _IPEXLlamaMLP(module.mlp, config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Please see the original model's forward to check the parameter
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present, attn_weights = self.self_attn(hidden_states=hidden_states, **kwargs)

        if hasattr(self.self_attn, "mha_linear_add"):
            hidden_states = self.self_attn.mha_linear_add(hidden_states, residual)
        else:
            hidden_states = self.self_attn.o_proj(hidden_states)
            hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual, **kwargs)

        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False):
            outputs += (attn_weights,)
        if kwargs.get("use_cache", False):
            outputs += (present,)

        return outputs


class _IPEXFalconDecoderLayer(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.self_attention = _IPEXFalconAttention(module.self_attention, config)
        self.mlp = _IPEXFalconMLP(module.mlp, config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Please see the original model's forward to check the parameter
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output, present, attn_weights = self.self_attention(hidden_states, **kwargs)
        attn_output = self.self_attention.dense(attn_output)
        hidden_states = self.mlp(hidden_states, attn_output, residual)

        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False):
            outputs += (attn_weights,)
        if kwargs.get("use_cache", False):
            outputs += (present,)

        return outputs


# Adapted from https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L524
class _IPEXIntermediate(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.module_device = next(module.parameters()).device
        if self.module_device.type == "cpu":
            self.linear_gelu = LinearGelu(module.dense)
        elif self.module_device.type == "xpu":
            self.linear_gelu = XPULinearGelu(module.dense)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_gelu(hidden_states)
        return hidden_states
