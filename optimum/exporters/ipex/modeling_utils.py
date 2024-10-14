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
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from optimum.intel.utils.import_utils import is_ipex_version
from optimum.intel.utils.modeling_utils import _setattr_from_module

from .cache_utils import IPEXPagedCache


logger = logging.getLogger(__name__)

_IPEX_MINIMUM_VERSION_FOR_PATCHING = "2.3.0"


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
        PagedAttention,
    )


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
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    if past_key_values_length == 0:
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
    input_lens = attention_mask.cumsum(-1)[:, -1]
    lens_list = input_lens.tolist()
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
            input_lens=input_lens.int(),
            lens_list=lens_list,
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
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states.view(batch_size, -1, hidden_states.shape[-1]),
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _gpt2_block_forward(self, hidden_states, *args, **kwargs):
    attention_mask = kwargs.get("attention_mask", None)
    if attention_mask is not None:
        bsz, seq_len, _ = hidden_states.size()
        layer_past = kwargs.get("layer_past", None)
        past_len = layer_past[0].size(-2) if layer_past is not None else 0
        attention_mask = (1 - attention_mask / torch.finfo(attention_mask.dtype).min).squeeze(1, 2)
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (bsz, seq_len), hidden_states, past_len)
        kwargs["attention_mask"] = attention_mask

    return GPT2Block.forward(self, hidden_states, *args, **kwargs)


class _IPEXAttention(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = next(module.parameters()).device.type
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=self.module_device
        ).repeat_interleave(self.num_groups)

    def qkv_gemm(self, hidden_states):
        raise NotImplementedError("Need to implement in specific model class")

    def rope(self, *args, **kwargs):
        raise NotImplementedError("Need to implement in specific model class")

    def postprocess_attention_output(self, attn_output, bsz, seq_len):
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[IPEXPagedCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        input_lens: Optional[torch.Tensor] = None,
        lens_list: Optional[List] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None and kwargs.get("layer_past", None) is not None:
            past_key_value = kwargs.pop("layer_past", None)
        bsz, seq_len = position_ids.size()
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value.get_seq_length()
        qkv_out = self.qkv_gemm(hidden_states)
        if isinstance(qkv_out, tuple) and len(qkv_out) == 3:
            query, key, value = qkv_out[0], qkv_out[1], qkv_out[2]
            query, key = self.rope(query, key, **kwargs)
        else:
            query, key, value = self.rope(qkv_out, **kwargs)

        if past_key_value is not None:
            key_cache, value_cache = past_key_value.update(
                key, value, self.layer_idx, attention_mask, position_ids, lens_list
            )

        attn_output = torch.empty_like(query)
        if past_len == 0:
            # prefill, remove padding
            seq_len_tensor = torch.cat((input_lens.new_tensor([0]), input_lens.cumsum(-1).int()))
            varlen_attention(
                query.contiguous() if query.device.type == "xpu" else query,
                key.contiguous() if key.device.type == "xpu" else key,
                value.contiguous() if value.device.type == "xpu" else value,
                attn_output,
                seq_len_tensor,
                seq_len_tensor,
                max(input_lens),
                max(input_lens),
                0.0,
                1.0 / math.sqrt(self.head_dim),
                False,
                True,
                False,
                None,
            )
        else:
            # decode
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
                max(input_lens),
                None,
            )

        attn_output = attn_output.reshape(-1, attn_output.shape[-2] * attn_output.shape[-1])
        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value, attn_weights


class _IPEXLlamaAttention(_IPEXAttention):
    def __init__(self, module, config) -> None:
        super().__init__(module, config)
        if self.module_device == "cpu":
            if module.o_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mha_linear_add = LinearAdd(module.o_proj)
                del self.__dict__["_modules"]["o_proj"]

    def qkv_gemm(self, hidden_states):
        query = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        return query, key, value

    def rope(self, query, key, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)
        rotary_embedding(query, key, position_embeddings[1], position_embeddings[0], query.size(-1), True)
        return query, key


class _IPEXFalconAttention(_IPEXAttention):
    def qkv_gemm(self, hidden_states):
        return self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    def rope(self, fused_qkv, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)
        (query, key, value) = self._split_heads(fused_qkv)
        rotary_embedding(query, key, position_embeddings[1], position_embeddings[0], query.size(-1), True)
        return query, key, value


class _IPEXGPT2Attention(_IPEXAttention):
    def __init__(self, module, config) -> None:
        super().__init__(module, config)

    def _split_heads_ipex(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        return tensor.view(new_shape)  # (batch, seq_length, head, head_features)

    def qkv_gemm(self, hidden_states):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads_ipex(query, self.num_heads, self.head_dim)
        key = self._split_heads_ipex(key, self.num_heads, self.head_dim)
        value = self._split_heads_ipex(value, self.num_heads, self.head_dim)
        return query, key, value

    def rope(self, query, key, *args, **kwargs):
        return query, key

    def postprocess_attention_output(self, attn_output, bsz, seq_len):
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L186
class _IPEXLlamaMLP(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = next(module.parameters()).device.type
        if self.module_device == "cpu":
            # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
            if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mlp_linear_add = LinearAdd(module.down_proj)
                del self.__dict__["_modules"]["down_proj"]
            self.linear_silu_mul = Linear2SiluMul(module.gate_proj, module.up_proj)
            del self.__dict__["_modules"]["gate_proj"]
            del self.__dict__["_modules"]["up_proj"]

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
        self.linear_gelu = LinearGelu(module.dense_h_to_4h)
        del self.__dict__["_modules"]["dense_h_to_4h"]
        if module.dense_4h_to_h.__class__.__name__ not in ["LinearAllreduce"]:
            self.linear_add_add = LinearAddAdd(module.dense_4h_to_h)
            del self.__dict__["_modules"]["dense_4h_to_h"]

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
        self.linear_gelu = LinearGelu(module.dense)
        del self.__dict__["_modules"]["dense"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_gelu(hidden_states)
        return hidden_states
