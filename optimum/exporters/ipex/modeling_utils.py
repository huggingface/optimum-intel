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

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from optimum.intel.utils.import_utils import is_ipex_version
from optimum.intel.utils.modeling_utils import _setattr_from_module


_IPEX_MINIMUM_VERSION_FOR_PATCHING = "2.3.0"


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L83
def _llama_layer_norm_forward(self, hidden_states):
    return torch.ops.torch_ipex.rmsnorm(hidden_states, self.weight, self.variance_epsilon)


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L1130
def _llama_model_forward(
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
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L321
class _IPEXLlamaAttention(nn.Module):
    def __init__(self, module, config) -> None:
        if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
            raise ImportError(
                f"Only ipex version > {_IPEX_MINIMUM_VERSION_FOR_PATCHING} supports IndirectAccessKVCacheAttention, LinearAdd, RotaryEmbedding"
            )
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        from intel_extension_for_pytorch.llm.modules import IndirectAccessKVCacheAttention, LinearAdd, RotaryEmbedding

        if module.o_proj.__class__.__name__ not in ["LinearAllreduce"]:
            self.mha_linear_add = LinearAdd(module.o_proj)
            del self.__dict__["_modules"]["o_proj"]
        self.ipex_scale_dot_product = IndirectAccessKVCacheAttention(
            text_max_length=module.config.max_position_embeddings
        )
        self.ipex_rope = RotaryEmbedding(
            module.config.max_position_embeddings,
            module.config.hidden_size // module.config.num_attention_heads,
            module.config.rope_theta,
            module.config.architectures[0],
        )

    def qkv_gemm(self, hidden_states):
        bsz, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim)
        key = key.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        return query, key, value

    def rope(self, query, key, kv_seq_len, position_ids, use_cache):
        if use_cache:
            key = self.ipex_rope(
                key,
                position_ids,
                self.num_key_value_heads,
                self.head_dim,
                self.head_dim // 2,
                self.head_dim,
                kv_seq_len,
            )
            query = self.ipex_rope(
                query,
                position_ids,
                self.num_heads,
                self.head_dim,
                self.head_dim // 2,
                self.head_dim,
                kv_seq_len,
            )
        return query, key

    def sdpa_with_cache(self, query, key, value, past_key_value, attention_mask, position_ids):
        # This ipex op pre-allocates buffers for past_key_values and use beam index history
        # which to decide which beam should be used to make attention scale dot more efficient.
        (attn_output, attn_weights, past_key_value) = self.ipex_scale_dot_product(
            query,
            key,
            value,
            math.sqrt(self.head_dim),
            past_key_value,
            None,
            attention_mask,
        )
        return attn_output, past_key_value, attn_weights

    # Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L341
    def sdpa_without_cache(self, query, key, value, past_key_value, attention_mask, position_ids):
        value_states = value.transpose(1, 2)
        query_states = query.transpose(1, 2)
        key_states = key.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = None
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = torch.tensor(attn_weights) + torch.tensor(attention_mask)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, past_key_value, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                Attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
                this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
                the complete sequence length.
            residual (`torch.Tensor`): residual tensor to the layer of shape (batch, seq_len, embed_dim)`
        """
        bsz, seq_len, _ = hidden_states.size()
        kv_seq_len = seq_len + past_key_value[0].size(-2) if past_key_value is not None else seq_len

        query, key, value = self.qkv_gemm(hidden_states)
        query, key = self.rope(query, key, kv_seq_len, position_ids, use_cache)

        sdpa = self.sdpa_with_cache if use_cache else self.sdpa_without_cache
        attn_output, past_key_value, attn_weights = sdpa(
            query, key, value, past_key_value, attention_mask, position_ids
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)

        if hasattr(self, "mha_linear_add"):
            attn_output = self.mha_linear_add(attn_output, residual)
        else:
            attn_output = self.o_proj(attn_output)
            attn_output = residual + attn_output

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L186
class _IPEXLlamaMLP(nn.Module):
    def __init__(self, module, config) -> None:
        if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
            raise ImportError(
                f"Only ipex version > {_IPEX_MINIMUM_VERSION_FOR_PATCHING} supports Linear2SiluMul, LinearAdd"
            )
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        from intel_extension_for_pytorch.llm.modules import Linear2SiluMul, LinearAdd

        # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
        if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
            self.mlp_linear_add = LinearAdd(module.down_proj)
            del self.__dict__["_modules"]["down_proj"]
        self.linear_silu_mul = Linear2SiluMul(module.gate_proj, module.up_proj)
        del self.__dict__["_modules"]["gate_proj"]
        del self.__dict__["_modules"]["up_proj"]

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None, **kwargs):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            residual (`torch.Tensor`): residual tensor to the layer of shape (batch, seq_len, embed_dim)`
        """
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


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L694
class _IPEXLlamaDecoderLayer(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.self_attn = _IPEXLlamaAttention(module.self_attn, config)
        self.mlp = _IPEXLlamaMLP(module.mlp, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                Attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=None,
            residual=residual,
            **kwargs,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual, **kwargs)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
