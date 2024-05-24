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
from transformers.models.llama.modeling_llama import repeat_kv

from optimum.intel.utils.import_utils import is_ipex_version


def matmul_add_add(attn_output, weight, bias=None, residual=None):
    seq_len, bs, _ = attn_output.size()
    if residual is None:
        attn_output = torch.matmul(attn_output, weight)
        if bias is not None:
            attn_output += bias
    else:
        if bias is not None:
            attn_output = torch.ops.torch_ipex.mm_bias_resadd(attn_output, weight, bias, 1.0, residual, 1.0)
        else:
            attn_output = torch.addmm(
                residual.flatten(0, -2),
                attn_output.flatten(0, -2),
                weight,
                beta=1.0,
            )
    attn_output = attn_output.view(seq_len, bs, -1)
    return attn_output

def padding_attn_mask(attn_mask, alignment):
    if attn_mask is None:
        return None
    assert isinstance(
        attn_mask, torch.Tensor
    ), f"attn mask is supposed to be a tensor, instead we got {type(attn_mask)}"
    if attn_mask.device == torch.device("cpu"):
        return attn_mask
    last_dim_size = attn_mask.size(-1)
    aligned_size = (last_dim_size + alignment - 1) // alignment * alignment
    mask_size = [*attn_mask.size()[:-1], aligned_size]
    new_attn_mask = torch.empty(mask_size, dtype=attn_mask.dtype, device=attn_mask.device).fill_(-65504.0)
    new_attn_mask[..., :last_dim_size] = attn_mask
    return new_attn_mask

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

    attention_mask = padding_attn_mask(attention_mask, 8)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    seqlen = hidden_states.size(1)
    head_dim = self.layers[0].self_attn.head_dim
    sin, cos = self.layers[0].self_attn.ipex_rope.get_sin_cos(seqlen, head_dim // 2)
    sin = sin.squeeze()[position_ids].unsqueeze(2)
    cos = cos.squeeze()[position_ids].unsqueeze(2)
    sin_cos = {"sin": sin, "cos": cos}
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None and len(past_key_values) > idx else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **sin_cos,
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
class _IPEXLlamaAttentionRef(nn.Module):
    def __init__(self, module, config, distributed=False) -> None:
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__") or k.startswith("forward"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))
        self.config = config
        self.distributed = distributed
        self.module_device = module.q_proj.weight.device.type
        if self.module_device == "xpu":
            from intel_extension_for_pytorch.transformers.models.xpu.fusions.mha_fusion import _IPEXRopeXPU
            self.ipex_rope = _IPEXRopeXPU(
                module.config.max_position_embeddings,
                module.config.hidden_size // module.config.num_attention_heads,
                module.config.rope_theta,
                module.config.architectures[0],
            )
            self.port_parameters(module)
            torch.xpu.empty_cache()
        else:
            from intel_extension_for_pytorch.llm.modules import IndirectAccessKVCacheAttention, LinearAdd, RotaryEmbedding
            if not self.distributed:
                self.mha_linear_add = LinearAdd(self.o_proj)
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
        if self.module_device == "xpu":
            if self.num_key_value_heads == self.num_heads:
                query = torch.empty(
                    (bsz, seq_len, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
                )
                key = torch.empty(
                    (bsz, seq_len, self.num_heads * self.head_dim),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                value = torch.empty(
                    (bsz, seq_len, self.num_heads * self.head_dim),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
                torch.ops.torch_ipex.mm_qkv_out(
                    hidden_states,
                    self.qkv_proj_weight,
                    self.qkv_proj_bias,
                    query,
                    key,
                    value,
                )
            else:
                query = torch.empty(
                    (bsz, seq_len, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
                )
                key = torch.empty(
                    (bsz, seq_len, self.num_key_value_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
                )
                value = torch.empty(
                    (bsz, seq_len, self.num_key_value_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
                )
                torch.ops.torch_ipex.mm_qkv_group_out(
                    hidden_states, self.qkv_proj_weight, self.qkv_proj_bias, query, key, value
                )
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim)
        key = key.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        
        return query, key, value
    
    def rope(self, query, key, kv_seq_len, position_ids, **kwargs):
        if self.module_device == "xpu":
            sin = kwargs.pop("sin", None)
            cos = kwargs.pop("cos", None)
            self.ipex_rope.apply_embedding(query, sin, cos, self.head_dim // 2, key)
        else:
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
    
    def sdpa(self, query, key, value, past_key_value, attention_mask, use_cache):
        
        if self.module_device == "xpu":
            scale = 1.0 / math.sqrt(self.head_dim)
            is_causal = False
            attn_output = torch.xpu.IpexSDP(
                query, key, value, None, attention_mask, None, scale, 1.0, 0.0, is_causal, False
            )
            attn_weights = None
            past_key_value = (key, value) if use_cache else None
        else:
            if use_cache:
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
            else:
                value_states = value.transpose(1, 2)
                query_states = query.transpose(1, 2)
                key_states = key.transpose(1, 2)

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
        
        return attn_output, attn_weights, past_key_value
    
        
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
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            residual (`torch.Tensor`): residual tensor to the layer of shape `
        """
        bsz, seq_len, _ = hidden_states.size()
        kv_seq_len = seq_len + past_key_value[0].size(-2) if past_key_value is not None else seq_len
        
        query, key, value = self.qkv_gemm(hidden_states)
        query, key = self.rope(query, key, kv_seq_len, position_ids, **kwargs)
        
        if self.module_device == "xpu":
            if past_key_value is not None:
                key = torch.cat([past_key_value[0].transpose(1, 2), key], dim=1)
                value = torch.cat([past_key_value[1].transpose(1, 2), value], dim=1)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        
        attn_output, attn_weights, past_key_value= self.sdpa(query, key, value, past_key_value, attention_mask, use_cache)
        attn_output = attn_output.transpose(1, 2).view(bsz, seq_len, self.hidden_size)
        
        if self.module_device == "xpu":
            attn_output = matmul_add_add(attn_output, self.o_proj_weight, self.o_proj_bias, residual).view([bsz, seq_len, self.hidden_size])
        else:
            if hasattr(self, "mha_linear_add"):
                attn_output = self.mha_linear_add(attn_output, residual)
            else:
                attn_output = self.o_proj(attn_output)
                attn_output = residual + attn_output
            
        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value, attn_weights
    
    
    def port_parameters(self, module):
        self.qkv_proj_bias = None
        self.qkv_proj_weight = None
        if self.num_heads == self.num_key_value_heads:
            q_proj = module.q_proj.weight.transpose(0, 1)
            k_proj = module.k_proj.weight.transpose(0, 1)
            v_proj = module.v_proj.weight.transpose(0, 1)
            self.qkv_proj_weight = torch.stack([q_proj, k_proj, v_proj]).contiguous().view([3, -1, q_proj.shape[-1]])
            module.q_proj.weight.data = self.qkv_proj_weight[0, :, :].transpose(0, 1)
            module.k_proj.weight.data = self.qkv_proj_weight[1, :, :].transpose(0, 1)
            module.v_proj.weight.data = self.qkv_proj_weight[2, :, :].transpose(0, 1)
            if module.q_proj.bias is not None:
                self.qkv_proj_bias = (
                    torch.stack([module.q_proj.bias, module.k_proj.bias, module.v_proj.bias])
                    .contiguous()
                    .view([3, -1])
                )
                module.q_proj.bias.data = self.qkv_proj_bias[0]
                module.k_proj.bias.data = self.qkv_proj_bias[1]
                module.v_proj.bias.data = self.qkv_proj_bias[2]
        else:
            q_proj = module.q_proj.weight.view(self.num_kv_heads, self.num_key_value_groups, self.head_dim, self.embed_dim)
            k_proj = module.k_proj.weight.view(self.num_kv_heads, 1, self.head_dim, self.embed_dim)
            v_proj = module.v_proj.weight.view(self.num_kv_heads, 1, self.head_dim, self.embed_dim)
            self.qkv_proj_weight = torch.cat([q_proj, k_proj, v_proj], dim=1).view(
                [self.num_kv_heads, self.num_key_value_groups + 2, self.head_dim, self.embed_dim]
            )
            module.q_proj.data = self.qkv_proj_weight[:, :self.num_key_value_groups, :, :].view(
                [self.num_kv_heads * self.num_key_value_groups * self.head_dim, self.embed_dim]
            )
            module.k_proj.data = self.qkv_proj_weight[:, self.num_key_value_groups, :, :].view(
                [self.num_kv_heads * self.head_dim, self.embed_dim]
            )
            module.v_proj.data = self.qkv_proj_weight[:, self.num_key_value_groups + 1, :, :].view(
                [self.num_kv_heads * self.head_dim, self.embed_dim]
            )
            if module.q_proj.bias is not None:
                q_bias = module.q_proj.bias.view(self.num_kv_heads, self.num_key_value_groups, self.head_dim)
                k_bias = module.k_proj.bias.view(self.num_kv_heads, 1, self.head_dim)
                v_bias = module.v_proj.bias.view(self.num_kv_heads, 1, self.head_dim)
                self.qkv_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(
                    [self.num_kv_heads, self.num_key_value_groups + 2, self.head_dim]
                )
                module.q_proj.bias.data = self.qkv_proj_bias[:, :self.num_key_value_groups, self.head_dim].view(-1)
                module.k_proj.bias.data = self.qkv_proj_bias[:, self.num_key_value_groups, self.head_dim].view(-1)
                module.v_proj.bias.data = self.qkv_proj_bias[:, self.num_key_value_groups + 1, self.head_dim].view(-1)

        self.o_proj_weight = module.o_proj.weight.transpose(0, 1).contiguous()
        module.o_proj.weight.data = self.o_proj_weight.transpose(0, 1)
        self.o_proj_bias = module.o_proj.bias


class _IPEXLlamaMLPRef(nn.Module):
    def __init__(self, module, config, distributed=False) -> None:
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__") or k.startswith("forward"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))
        self.config = config
        self.distributed = distributed
        self.module_device = module.gate_proj.weight.device.type
        if self.module_device == "xpu":
            self.port_parameter(module)
            torch.xpu.empty_cache()
        else:
            from intel_extension_for_pytorch.llm.modules import Linear2SiluMul, LinearAdd
            if not self.distributed:
                self.mlp_linear_add = LinearAdd(module.down_proj)
                del self.__dict__["_modules"]["down_proj"]
            self.linear_silu_mul = Linear2SiluMul(module.gate_proj, module.up_proj)
            del self.__dict__["_modules"]["gate_proj"]
            del self.__dict__["_modules"]["up_proj"]

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None, **kwargs):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        if self.module_device == "xpu":
            up = torch.ops.torch_ipex.mm_silu(hidden_states, self.gate_proj_weight)
            hidden_states = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_proj_weight, up)
            hidden_states = matmul_add_add(hidden_states, self.down_proj_weight, self.down_proj_bias, residual)
        else:
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

    def port_parameter(self, module):
        self.up_proj_weight = module.up_proj.weight.transpose(0, 1).contiguous()
        module.up_proj.weight.data = self.up_proj_weight.transpose(0, 1)
        self.gate_proj_weight = module.gate_proj.weight.transpose(0, 1).contiguous()
        module.gate_proj.weight.data = self.gate_proj_weight.transpose(0, 1)
        self.down_proj_weight = module.down_proj.weight.transpose(0, 1).contiguous()
        module.down_proj.weight.data = self.down_proj_weight.transpose(0, 1)
        self.up_proj_bias = module.up_proj.bias
        self.gate_proj_bias = module.gate_proj.bias
        self.down_proj_bias = module.down_proj.bias

# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L694
class _IPEXLlamaDecoderLayerRef(nn.Module):
    def __init__(self, module, config, distributed=False):
        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__") or k.startswith("forward"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))
        self.distributed = distributed
        self.self_attn = _IPEXLlamaAttentionRef(module.self_attn, config, distributed)
        self.mlp = _IPEXLlamaMLPRef(module.mlp, config, distributed)
        self.module_device = module.mlp.gate_proj.weight.device.type
        from intel_extension_for_pytorch.llm.modules import RMSNorm
        if self.module_device == "xpu":
            self.input_layernorm = RMSNorm(
                self.input_layernorm.weight, self.input_layernorm.variance_epsilon
            )
            self.post_attention_layernorm = RMSNorm(
                self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon
            )
        else:
            self.input_layernorm = RMSNorm(
                self.hidden_size, self.input_layernorm.variance_epsilon, self.input_layernorm.weight 
            )
            self.post_attention_layernorm = RMSNorm(
                self.hidden_size, self.post_attention_layernorm.variance_epsilon, self.post_attention_layernorm.weight 
            )
            
        
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
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, self_attn_weights = self.self_attn(
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
