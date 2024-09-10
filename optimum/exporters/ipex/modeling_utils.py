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
from intel_extension_for_pytorch.llm.functional import rms_norm
from torch import nn
from torch.nn import functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from optimum.intel.utils.import_utils import is_ipex_version
from optimum.intel.utils.modeling_utils import _setattr_from_module


logger = logging.getLogger(__name__)

_IPEX_MINIMUM_VERSION_FOR_PATCHING = "2.3.0"

if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
    logger.warning(
        f"Please upgrade the IPEX version to at least {_IPEX_MINIMUM_VERSION_FOR_PATCHING} if you want to patch the model."
    )
else:
    from intel_extension_for_pytorch.llm.modules import (
        IndirectAccessKVCacheAttention,
        Linear2SiluMul,
        LinearAdd,
        LinearAddAdd,
        LinearGelu,
        RotaryEmbedding,
    )


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


def _llama_layer_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu":
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)
    else:
        # Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L83
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

    attention_mask = padding_attn_mask(attention_mask, 8)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    if hidden_states.device.type == "xpu":
        seqlen = hidden_states.size(1)
        head_dim = self.layers[0].self_attn.head_dim
        sin, cos = self.layers[0].self_attn.ipex_rope.get_sin_cos(seqlen, head_dim // 2)
        sin = sin.squeeze()[position_ids].unsqueeze(2)
        cos = cos.squeeze()[position_ids].unsqueeze(2)
        decoder_layer_kwargs = {"sin": sin, "cos": cos}
    else:
        decoder_layer_kwargs = {}
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
            **decoder_layer_kwargs,
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
            self.ipex_scale_dot_product = IndirectAccessKVCacheAttention(
                text_max_length=config.max_position_embeddings
            )
            if hasattr(config, "rope_theta"):
                self.ipex_rope = RotaryEmbedding(
                    config.max_position_embeddings,
                    config.hidden_size // config.num_attention_heads,
                    config.rope_theta,
                    config.architectures[0],
                )

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
            q_proj = module.q_proj.weight.view(
                self.num_key_value_heads, self.num_key_value_groups, self.head_dim, self.hidden_size
            )
            k_proj = module.k_proj.weight.view(self.num_key_value_heads, 1, self.head_dim, self.hidden_size)
            v_proj = module.v_proj.weight.view(self.num_key_value_heads, 1, self.head_dim, self.hidden_size)
            self.qkv_proj_weight = torch.cat([q_proj, k_proj, v_proj], dim=1).view(
                [self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim, self.hidden_size]
            )
            module.q_proj.data = self.qkv_proj_weight[:, : self.num_key_value_groups, :, :].reshape(
                [self.num_key_value_heads * self.num_key_value_groups * self.head_dim, self.hidden_size]
            )
            module.k_proj.data = self.qkv_proj_weight[:, self.num_key_value_groups, :, :].reshape(
                [self.num_key_value_heads * self.head_dim, self.hidden_size]
            )
            module.v_proj.data = self.qkv_proj_weight[:, self.num_key_value_groups + 1, :, :].reshape(
                [self.num_key_value_heads * self.head_dim, self.hidden_size]
            )
            self.qkv_proj_weight = self.qkv_proj_weight.permute(3, 0, 1, 2).contiguous()
            if module.q_proj.bias is not None:
                q_bias = module.q_proj.bias.view(self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                k_bias = module.k_proj.bias.view(self.num_key_value_heads, 1, self.head_dim)
                v_bias = module.v_proj.bias.view(self.num_key_value_heads, 1, self.head_dim)
                self.qkv_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(
                    [self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim]
                )
                module.q_proj.bias.data = self.qkv_proj_bias[:, : self.num_key_value_groups, self.head_dim].view(-1)
                module.k_proj.bias.data = self.qkv_proj_bias[:, self.num_key_value_groups, self.head_dim].view(-1)
                module.v_proj.bias.data = self.qkv_proj_bias[:, self.num_key_value_groups + 1, self.head_dim].view(-1)
        self.o_proj_weight = module.o_proj.weight.transpose(0, 1).contiguous()
        module.o_proj.weight.data = self.o_proj_weight.transpose(0, 1)
        self.o_proj_bias = module.o_proj.bias

    def qkv_gemm(self, hidden_states):
        raise NotImplementedError("Need to implement in specific model class")

    def rope(self, *args, **kwargs):
        raise NotImplementedError("Need to implement in specific model class")

    def sdpa_with_cache(self, query, key, value, past_key_value, attention_mask, **kwargs):
        # This ipex op pre-allocates buffers for past_key_values and use beam index history
        # which to decide which beam should be used to make attention scale dot more efficient.
        if self.module_device == "xpu":
            scale = 1.0 / math.sqrt(self.head_dim)
            is_causal = False
            attn_output = torch.xpu.IpexSDP(
                query, key, value, None, attention_mask, None, scale, 1.0, 0.0, is_causal, False
            )
            attn_weights = None
            past_key_value = (key, value)
        else:
            (attn_output, attn_weights, past_key_value) = self.ipex_scale_dot_product(
                query,
                key,
                value,
                math.sqrt(self.head_dim),
                past_key_value,
                kwargs.get("head_mask", None),
                attention_mask,
                kwargs.get("alibi", None),
            )
        return attn_output, past_key_value, attn_weights

    def sdpa_without_cache(self, query, key, value, past_key_value, attention_mask, **kwargs):
        raise NotImplementedError("Need to implement in specific model class")

    def prepare_attention_mask_float(self, attention_mask, *args):
        return attention_mask

    def postprocess_attention_output(self, attn_output, bsz, seq_len):
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # For llama inputs: https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L308
        # For falcon inputs: https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/falcon/modeling_falcon.py#L370
        if past_key_value is None and kwargs.get("layer_past", None) is not None:
            past_key_value = kwargs.pop("layer_past", None)
        bsz, seq_len, _ = hidden_states.size()
        past_len = past_key_value[0].size(-2) if past_key_value is not None else 0
        kv_seq_len = seq_len + past_len

        qkv_out = self.qkv_gemm(hidden_states)
        if isinstance(qkv_out, tuple) and len(qkv_out) == 3:
            query, key, value = self.qkv_gemm(hidden_states)
            query, key = self.rope(query, key, kv_seq_len, use_cache, position_ids, **kwargs)
        else:
            query, key, value = self.rope(qkv_out, kv_seq_len, use_cache, past_len=past_len)

        if self.module_device == "xpu":
            if past_key_value is not None:
                key = torch.cat([past_key_value[0].transpose(1, 2), key], dim=1)
                value = torch.cat([past_key_value[1].transpose(1, 2), value], dim=1)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        attention_mask = self.prepare_attention_mask_float(attention_mask, query.dtype)
        sdpa = self.sdpa_with_cache if use_cache else self.sdpa_without_cache
        attn_output, past_key_value, attn_weights = sdpa(
            query,
            key,
            value,
            past_key_value,
            attention_mask,
            position_ids=position_ids,
            head_mask=kwargs.get("head_mask", None),
            alibi=kwargs.get("alibi", None),
        )

        attn_output = self.postprocess_attention_output(attn_output, bsz, seq_len)

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
        bsz, seq_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        return query, key, value

    def rope(self, query, key, kv_seq_len, use_cache, position_ids, **kwargs):
        if self.module_device == "xpu":
            sin = kwargs.pop("sin", None)
            cos = kwargs.pop("cos", None)
            self.ipex_rope.apply_embedding(query, sin, cos, self.head_dim // 2, key)
        else:
            if use_cache:
                args = (self.head_dim, self.head_dim // 2, self.head_dim, kv_seq_len)
                key = self.ipex_rope(key, position_ids, self.num_key_value_heads, *args)
                query = self.ipex_rope(query, position_ids, self.num_heads, *args)
        return query, key

    # Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L341
    def sdpa_without_cache(self, query, key, value, past_key_value, attention_mask, position_ids, **kwargs):
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        cos, sin = self.rotary_emb(value, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        # repeat k/v heads if n_kv_heads < n_heads
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = torch.tensor(attn_weights) + torch.tensor(attention_mask)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, None, attn_weights


class _IPEXFalconAttention(_IPEXAttention):
    def qkv_gemm(self, hidden_states):
        return self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    def rope(self, fused_qkv, seq_len, use_cache, past_len):
        if use_cache:
            query, key, value = self.ipex_rope(
                fused_qkv,
                torch.tensor(past_len),
                self.num_heads,
                self.head_dim,
                self.head_dim // 2,
                self.head_dim,
                seq_len,
                3,
            )
        else:
            (query, key, value) = self._split_heads(fused_qkv)
        return query, key, value

    def prepare_attention_mask_float(self, attention_mask, dtype):
        attention_mask_float = (
            (attention_mask * 1.0).masked_fill(attention_mask.to(torch.bool), float("-1e9")).to(dtype)
        )
        return attention_mask_float

    def sdpa_without_cache(self, query, key, value, past_key_value, attention_mask, **kwargs):
        bs, q_len = query.shape[0], query.shape[1]
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(query, key, value, attention_mask, 0.0, is_causal=False)
        attn_output = attn_output.view(bs, self.num_heads, q_len, self.head_dim)

        return attn_output, None, None


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

    def sdpa_without_cache(self, query, key, value, past_key_value, attention_mask, **kwargs):
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(query, key, value, attention_mask, 0.0, is_causal=True)

        return attn_output, None, None

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
        if self.module_device == "xpu":
            self.port_parameter(module)
            torch.xpu.empty_cache()
        else:
            # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
            if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mlp_linear_add = LinearAdd(module.down_proj)
                del self.__dict__["_modules"]["down_proj"]
            self.linear_silu_mul = Linear2SiluMul(module.gate_proj, module.up_proj)
            del self.__dict__["_modules"]["gate_proj"]
            del self.__dict__["_modules"]["up_proj"]

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None, **kwargs):
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
                    hidden_states = self.down_proj(
                        self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
                    )
                    hidden_states = residual + hidden_states
            else:
                hidden_states = self.down_proj(
                    self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
                )
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
