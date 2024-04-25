from typing import Tuple
import torch
import torch.nn as nn
from typing import Optional
import math

import intel_extension_for_pytorch
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.llama import NewIPEXLLAMABlock

from ..modeling_llama import _IPEXLlamaDecoderLayer, _IPEXLlamaAttention, _IPEXLlamaMLP


def matmul_add_add(attn_output, weight, bias=None, residual=None):
    seq_len, bs, _ = attn_output.size()
    if residual is None:
        attn_output = torch.matmul(attn_output, weight)
        if bias is not None:
            attn_output += bias
    else:
        if bias is not None:
            attn_output = torch.ops.torch_ipex.mm_bias_resadd(
                attn_output, weight, bias, 1.0, residual, 1.0
            )
        else:
            attn_output = torch.addmm(
                residual.flatten(0, -2),
                attn_output.flatten(0, -2),
                weight,
                beta=1.0,
            )
    attn_output = attn_output.view(seq_len, bs, -1)
    return attn_output

class _IPEXLlamaAttentionXPU(_IPEXLlamaAttention):
    def __init__(self, module, config, distributed=False, optimized_module=None) -> None:
        super().__init__(module, config, distributed)
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.num_kv_heads = module.num_key_value_heads
        self.embed_dim = module.config.hidden_size
        self.port_parameters(module)
        from intel_extension_for_pytorch.llm.modules import RotaryEmbedding

        self.ipex_rope = RotaryEmbedding(
                    module.config.max_position_embeddings,
                    module.config.hidden_size // module.config.num_attention_heads,
                    module.config.rope_theta,
                    module.config.architectures[0],
                                                        )

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
        # allocate cache and copy past_key_value
        bs, seqlen, _ = hidden_states.size()
        prev_seqlen = 0
        if past_key_value:
            _, _, prev_seqlen, _ = past_key_value[0].size()
        if self.num_kv_heads == self.num_heads:
            query = torch.empty((bs, seqlen, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            key = torch.empty((bs, prev_seqlen + seqlen, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            value = torch.empty((bs, prev_seqlen + seqlen, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            torch.ops.torch_ipex.mm_qkv_out(
                hidden_states, self.qkv_proj_weight, self.qkv_proj_bias, query, key[:, prev_seqlen:, :], value[:, prev_seqlen:, :])
        else:
            query = torch.empty((bs, seqlen, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            key = torch.empty((bs, prev_seqlen + seqlen, self.num_kv_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            value = torch.empty((bs, prev_seqlen + seqlen, self.num_kv_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            torch.ops.torch_ipex.mm_qkv_group_out(
                hidden_states, self.qkv_proj_weight, self.qkv_proj_bias, query, key, value)
        if past_key_value:
            key[:, :prev_seqlen, :] = past_key_value[0].transpose(1, 2).view(bs, prev_seqlen, -1)
            value[:, :prev_seqlen, :] = past_key_value[1].transpose(1, 2).view(bs, prev_seqlen, -1)

        # rope
        #query = query.view([-1, seqlen, self.num_heads, self.head_dim])
        #key = key.view([-1, seqlen, self.num_kv_heads, self.head_dim])
        value = value.view([bs, prev_seqlen + seqlen, self.num_kv_heads, self.head_dim])

        query = self.ipex_rope(
            query,
            position_ids,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
            seqlen,
        )

        key = self.ipex_rope(
            key,
            position_ids,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
            seqlen,
        )

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        present = (key, value) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = torch.xpu.IpexSDP(query.transpose(1,2), key, value, None, attention_mask, None, scale, 1.0, 0.0, True, False)
        # attn_output, attn_weight = torch.nn.functional.scaled_dot_product_attention(query, key, value, attention_mask, dropout_p=0.0, scale=scale)
        attn_output = attn_output.transpose(1, 2).view([bs, seqlen, self.embed_dim])
        attn_output = matmul_add_add(attn_output, self.o_proj_weight, self.o_proj_bias, residual).view([bs, seqlen, self.embed_dim])
        outputs = (attn_output, present)
        if output_attentions:
            raise ValueError("not support output attn_weight")
            # outputs += (attn_weight, )
        else:
            outputs += (None, )
        return outputs


    def port_parameters(self, module):
        self.qkv_proj_bias = None
        self.qkv_proj_weight = None
        if self.num_heads == self.num_kv_heads:
            q_proj = module.q_proj.weight.transpose(0, 1)
            k_proj = module.k_proj.weight.transpose(0, 1)
            v_proj = module.v_proj.weight.transpose(0, 1)
            self.qkv_proj_weight = torch.stack([q_proj, k_proj, v_proj]).contiguous().view([3, -1, q_proj.shape[-1]])
            if module.q_proj.bias is not None:
                self.qkv_proj_bias = torch.stack([module.q_proj.bias, module.k_proj.bias, module.v_proj.bias]).contiguous().view([3, -1])
        else:
            group = self.num_heads // self.num_kv_heads
            q_proj = module.q_proj.weight.view(self.num_kv_heads, group, self.head_dim, self.embed_dim)
            k_proj = module.k_proj.weight.view(self.num_kv_heads, 1, self.head_dim, self.embed_dim)
            v_proj = module.v_proj.weight.view(self.num_kv_heads, 1, self.head_dim, self.embed_dim)
            self.qkv_proj_weight = torch.cat([q_proj, k_proj, v_proj], dim=1).view([self.num_kv_heads, group + 2, self.head_dim, self.embed_dim])
            if module.q_proj.bias is not None:
                q_bias = module.q_proj.bias.view(self.num_kv_heads, group, self.head_dim)
                k_bias = module.k_proj.bias.view(self.num_kv_heads, 1, self.head_dim)
                v_bias = module.v_proj.bias.view(self.num_kv_heads, 1, self.head_dim)
                self.qkv_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view([self.num_kv_heads, group + 2, self.head_dim])
        self.o_proj_weight = module.o_proj.weight
        self.o_proj_bias = module.o_proj.bias



class _IPEXLlamaMLPXPU(_IPEXLlamaMLP):
    def __init__(self, module, config, distributed=False, optimized_module=None) -> None:
        super().__init__(module, config, distributed)
        self.mlp_impl = None
        if optimized_module is not None:
            self.mlp_impl = optimized_module
        self.port_parameter(module)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor = None,
        **kwargs
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        up = torch.ops.torch_ipex.mm_silu(hidden_states, self.gate_proj_weight)
        out = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_proj_weight, up)
        out = matmul_add_add(out, self.down_proj_weight, self.down_proj_bias, residual)
        return out


    def port_parameter(self, module):
        self.up_proj_weight = module.up_proj.weight.transpose(0, 1).contiguous()
        self.gate_proj_weight = module.gate_proj.weight.transpose(0, 1).contiguous()
        self.down_proj_weight = module.down_proj.weight.transpose(0, 1).contiguous()
        self.up_proj_bias = module.up_proj.bias
        self.gate_proj_bias = module.gate_proj.bias
        self.down_proj_bias = module.down_proj.bias



# class _IPEXLlamaDecoderLayerXPU(_IPEXLlamaDecoderLayer):
#     def __init__(self, module, config, distributed=False) -> None:
#         super().__init__(module, config, distributed)
#         self.block_impl = NewIPEXLLAMABlock(module, config)
#         self.attn = _IPEXLlamaAttentionXPU(module.self_attn, config, self.block_impl.attn)
#         self.mlp = _IPEXLlamaMLPXPU(module.mlp, config, self.block_impl.mlp)

#     def preprocess_for_optimize(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attention: Optional[bool] = True,
#         use_cache: Optional[bool] = False,
#         **kwargs
#     ):
#         return self.block_impl.preprocess_for_optimize(
#             hidden_states,
#             attention_mask,
#             position_ids,
#             past_key_value,
#             output_attention,
#             use_cache,
#             **kwargs
#         )



#     def postprocess_for_optimize(self, hidden_states, output_attention, use_cache, self_attn_weight, present_key_value, **kwargs):
#         return self.block_impl.postprocess_for_optimize(
#             hidden_states,
#             output_attention,
#             use_cache,
#             self_attn_weight,
#             present_key_value,
#             **kwargs
#         )

