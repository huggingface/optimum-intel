import torch
import torch.nn as nn
from typing import Optional, Tuple
import intel_extension_for_pytorch as ipex



class _IPEXLlamaAttention(nn.Module):
    def __init__(self, module, config, distributed=False) -> None:
        self.module = module
        self.config = config
        self.distributed = distributed

    def preprocess_for_optimize(self, hidden_states, layer_past, **kwargs):
        pass

    def qkv_gemm(self, hidden_states, **kwargs):
        pass

    def rope(self, query, key, value, position_ids, layer_past, **kwargs):
        pass

    def get_present(self, query, key, value, use_cache, **kwargs):
        pass

    def sdpa(self, query, key, value, attention_mask, past_key_value, **kwargs):
        pass

    def out_proj(self, hidden_states, residual, **kwargs):
        pass

    def post_process_for_optimize(self):
        pass

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

        self.preprocess_for_optimize(hidden_states, past_key_value, kwargs)

        query, key, value = self.qkv_gemm(hidden_states, kwargs)

        key, value = self.rope(key, value, position_ids, past_key_value, kwargs)

        present = self.get_present(query, key, value, use_cache)

        attn_output, attn_weight = self.sdpa(query, key, value, attention_mask, past_key_value, kwargs)

        attn_output = self.out_proj(attn_output, residual)

        self.post_process_for_optimize()
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight, )
        else:
            outputs += (None, )
        return outputs

class _IPEXLlamaMLP(nn.Module):
    def __init__(self, module, config, distributed=False) -> None:
        self.module = module
        self.config = config
        self.distributed = distributed

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        **kwargs
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        pass


class _IPEXLlamaDecoderLayer(nn.Module):
    def __init__(self, module, config, distributed=False) -> None:
        super().__init__()
        self.layer_idx = module.self_attn.layer_idx
        self.attn = _IPEXLlamaAttention(module.self_attn, config, distributed)
        self.mlp = _IPEXLlamaMLP(module.mlp, config, distributed)
        self.input_layernorm = ipex.llm.modules.RMSNorm(module.input_layernorm.weight, module.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = ipex.llm.modules.RMSNorm(module.post_attention_layernorm.weight, module.post_attention_layernorm.variance_epsilon)



    def preprocess_for_optimize(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        postion_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attention: Optional[bool] = True,
        use_cache: Optional[bool] = False,
        **kwargs
    ):
        return hidden_states, attention_mask, postion_ids, past_key_value

    def postprocess_for_optimize(
        self,
        hidden_states,
        output_attention,
        use_cache,
        self_attn_weight,
        present_key_value,
        **kwargs
    ):
        outputs = (hidden_states,)
        if output_attention:
            outputs += (self_attn_weight,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


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
        outputs = self.preprocess_for_optimize(
            hidden_states, 
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            kwargs
        )
        (hidden_states, attention_mask, position_ids, past_key_value) = outputs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weight, present_key_value = self.attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            residual,
            kwargs,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual, kwargs)

        outputs = self.postprocess_for_optimize(
            hidden_states,
            output_attentions,
            use_cache,
            self_attn_weight,
            present_key_value,
            kwargs
        )

        return outputs

    

