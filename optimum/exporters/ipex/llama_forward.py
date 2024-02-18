from typing import List, Optional, Tuple, Union
import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    concat_qkv = None
    if hasattr(self, "concat_qkv") and self.concat_qkv is not None:
        concat_qkv = self.concat_qkv(hidden_states)
    else:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

    kv_seq_len = (
        q_len + past_key_value[0].size(-2) if past_key_value is not None else q_len
    )

    if concat_qkv is not None and type(concat_qkv) is not tuple:
        query, key, value = self.ipex_rope(
            concat_qkv,
            position_ids,
            self.num_heads,
            self.head_dim,
            self.head_dim // 2,
            self.head_dim,
            kv_seq_len,
            self.concat_qkv._num_concats,
        )
    else:
        if concat_qkv is not None:
            query, key, value = concat_qkv
        query = query.view(bsz, q_len, self.num_heads, self.head_dim)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
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

    if use_cache:
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
        kv_seq_len = key_states.shape[-2]

        past_key_value = None
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = torch.tensor(attn_weights) + torch.tensor(attention_mask)
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value