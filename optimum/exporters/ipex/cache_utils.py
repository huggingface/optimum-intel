import os
from typing import List, Optional, Tuple

import torch
from intel_extension_for_pytorch.llm.modules import PagedAttention
from transformers import Cache, PretrainedConfig


class IPEXPagedCache(Cache):
    """
    A PagedCache that grows dynamically as more tokens are generated. everytime it grows block-size memory, vendor could set the pageCache memory layout.
    ipex-xpu:
    ipex-cpu:

    Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from optimum.intel import IPEXModelForCausalLM
        >>> from optimum.exporters.ipex.cache_utils import IPEXPagedCache

        >>> model = IPEXModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", export=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = IPEXPagedCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: int,
        device,
        dtype=None,
        layer_device_map=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        # Used in `generate` to keep tally of how many tokens the cache has seen

        self._seen_tokens = torch.zeros([max_batch_size], dtype=torch.int32, device=device)
        default_block_size = 16 if device.type == "cpu" else 64
        self.block_size = int(os.environ.get("OI_PAGED_ATTN_BLOCK_SIZE", str(default_block_size)))
        self.num_blocks = (max_cache_len // self.block_size + (max_cache_len % self.block_size != 0)) * max_batch_size
        self.block_tables = -1 * torch.ones([self.num_blocks], dtype=torch.int32, device=device).reshape(
            max_batch_size, -1
        )
        self.free_blocks = torch.ones([self.num_blocks], dtype=torch.int32, device=device)
        self.max_cache_len = max_cache_len
        self.num_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        if hasattr(config, "head_dim"):
            head_size = config.head_dim
        else:
            head_size = config.hidden_size // config.num_attention_heads
        self.head_size = head_size
        self.max_seq_len = 0

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        if device.type == "cpu":
            key_cache_shape = (self.num_blocks, self.num_kv_heads, self.block_size, head_size)
            value_cache_shape = (self.num_blocks, self.num_kv_heads, self.block_size, head_size)
        elif device.type == "xpu":
            key_cache_shape = (self.num_blocks, self.num_kv_heads, head_size, self.block_size, 1)
            value_cache_shape = (self.num_blocks, self.num_kv_heads, head_size, self.block_size)
        for i in range(config.num_hidden_layers):
            new_layer_key_cache = torch.zeros(key_cache_shape, dtype=dtype, device=device)
            new_layer_value_cache = torch.zeros(value_cache_shape, dtype=dtype, device=device)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update_for_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        input_lens: torch.Tensor,
    ):
        if layer_idx == 0:
            all_block_indices = []
            all_slot_offsets = []
            num_blocks = (input_lens + self.block_size - 1) // self.block_size
            for i in range(batch_size):
                nb = num_blocks[i]
                block_table = self.free_blocks.nonzero().view(-1)[0:nb]
                self.block_tables[i][0:nb] = block_table
                self.free_blocks[block_table] = 0
                slots_range = torch.arange(input_lens[i], device=key_states.device)
                block_indices = slots_range // self.block_size
                slot_offsets = slots_range % self.block_size
                all_block_indices.append(self.block_tables[i][block_indices])
                all_slot_offsets.append(slot_offsets)

            all_block_indices = torch.cat(all_block_indices)
            all_slot_offsets = torch.cat(all_slot_offsets)
            self.slots = all_block_indices * self.block_size + all_slot_offsets
        # Update the cache
        PagedAttention.reshape_and_cache(
            key_states,
            value_states,
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            self.slots,
        )

        # Update the number of seen tokens
        if layer_idx == self.num_hidden_layers - 1:
            self._seen_tokens = self._seen_tokens + input_lens
            self.max_seq_len, _ = self._seen_tokens.max(dim=0)

    def update_for_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        batch_size: int,
    ):
        if layer_idx == 0:
            start_block_idx = self._seen_tokens // self.block_size
            slot_offset_in_block = (self._seen_tokens) % self.block_size
            self.slots = torch.zeros([batch_size], device=key_states.device, dtype=torch.int32)
            for i in range(batch_size):
                if slot_offset_in_block[i] == 0:
                    # need a new block:
                    b_idx = start_block_idx[i]
                    if self.block_tables[i][b_idx] == -1:
                        # need a free block
                        self.block_tables[i][b_idx] = self.free_blocks.nonzero().view(-1)[0:1]
                        self.free_blocks[self.block_tables[i][b_idx]] = 0
                self.slots[i] = self.block_tables[i][start_block_idx[i]] * self.block_size + slot_offset_in_block[i]
        # Update the cache
        PagedAttention.reshape_and_cache(
            key_states,
            value_states,
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            self.slots,
        )

        # Update the number of seen tokens
        if layer_idx == self.num_hidden_layers - 1:
            self._seen_tokens = self._seen_tokens + 1
            self.max_seq_len = self.max_seq_len + 1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor,
        input_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
        Return:
            A tuple containing the updated key and value states.
        """

        batch_size = input_lens.shape[-1]
        if self.get_seq_length() == 0:
            # prefill
            self.update_for_prefill(key_states, value_states, layer_idx, batch_size, input_lens)
        else:
            # decode
            self.update_for_decode(key_states, value_states, layer_idx, batch_size)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        return self.max_seq_len

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        self._seen_tokens = torch.zeros([self.max_batch_size], dtype=torch.int32, device=self.block_tables.device)
        self.block_tables.fill_(-1)
        self.free_blocks = torch.ones([self.num_blocks], dtype=torch.int32, device=self.block_tables.device)
        self.max_seq_len = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.block_tables.device
        origin_table = self.block_tables.clone()
        updated_block_tables = self.block_tables.index_select(0, beam_idx.to(device))
        mask = self.block_tables.masked_fill(self.block_tables != -1, 1).masked_fill(self.block_tables == -1, 0)
        num_blocks = mask.cumsum(-1)[:, -1]
        updated_table = torch.zeros_like(beam_idx)
        for i in range(beam_idx.shape[0]):
            nb = num_blocks[i]
            self.block_tables[i, 0 : nb - 1] = updated_block_tables[i, 0 : nb - 1]
            updated_table[i] = self.block_tables[i][nb - 1]
        for layer_idx in range(self.num_hidden_layers):
            self.key_cache[layer_idx][updated_table] = self.key_cache[layer_idx][updated_table[beam_idx]]
            self.value_cache[layer_idx][updated_table] = self.value_cache[layer_idx][updated_table[beam_idx]]
        free_table = torch.unique((origin_table[origin_table != self.block_tables]).view(-1))
        for i in free_table:
            if not (self.block_tables == i).any():
                self.free_blocks[i] = 1

    def crop(self, maximum_length: int):
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""

        max_seq_len = self.get_seq_length()
        if maximum_length < 0:
            maximum_length = max_seq_len - abs(maximum_length)

        if max_seq_len <= maximum_length:
            return
        origin_table = self.block_tables.clone()
        for bs in range(self._seen_tokens.shape[0]):
            new_tokens = self._seen_tokens[bs] + maximum_length - max_seq_len
            num_blocks = (new_tokens + self.block_size - 1) // self.block_size
            self.block_tables[bs, num_blocks:] = -1
            self._seen_tokens[bs] = new_tokens
        self.max_seq_len, _ = self._seen_tokens.max(dim=0)
        free_table = torch.unique((origin_table[origin_table != self.block_tables]).view(-1))
        for i in free_table:
            if not (self.block_tables == i).any():
                self.free_blocks[i] = 1
