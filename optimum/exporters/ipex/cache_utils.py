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
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.batch_size = max_batch_size
        self.kv_cache = []

        self._seen_tokens = max_batch_size * [
            0
        ]  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.block_size = 16
        self.num_blocks = (max_cache_len // self.block_size + (max_cache_len % self.block_size != 0)) * max_batch_size
        self.block_tables = -1 * torch.ones([self.num_blocks], dtype=torch.int32, device=device).reshape(
            max_batch_size, -1
        )
        self.free_blocks = list(range(0, self.num_blocks))
        self.max_cache_len = max_cache_len
        self.num_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        if hasattr(config, "head_dim"):
            head_size = config.head_dim
        else:
            head_size = config.hidden_size // config.num_attention_heads
        self.head_size = head_size

        if device.type == "cpu":
            self.kv_cache = [
                (
                    torch.empty(
                        (self.num_blocks, self.num_kv_heads, self.block_size, head_size),
                        dtype=dtype,
                        device=device,
                    ),
                    torch.empty(
                        (self.num_blocks, self.num_kv_heads, self.block_size, head_size),
                        dtype=dtype,
                        device=device,
                    ),
                )
                for _ in range(self.num_hidden_layers)
            ]
        elif device.type == "xpu":
            self.kv_cache = [
                (
                    torch.empty(
                        (self.num_blocks, self.num_kv_heads, head_size, self.block_size, 1),
                        dtype=dtype,
                        device=device,
                    ),
                    torch.empty(
                        (self.num_blocks, self.num_kv_heads, head_size, self.block_size),
                        dtype=dtype,
                        device=device,
                    ),
                )
                for _ in range(self.num_hidden_layers)
            ]

    def update_for_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        batch_size: int,
        length_list: Optional[List],
    ):
        all_block_indices = []
        all_slot_offsets = []
        for i in range(batch_size):
            num_blocks = (length_list[i] + self.block_size - 1) // self.block_size
            for b_idx in range(num_blocks):
                if self.block_tables[i][b_idx] == -1:
                    # need a free block
                    self.block_tables[i][b_idx] = self.free_blocks.pop(0)

            slots_range = torch.arange(length_list[i], device=key_states.device)
            block_indices = slots_range // self.block_size
            slot_offsets = slots_range % self.block_size
            all_block_indices.append(self.block_tables[i][block_indices])
            all_slot_offsets.append(slot_offsets)

        all_block_indices = torch.cat(all_block_indices)
        all_slot_offsets = torch.cat(all_slot_offsets)
        slots_tensor = all_block_indices * self.block_size + all_slot_offsets
        # Update the cache
        PagedAttention.reshape_and_cache(
            key_states,
            value_states,
            self.kv_cache[layer_idx][0],
            self.kv_cache[layer_idx][1],
            slots_tensor,
        )

        # Update the number of seen tokens
        if layer_idx == self.num_hidden_layers - 1:
            for i in range(batch_size):
                self._seen_tokens[i] += length_list[i]

    def update_for_decode(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        batch_size: int,
    ):
        slots = []
        for i in range(batch_size):
            start_block_idx = self._seen_tokens[i] // self.block_size
            num_blocks = (self._seen_tokens[i] + self.block_size) // self.block_size
            for b_idx in range(start_block_idx, num_blocks):
                if self.block_tables[i][b_idx] == -1:
                    # need a free block
                    self.block_tables[i][b_idx] = self.free_blocks.pop(0)
            block_idx = (self._seen_tokens[i]) // self.block_size
            slot_offset_in_block = (self._seen_tokens[i]) % self.block_size
            slots.append(self.block_tables[i][block_idx].item() * self.block_size + slot_offset_in_block)

        # Update the cache
        PagedAttention.reshape_and_cache(
            key_states,
            value_states,
            self.kv_cache[layer_idx][0],
            self.kv_cache[layer_idx][1],
            torch.tensor(slots, device=key_states.device),
        )

        # Update the number of seen tokens
        if layer_idx == self.num_hidden_layers - 1:
            for i in range(batch_size):
                self._seen_tokens[i] += 1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        length_list: Optional[List],
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

        batch_size = position_ids.shape[0]
        if self.get_seq_length() == 0:
            # prefill
            self.update_for_prefill(key_states, value_states, layer_idx, batch_size, length_list)
        else:
            # decode
            self.update_for_decode(key_states, value_states, layer_idx, batch_size)

        return self.kv_cache[layer_idx][0], self.kv_cache[layer_idx][1]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        return max(self._seen_tokens)

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        self._seen_tokens = self.max_batch_size * [0]
        self.block_tables.fill_(-1)
        self.free_blocks = list(range(0, self.num_blocks))

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.block_tables.device
        origin_table = self.block_tables.clone()
        updated_block_tables = self.block_tables.index_select(0, beam_idx.to(device))
        mask = self.block_tables.masked_fill(self.block_tables != -1, 1).masked_fill(self.block_tables == -1, 0)
        num_blocks = mask.cumsum(-1)[:, -1]
        updated_table = []
        for i in range(beam_idx.shape[0]):
            self.block_tables[i, 0 : num_blocks[i] - 1] = updated_block_tables[i, 0 : num_blocks[i] - 1]
            updated_table.append(self.block_tables[i : i + 1, num_blocks[i] - 1 : num_blocks[i]])
        updated_table = torch.cat(tuple(updated_table), dim=0)
        for layer_idx in range(len(self.kv_cache)):
            self.kv_cache[layer_idx][0][updated_table] = self.kv_cache[layer_idx][0][updated_table[beam_idx]]
            self.kv_cache[layer_idx][1][updated_table] = self.kv_cache[layer_idx][1][updated_table[beam_idx]]

        free_table = origin_table[origin_table != self.block_tables]
        for i in range(free_table.shape[0]):
            if free_table[i] not in self.free_blocks and not torch.any(self.block_tables.view(-1) == free_table[i]):
                self.free_blocks.insert(0, free_table[i].item())

    def crop(self, maximum_length: int):
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""

        max_seq_len = self.get_seq_length()
        if maximum_length < 0:
            maximum_length = max_seq_len - abs(maximum_length)

        if max_seq_len <= maximum_length:
            return
        origin_table = self.block_tables.clone()
        for bs in range(len(self._seen_tokens)):
            new_tokens = self._seen_tokens[bs] + maximum_length - max_seq_len
            num_blocks = (new_tokens + self.block_size - 1) // self.block_size
            self.block_tables[bs, num_blocks:] = -1
            self._seen_tokens[bs] = new_tokens
        free_table = origin_table[origin_table != self.block_tables]
        for i in range(free_table.shape[0]):
            if free_table[i] not in self.free_blocks and not torch.any(self.block_tables.view(-1) == free_table[i]):
                self.free_blocks.insert(0, free_table[i].item())
