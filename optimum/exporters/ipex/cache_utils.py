import os
from typing import List, Optional, Tuple

import intel_extension_for_pytorch as ipex
import torch
from intel_extension_for_pytorch.llm.modules import PagedAttention
from transformers import Cache, PretrainedConfig

from optimum.intel.utils.import_utils import is_ipex_version


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
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        default_device = torch.device("xpu") if ipex._C._has_xpu() else torch.device("cpu")
        device = device or default_device
        self.device = device
        self._supports_flash_decoding = (
            is_ipex_version(">", "2.4.99") if device.type == "cpu" else is_ipex_version(">", "2.5.99")
        )
        # Used in `generate` to keep tally of how many tokens the cache has seen

        self._seen_tokens = torch.zeros([max_batch_size], dtype=torch.int32, device=device)
        self.slots = torch.zeros([max_cache_len * max_batch_size], dtype=torch.int32, device=device)
        torch._dynamo.mark_static_address(self._seen_tokens)
        torch._dynamo.mark_static_address(self.slots)
        default_block_size = 16 if max_cache_len <= 64 else 64
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

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        if device.type == "cpu":
            key_cache_shape = (self.num_blocks, self.num_kv_heads, self.block_size, head_size)
            value_cache_shape = (self.num_blocks, self.num_kv_heads, self.block_size, head_size)
        elif device.type == "xpu":
            if self._supports_flash_decoding:
                key_cache_shape = (self.num_blocks, self.block_size, self.num_kv_heads, head_size)
                value_cache_shape = (self.num_blocks, self.block_size, self.num_kv_heads, head_size)
            else:
                key_cache_shape = (self.num_blocks, self.num_kv_heads, head_size, self.block_size, 1)
                value_cache_shape = (self.num_blocks, self.num_kv_heads, head_size, self.block_size)
        for i in range(config.num_hidden_layers):
            new_layer_key_cache = torch.zeros(key_cache_shape, dtype=dtype, device=device)
            new_layer_value_cache = torch.zeros(value_cache_shape, dtype=dtype, device=device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slots: torch.Tensor,
    ):
        # TODO: unify API definition between CPU and XPU in IPEX version > 2.6
        if self.device.type == "xpu" and self._supports_flash_decoding:
            # make a WA here as slots here is padded but XPU does not support slots with length not equal to key length, will fix it in IPEX 2.8
            valid_len = key.shape[0]
            truncated_slots = slots[:valid_len]
            PagedAttention.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                truncated_slots,
            )
        else:
            PagedAttention.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slots,
            )

    # outside the model forward
    def alloc_slot_for_prefill(self, input_lens: torch.Tensor, batch_size: int):
        all_block_indices = []
        all_slot_offsets = []
        num_blocks = (input_lens + self.block_size - 1) // self.block_size
        for i in range(batch_size):
            nb = num_blocks[i]
            scores = self.free_blocks * torch.arange(self.free_blocks.shape[0], 0, -1, device=self.device)
            block_table = torch.topk(scores, nb).indices
            self.block_tables[i][0:nb] = block_table
            self.free_blocks[block_table] = 0
            slots_range = torch.arange(input_lens[i], device=self.device)
            block_indices = slots_range // self.block_size
            slot_offsets = slots_range % self.block_size
            all_block_indices.append(self.block_tables[i][block_indices])
            all_slot_offsets.append(slot_offsets)

        all_block_indices = torch.cat(all_block_indices)
        all_slot_offsets = torch.cat(all_slot_offsets).int()
        # Use inplace op to keep the same memory address, avoid recompile
        self.slots[: all_block_indices.shape[0]].copy_(all_block_indices * self.block_size + all_slot_offsets)

    # outside the model forward
    def alloc_slot_for_decode(self, batch_size: int):
        start_block_idx = self._seen_tokens // self.block_size
        slot_offset_in_block = (self._seen_tokens) % self.block_size
        # Use inplace op to keep the same memory address, avoid recompile
        self.slots.zero_()
        for i in range(batch_size):
            if slot_offset_in_block[i] == 0:
                # need a new block:
                b_idx = start_block_idx[i]
                if self.block_tables[i][b_idx] == -1:
                    # Need a free block. Get indices of free blocks, select the first free block
                    scores = self.free_blocks * torch.arange(self.free_blocks.shape[0], 0, -1, device=self.device)
                    self.block_tables[i][b_idx] = scores.argmax()
                    self.free_blocks[self.block_tables[i][b_idx]] = 0
            self.slots[i] = self.block_tables[i][start_block_idx[i]] * self.block_size + slot_offset_in_block[i]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
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

        self.reshape_and_cache(
            key_states, value_states, self.key_cache[layer_idx], self.value_cache[layer_idx], self.slots
        )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        return self._seen_tokens.max()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        self._seen_tokens.zero_()
        self.block_tables.fill_(-1)
        self.free_blocks.fill_(1)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        origin_table = self.block_tables.clone()
        updated_block_tables = self.block_tables.index_select(0, beam_idx.to(self.device))
        mask = torch.where(self.block_tables == -1, 0, 1)
        num_blocks = mask.sum(-1)
        updated_table = torch.zeros_like(beam_idx)
        for i in range(beam_idx.shape[0]):
            nb = num_blocks[i]
            self.block_tables[i, 0 : nb - 1] = updated_block_tables[i, 0 : nb - 1]
            updated_table[i] = self.block_tables[i][nb - 1]
        for layer_idx in range(self.num_hidden_layers):
            # The updated_table cannot contain the whole block table, otherwise will cause core-dump.
            self.key_cache[layer_idx][updated_table] = self.key_cache[layer_idx].index_select(
                0, updated_table[beam_idx]
            )
            self.value_cache[layer_idx][updated_table] = self.value_cache[layer_idx].index_select(
                0, updated_table[beam_idx]
            )

        free_table = torch.unique((origin_table[origin_table != self.block_tables]).view(-1))
        for i in free_table:
            if not (self.block_tables == i).any():
                self.free_blocks[i] = 1

    def crop(self, maximum_length: int):
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""

        max_seq_len = self._seen_tokens.max()
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

        free_table = torch.unique((origin_table[origin_table != self.block_tables]).view(-1))
        for i in free_table:
            if not (self.block_tables == i).any():
                self.free_blocks[i] = 1
