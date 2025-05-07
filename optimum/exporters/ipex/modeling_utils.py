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
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

from optimum.intel.utils.import_utils import is_ipex_version, is_torch_version
from optimum.intel.utils.modeling_utils import _setattr_from_module

from .cache_utils import IPEXPagedCache


logger = logging.getLogger(__name__)

_IPEX_MINIMUM_VERSION_FOR_PATCHING = "2.6.0"
_accelerate_added_attributes = ["to", "xpu"]


if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
    logger.warning(
        f"Please upgrade the IPEX version to at least {_IPEX_MINIMUM_VERSION_FOR_PATCHING} if you want to patch the model."
    )
else:
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.llm.functional import varlen_attention
    from intel_extension_for_pytorch.llm.modules import (
        Linear2SiluMul,
        LinearAdd,
        LinearAddAdd,
        LinearGelu,
        LinearNewGelu,
        PagedAttention,
        RMSNorm,
        RotaryEmbedding,
    )

    device_type = "xpu" if ipex._C._has_xpu() else "cpu"
    # Assign device type earlier to void recompile in ipex.
    PagedAttention.runtime_ops.device_type = device_type
    RMSNorm.runtime_ops.device_type = device_type
    RotaryEmbedding.runtime_ops.device_type = device_type


# Adapted from https://github.com/huggingface/accelerate/blob/v1.2.1/src/accelerate/hooks.py#L183
def _remove_hooks_for_ipex(module, recurse):
    if hasattr(module, "_hf_hook"):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

    if hasattr(module, "_old_forward"):
        # Overriding a GraphModuleImpl forward freezes the forward call and later modifications on the graph will fail.
        # Reference: https://pytorch.slack.com/archives/C3PDTEV8E/p1705929610405409
        if "GraphModuleImpl" in str(type(module)):
            module.__class__.forward = module.__class__.forward.__get__(module)
        else:
            module.forward = module.__class__.forward.__get__(module)
        delattr(module, "_old_forward")

    # Remove accelerate added warning hooks from dispatch_model
    for attr in _accelerate_added_attributes:
        module.__dict__.pop(attr, None)

    if recurse:
        for child in module.children():
            _remove_hooks_for_ipex(child, recurse)

    return module


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L83
def _ipex_rms_layer_norm_forward(self, hidden_states):
    return RMSNorm.apply_function(hidden_states, self.weight, self.variance_epsilon)


# Adapted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/falcon/modeling_falcon.py#L1161
# For passing kwargs, we can remove it when falcon model support passing kwargs to self.transformer.
def _falcon_for_causal_lm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

    logits_to_keep (`int` or `torch.Tensor`, *optional*):
        If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
        `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
        token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
        If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
        This is useful when using packed tensor format (single dimension for batch and sequence length).
    """

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = transformer_outputs[0]

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    lm_logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(
            lm_logits,
            labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gpt2/modeling_gpt2.py#L1036
# For passing kwargs, we can remove it when gpt2 model support passing kwargs to self.transformer.
def _gpt2_lm_head_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )
    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.transformer.first_device)
        hidden_states = hidden_states.to(self.lm_head.weight.device)

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Flatten the tokens
        loss = self.loss_function(
            lm_logits,
            labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=transformer_outputs.cross_attentions,
    )


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

    max_input_lens = self.config.max_input_lens
    past_key_values_length = max_input_lens - seq_length

    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(past_key_values_length, max_input_lens, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    index = kwargs.pop("index", None)
    cos = position_embeddings[0]
    sin = position_embeddings[1]

    hidden_states_copy = hidden_states
    hidden_states = (hidden_states.view(-1, hidden_states.shape[-1])).index_select(0, index)
    cos = (cos.reshape(-1, cos.shape[-1])).index_select(0, index)
    sin = (sin.reshape(-1, sin.shape[-1])).index_select(0, index)
    position_embeddings = (cos.unsqueeze(1), sin.unsqueeze(1))

    if past_key_values is None:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(input_ids.shape[0], input_ids.shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

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
            past_key_values_length=past_key_values_length,
            max_input_lens=self.config.max_input_lens,
            query_max_len=seq_length,
            **kwargs,
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
    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy
    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.46.1/src/transformers/models/falcon/modeling_falcon.py#L945
def _falcon_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    max_input_lens = self.config.max_input_lens
    batch_size, seq_length, _ = inputs_embeds.shape
    past_key_values_length = max_input_lens - seq_length
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if cache_position is None:
        cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_attention_heads x N x N
    # head_mask has shape n_layer x batch x num_attention_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    index = kwargs.pop("index", None)
    cos = position_embeddings[0]
    sin = position_embeddings[1]

    hidden_states_copy = hidden_states
    hidden_states = (hidden_states.view(-1, hidden_states.shape[-1])).index_select(0, index)
    cos = (cos.reshape(-1, cos.shape[-1])).index_select(0, index)
    sin = (sin.reshape(-1, sin.shape[-1])).index_select(0, index)
    position_embeddings = (cos.unsqueeze(1), sin.unsqueeze(1))

    if past_key_values is None:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(input_ids.shape[0], input_ids.shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

    next_decoder_cache = None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    for i, block in enumerate(self.h):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = block(
            hidden_states,
            layer_past=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask[i],
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            past_key_values_length=past_key_values_length,
            max_input_lens=self.config.max_input_lens,
            query_max_len=seq_length,
            **kwargs,
        )

        hidden_states = outputs[0]
        if use_cache is True:
            next_decoder_cache = outputs[1]

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy

    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def _gpt2_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    max_input_lens = self.config.max_input_lens
    seq_length = input_ids.shape[-1]
    past_key_values_length = max_input_lens - seq_length
    if position_ids is None:
        position_ids = torch.arange(
            past_key_values_length, input_shape[-1] + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    batch_size, seq_length, _ = inputs_embeds.shape
    position_embeddings = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeddings

    encoder_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    index = kwargs.pop("index", None)

    hidden_states_copy = hidden_states
    hidden_states = (hidden_states.view(-1, hidden_states.shape[-1])).index_select(0, index)

    if past_key_values is None:
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask=attention_mask,
            input_shape=(input_ids.shape[0], input_ids.shape[-1]),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

    presents = None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, block in enumerate(self.h):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = block(
            hidden_states,
            layer_past=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask[i],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            past_key_values_length=past_key_values_length,
            max_input_lens=self.config.max_input_lens,
            query_max_len=seq_length,
            **kwargs,
        )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = outputs[1]

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

    hidden_states = self.ln_f(hidden_states)
    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy

    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/qwen2/modeling_qwen2.py#L499
def _qwen2_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # avoid multi inputs
    kwargs.pop("max_input_lens", None)
    max_input_lens = self.config.max_input_lens
    past_key_values_length = max_input_lens - seq_length
    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    index = kwargs.pop("index", None)
    cos = position_embeddings[0]
    sin = position_embeddings[1]

    hidden_states_copy = hidden_states
    hidden_states = (hidden_states.view(-1, hidden_states.shape[-1])).index_select(0, index)
    cos = (cos.reshape(-1, cos.shape[-1])).index_select(0, index)
    sin = (sin.reshape(-1, sin.shape[-1])).index_select(0, index)
    position_embeddings = (cos.unsqueeze(1), sin.unsqueeze(1))

    if past_key_values is None:
        attention_mask = causal_mask

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            past_key_values_length=past_key_values_length,
            max_input_lens=max_input_lens,
            query_max_len=seq_length,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy
    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()


# Adapted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/mistral/modeling_mistral.py#L459
def _mistral_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # avoid multi inputs
    kwargs.pop("max_input_lens", None)
    max_input_lens = self.config.max_input_lens
    past_key_values_length = max_input_lens - seq_length
    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + inputs_embeds.shape[1], device=device
        )

    if position_ids is None:
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).repeat_interleave(input_ids.shape[0], 0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    index = kwargs.pop("index", None)
    cos = position_embeddings[0]
    sin = position_embeddings[1]
    hidden_states_copy = hidden_states
    hidden_states = (hidden_states.view(-1, hidden_states.shape[-1])).index_select(0, index)
    cos = (cos.reshape(-1, cos.shape[-1])).index_select(0, index)
    sin = (sin.reshape(-1, sin.shape[-1])).index_select(0, index)
    position_embeddings = (cos.unsqueeze(1), sin.unsqueeze(1))
    # TODO: remove this WA after IPEX 2.7
    if device.type == "xpu":
        cos = cos.reshape(-1, cos.shape[-1])
        sin = sin.reshape(-1, sin.shape[-1])
        position_embeddings = (cos.unsqueeze(1), sin.unsqueeze(1))
    if past_key_values is None:
        attention_mask = causal_mask
    # part of the code that was modified above

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            past_key_values_length=past_key_values_length,
            max_input_lens=max_input_lens,
            query_max_len=seq_length,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if hidden_states.shape[0] != batch_size * seq_length:
        (hidden_states_copy.view(-1, hidden_states.shape[-1]))[attention_mask.view(-1) != 0] = hidden_states
        hidden_states = hidden_states_copy
    hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()


class _IPEXAttention(nn.Module):
    def __init__(self, module, device, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = device
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_groups = self.num_attention_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=self.module_device
        ).repeat_interleave(self.num_groups)
        self.use_sdpa = False

    def qkv_gemm(self, hidden_states):
        raise NotImplementedError("Need to implement in specific model class")

    def rope(self, query, key, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)
        RotaryEmbedding.apply_function(
            query, key, position_embeddings[1], position_embeddings[0], query.size(-1), True
        )
        return query, key

    def postprocess_attention_output(self, attn_output):
        if self.use_sdpa:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(-1, attn_output.shape[-2] * attn_output.shape[-1])
        return attn_output

    # Maybe removed after torch 2.6 released
    def has_flash_attn(self):
        if self.module_device.type == "cpu":
            return is_torch_version(">", "2.4.99")
        elif self.module_device.type == "xpu":
            return is_torch_version(">", "2.5.99")

    def attention_interface(
        self,
        query,
        key_cache,
        value_cache,
        key,
        value,
        past_key_value,
        attention_mask,
        input_lens,
        past_key_values_length,
        seq_len_tensor,
        query_len_tensor,
        max_input_lens,
        query_max_len,
    ):
        if past_key_value is None:
            n_rep = query.shape[1] // key.shape[1]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query.reshape(input_lens.shape[0], max_input_lens, -1, query.shape[-1]).transpose(1, 2),
                key.reshape(input_lens.shape[0], max_input_lens, -1, key.shape[-1])
                .transpose(1, 2)
                .repeat_interleave(n_rep, 1),
                value.reshape(input_lens.shape[0], max_input_lens, -1, value.shape[-1])
                .transpose(1, 2)
                .repeat_interleave(n_rep, 1),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=True,
            )
            self.use_sdpa = True
        elif self.has_flash_attn():
            attn_output = torch.empty_like(query)
            PagedAttention.flash_attn_varlen_func(
                attn_output,
                query.contiguous() if query.device.type == "xpu" else query,
                key_cache,
                value_cache,
                query_len_tensor,
                seq_len_tensor,
                query_max_len,
                max_input_lens,
                1.0 / math.sqrt(self.head_dim),
                True,
                past_key_value.block_tables,
                None,
            )
        elif past_key_values_length == 0:
            # prefill, remove padding
            attn_output = torch.empty_like(query)
            varlen_attention(
                query.contiguous() if query.device.type == "xpu" else query,
                key.contiguous() if key.device.type == "xpu" else key,
                value.contiguous() if value.device.type == "xpu" else value,
                attn_output,
                seq_len_tensor,
                seq_len_tensor,
                max_input_lens,
                max_input_lens,
                0.0,
                1.0 / math.sqrt(self.head_dim),
                False,
                True,
                False,
                None,
            )
        else:
            # decode
            attn_output = torch.empty_like(query)
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
                max_input_lens,
                None,
            )

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[IPEXPagedCache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None and kwargs.get("layer_past", None) is not None:
            past_key_value = kwargs.pop("layer_past", None)
        input_lens = kwargs.pop("input_lens", None)
        seq_len_tensor = kwargs.pop("seq_len_tensor", None)
        query_len_tensor = kwargs.pop("query_len_tensor", None)
        max_input_lens = kwargs.pop("max_input_lens", 0)
        query_max_len = kwargs.pop("query_max_len", 0)
        past_key_values_length = kwargs.pop("past_key_values_length", 0)
        query, key, value = self.qkv_gemm(hidden_states)
        query, key = self.rope(query, key, **kwargs)

        key_cache, value_cache = None, None
        if past_key_value is not None:
            key_cache, value_cache = past_key_value.update(key, value, self.layer_idx)

        attn_output = self.attention_interface(
            query,
            key_cache,
            value_cache,
            key,
            value,
            past_key_value,
            attention_mask,
            input_lens,
            past_key_values_length,
            seq_len_tensor,
            query_len_tensor,
            max_input_lens,
            query_max_len,
        )

        attn_output = self.postprocess_attention_output(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value, attn_weights


class _IPEXLlamaAttention(_IPEXAttention):
    def __init__(self, module, device, config) -> None:
        super().__init__(module, device, config)
        if getattr(config, "quantization_config", None) is None:
            concat_weight = torch.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]).contiguous()
            bias_list = [bias for bias in [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias] if bias is not None]
            use_bias = bias_list != []
            self.concat_qkv = nn.Linear(concat_weight.shape[1], concat_weight.shape[0], bias=use_bias)
            self.concat_qkv.weight = nn.Parameter(concat_weight)
            if use_bias:
                concat_bias = torch.concat(bias_list, 0).contiguous()
                self.concat_qkv.bias = nn.Parameter(concat_bias)
            self.q_slice = self.q_proj.weight.shape[0]
            self.k_slice = self.q_slice + self.k_proj.weight.shape[0]
            self.v_slice = self.k_slice + self.v_proj.weight.shape[0]

            if not config.compile and module.o_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mha_linear_add = LinearAdd(module.o_proj)

    def qkv_gemm(self, hidden_states):
        if hasattr(self, "concat_qkv"):
            qkv_out = self.concat_qkv(hidden_states)
            query = qkv_out[:, : self.q_slice].view(-1, self.num_attention_heads, self.head_dim)
            key = qkv_out[:, self.q_slice : self.k_slice].view(-1, self.num_key_value_heads, self.head_dim)
            value = qkv_out[:, self.k_slice :].view(-1, self.num_key_value_heads, self.head_dim)
        else:
            query = self.q_proj(hidden_states).view(-1, self.num_attention_heads, self.head_dim)
            key = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
            value = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        return query, key, value


class _IPEXFalconAttention(_IPEXAttention):
    def __init__(self, module, device, config):
        self.num_key_value_heads = config.num_key_value_heads
        super().__init__(module, device, config)
        self.q_slice = self.head_dim * config.num_kv_heads
        self.k_slice = self.q_slice + self.head_dim
        self.v_slice = self.k_slice + self.head_dim

    def qkv_gemm(self, hidden_states):
        qkv_out = self.query_key_value(hidden_states)
        if self.new_decoder_architecture:
            qkv_out = qkv_out.view(
                qkv_out.shape[0], -1, self.num_attention_heads // self.num_kv_heads + 2, self.head_dim
            )
            query = qkv_out[:, :, :-2, :].flatten(1, 2)
            key = qkv_out[:, :, [-2], :].flatten(1, 2)
            value = qkv_out[:, :, [-1], :].flatten(1, 2)
        else:
            query = qkv_out[:, : self.q_slice].view(-1, self.num_attention_heads, self.head_dim)
            key = qkv_out[:, self.q_slice : self.k_slice].view(-1, self.num_key_value_heads, self.head_dim)
            value = qkv_out[:, self.k_slice :].view(-1, self.num_key_value_heads, self.head_dim)
        return query, key, value


class _IPEXGPT2Attention(_IPEXAttention):
    def __init__(self, module, device, config) -> None:
        super().__init__(module, device, config)
        _setattr_from_module(self, module)
        if not config.compile and getattr(config, "quantization_config", None) is None:
            self.c_attn_linear = nn.Linear(self.c_attn.weight.shape[0], self.c_attn.weight.shape[1])
            self.c_attn_linear.weight = nn.Parameter(self.c_attn.weight.t())
            self.c_attn_linear.bias = self.c_attn.bias
            self.c_proj_linear = nn.Linear(self.c_proj.weight.shape[0], self.c_proj.weight.shape[1])
            self.c_proj_linear.weight = nn.Parameter(self.c_proj.weight.t())
            self.c_proj_linear.bias = self.c_proj.bias
            if self.c_proj_linear not in ["LinearAllreduce"]:
                self.linear_add = LinearAdd(self.c_proj_linear)

    def qkv_gemm(self, hidden_states):
        if hasattr(self, "c_attn_linear"):
            query, key, value = self.c_attn_linear(hidden_states).split(self.split_size, dim=-1)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=-1)
        query = query.view(-1, self.num_attention_heads, self.head_dim)
        key = key.view(-1, self.num_attention_heads, self.head_dim)
        value = value.view(-1, self.num_attention_heads, self.head_dim)
        return query, key, value

    def rope(self, query, key, *args, **kwargs):
        return query, key

    def postprocess_attention_output(self, attn_output):
        if self.use_sdpa:
            attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(-1, attn_output.shape[-2] * attn_output.shape[-1])
        if not hasattr(self, "linear_add"):
            attn_output = self.c_proj(attn_output)
        return attn_output


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L186
class _IPEXLlamaMLP(nn.Module):
    def __init__(self, module, device, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = device

        if not config.compile and getattr(config, "quantization_config", None) is None:
            # LinearAllreduce cannot use fused op LinearAdd
            if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mlp_linear_add = LinearAdd(module.down_proj)
            if isinstance(self.act_fn, nn.SiLU):
                self.linear_silu_mul = Linear2SiluMul(module.gate_proj, module.up_proj)

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
    def __init__(self, module, device, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = device
        if not config.compile and getattr(config, "quantization_config", None) is None:
            # LinearAllreduce cannot use fused op LinearAdd
            self.linear_gelu = LinearGelu(module.dense_h_to_4h)

            if module.dense_4h_to_h.__class__.__name__ not in ["LinearAllreduce"]:
                self.linear_add_add = LinearAddAdd(module.dense_4h_to_h)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor = None,
        residual: torch.Tensor = None,
        **kwargs,
    ):
        if hasattr(self, "linear_gelu"):
            mlp_hidden_states = self.linear_gelu(hidden_states)
        else:
            mlp_hidden_states = self.act(self.dense_h_to_4h(hidden_states))

        if hasattr(self, "linear_add_add"):
            output = self.linear_add_add(mlp_hidden_states, attention_output, residual)
        else:
            mlp_output = self.dense_4h_to_h(mlp_hidden_states)
            output = mlp_output + attention_output + residual

        return output


class _IPEXGPT2MLP(nn.Module):
    def __init__(self, module, device, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = device

        if not config.compile and getattr(config, "quantization_config", None) is None:
            self.c_fc_linear = nn.Linear(self.c_fc.weight.shape[0], self.c_fc.weight.shape[1])
            self.c_fc_linear.weight = nn.Parameter(self.c_fc.weight.t())
            self.c_fc_linear.bias = self.c_fc.bias
            self.c_proj_linear = nn.Linear(self.c_proj.weight.shape[0], self.c_proj.weight.shape[1])
            self.c_proj_linear.weight = nn.Parameter(self.c_proj.weight.t())
            self.c_proj_linear.bias = self.c_proj.bias
            if self.module_device.type == "cpu":
                self.linear_new_gelu = LinearNewGelu(self.c_fc_linear)

            if self.c_proj_linear not in ["LinearAllreduce"]:
                self.linear_add = LinearAdd(self.c_proj_linear)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        if hasattr(self, "linear_new_gelu"):
            hidden_states = self.linear_new_gelu(hidden_states)
        else:
            hidden_states = self.c_fc(hidden_states)
            hidden_states = self.act(hidden_states)
        if not hasattr(self, "linear_add"):
            hidden_states = self.c_proj(hidden_states)
        return hidden_states


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L694
class _IPEXLlamaDecoderLayer(nn.Module):
    def __init__(self, module, device, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.self_attn = _IPEXLlamaAttention(module.self_attn, device, config)
        self.mlp = _IPEXLlamaMLP(module.mlp, device, config)
        if getattr(config, "quantization_config", None):
            _remove_hooks_for_ipex(self, True)

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
    def __init__(self, module, device, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.self_attention = _IPEXFalconAttention(module.self_attention, device, config)
        self.mlp = _IPEXFalconMLP(module.mlp, device, config)
        if getattr(config, "quantization_config", None):
            _remove_hooks_for_ipex(self, True)

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


class _IPEXGPT2Block(nn.Module):
    def __init__(self, module, device, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.attn = _IPEXGPT2Attention(module.attn, device, config)
        self.mlp = _IPEXGPT2MLP(module.mlp, device, config)
        if getattr(config, "quantization_config", None):
            _remove_hooks_for_ipex(self, True)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        if hasattr(self.attn, "linear_add"):
            hidden_states = self.attn.linear_add(attn_output, residual)
        else:
            hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        if hasattr(self.mlp, "linear_add"):
            hidden_states = self.mlp.linear_add(feed_forward_hidden_states, residual)
        else:
            hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


# Currently can just apply llama decoder layer.
class _IPEXQwen2DecoderLayer(_IPEXLlamaDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _IPEXMistralDecoderLayer(_IPEXLlamaDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Adapted from https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L524
class _IPEXIntermediate(nn.Module):
    def __init__(self, module, device, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.module_device = device

        if not config.compile and getattr(config, "quantization_config", None) is None:
            self.linear_gelu = LinearGelu(module.dense)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "linear_gelu"):
            hidden_states = self.linear_gelu(hidden_states)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
