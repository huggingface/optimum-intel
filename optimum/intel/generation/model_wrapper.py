from typing import Tuple

import torch
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class _ModelGenerationWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        try:
            trace_graph_inputs = {
                "input_ids": kwargs["input_ids"],
                "past_key_values": kwargs["past_key_values"],
                "attention_mask": kwargs["attention_mask"],
            }
            if args:
                trace_graph_inputs = args + tuple(trace_graph_inputs.values())
            if isinstance(trace_graph_inputs, tuple):
                outputs = self._optimized(*trace_graph_inputs)
            elif isinstance(trace_graph_inputs, dict):
                outputs = self._optimized(**trace_graph_inputs)
            fixed_output = CausalLMOutputWithPast(
                loss=None,
                logits=outputs[0],
                past_key_values=outputs[1],
                hidden_states=None,
                attentions=None,
            )
            return fixed_output
        except Exception:
            return self._default(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)
