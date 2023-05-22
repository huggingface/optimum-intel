#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast

from optimum.utils import NormalizedConfigManager

from ..utils.import_utils import is_transformers_version
from .modeling_base import _TOKENIZER_FOR_DOC, INPUTS_DOCSTRING, MODEL_START_DOCSTRING, INCBaseModel


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)


TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> inputs = tokenizer("I love this story because", return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)
    ```
    Example using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    >>> text = "I love this story because"
    >>> gen = gen_pipeline(text)
    ```
"""


@add_start_docstrings(
    """
    Base INCBaseDecoderModel class.
    """,
)
class INCBaseDecoderModel(INCBaseModel):
    main_input_name = "input_ids"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(
            model,
            config,
            model_save_dir=model_save_dir,
            **kwargs,
        )

        self.use_cache = use_cache
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)


@add_start_docstrings(
    """
    Neural-compressor Model with a causal language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MODEL_START_DOCSTRING,
)
class INCModelForCausalLM(INCBaseDecoderModel, GenerationMixin):
    export_feature = "text-generation"
    auto_model_class = AutoModelForCausalLM

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="INCModelForCausalLM",
            checkpoint="EleutherAI/gpt-j-6b",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        nb_pkv = 2
        num_layers = self.normalized_config.num_layers
        first_token = False
        if self.use_cache:
            if past_key_values is None:
                first_token = True
                num_attention_heads = self.normalized_config.num_attention_heads
                hidden_size = self.normalized_config.hidden_size
                d_k = hidden_size // num_attention_heads

                if self.config.model_type != "bloom":
                    if self.backend == "neural_engine":
                        new_shape = [input_ids.shape[0], 0, num_attention_heads, d_k]
                    else:
                        new_shape = [input_ids.shape[0], num_attention_heads, 0, d_k]
                    empty_tensor = torch.empty(size=new_shape)
                    past_key_values = tuple(tuple(empty_tensor for _ in range(nb_pkv)) for _ in range(num_layers))
                    pkv = tuple(empty_tensor for _ in range(nb_pkv))
                else:
                    pkv = ()
                    for nb_pkv in range(nb_pkv):
                        if nb_pkv % 2 == 0:
                            new_shape = [input_ids.shape[0] * num_attention_heads, d_k, 0]
                        else:
                            new_shape = [input_ids.shape[0] * num_attention_heads, 0, d_k]
                        pkv = pkv + (torch.empty(size=new_shape),)
                past_key_values = tuple(tuple(pkv) for _ in range(num_layers))

            inputs["past_key_values"] = past_key_values
        if self.backend == "neural_engine":
            past_key_values = [past_key_values[i][j] for i in range(num_layers) for j in range(nb_pkv)]
            predictions = self.model.inference([input_ids] + past_key_values + [attention_mask])
            for key in predictions:
                predictions[key] = torch.from_numpy(predictions[key])

            torchout = CausalLMOutputWithPast()
            torchout.logits = list(predictions.values())[0]
            torchout.past_key_values = [
                (list(predictions.values())[2 * i + 1], list(predictions.values())[2 * i + 2])
                for i in range(num_layers)
            ]
            outputs = torchout
            if first_token:
                input_bs = input_ids.size()[0]
                seq_len = input_ids.size()[1]
                outputs.logits = outputs.logits.expand(input_bs, seq_len, -1)
                past_key_values = []
                for key, value in outputs.past_key_values:
                    key_dim = key.dim()
                    value_dim = value.dim()
                    key = key.expand(input_bs, -1, -1, -1).contiguous()
                    value = value.expand(input_bs, -1, -1, -1).contiguous()
                    if key_dim == 3:
                        key = key.view(key.size(1) * key.size(0), key.size(2), key.size(3))
                    if value_dim == 3:
                        value = value.view(value.size(1) * value.size(0), value.size(2), value.size(3))
                    past_key_values.append((key, value))
                outputs.past_key_values = tuple(past_key_values)
        else:
            outputs = self.model(**inputs)
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            return CausalLMOutputWithPast(logits=outputs[0], past_key_values=outputs[1] if self.use_cache else None)
        elif isinstance(outputs, CausalLMOutputWithPast):
            return outputs
        else:
            raise ValueError(
                f"output should be a list or an instance of CausalLMOutputWithPast, but got {type(outputs)}"
            )

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = past_key_values or kwargs.get("past", None)

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # `past_key_values` may be in the stardard format (e.g. in contrastive search), converts to bloom's format if needed
        if past_key_values is not None and self.config.model_type == "bloom":
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_cache,
            "position_ids": None,
            "attention_mask": kwargs.get("attention_mask", None),
            "token_type_ids": None,
        }

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.config.model_type == "bloom":
            return self._reorder_cache_bloom(past_key_values, beam_idx)

        # from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    # Copied from transformers.models.bloom.modeling_bloom.BloomForCausalLM._reorder_cache
    def _reorder_cache_bloom(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called for bloom architecture.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        standardized_past = self._convert_to_standard_cache(past_key_values, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device)
            for layer_past in past_key_values
            for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)

    # Copied from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_bloom_cache
    @staticmethod
    def _convert_to_bloom_cache(past_key_value: Tuple[Tuple[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    # Adapted from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_standard_cache
    def _convert_to_standard_cache(
        self, past_key_value: Tuple[Tuple[torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size, num_heads, ...]))
        """
        if self.config.model_type != "bloom":
            return past_key_value

        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
