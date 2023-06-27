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

from optimum.intel.generation import BaseModelForCausalLM

from .modeling_base import _TOKENIZER_FOR_DOC, INPUTS_DOCSTRING, MODEL_START_DOCSTRING, INCBaseModel


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
    Neural-compressor Model with a causal language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MODEL_START_DOCSTRING,
)
class INCModelForCausalLM(INCBaseModel, BaseModelForCausalLM):
    export_feature = "text-generation"
    auto_model_class = AutoModelForCausalLM

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super(INCBaseModel, self).__init__(
            model=model, config=config, model_save_dir=model_save_dir, use_cache=use_cache, **kwargs
        )
        self.backend = getattr(config, "backend", None)

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
        else:
            return CausalLMOutputWithPast(
                logits=outputs["logits"], past_key_values=outputs["past_key_values"] if self.use_cache else None
            )
