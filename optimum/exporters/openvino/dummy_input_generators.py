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

from typing import Optional, Tuple

import torch

from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    NormalizedTextConfig,
)


class ChatGLN2DummyTextInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    }

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        input = super().generate(input_name, framework, int_dtype, float_dtype)
        if input_name == "attention_mask":
            input = torch.ones((input.shape[0], input.shape[1] + 1), dtype=input.dtype)
            # input[0] = 0
        if input_name == "position_ids":
            input = torch.range(0, input.shape[1] + 1, dtype=input.dtype).repeat(1, 1)
            # input[0] = 0
        return input


class ChatGLM2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.multi_query_group_num = normalized_config.multi_query_group_num
        self.head_dim = self.hidden_size // self.num_attention_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (
            self.sequence_length,
            self.batch_size,
            self.multi_query_group_num,
            self.head_dim,
        )
        past_value_shape = (
            self.sequence_length,
            self.batch_size,
            self.multi_query_group_num,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]
