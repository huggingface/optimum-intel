# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from optimum.utils import (
    DummyAudioInputGenerator,
    DummyPastKeyValuesGenerator,
    DummyTransformerTextInputGenerator,
    NormalizedTextConfig,
    is_transformers_version,
)


class GPTBigCodeDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(self, task: str, normalized_config: NormalizedTextConfig, **kwargs):
        super().__init__(task=task, normalized_config=normalized_config, **kwargs)
        self.multi_query = normalized_config.multi_query

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if is_transformers_version("<", "4.54"):
            if self.multi_query:
                shape = (
                    self.batch_size,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads * 2,
                )
            else:
                shape = (
                    self.batch_size,
                    self.num_attention_heads,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads * 2,
                )
            pkv = [
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype) for _ in range(self.num_layers)
            ]

        else:
            if self.multi_query:
                shape = (
                    self.batch_size,
                    1,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads,
                )
            else:
                shape = (
                    self.batch_size,
                    self.num_attention_heads,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads,
                )
            pkv = [
                (
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]

        return pkv


class DummyMoonshineAudioInputGenerator(DummyAudioInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input_values", "attention_mask")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "input_values":  # raw waveform
            return self.random_float_tensor(
                shape=[self.batch_size, self.sequence_length],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        elif input_name == "attention_mask":  # attention mask
            return self.random_mask_tensor(
                shape=[self.batch_size, self.sequence_length],
                framework=framework,
                dtype=int_dtype,
            )
        else:
            raise ValueError(f"Unsupported input name: {input_name}")


class DummySanaTransforemerTextInputGenerator(DummyTransformerTextInputGenerator):
    SUPPORTED_INPUT_NAMES = ("encoder_hidden_states", "encoder_attention_mask")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "encoder_attention_mask":
            return self.random_mask_tensor(
                shape=[self.batch_size, self.sequence_length],
                framework=framework,
                dtype=int_dtype,
            )
        else:
            return super().generate(
                input_name=input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype
            )
