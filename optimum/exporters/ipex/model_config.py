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

from typing import Optional, Tuple

from optimum.exporters.onnx.model_configs import (
    FalconOnnxConfig,
    GPT2OnnxConfig,
    LlamaOnnxConfig,
)
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.utils.input_generators import DummyPastKeyValuesGenerator, DummyTextInputGenerator
from optimum.utils.normalized_config import NormalizedTextConfig


DEFAULT_DUMMY_SHAPES["batch_size"] = 1


class IPEXDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        self.num_key_value_heads = getattr(normalized_config, "num_key_value_heads", 1)
        self.max_position_embeddings = normalized_config.max_position_embeddings

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape_init = (1, self.sequence_length, self.sequence_length, 1)
        shape_beam_idx_tmp = (self.max_position_embeddings, self.batch_size)
        shape_kv = (
            self.max_position_embeddings,
            self.batch_size,
            self.num_key_value_heads,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_int_tensor(shape_init, max_value=1, framework=framework).contiguous(),
                self.random_float_tensor(shape_kv, framework=framework, dtype=float_dtype).contiguous(),
                self.random_float_tensor(shape_kv, framework=framework, dtype=float_dtype).contiguous(),
                self.random_int_tensor(shape_beam_idx_tmp, max_value=1, framework=framework).contiguous(),
            )
            for _ in range(self.num_layers)
        ]


class IPEXDummyTextInputGenerator(DummyTextInputGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, **kwargs)


class LlamaIPEXConfig(LlamaOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (IPEXDummyTextInputGenerator, IPEXDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = IPEXDummyPastKeyValuesGenerator


class FalconIPEXConfig(FalconOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (IPEXDummyTextInputGenerator, IPEXDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = IPEXDummyPastKeyValuesGenerator


class GPT2IPEXConfig(GPT2OnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (IPEXDummyTextInputGenerator, IPEXDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = IPEXDummyPastKeyValuesGenerator


ipex_onnx_config = {"llama": LlamaIPEXConfig, "falcon": FalconIPEXConfig, "gpt2": GPT2IPEXConfig}
