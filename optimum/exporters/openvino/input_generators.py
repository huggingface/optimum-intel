# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import math
from typing import Optional, Tuple

import torch

from optimum.intel.utils.import_utils import is_diffusers_version
from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    DummyAudioInputGenerator,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyVisionInputGenerator,
    FalconDummyPastKeyValuesGenerator,
    MistralDummyPastKeyValuesGenerator,
    NormalizedTextConfig,
    is_transformers_version,
)
from optimum.utils.input_generators import DTYPE_MAPPER
from optimum.utils.normalized_config import NormalizedConfig, NormalizedVisionConfig


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


class DummyQwen3VLLMInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "visual_pos_masks",
        "deepstack_visual_embeds",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        padding_side: str = "right",
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
            padding_side=padding_side,
            **kwargs,
        )
        self.embed_dim = normalized_config.hidden_size
        self.num_layers = len(self.normalized_config.deepstack_visual_indexes)

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        bool_dtype: str = "bool",
    ):
        if input_name == "deepstack_visual_embeds":
            return self.random_float_tensor(
                [self.num_layers, 2 * self.sequence_length, self.embed_dim], framework=framework, dtype=float_dtype
            )
        if input_name == "visual_pos_masks":
            return self.constant_tensor(
                shape=[self.batch_size, self.sequence_length],
                framework=framework,
                value=1,
                dtype=DTYPE_MAPPER.pt(bool_dtype),
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)


class OVMiniCPM3DummyPastKeyValuesGenerator(MistralDummyPastKeyValuesGenerator):
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
            **kwargs,
        )
        self.v_head_dim = getattr(normalized_config, "v_head_dim", self.hidden_size // self.num_attention_heads)
        self.k_head_dim = normalized_config.qk_nope_head_dim + normalized_config.qk_rope_head_dim

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        v_shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.v_head_dim,
        )
        k_shape = (self.batch_size, self.num_key_value_heads, self.sequence_length, self.k_head_dim)
        return [
            (
                self.random_float_tensor(k_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(v_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


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
        self.head_dim = normalized_config.kv_channels
        self.standart_cache_layout = hasattr(normalized_config, "rope_ratio")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if not self.standart_cache_layout:
            pkv_shape = (
                self.sequence_length,
                self.batch_size,
                self.multi_query_group_num,
                self.head_dim,
            )
        else:
            pkv_shape = (
                self.batch_size,
                self.multi_query_group_num,
                self.sequence_length,
                self.head_dim,
            )
        return [
            (
                self.random_float_tensor(pkv_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(pkv_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class Eagle3DummyGenerator(DummyInputGenerator):
    """
    Dummy input generator for Eagle-3 speculative decoding.

    This generator produces synthetic `hidden_states` tensors that mimic the
    intermediate hidden-state outputs of a *main (target) model*, which are
    required by the Eagle-3 draft model during speculative decoding.
    """

    SUPPORTED_INPUT_NAMES = ("hidden_states",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = normalized_config.hidden_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        # hidden_states is provided as a concatenation of three hidden-layer outputs from the main model
        shape = (
            self.batch_size,
            self.sequence_length,
            self.hidden_size * 3,
        )
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class Eagle3VLMDummyGenerator(DummyInputGenerator):
    """
    Dummy input generator for VLM Eagle-3 speculative decoding.

    Produces `inputs_embeds` (float) and 3D `position_ids` (MRoPE)
    required by VLM Eagle-3 draft models targeting Qwen3-VL.
    """

    SUPPORTED_INPUT_NAMES = ("inputs_embeds", "position_ids")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_size = normalized_config.hidden_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "inputs_embeds":
            shape = (self.batch_size, self.sequence_length, self.hidden_size)
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        if input_name == "position_ids":
            # The rotary embedding is MRoPE (Multimodal RoPE)
            # MRoPE encodes position along three independent axes: temporal, height, and width
            # https://github.com/Tencent/AngelSlim/blob/main/angelslim/compressor/speculative/train/models/draft/llama_eagle3.py#L211
            shape = (3, self.batch_size, self.sequence_length)
            return self.random_int_tensor(shape, max_value=self.sequence_length, framework=framework, dtype=int_dtype)


class QwenDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        self.kv_channels = normalized_config.kv_channels

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (self.batch_size, self.sequence_length, self.num_attention_heads, self.kv_channels)
        past_value_shape = (self.batch_size, self.sequence_length, self.num_attention_heads, self.kv_channels)
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class OVFalconDummyPastKeyValuesGenerator(FalconDummyPastKeyValuesGenerator):
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
            **kwargs,
        )
        if normalized_config.new_decoder_architecture:
            self.num_kv_heads = normalized_config.num_attention_heads
        else:
            self.num_kv_heads = normalized_config.num_kv_heads if not normalized_config.multi_query else 1

        self.head_dim = self.hidden_size // self.num_attention_heads


class AquilaDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
            task,
            normalized_config,
            batch_size,
            sequence_length,
            random_batch_size_range,
            random_sequence_length_range,
            **kwargs,
        )
        self.num_key_value_heads = getattr(
            normalized_config, "num_key_value_heads", normalized_config.num_attention_heads
        )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class Gemma4DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        self.num_key_value_heads = normalized_config.num_key_value_heads
        self.head_dim = normalized_config.head_dim
        self.global_head_dim = getattr(normalized_config.config, "global_head_dim", self.head_dim)
        self.layer_types = normalized_config.config.layer_types
        self.num_kv_shared_layers = normalized_config.config.num_kv_shared_layers
        self.sliding_window = normalized_config.config.sliding_window
        # Full-attention layers use fewer KV heads than sliding-attention layers (e.g. 2 vs 8 for 26B-A4B)
        self.num_global_key_value_heads = (
            getattr(normalized_config.config, "num_global_key_value_heads", None) or self.num_key_value_heads
        )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        # some layers do not produce their own KV-cache, they use the shared KV-cache
        if self.num_kv_shared_layers > 0:
            layer_types = self.layer_types[: -self.num_kv_shared_layers]
        else:
            layer_types = self.layer_types
        past_kv_values = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                shape = (
                    self.batch_size,
                    self.num_key_value_heads,
                    self.sliding_window,
                    self.head_dim,
                )
            else:
                shape = (
                    self.batch_size,
                    self.num_global_key_value_heads,
                    self.sequence_length,
                    self.global_head_dim,
                )
            past_kv_value = (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            past_kv_values.append(past_kv_value)

        return past_kv_values


class DummyGemma4UnifiedVisionInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("pixel_values", "image_position_ids")

    def __init__(self, task, normalized_config, batch_size=DEFAULT_DUMMY_SHAPES["batch_size"], **kwargs):
        super().__init__(task, normalized_config, batch_size, **kwargs)
        self.patch_size = getattr(normalized_config, "patch_size", 16)
        self.pooling_kernel_size = getattr(normalized_config, "pooling_kernel_size", 3)
        self.mm_posemb_size = getattr(normalized_config, "mm_posemb_size", 1120)
        # The gemma4_unified vision embedder is encoder-free and consumes pre-merged patches:
        # each merged patch has model_patch_size = patch_size * pooling_kernel_size pixels per side.
        # The processor pads to max_soft_tokens merged patches, so num_patches == max_soft_tokens.
        max_soft_tokens = getattr(normalized_config, "image_seq_length", None)
        if max_soft_tokens is None:
            max_soft_tokens = getattr(normalized_config, "max_soft_tokens", 280)
        self.num_patches = max_soft_tokens
        self.model_patch_size = self.patch_size * self.pooling_kernel_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "pixel_values":
            # Pre-merged pixel patches: [batch, num_patches, 3 * model_patch_size^2]
            return self.random_float_tensor(
                shape=[self.batch_size, self.num_patches, 3 * self.model_patch_size**2],
                framework=framework,
                dtype=float_dtype,
            )
        if input_name == "image_position_ids":
            # 2D (x, y) patch coordinates. Build a roughly square grid of valid positions
            # bounded by the factorized position embedding table size.
            side = int(math.sqrt(self.num_patches))
            side = max(1, min(side, self.mm_posemb_size - 1))
            dtype = DTYPE_MAPPER.pt(int_dtype)
            grid = torch.stack(
                torch.meshgrid(
                    torch.arange(side, dtype=dtype),
                    torch.arange(side, dtype=dtype),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(1, -1, 2)
            if grid.shape[1] < self.num_patches:
                pad = torch.full((1, self.num_patches - grid.shape[1], 2), -1, dtype=grid.dtype)
                grid = torch.cat([grid, pad], dim=1)
            else:
                grid = grid[:, : self.num_patches, :]
            return grid.expand(self.batch_size, -1, -1).clone()
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DeciDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        self.num_key_value_heads_per_layer = normalized_config.num_key_value_heads_per_layer

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_values = []

        for layer_id in range(self.num_layers):
            shape = (
                self.batch_size,
                self.num_key_value_heads_per_layer[layer_id],
                self.sequence_length,
                self.hidden_size // self.num_attention_heads,
            )
            past_key_values.append(
                (
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                )
            )
        return past_key_values


class DummyLLavaMultiModalProjectorInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ["image_features"]

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task

        self.batch_size = batch_size
        self.hidden_size = normalized_config.hidden_size
        self.num_patches = (normalized_config.image_size // normalized_config.patch_size) ** 2
        self.normalized_config = normalized_config

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        shape = [self.batch_size, self.num_patches, self.hidden_size]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class PooledProjectionsDummyInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ["pooled_projections"]

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.pooled_projection_dim = normalized_config.config.pooled_projection_dim

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size, self.pooled_projection_dim]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class DummyTransformerTimestpsInputGenerator(DummyTimestepInputGenerator):
    SUPPORTED_INPUT_NAMES = ("timestep", "text_embeds", "time_ids", "timestep_cond", "guidance")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["timestep", "guidance"]:
            shape = [self.batch_size]
            return self.random_float_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyUnetVisionInputGenerator(DummyVisionInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name not in ["sample", "latent_sample"]:
            return super().generate(input_name, framework, int_dtype, float_dtype)
        # add height and width discount for enable any resolution generation
        return self.random_float_tensor(
            shape=[self.batch_size, self.num_channels, self.height - 1, self.width - 1],
            framework=framework,
            dtype=float_dtype,
        )


class DummyUnetTimestepInputGenerator(DummyTimestepInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name != "timestep":
            return super().generate(input_name, framework, int_dtype, float_dtype)
        shape = [self.batch_size]
        return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=int_dtype)


class DummySanaTimestepInputGenerator(DummyTimestepInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name != "timestep":
            return super().generate(input_name, framework, int_dtype, float_dtype)
        shape = [self.batch_size]
        return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)


class DummyUnetEncoderInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
            **kwargs,
        )
        if hasattr(normalized_config.config, "model_max_length"):
            self.sequence_length = normalized_config.config.model_max_length


class DummySanaSeq2SeqDecoderTextWithEncMaskInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_outputs",
        "encoder_hidden_states",
        "encoder_attention_mask",
    )


# DummySanaTransformerVisionInputGenerator inherits from DummyUnetVisionInputGenerator (defined above)
class DummySanaTransformerVisionInputGenerator(DummyUnetVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
        "sample",
        "latent_sample",
        "guidance",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"] // 8,
        height: int = DEFAULT_DUMMY_SHAPES["height"] // 8,
        # Reduce img shape by 4 for FLUX to reduce memory usage on conversion
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width=width, height=height, **kwargs)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "guidance":
            return self.random_float_tensor([self.batch_size], framework=framework, dtype=float_dtype)
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyFluxTransformerInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
        "sample",
        "latent_sample",
        "hidden_states",
        "img_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"] // 4,
        height: int = DEFAULT_DUMMY_SHAPES["height"] // 4,
        # Reduce img shape by 4 for FLUX to reduce memory usage on conversion
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        if getattr(normalized_config, "in_channels", None):
            self.num_channels = normalized_config.in_channels // 4

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["hidden_states", "sample"]:
            shape = [self.batch_size, (self.height // 2) * (self.width // 2), self.num_channels * 4]
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        if input_name == "img_ids":
            img_ids_height = self.height // 2
            img_ids_width = self.width // 2
            return self.random_int_tensor(
                (
                    [self.batch_size, img_ids_height * img_ids_width, 3]
                    if is_diffusers_version("<", "0.31.0")
                    else [img_ids_height * img_ids_width, 3]
                ),
                min_value=0,
                max_value=min(img_ids_height, img_ids_width),
                framework=framework,
                dtype=float_dtype,
            )

        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyFluxTextInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_outputs",
        "encoder_hidden_states",
        "txt_ids",
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "txt_ids":
            import torch

            shape = (
                [self.batch_size, self.sequence_length, 3]
                if is_diffusers_version("<", "0.31.0")
                else [self.sequence_length, 3]
            )
            dtype = DTYPE_MAPPER.pt(float_dtype)
            return torch.full(shape, 0, dtype=dtype)
        return super().generate(input_name, framework, int_dtype, float_dtype)


class LTXVaeDummyInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("pixel_values", "pixel_mask", "sample", "latent_sample", "timestep")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        num_frames: int = 2,
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        self.num_frames = num_frames

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["sample", "latent_sample"]:
            return self.random_float_tensor(
                [self.batch_size, self.num_channels, self.num_frames, self.height, self.width]
            )
        if input_name == "timestep":
            return self.random_int_tensor([1], max_value=20, min_value=1, framework=framework, dtype=int_dtype)

        return super().generate(input_name, framework, int_dtype, float_dtype)


class LTXTransformerDummyInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("hidden_states", "width", "height", "num_frames", "rope_interpolation_scale")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = 16,
        height: int = 8,
        num_frames: int = 2,
        frame_rate: int = 10,
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.vae_spatial_compression_ratio = normalized_config.config.vae_spatial_compression_ratio
        self.vae_temporal_compression_ratio = normalized_config.config.vae_temporal_compression_ratio

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        import torch

        if input_name == "hidden_states":
            return self.random_float_tensor(
                [self.batch_size, self.num_frames * self.height * self.width, self.num_channels]
            )
        if input_name == "width":
            return torch.tensor(self.width)
        if input_name == "height":
            return torch.tensor(self.height)
        if input_name == "num_frames":
            return torch.tensor(self.num_frames)
        if input_name == "rope_interpolation_scale":
            import torch

            return torch.tensor(
                [
                    self.vae_temporal_compression_ratio / self.frame_rate,
                    self.vae_spatial_compression_ratio,
                    self.vae_spatial_compression_ratio,
                ]
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyMiniCPMVImageInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("pixel_values", "patch_attention_mask", "position_ids")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height)
        self.patch_size = normalized_config.config.patch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "pixel_values":
            return self.random_float_tensor(
                shape=[
                    self.batch_size,
                    self.num_channels,
                    self.patch_size,
                    (self.height * self.width) // self.patch_size,
                ],
                framework=framework,
                dtype=float_dtype,
            )

        if input_name == "patch_attention_mask":
            return self.random_int_tensor(
                shape=[self.batch_size, 1, (self.height // self.patch_size) * (self.width // self.patch_size)],
                framework=framework,
                dtype=float_dtype,
                min_value=0,
                max_value=2,
            )

        if input_name == "position_ids":
            return self.random_int_tensor(
                shape=[self.batch_size, (self.height // self.patch_size) * (self.width // self.patch_size)],
                max_value=self.patch_size,
            )


class DummyMiniCPMVResampleInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("image_feature", "pos_embed", "key_padding_mask")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height)
        self.patch_size = normalized_config.config.patch_size
        self.hidden_size = normalized_config.config.hidden_size
        self.img_hidden_size = normalized_config.config.vision_config.hidden_size
        self.feat_size = (normalized_config.config.vision_config.image_size // self.patch_size) * (
            normalized_config.config.vision_config.image_size // self.patch_size
        )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "image_feature":
            return self.random_float_tensor(
                shape=[self.batch_size, self.feat_size, self.img_hidden_size], framework=framework, dtype=float_dtype
            )

        if input_name == "key_padding_mask":
            return self.constant_tensor(
                shape=[self.batch_size, self.feat_size],
                framework=framework,
                value=1,
                dtype=DTYPE_MAPPER.pt(float_dtype),
            )

        if input_name == "pos_embed":
            return self.random_float_tensor(shape=[self.feat_size, self.batch_size, self.hidden_size])


class DummyPhi3VisionProjectionInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = 336,
        height: int = 336,
        crop_size=336,
        **kwargs,
    ):
        self.batch_size = batch_size
        self._embed_layer_realization = (
            normalized_config.config.embd_layer["embedding_cls"]
            if hasattr(normalized_config.config, "embd_layer")
            else "image_audio"
        )
        if not hasattr(normalized_config.config, "vision_config"):
            self.image_dim_out = (
                normalized_config.config.img_processor.get(
                    "image_dim_out", normalized_config.config.img_processor.get("hidden_size")
                )
                if normalized_config.config.img_processor is not None
                else 1152
            )
            if "image_embd_layer" in normalized_config.config.embd_layer:
                self.crop_size = normalized_config.config.embd_layer["image_embd_layer"].get("crop_size", crop_size)
            else:
                self.crop_size = normalized_config.config.embd_layer.get("crop_size", crop_size)
        else:
            self.image_dim_out = normalized_config.config.vision_config.hidden_size
            self.crop_size = normalized_config.config.vision_config.crop_size
        self.height = height
        self.width = width

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        h = self.height // self.crop_size
        w = self.width // self.crop_size
        feat_size = (h * w + 1) * 144 + 1 + (h + 1) * 12
        if self._embed_layer_realization in ["linear", "image_audio"]:
            shape = [self.batch_size, feat_size, self.image_dim_out]
        else:
            shape = [self.batch_size, feat_size, self.image_dim_out * 4]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class DummyAudioPhi4MMInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("audio_input", "audio_feature", "audio_mask")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        signal_length=498,
        **kwargs,
    ):
        self.signal_length = signal_length
        if hasattr(normalized_config.config, "audio_processor"):
            self.audio_chunk_size = (
                signal_length // normalized_config.config.audio_processor["config"]["time_reduction"] + 1
            )
            self.input_size = normalized_config.config.audio_processor["config"]["input_size"]
            self.attention_dim = normalized_config.config.audio_processor["config"]["attention_dim"]
        else:
            self.audio_chunk_size = signal_length // normalized_config.config.audio_config.time_reduction + 1
            self.input_size = normalized_config.config.audio_config.input_size
            self.attention_dim = normalized_config.config.audio_config.hidden_size
        self.batch_size = batch_size
        self.task = task
        self.normalized_config = normalized_config

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "audio_input":
            return self.random_float_tensor(
                [self.batch_size, self.signal_length, self.input_size], framework=framework, dtype=float_dtype
            )

        if input_name == "audio_feature":
            return self.random_float_tensor(
                [self.batch_size, self.audio_chunk_size, self.attention_dim], framework=framework, dtype=float_dtype
            )

        if input_name == "audio_mask":
            return self.random_int_tensor(
                [self.batch_size, self.audio_chunk_size, self.audio_chunk_size],
                max_value=2,
                framework=framework,
                dtype="bool",
            )


class DummyVisionPositionIdsPhi4InputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("patch_position_ids", "patch_attention_mask")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        if hasattr(normalized_config.config, "vision_conifg"):
            self.patch_size = getattr(normalized_config.config.vision_config, "patch_size", 14)
        else:
            self.patch_size = 14
        self.num_patches_per_side = self.height // self.patch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "patch_position_ids":
            return self.get_vision_position_ids()
        if input_name == "patch_attention_mask":
            return self.random_int_tensor(
                [self.batch_size, self.height // self.patch_size, self.width // self.patch_size],
                framework=framework,
                dtype="bool",
                max_value=2,
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)

    def get_vision_position_ids(self):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L494-L512
        import torch

        batch_size = self.batch_size
        max_im_h, max_im_w = self.height, self.width
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )
        patch_attention_mask = torch.ones(
            [self.batch_size, self.height // self.patch_size, self.width // self.patch_size], dtype=torch.int64
        )
        patch_attention_mask[0, self.height - 2 :] = 0

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
        return position_ids


class DummyQwen2VLLMInputGenerator(DummyTextInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        generated_input = super().generate(input_name, framework, int_dtype, float_dtype)
        if input_name == "position_ids":
            return generated_input.unsqueeze(0).expand(3, -1, -1)
        return generated_input


class DummyQwen3_5LMInputGenerator(DummyTextInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        generated_input = super().generate(input_name, framework, int_dtype, float_dtype)
        if input_name == "position_ids":
            return generated_input.unsqueeze(0).expand(4, -1, -1)
        return generated_input


class DummyQwen2VLVisionEmbedInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "hidden_states",
        "attention_mask",
        "window_attention_mask",
        "window_index",
        "rotary_pos_emb",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = 1,
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = 420,
        height: int = 420,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.temporal_patch_size = normalized_config.config.temporal_patch_size
        self.patch_size = normalized_config.config.patch_size
        if normalized_config.use_embed_dim:
            self.embed_dim = (
                normalized_config.config.embed_dim
                if hasattr(normalized_config.config, "embed_dim")
                else normalized_config.hidden_size
            )
        else:
            self.embed_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        self.num_heads = normalized_config.config.num_heads
        self.spatial_merge_size = None
        if hasattr(normalized_config.config, "spatial_merge_size"):
            self.spatial_merge_size = normalized_config.config.spatial_merge_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        grid_h, grid_w = self.height // self.patch_size, self.width // self.patch_size
        grid_t = self.batch_size

        if input_name == "hidden_states":
            return self.random_float_tensor(
                [grid_t * grid_h * grid_w, self.embed_dim], framework=framework, dtype=float_dtype
            )

        if input_name in ["attention_mask", "window_attention_mask"]:
            return self.random_mask_tensor(
                [1, grid_t * grid_h * grid_w, grid_t * grid_h * grid_w], framework=framework, dtype=float_dtype
            )

        if input_name == "rotary_pos_emb":
            dim = self.embed_dim // self.num_heads // 2
            return self.random_float_tensor([grid_h * grid_t * grid_w, dim], framework=framework, dtype=float_dtype)

        if input_name == "window_index":
            if self.spatial_merge_size is None:
                raise ValueError(
                    "`spatial_merge_size` parameter is not found in model config. Can not generate dummy input data for `window_index` input"
                )
            spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
            hidden_size = (grid_t * grid_h * grid_w) // spatial_merge_unit
            return self.random_int_tensor([hidden_size], max_value=hidden_size)


# DummyQwen3VLVisionEmbedInputGenerator inherits from DummyQwen2VLVisionEmbedInputGenerator (defined above)
class DummyQwen3VLVisionEmbedInputGenerator(DummyQwen2VLVisionEmbedInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "hidden_states",
        "attention_mask",
        "rotary_pos_emb",
        "input",
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        grid_h, grid_w = self.height // self.patch_size, self.width // self.patch_size
        grid_t = self.batch_size

        if input_name == "hidden_states":
            return self.random_float_tensor(
                [grid_t * grid_h * grid_w, self.embed_dim], framework=framework, dtype=float_dtype
            )

        if input_name in ["attention_mask"]:
            return self.random_mask_tensor(
                [1, grid_t * grid_h * grid_w, grid_t * grid_h * grid_w], framework=framework, dtype=float_dtype
            )

        if input_name == "rotary_pos_emb":
            dim = self.embed_dim // self.num_heads // 2
            return self.random_float_tensor([grid_h * grid_t * grid_w, dim], framework=framework, dtype=float_dtype)

        if input_name == "input":
            return self.constant_tensor(
                [4, DEFAULT_DUMMY_SHAPES["sequence_length"]],
                framework=framework,
                value=0,
                dtype=DTYPE_MAPPER.pt(int_dtype),
            )


class Qwen3ASRDummySeq2SeqPastKeyValuesGenerator(DummySeq2SeqPastKeyValuesGenerator):
    """Custom KV cache generator for Qwen3-ASR with GQA (num_key_value_heads != num_attention_heads).
    Qwen3-ASR has no cross-attention, so only self-attention KV cache is generated (2 per layer)."""

    def __init__(self, task, normalized_config, **kwargs):
        super().__init__(task, normalized_config, **kwargs)
        # Override head count and head_dim for GQA
        self.decoder_num_attention_heads = normalized_config.decoder_num_attention_heads
        self.decoder_head_dim = getattr(normalized_config, "head_dim", None)
        if self.decoder_head_dim is None:
            self.decoder_head_dim = self.decoder_hidden_size // normalized_config.num_attention_heads

    def generate(self, input_name, framework="pt", int_dtype="int64", float_dtype="fp32"):
        if input_name == "past_key_values":
            decoder_shape = (
                self.batch_size,
                self.decoder_num_attention_heads,
                self.sequence_length,
                self.decoder_head_dim,
            )
            # Qwen3-ASR has no cross-attention, only self-attention KV cache
            return [
                (
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(decoder_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.decoder_num_layers)
            ]
        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)


class FunASRDummyAudioInputGenerator(DummyAudioInputGenerator):
    """Dummy audio feature generator for FunASR.

    FunASR's encoder consumes fbank features laid out as (batch, num_frames, feature_size),
    unlike the default (batch, feature_size, num_frames) layout used by the base generator.
    """

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "input_features":
            return self.random_float_tensor(
                shape=[self.batch_size, self.nb_max_frames, self.feature_size],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)


class DummyGemma4VisionInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("pixel_values", "image_position_ids")

    def __init__(self, task, normalized_config, batch_size=DEFAULT_DUMMY_SHAPES["batch_size"], **kwargs):
        super().__init__(task, normalized_config, batch_size, **kwargs)
        self.patch_size = getattr(normalized_config, "patch_size", 16)
        self.pooling_kernel_size = getattr(normalized_config, "pooling_kernel_size", 3)
        # Gemma4 processor always pads pixel_values to max_soft_tokens * pooling_kernel_size^2 patches.
        # The vision model's pooling uses shape-dependent Python operations that get baked in during tracing,
        # so the dummy input must match the actual inference shapes.
        max_soft_tokens = getattr(normalized_config, "image_seq_length", None)
        if max_soft_tokens is None:
            max_soft_tokens = getattr(normalized_config, "max_soft_tokens", 280)
        self.num_patches = max_soft_tokens * self.pooling_kernel_size**2

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "pixel_values":
            # Gemma4 expects pre-patchified pixel_values: [batch, num_patches, 3 * patch_size^2]
            return self.random_float_tensor(
                shape=[self.batch_size, self.num_patches, 3 * self.patch_size**2],
                framework=framework,
                dtype=float_dtype,
            )
        if input_name == "image_position_ids":
            # Create position ids as a grid. The patch count = h_patches * w_patches
            # where both are divisible by pooling_kernel_size for correct pooling.
            import math

            k = self.pooling_kernel_size
            total_pooled = self.num_patches // (k * k)
            # Find roughly square grid for pooled side
            pooled_side = int(math.sqrt(total_pooled))
            if pooled_side * pooled_side < total_pooled:
                pooled_h = pooled_side
                pooled_w = total_pooled // pooled_h
            else:
                pooled_h = pooled_w = pooled_side
            h_patches = pooled_h * k
            w_patches = pooled_w * k
            pos_ids = torch.stack(
                torch.meshgrid(torch.arange(h_patches), torch.arange(w_patches), indexing="ij"), dim=-1
            ).reshape(1, -1, 2)
            # Pad to num_patches with -1 (padding position)
            if pos_ids.shape[1] < self.num_patches:
                pad = torch.full((1, self.num_patches - pos_ids.shape[1], 2), -1, dtype=pos_ids.dtype)
                pos_ids = torch.cat([pos_ids, pad], dim=1)
            return pos_ids.expand(self.batch_size, -1, -1).clone()
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyVisionPositionIdsInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("patch_attention_mask", "patch_position_ids")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        self.patch_size = normalized_config.config.patch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "patch_attention_mask":
            shape = [self.batch_size, self.height // self.patch_size, self.width // self.patch_size]
            return self.random_int_tensor(shape, max_value=2, framework=framework, dtype="bool")
        if input_name == "patch_position_ids":
            max_nb_patches_h, max_nb_patches_w = self.height // self.patch_size, self.width // self.patch_size
            shape = [self.batch_size, max_nb_patches_h * max_nb_patches_w]
            return self.random_int_tensor(
                shape, max_value=min(max_nb_patches_h, max_nb_patches_w), framework=framework, dtype=int_dtype
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummySpeechT5InputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "inputs_embeds",
        "output_sequence",
        "speaker_embeddings",
        "spectrogram",
        "raw_spectrogram",
        "encoder_hidden_states",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = 1

        self.sequence_length = sequence_length
        self.speaker_embedding_dim = normalized_config.speaker_embedding_dim
        self.num_mel_bins = normalized_config.num_mel_bins
        self.reduction_factor = normalized_config.config.reduction_factor
        self.hidden_size = normalized_config.config.hidden_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["output_sequence", "inputs_embeds"]:
            shape = [self.batch_size, self.sequence_length, self.num_mel_bins]
        elif input_name == "speaker_embeddings":
            shape = [self.batch_size, self.speaker_embedding_dim]
        elif input_name == "raw_spectrogram":
            shape = [self.sequence_length, self.batch_size, self.reduction_factor, self.num_mel_bins]
        elif input_name == "encoder_hidden_states":
            shape = [self.batch_size, self.sequence_length, self.hidden_size]
        elif input_name == "spectrogram":
            shape = [self.batch_size, self.sequence_length, self.num_mel_bins]
        else:
            raise ValueError(f"Unsupported input {input_name} for DummySpeechT5InputGenerator")

        return self.random_float_tensor(
            shape=shape,
            min_value=0,
            max_value=1,
            framework=framework,
            dtype=float_dtype,
        )


class MambaCacheDummyInputGenerator(DummyInputGenerator):
    """
    Generates dummy past_ssm_states, past_conv_states and cache_position inputs for Mamba architectures.
    """

    SUPPORTED_INPUT_NAMES = ("cache_params", "cache_position")

    def __init__(
        self,
        task: str,
        normalized_config,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.normalized_config = normalized_config
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.intermediate_size = self.normalized_config.config.intermediate_size
        self.ssm_state_size = self.normalized_config.config.state_size
        self.conv_kernel_size = self.normalized_config.config.conv_kernel

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "cache_params":
            ssm_shape = [self.batch_size, self.intermediate_size, self.ssm_state_size]
            conv_shape = [self.batch_size, self.intermediate_size, self.conv_kernel_size]
            return [
                (
                    self.random_float_tensor(ssm_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(conv_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.normalized_config.num_layers)
            ]
        elif input_name == "cache_position":
            return self.random_int_tensor(
                shape=[self.conv_kernel_size],
                max_value=self.sequence_length,
                framework=framework,
                dtype=int_dtype,
            )

        raise ValueError(f"Unsupported input name {input_name}")


class Zamba2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy cache_params inputs for Zamba2 architectures.
    """

    SUPPORTED_INPUT_NAMES = ("cache_params",)

    def __init__(
        self,
        task: str,
        normalized_config,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            **kwargs,
        )

        config = normalized_config.config
        self.intermediate_size = int(config.mamba_expand * config.hidden_size)
        self.conv_kernel_size = config.mamba_d_conv
        self.mamba_d_state = config.mamba_d_state
        if config.model_type == "zamba2":
            self.n_mamba_heads = config.n_mamba_heads
            self.mamba_ngroups = config.mamba_ngroups
            self.mamba_headdim = config.mamba_headdim
            self.head_dim = config.attention_head_dim
            # in Zamba2, all layers contain Mamba block
            # some of these layers are hybrid so they contain both attention and mamba blocks
            self.num_attention_layers = len(config.hybrid_layer_ids)
            self.num_mamba_layers = self.num_layers
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "The current support for the 'Zamba2' model type is experimental. "
                "Performance is not optimal with high memory consumption. "
                "Optimizations and improved support will be available in a future OpenVINO release."
            )
        else:
            # currently, this else-branch is applied for GraniteMoeHybrid models
            self.n_mamba_heads = config.mamba_n_heads
            self.mamba_ngroups = config.mamba_n_groups
            self.mamba_headdim = config.mamba_d_head
            self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            self.num_attention_layers = config.layer_types.count("attention")
            self.num_mamba_layers = config.layer_types.count("mamba")
            self.num_attention_heads = config.num_key_value_heads
            self.sequence_length = 0

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_values = []
        for i in range(self.num_mamba_layers):
            conv_state_shape = (
                self.batch_size,
                self.intermediate_size + 2 * self.mamba_ngroups * self.mamba_d_state,
                self.conv_kernel_size,
            )
            conv_state = self.random_float_tensor(conv_state_shape, framework=framework, dtype=float_dtype)
            past_key_values.append(conv_state)
            ssm_state_shape = (self.batch_size, self.n_mamba_heads, self.mamba_headdim, self.mamba_d_state)
            ssm_state = self.random_float_tensor(ssm_state_shape, framework=framework, dtype=float_dtype)
            past_key_values.append(ssm_state)

        for i in range(self.num_attention_layers):
            kv_shape = (self.batch_size, self.num_attention_heads, self.sequence_length, self.head_dim)
            k = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            v = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            past_key_values.append(k)
            past_key_values.append(v)

        return past_key_values


class Lfm2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy past_key_values inputs for Lfm2 architectures.
    """

    SUPPORTED_INPUT_NAMES = ("cache_params",)

    def __init__(
        self,
        task: str,
        normalized_config,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            **kwargs,
        )
        config = normalized_config.config
        self.num_conv_layers = config.layer_types.count("conv")
        self.num_atten_layers = config.layer_types.count("full_attention")
        self.batch_size = batch_size
        self.normalized_config = normalized_config
        self.hidden_size = self.normalized_config.hidden_size
        self.conv_L_cache = self.normalized_config.conv_L_cache
        self.num_key_value_heads = self.normalized_config.num_key_value_heads
        self.num_hidden_layers = self.normalized_config.num_hidden_layers

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_values = []

        for i in range(self.num_conv_layers):
            conv_state_shape = (self.batch_size, self.hidden_size, self.conv_L_cache)
            conv_state = self.random_float_tensor(conv_state_shape, framework=framework, dtype=float_dtype)
            past_key_values.append(conv_state)

        for i in range(self.num_atten_layers):
            shape = (
                self.batch_size,
                self.num_key_value_heads,
                self.sequence_length,
                self.hidden_size // self.num_attention_heads,
            )

            kv_shape = shape  # (self.batch_size, self.num_attention_heads, self.sequence_length, self.head_dim)
            k = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            v = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            past_key_values.append(k)
            past_key_values.append(v)

        return past_key_values


class DummyVideoChatFlashQwenInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("hidden_states", "rotary_pos_emb")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        visual_seq_length: int = DEFAULT_DUMMY_SHAPES["visual_seq_length"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, visual_seq_length, **kwargs)
        self.num_frames = normalized_config.config.mm_local_num_frames
        self.embed_dim = normalized_config.config.mm_hidden_size
        self.height = normalized_config.config.image_size
        self.width = normalized_config.config.image_size
        self.image_size = (self.height, self.width)
        self.patch_size = normalized_config.config.patch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "hidden_states":
            return self.random_float_tensor(
                shape=[
                    self.batch_size,
                    self.num_channels,
                    self.num_frames,
                    self.height,
                    self.width,
                ],
                framework=framework,
                dtype=float_dtype,
            )
        elif input_name == "rotary_pos_emb":
            grid_h, grid_w = self.height // self.patch_size, self.width // self.patch_size
            grid_t = self.num_frames
            # Source: https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B/blob/main/vision_tower_builder.py#L523
            # The first dimension of rotary_pos_emb is fixed to 1 in the original model.
            # And the second dimension is the total number of tokens for all frames, which is calculated as grid_h * grid_w * grid_t plus 1 for the cls token.
            return self.random_float_tensor(
                [1, 1 + grid_h * grid_t * grid_w, self.embed_dim], framework=framework, dtype=float_dtype
            )


class DummyVideoChatFlashQwenProjectorInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ["input"]

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.hidden_size = normalized_config.config.mm_hidden_size
        self.num_patches = normalized_config.config.mm_projector_num_tome_tokens
        self.normalized_config = normalized_config

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        shape = [self.batch_size, self.num_patches, self.hidden_size]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class Qwen3NextDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy cache_params inputs for Qwen3-Next architectures.
    """

    SUPPORTED_INPUT_NAMES = ("cache_params",)

    def __init__(
        self,
        task: str,
        normalized_config,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            **kwargs,
        )

        config = normalized_config.config
        self.num_full_attn_layers = config.layer_types.count("full_attention")
        self.num_linear_attn_layers = config.layer_types.count("linear_attention")
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.num_key_value_heads = config.num_key_value_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        cache_params = []

        for idx in range(self.num_linear_attn_layers):
            d_inner = self.num_k_heads * (2 * self.head_k_dim + self.head_v_dim * self.num_v_heads // self.num_k_heads)
            conv_state_shape = (
                self.batch_size,
                d_inner,
                self.conv_kernel_size,
            )
            conv_state = self.random_float_tensor(conv_state_shape, framework=framework, dtype=float_dtype)
            cache_params.append(conv_state)
            num_heads = self.num_v_heads
            recurrent_state_shape = (self.batch_size, num_heads, self.head_k_dim, self.head_v_dim)
            recurrent_state = self.random_float_tensor(recurrent_state_shape, framework=framework, dtype=float_dtype)
            cache_params.append(recurrent_state)

        for idx in range(self.num_full_attn_layers):
            kv_shape = (self.batch_size, self.num_key_value_heads, self.sequence_length, self.head_dim)
            k = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            v = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            cache_params.append(k)
            cache_params.append(v)

        return cache_params


class Qwen3_5DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    Generates dummy cache_params inputs for Qwen3.5 architectures.
    """

    SUPPORTED_INPUT_NAMES = ("cache_params",)

    def __init__(
        self,
        task: str,
        normalized_config,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            **kwargs,
        )

        config = normalized_config.config
        self.num_full_attn_layers = config.layer_types.count("full_attention")
        self.num_linear_attn_layers = config.layer_types.count("linear_attention")
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.num_key_value_heads = config.num_key_value_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        cache_params = []

        for idx in range(self.num_linear_attn_layers):
            d_inner = self.num_k_heads * (2 * self.head_k_dim + self.head_v_dim * self.num_v_heads // self.num_k_heads)
            conv_state_shape = (
                self.batch_size,
                d_inner,
                self.conv_kernel_size,
            )
            conv_state = self.random_float_tensor(conv_state_shape, framework=framework, dtype=float_dtype)
            cache_params.append(conv_state)
            num_heads = self.num_v_heads
            recurrent_state_shape = (self.batch_size, num_heads, self.head_k_dim, self.head_v_dim)
            recurrent_state = self.random_float_tensor(recurrent_state_shape, framework=framework, dtype=float_dtype)
            cache_params.append(recurrent_state)

        for idx in range(self.num_full_attn_layers):
            kv_shape = (self.batch_size, self.num_key_value_heads, self.sequence_length, self.head_dim)
            k = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            v = self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype)
            cache_params.append(k)
            cache_params.append(v)

        return cache_params


class DummyKokoroInputGenerator(DummyInputGenerator):
    """Generates dummy inputs for the Kokoro TTS model."""

    SUPPORTED_INPUT_NAMES = ("input_ids", "ref_s", "speed")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = 1
        self.sequence_length = sequence_length
        self.style_dim = getattr(normalized_config, "style_dim", 128)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "input_ids":
            shape = [self.batch_size, self.sequence_length]
            input_ids_value = self.random_int_tensor(
                shape=shape, min_value=0, max_value=178, framework=framework, dtype=int_dtype
            )
            input_ids_value[:, 0] = 0
            input_ids_value[:, -1] = 0
            return input_ids_value
        elif input_name == "ref_s":
            shape = [self.batch_size, self.style_dim * 2]
            return self.random_float_tensor(
                shape=shape, min_value=-1, max_value=1, framework=framework, dtype=float_dtype
            )
        elif input_name == "speed":
            return self.random_int_tensor(shape=[1], min_value=1, max_value=10, framework=framework, dtype=float_dtype)
        else:
            raise ValueError(f"Unsupported input {input_name} for DummyKokoroInputGenerator")
