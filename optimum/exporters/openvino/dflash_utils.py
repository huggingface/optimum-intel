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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from transformers import PretrainedConfig


DFLASH_ARCHITECTURES = {"DFlashDraftModel"}
DSPARK_ARCHITECTURES = {"Qwen3DSparkModel"}
DSPARK_SUPPORTED_MARKOV_HEAD_TYPES = ("vanilla",)
MARKOV_HEAD_INTERFACE = "markov_head_v1"
# K of the TopK in the exported candidate scorer. Runtimes may replace it at load time.
MARKOV_HEAD_DEFAULT_TOP_K = 64


@dataclass(frozen=True)
class DSparkConfig:
    markov_rank: int
    markov_head_type: str
    sample_from_anchor: bool

    @property
    def candidate_position_offset(self) -> int:
        # DeepSpec drafts predict the first token from the anchor row. Drafts with
        # sample_from_anchor=False use the DFlash [anchor ; masks] layout instead.
        return 0 if self.sample_from_anchor else 1


def _dspark_config_value(config: PretrainedConfig, name: str, default: Any) -> Any:
    raw_config = getattr(config, "dflash_config", None)
    nested_value = raw_config.get(name) if isinstance(raw_config, Mapping) else None
    flat_value = getattr(config, name, None)
    if nested_value is not None:
        if flat_value is not None and flat_value != nested_value:
            raise ValueError(f"DSpark configuration field {name!r} has conflicting nested and flat values.")
        return nested_value
    return default if flat_value is None else flat_value


def parse_and_validate_dspark_config(config: PretrainedConfig) -> DSparkConfig:
    """Read the DSpark Markov-head fields from flat config attributes or the nested dflash_config."""
    markov_rank = _dspark_config_value(config, "markov_rank", 0)
    if not isinstance(markov_rank, int) or isinstance(markov_rank, bool) or markov_rank < 0:
        raise ValueError("DSpark markov_rank must be a non-negative integer.")
    markov_head_type = str(_dspark_config_value(config, "markov_head_type", "vanilla")).lower()
    if markov_rank > 0 and markov_head_type not in DSPARK_SUPPORTED_MARKOV_HEAD_TYPES:
        raise ValueError(
            f"DSpark markov_head_type={markov_head_type!r} is not supported; "
            f"supported Markov heads: {', '.join(DSPARK_SUPPORTED_MARKOV_HEAD_TYPES)}."
        )
    sample_from_anchor = _dspark_config_value(config, "sample_from_anchor", True)
    if not isinstance(sample_from_anchor, bool):
        raise ValueError("DSpark sample_from_anchor must be a boolean.")
    return DSparkConfig(
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
        sample_from_anchor=sample_from_anchor,
    )


def markov_head_default_top_k(config: PretrainedConfig) -> int:
    return min(MARKOV_HEAD_DEFAULT_TOP_K, config.vocab_size)
