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
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict

from transformers import PretrainedConfig

from openvino import Model
from optimum.intel.utils.import_utils import is_transformers_version


DFLASH_ARCHITECTURE_VERSIONS = {
    "DFlashDraftModel": 1,
    # DeepSpec's DSpark draft is the DFlash v1 backbone, optionally with a Markov head.
    "Qwen3DSparkModel": 1,
    "DFlash2DraftModel": 2,
}
DFLASH_ARCHITECTURES = set(DFLASH_ARCHITECTURE_VERSIONS)
DSPARK_ARCHITECTURES = {"Qwen3DSparkModel"}
DSPARK_SUPPORTED_MARKOV_HEAD_TYPES = ("vanilla",)
# The DFlash-2 candidate selector is a Markov head whose transitions are gated by the draft hidden states.
DFLASH2_MARKOV_HEAD_TYPE = "dflash2"
MARKOV_HEAD_INTERFACE = "markov_head_v1"
# K of the TopK in the exported candidate scorer of a DSpark Markov head. Runtimes may replace it at load time.
MARKOV_HEAD_DEFAULT_TOP_K = 64
# The DFlash-2 draft requires at least this activation scale for correct GPU FP16 execution. It is
# stored as DFlash-specific IR metadata and applied by GenAI as an explicit compile property, since
# the GPU plugin intentionally does not consume generic activation-scale RT info for LLM graphs.
DFLASH2_GPU_MIN_ACTIVATIONS_SCALE_FACTOR = 32.0
DFLASH_COMMON_CONFIG_FIELDS = (
    "mask_token_id",
    "target_layer_ids",
)
DFLASH2_REQUIRED_CONFIG_FIELDS = (
    "conv_group_size",
    "conv_kernel_size",
    "selector_rank",
    "selector_top_k",
)
# These transforms are applied by the serving runtime because the DFlash export
# intentionally excludes the target embedding and LM head. `input_embedding_scale`
# scales only target embedding rows reused as draft inputs before the DFlash-2
# backbone; it does not alter the target model's own embedding path. When absent,
# runtimes use the identity scale (1.0). The remaining fields transform selected
# unary logits before candidate-path scoring.
DFLASH2_OPTIONAL_CONFIG_FIELDS = (
    "final_logit_softcapping",
    "input_embedding_scale",
    "output_multiplier",
)


@dataclass(frozen=True)
class DFlashConfig:
    version: int
    values: Dict[str, Any]


def config_architecture(config: PretrainedConfig) -> str | None:
    """architectures[0] of the config, or None when it names no architecture."""
    architectures = getattr(config, "architectures", None)
    return architectures[0] if isinstance(architectures, list) and architectures else None


def _detect_dflash_version(config: PretrainedConfig) -> int:
    architecture = config_architecture(config)
    if architecture not in DFLASH_ARCHITECTURE_VERSIONS:
        raise ValueError(
            "DFlash export requires architectures[0] to be "
            f"one of {sorted(DFLASH_ARCHITECTURES)}, got {architecture!r}."
        )
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError(
            f"DFlash export supports only Qwen3-based draft models, got model_type={config.model_type!r}."
        )
    return DFLASH_ARCHITECTURE_VERSIONS[architecture]


def _normalize_dflash_config(config: PretrainedConfig, version: int) -> Dict[str, Any]:
    """Return the canonical nested DFlash config, using flat fields only as fallbacks."""
    raw_config = getattr(config, "dflash_config", None)
    if raw_config is None:
        dflash_config = {}
    elif isinstance(raw_config, Mapping):
        dflash_config = dict(raw_config)
    else:
        raise ValueError("DFlash dflash_config must be a mapping.")

    config_fields = DFLASH_COMMON_CONFIG_FIELDS
    if version == 2:
        config_fields += DFLASH2_REQUIRED_CONFIG_FIELDS + DFLASH2_OPTIONAL_CONFIG_FIELDS
    for name in config_fields:
        flat_value = getattr(config, name, None)
        if name in dflash_config:
            if flat_value is not None and flat_value != dflash_config[name]:
                raise ValueError(f"DFlash configuration field {name!r} has conflicting nested and flat values.")
        elif flat_value is not None:
            dflash_config[name] = flat_value
    return dflash_config


def _validate_common_dflash_config(config: PretrainedConfig, dflash_config: Dict[str, Any], version: int) -> None:
    missing = [name for name in DFLASH_COMMON_CONFIG_FIELDS if dflash_config.get(name) is None]
    if missing:
        raise ValueError(f"DFlash v{version} export requires configuration fields: {', '.join(missing)}.")

    mask_token_id = dflash_config["mask_token_id"]
    vocab_size = getattr(config, "vocab_size", None)

    if not _is_positive_int(vocab_size):
        raise ValueError(f"DFlash v{version} export requires a positive integer vocabulary size.")
    if not isinstance(mask_token_id, int) or isinstance(mask_token_id, bool) or not 0 <= mask_token_id < vocab_size:
        raise ValueError(f"DFlash v{version} dflash_config['mask_token_id'] must be an integer in [0, vocab_size).")

    target_layer_ids = dflash_config["target_layer_ids"]
    if (
        not isinstance(target_layer_ids, (list, tuple))
        or not target_layer_ids
        or any(
            not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0
            for layer_id in target_layer_ids
        )
        or len(set(target_layer_ids)) != len(target_layer_ids)
    ):
        raise ValueError(
            f"DFlash v{version} dflash_config['target_layer_ids'] must be a non-empty sequence "
            "of unique non-negative integers."
        )


def _validate_dflash2_config(config: PretrainedConfig, dflash_config: Dict[str, Any]) -> None:
    missing = [name for name in DFLASH2_REQUIRED_CONFIG_FIELDS if dflash_config.get(name) is None]
    if missing:
        raise ValueError(f"DFlash v2 export requires configuration fields: {', '.join(missing)}.")

    conv_group_size = dflash_config["conv_group_size"]
    conv_kernel_size = dflash_config["conv_kernel_size"]
    selector_rank = dflash_config["selector_rank"]
    selector_top_k = dflash_config["selector_top_k"]
    vocab_size = config.vocab_size
    num_target_layers = getattr(config, "num_target_layers", None)

    if not _is_positive_int(conv_kernel_size):
        raise ValueError("DFlash v2 dflash_config['conv_kernel_size'] must be a positive integer.")
    if not _is_positive_int(conv_group_size) or config.hidden_size % conv_group_size:
        raise ValueError("DFlash v2 dflash_config['conv_group_size'] must be a positive divisor of hidden_size.")
    if not _is_positive_int(selector_rank):
        raise ValueError("DFlash v2 dflash_config['selector_rank'] must be a positive integer.")
    if not _is_positive_int(selector_top_k) or selector_top_k > vocab_size:
        raise ValueError("DFlash v2 dflash_config['selector_top_k'] must be between 1 and vocab_size.")
    if not _is_positive_int(num_target_layers):
        raise ValueError("DFlash v2 export requires a positive integer num_target_layers.")
    if any(layer_id >= num_target_layers for layer_id in dflash_config["target_layer_ids"]):
        raise ValueError("DFlash v2 dflash_config['target_layer_ids'] must be smaller than num_target_layers.")

    for name in ("input_embedding_scale", "output_multiplier"):
        value = dflash_config.get(name)
        if value is not None and (
            not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"DFlash v2 dflash_config[{name!r}] must be a finite positive number.")

    final_logit_softcapping = dflash_config.get("final_logit_softcapping")
    if final_logit_softcapping is not None and (
        not isinstance(final_logit_softcapping, Real)
        or isinstance(final_logit_softcapping, bool)
        or not math.isfinite(final_logit_softcapping)
        or final_logit_softcapping < 0
    ):
        raise ValueError("DFlash v2 dflash_config['final_logit_softcapping'] must be a finite non-negative number.")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_dflash_draft_model(ov_model: Model) -> bool:
    """Whether an IR was exported as a DFlash draft backbone (a Markov head model is not one)."""
    return ov_model.has_rt_info(["dflash_mode"]) and ov_model.get_rt_info(["dflash_mode"]).value == "True"


def parse_and_validate_dflash_config(config: PretrainedConfig, expected_version: int | None = None) -> DFlashConfig:
    """Detect and validate a DFlash draft config, branching by checkpoint version."""
    version = _detect_dflash_version(config)
    if expected_version is not None and version != expected_version:
        raise ValueError(f"Expected a DFlash v{expected_version} checkpoint, got DFlash v{version}.")
    if version == 2 and not is_transformers_version(">=", "4.57"):
        raise ValueError("DFlash v2 export requires Transformers >= 4.57.")

    dflash_config = _normalize_dflash_config(config, version)
    _validate_common_dflash_config(config, dflash_config, version)
    if version == 2:
        _validate_dflash2_config(config, dflash_config)
    return DFlashConfig(version=version, values=dflash_config)


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


@dataclass(frozen=True)
class MarkovHeadSpec:
    markov_rank: int
    markov_head_type: str
    top_k: int


def dflash_candidate_position_offset(config: PretrainedConfig) -> int:
    """The block row of the first drafted candidate: 1 after the anchor row, or 0 when the anchor row drafts it."""
    if config_architecture(config) in DSPARK_ARCHITECTURES:
        return parse_and_validate_dspark_config(config).candidate_position_offset
    return 1


def markov_head_spec(config: PretrainedConfig) -> MarkovHeadSpec | None:
    """The Markov head exported next to a DFlash draft backbone, or None for a draft without one."""
    architecture = config_architecture(config)
    if architecture in DSPARK_ARCHITECTURES:
        dspark_config = parse_and_validate_dspark_config(config)
        if dspark_config.markov_rank == 0:
            return None
        return MarkovHeadSpec(
            dspark_config.markov_rank, dspark_config.markov_head_type, markov_head_default_top_k(config)
        )
    if DFLASH_ARCHITECTURE_VERSIONS.get(architecture) == 2:
        dflash_config = parse_and_validate_dflash_config(config, expected_version=2).values
        return MarkovHeadSpec(
            dflash_config["selector_rank"], DFLASH2_MARKOV_HEAD_TYPE, dflash_config["selector_top_k"]
        )
    return None
