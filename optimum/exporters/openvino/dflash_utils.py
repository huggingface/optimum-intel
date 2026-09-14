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

from optimum.intel.utils.import_utils import is_transformers_version


DFLASH_ARCHITECTURE_VERSIONS = {
    "DFlashDraftModel": 1,
    "DFlash2DraftModel": 2,
}
DFLASH_ARCHITECTURES = set(DFLASH_ARCHITECTURE_VERSIONS)
DFLASH2_RECOMMENDED_ACTIVATIONS_SCALE_FACTOR = 32.0
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


def _detect_dflash_version(config: PretrainedConfig) -> int:
    architectures = getattr(config, "architectures", None)
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
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
