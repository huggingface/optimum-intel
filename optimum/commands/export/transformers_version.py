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
"""Resolve and switch the Transformers version required to export a given model."""

import logging
import os
import shutil
import sys

from packaging.version import Version

from ...intel.utils.import_utils import _transformers_version


logger = logging.getLogger(__name__)

_REEXEC_GUARD = "_OPTIMUM_INTEL_TF_REEXEC"


def _export_config_bounds(model_type):
    try:
        import optimum.exporters.openvino.model_configs  # noqa: F401
        from optimum.exporters.tasks import TasksManager
    except Exception:
        return None, None

    entry = TasksManager._SUPPORTED_MODEL_TYPE.get(model_type, {})
    config_constructors = entry.get("openvino", {})
    for constructor in config_constructors.values():
        config_class = getattr(constructor, "func", constructor)
        min_version = getattr(config_class, "MIN_TRANSFORMERS_VERSION", None)
        max_version = getattr(config_class, "MAX_TRANSFORMERS_VERSION", None)
        if min_version is not None or max_version is not None:
            return _as_version_string(min_version), _as_version_string(max_version)
    return None, None


def _as_version_string(version):
    if isinstance(version, Version):
        return version.base_version
    return version


def _read_model_config(model, cache_dir):
    from transformers import PretrainedConfig

    config_dict, _ = PretrainedConfig.get_config_dict(model, cache_dir=cache_dir)
    return config_dict.get("model_type"), config_dict.get("transformers_version")


def resolve_transformers_specifier(model, requested, cache_dir):
    """Return a pip requirement specifier to switch Transformers to, or None if no switch is needed.

    `requested` is the raw value of `--transformers-version`: a concrete version, or "auto".
    """
    current = Version(Version(_transformers_version).base_version)

    if requested != "auto":
        target = Version(requested)
        if target == current:
            return None
        return f"transformers=={requested}"

    model_type, config_version = _read_model_config(model, cache_dir)
    min_version, max_version = _export_config_bounds(model_type)

    if min_version is not None or max_version is not None:
        if config_version is not None and max_version is not None and Version(config_version) > Version(max_version):
            raise ValueError(
                f"The model was saved with transformers=={config_version}, but optimum-intel supports it only up to "
                f"transformers=={max_version}. Export may not work. To attempt it anyway, pass an explicit version, "
                f'e.g. --transformers-version="{config_version}".'
            )

        in_range = (min_version is None or current >= Version(min_version)) and (
            max_version is None or current <= Version(max_version)
        )
        if in_range:
            return None

        bounds = []
        if min_version is not None:
            bounds.append(f">={min_version}")
        if max_version is not None:
            bounds.append(f"<={max_version}")
        return "transformers" + ",".join(bounds)

    if config_version is not None and current < Version(config_version):
        return f"transformers=={config_version}"

    return None


def maybe_switch_transformers_version(model, requested, cache_dir):
    """Re-exec the current command under an ephemeral `uv` environment with the required Transformers version.

    Returns without re-execing when the current environment already satisfies the requirement.
    """
    if requested is None:
        return

    specifier = resolve_transformers_specifier(model, requested, cache_dir)
    if specifier is None:
        return

    if os.environ.get(_REEXEC_GUARD):
        raise RuntimeError(
            f"Failed to switch the Transformers version: `{specifier}` is required, but the isolated environment "
            f"still resolves to transformers=={_transformers_version}. This is likely due to a `uv` version whose "
            f"overlay does not take precedence over the base environment; please upgrade `uv`."
        )

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "`--transformers-version` requires `uv` to switch the Transformers version in an isolated environment. "
            "Install it with `pip install optimum-intel[transformers-switch]` or `pip install uv`."
        )

    logger.warning(
        f"Re-running export in an isolated `uv` environment with `{specifier}` "
        f"(current: transformers=={_transformers_version})."
    )
    env = {**os.environ, _REEXEC_GUARD: "1", "VIRTUAL_ENV": sys.prefix}
    args = [uv, "run", "--active", "--no-project", "--with", specifier, "--", "optimum-cli", *sys.argv[1:]]
    os.execve(uv, args, env)
