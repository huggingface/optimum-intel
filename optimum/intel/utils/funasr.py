#  Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import PretrainedConfig


def _is_funasr_model(
    model_name_or_path: Union[str, Path],
    all_files: list,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
) -> bool:
    """Detect FunASR models (e.g. Fun-ASR-Nano) by checking for funasr-specific artifacts.

    FunASR models are loaded via the `funasr` library (not transformers): they ship a
    `config.yaml` describing the model and a `configuration.json` declaring `model.type == "funasr"`,
    and there is no root `config.json`.
    """
    if "configuration.json" not in all_files or "config.yaml" not in all_files:
        return False
    try:
        config_path = Path(model_name_or_path)
        if config_path.is_dir():
            config_file = config_path / "configuration.json"
        else:
            config_file = hf_hub_download(
                repo_id=str(model_name_or_path), filename="configuration.json", cache_dir=cache_dir, token=token
            )
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model", {}).get("type", None) == "funasr"
    except Exception:
        return False


def _is_funasr_source(model_id, **kwargs) -> bool:
    """Check whether model_id points to a FunASR source (original repo or exported OV model)."""
    from optimum.exporters.tasks import TasksManager

    cache_dir = kwargs.get("cache_dir", HUGGINGFACE_HUB_CACHE)
    token = kwargs.get("token")
    subfolder = kwargs.get("subfolder", "")
    revision = kwargs.get("revision")
    try:
        all_files, _ = TasksManager.get_model_files(
            model_id, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
        )
    except Exception:
        all_files = []

    if _is_funasr_model(model_id, all_files, cache_dir=cache_dir, token=token):
        return True

    if "config.json" in all_files:
        try:
            cfg = PretrainedConfig.from_pretrained(
                model_id, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
            )
            return getattr(cfg, "export_model_type", None) == "fun_asr"
        except Exception:
            return False
    return False
