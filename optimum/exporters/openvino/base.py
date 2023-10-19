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
from copy import deepcopy
from typing import Callable, Type

from optimum.exporters.tasks import TasksManager
from optimum.utils.normalized_config import NormalizedConfigManager


def init_model_configs():
    suppored_models = TasksManager._SUPPORTED_MODEL_TYPE
    for model, export_configs in suppored_models.items():
        if "onnx" not in export_configs:
            continue
        TasksManager._SUPPORTED_MODEL_TYPE[model]["openvino"] = deepcopy(
            TasksManager._SUPPORTED_MODEL_TYPE[model]["onnx"]
        )


def register_normalized_config(model_type: str) -> Callable[[Type], Type]:
    def decorator(config_cls: Type) -> Type:
        if model_type in NormalizedConfigManager._conf:
            return config_cls
        NormalizedConfigManager._conf[model_type] = config_cls
        return config_cls

    return decorator
