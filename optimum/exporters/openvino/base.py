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
