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

from optimum.exporters.tasks import TasksManager


def init_model_configs():
    suppored_models = TasksManager._SUPPORTED_MODEL_TYPE
    for model, export_configs in suppored_models.items():
        if "onnx" not in export_configs:
            continue
        TasksManager._SUPPORTED_MODEL_TYPE[model]["openvino"] = deepcopy(
            TasksManager._SUPPORTED_MODEL_TYPE[model]["onnx"]
        )
    supported_diffusers_models = TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE
    for model, export_configs in supported_diffusers_models.items():
        if "onnx" not in export_configs:
            continue
        TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE[model]["openvino"] = deepcopy(
            TasksManager._DIFFUSERS_SUPPORTED_MODEL_TYPE[model]["onnx"]
        )
    supported_timm_models = TasksManager._TIMM_SUPPORTED_MODEL_TYPE
    for model, export_configs in supported_timm_models.items():
        if "onnx" not in export_configs:
            continue
        TasksManager._TIMM_SUPPORTED_MODEL_TYPE[model]["openvino"] = deepcopy(
            TasksManager._TIMM_SUPPORTED_MODEL_TYPE[model]["onnx"]
        )
    
    supported_sentence_transformer_models = TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE
    for model, export_configs in supported_sentence_transformer_models.items():
        if "onnx" not in export_configs:
            continue
        TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE[model]["openvino"] = deepcopy(
            TasksManager._SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE[model]["onnx"]
        )
