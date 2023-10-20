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
from typing import Callable, Dict, Type

from optimum.exporters.onnx import TextDecoderOnnxConfig
from optimum.exporters.tasks import TasksManager, make_backend_config_constructor_for_task

from .dummy_input_generators import ChatGLM2DummyPastKeyValuesGenerator, ChatGLN2DummyTextInputGenerator
from .normalized_configs import ChatGLM2NormalizedConfig


def create_register(overwrite_existing: bool = False):
    def wrapper(model_type: str, *supported_tasks: str) -> Callable[[Type], Type]:
        def decorator(config_cls: Type) -> Type:
            mapping = TasksManager._SUPPORTED_MODEL_TYPE.get(model_type, {})
            mapping_backend = mapping.get("openvino", {})
            for task in supported_tasks:
                normalized_task = task
                if "-with-past" in task:
                    normalized_task = task.split("-with-past")[0]
                if normalized_task not in TasksManager.get_all_tasks():
                    known_tasks = ", ".join(TasksManager.get_all_tasks())
                    raise ValueError(
                        f'The TasksManager does not know the task called "{task}", known tasks: {known_tasks}.'
                    )
                if not overwrite_existing and task in mapping_backend:
                    continue
                mapping_backend[task] = make_backend_config_constructor_for_task(config_cls, task)
            mapping["openvino"] = mapping_backend
            TasksManager._SUPPORTED_MODEL_TYPE[model_type] = mapping
            return config_cls

        return decorator

    return wrapper


register_in_tasks_manager = create_register(True)


@register_in_tasks_manager("chatglm", *["text-generation", "text-generation-with-past"])
class ChatGLM2OpenVINOConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = ChatGLM2NormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (ChatGLN2DummyTextInputGenerator, ChatGLM2DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = ChatGLM2DummyPastKeyValuesGenerator
    no_position_ids = False

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        if not self.no_position_ids and self.task == "text-generation":
            common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        """
        Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.

        Args:
            inputs_or_outputs (`Dict[str, Dict[int, str]]`):
                The mapping to fill.
            direction (`str`):
                either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                output mapping, this is important for axes naming.
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {1: "batch_size", 0: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {1: "batch_size", 0: decoder_sequence_name}
