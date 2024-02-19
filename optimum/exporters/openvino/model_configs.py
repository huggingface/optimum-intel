#  Copyright 2024 The HuggingFace Team. All rights reserved.
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


from optimum.exporters.onnx.config import TextDecoderOnnxConfig, TextDecoderWithPositionIdsOnnxConfig
from optimum.exporters.tasks import TasksManager
from optimum.utils.input_generators import DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator
from optimum.utils.normalized_config import NormalizedTextConfig


register_in_tasks_manager = TasksManager.create_register("openvino", overwrite_existing=True)


@register_in_tasks_manager("baichuan", *["text-generation", "text-generation-with-past"])
class BaichaunOpenVINOConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads", hidden_size="hidden_size"
    )


@register_in_tasks_manager("jais", *["text-generation", "text-generation-with-past"])
class JaisOpenVINOConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="n_layer", num_attention_heads="n_head", hidden_size="n_embd"
    )


@register_in_tasks_manager("qwen2", *["text-generation", "text-generation-with-past"])
class Qwen2OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager("minicpm", *["text-generation", "text-generation-with-past"])
class MiniCPMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
