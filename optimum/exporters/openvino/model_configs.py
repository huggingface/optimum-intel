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

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from packaging import version
from transformers.utils import is_tf_available

from optimum.exporters.onnx.config import TextDecoderOnnxConfig, TextDecoderWithPositionIdsOnnxConfig
from optimum.exporters.tasks import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.utils.input_generators import (
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    MistralDummyPastKeyValuesGenerator,
)
from optimum.utils.normalized_config import NormalizedTextConfig

from .model_patcher import ChatGLMModelPatcher, MixtralModelPatcher


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from optimum.exporters.onnx.model_patcher import ModelPatcher

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel

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


class ChatGLM2DummyTextInputGenerator(DummyTextInputGenerator):
    SUPPORTED_INPUT_NAMES = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    }

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        import torch

        input = super().generate(input_name, framework, int_dtype, float_dtype)
        if input_name == "attention_mask":
            input = torch.ones(input.shape, dtype=input.dtype)
        if input_name == "position_ids":
            bs = input.shape[0]
            input = torch.range(0, input.shape[1], dtype=input.dtype).repeat(bs, 1)
        return input


class ChatGLM2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task=task,
            normalized_config=normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
        )
        self.multi_query_group_num = normalized_config.multi_query_group_num
        self.head_dim = self.hidden_size // self.num_attention_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (
            self.sequence_length,
            self.batch_size,
            self.multi_query_group_num,
            self.head_dim,
        )
        past_value_shape = (
            self.sequence_length,
            self.batch_size,
            self.multi_query_group_num,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


@register_in_tasks_manager("chatglm", *["text-generation", "text-generation-with-past"])
class ChatGLM2OpenVINOConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(vocab_size="padded_vocab_size", num_layers="num_layers")
    DUMMY_INPUT_GENERATOR_CLASSES = (ChatGLM2DummyTextInputGenerator, ChatGLM2DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = ChatGLM2DummyPastKeyValuesGenerator
    no_position_ids = False

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs.keys() if not key.startswith("past_key_values")]
        if self.use_past_in_inputs and self.use_cache_branch is not False:
            input_names.append("past_key_values")

        for input_name in input_names:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = self.overwrite_shape_and_generate_input(
                        dummy_input_gen,
                        input_name,
                        framework,
                        input_shapes=kwargs,
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                )

        # refer to https://github.com/huggingface/optimum/pull/764
        cond1 = self.use_past_in_inputs
        cond2 = self.PAD_ATTENTION_MASK_TO_PAST
        cond3 = self.use_cache_branch is not False
        cond4 = "attention_mask" in dummy_inputs
        if cond1 and cond2 and cond3 and cond4:
            # Obtain the past sequence length from the value instead of the key (Bloom).
            past_length = dummy_inputs["past_key_values"][0][1].shape[0]
            for k, v in dummy_inputs.items():
                if k not in ["attention_mask", "past_key_values"]:
                    dummy_inputs[k] = v[:, -1:]

            dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                dummy_inputs["attention_mask"],
                desired_length=past_length + 1,
                dim=1,
                dtype=dummy_inputs["attention_mask"].dtype,
            )

        return dummy_inputs

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
            inputs_or_outputs (`Dict[str, Dict[int, str]]`): The mapping to fill.
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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return ChatGLMModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("mixtral", *["text-generation", "text-generation-with-past"])
class MixtralOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    # This is because of the patching of torch.triu in AttentionMaskConverter, that exists from transformers>=4.35
    MIN_TRANSFORMERS_VERSION = version.parse("4.34.99")

    # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MixtralModelPatcher(self, model, model_kwargs=model_kwargs)
