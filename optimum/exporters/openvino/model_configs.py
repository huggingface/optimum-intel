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

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from packaging import version
from transformers.utils import is_tf_available

from optimum.exporters.onnx.config import (
    TextDecoderOnnxConfig,
    TextDecoderWithPositionIdsOnnxConfig,
)
from optimum.exporters.onnx.model_configs import (
    CodeGenOnnxConfig,
    FalconOnnxConfig,
    GemmaOnnxConfig,
    LlamaOnnxConfig,
    M2M100OnnxConfig,
    MPTOnnxConfig,
    PhiOnnxConfig,
    T5OnnxConfig,
    UNetOnnxConfig,
    VaeDecoderOnnxConfig,
    VaeEncoderOnnxConfig,
    WhisperOnnxConfig,
)
from optimum.exporters.tasks import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES, DummyInputGenerator
from optimum.utils.input_generators import (
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    FalconDummyPastKeyValuesGenerator,
    MistralDummyPastKeyValuesGenerator,
)
from optimum.utils.normalized_config import NormalizedTextConfig

from ...intel.utils.import_utils import _transformers_version, is_transformers_version
from .model_patcher import (
    AquilaModelPatcher,
    ArcticModelPatcher,
    BaichuanModelPatcher,
    ChatGLMModelPatcher,
    CodeGenModelPatcher,
    DBRXModelPatcher,
    GemmaModelPatcher,
    InternLM2Patcher,
    InternLMModelPatcher,
    JaisModelPatcher,
    LlamaModelPatcher,
    MixtralModelPatcher,
    MPTModelPatcher,
    PersimmonModelPatcher,
    Phi3ModelPatcher,
    QwenModelPatcher,
    XverseModelPatcher,
)


def init_model_configs():
    supported_model_types = [
        "_SUPPORTED_MODEL_TYPE",
        "_DIFFUSERS_SUPPORTED_MODEL_TYPE",
        "_TIMM_SUPPORTED_MODEL_TYPE",
        "_SENTENCE_TRANSFORMERS_SUPPORTED_MODEL_TYPE",
    ]

    for supported_models_config in supported_model_types:
        supported_models = getattr(TasksManager, supported_models_config)
        for model, export_configs in supported_models.items():
            if "onnx" not in export_configs:
                continue
            onnx_config = export_configs["onnx"]
            supported_models[model]["openvino"] = deepcopy(onnx_config)

        setattr(TasksManager, supported_models_config, supported_models)


init_model_configs()


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from optimum.exporters.onnx.model_patcher import ModelPatcher

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel


register_in_tasks_manager = TasksManager.create_register("openvino", overwrite_existing=True)


@register_in_tasks_manager("baichuan", *["text-generation", "text-generation-with-past"], library_name="transformers")
class BaichaunOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads", hidden_size="hidden_size"
    )

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return BaichuanModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("qwen2", *["text-generation", "text-generation-with-past"], library_name="transformers")
class Qwen2OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager("qwen2-moe", *["text-generation", "text-generation-with-past"], library_name="transformers")
class Qwen2MoEOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager("minicpm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MiniCPMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager("stablelm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class StableLMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


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
        self.head_dim = normalized_config.kv_channels

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


@register_in_tasks_manager("chatglm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class ChatGLM2OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(vocab_size="padded_vocab_size", num_layers="num_layers")
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, ChatGLM2DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = ChatGLM2DummyPastKeyValuesGenerator

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
        if (
            self.use_past_in_inputs
            and self.PAD_ATTENTION_MASK_TO_PAST
            and self.use_cache_branch is not False
            and "attention_mask" in dummy_inputs
        ):
            # Obtain the past sequence length from the value instead of the key (Bloom). ChatGLM has seq_len in 0 dim instead of -2
            past_present_length = dummy_inputs["input_ids"].shape[1] + dummy_inputs["past_key_values"][0][1].shape[0]

            dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                dummy_inputs["attention_mask"],
                desired_length=past_present_length,
                dim=1,
                dtype=dummy_inputs["attention_mask"].dtype,
            )

        return dummy_inputs

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
            decoder_sequence_name = "past_sequence_length + present_lenght"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {1: "batch_size", 0: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {1: "batch_size", 0: decoder_sequence_name}

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return ChatGLMModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("mixtral", *["text-generation", "text-generation-with-past"], library_name="transformers")
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


@register_in_tasks_manager(
    "gemma",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GemmaOpenVINOConfig(GemmaOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return GemmaModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "llama",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class LlamaOpenVINOConfig(LlamaOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return LlamaModelPatcher(self, model, model_kwargs=model_kwargs)


class QwenDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        self.kv_channels = normalized_config.kv_channels

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (self.batch_size, self.sequence_length, self.num_attention_heads, self.kv_channels)
        past_value_shape = (self.batch_size, self.sequence_length, self.num_attention_heads, self.kv_channels)
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


@register_in_tasks_manager("qwen", *["text-generation", "text-generation-with-past"])
class QwenOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads", hidden_size="hidden_size"
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, QwenDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = QwenDummyPastKeyValuesGenerator
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
        if (
            self.use_past_in_inputs
            and self.PAD_ATTENTION_MASK_TO_PAST
            and self.use_cache_branch is not False
            and "attention_mask" in dummy_inputs
        ):
            # Obtain the past sequence length from the value instead of the key (Bloom). Qwen has seq_len in 1 dim instead of -2
            past_present_length = dummy_inputs["input_ids"].shape[1] + dummy_inputs["past_key_values"][0][1].shape[1]

            dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                dummy_inputs["attention_mask"],
                desired_length=past_present_length,
                dim=1,
                dtype=dummy_inputs["attention_mask"].dtype,
            )

        return dummy_inputs

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
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 1: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 1: decoder_sequence_name}

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return QwenModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "starcoder2", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class Starcoder2OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager("internlm2", *["text-generation", "text-generation-with-past"], library_name="transformers")
class InternLM2OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return InternLM2Patcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("orion", *["text-generation", "text-generation-with-past"], library_name="transformers")
class OrionOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager("olmo", *["text-generation", "text-generation-with-past"], library_name="transformers")
class OlmoOpenVINOConfig(LlamaOpenVINOConfig):
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager(
    "mpt", *["text-generation", "text-generation-with-past", "text-classification"], library_name="transformers"
)
class MPTOpenVINOConfig(MPTOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MPTModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "phi3",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class Phi3OpenVINOConfig(PhiOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return Phi3ModelPatcher(self, model, model_kwargs=model_kwargs)


class OVFalconDummyPastKeyValuesGenerator(FalconDummyPastKeyValuesGenerator):
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
            **kwargs,
        )
        if normalized_config.new_decoder_architecture:
            self.num_kv_heads = normalized_config.num_attention_heads
        else:
            self.num_kv_heads = normalized_config.num_kv_heads if not normalized_config.multi_query else 1

        self.head_dim = self.hidden_size // self.num_attention_heads


@register_in_tasks_manager(
    "falcon",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "question-answering",
        "text-generation",
        "text-generation-with-past",
        "token-classification",
    ],
    library_name="transformers",
)
class FalconOpenVINOConfig(FalconOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        OVFalconDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = OVFalconDummyPastKeyValuesGenerator


@register_in_tasks_manager("unet", *["semantic-segmentation"], library_name="diffusers")
class UNetOpenVINOConfig(UNetOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "timestep": {0: "steps"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }

        # TODO : add text_image, image and image_embeds
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs["text_embeds"] = {0: "batch_size"}
            common_inputs["time_ids"] = {0: "batch_size"}

        if getattr(self._normalized_config, "time_cond_proj_dim", None) is not None:
            common_inputs["timestep_cond"] = {0: "batch_size"}
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "out_sample": {0: "batch_size", 2: "height", 3: "width"},
        }


@register_in_tasks_manager("vae-encoder", *["semantic-segmentation"], library_name="diffusers")
class VaeEncoderOpenVINOConfig(VaeEncoderOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent_sample": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        }


@register_in_tasks_manager("vae-decoder", *["semantic-segmentation"], library_name="diffusers")
class VaeDecoderOpenVINOConfig(VaeDecoderOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent_sample": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
        }


@register_in_tasks_manager(
    "persimmon",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class PersimmonOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return PersimmonModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("biogpt", *["text-generation", "text-generation-with-past"], library_name="transformers")
class BioGPTOpenVINOConfig(TextDecoderOnnxConfig):
    # BioGPT does not require position_ids input.
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager(
    "gpt-neox-japanese", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class GPTNeoxJapaneseOpenVINOConfig(TextDecoderOnnxConfig):
    # GPTNeoxJapanese does not require position_ids input.
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager(
    "cohere",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class CohereOpenVINOConfig(LlamaOpenVINOConfig):
    pass


@register_in_tasks_manager("xglm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class XGLMConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="attention_heads", hidden_size="d_model"
    )


class AquilaDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
            task,
            normalized_config,
            batch_size,
            sequence_length,
            random_batch_size_range,
            random_sequence_length_range,
            **kwargs,
        )
        self.num_key_value_heads = getattr(
            normalized_config, "num_key_value_heads", normalized_config.num_attention_heads
        )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


@register_in_tasks_manager("aquila", *["text-generation", "text-generation-with-past"], library_name="transformers")
class AquilaMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, AquilaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = AquilaDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return AquilaModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("xverse", *["text-generation", "text-generation-with-past"], library_name="transformers")
class XverseMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return XverseModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("internlm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class InternLMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return InternLMModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "codegen",
    *["feature-extraction", "feature-extraction-with-past", "text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class CodeGenOpenVINOConfig(CodeGenOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return CodeGenModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "dbrx",
    *["text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class DBRXOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="n_heads",
        hidden_size="d_model",
        num_layers="n_layers",
        num_key_value_heads="attn_config.kv_n_heads",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return DBRXModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "jais",
    *["text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class JaisOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return JaisModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("arctic", *["text-generation", "text-generation-with-past"], library_name="transformers")
class ArcticOpenVINOConfig(MixtralOpenVINOConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        if is_transformers_version("<=", "4.36.0"):
            raise ValueError(
                f"Model patching for Arctic models only available for transformers >= v4.37.0, found {_transformers_version}"
            )

        return ArcticModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "whisper",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "audio-classification",
        "automatic-speech-recognition",
        "automatic-speech-recognition-with-past",
    ],
    library_name="transformers",
)
class WhisperOpenVINOConfig(WhisperOnnxConfig):
    def _create_dummy_input_generator_classes(self, **kwargs) -> List[DummyInputGenerator]:
        """
        Instantiates the dummy input generators from `self.DUMMY_INPUT_GENERATOR_CLASSES`.
        Each dummy input generator is independent, so this method instantiates the first generator, and
        forces the other generators to use the same batch size, meaning they will all produce inputs of the same batch
        size. Override this method for custom behavior.
        """
        if getattr(self, "stateful", False):
            if "encoder_sequence_length" not in kwargs:
                sequence_len = kwargs.get("sequence_length", DEFAULT_DUMMY_SHAPES["sequence_length"])
                kwargs["encoder_sequence_length"] = sequence_len + 2
        return super()._create_dummy_input_generator_classes(**kwargs)


@register_in_tasks_manager(
    "t5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class T5OpenVINOConfig(T5OnnxConfig):
    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](
            self.task,
            self._normalized_config,
            **kwargs,
        )
        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2](
            self.task,
            self._normalized_config,
            encoder_sequence_length=dummy_text_input_generator.sequence_length
            if not getattr(self, "stateful", False)
            else dummy_text_input_generator.sequence_length + 2,
            **kwargs,
        )
        dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

        return dummy_inputs_generators


@register_in_tasks_manager(
    "mt5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class MT5OpenVINOConfig(T5OpenVINOConfig):
    pass


@register_in_tasks_manager(
    "longt5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class LongT5OpenVINOConfig(T5OpenVINOConfig):
    pass


@register_in_tasks_manager(
    "m2m-100",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class M2M100OpenVINOConfig(M2M100OnnxConfig):
    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        task = "feature-extraction" if self.task != "text-generation" else "text-generation"
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1][task](
            self.task, self._normalized_config, **kwargs
        )
        if self.task != "text-generation":
            kwargs["encoder_sequence_length"] = dummy_text_input_generator.sequence_length
            if getattr(self, "stateful", False):
                kwargs["encoder_sequence_length"] = kwargs["encoder_sequence_length"] + 2

        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2][task](
            self.task, self._normalized_config, **kwargs
        )
        dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

        return dummy_inputs_generators


@register_in_tasks_manager(
    "bart",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text2text-generation",
        "text2text-generation-with-past",
        "text-classification",
        "question-answering",
    ],
    library_name="transformers",
)
class BartOpenVINOConfig(M2M100OpenVINOConfig):
    pass


@register_in_tasks_manager(
    "mbart",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text2text-generation",
        "text2text-generation-with-past",
        "text-classification",
        "question-answering",
    ],
    library_name="transformers",
)
class MBartOpenVINOConfig(M2M100OpenVINOConfig):
    pass


@register_in_tasks_manager(
    "blenderbot",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text2text-generation",
        "text2text-generation-with-past",
    ],
    library_name="transformers",
)
class BlenderbotOpenVINOConfig(M2M100OpenVINOConfig):
    pass


@register_in_tasks_manager(
    "blenderbot-small",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text2text-generation",
        "text2text-generation-with-past",
    ],
    library_name="transformers",
)
class BlenderbotSmallOpenVINOConfig(M2M100OpenVINOConfig):
    pass


@register_in_tasks_manager(
    "marian",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text2text-generation",
        "text2text-generation-with-past",
    ],
    library_name="transformers",
)
class MarianOpenVINOConfig(M2M100OpenVINOConfig):
    pass


@register_in_tasks_manager(
    "pegasus",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text2text-generation",
        "text2text-generation-with-past",
    ],
    library_name="transformers",
)
class PegasusOpenVINOConfig(M2M100OpenVINOConfig):
    pass
