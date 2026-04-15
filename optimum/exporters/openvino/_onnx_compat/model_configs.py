# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Model specific ONNX configurations."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal

from packaging import version

from optimum.exporters.openvino._onnx_compat.base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from optimum.exporters.openvino._onnx_compat.config import (
    AudioOnnxConfig,
    AudioToTextOnnxConfig,
    EncoderDecoderBaseOnnxConfig,
    TextAndVisionOnnxConfig,
    TextDecoderOnnxConfig,
    TextDecoderWithPositionIdsOnnxConfig,
    TextEncoderOnnxConfig,
    TextSeq2SeqOnnxConfig,
    VisionOnnxConfig,
)
from optimum.exporters.openvino._onnx_compat.input_generators import (
    DummyMoonshineAudioInputGenerator,
    DummySanaTransforemerTextInputGenerator,
    GPTBigCodeDummyPastKeyValuesGenerator,
)
from optimum.exporters.openvino._onnx_compat.model_patcher import (
    BigBirdPegasusModelPatcher,
    CLIPModelPatcher,
    CohereModelPatcher,
    FluxTransformerModelPatcher,
    GptOssModelPatcher,
    MetaCLIP2Patcher,
    MgpstrModelPatcher,
    MoonshineModelPatcher,
    MusicgenModelPatcher,
    Qwen3MoeModelPatcher,
    SAMModelPatcher,
    SentenceTransformersCLIPPatcher,
    SentenceTransformersTransformerPatcher,
    SpeechT5ModelPatcher,
    VitPoseModelPatcher,
)
from optimum.exporters.openvino._onnx_compat.model_patcher_dynamo import ViTForImageClassificationPatcher
from optimum.exporters.tasks import TasksManager
from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    ASTDummyAudioInputGenerator,
    BartDummyTextInputGenerator,
    BloomDummyPastKeyValuesGenerator,
    DeepSeekV3DummyPastKeyValuesGenerator,
    Dinov2DummyInputGenerator,
    DummyCodegenDecoderTextInputGenerator,
    DummyDecisionTransformerInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyEncodecInputGenerator,
    DummyFluxTransformerTextInputGenerator,
    DummyFluxTransformerVisionInputGenerator,
    DummyInputGenerator,
    DummyIntGenerator,
    DummyPastKeyValuesGenerator,
    DummyPatchTSTInputGenerator,
    DummyPix2StructInputGenerator,
    DummyPointsGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummySpeechT5InputGenerator,
    DummyTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyTransformerTextInputGenerator,
    DummyTransformerTimestepInputGenerator,
    DummyTransformerVisionInputGenerator,
    DummyVisionEmbeddingsGenerator,
    DummyVisionEncoderDecoderPastKeyValuesGenerator,
    DummyVisionInputGenerator,
    DummyXPathSeqInputGenerator,
    FalconDummyPastKeyValuesGenerator,
    GemmaDummyPastKeyValuesGenerator,
    LongformerDummyTextInputGenerator,
    MCTCTDummyAudioInputGenerator,
    MistralDummyPastKeyValuesGenerator,
    NormalizedConfig,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedTextConfigWithGQA,
    NormalizedTimeSeriesForecastingConfig,
    NormalizedVisionConfig,
    PerceiverDummyInputGenerator,
    Speech2TextDummyAudioInputGenerator,
    T5DummySeq2SeqPastKeyValuesGenerator,
    VitPoseDummyInputGenerator,
    is_diffusers_version,
    is_transformers_version,
    logging,
)
from optimum.utils.normalized_config import NormalizedConfigManager


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.get_logger(__name__)


COMMON_TEXT_TASKS = [
    "feature-extraction",
    "fill-mask",
    "multiple-choice",
    "question-answering",
    "text-classification",
    "token-classification",
]


COMMON_TEXT_GENERATION_TASKS = [
    "feature-extraction",
    "feature-extraction-with-past",
    "text-generation",
    "text-generation-with-past",
]

COMMON_TEXT2TEXT_GENERATION_TASKS = [
    *COMMON_TEXT_GENERATION_TASKS,
    "text2text-generation",
    "text2text-generation-with-past",
]


register_tasks_manager_onnx = TasksManager.create_register("onnx")


@register_tasks_manager_onnx("bert", *COMMON_TEXT_TASKS)
class BertOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "token_type_ids": dynamic_axis,
        }


@register_tasks_manager_onnx("visual_bert", *["feature-extraction"])
class VisualBertOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyVisionInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "visual_embeds": {0: "batch_size", 1: "visual_seq_length", 2: "visual_embedding_dim"},
            "visual_attention_mask": {0: "batch_size", 1: "visual_seq_length"},
            "visual_token_type_ids": {0: "batch_size", 1: "visual_seq_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length + visual_seq_length"},
        }


@register_tasks_manager_onnx("albert", *COMMON_TEXT_TASKS)
class AlbertOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("convbert", *COMMON_TEXT_TASKS)
class ConvBertOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("electra", *COMMON_TEXT_TASKS)
class ElectraOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("roformer", *COMMON_TEXT_TASKS)
class RoFormerOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("squeezebert", *COMMON_TEXT_TASKS)
class SqueezeBertOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("nystromformer", *COMMON_TEXT_TASKS)
class NystromformerOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("xlm", *COMMON_TEXT_TASKS)
class XLMOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("splinter", *["feature-extraction", "question-answering"])
class SplinterOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("rembert", *COMMON_TEXT_TASKS)
class RemBertOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("longformer", *COMMON_TEXT_TASKS)
class LongformerOnnxConfig(BertOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (LongformerDummyTextInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        inputs = super().inputs

        inputs["global_attention_mask"] = inputs["attention_mask"]

        return inputs


@register_tasks_manager_onnx("megatron-bert", *COMMON_TEXT_TASKS)
class MegatronBertOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("distilbert", *COMMON_TEXT_TASKS)
class DistilBertOnnxConfig(BertOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}


@register_tasks_manager_onnx(
    "modernbert",
    *[
        "feature-extraction",
        "fill-mask",
        "text-classification",
        "token-classification",
    ],
)
class ModernBertOnnxConfig(DistilBertOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.48.0")


@register_tasks_manager_onnx("mpnet", *COMMON_TEXT_TASKS)
class MPNetOnnxConfig(DistilBertOnnxConfig):
    pass


@register_tasks_manager_onnx("roberta", *COMMON_TEXT_TASKS)
class RobertaOnnxConfig(DistilBertOnnxConfig):
    pass


@register_tasks_manager_onnx("camembert", *COMMON_TEXT_TASKS)
class CamembertOnnxConfig(DistilBertOnnxConfig):
    pass


@register_tasks_manager_onnx("flaubert", *COMMON_TEXT_TASKS)
class FlaubertOnnxConfig(BertOnnxConfig):
    pass


@register_tasks_manager_onnx("ibert", *COMMON_TEXT_TASKS)
class IBertOnnxConfig(DistilBertOnnxConfig):
    pass


@register_tasks_manager_onnx("xlm-roberta", *COMMON_TEXT_TASKS)
class XLMRobertaOnnxConfig(DistilBertOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "deberta",
    *["feature-extraction", "fill-mask", "text-classification", "token-classification", "question-answering"],
)
class DebertaOnnxConfig(BertOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            common_inputs.pop("token_type_ids")
        return common_inputs


@register_tasks_manager_onnx(
    "markuplm", *["feature-extraction", "text-classification", "token-classification", "question-answering"]
)
class MarkupLMOnnxConfig(BertOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyXPathSeqInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        xpath_dynamic_axis = {0: "batch_size", 1: "sequence_length", 2: "max_depth"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "token_type_ids": dynamic_axis,
            "xpath_subs_seq": xpath_dynamic_axis,
            "xpath_tags_seq": xpath_dynamic_axis,
        }


@register_tasks_manager_onnx("deberta-v2", *COMMON_TEXT_TASKS)
class DebertaV2OnnxConfig(DebertaOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "esm", *["feature-extraction", "fill-mask", "text-classification", "token-classification"]
)
class EsmOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
        }


@register_tasks_manager_onnx("gpt2", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "token-classification"])
class GPT2OnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")


@register_tasks_manager_onnx("gptj", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "question-answering"])
class GPTJOnnxConfig(GPT2OnnxConfig):
    pass


@register_tasks_manager_onnx("codegen", *COMMON_TEXT_GENERATION_TASKS)
class CodeGenOnnxConfig(GPT2OnnxConfig):
    pass


@register_tasks_manager_onnx("imagegpt", *["feature-extraction", "image-classification"])
class ImageGPTOnnxConfig(GPT2OnnxConfig):
    pass


@register_tasks_manager_onnx("decision_transformer", *["feature-extraction", "reinforcement-learning"])
class DecisionTransformerOnnxConfig(OnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyDecisionTransformerInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "states": {0: "batch_size", 1: "sequence_length"},
            "actions": {0: "batch_size", 1: "sequence_length"},
            "timesteps": {0: "batch_size", 1: "sequence_length"},
            "returns_to_go": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "state_preds": {0: "batch_size", 1: "sequence_length"},
            "action_preds": {0: "batch_size", 1: "sequence_length"},
            "return_preds": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }


@register_tasks_manager_onnx("gpt_neo", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class GPTNeoOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_attention_heads="num_heads")


@register_tasks_manager_onnx("gpt_neox", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class GPTNeoXOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_tasks_manager_onnx("opt", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "question-answering"])
class OPTOnnxConfig(
    TextDecoderWithPositionIdsOnnxConfig if is_transformers_version(">=", "4.46.0") else TextDecoderOnnxConfig
):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_tasks_manager_onnx("llama", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class LlamaOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_tasks_manager_onnx("arcee", *COMMON_TEXT_GENERATION_TASKS)
class ArceeOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.53.0")
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_tasks_manager_onnx("deepseek_v3", *COMMON_TEXT_GENERATION_TASKS)
class DeepSeekV3OnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.51.0")
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DeepSeekV3DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DeepSeekV3DummyPastKeyValuesGenerator


@register_tasks_manager_onnx("cohere", *COMMON_TEXT_GENERATION_TASKS)
class CohereOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.38.0")
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = CohereModelPatcher


@register_tasks_manager_onnx("glm", *COMMON_TEXT_GENERATION_TASKS)
class GLMOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.46.0")


@register_tasks_manager_onnx("helium", *COMMON_TEXT_GENERATION_TASKS)
class HeliumOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.49.0")


@register_tasks_manager_onnx("smollm3", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class SmolLM3OnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.53.0")


@register_tasks_manager_onnx("stablelm", *COMMON_TEXT_GENERATION_TASKS)
class StableLMOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.38.0")


@register_tasks_manager_onnx("olmo", *COMMON_TEXT_GENERATION_TASKS)
class OlmoOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.40.0")


@register_tasks_manager_onnx("olmo2", *COMMON_TEXT_GENERATION_TASKS)
class Olmo2OnnxConfig(OlmoOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.47.0")


@register_tasks_manager_onnx("qwen2", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "token-classification"])
class Qwen2OnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
    MIN_TRANSFORMERS_VERSION = version.parse("4.37.0")


@register_tasks_manager_onnx("qwen3", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class Qwen3OnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
    MIN_TRANSFORMERS_VERSION = version.parse("4.51.0")


@register_tasks_manager_onnx(
    "qwen3_moe", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "token-classification"]
)
class Qwen3MoeOnnxConfig(LlamaOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
    MIN_TRANSFORMERS_VERSION = version.parse("4.51.0")
    _MODEL_PATCHER = Qwen3MoeModelPatcher


@register_tasks_manager_onnx("gemma", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class GemmaOnnxConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    MIN_TRANSFORMERS_VERSION = version.parse("4.38.0")


@register_tasks_manager_onnx("gemma2", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class Gemma2OnnxConfig(GemmaOnnxConfig):
    # Gemma 2 was added in transformers v4.42 using HybridCache
    # DynamicCache support was added since v4.53
    MIN_TRANSFORMERS_VERSION = version.parse("4.53.0")


@register_tasks_manager_onnx("gemma3_text", *COMMON_TEXT_GENERATION_TASKS)
class Gemma3TextOnnxConfig(GemmaOnnxConfig):
    # Gemma 3 was added in transformers v4.50 using HybridCache
    # DynamicCache support was added since v4.53
    MIN_TRANSFORMERS_VERSION = version.parse("4.53.0")


# we still don't support gemma3 for multimodal feature-extraction(-with-past) and image-text-to-text(-with-past) tasks
@register_tasks_manager_onnx("gemma3", *COMMON_TEXT_GENERATION_TASKS, "text-classification")
class Gemma3OnnxConfig(GemmaOnnxConfig):
    # Gemma 3 was added in transformers v4.50 using HybridCache
    # DynamicCache support was added since v4.53
    MIN_TRANSFORMERS_VERSION = version.parse("4.53.0")

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config.text_config, **kwargs)


@register_tasks_manager_onnx("gpt_oss", *COMMON_TEXT_GENERATION_TASKS)
class GPTOssOnnxConfig(GemmaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.55.0")
    _MODEL_PATCHER = GptOssModelPatcher


@register_tasks_manager_onnx("nemotron", *COMMON_TEXT_GENERATION_TASKS)
class NemotronOnnxConfig(GemmaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.48.0")  # More stable version than 4.44.0
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA


@register_tasks_manager_onnx("granite", *COMMON_TEXT_GENERATION_TASKS)
class GraniteOnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.45.0")


@register_tasks_manager_onnx("phi", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class PhiOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = version.parse("4.36.0")


@register_tasks_manager_onnx("phi3", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class Phi3OnnxConfig(PhiOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
    MIN_TRANSFORMERS_VERSION = version.parse("4.41.0")


@register_tasks_manager_onnx("internlm2", *["text-generation", "text-generation-with-past"])
class InternLM2OnnxConfig(LlamaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.41.0")


@register_tasks_manager_onnx("mistral", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"])
class MistralOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator


@register_tasks_manager_onnx("mpt", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "token-classification"])
class MPTOnnxConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="n_heads", hidden_size="d_model", num_layers="n_layers"
    )


@register_tasks_manager_onnx("bloom", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "token-classification"])
class BloomOnnxConfig(TextDecoderOnnxConfig):
    # Bloom does not require position_ids input.
    MIN_TRANSFORMERS_VERSION = version.parse("4.36.0")
    DUMMY_PKV_GENERATOR_CLASS = BloomDummyPastKeyValuesGenerator
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, BloomDummyPastKeyValuesGenerator)
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        if is_transformers_version(">=", "4.44"):
            super().add_past_key_values(inputs_or_outputs, direction)
        else:
            if direction not in ["inputs", "outputs"]:
                raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

            if direction == "inputs":
                decoder_sequence_name = "past_sequence_length"
                name = "past_key_values"
            else:
                decoder_sequence_name = "past_sequence_length + sequence_length"
                name = "present"

            for i in range(self._normalized_config.num_layers):
                inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size * num_heads", 2: decoder_sequence_name}
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size * num_heads", 1: decoder_sequence_name}


@register_tasks_manager_onnx(
    "gpt_bigcode", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "token-classification"]
)
class GPTBigCodeOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GPTBigCodeDummyPastKeyValuesGenerator)
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("gpt_bigcode")
    DUMMY_PKV_GENERATOR_CLASS = GPTBigCodeDummyPastKeyValuesGenerator

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        if is_transformers_version(">=", "4.54"):
            super().add_past_key_values(inputs_or_outputs, direction)
        else:
            if direction not in ["inputs", "outputs"]:
                raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

            if direction == "inputs":
                decoder_sequence_name = "past_sequence_length"
                name = "past_key_values"
            else:
                decoder_sequence_name = "past_sequence_length + sequence_length"
                name = "present"

            if self._normalized_config.multi_query:
                decoder_sequence_dim = 1
            else:
                decoder_sequence_dim = 2

            for i in range(self._normalized_config.num_layers):
                inputs_or_outputs[f"{name}.{i}.key_value"] = {
                    0: "batch_size",
                    decoder_sequence_dim: decoder_sequence_name,
                }

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        if is_transformers_version(">=", "4.54"):
            super().flatten_past_key_values(flattened_output, name, idx, t)
        else:
            flattened_output[f"{name}.{idx}.key_value"] = t


@register_tasks_manager_onnx("falcon", *[*COMMON_TEXT_GENERATION_TASKS, "question-answering", "token-classification"])
class FalconOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, FalconDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = FalconDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = super().inputs

        if self._config.alibi:
            common_inputs.pop("position_ids", None)

        return common_inputs


@register_tasks_manager_onnx(
    "t5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
)
class T5OnnxConfig(TextSeq2SeqOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        *TextSeq2SeqOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[:-1],
        T5DummySeq2SeqPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = T5DummySeq2SeqPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="d_model",
        num_attention_heads="num_heads",
        encoder_num_layers="num_layers",
        decoder_num_layers="num_decoder_layers",
        key_value_dim="d_kv",
        allow_new=True,
    )


@register_tasks_manager_onnx(
    "mt5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
)
class MT5OnnxConfig(T5OnnxConfig):
    pass


@register_tasks_manager_onnx(
    "longt5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
)
class LongT5OnnxConfig(T5OnnxConfig):
    pass


@register_tasks_manager_onnx(
    "m2m_100",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
)
class M2M100OnnxConfig(TextSeq2SeqOnnxConfig):
    PAD_ATTENTION_MASK_TO_PAST = True

    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",  # Used for the text-generation task past key values input generation.
        encoder_num_attention_heads="encoder_attention_heads",
        decoder_num_attention_heads="decoder_attention_heads",
        eos_token_id="eos_token_id",
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        BartDummyTextInputGenerator,
        {
            "feature-extraction": DummySeq2SeqDecoderTextInputGenerator,
            "text-generation": DummyDecoderTextInputGenerator,
        },
        {
            "feature-extraction": DummySeq2SeqPastKeyValuesGenerator,
            "text-generation": DummyPastKeyValuesGenerator,
        },
    )

    def _create_dummy_input_generator_classes(self, **kwargs) -> list[DummyInputGenerator]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        task = "feature-extraction" if self.task != "text-generation" else "text-generation"
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1][task](
            self.task, self._normalized_config, **kwargs
        )
        if self.task != "text-generation":
            kwargs["encoder_sequence_length"] = dummy_text_input_generator.sequence_length

        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2][task](
            self.task, self._normalized_config, **kwargs
        )
        dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

        return dummy_inputs_generators

    @property
    def inputs_for_default_and_seq2seq_lm(self):
        return super().inputs

    @property
    def inputs_for_causal_lm(self):
        if self.use_past_in_inputs:
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "past_sequence_length + sequence_length"},
            }
            for i in range(self._normalized_config.decoder_num_layers):
                common_inputs[f"past_key_values.{i}.key"] = {
                    0: "batch_size",
                    2: "past_sequence_length",
                }
                common_inputs[f"past_key_values.{i}.value"] = {
                    0: "batch_size",
                    2: "past_sequence_length",
                }
        else:
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }

        return common_inputs

    @property
    def inputs_for_other_tasks(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        inputs_properties = {
            "feature-extraction": self.inputs_for_default_and_seq2seq_lm,
            "text2text-generation": self.inputs_for_default_and_seq2seq_lm,
            "text-generation": self.inputs_for_causal_lm,
            "other": self.inputs_for_other_tasks,
        }
        return inputs_properties.get(self.task, inputs_properties["other"])

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.task in ["feature-extraction", "text2text-generation"]:
            common_outputs = super().outputs
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs
            if self.use_past:
                # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
                for i in range(
                    self._normalized_config.encoder_num_layers
                    if self.task != "text-generation"
                    else self._normalized_config.decoder_num_layers
                ):
                    common_outputs[f"present.{i}.key"] = {0: "batch_size", 2: "past_sequence_length + sequence_length"}
                    common_outputs[f"present.{i}.value"] = {
                        0: "batch_size",
                        2: "past_sequence_length + sequence_length",
                    }
        return common_outputs

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        if self.task in ["feature-extraction", "text2text-generation"]:
            flattened_output = super().flatten_past_key_values(flattened_output, name, idx, t)
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self).flatten_past_key_values(
                flattened_output, name, idx, t
            )


@register_tasks_manager_onnx(
    "bart", *[*COMMON_TEXT2TEXT_GENERATION_TASKS, "text-classification", "question-answering"]
)
class BartOnnxConfig(M2M100OnnxConfig):
    pass


@register_tasks_manager_onnx(
    "mbart", *[*COMMON_TEXT2TEXT_GENERATION_TASKS, "text-classification", "question-answering"]
)
class MBartOnnxConfig(BartOnnxConfig):
    pass


@register_tasks_manager_onnx("blenderbot", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class BlenderbotOnnxConfig(BartOnnxConfig):
    pass


@register_tasks_manager_onnx("blenderbot-small", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class BlenderbotSmallOnnxConfig(BartOnnxConfig):
    pass


@register_tasks_manager_onnx("big_bird", *COMMON_TEXT_TASKS)
class BigBirdOnnxConfig(DistilBertOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "bigbird_pegasus", *[*COMMON_TEXT2TEXT_GENERATION_TASKS, "text-classification", "question-answering"]
)
class BigBirdPegasusOnnxConfig(BartOnnxConfig):
    _MODEL_PATCHER = BigBirdPegasusModelPatcher


@register_tasks_manager_onnx("pegasus", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class PegasusOnnxConfig(BartOnnxConfig):
    pass


@register_tasks_manager_onnx("marian", *COMMON_TEXT2TEXT_GENERATION_TASKS)
class MarianOnnxConfig(BartOnnxConfig):
    pass


@register_tasks_manager_onnx("vit", *["feature-extraction", "image-classification", "masked-im"])
class ViTOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    _MODEL_PATCHER = ViTForImageClassificationPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super().outputs

        if self.task == "feature-extraction":
            common_outputs["last_hidden_state"] = {0: "batch_size"}

        return common_outputs


@register_tasks_manager_onnx("vitpose", *["keypoint-detection"])
class VitPoseOnnxConfig(ViTOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (VitPoseDummyInputGenerator,)

    _MODEL_PATCHER = VitPoseModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"pixel_values": {0: "batch_size"}}


@register_tasks_manager_onnx("cvt", *["feature-extraction", "image-classification"])
class CvTOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("levit", *["feature-extraction", "image-classification"])
class LevitOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("deit", *["feature-extraction", "image-classification", "masked-im"])
class DeiTOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("beit", *["feature-extraction", "image-classification"])
class BeitOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("convnext", *["feature-extraction", "image-classification"])
class ConvNextOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("convnextv2", *["feature-extraction", "image-classification"])
class ConvNextV2OnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("hiera", *["feature-extraction", "image-classification"])
class HieraOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("pvt", *["feature-extraction", "image-classification"])
class PvtOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("vit_mae", *["feature-extraction"])
class VitMAEOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("vit_msn", *["feature-extraction", "image-classification"])
class VitMSNOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("dinov2", *["feature-extraction", "image-classification"])
class Dinov2OnnxConfig(ViTOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (Dinov2DummyInputGenerator,)


@register_tasks_manager_onnx("mobilevit", *["feature-extraction", "image-classification", "image-segmentation"])
class MobileViTOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("regnet", *["feature-extraction", "image-classification"])
class RegNetOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("resnet", *["feature-extraction", "image-classification"])
class ResNetOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("detr", *["feature-extraction", "object-detection", "image-segmentation"])
class DetrOnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.task == "image-segmentation":
            return {
                "logits": {0: "batch_size", 1: "num_queries"},
                "pred_masks": {0: "batch_size", 1: "num_queries"},
            }
        else:
            return super().outputs


@register_tasks_manager_onnx("table-transformer", *["feature-extraction", "object-detection"])
class TableTransformerOnnxConfig(DetrOnnxConfig):
    pass


@register_tasks_manager_onnx("yolos", *["feature-extraction", "object-detection"])
class YolosOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("swin", *["feature-extraction", "image-classification", "masked-im"])
class SwinOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("swinv2", *["feature-extraction", "image-classification", "masked-im"])
class SwinV2OnnxConfig(SwinOnnxConfig):
    pass


@register_tasks_manager_onnx("swin2sr", *["feature-extraction", "image-to-image"])
class Swin2srOnnxConfig(SwinOnnxConfig):
    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        outputs = super().outputs

        if self.task == "image-to-image":
            scale_factor = self._config.upscale
            outputs["reconstruction"] = {
                0: "batch_size",
                1: "num_channels",
                2: f"height  * {scale_factor}",
                3: f"width * {scale_factor}",
            }

        return outputs


@register_tasks_manager_onnx(
    "dpt", *["feature-extraction", "depth-estimation", "image-segmentation", "semantic-segmentation"]
)
class DptOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("glpn", *["feature-extraction", "depth-estimation"])
class GlpnOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("poolformer", *["feature-extraction", "image-classification"])
class PoolFormerOnnxConfig(ViTOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig


@register_tasks_manager_onnx(
    "segformer", *["feature-extraction", "image-classification", "image-segmentation", "semantic-segmentation"]
)
class SegformerOnnxConfig(YolosOnnxConfig):
    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        outputs = super().outputs

        if self.task == "image-segmentation":
            outputs["logits"] = {0: "batch_size"}

        return outputs


@register_tasks_manager_onnx("mobilenet_v1", *["feature-extraction", "image-classification"])
class MobileNetV1OnnxConfig(ViTOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"pixel_values": {0: "batch_size"}}


@register_tasks_manager_onnx("mobilenet_v2", *["feature-extraction", "image-classification"])
class MobileNetV2OnnxConfig(MobileNetV1OnnxConfig):
    pass


@register_tasks_manager_onnx("maskformer", *["feature-extraction", "image-segmentation"])
class MaskFormerOnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.task == "image-segmentation":
            return {
                "class_queries_logits": {0: "batch_size", 1: "num_queries"},
                "masks_queries_logits": {0: "batch_size", 1: "num_queries", 2: "height", 3: "width"},
            }
        else:
            return super().outputs

    @property
    def torch_to_onnx_output_map(self) -> dict[str, str]:
        return {
            "transformer_decoder_last_hidden_state": "last_hidden_state",
        }


@register_tasks_manager_onnx("donut-swin", *["feature-extraction"])
class DonutSwinOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("default-timm-config", *["image-classification"], library_name="timm")
class TimmDefaultOnnxConfig(ViTOnnxConfig):
    def rename_ambiguous_inputs(self, inputs):
        #  The input name in the model signature is `x, hence the export input name is updated.
        model_inputs = {}
        model_inputs["x"] = inputs["pixel_values"]

        return model_inputs

    @property
    def torch_to_onnx_input_map(self) -> dict[str, str]:
        return {"x": "pixel_values"}


@register_tasks_manager_onnx("mgp-str", *["feature-extraction"])
class MgpstrOnnxConfig(ViTOnnxConfig):
    _MODEL_PATCHER = MgpstrModelPatcher

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "char_logits": {0: "batch_size"},
            "bpe_logits": {0: "batch_size"},
            "wp_logits": {0: "batch_size"},
        }


@register_tasks_manager_onnx("efficientnet", *["feature-extraction", "image-classification"])
class EfficientNetOnnxConfig(ViTOnnxConfig):
    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super().outputs

        if self.task == "image-classification":
            common_outputs["logits"] = {0: "batch_size", 1: "num_classes"}

        return common_outputs


@register_tasks_manager_onnx(
    "transformer", *["feature-extraction", "sentence-similarity"], library_name="sentence_transformers"
)
class SentenceTransformersTransformerOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = SentenceTransformersTransformerPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "token_embeddings": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"},
        }


class CLIPNormalizedConfig(NormalizedTextAndVisionConfig):
    TEXT_CONFIG = "text_config"
    VISION_CONFIG = "vision_config"


@register_tasks_manager_onnx("clip_vision_model", *["feature-extraction"])
class CLIPVisionModelOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super().outputs
        common_outputs["last_hidden_state"] = {0: "batch_size"}
        common_outputs["pooler_output"] = {0: "batch_size"}

        return common_outputs


@register_tasks_manager_onnx("clip", *["feature-extraction", "zero-shot-image-classification", "image-classification"])
class CLIPOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        inputs = {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}

        if self.task in ["feature-extraction", "zero-shot-image-classification"]:
            inputs.update(
                {
                    "input_ids": {0: "text_batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
                }
            )

        return inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.task in ["feature-extraction", "zero-shot-image-classification"]:
            return {
                "logits_per_image": {0: "image_batch_size", 1: "text_batch_size"},
                "logits_per_text": {0: "text_batch_size", 1: "image_batch_size"},
                "text_embeds": {0: "text_batch_size"},
                "image_embeds": {0: "image_batch_size"},
            }
        else:
            return super().outputs


@register_tasks_manager_onnx(
    "clip", *["feature-extraction", "sentence-similarity"], library_name="sentence_transformers"
)
class SentenceTransformersCLIPOnnxConfig(CLIPOnnxConfig):
    _MODEL_PATCHER = SentenceTransformersCLIPPatcher

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "text_embeds": {0: "text_batch_size"},
            "image_embeds": {0: "image_batch_size"},
        }


@register_tasks_manager_onnx("clip-text-with-projection", *["feature-extraction"], library_name="diffusers")
class CLIPTextWithProjectionOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        num_layers="num_hidden_layers",
        allow_new=True,
    )
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = {
            "text_embeds": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }
        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs


@register_tasks_manager_onnx("clip-text", *["feature-extraction"], library_name="diffusers")
class CLIPTextOnnxConfig(CLIPTextWithProjectionOnnxConfig):
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }

        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs


@register_tasks_manager_onnx(
    "metaclip_2",
    *["feature-extraction", "zero-shot-image-classification", "image-classification"],
    library_name="transformers",
)
class MetaCLIP2OnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    MIN_TRANSFORMERS_VERSION = version.parse("4.56.2")
    VARIANTS = {  # noqa: RUF012
        "monolith": "All the MetaClip2 model components are exported as a single model.onnx.",
        "split": "The vision model is exported as a separate vision_model.onnx, and the text_model is exported as text_model.onnx",
    }
    DEFAULT_VARIANT = "monolith"
    _MODEL_PATCHER = MetaCLIP2Patcher

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        variant: str = "monolith",
        vision_model: bool | None = None,
        preprocessors: list[Any] | None = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self.variant = variant
        self.vision_model = vision_model

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.variant == "monolith":
            inputs = {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}
            if self.task in ["feature-extraction", "zero-shot-image-classification"]:
                inputs.update(
                    {
                        "input_ids": {0: "text_batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
                    }
                )
        else:
            if self.vision_model:
                inputs = {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}
            else:
                inputs = {
                    "input_ids": {0: "text_batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
                }
        return inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.variant == "split":
            if self.vision_model:
                return {
                    "image_embeds": {0: "batch_size"},
                }
            else:
                return {
                    "text_embeds": {0: "batch_size"},
                }
        else:
            if self.task in ["feature-extraction", "zero-shot-image-classification"]:
                return {
                    "logits_per_image": {0: "image_batch_size", 1: "text_batch_size"},
                    "logits_per_text": {0: "text_batch_size", 1: "image_batch_size"},
                    "text_embeds": {0: "text_batch_size"},
                    "image_embeds": {0: "image_batch_size"},
                }
            else:
                return super().outputs


class SiglipNormalizedConfig(CLIPNormalizedConfig):
    pass


@register_tasks_manager_onnx("chinese_clip", *["feature-extraction", "zero-shot-image-classification"])
class ChineseCLIPOnnxConfig(CLIPOnnxConfig):
    pass


@register_tasks_manager_onnx("siglip", *["feature-extraction", "zero-shot-image-classification"])
class SiglipOnnxConfig(CLIPOnnxConfig):
    NORMALIZED_CONFIG_CLASS = SiglipNormalizedConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size", 1: "sequence_length"},
            "pixel_values": {0: "image_batch_size", 1: "num_channels", 2: "height", 3: "width"},
            # NOTE: No attention_mask
        }


@register_tasks_manager_onnx("siglip-text-with-projection", *["feature-extraction"])
class SiglipTextWithProjectionOnnxConfig(CLIPTextWithProjectionOnnxConfig):
    pass


@register_tasks_manager_onnx("siglip-text", *["feature-extraction"])
class SiglipTextOnnxConfig(CLIPTextOnnxConfig):
    pass


@register_tasks_manager_onnx("siglip_vision_model", *["feature-extraction"])
class SiglipVisionModelOnnxConfig(CLIPVisionModelOnnxConfig):
    pass


@register_tasks_manager_onnx("unet-2d-condition", *["semantic-segmentation"], library_name="diffusers")
class UNetOnnxConfig(VisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        DummyTimestepInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "timestep": {},  # a scalar with no dimension
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }

        # TODO : add addition_embed_type == text_image, image and image_embeds
        # https://github.com/huggingface/diffusers/blob/9366c8f84bfe47099ff047272661786ebb54721d/src/diffusers/models/unets/unet_2d_condition.py#L671
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs["text_embeds"] = {0: "batch_size"}
            common_inputs["time_ids"] = {0: "batch_size"}

        if getattr(self._normalized_config, "time_cond_proj_dim", None) is not None:
            common_inputs["timestep_cond"] = {0: "batch_size"}

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "out_sample": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def torch_to_onnx_output_map(self) -> dict[str, str]:
        return {
            "sample": "out_sample",
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            dummy_inputs["added_cond_kwargs"] = {
                "text_embeds": dummy_inputs.pop("text_embeds"),
                "time_ids": dummy_inputs.pop("time_ids"),
            }

        return dummy_inputs

    def ordered_inputs(self, model) -> dict[str, dict[int, str]]:
        inputs = super().ordered_inputs(model=model)
        # to fix mismatch between model forward signature and expected inputs
        # a dictionary of additional embeddings `added_cond_kwargs` is expected depending on config.addition_embed_type
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            inputs["text_embeds"] = self.inputs["text_embeds"]
            inputs["time_ids"] = self.inputs["time_ids"]

        return inputs


@register_tasks_manager_onnx("vae-encoder", *["semantic-segmentation"], library_name="diffusers")
class VaeEncoderOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels", image_size="sample_size", allow_new=True
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "sample_height", 3: "sample_width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        down_sampling_factor = 2 ** (len(self._normalized_config.down_block_types) - 1)

        return {
            "latent_parameters": {
                0: "batch_size",
                2: f"sample_height / {down_sampling_factor}",
                3: f"sample_width / {down_sampling_factor}",
            },
        }


@register_tasks_manager_onnx("vae-decoder", *["semantic-segmentation"], library_name="diffusers")
class VaeDecoderOnnxConfig(VisionOnnxConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_channels="latent_channels", allow_new=True)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "latent_sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        up_sampling_factor = 2 ** (len(self._normalized_config.up_block_types) - 1)

        return {
            "sample": {
                0: "batch_size",
                2: f"latent_height * {up_sampling_factor}",
                3: f"latent_width * {up_sampling_factor}",
            },
        }


@register_tasks_manager_onnx("t5-encoder", *["feature-extraction"], library_name="diffusers")
class T5EncoderOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self):
        return {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }


@register_tasks_manager_onnx("sd3-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class SD3TransformerOnnxConfig(VisionOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestepInputGenerator,
        DummyTransformerVisionInputGenerator,
        DummyTransformerTextInputGenerator,
    )

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        vocab_size="attention_head_dim",
        hidden_size="joint_attention_dim",
        projection_size="pooled_projection_dim",
        allow_new=True,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {
            "hidden_states": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "step"},
        }

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "out_hidden_states": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def torch_to_onnx_output_map(self) -> dict[str, str]:
        return {
            "sample": "out_hidden_states",
        }


@register_tasks_manager_onnx("flux-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class FluxTransformerOnnxConfig(SD3TransformerOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestepInputGenerator,
        DummyFluxTransformerVisionInputGenerator,
        DummyFluxTransformerTextInputGenerator,
    )
    _MODEL_PATCHER = FluxTransformerModelPatcher

    @property
    def inputs(self):
        common_inputs = super().inputs
        common_inputs["hidden_states"] = {0: "batch_size", 1: "packed_height_width"}
        common_inputs["txt_ids"] = (
            {0: "sequence_length"} if is_diffusers_version(">=", "0.31.0") else {0: "batch_size", 1: "sequence_length"}
        )
        common_inputs["img_ids"] = (
            {0: "packed_height_width"}
            if is_diffusers_version(">=", "0.31.0")
            else {0: "batch_size", 1: "packed_height_width"}
        )

        if getattr(self._normalized_config, "guidance_embeds", False):
            common_inputs["guidance"] = {0: "batch_size"}

        return common_inputs

    @property
    def outputs(self):
        return {
            "out_hidden_states": {0: "batch_size", 1: "packed_height_width"},
        }


@register_tasks_manager_onnx("groupvit", *["feature-extraction"])
class GroupViTOnnxConfig(CLIPOnnxConfig):
    pass


@register_tasks_manager_onnx("owlvit", *["feature-extraction", "zero-shot-object-detection"])
class OwlViTOnnxConfig(CLIPOnnxConfig):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        preprocessors: list[Any] | None = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        if task == "zero-shot-object-detection":
            logger.warning(
                "The batch size of this model will not be dynamic because non-maximum suppression is performed. "
                "Make sure to export the model with the same batch size as the one you will use at inference "
                "with `--batch_size N`."
            )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        inputs = {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}

        if self.task in ["feature-extraction", "zero-shot-object-detection"]:
            inputs.update(
                {
                    "input_ids": {0: "text_batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
                }
            )

        return inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        outputs = {}
        if self.task == "feature-extraction":
            outputs["logits_per_image"] = {0: "image_batch_size", 1: "text_batch_size"}
            outputs["logits_per_text"] = {0: "text_batch_size", 1: "image_batch_size"}
        elif self.task == "zero-shot-object-detection":
            outputs["logits"] = {0: "image_batch_size", 2: "num_queries"}
            outputs["pred_boxes"] = {0: "image_batch_size", 1: "num_boxes"}

        outputs["text_embeds"] = {0: "text_batch_size", 1: "max_text_queries"}
        outputs["image_embeds"] = {0: "image_batch_size"}
        return outputs


@register_tasks_manager_onnx("owlv2", *["feature-extraction", "zero-shot-object-detection"])
class OwlV2OnnxConfig(OwlViTOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.35.0")


@register_tasks_manager_onnx(
    "layoutlm", *["feature-extraction", "fill-mask", "text-classification", "token-classification"]
)
class LayoutLMOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        allow_new=True, MAX_2D_POSITION_EMBEDDINGS="max_2d_position_embeddings"
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        }


@register_tasks_manager_onnx(
    "layoutlmv3", *["feature-extraction", "question-answering", "text-classification", "token-classification"]
)
class LayoutLMv3OnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        allow_new=True, MAX_2D_POSITION_EMBEDDINGS="max_2d_position_embeddings", image_size="input_size"
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.task in ["text-classification", "question-answering"]:
            pixel_values_dynamic_axes = {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        else:
            pixel_values_dynamic_axes = {0: "batch_size", 1: "num_channels"}
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": pixel_values_dynamic_axes,
        }


@register_tasks_manager_onnx(
    "lilt", *["feature-extraction", "question-answering", "text-classification", "token-classification"]
)
class LiltOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        allow_new=True,
        MAX_2D_POSITION_EMBEDDINGS="max_2d_position_embeddings",
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "bbox": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }


@register_tasks_manager_onnx("data2vec-text", *COMMON_TEXT_TASKS)
class Data2VecTextOnnxConfig(DistilBertOnnxConfig):
    pass


@register_tasks_manager_onnx("data2vec-vision", *["feature-extraction", "image-classification"])
class Data2VecVisionOnnxConfig(ViTOnnxConfig):
    pass


@register_tasks_manager_onnx("perceiver", *["fill-mask", "text-classification", "image-classification"])
class PerceiverOnnxConfig(TextAndVisionOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        PerceiverDummyInputGenerator,
        *TextAndVisionOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES,
    )

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        preprocessors: list[Any] | None = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self.is_generating_dummy_inputs = False

    @property
    def inputs_name(self):
        if self.is_generating_dummy_inputs:
            if self.task in ["fill-mask", "text-classification"]:
                return "input_ids"
            else:
                return "pixel_values"
        else:
            return "inputs"

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.inputs_name in ["input_ids", "inputs"]:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
            return {
                "input_ids": dynamic_axis,
                "attention_mask": dynamic_axis,
            }
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length", 2: "width", 3: "height"}
            return {
                "pixel_values": dynamic_axis,
            }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        outputs = super().outputs

        if "logits" in outputs:
            # default is {0: "batch_size", 1: "sequence_length"} where sequence_length is dynamic axis
            # but perceiver always return the same max sequence length in the second dimension
            outputs["logits"] = {0: "batch_size"}

        return outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        self.is_generating_dummy_inputs = True
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs[self.inputs_name] = dummy_inputs.pop(self.inputs_name)
        return dummy_inputs


@register_tasks_manager_onnx("hubert", *["feature-extraction", "automatic-speech-recognition", "audio-classification"])
class HubertOnnxConfig(AudioOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        outputs = super().outputs

        # Hubert output formula adapted from:
        # https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/hubert/modeling_hubert.py#L721
        if self.task == "automatic-speech-recognition":
            sequence_length = "sequence_length"
            for kernel_size, stride in zip(self._config.conv_kernel, self._config.conv_stride):
                sequence_length = f"( {sequence_length} - {kernel_size} ) // {stride} + 1"
            outputs["logits"] = {0: "batch_size", 1: sequence_length}

        return outputs


@register_tasks_manager_onnx(
    "data2vec-audio",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class Data2VecAudioOnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "wav2vec2",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class Wav2Vec2OnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "wav2vec2-conformer",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class Wav2Vec2ConformerOnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx("sew", *["feature-extraction", "automatic-speech-recognition", "audio-classification"])
class SEWOnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx("sew-d", *["feature-extraction", "automatic-speech-recognition", "audio-classification"])
class SEWDOnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "unispeech", *["feature-extraction", "automatic-speech-recognition", "audio-classification"]
)
class UniSpeechOnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "unispeech-sat",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class UniSpeechSATOnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx(
    "wavlm",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class WavLMOnnxConfig(HubertOnnxConfig):
    pass


@register_tasks_manager_onnx("mctct", *["feature-extraction", "automatic-speech-recognition"])
class MCTCTOnnxConfig(AudioOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        input_features_per_channel="input_feat_per_channel", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (MCTCTDummyAudioInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"input_features": {0: "batch_size", 1: "sequence_length"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        outputs = super().outputs

        # mctct output formula adapted from:
        # https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/deprecated/mctct/modeling_mctct.py#L455
        if self.task == "automatic-speech-recognition":
            sequence_length = "sequence_length"
            for kernel_size, stride in zip(self._config.conv_kernel, self._config.conv_stride):
                dilation = 1
                padding = kernel_size // 2
                sequence_length = f"( {sequence_length} + 2 * {padding} - {dilation} * ({kernel_size} - 1) - 1 )"
                sequence_length = f"( {sequence_length} // {stride} ) + 1"
            outputs["logits"] = {0: "batch_size", 1: sequence_length}

        return outputs


@register_tasks_manager_onnx("audio-spectrogram-transformer", *["feature-extraction", "audio-classification"])
class ASTOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_mel_bins="num_mel_bins", max_length="max_length", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (ASTDummyAudioInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"input_values": {0: "batch_size"}}


@register_tasks_manager_onnx(
    "moonshine",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "automatic-speech-recognition",
        "automatic-speech-recognition-with-past",
    ],
)
class MoonshineOnnxConfig(AudioToTextOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.48.0")
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig
    _MODEL_PATCHER = MoonshineModelPatcher
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyMoonshineAudioInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {}
        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            common_inputs["input_values"] = {0: "batch_size", 1: "encoder_sequence_length"}
        else:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}
        common_inputs["attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior in {ConfigBehavior.DECODER, ConfigBehavior.MONOLITH}:
            common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super().outputs
        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            if self._behavior is ConfigBehavior.MONOLITH:
                output_name = "encoder_last_hidden_state"
            else:
                output_name = "last_hidden_state"
            # Moonshine encoder output formula adapted from:
            # transformers.models.moonshine.modeling_moonshine.MoonshinePreTrainedModel._get_feat_extract_output_lengths
            # output_conv1_length = int((input_lengths - 127) / 64 + 1)
            # output_conv2_length = int((output_conv1_length - 7) / 3 + 1)
            # output_conv3_length = int((output_conv2_length - 3) / 2 + 1)
            output_sequence_length = "( ( ( encoder_sequence_length - 127 ) // 64 + 1 - 7 ) // 3 + 1 - 3 ) // 2 + 1"
            common_outputs[output_name] = {0: "batch_size", 1: output_sequence_length}
        return common_outputs


@register_tasks_manager_onnx(
    "whisper",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "audio-classification",
        "automatic-speech-recognition",
        "automatic-speech-recognition-with-past",
    ],
)
class WhisperOnnxConfig(AudioToTextOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        feature_size="num_mel_bins",
        allow_new=True,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.task == "audio-classification":
            return {"input_features": {0: "batch_size"}}

        common_inputs = super().inputs
        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            common_inputs["input_features"] = {0: "batch_size"}  # Remove unnecessary dynamic axis.
        else:
            # the dynamic encoder sequence length is only needed here because the input generator generates
            # encoder_outputs with a seq_len=16 but the model expects at inference time seq_len=1500
            # TODO: this can be fixed by generating the correct inputs in the input generator
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior in {ConfigBehavior.DECODER, ConfigBehavior.MONOLITH}:
            if is_transformers_version(">=", "4.43.0") and is_transformers_version("<", "4.46.0"):
                # since https://github.com/huggingface/transformers/pull/31166
                if self._behavior is not ConfigBehavior.ENCODER and self.use_past_in_inputs:
                    common_inputs["cache_position"] = {0: "decoder_sequence_length"}

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.task == "audio-classification":
            return {"logits": {0: "batch_size"}}

        common_outputs = super().outputs
        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            if self._behavior is ConfigBehavior.MONOLITH:
                output_name = "encoder_last_hidden_state"
            else:
                output_name = "last_hidden_state"
            common_outputs[output_name] = {0: "batch_size"}  # Remove unnecessary dynamic axis.
        return common_outputs


@register_tasks_manager_onnx("musicgen", *["text-to-audio"])
class MusicgenOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # NOTE: Several warnings during the export are not to worry about:
    # * for i, indices in enumerate(codes): --> can be unrolled, fixed length (num_quantizers).
    # * max_pad = max(padding_left, padding_right) --> does not impact later controlflows.
    # if length <= max_pad:  --> appears to be always False for Musicgen.

    VARIANTS = {  # noqa: RUF012
        "text-conditional-with-past": """Exports Musicgen to ONNX to generate audio samples conditioned on a text prompt (Reference: https://huggingface.co/docs/transformers/model_doc/musicgen#text-conditional-generation).
        This uses the decoder KV cache. The following subcomponents are exported:
        * text_encoder.onnx: corresponds to the text encoder part in https://github.com/huggingface/transformers/blob/v4.39.1/src/transformers/models/musicgen/modeling_musicgen.py#L1457.
        * encodec_decode.onnx: corresponds to the Encodec audio encoder part in https://github.com/huggingface/transformers/blob/v4.39.1/src/transformers/models/musicgen/modeling_musicgen.py#L2472-L2480.
        * decoder_model.onnx: The Musicgen decoder, without past key values input, and computing cross attention. Not required at inference (use decoder_model_merged.onnx instead).
        * decoder_with_past_model.onnx: The Musicgen decoder, with past_key_values input (KV cache filled), not computing cross attention. Not required at inference (use decoder_model_merged.onnx instead).
        * decoder_model_merged.onnx: The two previous models fused in one, to avoid duplicating weights. A boolean input `use_cache_branch` allows to select the branch to use. In the first forward pass where the KV cache is empty, dummy past key values inputs need to be passed and are ignored with use_cache_branch=False.
        * build_delay_pattern_mask.onnx: A model taking as input `input_ids`, `pad_token_id`, `max_length`, and building a delayed pattern mask to the input_ids. Implements https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/musicgen/modeling_musicgen.py#L1054.""",
    }
    # TODO: support audio-prompted generation (audio_encoder_encode.onnx: corresponds to the audio encoder part
    # in https://github.com/huggingface/transformers/blob/f01e1609bf4dba146d1347c1368c8c49df8636f6/src/transformers/models/musicgen/modeling_musicgen.py#L2087.)
    # With that, we have full Encodec support.
    DEFAULT_VARIANT = "text-conditional-with-past"

    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyCodegenDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummyEncodecInputGenerator,
        DummyIntGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator
    _MODEL_PATCHER = MusicgenModelPatcher

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        behavior: ConfigBehavior = ConfigBehavior.ENCODER,
        preprocessors: list[Any] | None = None,
        model_part: Literal["text_encoder", "encodec_decode", "decoder", "build_delay_pattern_mask"] | None = None,
        variant: str = "text-conditional-with-past",
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            behavior=behavior,
            preprocessors=preprocessors,
        )

        if (
            model_part in ["text_encoder", "encodec_decode", "build_delay_pattern_mask"]
            and behavior != ConfigBehavior.ENCODER
        ):
            raise ValueError(
                f"model_part is {model_part} and behavior is {behavior}. This is not supported, please open an issue at https://github.com/huggingface/optimum/issues."
            )

        if model_part == "decoder" and behavior != ConfigBehavior.DECODER:
            raise ValueError(
                f"model_part is {model_part} and behavior is {behavior}. This is not supported, please open an issue at https://github.com/huggingface/optimum/issues."
            )

        if behavior == ConfigBehavior.MONOLITH:
            raise ValueError(
                "Musicgen does not support behavior=ConfigBehavior.MONOLITH. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        if config.audio_encoder.model_type != "encodec":
            raise ValueError(
                f"Optimum ONNX export for Musicgen supports only Encodec as the audio encoder, got: {config.audio_encoder.model_type}. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        # Handling it would require to trace the audio_encoder.decode with torch.jit.script as we than have an unrollable loop.
        if config.audio_encoder.chunk_length_s is not None:
            raise ValueError(
                f"Musicgen ONNX export currently does not support audio_encoder.chunk_length_s not None (got {config.audio_encoder.chunk_length_s}). Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        self.model_part = model_part
        if self.model_part == "decoder":
            self.use_past = True  # without past is not supported, hard-code it here.

        self._normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS = NormalizedTextConfig(self._config.text_encoder)
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS = NormalizedConfig(self._config.decoder)
        self._normalized_config.decoder_num_layers = self._config.decoder.num_hidden_layers
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_layers = self._config.decoder.num_hidden_layers
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.encoder_num_attention_heads = (
            self._config.decoder.num_attention_heads
        )
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_attention_heads = (
            self._config.decoder.num_attention_heads
        )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        # Batched inference is not supported in Transformers.
        if self.model_part == "text_encoder":
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
                "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
            }
        elif self.model_part == "encodec_decode":
            # 0: always 1 for chunk_length_s=None, 2: num_quantizers fixed.
            common_inputs = {"audio_codes": {1: "batch_size", 3: "chunk_length"}}
        elif self.model_part == "build_delay_pattern_mask":
            common_inputs = {
                "input_ids": {0: "batch_size_x_num_codebooks"},
                "pad_token_id": {},
                "max_length": {},
            }
        elif self._behavior is ConfigBehavior.DECODER:
            # Naming it total_batch_size as in case we use guidance_scale, the dimension 0 may be larger than simply the batch_size.
            # Reference: https://github.com/huggingface/transformers/blob/31c575bcf13c2b85b65d652dd1b5b401f99be999/src/transformers/models/musicgen/modeling_musicgen.py#L1932-L1935
            common_inputs = {
                "decoder_input_ids": {0: "total_batch_size_x_num_codebooks"},
                "encoder_outputs": {0: "total_batch_size", 1: "encoder_sequence_length"},
                # MusicgenForConditionalGeneration maps attention_mask to encoder_attention_mask.
                "attention_mask": {
                    0: "batch_size",
                    1: "encoder_sequence_length",
                },
            }
            if self.use_past_in_inputs:
                # TODO: validate the axis name for attention_mask
                # common_inputs["attention_mask"][1] = "past_encoder_sequence_length + sequence_length"
                self.add_past_key_values(common_inputs, direction="inputs")
            else:
                common_inputs["decoder_input_ids"] = {
                    0: "total_batch_size_x_num_codebooks",
                    1: "decoder_sequence_length",
                }
        else:
            raise ValueError(
                "This should not happen. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = {}

        if self.model_part == "text_encoder":
            common_outputs = super().outputs
        elif self.model_part == "encodec_decode":
            common_outputs["audio_values"] = {0: "batch_size", 2: "audio_length"}
        elif self.model_part == "build_delay_pattern_mask":
            common_outputs["input_ids_edited"] = {0: "total_batch_size_x_num_codebooks"}
            common_outputs["delay_pattern_mask"] = {0: "total_batch_size_x_num_codebooks", 1: "max_length"}
        elif self._behavior is ConfigBehavior.DECODER:
            common_outputs = super().outputs

            # MusicgenForConditionalGeneration output is named logits, not last_hidden_state.
            # Rename last_hidden_state -> logits while keeping the order.
            common_outputs = {
                "logits" if name == "last_hidden_state" else name: value for name, value in common_outputs.items()
            }
        else:
            raise ValueError(
                "This should not happen. Please open an issue at https://github.com/huggingface/optimum/issues."
            )

        return common_outputs

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: DummyInputGenerator, input_name: str, framework: str, input_shapes: dict
    ):
        if self.model_part == "build_delay_pattern_mask" and input_name == "input_ids":
            original_batch_size = dummy_input_gen.batch_size
            dummy_input_gen.batch_size = (
                original_batch_size * dummy_input_gen.normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.num_codebooks
            )
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )
            dummy_input_gen.batch_size = original_batch_size
        else:
            dummy_input = super().overwrite_shape_and_generate_input(
                dummy_input_gen, input_name, framework, input_shapes
            )

        return dummy_input


@register_tasks_manager_onnx("speecht5", *["text-to-audio"])
class SpeechT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    # TODO: Transformers batched generation for Speecht5 is BROKEN (https://github.com/huggingface/transformers/pull/25943),
    # so we won't support for now.
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="hidden_size",
        num_attention_heads="encoder_attention_heads",  # TODO: bugged in case encoder and decoder have different number of heads
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummySpeechT5InputGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator

    VARIANTS = {  # noqa: RUF012
        "with-past": "The export follows the Transformers implementation using the KV cache, with the following components exported:\n\t - encoder_model.onnx: corresponds to the encoding part in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2544-L2556.\n\t - decoder_model.onnx: corresponds to the decoder part in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2572-L2602.\n\t - decoder_with_past_model.onnx: same as the above, with past_key_values input (KV cache filled).\n\t - decoder_postnet_and_vocoder.onnx: Decoder speech postnet and vocoder (e.g. a SpeechT5HifiGan) to generate speech from the spectrogram, as in https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2605-L2614.",
        "without-past": "The same as `with-past`, just without KV cache support. This is not a recommended export as slower than `with-past`.",
    }
    DEFAULT_VARIANT = "with-past"
    _MODEL_PATCHER = SpeechT5ModelPatcher

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        behavior: ConfigBehavior = ConfigBehavior.MONOLITH,
        preprocessors: list[Any] | None = None,
        is_postnet_and_vocoder: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            behavior=behavior,
            preprocessors=preprocessors,
        )
        if float_dtype == "fp16":
            raise ValueError(
                "The ONNX export of SpeechT5 in float16 is currently not supported due to a bug in PyTorch: https://github.com/pytorch/pytorch/pull/110078. Please open an issue in Optimum if you would like to export SpeechT5 in float16."
            )
        self.is_postnet_and_vocoder = is_postnet_and_vocoder

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {}

        # Batched inference is not supported in Transformers.
        if self._behavior is ConfigBehavior.ENCODER:
            common_inputs["input_ids"] = {1: "encoder_sequence_length"}
        elif self._behavior is ConfigBehavior.DECODER:
            # NOTE: even when past is used, the decoder takes the full sequence as input as the prenet seem to require it:
            # https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/speecht5/modeling_speecht5.py#L2573
            common_inputs["output_sequence"] = {1: "decoder_sequence_length"}
            common_inputs["speaker_embeddings"] = {}  # No dynamic shape here.
            common_inputs["encoder_outputs"] = {1: "encoder_sequence_length"}
            common_inputs["encoder_attention_mask"] = {1: "encoder_sequence_length"}

            if self.variant == "with-past" and self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")
        elif self.is_postnet_and_vocoder:
            common_inputs["spectrogram"] = {0: "n_spectrums x reduction_factor"}
        else:
            raise ValueError(
                "self._behavior is neither encoder or decoder, and is_postnet_and_vocoder=False. This should not happen."
            )

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = {}
        if self._behavior is ConfigBehavior.ENCODER:
            common_outputs["encoder_outputs"] = {1: "encoder_sequence_length"}
            common_outputs["encoder_attention_mask"] = {1: "encoder_sequence_length"}
        elif self._behavior is ConfigBehavior.DECODER:
            common_outputs["output_sequence_out"] = {1: "decoder_sequence_length + 1"}
            common_outputs["spectrum"] = {}  # No dynamic shape here.
            common_outputs["prob"] = {}  # No dynamic shape here.

            if self.variant == "with-past" and self.use_past:
                # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
                self.add_past_key_values(common_outputs, direction="outputs")
        elif self.is_postnet_and_vocoder:
            common_outputs["waveform"] = {0: "n_samples"}
        else:
            raise ValueError(
                "self._behavior is neither encoder or decoder, and is_postnet_and_vocoder=False. This should not happen."
            )

        return common_outputs

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: DummyInputGenerator, input_name: str, framework: str, input_shapes: dict
    ):
        dummy_input_gen.batch_size = 1
        dummy_input = dummy_input_gen.generate(
            input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
        )
        return dummy_input


@register_tasks_manager_onnx("vits", *["text-to-audio"])
class VitsOnnxConfig(TextEncoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size", 1: "sequence_length"},
            "attention_mask": {0: "text_batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "waveform": {0: "text_batch_size", 1: "n_samples"},
            "spectrogram": {0: "text_batch_size", 2: "num_bins"},
        }


@register_tasks_manager_onnx(
    "speech_to_text",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "automatic-speech-recognition",
        "automatic-speech-recognition-with-past",
    ],
)
class Speech2TextOnnxConfig(AudioToTextOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",
        input_features_per_channel="input_feat_per_channel",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        Speech2TextDummyAudioInputGenerator,
        *AudioToTextOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[1:],
        DummyTextInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {}

        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            common_inputs["input_features"] = {0: "batch_size", 1: "encoder_sequence_length"}
        else:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}
        common_inputs["attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior in {ConfigBehavior.DECODER, ConfigBehavior.MONOLITH}:
            common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super().outputs
        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            if self._behavior is ConfigBehavior.MONOLITH:
                output_name = "encoder_last_hidden_state"
            else:
                output_name = "last_hidden_state"
            # Speech2Text encoder output formula adapted from:
            # Speech2TextPreTrainedModel._get_feat_extract_output_lengths
            # for i in range(self.config.num_conv_layers):
            #     input_lengths = (input_lengths - 1) // 2 + 1
            downsample_factor = 2 * self._config.num_conv_layers
            output_sequence_length = f"( encoder_sequence_length + {downsample_factor} - 1 ) // {downsample_factor}"
            common_outputs[output_name] = {0: "batch_size", 1: output_sequence_length}
        return common_outputs


# TrOCR is a causal model, used as the decoder in some vision encoder-decoder models.
@register_tasks_manager_onnx(
    "trocr",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
    ],
)
class TrOCROnnxConfig(TextSeq2SeqOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",
        decoder_num_attention_heads="decoder_attention_heads",
        hidden_size="hidden_size",
    )


@register_tasks_manager_onnx(
    "vision-encoder-decoder",
    *[
        "image-to-text",
        "image-to-text-with-past",
    ],
)
class VisionEncoderDecoderOnnxConfig(EncoderDecoderBaseOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyVisionEncoderDecoderPastKeyValuesGenerator)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {}

        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            common_inputs["pixel_values"] = {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        else:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior in {ConfigBehavior.DECODER, ConfigBehavior.MONOLITH}:
            common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self._behavior == ConfigBehavior.ENCODER:
            # Some encoders have static sequence length so it is useful to rely on the encoder ONNX config to grab this information.
            return self._encoder_onnx_config.outputs
        else:
            # Ideally, we would want here to have self._decoder_onnx_config.outputs, which is currently not possible
            # as we hard-code the task to feature-extraction, that has the wrong output names (e.g. mbart does not support document-question-answering
            # so we can not initializer MBartONNXConfig with document-question-answering).
            return super().outputs


@register_tasks_manager_onnx("sam", *["feature-extraction"])
class SamOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyPointsGenerator, DummyVisionEmbeddingsGenerator)
    VARIANTS = {  # noqa: RUF012
        "monolith": "All the SAM model components are exported as a single model.onnx.",
        "split": "The vision encoder is exported as a separate vision_encoder.onnx, and the prompt encoder and mask decoder are exported as a prompt_encoder_mask_decoder.onnx. This allows to encoder the image only once for multiple point queries.",
    }
    DEFAULT_VARIANT = "split"
    _MODEL_PATCHER = SAMModelPatcher

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        variant: str = "split",
        vision_encoder: bool | None = None,
        preprocessors: list[Any] | None = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self.variant = variant
        self.vision_encoder = vision_encoder
        self._normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig(self._config.vision_config)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.variant == "monolith":
            inputs = {
                "pixel_values": {0: "batch_size"},
                "input_points": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
                "input_labels": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
            }
        else:
            if self.vision_encoder:
                inputs = {"pixel_values": {0: "batch_size"}}
            else:
                inputs = {
                    "image_positional_embeddings": {0: "batch_size"},
                    "image_embeddings": {0: "batch_size"},
                    "input_points": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
                    "input_labels": {0: "batch_size", 1: "point_batch_size", 2: "nb_points_per_image"},
                }
        return inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.variant == "split" and self.vision_encoder:
            return {"image_embeddings": {0: "batch_size"}, "image_positional_embeddings": {0: "batch_size"}}
        else:
            return {
                "iou_scores": {0: "batch_size", 1: "point_batch_size"},
                "pred_masks": {0: "batch_size", 1: "point_batch_size"},
            }


class Pix2StructNormalizedConfig(NormalizedSeq2SeqConfig):
    ENCODER_NUM_LAYERS = "vision_config.num_hidden_layers"
    DECODER_NUM_LAYERS = "text_config.num_layers"
    ENCODER_NUM_ATTENTION_HEADS = "vision_config.num_attention_heads"
    DECODER_NUM_ATTENTION_HEADS = "text_config.num_heads"
    HIDDEN_SIZE = "text_config.hidden_size"
    VOCAB_SIZE = "text_config.vocab_size"


@register_tasks_manager_onnx(
    "pix2struct",
    *[
        "image-to-text",
        "image-to-text-with-past",
    ],
)
class Pix2StructOnnxConfig(OnnxSeq2SeqConfigWithPast):
    PAD_ATTENTION_MASK_TO_PAST = True
    NORMALIZED_CONFIG_CLASS = Pix2StructNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummyPix2StructInputGenerator,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if is_transformers_version("==", "4.46.0"):
            logging.warn_once(
                logger,
                "Found transformers v4.46.0 while trying to export a Pix2Struct model, "
                "this specific version of transformers is broken for this model. Please "
                "upgrade to v4.46.1 or higher, or downgrade to v4.45.x.",
            )

    @property
    def inputs(self):
        common_inputs = {}

        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            common_inputs["flattened_patches"] = {0: "batch_size"}
        else:
            common_inputs["encoder_outputs"] = {0: "batch_size"}
        common_inputs["attention_mask"] = {0: "batch_size"}

        if self._behavior in {ConfigBehavior.DECODER, ConfigBehavior.MONOLITH}:
            common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")
                decoder_attention_mask_dim = "past_decoder_sequence_length + decoder_sequence_length"
            else:
                decoder_attention_mask_dim = "decoder_sequence_length"
            common_inputs["decoder_attention_mask"] = {0: "batch_size", 1: decoder_attention_mask_dim}

        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super().outputs
        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            if self._behavior is ConfigBehavior.MONOLITH:
                output_name = "encoder_last_hidden_state"
            else:
                output_name = "last_hidden_state"
            common_outputs[output_name] = {0: "batch_size"}  # Remove unnecessary dynamic axis.

        return common_outputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> list[DummyInputGenerator]:
        if self._preprocessors is None or len(self._preprocessors) < 2:
            raise ValueError(
                f"Preprocessors for pix2struct need to be available for the ONNX export to infer input static shapes. Got: {self._preprocessors}"
            )

        dummy_inputs_generators = []
        dummy_inputs_generators.append(
            self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config, **kwargs)
        )
        # A hack for DummyPix2StructInputGenerator to gain access to the preprocessors.
        # TODO: we probably pass preprocessors to all dummy input generators.
        encoder_sequence_length = self._preprocessors[1].image_processor.max_patches
        kwargs["preprocessors"] = self._preprocessors
        for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES[1:]:
            dummy_inputs_generators.append(
                cls_(self.task, self._normalized_config, encoder_sequence_length=encoder_sequence_length, **kwargs)
            )

        return dummy_inputs_generators

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: DummyInputGenerator, input_name: str, framework: str, input_shapes: dict
    ):
        if self._preprocessors is None or len(self._preprocessors) < 2:
            raise ValueError(
                f"Preprocessors for pix2struct need to be available for the ONNX export to infer input static shapes. Got: {self._preprocessors}"
            )

        # it would been simpler if pix2struct dummy input generator took care of generating these as well
        if input_name in ["encoder_outputs", "attention_mask"]:
            # Pix2struct takes inputs encoder inputs/outputs with a fixed sequence length (max_patches).
            original_seq_length = dummy_input_gen.sequence_length
            dummy_input_gen.sequence_length = self._preprocessors[1].image_processor.max_patches
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )
            dummy_input_gen.sequence_length = original_seq_length
        else:
            dummy_input = super().overwrite_shape_and_generate_input(
                dummy_input_gen, input_name, framework, input_shapes
            )

        return dummy_input


@register_tasks_manager_onnx("encoder-decoder", *["text2text-generation", "text2text-generation-with-past"])
class EncoderDecoderOnnxConfig(EncoderDecoderBaseOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig


@register_tasks_manager_onnx("patchtst", *["feature-extraction", "time-series-forecasting"])
class PatchTSTOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTimeSeriesForecastingConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyPatchTSTInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"past_values": {0: "batch_size", 1: "sequence_length"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.task == "feature-extraction":
            return {"last_hidden_state": {0: "batch_size"}}
        else:
            return super().outputs


@register_tasks_manager_onnx("patchtsmixer", *["feature-extraction", "time-series-forecasting"])
class PatchTSMixerOnnxConfig(PatchTSTOnnxConfig):
    pass


@register_tasks_manager_onnx("rt_detr", *["object-detection"])
class RTDetrOnnxConfig(ViTOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        }

    def _create_dummy_input_generator_classes(self, **kwargs) -> list[DummyInputGenerator]:
        min_image_size = int(math.ceil(self._config.num_queries / 32) * 32)
        if kwargs["height"] < min_image_size:
            warnings.warn(
                f"Exporting model with image `height={kwargs['height']}` which is less than "
                f"minimal {min_image_size}, setting `height` to {min_image_size}.",
                stacklevel=2,
            )
            kwargs["height"] = min_image_size
        if kwargs["width"] < min_image_size:
            warnings.warn(
                f"Exporting model with image `width={kwargs['width']}` which is less than "
                f"minimal {min_image_size}, setting `width` to {min_image_size}.",
                stacklevel=2,
            )
            kwargs["width"] = min_image_size
        return super()._create_dummy_input_generator_classes(**kwargs)


@register_tasks_manager_onnx("rt_detr_v2", *["object-detection"])
class RTDetrV2OnnxConfig(RTDetrOnnxConfig):
    pass


@register_tasks_manager_onnx("colpali", *["feature-extraction"])
class ColPaliOnnxConfig(GemmaOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyVisionInputGenerator)
    NORMALIZED_CONFIG_CLASS = NormalizedTextAndVisionConfig.with_args(
        allow_new=True,
        text_config="text_config",
        vision_config="vlm_config.vision_config",
        vlm_config="vlm_config",
    )

    VARIANTS = {  # noqa: RUF012
        "vision": "Embedding extraction for image.",
        "text": "Embedding extraction for text.",
    }
    DEFAULT_VARIANT = "vision"

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        if self.variant == "vision":
            return {
                "input_ids": dynamic_axis,
                "attention_mask": dynamic_axis,
                "pixel_values": {0: "batch_size"},
            }
        else:
            return {
                "input_ids": dynamic_axis,
                "attention_mask": dynamic_axis,
            }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "embeddings": {0: "batch_size", 1: "sequence_length"},
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        if self.variant == "vision":
            image_token_index = self._normalized_config.vlm_config.image_token_index
            num_image_tokens = self._normalized_config.vision_config.num_image_tokens
            if "sequence_length" in kwargs:
                kwargs["sequence_length"] += num_image_tokens
            else:
                kwargs["sequence_length"] = DEFAULT_DUMMY_SHAPES["sequence_length"] + num_image_tokens

        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)

        if self.variant == "vision":
            dummy_inputs["input_ids"][:, :num_image_tokens] = image_token_index
        return dummy_inputs


@register_tasks_manager_onnx("d_fine", *["object-detection"])
class DFineOnnxConfig(RTDetrOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.52.0")


@register_tasks_manager_onnx("gemma2-text-encoder", *["feature-extraction"], library_name="diffusers")
class Gemma2TextEncoderOnnxConfig(Gemma2OnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.42.0")


@register_tasks_manager_onnx("sana-transformer", *["semantic-segmentation"], library_name="diffusers")
class SanaTransformerOnnxConfig(SD3TransformerOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerVisionInputGenerator,
        DummySanaTransforemerTextInputGenerator,
        DummyTransformerTimestepInputGenerator,
    )
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        hidden_size="caption_channels",
        num_channels="in_channels",
        allow_new=True,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "hidden_states": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "sequence_length"},
            "timestep": {0: "batch_size"},
        }


@register_tasks_manager_onnx("dcae-encoder", *["semantic-segmentation"], library_name="diffusers")
class DcaeEncoderOnnxConfig(VaeEncoderOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        down_sampling_factor = 2 ** (len(self._normalized_config.encoder_block_out_channels) - 1)

        return {
            "latent_sample": {
                0: "batch_size",
                2: f"height / {down_sampling_factor}",
                3: f"width / {down_sampling_factor}",
            }
        }


@register_tasks_manager_onnx("dcae-decoder", *["semantic-segmentation"], library_name="diffusers")
class DcaeDecoderOnnxConfig(VaeDecoderOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "latent_sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        up_sampling_factor = 2 ** (len(self._normalized_config.decoder_block_out_channels) - 1)

        return {
            "sample": {
                0: "batch_size",
                2: f"latent_height * {up_sampling_factor}",
                3: f"latent_width * {up_sampling_factor}",
            }
        }
