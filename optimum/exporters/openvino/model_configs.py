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

import enum
import logging
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from optimum.exporters.openvino.base import (
    ConfigBehavior,
    OpenVINOConfig,
    OpenVINOConfigWithPast,
    OpenVINOSeq2SeqConfigWithPast,
)
from optimum.exporters.openvino.config import (
    AudioOpenVINOConfig,
    AudioToTextOpenVINOConfig,
    EncoderDecoderBaseOpenVINOConfig,
    TextAndVisionOpenVINOConfig,
    TextDecoderOpenVINOConfig,
    TextDecoderWithPositionIdsOpenVINOConfig,
    TextEncoderOpenVINOConfig,
    TextSeq2SeqOpenVINOConfig,
    VisionOpenVINOConfig,
)
from optimum.exporters.openvino.input_generators import (
    AquilaDummyPastKeyValuesGenerator,
    ChatGLM2DummyPastKeyValuesGenerator,
    DeciDummyPastKeyValuesGenerator,
    DummyAudioPhi4MMInputGenerator,
    DummyFluxTextInputGenerator,
    DummyFluxTransformerInputGenerator,
    DummyGemma4UnifiedVisionInputGenerator,
    DummyGemma4VisionInputGenerator,
    DummyKokoroInputGenerator,
    DummyLLavaMultiModalProjectorInputGenerator,
    DummyMiniCPMVImageInputGenerator,
    DummyMiniCPMVResampleInputGenerator,
    DummyPhi3VisionProjectionInputGenerator,
    DummyQwen2VLLMInputGenerator,
    DummyQwen2VLVisionEmbedInputGenerator,
    DummyQwen3_5LMInputGenerator,
    DummyQwen3VLLMInputGenerator,
    DummyQwen3VLVisionEmbedInputGenerator,
    DummySanaSeq2SeqDecoderTextWithEncMaskInputGenerator,
    DummySanaTimestepInputGenerator,
    DummySanaTransformerVisionInputGenerator,
    DummySpeechT5InputGenerator,
    DummyTransformerTimestpsInputGenerator,
    DummyUnetEncoderInputGenerator,
    DummyUnetTimestepInputGenerator,
    DummyUnetVisionInputGenerator,
    DummyVideoChatFlashQwenInputGenerator,
    DummyVideoChatFlashQwenProjectorInputGenerator,
    DummyVisionPositionIdsInputGenerator,
    DummyVisionPositionIdsPhi4InputGenerator,
    Eagle3DummyGenerator,
    Eagle3VLMDummyGenerator,
    Gemma4DummyPastKeyValuesGenerator,
    GPTBigCodeDummyPastKeyValuesGenerator,
    Lfm2DummyPastKeyValuesGenerator,
    LTXTransformerDummyInputGenerator,
    LTXVaeDummyInputGenerator,
    MambaCacheDummyInputGenerator,
    OVFalconDummyPastKeyValuesGenerator,
    OVMiniCPM3DummyPastKeyValuesGenerator,
    PooledProjectionsDummyInputGenerator,
    Qwen3_5DummyPastKeyValuesGenerator,
    Qwen3ASRDummySeq2SeqPastKeyValuesGenerator,
    Qwen3NextDummyPastKeyValuesGenerator,
    QwenDummyPastKeyValuesGenerator,
    Zamba2DummyPastKeyValuesGenerator,
)
from optimum.exporters.openvino.model_patcher import (
    AfmoeModelPatcher,
    AquilaModelPatcher,
    ArcticModelPatcher,
    BaichuanModelPatcher,
    BigBirdPegasusModelPatcher,
    BlenderbotModelPatcher,
    BlenderbotSmallModelPatcher,
    BloomModelPatcher,
    ChatGLMModelPatcher,
    CLIPModelPatcher,
    CodeGenModelPatcher,
    CommonImageEmbeddingsModelPatcher,
    DBRXModelPatcher,
    DeciLMModelPatcher,
    DeepseekPatcher,
    FalconModelPatcher,
    FluxTransformerModelPatcher,
    Gemma2ModelPatcher,
    Gemma3LMModelPatcher,
    Gemma4ImageEmbeddingsModelPatcher,
    Gemma4LMModelPatcher,
    Gemma4UnifiedImageEmbeddingsModelPatcher,
    Gemma4UnifiedLMModelPatcher,
    GptJModelPatcher,
    GptNeoModelPatcher,
    GptNeoxModelPatcher,
    GptOssModelPatcher,
    GraniteMoeHybridModelPatcher,
    GraniteMoEModelPatcher,
    IBertModelPatcher,
    Idefics3ImageEmbeddingsModelPatcher,
    InputEmbeddingPatcher,
    InternLM2Patcher,
    InternLMModelPatcher,
    InternVL2ChatLangModelPatcher,
    InternVLChatImageEmbeddingModelPatcher,
    JaisModelPatcher,
    KokoroModelPatcher,
    Lfm2ModelPatcher,
    Lfm2MoeModelPatcher,
    Llama4ImageEmbeddingsModelPatcher,
    Llama4TextModelPatcher,
    LlavaImageEmbeddingModelPatcher,
    LlavaNextVideoImageEmbeddingModelPatcher,
    LlavaQwen2ImageEmbeddingsModelPatcher,
    MairaImageEmbeddingModelPatcher,
    MambaPatcher,
    MarianModelPatcher,
    MiniCPM3Patcher,
    MiniCPMModelPatcher,
    MiniCPMVImageEmbeddingsModelPatcher,
    MiniCPMVResamplerModelPatcher,
    MistralModelPatcher,
    MixtralModelPatcher,
    ModelPatcher,
    MPTModelPatcher,
    OVDecoderModelPatcher,
    OVSeq2SeqModelPatcher,
    PegasusModelPatcher,
    PersimmonModelPatcher,
    Phi3ModelPatcher,
    Phi3VisionImageEmbeddingsPatcher,
    Phi4MMAudioEncoderPatcher,
    Phi4MMAudioForwardEmbeddingsPatcher,
    Phi4MMLanguageModelPatcher,
    Phi4MMVisionEmbeddingsPatcher,
    PhiMoEModelPatcher,
    Qwen2_5_VLVisionEmbMergerPatcher,
    Qwen2MoEPatcher,
    Qwen2VLLanguageModelPatcher,
    Qwen2VLVisionEmbMergerPatcher,
    Qwen3_5ModelPatcher,
    Qwen3_5MoeModelPatcher,
    Qwen3_5VisionEmbMergerPatcher,
    Qwen3ASRModelPatcher,
    Qwen3MoeModelPatcher,
    Qwen3NextModelPatcher,
    Qwen3VLLanguageModelPatcher,
    Qwen3VLVisionEmbMergerPatcher,
    QwenModelPatcher,
    SAMModelPatcher,
    SanaTextEncoderModelPatcher,
    SentenceTransformersTransformerPatcher,
    SpeechT5ModelPatcher,
    VideoChatFlashQwenVisionEmbeddingModelPatcher,
    XverseModelPatcher,
    Zamba2ModelPatcher,
    _get_model_attribute,
)
from optimum.exporters.tasks import TasksManager
from optimum.intel.utils.import_utils import (
    is_diffusers_available,
    is_diffusers_version,
    is_openvino_version,
    is_transformers_version,
)
from optimum.utils.input_generators import (
    ASTDummyAudioInputGenerator,
    BartDummyTextInputGenerator,
    BloomDummyPastKeyValuesGenerator,
    DummyAudioInputGenerator,
    DummyDecoderTextInputGenerator,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummyPix2StructInputGenerator,
    DummyPointsGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionEmbeddingsGenerator,
    DummyVisionEncoderDecoderPastKeyValuesGenerator,
    DummyVisionInputGenerator,
    GemmaDummyPastKeyValuesGenerator,
    MistralDummyPastKeyValuesGenerator,
    PerceiverDummyInputGenerator,
    T5DummySeq2SeqPastKeyValuesGenerator,
)
from optimum.utils.normalized_config import (
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedEncoderDecoderConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)


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


logger = logging.getLogger(__name__)


def _warn_potential_accuracy_issue_ov_2026_1(model_type: str, min_transformers_version: Optional[str] = None):
    # Fix CVS-185350: OpenVINO 2026.1.0 inference results mismatch
    if not is_openvino_version(">=", "2026.1.0"):
        return
    if min_transformers_version is not None and not is_transformers_version(">=", min_transformers_version):
        return
    logger.warning(f"Model type '{model_type}' may have potential accuracy issues with OpenVINO >= 2026.1.0.")


def init_model_configs():
    if "open_clip" not in TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES:
        TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES["open_clip"] = {}
    if "kokoro" not in TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES:
        TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES["kokoro"] = {}
    if "kokoro" not in TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP:
        from optimum.intel.utils.modeling_utils import _KokoroForTextToSpeech

        try:
            import kokoro as _kokoro_module

            if not hasattr(_kokoro_module, "_KokoroForTextToSpeech"):
                _kokoro_module._KokoroForTextToSpeech = _KokoroForTextToSpeech
            TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP["kokoro"] = {
                "text-to-audio": "_KokoroForTextToSpeech",
            }
        except ImportError:
            pass

    TasksManager._CUSTOM_CLASSES[("pt", "phi4mm", "image-text-to-text")] = ("transformers", "AutoModelForCausalLM")
    TasksManager._CUSTOM_CLASSES[("pt", "phi4mm", "automatic-speech-recognition")] = (
        "transformers",
        "AutoModelForCausalLM",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "phi4_multimodal", "image-text-to-text")] = (
        "transformers",
        "AutoModelForCausalLM",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "phi4_multimodal", "automatic-speech-recognition")] = (
        "transformers",
        "AutoModelForCausalLM",
    )

    # since transformers v4.46, model can be loaded using default AutoModelForImageTextToText
    # https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/models/auto/modeling_auto.py#L776
    if is_transformers_version("<", "4.46"):
        TasksManager._CUSTOM_CLASSES[("pt", "llava", "image-text-to-text")] = (
            "transformers",
            "LlavaForConditionalGeneration",
        )
        TasksManager._CUSTOM_CLASSES[("pt", "llava_next", "image-text-to-text")] = (
            "transformers",
            "LlavaNextForConditionalGeneration",
        )
        TasksManager._CUSTOM_CLASSES[("pt", "qwen2_vl", "image-text-to-text")] = (
            "transformers",
            "Qwen2VLForConditionalGeneration",
        )

    # since transformers v4.50, model can be loaded using default AutoModelForImageTextToText
    # https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/auto/modeling_auto.py#L835
    if is_transformers_version("<", "4.50"):
        TasksManager._CUSTOM_CLASSES[("pt", "gemma3", "image-text-to-text")] = (
            "transformers",
            "Gemma3ForConditionalGeneration",
        )

    # since transformers v4.52, model can be loaded using default AutoModelForImageTextToText
    # https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/models/auto/modeling_auto.py#L899
    if is_transformers_version("<", "4.52"):
        TasksManager._CUSTOM_CLASSES[("pt", "llava_next_video", "image-text-to-text")] = (
            "transformers",
            "AutoModelForVision2Seq",
        )

    # Qwen3-ASR is loaded via trust_remote_code; register custom classes for task lookup.
    if is_transformers_version("==", "4.57.6"):
        TasksManager._CUSTOM_CLASSES[("pt", "qwen3_asr", "automatic-speech-recognition")] = (
            "transformers",
            "AutoModel",
        )
        TasksManager._CUSTOM_CLASSES[("pt", "qwen3_asr", "automatic-speech-recognition-with-past")] = (
            "transformers",
            "AutoModel",
        )

    if is_diffusers_available() and "fill" not in TasksManager._DIFFUSERS_TASKS_TO_MODEL_LOADERS:
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_LOADERS["fill"] = "FluxFillPipeline"
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["fill"] = {"flux": "FluxFillPipeline"}
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_LOADERS["text-to-image"] = ("AutoPipelineForText2Image", "SanaPipeline")
        if "text-to-image" not in TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS:
            TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-image"] = {}
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-image"]["sana"] = "SanaPipeline"
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-image"]["sana-sprint"] = "SanaSprintPipeline"
    if is_diffusers_available() and "text-to-video" not in TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS:
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-video"] = {}
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-video"]["ltx-video"] = "LTXPipeline"


init_model_configs()


register_in_tasks_manager = TasksManager.create_register("openvino", overwrite_existing=True)


@register_in_tasks_manager("baichuan", *["text-generation", "text-generation-with-past"], library_name="transformers")
class BaichaunOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads", hidden_size="hidden_size"
    )
    _MODEL_PATCHER = BaichuanModelPatcher
    MAX_TRANSFORMERS_VERSION = "4.57.6"


@register_in_tasks_manager(
    "qwen2",
    *[
        "text-generation",
        "text-generation-with-past",
        "feature-extraction",
        "feature-extraction-with-past",
        "text-classification",
        "token-classification",
    ],
    library_name="transformers",
)
class Qwen2OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager("qwen2_moe", *["text-generation", "text-generation-with-past"], library_name="transformers")
class Qwen2MoEOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = Qwen2MoEPatcher


@register_in_tasks_manager(
    "qwen3",
    *[
        "text-generation",
        "text-generation-with-past",
        "feature-extraction",
        "feature-extraction-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class Qwen3OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.51.0"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.task in ["feature-extraction"]:
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }
        else:
            common_inputs = super().inputs
        return common_inputs


@register_in_tasks_manager(
    "qwen3_vl_text",
    *[
        "text-generation",
        "text-generation-with-past",
    ],
    library_name="transformers",
)
class Qwen3VLTextOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.57.0"

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyQwen3VLLMInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        common_inputs["visual_pos_masks"] = {0: "batch_size", 1: "sequence_length"}
        common_inputs["deepstack_visual_embeds"] = {0: "num_layers", 1: "visual_seqlen"}
        return common_inputs


@register_in_tasks_manager(
    "qwen3_moe",
    *["text-generation", "text-generation-with-past", "feature-extraction", "feature-extraction-with-past"],
    library_name="transformers",
)
class Qwen3MoEOpenVINOConfig(Qwen3OpenVINOConfig):
    _MODEL_PATCHER = Qwen3MoeModelPatcher


@register_in_tasks_manager("minicpm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MiniCPMOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.53.3"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = MiniCPMModelPatcher


@register_in_tasks_manager("minicpm3", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MiniCPM3OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.53.3"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, OVMiniCPM3DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = OVMiniCPM3DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = MiniCPM3Patcher


@register_in_tasks_manager(
    "smollm3",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class SmolLM3OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.53.0"
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager("stablelm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class StableLMOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager("chatglm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class ChatGLM2OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(vocab_size="padded_vocab_size", num_layers="num_layers")
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, ChatGLM2DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = ChatGLM2DummyPastKeyValuesGenerator
    _MODEL_PATCHER = ChatGLMModelPatcher
    MAX_TRANSFORMERS_VERSION = "4.55.4"

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
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model OpenVINO config.'
                )

        # refer to https://github.com/huggingface/optimum/pull/764
        if (
            self.use_past_in_inputs
            and self.PAD_ATTENTION_MASK_TO_PAST
            and self.use_cache_branch is not False
            and "attention_mask" in dummy_inputs
        ):
            # Obtain the past sequence length from the value instead of the key (Bloom). ChatGLM has seq_len in 0 dim instead of -2
            seq_len_dim = 0 if not hasattr(self._normalized_config, "rope_ratio") else -2
            past_present_length = (
                dummy_inputs["input_ids"].shape[1] + dummy_inputs["past_key_values"][0][1].shape[seq_len_dim]
            )

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
            decoder_sequence_name = "past_sequence_length + present_length"
            name = "present"

        is_v4 = hasattr(self._normalized_config, "rope_ratio")
        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = (
                {1: "batch_size", 0: decoder_sequence_name}
                if not is_v4
                else {0: "batch_size", 2: decoder_sequence_name}
            )
            inputs_or_outputs[f"{name}.{i}.value"] = (
                {1: "batch_size", 0: decoder_sequence_name}
                if not is_v4
                else {0: "batch_size", 2: decoder_sequence_name}
            )


@register_in_tasks_manager("mixtral", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MixtralOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOpenVINOConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)
    _MODEL_PATCHER = MixtralModelPatcher


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
class GemmaOpenVINOConfig(TextDecoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    MIN_TRANSFORMERS_VERSION = "4.38.0"
    _MODEL_PATCHER = OVDecoderModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        # position_ids was removed from optimum-openvino's gemma config because
        # it's not necessary (it's correctly generated inside the model)
        # but openvino genai requires it to be present to work properly
        inputs = super().inputs
        if "position_ids" not in inputs:
            inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}
        return inputs


@register_in_tasks_manager(
    "llama",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
        "image-text-to-text",
    ],
    library_name="transformers",
)
class LlamaOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        preprocessors: list[Any] | None = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            preprocessors=preprocessors,
        )
        archs = getattr(config, "architectures", None)
        self.eagle3 = False
        self.eagle3_vlm = False
        if isinstance(archs, list) and len(archs) > 0 and "eagle3" in archs[0].lower():
            self.MIN_TRANSFORMERS_VERSION = "4.54.0"
            self.eagle3 = True
            # VLM Eagle3 targets a VLM model (e.g. Qwen3-VL) and requires
            # inputs_embeds instead of input_ids and 3D MRoPE position_ids.
            target_model_type = getattr(config, "target_model_type", "")
            modal_type = getattr(config, "modal_type", "")
            if modal_type == "VLM" or target_model_type in {"qwen2_vl", "qwen3_vl"}:
                self.eagle3_vlm = True
                # VLM Eagle3 always needs KV cache for speculative decoding,
                # regardless of whether the task includes "-with-past".
                self.use_past = True
                self.use_past_in_inputs = True
                # Eagle3VLMDummyGenerator must precede DummyTextInputGenerator
                # so it wins for inputs_embeds and position_ids generation.
                self.DUMMY_INPUT_GENERATOR_CLASSES = (
                    (Eagle3VLMDummyGenerator,) + self.DUMMY_INPUT_GENERATOR_CLASSES + (Eagle3DummyGenerator,)
                )
                self.MIN_TRANSFORMERS_VERSION = "4.57.0"
                # VLM Eagle3 export uses transformers modeling APIs that changed in 5.0.
                self.MAX_TRANSFORMERS_VERSION = "4.57.6"
            else:
                self.DUMMY_INPUT_GENERATOR_CLASSES += (Eagle3DummyGenerator,)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        # Eagle3 model has additional conditional input
        if self.eagle3:
            common_inputs["hidden_states"] = {0: "batch_size", 1: "sequence_length", 2: "hidden_size"}
        # VLM Eagle3 uses inputs_embeds (not input_ids) and 3D MRoPE position_ids
        if self.eagle3_vlm:
            common_inputs.pop("input_ids", None)
            common_inputs["inputs_embeds"] = {0: "batch_size", 1: "sequence_length", 2: "hidden_size"}
            common_inputs["position_ids"] = {0: "num_dims", 1: "batch_size", 2: "sequence_length"}
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs
        # d2t map for Eagle3 is required to map draft tokens to target model token
        if self.eagle3:
            common_outputs["d2t"] = {0: "vocab_size"}
        return common_outputs


@register_in_tasks_manager(
    "gpt_oss",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GptOssOpenVINOConfig(LlamaOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    MIN_TRANSFORMERS_VERSION = "4.55.1"
    _MODEL_PATCHER = GptOssModelPatcher


@register_in_tasks_manager(
    "bitnet",
    *[
        "text-generation",
        "text-generation-with-past",
    ],
    library_name="transformers",
)
class BitnetOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.52.1"
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager(
    "exaone",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class ExaoneOpenVINOConfig(LlamaOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"


@register_in_tasks_manager(
    "exaone4",
    *[
        "text-generation",
        "text-generation-with-past",
    ],
    library_name="transformers",
)
class Exaone4OpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.54.0"
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"


@register_in_tasks_manager(
    "arcee",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class ArceeOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.53.0"


@register_in_tasks_manager(
    "cohere2",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class Cohere2OpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.48.0"


@register_in_tasks_manager("qwen", *["text-generation", "text-generation-with-past"])
class QwenOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.55.4"
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads", hidden_size="hidden_size"
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, QwenDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = QwenDummyPastKeyValuesGenerator
    _MODEL_PATCHER = QwenModelPatcher

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
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model OpenVINO config.'
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
            decoder_sequence_name = "past_sequence_length + sequence_length"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 1: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 1: decoder_sequence_name}


@register_in_tasks_manager(
    "starcoder2", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class Starcoder2OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager("internlm2", *["text-generation", "text-generation-with-past"], library_name="transformers")
class InternLM2OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = InternLM2Patcher


@register_in_tasks_manager("orion", *["text-generation", "text-generation-with-past"], library_name="transformers")
class OrionOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager("olmo", *["text-generation", "text-generation-with-past"], library_name="transformers")
class OlmoOpenVINOConfig(LlamaOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


@register_in_tasks_manager(
    "mpt", *["text-generation", "text-generation-with-past", "text-classification"], library_name="transformers"
)
class MPTOpenVINOConfig(TextDecoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="n_heads", hidden_size="d_model", num_layers="n_layers"
    )
    _MODEL_PATCHER = MPTModelPatcher


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
class Phi3OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOpenVINOConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    MIN_TRANSFORMERS_VERSION = "4.49.0"
    _MODEL_PATCHER = Phi3ModelPatcher


@register_in_tasks_manager(
    "phimoe",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class PhiMoEOpenVINOConfig(Phi3OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.46.0"
    _MODEL_PATCHER = PhiMoEModelPatcher


@register_in_tasks_manager(
    "phi",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class PhiOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.36.0"
    _MODEL_PATCHER = OVDecoderModelPatcher


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
class FalconOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        OVFalconDummyPastKeyValuesGenerator,
    ) + TextDecoderOpenVINOConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = OVFalconDummyPastKeyValuesGenerator
    _MODEL_PATCHER = FalconModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs

        if self._config.alibi:
            common_inputs.pop("position_ids", None)

        return common_inputs


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
class PersimmonOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = PersimmonModelPatcher


@register_in_tasks_manager("biogpt", *["text-generation", "text-generation-with-past"], library_name="transformers")
class BioGPTOpenVINOConfig(
    TextDecoderWithPositionIdsOpenVINOConfig if is_transformers_version(">=", "4.52.0") else TextDecoderOpenVINOConfig
):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager(
    "gpt_neo",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GPTNeoOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_attention_heads="num_heads")
    _MODEL_PATCHER = GptNeoModelPatcher


@register_in_tasks_manager(
    "gptj",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GPTJOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")
    _MODEL_PATCHER = GptJModelPatcher


@register_in_tasks_manager(
    "bloom",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
        "token-classification",
    ],
    library_name="transformers",
)
class BloomOpenVINOConfig(TextDecoderOpenVINOConfig):
    # Bloom does not require position_ids input.
    MIN_TRANSFORMERS_VERSION = "4.36.0"
    DUMMY_PKV_GENERATOR_CLASS = BloomDummyPastKeyValuesGenerator
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, BloomDummyPastKeyValuesGenerator)
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")
    _MODEL_PATCHER = BloomModelPatcher

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
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
class XGLMConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="attention_heads", hidden_size="d_model"
    )
    _MODEL_PATCHER = OVDecoderModelPatcher

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warn_potential_accuracy_issue_ov_2026_1("xglm")


@register_in_tasks_manager("aquila", *["text-generation", "text-generation-with-past"], library_name="transformers")
class AquilaMOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, AquilaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = AquilaDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)
    _MODEL_PATCHER = AquilaModelPatcher


@register_in_tasks_manager("xverse", *["text-generation", "text-generation-with-past"], library_name="transformers")
class XverseMOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = XverseModelPatcher


@register_in_tasks_manager("internlm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class InternLMOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = InternLMModelPatcher


@register_in_tasks_manager(
    "codegen",
    *["feature-extraction", "feature-extraction-with-past", "text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class CodeGenOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")
    _MODEL_PATCHER = CodeGenModelPatcher


@register_in_tasks_manager(
    "dbrx",
    *["text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class DBRXOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_attention_heads="n_heads",
        hidden_size="d_model",
        num_layers="n_layers",
        num_key_value_heads="attn_config.kv_n_heads",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    _MODEL_PATCHER = DBRXModelPatcher


@register_in_tasks_manager(
    "jais",
    *["text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class JaisOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator
    _MODEL_PATCHER = JaisModelPatcher


@register_in_tasks_manager("arctic", *["text-generation", "text-generation-with-past"], library_name="transformers")
class ArcticOpenVINOConfig(MixtralOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.53.3"
    _MODEL_PATCHER = ArcticModelPatcher


@register_in_tasks_manager(
    "mistral",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class MistralOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOpenVINOConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    _MODEL_PATCHER = MistralModelPatcher


@register_in_tasks_manager(
    "gpt_neox",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GPTNeoxOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = GptNeoxModelPatcher


@register_in_tasks_manager(
    "gpt_neox_japanese", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class GPTNeoxJapaneseOpenVINOConfig(TextDecoderOpenVINOConfig):
    # GPTNeoxJapanese does not require position_ids input.
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = GptNeoxModelPatcher


@register_in_tasks_manager(
    "gemma2",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class Gemma2OpenVINOConfig(GemmaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.43.0"
    _MODEL_PATCHER = Gemma2ModelPatcher


@register_in_tasks_manager(
    "gemma3_text",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class Gemma3TextOpenVINOConfig(Gemma2OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.50.0"


@register_in_tasks_manager(
    "gemma4_text",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class Gemma4TextOpenVINOConfig(Gemma3TextOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, Gemma4DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = Gemma4DummyPastKeyValuesGenerator
    MIN_TRANSFORMERS_VERSION = "5.5"

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            name = "present"

        num_kv_shared_layers = self._normalized_config.config.num_kv_shared_layers
        if num_kv_shared_layers > 0:
            layer_types = self._normalized_config.config.layer_types[:-num_kv_shared_layers]
        else:
            layer_types = self._normalized_config.config.layer_types

        for i, layer_type in enumerate(layer_types):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: decoder_sequence_name}


@register_in_tasks_manager(
    "gemma4_unified_text",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class Gemma4UnifiedTextOpenVINOConfig(Gemma4TextOpenVINOConfig):
    # The gemma4_unified text model shares gemma4's KV-cache layout (mixed sliding/full
    # attention, optional global KV heads / head dim), so add_past_key_values is inherited.
    # It has no per-layer embeddings (PLE), so no extra inputs are required.
    MIN_TRANSFORMERS_VERSION = "5.10"
    MAX_TRANSFORMERS_VERSION = "5.10.99"


@register_in_tasks_manager("deci", *["text-generation", "text-generation-with-past"], library_name="transformers")
class DeciOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DeciDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DeciDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = DeciLMModelPatcher


class CLIPNormalizedConfig(NormalizedTextAndVisionConfig):
    TEXT_CONFIG = "text_config"
    VISION_CONFIG = "vision_config"


class SiglipNormalizedConfig(CLIPNormalizedConfig):
    pass


@register_in_tasks_manager("clip", *["zero-shot-image-classification"], library_name="open_clip")
class OpenCLIPOpenVINOConfig(TextAndVisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size"},
            "pixel_values": {0: "image_batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "attention_mask": {0: "text_batch_size"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "text_features": {0: "text_batch_size"},
            "image_features": {0: "image_batch_size"},
        }

    def rename_ambiguous_inputs(self, inputs):
        model_inputs = {}
        model_inputs["image"] = inputs["pixel_values"]
        model_inputs["text"] = inputs["input_ids"]
        return model_inputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        # override sequence_length shape here in the kwargs
        kwargs["sequence_length"] = self._config.text_config.context_length
        return super().generate_dummy_inputs(framework, **kwargs)

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: Dict[str, Any], ov_input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if "attention_mask" in reference_model_inputs:
            reference_model_inputs.pop("attention_mask")
        if "image" in ov_input_names and "pixel_values" in reference_model_inputs:
            reference_model_inputs["image"] = reference_model_inputs.pop("pixel_values")
        if "text" in ov_input_names and "input_ids" in reference_model_inputs:
            reference_model_inputs["text"] = reference_model_inputs.pop("input_ids")
        return super().generate_dummy_inputs_for_validation(reference_model_inputs)


@register_in_tasks_manager("clip_text_model", *["feature-extraction"], library_name="open_clip")
class OpenCLIPTextOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        num_layers="num_hidden_layers",
        allow_new=True,
    )
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "text_batch_size"},
            "attention_mask": {0: "text_batch_size"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "text_features": {0: "text_batch_size"},
        }

    def rename_ambiguous_inputs(self, inputs):
        model_inputs = {}
        model_inputs["text"] = inputs["input_ids"]
        # model_inputs["attn_mask"] = inputs["attention_mask"]
        return model_inputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        # override sequence_length shape here in the kwargs
        kwargs["sequence_length"] = self._config.context_length
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        return dummy_inputs


@register_in_tasks_manager("clip_vision_model", *["feature-extraction"], library_name="open_clip")
class OpenCLIPVisualOpenVINOConfig(VisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "pixel_values": {0: "image_batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "image_features": {0: "image_batch_size"},
        }

    def rename_ambiguous_inputs(self, inputs):
        model_inputs = {}
        model_inputs["x"] = inputs["pixel_values"]
        return model_inputs


@register_in_tasks_manager(
    "clip", *["feature-extraction", "zero-shot-image-classification"], library_name="transformers"
)
class CLIPOpenVINOConfig(TextAndVisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
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
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.task in ["feature-extraction", "zero-shot-image-classification"]:
            return {
                "logits_per_image": {0: "image_batch_size", 1: "text_batch_size"},
                "logits_per_text": {0: "text_batch_size", 1: "image_batch_size"},
                "text_embeds": {0: "text_batch_size"},
                "image_embeds": {0: "image_batch_size"},
            }
        else:
            return super().outputs


@register_in_tasks_manager("clip_text_model", *["feature-extraction"], library_name="transformers")
@register_in_tasks_manager("clip-text", *["feature-extraction"], library_name="diffusers")
class CLIPTextOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        num_layers="num_hidden_layers",
        allow_new=True,
    )
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }

        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs


@register_in_tasks_manager("clip-text-with-projection", *["feature-extraction"], library_name="diffusers")
class CLIPTextWithProjectionOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        num_layers="num_hidden_layers",
        allow_new=True,
    )
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {
            "text_embeds": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }
        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs


@register_in_tasks_manager("clip_vision_model", *["feature-extraction"], library_name="transformers")
class CLIPVisionModelOpenVINOConfig(VisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = super().outputs
        common_outputs["last_hidden_state"] = {0: "batch_size"}
        common_outputs["pooler_output"] = {0: "batch_size"}

        return common_outputs


@register_in_tasks_manager(
    "ibert",
    *[
        "feature-extraction",
        "fill-mask",
        "text-classification",
        "multiple-choice",
        "token-classification",
        "question-answering",
    ],
    library_name="transformers",
)
class IBertOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = IBertModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}


# TODO: this is a very confusing class TBH, why not simply decompose the VLM into components, like diffusion models ?
class LMInputEmbedsConfigHelper(TextDecoderWithPositionIdsOpenVINOConfig):
    def __init__(
        self, export_config, patcher_cls=None, dummy_input_generator=None, inputs_update=None, task="text-generation"
    ):
        self.orig_export_config = export_config
        if dummy_input_generator is not None:
            export_config.DUMMY_INPUT_GENERATOR_CLASSES = (
                dummy_input_generator,
            ) + export_config.DUMMY_INPUT_GENERATOR_CLASSES
        self.DUMMY_INPUT_GENERATOR_CLASSES = export_config.DUMMY_INPUT_GENERATOR_CLASSES
        self.DUMMY_PKV_GENERATOR_CLASS = export_config.DUMMY_PKV_GENERATOR_CLASS
        self._config = export_config._config
        self._normalized_config = export_config._normalized_config
        self.use_past = export_config.use_past
        self.patcher_cls = patcher_cls
        self.input_info_upd = inputs_update
        self.task = task

    def patch_model_for_export(
        self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        model_kwargs = model_kwargs or {}
        model_kwargs["use_cache"] = True
        if self.patcher_cls is not None:
            return self.patcher_cls(self, model, model_kwargs=model_kwargs)
        # Refer to DecoderModelPatcher.
        return self.orig_export_config.patch_model_for_export(model, model_kwargs=model_kwargs)

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        outputs = self.orig_export_config.outputs
        if self.task == "feature-extraction":
            outputs["last_hidden_state"] = {0: "batch_size", 1: "patch_height", 2: "patch_width"}
        return outputs

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        orig_inputs = self.orig_export_config.inputs
        input_ids_config = orig_inputs.pop("input_ids")
        orig_inputs["inputs_embeds"] = input_ids_config
        if self.input_info_upd is not None:
            orig_inputs.update(self.input_info_upd)
        return orig_inputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = self.orig_export_config.generate_dummy_inputs(framework, **kwargs)
        input_ids = dummy_inputs.pop("input_ids")
        pask_key_values = dummy_inputs.get("past_key_values")
        inputs_embed_shape = (input_ids.shape[0], input_ids.shape[1], self._normalized_config.hidden_size)
        inputs_embeds = self.orig_export_config.DUMMY_INPUT_GENERATOR_CLASSES[0].random_float_tensor(
            inputs_embed_shape
        )
        dummy_inputs["inputs_embeds"] = inputs_embeds
        if "token_type_ids" in self.inputs:
            if is_transformers_version(">=", "4.53"):
                token_type_ids_shape = (input_ids.shape[0], input_ids.shape[1] + pask_key_values[0][0].shape[-2])
            else:
                token_type_ids_shape = (input_ids.shape[0], input_ids.shape[1])
            dummy_inputs["token_type_ids"] = self.orig_export_config.DUMMY_INPUT_GENERATOR_CLASSES[
                0
            ].random_int_tensor(token_type_ids_shape, min_value=0, max_value=2)
        if "per_layer_inputs" in self.inputs:
            per_layer_inputs_shape = (
                input_ids.shape[0],
                input_ids.shape[1],
                self._normalized_config.config.num_hidden_layers,
                self._normalized_config.config.hidden_size_per_layer_input,
            )
            dummy_inputs["per_layer_inputs"] = self.orig_export_config.DUMMY_INPUT_GENERATOR_CLASSES[
                0
            ].random_float_tensor(per_layer_inputs_shape)
        return dummy_inputs


class InputEmbedOpenVINOConfig(TextDecoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = InputEmbeddingPatcher

    @property
    def inputs(self):
        return {"input_ids": {0: "batch_size", 1: "sequence_length"}}

    @property
    def outputs(self):
        return {"inputs_embeds": {0: "batch_size", 1: "sequence_length"}}

    def rename_ambiguous_inputs(self, inputs):
        model_inputs = {}
        model_inputs["input"] = inputs["input_ids"]
        return model_inputs


def get_vlm_internal_text_generation_config(model_type, model_config, int_dtype, float_dtype):
    model_type = model_type

    if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"Unsupported language model type provided `{model_type}`. Please define custom export config"
        )

    if "text-generation-with-past" not in TasksManager._SUPPORTED_MODEL_TYPE[model_type]["openvino"]:
        raise ValueError(
            f"Export config for text generation for `{model_type}` is not available. Please define custom export config"
        )
    export_config_class = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["openvino"]["text-generation-with-past"]
    export_config = export_config_class(
        model_config,
        use_past=True,
        use_past_in_inputs=True,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
    )
    return export_config


def get_vlm_text_embeddings_config(model_type, model_config, int_dtype, float_dtype):
    internal_export_config = get_vlm_internal_text_generation_config(model_type, model_config, int_dtype, float_dtype)
    InputEmbedOpenVINOConfig.NORMALIZED_CONFIG_CLASS = internal_export_config.NORMALIZED_CONFIG_CLASS
    export_config = InputEmbedOpenVINOConfig(
        model_config,
        task="feature-extraction",
        int_dtype=int_dtype,
        float_dtype=float_dtype,
    )
    return export_config


def get_vlm_text_generation_config(
    model_type,
    model_config,
    int_dtype,
    float_dtype,
    model_patcher=None,
    dummy_input_generator=None,
    inputs_update=None,
    task=None,
):
    internal_export_config = get_vlm_internal_text_generation_config(model_type, model_config, int_dtype, float_dtype)
    export_config = LMInputEmbedsConfigHelper(
        internal_export_config,
        patcher_cls=model_patcher,
        dummy_input_generator=dummy_input_generator,
        inputs_update=inputs_update,
        task=task,
    )
    export_config._normalized_config = internal_export_config._normalized_config
    return export_config


class VLMConfigBehavior(str, enum.Enum):
    VISION_EMBEDDINGS = "vision_embeddings"
    TEXT_EMBEDDINGS = "text_embeddings"
    LANGUAGE = "language"


class BaseVLMOpenVINOConfig(OpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in VLMConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    SUPPORTS_PAST = True

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._behavior = behavior

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if not self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return {}
        return {"pixel_values": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if not self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return {}
        return {"last_hidden_state": {0: "batch_size"}}

    def with_behavior(
        self,
        behavior: Union[str, VLMConfigBehavior],
    ):
        """
        Creates a config for different behaviour.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            model_type = self._orig_config.text_config.model_type
            return get_vlm_text_embeddings_config(
                model_type, self._orig_config.text_config, self.int_dtype, self.float_dtype
            )

        if behavior == VLMConfigBehavior.LANGUAGE:
            model_type = self._orig_config.text_config.model_type
            return get_vlm_text_generation_config(
                model_type, self._orig_config.text_config, self.int_dtype, self.float_dtype
            )

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    def get_model_for_behavior(self, model, behavior: Union[str, VLMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.LANGUAGE:
            return _get_model_attribute(model, "language_model") if not hasattr(model, "lm_head") else model

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.get_input_embeddings()
            text_embedding.config = _get_model_attribute(model, "language_model").config
            return text_embedding

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return CommonImageEmbeddingsModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager("llava", *["image-text-to-text"], library_name="transformers")
class LlavaOpenVINOConfig(BaseVLMOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.37.2"
    _OV_2026_1_MODEL_TYPE = "llava"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._orig_config = config
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
        _warn_potential_accuracy_issue_ov_2026_1(self._OV_2026_1_MODEL_TYPE, min_transformers_version="5.0")

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return LlavaImageEmbeddingModelPatcher(self, model, model_kwargs)

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> Dict:
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS and self._config.model_type == "pixtral":
            kwargs["batch_size"] = 1
        return super().generate_dummy_inputs(framework, **kwargs)


@register_in_tasks_manager("llava_next", *["image-text-to-text"], library_name="transformers")
class LlavaNextOpenVINOConfig(LlavaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.40.0"
    _OV_2026_1_MODEL_TYPE = "llava_next"


class LLavaMultimodalProjectorOpenVINOConfig(OpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyLLavaMultiModalProjectorInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"image_features": {0: "batch_size", 1: "sequence_length"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {"hidden_states": {0: "batch_size", 1: "sequence_length"}}


class LlavaNextVideoConfigBehavior(str, enum.Enum):
    LANGUAGE = "language"
    VISION_EMBEDDINGS = "vision_embeddings"
    VISION_RESAMPLER = "vision_resampler"
    MULTI_MODAL_PROJECTOR = "multi_modal_projector"
    TEXT_EMBEDDINGS = "text_embeddings"


@register_in_tasks_manager("llava_next_video", *["image-text-to-text"], library_name="transformers")
class LlavaNextVideoOpenVINOConfig(LlavaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.42.0"
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in LlavaNextVideoConfigBehavior]

    def with_behavior(
        self,
        behavior: Union[str, LlavaNextVideoConfigBehavior],
    ):
        """
        Creates a config for different behaviour.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, LlavaNextVideoConfigBehavior):
            behavior = LlavaNextVideoConfigBehavior(behavior)

        if behavior == LlavaNextVideoConfigBehavior.MULTI_MODAL_PROJECTOR:
            export_config = LLavaMultimodalProjectorOpenVINOConfig(
                self._orig_config.vision_config,
                task="feature-extraction",
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
            )
            return export_config

        if behavior == LlavaNextVideoConfigBehavior.VISION_RESAMPLER:
            export_config = LLavaMultimodalProjectorOpenVINOConfig(
                self._orig_config.vision_config,
                task="feature-extraction",
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
            )
            return export_config

        return super().with_behavior(behavior)

    def get_model_for_behavior(self, model, behavior: Union[str, LlavaNextVideoConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, LlavaNextVideoConfigBehavior):
            behavior = LlavaNextVideoConfigBehavior(behavior)

        if behavior == LlavaNextVideoConfigBehavior.MULTI_MODAL_PROJECTOR:
            return (
                model.multi_modal_projector
                if hasattr(model, "multi_modal_projector")
                else model.model.multi_modal_projector
            )

        if behavior == LlavaNextVideoConfigBehavior.VISION_RESAMPLER:
            return model.vision_resampler if hasattr(model, "vision_resampler") else model.model.vision_resampler

        return super().get_model_for_behavior(model, behavior)

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != LlavaNextVideoConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return LlavaNextVideoImageEmbeddingModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager(
    "maira2", *["image-text-to-text", "text-generation", "text-generation-with-past"], library_name="transformers"
)
class MairaOpenVINOConfig(LlavaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.46.0"
    SUPPORTS_PAST = True

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return MairaImageEmbeddingModelPatcher(self, model, model_kwargs)

    def get_model_for_behavior(self, model, behavior: Union[str, VLMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.language_model.get_input_embeddings()
            text_embedding.config = model.language_model.config
            return text_embedding

        if behavior == VLMConfigBehavior.LANGUAGE:
            return model.language_model

        return super().get_model_for_behavior(model, behavior)


@register_in_tasks_manager("internvl_chat", *["image-text-to-text"], library_name="transformers")
class InternVLChatOpenVINOConfig(BaseVLMOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._behavior = behavior
        self._orig_config = config
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)

    def with_behavior(
        self,
        behavior: Union[str, VLMConfigBehavior],
    ):
        """
        Creates a config for different behaviour.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            model_type = self._orig_config.llm_config.model_type
            return get_vlm_text_embeddings_config(
                model_type, self._orig_config.llm_config, self.int_dtype, self.float_dtype
            )

        if behavior == VLMConfigBehavior.LANGUAGE:
            model_type = self._orig_config.llm_config.model_type
            return get_vlm_text_generation_config(
                model_type,
                self._orig_config.llm_config,
                self.int_dtype,
                self.float_dtype,
                InternVL2ChatLangModelPatcher,
            )

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, VLMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.LANGUAGE:
            return _get_model_attribute(model, "language_model")

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = _get_model_attribute(model, "language_model").get_input_embeddings()
            text_embedding.config = _get_model_attribute(model, "language_model").config
            return text_embedding

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return InternVLChatImageEmbeddingModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager(
    "llava-qwen2", *["image-text-to-text", "text-generation", "text-generation-with-past"], library_name="transformers"
)
class LlavaQwen2OpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTS_PAST = True
    MIN_TRANSFORMERS_VERSION = "4.40.0"
    MAX_TRANSFORMERS_VERSION = "4.53.3"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
        use_past: bool = False,
    ):
        self._behavior = behavior
        self._orig_config = config
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            config = AutoConfig.from_pretrained(config.mm_vision_tower, trust_remote_code=True)
            if hasattr(config, "vision_config"):
                config = config.vision_config
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if not self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return {}
        return {"pixel_values": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if not self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return {}
        return {"last_hidden_state": {0: "batch_size"}}

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, VLMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.LANGUAGE:
            model.forward = super(type(model), model).forward
            return model

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.model.embed_tokens
            text_embedding.config = model.model.config
            return text_embedding

    def with_behavior(
        self,
        behavior: Union[str, VLMConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            model_type = self._orig_config.model_type.replace("llava-", "")
            return get_vlm_text_embeddings_config(model_type, self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == VLMConfigBehavior.LANGUAGE:
            model_type = self._orig_config.model_type.replace("llava-", "")
            return get_vlm_text_generation_config(model_type, self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return LlavaQwen2ImageEmbeddingsModelPatcher(self, model, model_kwargs)

    def rename_ambiguous_inputs(self, inputs):
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            model_inputs = {}
            model_inputs["images"] = inputs["pixel_values"]
            return model_inputs
        return super().rename_ambiguous_inputs(inputs)


@register_in_tasks_manager("unet", *["semantic-segmentation"], library_name="diffusers")
@register_in_tasks_manager("unet-2d-condition", *["semantic-segmentation"], library_name="diffusers")
class UNetOpenVINOConfig(VisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyUnetVisionInputGenerator,
        DummyUnetTimestepInputGenerator,
        DummyUnetEncoderInputGenerator,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "timestep": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }

        # TODO : add addition_embed_type == text_image, image and image_embeds
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs["text_embeds"] = {0: "batch_size"}
            common_inputs["time_ids"] = {0: "batch_size"}

        if getattr(self._normalized_config, "time_cond_proj_dim", None) is not None:
            common_inputs["timestep_cond"] = {0: "batch_size"}

        if hasattr(self._normalized_config.config, "model_max_length"):
            common_inputs["encoder_hidden_states"] = {0: "batch_size"}

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "out_sample": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def torch_to_ov_output_map(self) -> Dict[str, str]:
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

    def ordered_inputs(self, model) -> Dict[str, Dict[int, str]]:
        inputs = super().ordered_inputs(model=model)
        # to fix mismatch between model forward signature and expected inputs
        # a dictionary of additional embeddings `added_cond_kwargs` is expected depending on config.addition_embed_type
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            inputs["text_embeds"] = self.inputs["text_embeds"]
            inputs["time_ids"] = self.inputs["time_ids"]

        return inputs


@register_in_tasks_manager("sd3-transformer", *["semantic-segmentation"], library_name="diffusers")
@register_in_tasks_manager("sd3-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class SD3TransformerOpenVINOConfig(UNetOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        (DummyTransformerTimestpsInputGenerator,)
        + UNetOpenVINOConfig.DUMMY_INPUT_GENERATOR_CLASSES
        + (PooledProjectionsDummyInputGenerator,)
    )
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        hidden_size="joint_attention_dim",
        vocab_size="attention_head_dim",
        allow_new=True,
    )

    @property
    def inputs(self):
        common_inputs = super().inputs
        common_inputs["pooled_projections"] = {0: "batch_size"}
        return common_inputs

    def rename_ambiguous_inputs(self, inputs):
        #  The input name in the model signature is `x, hence the export input name is updated.
        hidden_states = inputs.pop("sample", None)
        if hidden_states is not None:
            inputs["hidden_states"] = hidden_states
        return inputs


@register_in_tasks_manager("t5-encoder-model", *["feature-extraction"], library_name="diffusers")
@register_in_tasks_manager("t5-encoder", *["feature-extraction"], library_name="diffusers")
class T5EncoderOpenVINOConfig(CLIPTextOpenVINOConfig):
    pass


@register_in_tasks_manager("gemma2-text-encoder", *["feature-extraction"], library_name="diffusers")
class Gemma2TextEncoderOpenVINOConfig(CLIPTextOpenVINOConfig):
    _MODEL_PATCHER = SanaTextEncoderModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }


@register_in_tasks_manager("sana-transformer", *["semantic-segmentation"], library_name="diffusers")
class SanaTransformerOpenVINOConfig(UNetOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        hidden_size="caption_channels",
        vocab_size="attention_head_dim",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummySanaTransformerVisionInputGenerator,
        DummySanaSeq2SeqDecoderTextWithEncMaskInputGenerator,
        DummySanaTimestepInputGenerator,
    )

    @property
    def inputs(self):
        common_inputs = super().inputs
        common_inputs["encoder_attention_mask"] = {0: "batch_size", 1: "sequence_length"}
        if getattr(self._normalized_config.config, "guidance_embeds", False):
            common_inputs["guidance"] = {0: "batch_size"}
        return common_inputs

    def rename_ambiguous_inputs(self, inputs):
        #  The input name in the model signature is `x, hence the export input name is updated.
        hidden_states = inputs.pop("sample", None)
        if hidden_states is not None:
            inputs["hidden_states"] = hidden_states
        return inputs


@register_in_tasks_manager("vae-encoder", *["semantic-segmentation"], library_name="diffusers")
class VaeEncoderOpenVINOConfig(VisionOpenVINOConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels", image_size="sample_size", allow_new=True
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "sample_height", 3: "sample_width"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        down_sampling_factor = 2 ** (len(self._normalized_config.down_block_types) - 1)

        return {
            "latent_parameters": {
                0: "batch_size",
                2: f"sample_height / {down_sampling_factor}",
                3: f"sample_width / {down_sampling_factor}",
            },
        }


@register_in_tasks_manager("vae-decoder", *["semantic-segmentation"], library_name="diffusers")
class VaeDecoderOpenVINOConfig(VisionOpenVINOConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_channels="latent_channels", allow_new=True)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent_sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        up_sampling_factor = 2 ** (len(self._normalized_config.up_block_types) - 1)

        return {
            "sample": {
                0: "batch_size",
                2: f"latent_height * {up_sampling_factor}",
                3: f"latent_width * {up_sampling_factor}",
            },
        }


@register_in_tasks_manager("dcae-encoder", *["semantic-segmentation"], library_name="diffusers")
class DcaeEncoderOpenVINOConfig(VisionOpenVINOConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels", image_size="sample_size", allow_new=True
    )

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


@register_in_tasks_manager("dcae-decoder", *["semantic-segmentation"], library_name="diffusers")
class DcaeDecoderOpenVINOConfig(VisionOpenVINOConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_channels="latent_channels", allow_new=True)

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


@register_in_tasks_manager("flux-transformer", *["semantic-segmentation"], library_name="diffusers")
@register_in_tasks_manager("flux-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class FluxTransformerOpenVINOConfig(SD3TransformerOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestpsInputGenerator,
        DummyFluxTransformerInputGenerator,
        DummyFluxTextInputGenerator,
        PooledProjectionsDummyInputGenerator,
    )
    _MODEL_PATCHER = FluxTransformerModelPatcher

    @property
    def inputs(self):
        common_inputs = super().inputs
        common_inputs.pop("sample", None)
        common_inputs["hidden_states"] = {0: "batch_size", 1: "packed_height_width"}
        common_inputs["txt_ids"] = (
            {0: "batch_size", 1: "sequence_length"} if is_diffusers_version("<", "0.31.0") else {0: "sequence_length"}
        )
        common_inputs["img_ids"] = (
            {0: "batch_size", 1: "packed_height_width"}
            if is_diffusers_version("<", "0.31.0")
            else {0: "packed_height_width"}
        )
        if getattr(self._normalized_config, "guidance_embeds", False):
            common_inputs["guidance"] = {0: "batch_size"}
        return common_inputs


def _num_rope_axes(normalized_config) -> int:
    # FLUX.1 uses 3 RoPE axes (T, H, W); FLUX.2 uses len(axes_dims_rope) axes
    # (e.g. [32, 32, 32, 32] -> 4 axes for T, H, W, L). img_ids / txt_ids must
    # therefore carry that many coordinate columns.
    axes_dims = getattr(normalized_config, "axes_dims_rope", None)
    return len(axes_dims) if axes_dims else 3


class DummyFlux2TransformerVisionInputGenerator(DummyFluxTransformerInputGenerator):
    """`img_ids` generator for FLUX.2: uses len(axes_dims_rope) coordinate columns."""

    def __init__(self, task, normalized_config, **kwargs):
        super().__init__(task, normalized_config, **kwargs)
        self.num_rope_axes = _num_rope_axes(normalized_config)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "img_ids":
            img_ids_height = self.height // 2
            img_ids_width = self.width // 2
            return self.random_int_tensor(
                [img_ids_height * img_ids_width, self.num_rope_axes],
                min_value=0,
                max_value=min(img_ids_height, img_ids_width),
                framework=framework,
                dtype=float_dtype,
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyFlux2TextInputGenerator(DummyFluxTextInputGenerator):
    """`txt_ids` generator for FLUX.2: uses len(axes_dims_rope) coordinate columns."""

    def __init__(self, task, normalized_config, **kwargs):
        super().__init__(task, normalized_config, **kwargs)
        self.num_rope_axes = _num_rope_axes(normalized_config)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "txt_ids":
            import torch

            shape = [self.sequence_length, self.num_rope_axes]
            return torch.full(shape, 0, dtype=DTYPE_MAPPER.pt(float_dtype))
        return super().generate(input_name, framework, int_dtype, float_dtype)


@register_in_tasks_manager("flux2-transformer", *["semantic-segmentation"], library_name="diffusers")
@register_in_tasks_manager("flux2-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class Flux2TransformerOpenVINOConfig(FluxTransformerOpenVINOConfig):
    """
    Export config for ``Flux2Transformer2DModel`` (FLUX.2-klein / -dev).

    Differs from FLUX.1 in two ways:
    - The text encoder is Qwen3 (not CLIP), so there is no pooled text embedding;
      ``pooled_projections`` is dropped from both the dummy generators and the inputs.
    - ``axes_dims_rope`` has 4 entries, so ``img_ids`` / ``txt_ids`` carry 4 coordinate
      columns (handled by the FLUX.2 dummy generators above).
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestpsInputGenerator,
        DummyFlux2TransformerVisionInputGenerator,
        DummyFlux2TextInputGenerator,
    )

    @property
    def inputs(self):
        common_inputs = super().inputs
        common_inputs.pop("pooled_projections", None)
        return common_inputs


@register_in_tasks_manager("qwen3-text-encoder", *["feature-extraction"], library_name="diffusers")
class Flux2Qwen3TextEncoderOpenVINOConfig(Gemma2TextEncoderOpenVINOConfig):
    """
    Export config for the FLUX.2-klein Qwen3 text encoder.

    Inherits the ``input_ids`` / ``attention_mask`` inputs and the LLM-style
    ``SanaTextEncoderModelPatcher`` from the gemma2 text-encoder config. The model
    passed for export is a thin wrapper (see ``get_flux2_models_for_export``) that
    stacks the configured Qwen3 hidden-state layers into a single prompt-embedding
    tensor, hence the single ``last_hidden_state`` output.
    """

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}


class LTXVaeDummyInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("pixel_values", "pixel_mask", "sample", "latent_sample", "timestep")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        num_frames: int = 2,
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        self.num_frames = num_frames

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["sample", "latent_sample"]:
            return self.random_float_tensor(
                [self.batch_size, self.num_channels, self.num_frames, self.height, self.width]
            )
        if input_name == "timestep":
            return self.random_int_tensor([1], max_value=20, min_value=1, framework=framework, dtype=int_dtype)

        return super().generate(input_name, framework, int_dtype, float_dtype)


@register_in_tasks_manager("ltx-vae-encoder", *["semantic-segmentation"], library_name="diffusers")
class LTXVaeEncoderOpenVINOConfig(VisionOpenVINOConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels", image_size="sample_size", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (LTXVaeDummyInputGenerator,)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "num_frames", 3: "height", 4: "width"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent_parameters": {0: "batch_size", 2: "num_frames", 3: "height_latent", 4: "width_latent"},
        }


@register_in_tasks_manager("ltx-vae-decoder", *["semantic-segmentation"], library_name="diffusers")
class LTXVaeDecoderOpenVINOConfig(VisionOpenVINOConfig):
    ATOL_FOR_VALIDATION = 3e-4  # TODO: this only happens in test_export.py
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_channels="latent_channels", allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (LTXVaeDummyInputGenerator,)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        base_input = {
            "latent_sample": {0: "batch_size", 2: "num_frames", 3: "latent_height", 4: "latent_width"},
        }
        if self._normalized_config.config.timestep_conditioning:
            base_input["timestep"] = {}
        return base_input

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "sample": {0: "batch_size", 2: "num_frames", 3: "height", 4: "width"},
        }


@register_in_tasks_manager("ltx-video-transformer", *["semantic-segmentation"], library_name="diffusers")
class LTXVideoTransformerOpenVINOConfig(SanaTransformerOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        LTXTransformerDummyInputGenerator,
        DummySanaSeq2SeqDecoderTextWithEncMaskInputGenerator,
        DummySanaTimestepInputGenerator,
    )

    @property
    def inputs(self):
        return {
            "hidden_states": {0: "batch_size", 1: "video_sequence_length"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "sequence_length"},
            "width": {},
            "height": {},
            "num_frames": {},
            "timestep": {0: "batch_size"},
            "rope_interpolation_scale": {},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "out_sample": {0: "batch_size", 1: "video_sequence_length"},
        }


class MiniCPMVConfigBehavior(str, enum.Enum):
    RESAMPLER = "resampler"
    LANGUAGE = "language"
    VISION_EMBEDDINGS = "vision_embeddings"
    TEXT_EMBEDDINGS = "text_embeddings"


@register_in_tasks_manager("minicpmv", *["image-text-to-text"], library_name="transformers")
class MiniCPMVOpenVINOConfig(BaseVLMOpenVINOConfig):
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in MiniCPMVConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = ()
    MODEL_TYPE = "minicpmv"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: MiniCPMVConfigBehavior = MiniCPMVConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._behavior = behavior
        self._orig_config = config
        if self._behavior == MiniCPMVConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyMiniCPMVImageInputGenerator,)
        if self._behavior == MiniCPMVConfigBehavior.RESAMPLER:
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyMiniCPMVResampleInputGenerator,)
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
        _warn_potential_accuracy_issue_ov_2026_1(self.MODEL_TYPE)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == MiniCPMVConfigBehavior.VISION_EMBEDDINGS:
            return {
                "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                "patch_attention_mask": {0: "batch_size", 1: "num_patches", 2: "patch_size"},
                "position_ids": {0: "batch_size", 1: "patch_size"},
            }
        if self._behavior == MiniCPMVConfigBehavior.RESAMPLER:
            return {
                "image_feature": {0: "batch_size", 1: "patch_height", 2: "patch_width"},
                "pos_embed": {0: "patch_size", 1: "batch_size", 2: "num_patches"},
                "key_padding_mask": {0: "batch_size", 1: "patch_size"},
            }
        return {}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == MiniCPMVConfigBehavior.VISION_EMBEDDINGS:
            return {"last_hidden_state": {0: "batch_size", 1: "patch_height", 2: "patch_width"}}
        if self._behavior == MiniCPMVConfigBehavior.RESAMPLER:
            return {"last_hidden_state": {0: "batch_size"}}

        return {}

    def with_behavior(
        self,
        behavior: Union[str, MiniCPMVConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, MiniCPMVConfigBehavior):
            behavior = MiniCPMVConfigBehavior(behavior)

        if behavior == MiniCPMVConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config("qwen2", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == MiniCPMVConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config("qwen2", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == MiniCPMVConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

        if behavior == MiniCPMVConfigBehavior.RESAMPLER:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, MiniCPMVConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, MiniCPMVConfigBehavior):
            behavior = MiniCPMVConfigBehavior(behavior)

        if behavior == MiniCPMVConfigBehavior.LANGUAGE:
            return model.llm

        if behavior == MiniCPMVConfigBehavior.VISION_EMBEDDINGS:
            return model.vpm

        if behavior == MiniCPMVConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.get_input_embeddings()
            text_embedding.config = model.llm.config
            return text_embedding
        if behavior == MiniCPMVConfigBehavior.RESAMPLER:
            model.resampler.config = model.vpm.config
            return model.resampler

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior == MiniCPMVConfigBehavior.VISION_EMBEDDINGS:
            return MiniCPMVImageEmbeddingsModelPatcher(self, model, model_kwargs)

        if self._behavior == MiniCPMVConfigBehavior.RESAMPLER:
            return MiniCPMVResamplerModelPatcher(self, model, model_kwargs)

        return super().patch_model_for_export(model, model_kwargs)


@register_in_tasks_manager("minicpmo", *["image-text-to-text"], library_name="transformers")
class MiniCPMOOpenVINOConfig(MiniCPMVOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.43.0"
    MAX_TRANSFORMERS_VERSION = "4.51.3"
    MODEL_TYPE = "minicpmo"


class Phi3VisionConfigBehavior(str, enum.Enum):
    LANGUAGE = "language"
    VISION_PROJECTION = "vision_projection"
    VISION_EMBEDDINGS = "vision_embeddings"
    TEXT_EMBEDDINGS = "text_embeddings"


@register_in_tasks_manager("phi3_v", *["image-text-to-text"], library_name="transformers")
class Phi3VisionOpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in Phi3VisionConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    MIN_TRANSFORMERS_VERSION = "4.40.0"
    MAX_TRANSFORMERS_VERSION = "4.53.3"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: Phi3VisionConfigBehavior = Phi3VisionConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._behavior = behavior
        self._orig_config = config
        if self._behavior == Phi3VisionConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "img_processor"):
            self._config = AutoConfig.from_pretrained(
                config.img_processor["model_name"], trust_remote_code=True
            ).vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
        if self._behavior == Phi3VisionConfigBehavior.VISION_PROJECTION and hasattr(config, "img_processor"):
            self._config = config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyPhi3VisionProjectionInputGenerator,)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == Phi3VisionConfigBehavior.VISION_EMBEDDINGS:
            return {"pixel_values": {0: "batch_size", 2: "height", 3: "width"}}
        if self._behavior == Phi3VisionConfigBehavior.VISION_PROJECTION:
            return {"input": {0: "batch_size", 1: "img_feat_size"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior in [Phi3VisionConfigBehavior.VISION_EMBEDDINGS, Phi3VisionConfigBehavior.VISION_PROJECTION]:
            return {"last_hidden_state": {0: "batch_size", 1: "height_width_projection"}}
        return {}

    def with_behavior(
        self,
        behavior: Union[str, Phi3VisionConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, Phi3VisionConfigBehavior):
            behavior = Phi3VisionConfigBehavior(behavior)

        if behavior == Phi3VisionConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config("phi3", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == Phi3VisionConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config("phi3", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == Phi3VisionConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )
        if behavior == Phi3VisionConfigBehavior.VISION_PROJECTION:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, Phi3VisionConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, Phi3VisionConfigBehavior):
            behavior = Phi3VisionConfigBehavior(behavior)

        if behavior == Phi3VisionConfigBehavior.LANGUAGE:
            return model

        if behavior == Phi3VisionConfigBehavior.VISION_EMBEDDINGS:
            vision_embeddings = model.model.vision_embed_tokens
            vision_embeddings.config = model.config
            return vision_embeddings

        if behavior == Phi3VisionConfigBehavior.VISION_PROJECTION:
            projection = model.model.vision_embed_tokens.img_projection
            projection.config = model.config
            return projection

        if behavior == Phi3VisionConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.model.embed_tokens
            text_embedding.config = model.config
            return text_embedding

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior == Phi3VisionConfigBehavior.VISION_EMBEDDINGS:
            return Phi3VisionImageEmbeddingsPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)


class Phi4MMConfigBehavior(str, enum.Enum):
    AUDIO_EMBEDDINGS = "audio_embeddings"
    AUDIO_ENCODER = "audio_encoder"
    AUDIO_FORWARD_EMBEDDINGS = "audio_forward_embeddings"
    AUDIO_VISION_PROJECTION = "audio_vision_projection"
    AUDIO_SPEECH_PROJECTION = "audio_speech_projection"
    LANGUAGE = "language"
    TEXT_EMBEDDINGS = "text_embeddings"
    VISION_PROJECTION = "vision_projection"
    VISION_EMBEDDINGS = "vision_embeddings"


@register_in_tasks_manager(
    "phi4mm", *["image-text-to-text", "automatic-speech-recognition"], library_name="transformers"
)
@register_in_tasks_manager(
    "phi4_multimodal", *["image-text-to-text", "automatic-speech-recognition"], library_name="transformers"
)
class Phi4MMOpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in Phi4MMConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    MIN_TRANSFORMERS_VERSION = "4.51.0"
    MAX_TRANSFORMERS_VERSION = "4.53.3"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: Phi4MMConfigBehavior = Phi4MMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._behavior = behavior
        self._orig_config = config

        if self._behavior == Phi4MMConfigBehavior.VISION_EMBEDDINGS:
            if hasattr(self._config, "vision_config"):
                self._config = self._config.vision_config
                self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            else:
                self._config.image_size = self._config.embd_layer.get("image_embd_layer", {}).get("crop_size", 448)
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyVisionPositionIdsPhi4InputGenerator)
        if self._behavior == Phi4MMConfigBehavior.VISION_PROJECTION:
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyPhi3VisionProjectionInputGenerator,)
        if self._behavior in (
            Phi4MMConfigBehavior.AUDIO_EMBEDDINGS,
            Phi4MMConfigBehavior.AUDIO_FORWARD_EMBEDDINGS,
            Phi4MMConfigBehavior.AUDIO_ENCODER,
            Phi4MMConfigBehavior.AUDIO_SPEECH_PROJECTION,
            Phi4MMConfigBehavior.AUDIO_VISION_PROJECTION,
        ):
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioPhi4MMInputGenerator,)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == Phi4MMConfigBehavior.VISION_EMBEDDINGS:
            return {
                "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                "patch_attention_mask": {0: "batch_size", 1: "patch_height", 2: "patch_width"},
                "patch_position_ids": {0: "batch_size", 1: "patch_size"},
            }
        if self._behavior == Phi4MMConfigBehavior.VISION_PROJECTION:
            return {"input": {0: "batch_size", 1: "img_feat_size"}}

        if self._behavior in [Phi4MMConfigBehavior.AUDIO_EMBEDDINGS, Phi4MMConfigBehavior.AUDIO_FORWARD_EMBEDDINGS]:
            return {"audio_input": {0: "batch_size", 1: "audio_length"}}

        if self._behavior == Phi4MMConfigBehavior.AUDIO_ENCODER:
            return {
                "audio_feature": {0: "batch_size", 1: "audio_length"},
                "audio_mask": {0: "batch_size", 1: "audio_length", 2: "audio_length"},
            }

        if self._behavior in [
            Phi4MMConfigBehavior.AUDIO_SPEECH_PROJECTION,
            Phi4MMConfigBehavior.AUDIO_VISION_PROJECTION,
        ]:
            return {"audio_feature": {0: "batch_size", 1: "audio_length"}}
        return {}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior in [
            Phi4MMConfigBehavior.VISION_EMBEDDINGS,
            Phi4MMConfigBehavior.VISION_PROJECTION,
            Phi4MMConfigBehavior.AUDIO_EMBEDDINGS,
            Phi4MMConfigBehavior.AUDIO_FORWARD_EMBEDDINGS,
            Phi4MMConfigBehavior.AUDIO_ENCODER,
            Phi4MMConfigBehavior.AUDIO_SPEECH_PROJECTION,
            Phi4MMConfigBehavior.AUDIO_VISION_PROJECTION,
        ]:
            return {"last_hidden_state": {0: "batch_size", 1: "projection_size"}}
        return {}

    def with_behavior(
        self,
        behavior: Union[str, Phi4MMConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, Phi4MMConfigBehavior):
            behavior = Phi4MMConfigBehavior(behavior)

        if behavior == Phi4MMConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config("phi3", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == Phi4MMConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config(
                "phi3", self._orig_config, self.int_dtype, self.float_dtype, model_patcher=Phi4MMLanguageModelPatcher
            )

        return self.__class__(
            self._orig_config,
            task=self.task,
            int_dtype=self.int_dtype,
            float_dtype=self.float_dtype,
            behavior=behavior,
            preprocessors=self._preprocessors,
        )

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, Phi4MMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, Phi4MMConfigBehavior):
            behavior = Phi4MMConfigBehavior(behavior)

        if behavior == Phi4MMConfigBehavior.LANGUAGE:
            return model

        if behavior == Phi4MMConfigBehavior.VISION_EMBEDDINGS:
            vision_embeddings = model.model.embed_tokens_extend.image_embed
            vision_embeddings.config = model.model.embed_tokens_extend.image_embed.img_processor.config
            return model.model.embed_tokens_extend.image_embed

        if behavior == Phi4MMConfigBehavior.VISION_PROJECTION:
            vision_model = model.model.embed_tokens_extend.image_embed
            if hasattr(vision_model, "img_projection"):
                projection = vision_model.img_projection
            else:
                import torch

                projection = torch.nn.Sequential(
                    *[vision_model.img_projection_up, torch.nn.GELU(), vision_model.img_projection_down]
                )
            projection.config = vision_model.img_processor.config
            return projection

        if behavior == Phi4MMConfigBehavior.TEXT_EMBEDDINGS:
            if hasattr(model.model, "_require_grads_hook"):
                model.model.disable_input_require_grads()
            text_embedding = model.model.embed_tokens
            text_embedding.config = model.config
            return text_embedding

        if behavior == Phi4MMConfigBehavior.AUDIO_EMBEDDINGS:
            audio_embeddings = model.model.embed_tokens_extend.audio_embed.encoder.encoder_embedding
            audio_embeddings.config = model.config
            return audio_embeddings

        if behavior == Phi4MMConfigBehavior.AUDIO_ENCODER:
            audio_encoder = model.model.embed_tokens_extend.audio_embed.encoder
            audio_encoder.config = model.config
            return audio_encoder

        if behavior == Phi4MMConfigBehavior.AUDIO_FORWARD_EMBEDDINGS:
            audio_encoder = model.model.embed_tokens_extend.audio_embed.encoder
            audio_encoder.config = model.config
            return audio_encoder

        if behavior == Phi4MMConfigBehavior.AUDIO_SPEECH_PROJECTION:
            if hasattr(model.model.embed_tokens_extend.audio_embed, "audio_projection"):
                audio_projection = model.model.embed_tokens_extend.audio_embed.audio_projection["speech"]
                audio_projection.config = model.config
                return audio_projection
            else:
                import torch

                audio_projection = torch.nn.Sequential(
                    *[
                        model.model.embed_tokens_extend.audio_embed.up_proj_for_speech,
                        torch.nn.GELU(),
                        model.model.embed_tokens_extend.audio_embed.down_proj_for_speech,
                    ]
                )
                audio_projection.config = model.config
                return audio_projection

        if behavior == Phi4MMConfigBehavior.AUDIO_VISION_PROJECTION:
            if hasattr(model.model.embed_tokens_extend.audio_embed, "audio_projection"):
                audio_projection = model.model.embed_tokens_extend.audio_embed.audio_projection["vision"]
                audio_projection.config = model.config
                return audio_projection
            else:
                import torch

                audio_projection = torch.nn.Sequential(
                    *[
                        model.model.embed_tokens_extend.audio_embed.up_proj_for_vision_speech,
                        torch.nn.GELU(),
                        model.model.embed_tokens_extend.audio_embed.down_proj_for_vision_speech,
                    ]
                )
                audio_projection.config = model.config
                return audio_projection

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior == Phi4MMConfigBehavior.VISION_EMBEDDINGS:
            return Phi4MMVisionEmbeddingsPatcher(self, model, model_kwargs)
        if self._behavior == Phi4MMConfigBehavior.AUDIO_FORWARD_EMBEDDINGS:
            return Phi4MMAudioForwardEmbeddingsPatcher(self, model, model_kwargs)
        if self._behavior == Phi4MMConfigBehavior.AUDIO_ENCODER:
            return Phi4MMAudioEncoderPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    def rename_ambiguous_inputs(self, inputs):
        if self._behavior == Phi4MMConfigBehavior.AUDIO_EMBEDDINGS:
            input_info = inputs.pop("audio_input")
            inputs["input_" if hasattr(self._normalized_config.config, "audio_processor") else "x"] = input_info
        if self._behavior in [
            Phi4MMConfigBehavior.AUDIO_SPEECH_PROJECTION,
            Phi4MMConfigBehavior.AUDIO_VISION_PROJECTION,
        ]:
            input_info = inputs.pop("audio_feature")
            inputs["input"] = input_info
        return inputs


class QwenVLConfigBehavior(str, enum.Enum):
    LANGUAGE = "language"
    VISION_EMBEDDINGS = "vision_embeddings"
    VISION_EMBEDDINGS_MERGER = "vision_embeddings_merger"
    TEXT_EMBEDDINGS = "text_embeddings"
    VISION_EMBEDDINGS_POS = "vision_embeddings_pos"


@register_in_tasks_manager("qwen2_vl", *["image-text-to-text"], library_name="transformers")
class Qwen2VLOpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [
        model_type.value for model_type in QwenVLConfigBehavior if model_type.value != "vision_embeddings_pos"
    ]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyQwen2VLVisionEmbedInputGenerator,)
    MIN_TRANSFORMERS_VERSION = "4.45.0"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: QwenVLConfigBehavior = QwenVLConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._behavior = behavior
        self._orig_config = config
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self._normalized_config.use_embed_dim = False
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self._normalized_config.use_embed_dim = True

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, QwenVLConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, QwenVLConfigBehavior):
            behavior = QwenVLConfigBehavior(behavior)

        if behavior == QwenVLConfigBehavior.LANGUAGE:
            return model

        if behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS:
            vision_embeddings = _get_model_attribute(model, "visual").patch_embed
            vision_embeddings.config = model.config.vision_config
            return vision_embeddings

        if behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            vision_emb_merger = _get_model_attribute(model, "visual")
            vision_emb_merger.config = model.config.vision_config
            return vision_emb_merger

        if behavior == QwenVLConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = (
                model.model.embed_tokens
                if hasattr(model, "model") and hasattr(model.model, "embed_tokens")
                else _get_model_attribute(model, "language_model").embed_tokens
            )
            text_embedding.config = model.config
            return text_embedding

    def with_behavior(
        self,
        behavior: Union[str, QwenVLConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, QwenVLConfigBehavior):
            behavior = QwenVLConfigBehavior(behavior)

        if behavior == QwenVLConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config(
                "qwen2",
                self._orig_config if is_transformers_version("<", "5") else self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
            )

        if behavior == QwenVLConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config(
                "qwen2",
                self._orig_config if is_transformers_version("<", "5") else self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Qwen2VLLanguageModelPatcher,
                dummy_input_generator=DummyQwen2VLLMInputGenerator,
                inputs_update={"position_ids": {1: "batch_size", 2: "sequence_length"}},
            )

        if behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )
        if behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return Qwen2VLVisionEmbMergerPatcher(self, model, model_kwargs)
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS:
            return ModelPatcher(self, model, model_kwargs=model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS:
            return {"hidden_states": {0: "patch_thw_grid", 1: "patch_temporal_channels"}}
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return {
                "hidden_states": {0: "sequence_length"},
                "attention_mask": {1: "sequence_length", 2: "sequence_length"},
                "rotary_pos_emb": {0: "sequence_length"},
            }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior in [QwenVLConfigBehavior.VISION_EMBEDDINGS, QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER]:
            return {"last_hidden_state": {0: "seq_len"}}
        return {}


@register_in_tasks_manager("qwen2_5_vl", *["image-text-to-text"], library_name="transformers")
class Qwen2_5_VLOpenVINOConfig(Qwen2VLOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.49.0"

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return {
                "hidden_states": {0: "sequence_length"},
                "attention_mask": {1: "sequence_length", 2: "sequence_length"},
                "window_attention_mask": {1: "sequence_length", 2: "sequence_length"},
                "window_index": {0: "unit_sequence_length"},
                "rotary_pos_emb": {0: "sequence_length"},
            }
        return super().inputs

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return Qwen2_5_VLVisionEmbMergerPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)


@register_in_tasks_manager(
    "qwen3_vl",
    *[
        "image-text-to-text",
        "feature-extraction",
        "feature-extraction-with-past",
    ],
    library_name="transformers",
)
class Qwen3VLOpenVINOConfig(Qwen2VLOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in QwenVLConfigBehavior]
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyQwen3VLVisionEmbedInputGenerator,)
    MIN_TRANSFORMERS_VERSION = "4.57.0"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: QwenVLConfigBehavior = QwenVLConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            behavior=behavior,
        )
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_POS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self._normalized_config.use_embed_dim = True

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, QwenVLConfigBehavior]):
        if behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_POS:
            vision_emb_pos = _get_model_attribute(model, "visual").pos_embed
            vision_emb_pos.config = model.config.vision_config
            return vision_emb_pos

        return Qwen2VLOpenVINOConfig.get_model_for_behavior(model, behavior)

    def with_behavior(
        self,
        behavior: Union[str, QwenVLConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, QwenVLConfigBehavior):
            behavior = QwenVLConfigBehavior(behavior)

        if behavior == QwenVLConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config(
                "qwen3_vl_text", self._orig_config.text_config, self.int_dtype, self.float_dtype
            )

        if behavior == QwenVLConfigBehavior.LANGUAGE:
            config = get_vlm_text_generation_config(
                "qwen3_vl_text",
                self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Qwen3VLLanguageModelPatcher,
                dummy_input_generator=DummyQwen2VLLMInputGenerator,
                inputs_update={"position_ids": {1: "batch_size", 2: "sequence_length"}},
                task=self.task,
            )
            config._normalized_config.deepstack_visual_indexes = (
                self._orig_config.vision_config.deepstack_visual_indexes
            )
            return config

        if behavior in (
            QwenVLConfigBehavior.VISION_EMBEDDINGS,
            QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER,
            QwenVLConfigBehavior.VISION_EMBEDDINGS_POS,
        ):
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    def patch_model_for_export(self, model: Union["PreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return Qwen3VLVisionEmbMergerPatcher(self, model, model_kwargs)
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS:
            return ModelPatcher(self, model, model_kwargs=model_kwargs)
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_POS:
            return InputEmbeddingPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_POS:
            return {
                "input": {1: "sequence_length"},
            }
        return super().inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS:
            return super().outputs
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return {"last_hidden_state": {0: "seq_len"}, "deepstack_feature_lists": {0: "seq_len"}}
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_POS:
            return {"last_hidden_state": {0: "seq_len", 1: "seq_len"}}
        if self._behavior == QwenVLConfigBehavior.TEXT_EMBEDDINGS:
            return {"inputs_embeds": {0: "batch_size", 1: "sequence_length"}}
        if self._behavior == QwenVLConfigBehavior.LANGUAGE:
            return get_vlm_internal_text_generation_config(
                "qwen3_vl_text", self._orig_config.text_config, self.int_dtype, self.float_dtype
            ).outputs
        raise Exception("Unknown Qwen3VL behavior type.")


@register_in_tasks_manager(
    "glm",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GLMOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.46.0"


@register_in_tasks_manager(
    "glm4",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GLM4OpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.51.3"


@register_in_tasks_manager(
    "granite",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GraniteOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.45.0"


@register_in_tasks_manager(
    "granitemoe", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class GraniteMoEOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.45.0"
    _MODEL_PATCHER = GraniteMoEModelPatcher


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
class WhisperOpenVINOConfig(AudioToTextOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        feature_size="num_mel_bins",
        allow_new=True,
    )
    _MODEL_PATCHER = OVSeq2SeqModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
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
    def outputs(self) -> Dict[str, Dict[int, str]]:
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


@register_in_tasks_manager(
    "qwen3_asr",
    *[
        "automatic-speech-recognition",
        "automatic-speech-recognition-with-past",
    ],
    library_name="transformers",
)
class Qwen3ASROpenVINOConfig(AudioToTextOpenVINOConfig):
    """OpenVINO export config for Qwen3-ASR model."""

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyAudioInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        Qwen3ASRDummySeq2SeqPastKeyValuesGenerator,
    )

    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        # Use num_key_value_heads for KV cache shape generation (GQA)
        decoder_num_attention_heads="num_key_value_heads",
        feature_size="num_mel_bins",
        allow_new=True,
    )
    MIN_TRANSFORMERS_VERSION = "4.57.6"
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    _MODEL_PATCHER = Qwen3ASRModelPatcher

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "automatic-speech-recognition",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        preprocessors: Optional[List[Any]] = None,
        **kwargs,
    ):
        # Flatten nested config for NormalizedSeq2SeqConfig
        thinker = getattr(config, "thinker_config", config)
        audio_config = getattr(thinker, "audio_config", None)
        text_config = getattr(thinker, "text_config", None)

        if audio_config is not None:
            config.encoder_layers = audio_config.encoder_layers
            config.num_mel_bins = audio_config.num_mel_bins
            # Use output_dim (post-projection) as d_model since that's the actual encoder output size
            config.d_model = getattr(audio_config, "output_dim", audio_config.d_model)

        if text_config is not None:
            config.num_hidden_layers = text_config.num_hidden_layers
            config.hidden_size = text_config.hidden_size
            config.num_attention_heads = text_config.num_attention_heads
            config.num_key_value_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
            config.head_dim = getattr(
                text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads
            )
            config.decoder_start_token_id = getattr(text_config, "bos_token_id", None) or 0
            config.vocab_size = text_config.vocab_size

        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            **kwargs,
        )

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        """Override to exclude encoder KV cache since Qwen3-ASR has no cross-attention."""
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_decoder_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_decoder_sequence_length + decoder_sequence_length"
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch_size", 2: decoder_sequence_name}


@register_in_tasks_manager(
    "t5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class T5OpenVINOConfig(TextSeq2SeqOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        *TextSeq2SeqOpenVINOConfig.DUMMY_INPUT_GENERATOR_CLASSES[:-1],
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
    _MODEL_PATCHER = OVSeq2SeqModelPatcher


@register_in_tasks_manager(
    "mt5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class MT5OpenVINOConfig(T5OpenVINOConfig):
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"


@register_in_tasks_manager(
    "longt5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class LongT5OpenVINOConfig(T5OpenVINOConfig):
    pass


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
class BartOpenVINOConfig(TextSeq2SeqOpenVINOConfig):
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
    _MODEL_PATCHER = OVSeq2SeqModelPatcher

    def _create_dummy_input_generator_classes(self, **kwargs):
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
    def inputs(self) -> dict:
        inputs_properties = {
            "feature-extraction": self.inputs_for_default_and_seq2seq_lm,
            "text2text-generation": self.inputs_for_default_and_seq2seq_lm,
            "text-generation": self.inputs_for_causal_lm,
            "other": self.inputs_for_other_tasks,
        }
        return inputs_properties.get(self.task, inputs_properties["other"])

    @property
    def outputs(self) -> dict:
        if self.task in ["feature-extraction", "text2text-generation"]:
            common_outputs = super().outputs
        else:
            common_outputs = super(OpenVINOConfigWithPast, self).outputs
            if self.use_past:
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
            flattened_output = super(OpenVINOSeq2SeqConfigWithPast, self).flatten_past_key_values(
                flattened_output, name, idx, t
            )


@register_in_tasks_manager(
    "bigbird_pegasus",
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
class BigBirdPegasusOpenVINOConfig(BartOpenVINOConfig):
    _MODEL_PATCHER = BigBirdPegasusModelPatcher


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
class MBartOpenVINOConfig(BartOpenVINOConfig):
    pass


@register_in_tasks_manager(
    "m2m_100",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class M2M100OpenVINOConfig(BartOpenVINOConfig):
    pass


@register_in_tasks_manager(
    "deepseek_v3", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
@register_in_tasks_manager(
    "deepseek_v2", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
@register_in_tasks_manager("deepseek", *["text-generation", "text-generation-with-past"], library_name="transformers")
class DeepseekOpenVINOConfig(MiniCPM3OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.46.0"
    MAX_TRANSFORMERS_VERSION = "4.53.3"
    _MODEL_PATCHER = DeepseekPatcher


@register_in_tasks_manager("got_ocr2", *["image-to-text", "image-text-to-text"], library_name="transformers")
class GotOCR2OpenVINOConfig(BaseVLMOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.49.0"
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
        **kwargs,
    ):
        if task == "image-to-text":
            logger.warning(
                "Support of task 'image-to-text' will be deprecated in optimum-intel v1.29 for GOT-OCR2 models, please use 'image-text-to-text'."
            )

        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._orig_config = config
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)


@register_in_tasks_manager("gemma3", *["image-text-to-text"], library_name="transformers")
class Gemma3OpenVINOConfig(BaseVLMOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.50.0"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._orig_config = config
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)

    def with_behavior(
        self,
        behavior: Union[str, VLMConfigBehavior],
    ):
        """
        Creates a config for different behaviour.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.LANGUAGE:
            model_type = self._orig_config.text_config.model_type
            return get_vlm_text_generation_config(
                model_type,
                self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Gemma3LMModelPatcher,
                inputs_update={"token_type_ids": {0: "batch_size", 1: "sequence_length"}},
            )
        return super().with_behavior(behavior)


class Gemma4ConfigBehavior(str, enum.Enum):
    VISION_EMBEDDINGS = "vision_embeddings"
    TEXT_EMBEDDINGS = "text_embeddings"
    LANGUAGE = "language"
    TEXT_EMBEDDINGS_PER_LAYER = "text_embeddings_per_layer"


@register_in_tasks_manager("gemma4", *["image-text-to-text"], library_name="transformers")
class Gemma4OpenVINOConfig(Gemma3OpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in Gemma4ConfigBehavior]
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyTextInputGenerator)

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: Gemma4ConfigBehavior = Gemma4ConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            behavior=behavior,
        )
        self._behavior = behavior
        if self._behavior == Gemma4ConfigBehavior.VISION_EMBEDDINGS:
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyGemma4VisionInputGenerator,)
            # Attach image_seq_length from preprocessor to normalized config so
            # the dummy input generator can compute the correct number of patches.
            # The vision model's pooling uses shape-dependent Python operations baked in
            # during tracing, so the dummy input must match actual inference shapes.
            image_seq_length = None
            if preprocessors is not None:
                for p in preprocessors:
                    if hasattr(p, "image_processor") and hasattr(p.image_processor, "image_seq_length"):
                        image_seq_length = p.image_processor.image_seq_length
                        break
                    if hasattr(p, "image_processor") and hasattr(p.image_processor, "max_soft_tokens"):
                        image_seq_length = p.image_processor.max_soft_tokens
                        break
                if image_seq_length is None:
                    for p in preprocessors:
                        if hasattr(p, "max_soft_tokens"):
                            image_seq_length = p.max_soft_tokens
                            break
                        if hasattr(p, "image_seq_length"):
                            image_seq_length = p.image_seq_length
                            break
            if image_seq_length is not None:
                self._normalized_config.image_seq_length = image_seq_length
        elif self._behavior in (
            Gemma4ConfigBehavior.TEXT_EMBEDDINGS,
            Gemma4ConfigBehavior.TEXT_EMBEDDINGS_PER_LAYER,
        ):
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
            self._config = config.text_config
            self._normalized_config = NormalizedTextConfig(self._config)

    def with_behavior(self, behavior: Union[str, Gemma4ConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, Gemma4ConfigBehavior):
            behavior = Gemma4ConfigBehavior(behavior)

        if behavior == Gemma4ConfigBehavior.LANGUAGE:
            model_type = "gemma4_text"
            inputs_update = {
                "per_layer_inputs": {0: "batch_size", 1: "sequence_length", 2: "num_hidden_layers"},
            }
            if getattr(self._orig_config.get_text_config(), "use_bidirectional_attention", None) == "vision":
                inputs_update["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            return get_vlm_text_generation_config(
                model_type,
                self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Gemma4LMModelPatcher,
                inputs_update=inputs_update,
            )
        if behavior == Gemma4ConfigBehavior.TEXT_EMBEDDINGS_PER_LAYER:
            config = self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )
            return config
        return super().with_behavior(behavior)

    def get_model_for_behavior(self, model, behavior: Union[str, VLMConfigBehavior]):
        if behavior == Gemma4ConfigBehavior.TEXT_EMBEDDINGS_PER_LAYER:
            import torch

            class PerLayerInputsModule(torch.nn.Module):
                def __init__(self, language_model, vocab_size_per_layer_input: int, config):
                    super().__init__()
                    self.language_model = language_model
                    self.vocab_size_per_layer_input = vocab_size_per_layer_input
                    self.config = config

                def forward(self, input_ids: torch.Tensor):
                    # 26B-A4B has hidden_size_per_layer_input=0 (PLE disabled)
                    if self.language_model.config.hidden_size_per_layer_input <= 0:
                        return torch.zeros(
                            input_ids.shape[0],
                            input_ids.shape[1],
                            self.language_model.config.num_hidden_layers,
                            0,
                            dtype=torch.float32,
                        )
                    # Replace multimodal token IDs with pad_token_id to match
                    # HF Gemma4Model.forward which uses llm_input_ids where
                    # image/video/audio positions are set to pad_token_id
                    pad_token_id = self.config.text_config.pad_token_id
                    per_layer_inputs_tokens = input_ids.clone()
                    for token_id_attr in ("image_token_id", "video_token_id", "audio_token_id"):
                        token_id = getattr(self.config, token_id_attr, None)
                        if token_id is not None:
                            per_layer_inputs_tokens = torch.where(
                                per_layer_inputs_tokens == token_id,
                                torch.full_like(per_layer_inputs_tokens, pad_token_id),
                                per_layer_inputs_tokens,
                            )
                    per_layer_inputs_mask = torch.logical_and(
                        per_layer_inputs_tokens >= 0,
                        per_layer_inputs_tokens < self.vocab_size_per_layer_input,
                    )
                    per_layer_inputs_tokens = torch.where(
                        per_layer_inputs_mask,
                        per_layer_inputs_tokens,
                        torch.zeros_like(per_layer_inputs_tokens),
                    )
                    per_layer_inputs = self.language_model.get_per_layer_inputs(per_layer_inputs_tokens, None)
                    return per_layer_inputs

            model = PerLayerInputsModule(
                model.model.language_model, model.config.text_config.vocab_size_per_layer_input, model.config
            )
            return model
        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model
        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            import torch

            class TextEmbeddingsModule(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids: torch.Tensor):
                    inputs_embeds = self.model.get_input_embeddings()(input_ids)
                    return inputs_embeds

            text_embedding = TextEmbeddingsModule(model)
            text_embedding.config = model.model.language_model.config
            return text_embedding

        return super().get_model_for_behavior(model, behavior)

    def patch_model_for_export(self, model, model_kwargs=None):
        model_kwargs = model_kwargs or {}
        if self._behavior == Gemma4ConfigBehavior.TEXT_EMBEDDINGS_PER_LAYER:
            return ModelPatcher(self, model, model_kwargs)
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return Gemma4ImageEmbeddingsModelPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == Gemma4ConfigBehavior.LANGUAGE:
            return super().inputs
        if self._behavior == Gemma4ConfigBehavior.TEXT_EMBEDDINGS_PER_LAYER:
            return {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
            }
        if self._behavior == Gemma4ConfigBehavior.VISION_EMBEDDINGS:
            return {
                "pixel_values": {0: "batch_size", 1: "num_patches"},
                "image_position_ids": {0: "batch_size", 1: "num_patches"},
            }
        return super().inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == Gemma4ConfigBehavior.TEXT_EMBEDDINGS_PER_LAYER:
            return {"text_embeds_per_layer": {}}
        return super().outputs


@register_in_tasks_manager("gemma4_unified", *["image-text-to-text"], library_name="transformers")
class Gemma4UnifiedOpenVINOConfig(Gemma3OpenVINOConfig):
    # gemma4_unified (e.g. google/gemma-4-12B) reuses the gemma3 VLM scaffolding but has an
    # encoder-free vision embedder and no per-layer text embeddings. We only support text and
    # vision (audio is not exported).
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in VLMConfigBehavior]
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyTextInputGenerator)
    MIN_TRANSFORMERS_VERSION = "5.10"
    MAX_TRANSFORMERS_VERSION = "5.10.99"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            behavior=behavior,
        )
        self._behavior = behavior
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyGemma4UnifiedVisionInputGenerator,)
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            # The encoder-free vision embedder pads patches to image_seq_length (max_soft_tokens)
            # merged patches. The dummy input must match the traced inference shapes.
            image_seq_length = None
            if preprocessors is not None:
                for p in preprocessors:
                    image_processor = getattr(p, "image_processor", p)
                    image_seq_length = getattr(image_processor, "image_seq_length", None) or getattr(
                        image_processor, "max_soft_tokens", None
                    )
                    if image_seq_length is not None:
                        break
            if image_seq_length is not None:
                self._normalized_config.image_seq_length = image_seq_length
        elif self._behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            self.DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
            self._config = config.text_config
            self._normalized_config = NormalizedTextConfig(self._config)

    def with_behavior(self, behavior: Union[str, VLMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.LANGUAGE:
            inputs_update = {}
            if getattr(self._orig_config.get_text_config(), "use_bidirectional_attention", None) == "vision":
                inputs_update["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            return get_vlm_text_generation_config(
                "gemma4_unified_text",
                self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Gemma4UnifiedLMModelPatcher,
                inputs_update=inputs_update,
            )
        return super().with_behavior(behavior)

    def get_model_for_behavior(self, model, behavior: Union[str, VLMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:

            class TextEmbeddingsModule(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids: torch.Tensor):
                    return self.model.get_input_embeddings()(input_ids)

            text_embedding = TextEmbeddingsModule(model)
            text_embedding.config = model.model.language_model.config
            return text_embedding

        return super().get_model_for_behavior(model, behavior)

    def patch_model_for_export(self, model, model_kwargs=None):
        model_kwargs = model_kwargs or {}
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return Gemma4UnifiedImageEmbeddingsModelPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return {
                "pixel_values": {0: "batch_size", 1: "num_patches"},
                "image_position_ids": {0: "batch_size", 1: "num_patches"},
            }
        return super().inputs


@register_in_tasks_manager("idefics3", *["image-text-to-text"], library_name="transformers")
class Idefics3OpenVINOConfig(BaseVLMOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyVisionPositionIdsInputGenerator)
    MIN_TRANSFORMERS_VERSION = "4.46.0"
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VLMConfigBehavior = VLMConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        self._orig_config = config
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return Idefics3ImageEmbeddingsModelPatcher(self, model, model_kwargs)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if not self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return {}
        return {
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "patch_attention_mask": {0: "batch_size", 1: "num_height_patches", 2: "num_width_patches"},
            "patch_position_ids": {0: "batch_size", 1: "num_patches"},
        }

    def get_model_for_behavior(self, model, behavior: Union[str, VLMConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VLMConfigBehavior):
            behavior = VLMConfigBehavior(behavior)

        if behavior == VLMConfigBehavior.LANGUAGE:
            return model

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model.model

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.model.text_model.get_input_embeddings()
            text_embedding.config = model.model.text_model.config
            return text_embedding


@register_in_tasks_manager("smolvlm", *["image-text-to-text"], library_name="transformers")
class SmolVLMOpenVINOConfig(Idefics3OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.50.0"
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"


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
class BlenderbotOpenVINOConfig(BartOpenVINOConfig):
    _MODEL_PATCHER = BlenderbotModelPatcher


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
class BlenderbotSmallOpenVINOConfig(BartOpenVINOConfig):
    _MODEL_PATCHER = BlenderbotSmallModelPatcher


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
class PegasusOpenVINOConfig(BartOpenVINOConfig):
    _MODEL_PATCHER = PegasusModelPatcher

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warn_potential_accuracy_issue_ov_2026_1("pegasus")


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
class MarianOpenVINOConfig(BartOpenVINOConfig):
    _MODEL_PATCHER = MarianModelPatcher
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"


class SpeechT5ConfigBehavior(str, enum.Enum):
    ENCODER = "encoder"
    DECODER = "decoder"
    POSTNET = "postnet"
    VOCODER = "vocoder"


@register_in_tasks_manager(
    "speecht5",
    *["text-to-audio", "text-to-audio-with-past"],
    library_name="transformers",
)
class SpeechT5OpenVINOConfig(OpenVINOSeq2SeqConfigWithPast):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummySpeechT5InputGenerator,
    )
    _MODEL_PATCHER = SpeechT5ModelPatcher

    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="hidden_size",
        num_attention_heads="encoder_attention_heads",
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        allow_new=True,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator
    VARIANTS = {  # noqa: RUF012
        "with-past": "The export follows the Transformers implementation using the KV cache.",
        "without-past": "The same as `with-past`, just without KV cache support.",
    }
    DEFAULT_VARIANT = "with-past"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "text-to-audio",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = True,
        use_past_in_inputs: bool = True,
        behavior: SpeechT5ConfigBehavior = SpeechT5ConfigBehavior.ENCODER,
        preprocessors: Optional[List[Any]] = None,
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

        self.is_postnet_and_vocoder = False

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_decoder_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_decoder_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch_size", 2: "encoder_sequence_length_out"}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch_size", 2: "encoder_sequence_length_out"}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}
        if self._behavior is SpeechT5ConfigBehavior.ENCODER:
            common_inputs["input_ids"] = {0: "batch_size", 1: "encoder_sequence_length"}
        elif self._behavior is SpeechT5ConfigBehavior.DECODER:
            common_inputs["inputs_embeds"] = {0: "batch_size", 1: "decoder_sequence_length"}
            common_inputs["speaker_embeddings"] = {0: "batch_size"}
            common_inputs["encoder_hidden_states"] = {0: "batch_size", 1: "encoder_sequence_length"}
            common_inputs["encoder_attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}
            if self.variant == "with-past" and self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")
        elif self._behavior is SpeechT5ConfigBehavior.POSTNET:
            common_inputs["raw_spectrogram"] = {0: "n_spectrums", 1: "batch_size"}
        elif self._behavior is SpeechT5ConfigBehavior.VOCODER:
            common_inputs["spectrogram"] = {0: "batch_size", 1: "n_spectrums"}
        else:
            raise ValueError(
                "self._behavior is neither encoder, decoder, postnet, or vocoder. This should not happen."
            )

        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        common_outputs = {}
        if self._behavior == SpeechT5ConfigBehavior.ENCODER:
            common_outputs = {
                "last_hidden_state": {1: "encoder_sequence_length"},
                "encoder_attention_mask": {1: "encoder_sequence_length"},
            }
        elif self._behavior is SpeechT5ConfigBehavior.DECODER:
            common_outputs["output_sequence_out"] = {1: "decoder_sequence_length + 1"}
            common_outputs["spectrum"] = {}  # No dynamic shape here.
            common_outputs["prob"] = {}  # No dynamic shape here.
            if self.variant == "with-past" and self.use_past:
                self.add_past_key_values(common_outputs, direction="outputs")
        elif self._behavior is SpeechT5ConfigBehavior.POSTNET:
            common_outputs["postnet_spectrogram"] = {}
        elif self._behavior is SpeechT5ConfigBehavior.VOCODER:
            common_outputs["waveform"] = {}
        return common_outputs

    def with_behavior(
        self,
        behavior: Union[str, SpeechT5ConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        """
        if isinstance(behavior, str) and not isinstance(behavior, SpeechT5ConfigBehavior):
            behavior = SpeechT5ConfigBehavior(behavior)

        if behavior == SpeechT5ConfigBehavior.ENCODER:
            return self.__class__(
                self._config,
                use_past=False,
                use_past_in_inputs=False,
                behavior=behavior,
            )
        elif behavior == SpeechT5ConfigBehavior.DECODER:
            return self.__class__(
                self._config,
                use_past=True,
                use_past_in_inputs=True,
                behavior=behavior,
            )
        elif behavior == SpeechT5ConfigBehavior.POSTNET:
            return self.__class__(
                self._config,
                use_past=False,
                use_past_in_inputs=False,
                behavior=behavior,
            )
        elif behavior == SpeechT5ConfigBehavior.VOCODER:
            return self.__class__(
                self._config,
                use_past=False,
                use_past_in_inputs=False,
                behavior=behavior,
            )
        else:
            raise ValueError(
                "self._behavior is neither encoder, decoder, postnet, or vocoder. This should not happen."
            )

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: DummyInputGenerator, input_name: str, framework: str, input_shapes: dict
    ):
        dummy_input_gen.batch_size = 1
        dummy_input = dummy_input_gen.generate(
            input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
        )
        return dummy_input


@register_in_tasks_manager(
    "llama4_text", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class Llama4TextOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.51.0"
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    _MODEL_PATCHER = Llama4TextModelPatcher


@register_in_tasks_manager(
    "llama4", *["image-text-to-text", "text-generation", "text-generation-with-past"], library_name="transformers"
)
class Llama4OpenVINOConfig(GotOCR2OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.51.0"
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warn_potential_accuracy_issue_ov_2026_1("llama4")

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return Llama4ImageEmbeddingsModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager(
    "falcon_mamba", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
@register_in_tasks_manager("mamba", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MambaOpenVINOConfig(TextDecoderOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MambaCacheDummyInputGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MambaCacheDummyInputGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.43.0"
    _MODEL_PATCHER = MambaPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "cache_position": {0: "cache_sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
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
            ssm_conv_states_name = "cache_params.past"
        else:
            ssm_conv_states_name = "cache_params.present"

        for i in range(self._normalized_config.num_layers):
            # [batch_size, d_state, d_model]
            inputs_or_outputs[f"{ssm_conv_states_name}.ssm.{i}"] = {0: "batch_size"}
            # [batch_size, conv_kernel_size - 1, d_model]
            inputs_or_outputs[f"{ssm_conv_states_name}.conv.{i}"] = {0: "batch_size"}

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        # need to override `generate_dummy_inputs` since mamba model has other states: ssm_states and conv_states
        # which we separate and call them as past_ssm_states and past_conv_states
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs.keys() if not key.startswith("cache_params")]
        if self.use_past_in_inputs:
            input_names.extend(["cache_params"])

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
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model OpenVINO config.'
                )

        return dummy_inputs


@register_in_tasks_manager(
    "gpt2",
    *[
        "text-generation",
        "text-generation-with-past",
        "feature-extraction",
        "feature-extraction-with-past",
        "text-classification",
        "token-classification",
    ],
    library_name="transformers",
)
class GPT2OpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers="n_layer", num_attention_heads="n_head")
    _MODEL_PATCHER = OVDecoderModelPatcher


@register_in_tasks_manager(
    "vision-encoder-decoder",
    *[
        "image-to-text",
        "image-to-text-with-past",
        "document-question-answering",
        "document-question-answering-with-past",
    ],
)
class VisionEncoderDecoderOpenVINOConfig(EncoderDecoderBaseOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyVisionEncoderDecoderPastKeyValuesGenerator)
    _MODEL_PATCHER = OVSeq2SeqModelPatcher

    @property
    def inputs(self) -> dict:
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
    def outputs(self) -> dict:
        if self._behavior == ConfigBehavior.ENCODER:
            return self._encoder_ov_config.outputs
        else:
            return super().outputs


@register_in_tasks_manager("zamba2", *["text-generation", "text-generation-with-past"], library_name="transformers")
class Zamba2OpenVINOConfig(MambaOpenVINOConfig):
    PAD_ATTENTION_MASK_TO_PAST = False
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, Zamba2DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = Zamba2DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.49.0"
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    # MIN_TRANSFORMERS_VERSION = "5.2.0"
    _MODEL_PATCHER = Zamba2ModelPatcher

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warn_potential_accuracy_issue_ov_2026_1("zamba2")

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            cache_name_prefix = "cache_params.past"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            cache_name_prefix = "cache_params.present"

        for i in range(self._normalized_config.num_layers):
            # [batch_size, conv_kernel_size - 1, d_model]
            inputs_or_outputs[f"{cache_name_prefix}.conv.{i}"] = {0: "batch_size"}
            # [batch_size, d_state, d_model]
            inputs_or_outputs[f"{cache_name_prefix}.ssm.{i}"] = {0: "batch_size"}

        for i in range(len(self._normalized_config.hybrid_layer_ids)):
            inputs_or_outputs[f"{cache_name_prefix}.key.{i}"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{cache_name_prefix}.value.{i}"] = {0: "batch_size", 2: decoder_sequence_name}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs


@register_in_tasks_manager(
    "lfm2",
    *[
        "text-generation",
        "text-generation-with-past",
    ],
    library_name="transformers",
)
class LFM2OpenVINOConfig(MambaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.54.0"
    _MODEL_PATCHER = Lfm2ModelPatcher

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, Lfm2DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = Lfm2DummyPastKeyValuesGenerator

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            cache_name_prefix = "cache_params.past"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            cache_name_prefix = "cache_params.present"

        self.num_conv_layers = self._normalized_config.layer_types.count("conv")
        self.num_atten_layers = self._normalized_config.layer_types.count("full_attention")

        for i in range(self.num_conv_layers):
            inputs_or_outputs[f"{cache_name_prefix}.conv.{i}"] = {0: "batch_size"}

        for i in range(self.num_atten_layers):
            inputs_or_outputs[f"{cache_name_prefix}.key.{i}"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{cache_name_prefix}.value.{i}"] = {0: "batch_size", 2: decoder_sequence_name}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs


@register_in_tasks_manager(
    "granitemoehybrid", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class GraniteMoeHybridOpenVINOConfig(MambaOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, Zamba2DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = Zamba2DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.53.0"
    _MODEL_PATCHER = GraniteMoeHybridModelPatcher

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warn_potential_accuracy_issue_ov_2026_1("granitemoehybrid")

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            cache_name_prefix = "cache_params.past"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            cache_name_prefix = "cache_params.present"

        self.num_mamba_layers = self._normalized_config.layer_types.count("mamba")
        self.num_attention_layers = self._normalized_config.layer_types.count("attention")
        for i in range(self.num_mamba_layers):
            # [batch_size, conv_kernel_size - 1, d_model]
            inputs_or_outputs[f"{cache_name_prefix}.conv.{i}"] = {0: "batch_size"}
            # [batch_size, d_state, d_model]
            inputs_or_outputs[f"{cache_name_prefix}.ssm.{i}"] = {0: "batch_size"}

        for i in range(self.num_attention_layers):
            inputs_or_outputs[f"{cache_name_prefix}.key.{i}"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{cache_name_prefix}.value.{i}"] = {0: "batch_size", 2: decoder_sequence_name}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }
        if self.use_past_in_inputs:
            common_inputs["attention_mask"] = {0: "batch_size", 1: "past_sequence_length + sequence_length"}
            self.add_past_key_values(common_inputs, direction="inputs")
        else:
            common_inputs["attention_mask"] = {0: "batch_size", 1: "sequence_length"}
        return common_inputs


@register_in_tasks_manager("audio-spectrogram-transformer", *["feature-extraction", "audio-classification"])
class ASTOpenVINOConfig(OpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_mel_bins="num_mel_bins", max_length="max_length", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (ASTDummyAudioInputGenerator,)

    @property
    def inputs(self) -> dict:
        return {"input_values": {0: "batch_size"}}


@register_in_tasks_manager(
    "afmoe",
    *[
        "text-generation",
        "text-generation-with-past",
    ],
    library_name="transformers",
)
class AfmoeOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.55.0"
    _MODEL_PATCHER = AfmoeModelPatcher

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warn_potential_accuracy_issue_ov_2026_1("afmoe")


@register_in_tasks_manager("olmo2", *COMMON_TEXT_GENERATION_TASKS, library_name="transformers")
class Olmo2OOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.47.0"


@register_in_tasks_manager("opt", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "question-answering"])
class OPTOpenVINOConfig(
    TextDecoderWithPositionIdsOpenVINOConfig if is_transformers_version(">=", "4.46.0") else TextDecoderOpenVINOConfig
):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warn_potential_accuracy_issue_ov_2026_1("opt")


@register_in_tasks_manager(
    "gpt_bigcode", *[*COMMON_TEXT_GENERATION_TASKS, "text-classification", "token-classification"]
)
class GPTBigCodeOpenVINOConfig(TextDecoderWithPositionIdsOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GPTBigCodeDummyPastKeyValuesGenerator)
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("gpt_bigcode")
    DUMMY_PKV_GENERATOR_CLASS = GPTBigCodeDummyPastKeyValuesGenerator

    def add_past_key_values(self, inputs_or_outputs: dict, direction: str):
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


class _Pix2StructNormalizedConfig(NormalizedSeq2SeqConfig):
    ENCODER_NUM_LAYERS = "vision_config.num_hidden_layers"
    DECODER_NUM_LAYERS = "text_config.num_layers"
    ENCODER_NUM_ATTENTION_HEADS = "vision_config.num_attention_heads"
    DECODER_NUM_ATTENTION_HEADS = "text_config.num_heads"
    HIDDEN_SIZE = "text_config.hidden_size"
    VOCAB_SIZE = "text_config.vocab_size"


@register_in_tasks_manager(
    "pix2struct",
    *[
        "image-to-text",
        "image-to-text-with-past",
    ],
)
class Pix2StructOpenVINOConfig(OpenVINOSeq2SeqConfigWithPast):
    PAD_ATTENTION_MASK_TO_PAST = True
    NORMALIZED_CONFIG_CLASS = _Pix2StructNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummyPix2StructInputGenerator,
    )
    _MODEL_PATCHER = OVSeq2SeqModelPatcher

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if is_transformers_version("==", "4.46.0"):
            logger.warning(
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
    def outputs(self) -> dict:
        common_outputs = super().outputs
        if self._behavior in {ConfigBehavior.ENCODER, ConfigBehavior.MONOLITH}:
            if self._behavior is ConfigBehavior.MONOLITH:
                output_name = "encoder_last_hidden_state"
            else:
                output_name = "last_hidden_state"
            common_outputs[output_name] = {0: "batch_size"}  # Remove unnecessary dynamic axis.

        return common_outputs

    def _create_dummy_input_generator_classes(self, **kwargs):
        if self._preprocessors is None or len(self._preprocessors) < 2:
            raise ValueError(
                f"Preprocessors for pix2struct need to be available for the OpenVINO export to infer input static shapes. Got: {self._preprocessors}"
            )

        dummy_inputs_generators = []
        dummy_inputs_generators.append(
            self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config, **kwargs)
        )
        encoder_sequence_length = self._preprocessors[1].image_processor.max_patches
        kwargs["preprocessors"] = self._preprocessors
        for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES[1:]:
            dummy_inputs_generators.append(
                cls_(self.task, self._normalized_config, encoder_sequence_length=encoder_sequence_length, **kwargs)
            )

        return dummy_inputs_generators

    def overwrite_shape_and_generate_input(self, dummy_input_gen, input_name: str, framework: str, input_shapes: dict):
        if self._preprocessors is None or len(self._preprocessors) < 2:
            raise ValueError(
                f"Preprocessors for pix2struct need to be available for the OpenVINO export to infer input static shapes. Got: {self._preprocessors}"
            )

        if input_name in ["encoder_outputs", "attention_mask"]:
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


@register_in_tasks_manager("bert", *COMMON_TEXT_TASKS)
class BertOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "token_type_ids": dynamic_axis,
        }


@register_in_tasks_manager("albert", *COMMON_TEXT_TASKS)
class AlbertOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("nystromformer", *COMMON_TEXT_TASKS)
class NystromformerOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("convbert", *COMMON_TEXT_TASKS)
class ConvBertOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("electra", *COMMON_TEXT_TASKS)
class ElectraOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("roformer", *COMMON_TEXT_TASKS)
class RoFormerOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("squeezebert", *COMMON_TEXT_TASKS)
class SqueezeBertOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("xlm", *COMMON_TEXT_TASKS)
class XLMOpenVINOConfig(BertOpenVINOConfig):
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"


@register_in_tasks_manager("distilbert", *COMMON_TEXT_TASKS)
class DistilBertOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}


@register_in_tasks_manager("xlm-roberta", *COMMON_TEXT_TASKS)
class XLMRobertaOpenVINOConfig(DistilBertOpenVINOConfig):
    pass


@register_in_tasks_manager("roberta", *COMMON_TEXT_TASKS)
class RobertaOpenVINOConfig(DistilBertOpenVINOConfig):
    pass


@register_in_tasks_manager("camembert", *COMMON_TEXT_TASKS)
class CamembertOpenVINOConfig(DistilBertOpenVINOConfig):
    pass


@register_in_tasks_manager("flaubert", *COMMON_TEXT_TASKS)
class FlaubertOpenVINOConfig(BertOpenVINOConfig):
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"


@register_in_tasks_manager(
    "deberta",
    *["feature-extraction", "fill-mask", "text-classification", "token-classification", "question-answering"],
)
class DebertaOpenVINOConfig(BertOpenVINOConfig):
    @property
    def inputs(self) -> dict:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            common_inputs.pop("token_type_ids")
        return common_inputs


@register_in_tasks_manager("deberta-v2", *COMMON_TEXT_TASKS)
class DebertaV2OpenVINOConfig(DebertaOpenVINOConfig):
    pass


@register_in_tasks_manager("hubert", *["feature-extraction", "automatic-speech-recognition", "audio-classification"])
class HubertOpenVINOConfig(AudioOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig

    @property
    def outputs(self) -> dict:
        outputs = super().outputs

        # Hubert output formula adapted from:
        # https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/hubert/modeling_hubert.py#L721
        if self.task == "automatic-speech-recognition":
            sequence_length = "sequence_length"
            for kernel_size, stride in zip(self._config.conv_kernel, self._config.conv_stride):
                sequence_length = f"( {sequence_length} - {kernel_size} ) // {stride} + 1"
            outputs["logits"] = {0: "batch_size", 1: sequence_length}

        return outputs


@register_in_tasks_manager(
    "data2vec-audio",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class Data2VecAudioOpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager("data2vec-text", *COMMON_TEXT_TASKS)
class Data2VecTextOpenVINOConfig(DistilBertOpenVINOConfig):
    # TODO (@echarlaix): add v5 support
    MAX_TRANSFORMERS_VERSION = "4.57.6"


@register_in_tasks_manager("vit", *["feature-extraction", "image-classification", "masked-im"])
class ViTOpenVINOConfig(VisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig

    @property
    def inputs(self) -> dict:
        return {"pixel_values": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> dict:
        common_outputs = super().outputs

        if self.task == "feature-extraction":
            common_outputs["last_hidden_state"] = {0: "batch_size"}

        return common_outputs


@register_in_tasks_manager("data2vec-vision", *["feature-extraction", "image-classification"])
class Data2VecVisionOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("perceiver", *["fill-mask", "text-classification", "image-classification"])
class PerceiverOpenVINOConfig(TextAndVisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        PerceiverDummyInputGenerator,
        *TextAndVisionOpenVINOConfig.DUMMY_INPUT_GENERATOR_CLASSES,
    )

    def __init__(
        self,
        config,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        preprocessors=None,
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
    def inputs(self) -> dict:
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
    def outputs(self) -> dict:
        outputs = super().outputs

        if "logits" in outputs:
            outputs["logits"] = {0: "batch_size"}

        return outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        self.is_generating_dummy_inputs = True
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
        dummy_inputs[self.inputs_name] = dummy_inputs.pop(self.inputs_name)
        return dummy_inputs


@register_in_tasks_manager("esm", *["feature-extraction", "fill-mask", "text-classification", "token-classification"])
class EsmOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> dict:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
        }


@register_in_tasks_manager("mpnet", *COMMON_TEXT_TASKS)
class MPNetOpenVINOConfig(DistilBertOpenVINOConfig):
    pass


@register_in_tasks_manager("beit", *["feature-extraction", "image-classification"])
class BeitOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("deit", *["feature-extraction", "image-classification", "masked-im"])
class DeiTOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("levit", *["feature-extraction", "image-classification"])
class LevitOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("mobilevit", *["feature-extraction", "image-classification", "image-segmentation"])
class MobileViTOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("mobilenet_v1", *["feature-extraction", "image-classification"])
class MobileNetV1OpenVINOConfig(ViTOpenVINOConfig):
    @property
    def inputs(self) -> dict:
        return {"pixel_values": {0: "batch_size"}}


@register_in_tasks_manager("mobilenet_v2", *["feature-extraction", "image-classification"])
class MobileNetV2OpenVINOConfig(MobileNetV1OpenVINOConfig):
    pass


@register_in_tasks_manager("poolformer", *["feature-extraction", "image-classification"])
class PoolFormerOpenVINOConfig(ViTOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig


@register_in_tasks_manager(
    "segformer", *["feature-extraction", "image-classification", "image-segmentation", "semantic-segmentation"]
)
class SegformerOpenVINOConfig(ViTOpenVINOConfig):
    @property
    def outputs(self) -> dict:
        outputs = super().outputs

        if self.task == "image-segmentation":
            outputs["logits"] = {0: "batch_size"}

        return outputs


@register_in_tasks_manager("swin", *["feature-extraction", "image-classification", "masked-im"])
class SwinOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("donut-swin", *["feature-extraction"])
class DonutSwinOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("convnext", *["feature-extraction", "image-classification"])
class ConvNextOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager("resnet", *["feature-extraction", "image-classification"])
class ResNetOpenVINOConfig(ViTOpenVINOConfig):
    pass


@register_in_tasks_manager(
    "wav2vec2",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class Wav2Vec2OpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager(
    "wav2vec2-conformer",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class Wav2Vec2ConformerOpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager("sew", *["feature-extraction", "automatic-speech-recognition", "audio-classification"])
class SEWOpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager("sew-d", *["feature-extraction", "automatic-speech-recognition", "audio-classification"])
class SEWDOpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager(
    "unispeech", *["feature-extraction", "automatic-speech-recognition", "audio-classification"]
)
class UniSpeechOpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager(
    "unispeech-sat",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class UniSpeechSATOpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager(
    "wavlm",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class WavLMOpenVINOConfig(HubertOpenVINOConfig):
    pass


@register_in_tasks_manager("sam", *["feature-extraction"])
class SamOpenVINOConfig(OpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedEncoderDecoderConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyPointsGenerator, DummyVisionEmbeddingsGenerator)
    VARIANTS = {  # noqa: RUF012
        "monolith": "All the SAM model components are exported as a single model.",
        "split": "The vision encoder is exported as a separate vision_encoder, and the prompt encoder and mask decoder are exported as a prompt_encoder_mask_decoder. This allows to encoder the image only once for multiple point queries.",
    }
    DEFAULT_VARIANT = "split"
    _MODEL_PATCHER = SAMModelPatcher

    def __init__(
        self,
        config,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        variant: str = "split",
        vision_encoder=None,
        preprocessors=None,
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
    def inputs(self) -> dict:
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
    def outputs(self) -> dict:
        if self.variant == "split" and self.vision_encoder:
            return {"image_embeddings": {0: "batch_size"}, "image_positional_embeddings": {0: "batch_size"}}
        else:
            return {
                "iou_scores": {0: "batch_size", 1: "point_batch_size"},
                "pred_masks": {0: "batch_size", 1: "point_batch_size"},
            }


@register_in_tasks_manager("siglip", *["feature-extraction", "zero-shot-image-classification"])
class SiglipOpenVINOConfig(TextAndVisionOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = SiglipNormalizedConfig
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> dict:
        return {
            "input_ids": {0: "text_batch_size", 1: "sequence_length"},
            "pixel_values": {0: "image_batch_size", 1: "num_channels", 2: "height", 3: "width"},
            # NOTE: No attention_mask
        }

    @property
    def outputs(self) -> dict:
        if self.task in ["feature-extraction", "zero-shot-image-classification"]:
            return {
                "logits_per_image": {0: "image_batch_size", 1: "text_batch_size"},
                "logits_per_text": {0: "text_batch_size", 1: "image_batch_size"},
                "text_embeds": {0: "text_batch_size"},
                "image_embeds": {0: "image_batch_size"},
            }
        else:
            return super().outputs


@register_in_tasks_manager(
    "transformer", *["feature-extraction", "sentence-similarity"], library_name="sentence_transformers"
)
class SentenceTransformersTransformerOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    _MODEL_PATCHER = SentenceTransformersTransformerPatcher

    @property
    def inputs(self) -> dict:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict:
        return {
            "token_embeddings": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"},
        }


@register_in_tasks_manager("rembert", *COMMON_TEXT_TASKS)
class RemBertOpenVINOConfig(BertOpenVINOConfig):
    pass


@register_in_tasks_manager("siglip-text-with-projection", *["feature-extraction"])
class SiglipTextWithProjectionOpenVINOConfig(TextEncoderOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        num_layers="num_hidden_layers",
        allow_new=True,
    )
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def inputs(self) -> dict:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict:
        common_outputs = {
            "text_embeds": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }
        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs


@register_in_tasks_manager("siglip-text", *["feature-extraction"])
class SiglipTextOpenVINOConfig(SiglipTextWithProjectionOpenVINOConfig):
    _MODEL_PATCHER = CLIPModelPatcher

    @property
    def outputs(self) -> dict:
        common_outputs = {
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output": {0: "batch_size"},
        }

        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs[f"hidden_states.{i}"] = {0: "batch_size", 1: "sequence_length"}

        return common_outputs


class VideoChatFlashQwenProjectorOpenVINOConfig(OpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVideoChatFlashQwenProjectorInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"input": {0: "batch_size", 1: "num_patches", 2: "hidden_size"}}


class VideoChatFlashQwenConfigBehavior(str, enum.Enum):
    LANGUAGE = "language"
    VISION_EMBEDDINGS = "vision_embeddings"
    VISION_PROJECTION = "vision_projection"
    TEXT_EMBEDDINGS = "text_embeddings"


@register_in_tasks_manager("videochat_flash_qwen", *["image-text-to-text"], library_name="transformers")
class VideoChatFlashQwenOpenVINOConfig(BaseVLMOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.49.0"
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in VideoChatFlashQwenConfigBehavior]
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVideoChatFlashQwenInputGenerator,)

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: VideoChatFlashQwenConfigBehavior = VideoChatFlashQwenConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            behavior=behavior,
            preprocessors=preprocessors,
        )
        self._orig_config = config
        self._model_config_prepared = False

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if not self._behavior == VideoChatFlashQwenConfigBehavior.VISION_EMBEDDINGS:
            return {}
        return {
            "hidden_states": {0: "batch_size", 2: "num_frames", 3: "height", 4: "width"},
            # rotary_pos_emb has a fixed leading dimension of 1 in the dummy generator,
            # so we do not associate axis 0 with batch_size and keep only dynamic axes here.
            "rotary_pos_emb": {1: "num_tokens", 2: "hidden_size"},
        }

    def with_behavior(
        self,
        behavior: Union[str, VideoChatFlashQwenConfigBehavior],
    ):
        """
        Creates a config for different behaviour.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, VideoChatFlashQwenConfigBehavior):
            behavior = VideoChatFlashQwenConfigBehavior(behavior)

        if behavior == VideoChatFlashQwenConfigBehavior.VISION_PROJECTION:
            export_config = VideoChatFlashQwenProjectorOpenVINOConfig(
                self._orig_config,
                task="feature-extraction",
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
            )
            return export_config

        if behavior == VideoChatFlashQwenConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config("qwen2", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == VideoChatFlashQwenConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config("qwen2", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == VideoChatFlashQwenConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    def get_model_for_behavior(self, model, behavior: Union[str, VideoChatFlashQwenConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, VideoChatFlashQwenConfigBehavior):
            behavior = VideoChatFlashQwenConfigBehavior(behavior)

        if not self._model_config_prepared:
            vision_tower = model.get_vision_tower()
            model.config.mm_num_attention_heads = vision_tower.config.num_attention_heads
            # num_tome_tokens=64 comes from the upstream projector_type "tome16_mlp_hd64", which uses a fixed 64-token output.
            # Source: https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B/blob/main/mm_projector_builder.py#L135-L146
            model.config.mm_projector_num_tome_tokens = 64
            model.config.patch_size = vision_tower.config.patch_size
            model.config.image_size = vision_tower.config.image_size
            model.config.image_mean = vision_tower.image_processor.image_mean
            model.config.image_std = vision_tower.image_processor.image_std
            self._model_config_prepared = True

        if behavior == VideoChatFlashQwenConfigBehavior.VISION_PROJECTION:
            vision_projector = model.get_model().mm_projector.mlp
            vision_projector.config = model.config
            return vision_projector

        if behavior == VideoChatFlashQwenConfigBehavior.VISION_EMBEDDINGS:
            vision_tower = model.get_vision_tower().vision_tower
            vision_tower.config = model.config
            return vision_tower

        if behavior == VideoChatFlashQwenConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.get_input_embeddings()
            text_embedding.config = model.config
            return text_embedding

        if behavior == VideoChatFlashQwenConfigBehavior.LANGUAGE:
            model.model.llm_compress_layer_list = []
            return model

    def patch_model_for_export(self, model: PreTrainedModel, model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}

        if self._behavior == VideoChatFlashQwenConfigBehavior.VISION_EMBEDDINGS:
            return VideoChatFlashQwenVisionEmbeddingModelPatcher(self, model, model_kwargs)

        return super().patch_model_for_export(model, model_kwargs)


@register_in_tasks_manager(
    "hunyuan_v1_dense",
    *[
        "text-generation",
        "text-generation-with-past",
    ],
    library_name="transformers",
)
class HunyuanV1DenseOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.57.0"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator


@register_in_tasks_manager(
    "qwen3_next",
    *["text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class Qwen3NextOpenVINOConfig(Qwen3OpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, Qwen3NextDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = Qwen3NextDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "4.57.0"
    MAX_TRANSFORMERS_VERSION = "4.57.6"
    _MODEL_PATCHER = Qwen3NextModelPatcher

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            cache_name_prefix = "cache_params.past"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            cache_name_prefix = "cache_params.present"

        self.num_full_attn_layers = self._normalized_config.layer_types.count("full_attention")
        self.num_linear_attn_layers = self._normalized_config.layer_types.count("linear_attention")

        for i in range(self.num_linear_attn_layers):
            inputs_or_outputs[f"{cache_name_prefix}.conv.{i}"] = {0: "batch_size"}
            inputs_or_outputs[f"{cache_name_prefix}.ssm.{i}"] = {0: "batch_size"}

        for i in range(self.num_full_attn_layers):
            inputs_or_outputs[f"{cache_name_prefix}.key.{i}"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{cache_name_prefix}.value.{i}"] = {0: "batch_size", 2: decoder_sequence_name}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        # need to override `generate_dummy_inputs` since mamba model has other states: ssm_states and conv_states
        # which we separate and call them as past_ssm_states and past_conv_states
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs.keys() if not key.startswith("cache_params")]
        if self.use_past_in_inputs:
            input_names.extend(["cache_params"])

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
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model OpenVINO config.'
                )

        return dummy_inputs


@register_in_tasks_manager(
    "lfm2_moe",
    *[
        "text-generation",
        "text-generation-with-past",
    ],
    library_name="transformers",
)
class LFM2MoeOpenVINOConfig(LFM2OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "5.0"
    _MODEL_PATCHER = Lfm2MoeModelPatcher


@register_in_tasks_manager(
    "qwen3_5_text",
    *["text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class Qwen3_5TextOpenVINOConfig(Qwen3VLTextOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, Qwen3_5DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = Qwen3_5DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    MIN_TRANSFORMERS_VERSION = "5.2.0"
    MAX_TRANSFORMERS_VERSION = "5.2.99"
    _MODEL_PATCHER = Qwen3_5ModelPatcher

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            cache_name_prefix = "cache_params.past"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            cache_name_prefix = "cache_params.present"

        self.num_full_attn_layers = self._normalized_config.layer_types.count("full_attention")
        self.num_linear_attn_layers = self._normalized_config.layer_types.count("linear_attention")

        for i in range(self.num_linear_attn_layers):
            inputs_or_outputs[f"{cache_name_prefix}.conv.{i}"] = {0: "batch_size"}
            inputs_or_outputs[f"{cache_name_prefix}.ssm.{i}"] = {0: "batch_size"}

        for i in range(self.num_full_attn_layers):
            inputs_or_outputs[f"{cache_name_prefix}.key.{i}"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{cache_name_prefix}.value.{i}"] = {0: "batch_size", 2: decoder_sequence_name}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
        }
        if self.use_past_in_inputs:
            self.add_past_key_values(common_inputs, direction="inputs")
        return common_inputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs.keys() if not key.startswith("cache_params")]
        if self.use_past_in_inputs:
            input_names.extend(["cache_params"])

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
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model OpenVINO config.'
                )

        return dummy_inputs


@register_in_tasks_manager(
    "qwen3_5",
    *["image-text-to-text"],
    library_name="transformers",
)
class Qwen3_5OpenVINOConfig(Qwen3VLOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in QwenVLConfigBehavior]
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyQwen3VLVisionEmbedInputGenerator,)
    MIN_TRANSFORMERS_VERSION = "5.2.0"
    MAX_TRANSFORMERS_VERSION = "5.2.99"

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: QwenVLConfigBehavior = QwenVLConfigBehavior.VISION_EMBEDDINGS,
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
            behavior=behavior,
        )
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_POS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self._normalized_config.use_embed_dim = True

    def with_behavior(
        self,
        behavior: Union[str, QwenVLConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, QwenVLConfigBehavior):
            behavior = QwenVLConfigBehavior(behavior)

        if behavior == QwenVLConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config(
                "qwen3_5_text", self._orig_config.text_config, self.int_dtype, self.float_dtype
            )

        if behavior == QwenVLConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config(
                "qwen3_5_text",
                self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Qwen3_5ModelPatcher,
                dummy_input_generator=DummyQwen3_5LMInputGenerator,
                inputs_update={"position_ids": {1: "batch_size", 2: "sequence_length"}},
            )

        if behavior in (
            QwenVLConfigBehavior.VISION_EMBEDDINGS,
            QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER,
            QwenVLConfigBehavior.VISION_EMBEDDINGS_POS,
        ):
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    def patch_model_for_export(self, model: Union["PreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None):
        model_kwargs = model_kwargs or {}
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return Qwen3_5VisionEmbMergerPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS:
            return super().outputs
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return {"last_hidden_state": {0: "seq_len"}}
        if self._behavior == QwenVLConfigBehavior.VISION_EMBEDDINGS_POS:
            return {"last_hidden_state": {0: "seq_len", 1: "seq_len"}}
        if self._behavior == QwenVLConfigBehavior.TEXT_EMBEDDINGS:
            return {"inputs_embeds": {0: "batch_size", 1: "sequence_length"}}
        if self._behavior == QwenVLConfigBehavior.LANGUAGE:
            return get_vlm_internal_text_generation_config(
                "qwen3_5_text", self._orig_config.text_config, self.int_dtype, self.float_dtype
            ).outputs
        raise Exception("Unknown Qwen3.5 behavior type.")


@register_in_tasks_manager(
    "qwen3_5_moe_text",
    *["text-generation", "text-generation-with-past"],
    library_name="transformers",
)
class Qwen3_5MoeTextOpenVINOConfig(Qwen3_5TextOpenVINOConfig):
    _MODEL_PATCHER = Qwen3_5MoeModelPatcher


@register_in_tasks_manager(
    "qwen3_5_moe",
    *["image-text-to-text"],
    library_name="transformers",
)
class Qwen3_5MoeOpenVINOConfig(Qwen3_5OpenVINOConfig):
    def with_behavior(
        self,
        behavior: Union[str, QwenVLConfigBehavior],
    ):
        if isinstance(behavior, str) and not isinstance(behavior, QwenVLConfigBehavior):
            behavior = QwenVLConfigBehavior(behavior)

        if behavior == QwenVLConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config(
                "qwen3_5_moe_text", self._orig_config.text_config, self.int_dtype, self.float_dtype
            )

        if behavior == QwenVLConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config(
                "qwen3_5_moe_text",
                self._orig_config.text_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Qwen3_5MoeModelPatcher,
                dummy_input_generator=DummyQwen3_5LMInputGenerator,
                inputs_update={"position_ids": {1: "batch_size", 2: "sequence_length"}},
            )

        if behavior in (
            QwenVLConfigBehavior.VISION_EMBEDDINGS,
            QwenVLConfigBehavior.VISION_EMBEDDINGS_MERGER,
            QwenVLConfigBehavior.VISION_EMBEDDINGS_POS,
        ):
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == QwenVLConfigBehavior.LANGUAGE:
            return get_vlm_internal_text_generation_config(
                "qwen3_5_moe_text", self._orig_config.text_config, self.int_dtype, self.float_dtype
            ).outputs
        return super().outputs


@register_in_tasks_manager(
    "kokoro",
    *["text-to-audio"],
    library_name="kokoro",
)
class KokoroOpenVINOConfig(OpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyKokoroInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    _MODEL_PATCHER = KokoroModelPatcher

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "text-to-audio",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        preprocessors: Optional[List[Any]] = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {1: ("sequence_length", 2, -1)},
            "ref_s": {1: "style_dim"},
            "speed": {},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "waveform": {0: "batch_size", 1: "audio_length"},
            "phonemes": {0: "batch_size", 1: "phoneme_length"},
        }


@register_in_tasks_manager(
    "trocr",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
    ],
)
class TrOCROpenVINOConfig(TextSeq2SeqOpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        decoder_num_layers="decoder_layers",
        num_layers="decoder_layers",
        decoder_num_attention_heads="decoder_attention_heads",
        hidden_size="hidden_size",
    )
