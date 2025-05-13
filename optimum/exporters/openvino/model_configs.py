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
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from packaging import version
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, TFPreTrainedModel
from transformers.utils import is_tf_available

from optimum.exporters.onnx.base import ConfigBehavior
from optimum.exporters.onnx.config import OnnxConfig, TextDecoderOnnxConfig, TextDecoderWithPositionIdsOnnxConfig
from optimum.exporters.onnx.model_configs import (
    BlenderbotOnnxConfig,
    BlenderbotSmallOnnxConfig,
    BloomOnnxConfig,
    CLIPOnnxConfig,
    CLIPTextOnnxConfig,
    CLIPTextWithProjectionOnnxConfig,
    CLIPVisionModelOnnxConfig,
    CodeGenOnnxConfig,
    FalconOnnxConfig,
    GemmaOnnxConfig,
    GPTBigCodeOnnxConfig,
    GPTJOnnxConfig,
    GPTNeoOnnxConfig,
    GPTNeoXOnnxConfig,
    IBertOnnxConfig,
    LlamaOnnxConfig,
    MarianOnnxConfig,
    MistralOnnxConfig,
    MPTOnnxConfig,
    PegasusOnnxConfig,
    PhiOnnxConfig,
    SpeechT5OnnxConfig,
    T5OnnxConfig,
    UNetOnnxConfig,
    VaeDecoderOnnxConfig,
    VaeEncoderOnnxConfig,
    VisionOnnxConfig,
    WhisperOnnxConfig,
)
from optimum.exporters.onnx.model_patcher import ModelPatcher
from optimum.exporters.tasks import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.utils.input_generators import (
    DTYPE_MAPPER,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyVisionInputGenerator,
    FalconDummyPastKeyValuesGenerator,
    GemmaDummyPastKeyValuesGenerator,
    MistralDummyPastKeyValuesGenerator,
)
from optimum.utils.normalized_config import NormalizedConfig, NormalizedTextConfig, NormalizedVisionConfig

from ...intel.utils.import_utils import (
    _transformers_version,
    is_diffusers_available,
    is_diffusers_version,
    is_transformers_version,
)
from .model_patcher import (
    AquilaModelPatcher,
    ArcticModelPatcher,
    BaichuanModelPatcher,
    BlenderbotModelPatcher,
    BlenderbotSmallModelPatcher,
    BloomModelPatcher,
    ChatGLMModelPatcher,
    CodeGenModelPatcher,
    CommonImageEmbeddingsModelPatcher,
    DBRXModelPatcher,
    DeciLMModelPatcher,
    DeepseekPatcher,
    FalconModelPatcher,
    FluxTransfromerModelPatcher,
    Gemma2ModelPatcher,
    Gemma3LMModelPatcher,
    GptBigCodeModelPatcher,
    GptJModelPatcher,
    GptNeoModelPatcher,
    GptNeoxJapaneseModelPatcher,
    GptNeoxModelPatcher,
    GraniteMoEModelPatcher,
    IBertModelPatcher,
    Idefics3ImageEmbeddingsModelPatcher,
    InputEmbeddingPatcher,
    InternLM2Patcher,
    InternLMModelPatcher,
    InternVL2ChatLangModelPatcher,
    InternVLChatImageEmbeddingModelPatcher,
    JaisModelPatcher,
    Llama4ImageEmbeddingsModelPatcher,
    Llama4TextModelPatcher,
    LlamaModelPatcher,
    LlavaImageEmbeddingModelPatcher,
    LlavaNextVideoImageEmbeddingModelPatcher,
    LlavaQwen2ImageEmbeddingsModelPatcher,
    MarianModelPatcher,
    MiniCPM3Patcher,
    MiniCPMModelPatcher,
    MiniCPMVImageEmbeddingsModelPatcher,
    MiniCPMVResamplerModelPatcher,
    MistralModelPatcher,
    MixtralModelPatcher,
    MPTModelPatcher,
    OVSpeechT5ModelPatcher,
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
    Qwen2VLLanguageModelPatcher,
    Qwen2VLVisionEmbMergerPatcher,
    QwenModelPatcher,
    RotaryEmbPatcher,
    SanaTextEncoderModelPatcher,
    StatefulSeq2SeqDecoderPatcher,
    UpdateCausalMaskModelPatcher,
    XverseModelPatcher,
)


def init_model_configs():
    if "open_clip" not in TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES:
        TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES["open_clip"] = {}
    TasksManager._CUSTOM_CLASSES[("pt", "llava", "image-text-to-text")] = (
        "transformers",
        "LlavaForConditionalGeneration",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "llava-next", "image-text-to-text")] = (
        "transformers",
        "LlavaNextForConditionalGeneration",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "qwen2-vl", "image-text-to-text")] = (
        "transformers",
        "Qwen2VLForConditionalGeneration",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "qwen2-5-vl", "image-text-to-text")] = (
        "transformers",
        "AutoModelForImageTextToText",
    )

    TasksManager._CUSTOM_CLASSES[("pt", "llava-next-video", "image-text-to-text")] = (
        "transformers",
        "AutoModelForVision2Seq",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "gemma3", "image-text-to-text")] = (
        "transformers",
        "Gemma3ForConditionalGeneration",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "idefics3", "image-text-to-text")] = (
        "transformers",
        "AutoModelForImageTextToText",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "smolvlm", "image-text-to-text")] = (
        "transformers",
        "AutoModelForImageTextToText",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "phi4mm", "image-text-to-text")] = ("transformers", "AutoModelForCausalLM")
    TasksManager._CUSTOM_CLASSES[("pt", "phi4mm", "automatic-speech-recognition")] = (
        "transformers",
        "AutoModelForCausalLM",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "phi4-multimodal", "image-text-to-text")] = (
        "transformers",
        "AutoModelForCausalLM",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "phi4-multimodal", "automatic-speech-recognition")] = (
        "transformers",
        "AutoModelForCausalLM",
    )
    TasksManager._CUSTOM_CLASSES[("pt", "llama4", "image-text-to-text")] = (
        "transformers",
        "AutoModelForImageTextToText",
    )

    TasksManager._TRANSFORMERS_TASKS_TO_MODEL_LOADERS[
        "image-text-to-text"
    ] = TasksManager._TRANSFORMERS_TASKS_TO_MODEL_LOADERS["text-generation"]

    TasksManager._TRANSFORMERS_TASKS_TO_MODEL_LOADERS["video-text-to-text"] = "AutoModelForVision2Seq"

    if is_diffusers_available() and "fill" not in TasksManager._DIFFUSERS_TASKS_TO_MODEL_LOADERS:
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_LOADERS["fill"] = "FluxFillPipeline"
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["fill"] = {"flux": "FluxFillPipeline"}
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_LOADERS["text-to-image"] = ("AutoPipelineForText2Image", "SanaPipeline")
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-image"]["sana"] = "SanaPipeline"
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-image"]["sana-sprint"] = "SanaSprintPipeline"
    if is_diffusers_available() and "text-to-video" not in TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS:
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-video"] = {}
        TasksManager._DIFFUSERS_TASKS_TO_MODEL_MAPPINGS["text-to-video"]["ltx-video"] = "LTXPipeline"

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
    from transformers.modeling_utils import PreTrainedModel  # noqa: F811

    from optimum.exporters.onnx.model_patcher import ModelPatcher  # noqa: F811

    if is_tf_available():
        from transformers.modeling_tf_utils import TFPreTrainedModel  # noqa: F811


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
class Qwen2OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return UpdateCausalMaskModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("qwen2-moe", *["text-generation", "text-generation-with-past"], library_name="transformers")
class Qwen2MoEOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return UpdateCausalMaskModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("qwen3", *["text-generation", "text-generation-with-past"], library_name="transformers")
@register_in_tasks_manager("qwen3-moe", *["text-generation", "text-generation-with-past"], library_name="transformers")
class Qwen3OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    MIN_TRANSFORMERS_VERSION = "4.51.0"

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return UpdateCausalMaskModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("minicpm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MiniCPMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MiniCPMModelPatcher(self, model, model_kwargs=model_kwargs)


class OVMiniCPM3DummyPastKeyValuesGenerator(MistralDummyPastKeyValuesGenerator):
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
        self.v_head_dim = getattr(normalized_config, "v_head_dim", self.hidden_size // self.num_attention_heads)
        self.k_head_dim = normalized_config.qk_nope_head_dim + normalized_config.qk_rope_head_dim

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        v_shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.v_head_dim,
        )
        k_shape = (self.batch_size, self.num_key_value_heads, self.sequence_length, self.k_head_dim)
        return [
            (
                self.random_float_tensor(k_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(v_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


@register_in_tasks_manager("minicpm3", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MiniCPM3OpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, OVMiniCPM3DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = OVMiniCPM3DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return MiniCPM3Patcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("stablelm", *["text-generation", "text-generation-with-past"], library_name="transformers")
class StableLMOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MistralDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return UpdateCausalMaskModelPatcher(self, model, model_kwargs=model_kwargs)


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
        self.standart_cache_layout = hasattr(normalized_config, "rope_ratio")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if not self.standart_cache_layout:
            pkv_shape = (
                self.sequence_length,
                self.batch_size,
                self.multi_query_group_num,
                self.head_dim,
            )
        else:
            pkv_shape = (
                self.batch_size,
                self.multi_query_group_num,
                self.sequence_length,
                self.head_dim,
            )
        return [
            (
                self.random_float_tensor(pkv_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(pkv_shape, framework=framework, dtype=float_dtype),
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
        return LlamaModelPatcher(self, model, model_kwargs=model_kwargs)


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
    pass


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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return UpdateCausalMaskModelPatcher(self, model, model_kwargs=model_kwargs)


def patch_model_for_export(
    self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
) -> "ModelPatcher":
    return RotaryEmbPatcher(self, model, model_kwargs=model_kwargs)


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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return PhiMoEModelPatcher(self, model, model_kwargs=model_kwargs)


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
class PhiOpenVINOConfig(PhiOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return UpdateCausalMaskModelPatcher(self, model, model_kwargs=model_kwargs)


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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return FalconModelPatcher(self, model, model_kwargs=model_kwargs)


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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return GptNeoxJapaneseModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "gpt-neo",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GPTNeoOpenVINOConfig(GPTNeoOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return GptNeoModelPatcher(self, model, model_kwargs=model_kwargs)


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
class GPTJOpenVINOConfig(GPTJOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return GptJModelPatcher(self, model, model_kwargs=model_kwargs)


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
class BloomOpenVINOConfig(BloomOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return BloomModelPatcher(self, model, model_kwargs=model_kwargs)


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


class OVMistralDummyPastKeyValuesGenerator(MistralDummyPastKeyValuesGenerator):
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
        self.head_dim = getattr(normalized_config, "head_dim", self.hidden_size // self.num_attention_heads)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


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
class MistralOpenVINOConfig(MistralOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        OVMistralDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = OVMistralDummyPastKeyValuesGenerator

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MistralModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "gpt-neox",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GPTNeoxOpenVINOConfig(GPTNeoXOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return GptNeoxModelPatcher(self, model, model_kwargs=model_kwargs)


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
class Gemma2OpenVINOConfig(GemmaOnnxConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.43.0")

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return Gemma2ModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "gemma3-text",
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
    MIN_TRANSFORMERS_VERSION = version.parse("4.50.0")


class DeciDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        self.num_key_value_heads_per_layer = normalized_config.num_key_value_heads_per_layer

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_values = []

        for layer_id in range(self.num_layers):
            shape = (
                self.batch_size,
                self.num_key_value_heads_per_layer[layer_id],
                self.sequence_length,
                self.hidden_size // self.num_attention_heads,
            )
            past_key_values.append(
                (
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                )
            )
        return past_key_values


@register_in_tasks_manager("deci", *["text-generation", "text-generation-with-past"], library_name="transformers")
class DeciOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DeciDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DeciDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return DeciLMModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("clip", *["zero-shot-image-classification"], library_name="open_clip")
class OpenCLIPOpenVINOConfig(CLIPOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

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
        self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if "attention_mask" in reference_model_inputs:
            reference_model_inputs.pop("attention_mask")
        if "image" in onnx_input_names and "pixel_values" in reference_model_inputs:
            reference_model_inputs["image"] = reference_model_inputs.pop("pixel_values")
        if "text" in onnx_input_names and "input_ids" in reference_model_inputs:
            reference_model_inputs["text"] = reference_model_inputs.pop("input_ids")
        return super().generate_dummy_inputs_for_validation(reference_model_inputs)

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return ModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("clip-text-model", *["feature-extraction"], library_name="open_clip")
class OpenCLIPTextOpenVINOConfig(CLIPTextOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return ModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("clip-vision-model", *["feature-extraction"], library_name="open_clip")
class OpenCLIPVisualOpenVINOConfig(VisionOnnxConfig):
    DEFAULT_ONNX_OPSET = 14

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
class CLIPOpenVINOConfig(CLIPOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return ModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("clip-text-model", *["feature-extraction"], library_name="transformers")
@register_in_tasks_manager("clip-text-model", *["feature-extraction"], library_name="diffusers")
@register_in_tasks_manager("clip-text", *["feature-extraction"], library_name="diffusers")
class CLIPTextOpenVINOConfig(CLIPTextOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return ModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("clip-text-with-projection", *["feature-extraction"], library_name="transformers")
@register_in_tasks_manager("clip-text-with-projection", *["feature-extraction"], library_name="diffusers")
class CLIPTextWithProjectionOpenVINOConfig(CLIPTextWithProjectionOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return ModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("clip-vision-model", *["feature-extraction"], library_name="transformers")
class CLIPVisionModelOpenVINOConfig(CLIPVisionModelOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return ModelPatcher(self, model, model_kwargs=model_kwargs)


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
class IBertOpenVINOConfig(IBertOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return IBertModelPatcher(self, model, model_kwargs=model_kwargs)


class LMInputEmbedsConfigHelper(TextDecoderWithPositionIdsOnnxConfig):
    def __init__(self, export_config, patcher_cls=None, dummy_input_generator=None, inputs_update=None):
        self.orig_export_config = export_config
        if dummy_input_generator is not None:
            export_config.DUMMY_INPUT_GENERATOR_CLASSES = (
                dummy_input_generator,
            ) + export_config.DUMMY_INPUT_GENERATOR_CLASSES
        self.DUMMY_INPUT_GENERATOR_CLASSES = export_config.DUMMY_INPUT_GENERATOR_CLASSES
        self.DEFAULT_ONNX_OPSET = export_config.DEFAULT_ONNX_OPSET
        self.DUMMY_PKV_GENERATOR_CLASS = export_config.DUMMY_PKV_GENERATOR_CLASS
        self._config = export_config._config
        self._normalized_config = export_config._normalized_config
        self.use_past = export_config.use_past
        self.patcher_cls = patcher_cls
        self.input_info_upd = inputs_update

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        if self.patcher_cls is not None:
            return self.patcher_cls(self, model, model_kwargs=model_kwargs)
        # Refer to DecoderModelPatcher.
        return self.orig_export_config.patch_model_for_export(model, model_kwargs=model_kwargs)

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return self.orig_export_config.outputs

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
        inputs_embed_shape = (input_ids.shape[0], input_ids.shape[1], self._normalized_config.hidden_size)
        inputs_embeds = self.orig_export_config.DUMMY_INPUT_GENERATOR_CLASSES[0].random_float_tensor(
            inputs_embed_shape
        )
        dummy_inputs["inputs_embeds"] = inputs_embeds
        if "token_type_ids" in self.inputs:
            dummy_inputs["token_type_ids"] = self.orig_export_config.DUMMY_INPUT_GENERATOR_CLASSES[
                0
            ].random_int_tensor(input_ids.shape, min_value=0, max_value=2)
        return dummy_inputs


class InputEmbedOpenvVINOConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        # making 16bit tracable overrides embeedings input signature these changes required to prevent this issue
        return InputEmbeddingPatcher(self, model, model_kwargs)


def get_vlm_internal_text_generation_config(model_type, model_config, int_dtype, float_dtype):
    model_type = model_type.replace("_", "-")

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
    InputEmbedOpenvVINOConfig.NORMALIZED_CONFIG_CLASS = internal_export_config.NORMALIZED_CONFIG_CLASS
    export_config = InputEmbedOpenvVINOConfig(
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
):
    internal_export_config = get_vlm_internal_text_generation_config(model_type, model_config, int_dtype, float_dtype)
    export_config = LMInputEmbedsConfigHelper(
        internal_export_config,
        patcher_cls=model_patcher,
        dummy_input_generator=dummy_input_generator,
        inputs_update=inputs_update,
    )
    export_config._normalized_config = internal_export_config._normalized_config
    return export_config


class VLMConfigBehavior(str, enum.Enum):
    VISION_EMBEDDINGS = "vision_embeddings"
    TEXT_EMBEDDINGS = "text_embeddings"
    LANGUAGE = "language"


class BaseVLMOpenVINOConfig(OnnxConfig):
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
            return model.language_model

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.get_input_embeddings()
            text_embedding.config = model.language_model.config
            return text_embedding

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return CommonImageEmbeddingsModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager("llava", *["image-text-to-text"], library_name="transformers")
class LlavaOpenVINOConfig(BaseVLMOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.37.2")

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return LlavaImageEmbeddingModelPatcher(self, model, model_kwargs)

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> Dict:
        if self._behavior == VLMConfigBehavior.VISION_EMBEDDINGS and self._config.model_type == "pixtral":
            kwargs["batch_size"] = 1
        return super().generate_dummy_inputs(framework, **kwargs)


@register_in_tasks_manager("llava-next", *["image-text-to-text"], library_name="transformers")
class LlavaNextOpenVINOConfig(LlavaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.40.0")


class DummyLLavaMultiModalProjectorInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ["image_features"]

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task

        self.batch_size = batch_size
        self.hidden_size = normalized_config.hidden_size
        self.num_patches = (normalized_config.image_size // normalized_config.patch_size) ** 2
        self.normalized_config = normalized_config

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        shape = [self.batch_size, self.num_patches, self.hidden_size]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class LLavaMultimodalProjectorOpenVINOConfig(OnnxConfig):
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


@register_in_tasks_manager(
    "llava-next-video", *["image-text-to-text", "video-text-to-text"], library_name="transformers"
)
class LlavaNextVideoOpenVINOConfig(LlavaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.42.0")
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
            return model.multi_modal_projector

        if behavior == LlavaNextVideoConfigBehavior.VISION_RESAMPLER:
            return model.vision_resampler

        return super().get_model_for_behavior(model, behavior)

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior != LlavaNextVideoConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return LlavaNextVideoImageEmbeddingModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager(
    "maira2", *["image-text-to-text", "text-generation", "text-generation-with-past"], library_name="transformers"
)
class MairaOpenVINOConfig(LlavaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.46.0")
    SUPPORTS_PAST = True


@register_in_tasks_manager("internvl-chat", *["image-text-to-text"], library_name="transformers")
class InternVLChatOpenVINOConfig(BaseVLMOpenVINOConfig):
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
            return model.language_model

        if behavior == VLMConfigBehavior.VISION_EMBEDDINGS:
            return model

        if behavior == VLMConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.language_model.get_input_embeddings()
            text_embedding.config = model.language_model.config
            return text_embedding

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return InternVLChatImageEmbeddingModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager(
    "llava-qwen2", *["image-text-to-text", "text-generation", "text-generation-with-past"], library_name="transformers"
)
class LlavaQwen2OpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTS_PAST = True
    MIN_TRANSFORMERS_VERSION = version.parse("4.40.0")

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
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


class PooledProjectionsDummyInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ["pooled_projections"]

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        self.batch_size = batch_size
        self.pooled_projection_dim = normalized_config.config.pooled_projection_dim

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size, self.pooled_projection_dim]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


class DummyTransformerTimestpsInputGenerator(DummyTimestepInputGenerator):
    SUPPORTED_INPUT_NAMES = ("timestep", "text_embeds", "time_ids", "timestep_cond", "guidance")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["timestep", "guidance"]:
            shape = [self.batch_size]
            return self.random_float_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)
        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyUnetVisionInputGenerator(DummyVisionInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name not in ["sample", "latent_sample"]:
            return super().generate(input_name, framework, int_dtype, float_dtype)
        # add height and width discount for enable any resolution generation
        return self.random_float_tensor(
            shape=[self.batch_size, self.num_channels, self.height - 1, self.width - 1],
            framework=framework,
            dtype=float_dtype,
        )


class DummyUnetTimestepInputGenerator(DummyTimestepInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name != "timestep":
            return super().generate(input_name, framework, int_dtype, float_dtype)
        shape = [self.batch_size]
        return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=int_dtype)


class DummySanaTimestepInputGenerator(DummyTimestepInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name != "timestep":
            return super().generate(input_name, framework, int_dtype, float_dtype)
        shape = [self.batch_size]
        return self.random_int_tensor(shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype)


class DummyUnetEncoderInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        random_sequence_length_range: Optional[Tuple[int, int]] = None,
        random_num_choices_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_choices=num_choices,
            random_batch_size_range=random_batch_size_range,
            random_sequence_length_range=random_sequence_length_range,
            random_num_choices_range=random_num_choices_range,
            **kwargs,
        )
        if hasattr(normalized_config.config, "model_max_length"):
            self.sequence_length = normalized_config.config.model_max_length


@register_in_tasks_manager("unet", *["semantic-segmentation"], library_name="diffusers")
@register_in_tasks_manager("unet-2d-condition", *["semantic-segmentation"], library_name="diffusers")
class UNetOpenVINOConfig(UNetOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyUnetVisionInputGenerator,
        DummyUnetTimestepInputGenerator,
        DummyUnetEncoderInputGenerator,
    )

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs
        common_inputs["timestep"] = {0: "batch_size"}
        if hasattr(self._normalized_config.config, "model_max_length"):
            common_inputs["encoder_hidden_states"] = {0: "batch_size"}
        return common_inputs


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
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return SanaTextEncoderModelPatcher(self, model, model_kwargs)


class DummySanaSeq2SeqDecoderTextWithEncMaskInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_outputs",
        "encoder_hidden_states",
        "encoder_attention_mask",
    )


class DummySanaTransformerVisionInputGenerator(DummyUnetVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
        "sample",
        "latent_sample",
        "guidance",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"] // 8,
        height: int = DEFAULT_DUMMY_SHAPES["height"] // 8,
        # Reduce img shape by 4 for FLUX to reduce memory usage on conversion
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width=width, height=height, **kwargs)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "guidance":
            return self.random_float_tensor([self.batch_size], framework=framework, dtype=float_dtype)
        return super().generate(input_name, framework, int_dtype, float_dtype)


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


@register_in_tasks_manager("dcae-encoder", *["semantic-segmentation"], library_name="diffusers")
class DcaeEncoderOpenVINOConfig(VaeEncoderOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "latent": {0: "batch_size", 2: "height_latent", 3: "width_latent"},
        }


class DummyFluxTransformerInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "pixel_values",
        "pixel_mask",
        "sample",
        "latent_sample",
        "hidden_states",
        "img_ids",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"] // 4,
        height: int = DEFAULT_DUMMY_SHAPES["height"] // 4,
        # Reduce img shape by 4 for FLUX to reduce memory usage on conversion
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        if getattr(normalized_config, "in_channels", None):
            self.num_channels = normalized_config.in_channels // 4

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["hidden_states", "sample"]:
            shape = [self.batch_size, (self.height // 2) * (self.width // 2), self.num_channels * 4]
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        if input_name == "img_ids":
            img_ids_height = self.height // 2
            img_ids_width = self.width // 2
            return self.random_int_tensor(
                (
                    [self.batch_size, img_ids_height * img_ids_width, 3]
                    if is_diffusers_version("<", "0.31.0")
                    else [img_ids_height * img_ids_width, 3]
                ),
                min_value=0,
                max_value=min(img_ids_height, img_ids_width),
                framework=framework,
                dtype=float_dtype,
            )

        return super().generate(input_name, framework, int_dtype, float_dtype)


class DummyFluxTextInputGenerator(DummySeq2SeqDecoderTextInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "decoder_attention_mask",
        "encoder_outputs",
        "encoder_hidden_states",
        "txt_ids",
    )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "txt_ids":
            import torch

            shape = (
                [self.batch_size, self.sequence_length, 3]
                if is_diffusers_version("<", "0.31.0")
                else [self.sequence_length, 3]
            )
            dtype = DTYPE_MAPPER.pt(float_dtype)
            return torch.full(shape, 0, dtype=dtype)
        return super().generate(input_name, framework, int_dtype, float_dtype)


@register_in_tasks_manager("flux-transformer", *["semantic-segmentation"], library_name="diffusers")
@register_in_tasks_manager("flux-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class FluxTransformerOpenVINOConfig(SD3TransformerOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestpsInputGenerator,
        DummyFluxTransformerInputGenerator,
        DummyFluxTextInputGenerator,
        PooledProjectionsDummyInputGenerator,
    )

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return FluxTransfromerModelPatcher(self, model, model_kwargs=model_kwargs)


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
class LTXVaeEncoderOpenVINOConfig(VaeEncoderOnnxConfig):
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
class LTXVaeDecoderOpenVINOConfig(VaeDecoderOnnxConfig):
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


class LTXTransformerDummyInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("hidden_states", "width", "height", "num_frames", "rope_interpolation_scale")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = 16,
        height: int = 8,
        num_frames: int = 2,
        frame_rate: int = 10,
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.vae_spatial_compression_ratio = normalized_config.config.vae_spatial_compression_ratio
        self.vae_temporal_compression_ratio = normalized_config.config.vae_temporal_compression_ratio

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        import torch

        if input_name == "hidden_states":
            return self.random_float_tensor(
                [self.batch_size, self.num_frames * self.height * self.width, self.num_channels]
            )
        if input_name == "width":
            return torch.tensor(self.width)
        if input_name == "height":
            return torch.tensor(self.height)
        if input_name == "num_frames":
            return torch.tensor(self.num_frames)
        if input_name == "rope_interpolation_scale":
            import torch

            return torch.tensor(
                [
                    self.vae_temporal_compression_ratio / self.frame_rate,
                    self.vae_spatial_compression_ratio,
                    self.vae_spatial_compression_ratio,
                ]
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)


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


class DummyMiniCPMVImageInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("pixel_values", "patch_attention_mask", "position_ids")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height)
        self.patch_size = normalized_config.config.patch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "pixel_values":
            return self.random_float_tensor(
                shape=[
                    self.batch_size,
                    self.num_channels,
                    self.patch_size,
                    (self.height * self.width) // self.patch_size,
                ],
                framework=framework,
                dtype=float_dtype,
            )

        if input_name == "patch_attention_mask":
            return self.random_int_tensor(
                shape=[self.batch_size, 1, (self.height // self.patch_size) * (self.width // self.patch_size)],
                framework=framework,
                dtype=float_dtype,
                min_value=0,
                max_value=2,
            )

        if input_name == "position_ids":
            return self.random_int_tensor(
                shape=[self.batch_size, (self.height // self.patch_size) * (self.width // self.patch_size)],
                max_value=self.patch_size,
            )


class DummyMiniCPMVResampleInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("image_feature", "pos_embed", "key_padding_mask")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height)
        self.patch_size = normalized_config.config.patch_size
        self.hidden_size = normalized_config.config.hidden_size
        self.img_hidden_size = normalized_config.config.vision_config.hidden_size
        self.feat_size = (normalized_config.config.vision_config.image_size // self.patch_size) * (
            normalized_config.config.vision_config.image_size // self.patch_size
        )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "image_feature":
            return self.random_float_tensor(
                shape=[self.batch_size, self.feat_size, self.img_hidden_size], framework=framework, dtype=float_dtype
            )

        if input_name == "key_padding_mask":
            return self.constant_tensor(
                shape=[self.batch_size, self.feat_size],
                framework=framework,
                value=1,
                dtype=DTYPE_MAPPER.pt(float_dtype),
            )

        if input_name == "pos_embed":
            return self.random_float_tensor(shape=[self.feat_size, self.batch_size, self.hidden_size])


class MiniCPMVConfigBehavior(str, enum.Enum):
    RESAMPLER = "resampler"
    LANGUAGE = "language"
    VISION_EMBEDDINGS = "vision_embeddings"
    TEXT_EMBEDDINGS = "text_embeddings"


@register_in_tasks_manager("minicpmv", *["image-text-to-text"], library_name="transformers")
class MiniCPMVOpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in MiniCPMVConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = ()

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior == MiniCPMVConfigBehavior.VISION_EMBEDDINGS:
            return MiniCPMVImageEmbeddingsModelPatcher(self, model, model_kwargs)

        if self._behavior == MiniCPMVConfigBehavior.RESAMPLER:
            return MiniCPMVResamplerModelPatcher(self, model, model_kwargs)

        return super().patch_model_for_export(model, model_kwargs)


class Phi3VisionConfigBehavior(str, enum.Enum):
    LANGUAGE = "language"
    VISION_PROJECTION = "vision_projection"
    VISION_EMBEDDINGS = "vision_embeddings"
    TEXT_EMBEDDINGS = "text_embeddings"


class DummyPhi3VisionProjectionInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = 336,
        height: int = 336,
        crop_size=336,
        **kwargs,
    ):
        self.batch_size = batch_size
        self._embed_layer_realization = (
            normalized_config.config.embd_layer["embedding_cls"]
            if hasattr(normalized_config.config, "embd_layer")
            else "image_audio"
        )
        if not hasattr(normalized_config.config, "vision_config"):
            self.image_dim_out = (
                normalized_config.config.img_processor.get(
                    "image_dim_out", normalized_config.config.img_processor.get("hidden_size")
                )
                if normalized_config.config.img_processor is not None
                else 1152
            )
            if "image_embd_layer" in normalized_config.config.embd_layer:
                self.crop_size = normalized_config.config.embd_layer["image_embd_layer"].get("crop_size", crop_size)
            else:
                self.crop_size = normalized_config.config.embd_layer.get("crop_size", crop_size)
        else:
            self.image_dim_out = normalized_config.config.vision_config.hidden_size
            self.crop_size = normalized_config.config.vision_config.crop_size
        self.height = height
        self.width = width

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        h = self.height // self.crop_size
        w = self.width // self.crop_size
        feat_size = (h * w + 1) * 144 + 1 + (h + 1) * 12
        if self._embed_layer_realization in ["linear", "image_audio"]:
            shape = [self.batch_size, feat_size, self.image_dim_out]
        else:
            shape = [self.batch_size, feat_size, self.image_dim_out * 4]
        return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


@register_in_tasks_manager("phi3-v", *["image-text-to-text"], library_name="transformers")
class Phi3VisionOpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in Phi3VisionConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    MIN_TRANSFORMERS_VERSION = version.parse("4.40.0")

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior == Phi3VisionConfigBehavior.VISION_EMBEDDINGS:
            return Phi3VisionImageEmbeddingsPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)


class DummyAudioPhi4MMInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("audio_input", "audio_feature", "audio_mask")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        signal_length=498,
        **kwargs,
    ):
        self.signal_length = signal_length
        if hasattr(normalized_config.config, "audio_processor"):
            self.audio_chunk_size = (
                signal_length // normalized_config.config.audio_processor["config"]["time_reduction"] + 1
            )
            self.input_size = normalized_config.config.audio_processor["config"]["input_size"]
            self.attention_dim = normalized_config.config.audio_processor["config"]["attention_dim"]
        else:
            self.audio_chunk_size = signal_length // normalized_config.config.audio_config.time_reduction + 1
            self.input_size = normalized_config.config.audio_config.input_size
            self.attention_dim = normalized_config.config.audio_config.hidden_size
        self.batch_size = batch_size
        self.task = task
        self.normalized_config = normalized_config

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "audio_input":
            return self.random_float_tensor(
                [self.batch_size, self.signal_length, self.input_size], framework=framework, dtype=float_dtype
            )

        if input_name == "audio_feature":
            return self.random_float_tensor(
                [self.batch_size, self.audio_chunk_size, self.attention_dim], framework=framework, dtype=float_dtype
            )

        if input_name == "audio_mask":
            return self.random_int_tensor(
                [self.batch_size, self.audio_chunk_size, self.audio_chunk_size],
                max_value=2,
                framework=framework,
                dtype="bool",
            )


class DummyVisionPositionIdsPhi4InputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("patch_position_ids", "patch_attention_mask")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        if hasattr(normalized_config.config, "vision_conifg"):
            self.patch_size = getattr(normalized_config.config.vision_config, "patch_size", 14)
        else:
            self.patch_size = 14
        self.num_patches_per_side = self.height // self.patch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "patch_position_ids":
            return self.get_vision_position_ids()
        if input_name == "patch_attention_mask":
            return self.random_int_tensor(
                [self.batch_size, self.height // self.patch_size, self.width // self.patch_size],
                framework=framework,
                dtype="bool",
                max_value=2,
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)

    def get_vision_position_ids(self):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L494-L512
        import torch

        batch_size = self.batch_size
        max_im_h, max_im_w = self.height, self.width
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )
        patch_attention_mask = torch.ones(
            [self.batch_size, self.height // self.patch_size, self.width // self.patch_size], dtype=torch.int64
        )
        patch_attention_mask[0, self.height - 2 :] = 0

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
        return position_ids


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
    "phi4-multimodal", *["image-text-to-text", "automatic-speech-recognition"], library_name="transformers"
)
class Phi4MMOpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in Phi4MMConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    MIN_TRANSFORMERS_VERSION = version.parse("4.51.0")

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
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


class DummyQwen2VLLMInputGenerator(DummyTextInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        generated_input = super().generate(input_name, framework, int_dtype, float_dtype)
        if input_name == "position_ids":
            return generated_input.unsqueeze(0).expand(3, -1, -1)
        return generated_input


class DummyQwen2VLVisionEmbedInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "hidden_states",
        "attention_mask",
        "window_attention_mask",
        "window_index",
        "rotary_pos_emb",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = 1,
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = 420,
        height: int = 420,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.temporal_patch_size = normalized_config.config.temporal_patch_size
        self.patch_size = normalized_config.config.patch_size
        if normalized_config.use_embed_dim:
            self.embed_dim = (
                normalized_config.config.embed_dim
                if hasattr(normalized_config.config, "embed_dim")
                else normalized_config.hidden_size
            )
        else:
            self.embed_dim = self.num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        self.num_heads = normalized_config.config.num_heads
        self.spatial_merge_size = None
        if hasattr(normalized_config.config, "spatial_merge_size"):
            self.spatial_merge_size = normalized_config.config.spatial_merge_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        grid_h, grid_w = self.height // self.patch_size, self.width // self.patch_size
        grid_t = self.batch_size

        if input_name == "hidden_states":
            return self.random_float_tensor(
                [grid_t * grid_h * grid_w, self.embed_dim], framework=framework, dtype=float_dtype
            )

        if input_name in ["attention_mask", "window_attention_mask"]:
            return self.random_mask_tensor(
                [1, grid_t * grid_h * grid_w, grid_t * grid_h * grid_w], framework=framework, dtype=float_dtype
            )

        if input_name == "rotary_pos_emb":
            dim = self.embed_dim // self.num_heads // 2
            return self.random_float_tensor([grid_h * grid_t * grid_w, dim], framework=framework, dtype=float_dtype)

        if input_name == "window_index":
            if self.spatial_merge_size is None:
                raise ValueError(
                    "`spatial_merge_size` parameter is not found in model config. Can not generate dummy input data for `window_index` input"
                )
            spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
            hidden_size = (grid_t * grid_h * grid_w) // spatial_merge_unit
            return self.random_int_tensor([hidden_size], max_value=hidden_size)


class Qwen2VLConfigBehavior(str, enum.Enum):
    LANGUAGE = "language"
    VISION_EMBEDDINGS = "vision_embeddings"
    VISION_EMBEDDINGS_MERGER = "vision_embeddings_merger"
    TEXT_EMBEDDINGS = "text_embeddings"


@register_in_tasks_manager("qwen2-vl", *["image-text-to-text", "video-text-to-text"], library_name="transformers")
class Qwen2VLOpenVINOConfig(BaseVLMOpenVINOConfig):
    SUPPORTED_BEHAVIORS = [model_type.value for model_type in Qwen2VLConfigBehavior]
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyQwen2VLVisionEmbedInputGenerator,)
    MIN_TRANSFORMERS_VERSION = version.parse("4.45.0")

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        behavior: Qwen2VLConfigBehavior = Qwen2VLConfigBehavior.VISION_EMBEDDINGS,
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
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self._normalized_config.use_embed_dim = False
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER and hasattr(config, "vision_config"):
            self._config = config.vision_config
            self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
            self._normalized_config.use_embed_dim = True

    @staticmethod
    def get_model_for_behavior(model, behavior: Union[str, Qwen2VLConfigBehavior]):
        if isinstance(behavior, str) and not isinstance(behavior, Qwen2VLConfigBehavior):
            behavior = Qwen2VLConfigBehavior(behavior)

        if behavior == Qwen2VLConfigBehavior.LANGUAGE:
            return model

        if behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS:
            vision_embeddings = model.visual.patch_embed
            vision_embeddings.config = model.config.vision_config
            return vision_embeddings

        if behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            vision_emb_merger = model.visual
            vision_emb_merger.config = model.config.vision_config
            return vision_emb_merger

        if behavior == Qwen2VLConfigBehavior.TEXT_EMBEDDINGS:
            text_embedding = model.model.embed_tokens
            text_embedding.config = model.config
            return text_embedding

    def with_behavior(
        self,
        behavior: Union[str, Qwen2VLConfigBehavior],
    ):
        """
        Creates a config for different behaviour.
        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
        """
        if isinstance(behavior, str) and not isinstance(behavior, Qwen2VLConfigBehavior):
            behavior = Qwen2VLConfigBehavior(behavior)

        if behavior == Qwen2VLConfigBehavior.TEXT_EMBEDDINGS:
            return get_vlm_text_embeddings_config("qwen2", self._orig_config, self.int_dtype, self.float_dtype)

        if behavior == Qwen2VLConfigBehavior.LANGUAGE:
            return get_vlm_text_generation_config(
                "qwen2",
                self._orig_config,
                self.int_dtype,
                self.float_dtype,
                model_patcher=Qwen2VLLanguageModelPatcher,
                dummy_input_generator=DummyQwen2VLLMInputGenerator,
                inputs_update={"position_ids": {1: "batch_size", 2: "sequence_length"}},
            )

        if behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )
        if behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return self.__class__(
                self._orig_config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                behavior=behavior,
                preprocessors=self._preprocessors,
            )

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return Qwen2VLVisionEmbMergerPatcher(self, model, model_kwargs)
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS:
            return ModelPatcher(self, model, model_kwargs=model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS:
            return {"hidden_states": {0: "patch_thw_grid", 1: "patch_temporal_channels"}}
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return {
                "hidden_states": {0: "sequence_length"},
                "attention_mask": {1: "sequence_length", 2: "sequence_length"},
                "rotary_pos_emb": {0: "sequence_length"},
            }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior in [Qwen2VLConfigBehavior.VISION_EMBEDDINGS, Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER]:
            return {"last_hidden_state": {0: "seq_len"}}
        return {}


@register_in_tasks_manager("qwen2-5-vl", *["image-text-to-text", "video-text-to-text"], library_name="transformers")
class Qwen2_5_VLOpenVINOConfig(Qwen2VLOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = version.parse("4.49.0")

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return {
                "hidden_states": {0: "sequence_length"},
                "attention_mask": {1: "sequence_length", 2: "sequence_length"},
                "window_attention_mask": {1: "sequence_length", 2: "sequence_length"},
                "window_index": {0: "unit_sequence_length"},
                "rotary_pos_emb": {0: "sequence_length"},
            }
        return super().inputs

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior == Qwen2VLConfigBehavior.VISION_EMBEDDINGS_MERGER:
            return Qwen2_5_VLVisionEmbMergerPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)


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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        return GraniteMoEModelPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager(
    "gpt-bigcode",
    *[
        "feature-extraction",
        "feature-extraction-with-past",
        "text-generation",
        "text-generation-with-past",
        "text-classification",
    ],
    library_name="transformers",
)
class GPTBigCodeOpenVINOConfig(GPTBigCodeOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return GptBigCodeModelPatcher(self, model, model_kwargs=model_kwargs)


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
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        if getattr(self, "stateful", False) and self._behavior == ConfigBehavior.DECODER:
            return StatefulSeq2SeqDecoderPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def inputs(self):
        common_inputs = super().inputs
        if getattr(self, "stateful", False) and self._behavior == ConfigBehavior.DECODER:
            common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

        if self._behavior is not ConfigBehavior.ENCODER and self.use_past_in_inputs:
            if is_transformers_version(">=", "4.43.0"):
                # since https://github.com/huggingface/transformers/pull/31166
                common_inputs["cache_position"] = {0: "decoder_sequence_length"}
        return common_inputs


@register_in_tasks_manager(
    "t5",
    *["feature-extraction", "feature-extraction-with-past", "text2text-generation", "text2text-generation-with-past"],
    library_name="transformers",
)
class T5OpenVINOConfig(T5OnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> ModelPatcher:
        if getattr(self, "stateful", False) and self._behavior == ConfigBehavior.DECODER:
            return StatefulSeq2SeqDecoderPatcher(self, model, model_kwargs)
        return super().patch_model_for_export(model, model_kwargs)

    @property
    def inputs(self):
        common_inputs = super().inputs
        if getattr(self, "stateful", False) and self._behavior == ConfigBehavior.DECODER:
            common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}
        return common_inputs


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
    "deepseek-v3", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
@register_in_tasks_manager(
    "deepseek-v2", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
@register_in_tasks_manager("deepseek", *["text-generation", "text-generation-with-past"], library_name="transformers")
class DeepseekOpenVINOConfig(MiniCPM3OpenVINOConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return DeepseekPatcher(self, model, model_kwargs=model_kwargs)


@register_in_tasks_manager("got-ocr2", *["image-to-text", "image-text-to-text"], library_name="transformers")
class GotOCR2OpenVINOConfig(BaseVLMOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.49.0"

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


class DummyVisionPositionIdsInputGenerator(DummyVisionInputGenerator):
    SUPPORTED_INPUT_NAMES = ("patch_attention_mask", "patch_position_ids")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        num_channels: int = DEFAULT_DUMMY_SHAPES["num_channels"],
        width: int = DEFAULT_DUMMY_SHAPES["width"],
        height: int = DEFAULT_DUMMY_SHAPES["height"],
        **kwargs,
    ):
        super().__init__(task, normalized_config, batch_size, num_channels, width, height, **kwargs)
        self.patch_size = normalized_config.config.patch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "patch_attention_mask":
            shape = [self.batch_size, self.height // self.patch_size, self.width // self.patch_size]
            return self.random_int_tensor(shape, max_value=2, framework=framework, dtype="bool")
        if input_name == "patch_position_ids":
            max_nb_patches_h, max_nb_patches_w = self.height // self.patch_size, self.width // self.patch_size
            shape = [self.batch_size, max_nb_patches_h * max_nb_patches_w]
            return self.random_int_tensor(
                shape, max_value=min(max_nb_patches_h, max_nb_patches_w), framework=framework, dtype=int_dtype
            )
        return super().generate(input_name, framework, int_dtype, float_dtype)


@register_in_tasks_manager("idefics3", *["image-text-to-text", "video-text-to-text"], library_name="transformers")
class Idefics3OpenVINOConfig(BaseVLMOpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyVisionPositionIdsInputGenerator)
    MIN_TRANSFORMERS_VERSION = "4.46.0"

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

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
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


@register_in_tasks_manager("smolvlm", *["image-text-to-text", "video-text-to-text"], library_name="transformers")
class SmolVLMOpenVINOConfig(Idefics3OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.50.0"


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
class BlenderbotOpenVINOConfig(BlenderbotOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return BlenderbotModelPatcher(self, model, model_kwargs=model_kwargs)


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
class BlenderbotSmallOpenVINOConfig(BlenderbotSmallOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return BlenderbotSmallModelPatcher(self, model, model_kwargs=model_kwargs)


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
class PegasusOpenVINOConfig(PegasusOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return PegasusModelPatcher(self, model, model_kwargs=model_kwargs)


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
class MarianOpenVINOConfig(MarianOnnxConfig):
    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return MarianModelPatcher(self, model, model_kwargs=model_kwargs)


class DummySpeechT5OpenVINOInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "inputs_embeds",
        "output_sequence",
        "speaker_embeddings",
        "spectrogram",
        "raw_spectrogram",
        "encoder_hidden_states",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        **kwargs,
    ):
        self.task = task
        self.batch_size = 1

        self.sequence_length = sequence_length
        self.speaker_embedding_dim = normalized_config.speaker_embedding_dim
        self.num_mel_bins = normalized_config.num_mel_bins
        self.reduction_factor = normalized_config.config.reduction_factor
        self.hidden_size = normalized_config.config.hidden_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name in ["output_sequence", "inputs_embeds"]:
            shape = [self.batch_size, self.sequence_length, self.num_mel_bins]
        elif input_name == "speaker_embeddings":
            shape = [self.batch_size, self.speaker_embedding_dim]
        elif input_name == "raw_spectrogram":
            shape = [self.sequence_length, self.batch_size, self.reduction_factor, self.num_mel_bins]
        elif input_name == "encoder_hidden_states":
            shape = [self.batch_size, self.sequence_length, self.hidden_size]
        elif input_name == "spectrogram":
            shape = [self.batch_size, self.sequence_length, self.num_mel_bins]
        else:
            raise ValueError(f"Unsupported input {input_name} for DummySpeechT5InputGenerator")

        return self.random_float_tensor(
            shape=shape,
            min_value=0,
            max_value=1,
            framework=framework,
            dtype=float_dtype,
        )


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
class SpeechT5OpenVINOConfig(SpeechT5OnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
        DummySpeechT5OpenVINOInputGenerator,
    )

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
        legacy: bool = False,
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
            is_postnet_and_vocoder=False,
            legacy=legacy,
        )

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ) -> "ModelPatcher":
        return OVSpeechT5ModelPatcher(self, model, model_kwargs=model_kwargs)

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
            common_inputs["input_ids"] = {1: "encoder_sequence_length"}
        elif self._behavior is SpeechT5ConfigBehavior.DECODER:
            common_inputs["inputs_embeds"] = {0: "batch_size", 1: "decoder_sequence_length"}
            common_inputs["speaker_embeddings"] = {}  # No dynamic shape here.
            common_inputs["encoder_hidden_states"] = {0: "batch_size", 1: "encoder_sequence_length"}
            common_inputs["encoder_attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}
            if self.variant == "with-past" and self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")
        elif self._behavior is SpeechT5ConfigBehavior.POSTNET:
            common_inputs["raw_spectrogram"] = {
                0: "n_spectrums",
                1: "batch_size",
            }
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


@register_in_tasks_manager(
    "llama4-text", *["text-generation", "text-generation-with-past"], library_name="transformers"
)
class Llama4TextOpenVINOConfig(LlamaOpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.51.0"
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        return Llama4TextModelPatcher(self, model, model_kwargs)


@register_in_tasks_manager(
    "llama4", *["image-text-to-text", "text-generation", "text-generation-with-past"], library_name="transformers"
)
class Llama4OpenVINOConfig(GotOCR2OpenVINOConfig):
    MIN_TRANSFORMERS_VERSION = "4.51.0"

    def patch_model_for_export(
        self, model: Union["PreTrainedModel", "TFPreTrainedModel"], model_kwargs: Optional[Dict[str, Any]] = None
    ):
        model_kwargs = model_kwargs or {}
        if self._behavior != VLMConfigBehavior.VISION_EMBEDDINGS:
            return super().patch_model_for_export(model, model_kwargs)
        return Llama4ImageEmbeddingsModelPatcher(self, model, model_kwargs)
