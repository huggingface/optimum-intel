#  Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Experimental OpenVINO export built directly on the Transformers (HF) exporter.

A deliberately small alternative to :func:`optimum.exporters.openvino.main_export`. It does *none* of
the usual optimum-intel preparation — no task-specific ``OnnxConfig``, no dummy-input generators, no
model patchers. The Transformers ``OpenVINOExporter`` (``torch.export`` based) owns tracing, IO naming
and stateful KV-cache fusion, so the whole path here is:

    load model + processor  →  build a tiny modality sample  →  hand the tensors to the exporter  →  save

Because there is no per-model export config, any model Transformers can ``torch.export`` is in scope.
Coverage keeps widening toward :func:`main_export`: text encoders/decoders, seq2seq, image, audio,
image-text-to-text (VLM) and zero-shot-image-classification (CLIP-family) tasks, with automatic task
inference from the model config. Not yet covered
(they need machinery beyond the Transformers exporter): weight/activation quantization (NNCF) and
diffusion pipelines — use the ``openvino`` command for those.

Outputs load in the matching optimum-intel ``OVModel*`` runtime classes (and, for text generation,
``openvino_genai.LLMPipeline``): text-generation as a single stateful ``openvino_model.xml`` (the
unified multi-token decode) + OpenVINO tokenizer/detokenizer + ``generation_config.json``;
seq2seq/speech-seq2seq as ``openvino_encoder_model.xml`` + a stateful ``openvino_decoder_model.xml``;
VLMs as text/vision-embeddings + a stateful language model. The runtime-specific decompositions live
in ``decomposers.py``; the default single/prefill-decode decomposition comes from Transformers.

Exposed on the CLI as ``optimum-cli export openvino-hf`` (see ``optimum.commands.export.openvino``).
"""

import contextlib
import logging
from pathlib import Path
from typing import Union

import torch
import transformers
from openvino import save_model
from transformers import AutoConfig, AutoTokenizer
from transformers.exporters import OpenVINOConfig, OpenVINOExporter
from transformers.exporters.utils import decompose_for_generation
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES

from optimum.exporters.openvino.convert import export_tokenizer

from .decomposers import decompose_encoder_decoder, decompose_vlm
from .inputs import _build_sample_inputs


logger = logging.getLogger(__name__)


# task -> (Transformers auto-model class, is the model generative, input modality). The modality
# picks both the processor and how the tiny example inputs are built. Extend as coverage grows.
_TASK_TO_AUTO_MODEL = {
    # text, non-generative (encoders)
    "feature-extraction": ("AutoModel", False, "text"),
    "fill-mask": ("AutoModelForMaskedLM", False, "text"),
    "text-classification": ("AutoModelForSequenceClassification", False, "text"),
    "token-classification": ("AutoModelForTokenClassification", False, "text"),
    "question-answering": ("AutoModelForQuestionAnswering", False, "text"),
    "multiple-choice": ("AutoModelForMultipleChoice", False, "text"),
    # text, generative
    "text-generation": ("AutoModelForCausalLM", True, "text"),
    "text2text-generation": ("AutoModelForSeq2SeqLM", True, "text"),
    # image
    "image-classification": ("AutoModelForImageClassification", False, "image"),
    "semantic-segmentation": ("AutoModelForSemanticSegmentation", False, "image"),
    "object-detection": ("AutoModelForObjectDetection", False, "image"),
    "depth-estimation": ("AutoModelForDepthEstimation", False, "image"),
    # audio
    "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq", True, "audio"),
    "audio-classification": ("AutoModelForAudioClassification", False, "audio"),
    "ctc": ("AutoModelForCTC", False, "audio"),
    "audio-frame-classification": ("AutoModelForAudioFrameClassification", False, "audio"),
    "audio-xvector": ("AutoModelForAudioXVector", False, "audio"),
    # multimodal (vision-language)
    "image-text-to-text": ("AutoModelForImageTextToText", True, "multimodal"),
    "zero-shot-image-classification": ("AutoModelForZeroShotImageClassification", False, "image-text"),
    # image-to-text (vision-encoder-decoder: vision encoder + stateful text decoder)
    "image-to-text": ("AutoModelForImageTextToText", True, "image"),
}

# modality -> the processor auto-class used to build inputs and saved alongside the IR.
_MODALITY_TO_PROCESSOR = {
    "text": "AutoTokenizer",
    "image": "AutoImageProcessor",
    "audio": "AutoFeatureExtractor",
    "multimodal": "AutoProcessor",
    "image-text": "AutoProcessor",
}

# CLIP-family dual-encoder architectures (their class name ends in plain ``Model``, so they need an
# explicit mapping rather than a ``For<Task>`` suffix) — exported as zero-shot image classification.
_ZERO_SHOT_IMAGE_ARCHITECTURES = {
    "CLIPModel",
    "SiglipModel",
    "Siglip2Model",
    "AltCLIPModel",
    "ChineseCLIPModel",
}

# Speech encoder-decoders (Whisper, Moonshine, Speech2Text, …) end in ``ForConditionalGeneration`` with
# no vision config, so without this set the ``ForConditionalGeneration`` fallback would misroute them to
# text2text-generation. Sourced from Transformers' own mapping so new speech models are covered.
_SPEECH_SEQ2SEQ_ARCHITECTURES = set(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values())

# architecture-class suffix -> task, for `task="auto"` inference from `config.architectures`.
_ARCH_SUFFIX_TO_TASK = {
    "ForCausalLM": "text-generation",
    "ForSeq2SeqLM": "text2text-generation",
    "ForMaskedLM": "fill-mask",
    "ForSequenceClassification": "text-classification",
    "ForTokenClassification": "token-classification",
    "ForQuestionAnswering": "question-answering",
    "ForMultipleChoice": "multiple-choice",
    "ForImageClassification": "image-classification",
    "ForSemanticSegmentation": "semantic-segmentation",
    "ForObjectDetection": "object-detection",
    "ForDepthEstimation": "depth-estimation",
    "ForImageTextToText": "image-text-to-text",
    "ForAudioClassification": "audio-classification",
    "ForCTC": "ctc",
    "ForAudioFrameClassification": "audio-frame-classification",
    "ForAudioXVector": "audio-xvector",
}


def _task_from_architecture(architecture: str, has_vision_config: bool = False) -> str:
    """Map a ``config.architectures[0]`` class name to an export task.

    A ``ForConditionalGeneration`` architecture is disambiguated by ``has_vision_config`` (a VLM).
    Falls back to ``feature-extraction`` (plain encoder) when nothing matches.
    """
    if architecture in _ZERO_SHOT_IMAGE_ARCHITECTURES:
        return "zero-shot-image-classification"
    if architecture in _SPEECH_SEQ2SEQ_ARCHITECTURES:
        return "automatic-speech-recognition"
    # A vision encoder-decoder loads via AutoModelForImageTextToText but is an encoder-decoder (image
    # in, text out) — export it as image-to-text so it routes through the seq2seq decomposition, not the
    # VLM one the other image-text-to-text architectures use. Pix2Struct is the same shape but feeds the
    # encoder ``flattened_patches`` instead of ``pixel_values`` (both come from the image processor).
    if architecture in {"VisionEncoderDecoderModel", "Pix2StructForConditionalGeneration"}:
        return "image-to-text"
    if architecture.endswith("ForConditionalGeneration"):
        return "image-text-to-text" if has_vision_config else "text2text-generation"
    for suffix, task in _ARCH_SUFFIX_TO_TASK.items():
        if architecture.endswith(suffix):
            return task
    return "feature-extraction"


def _infer_task(model_id: str, trust_remote_code: bool, subfolder: str = "") -> str:
    """Infer the export task from the model's ``config.architectures`` (see ``_task_from_architecture``).

    Requires ``config.architectures`` to be set (true for real checkpoints; some randomly-initialised
    test fixtures omit it — pass an explicit ``task`` for those).
    """
    config = AutoConfig.from_pretrained(model_id, subfolder=subfolder, trust_remote_code=trust_remote_code)
    if not config.architectures:
        raise ValueError(
            f"Could not infer a task for {model_id!r}: its config has no `architectures`. "
            f"Pass an explicit `task` (one of {sorted(_TASK_TO_AUTO_MODEL)})."
        )
    return _task_from_architecture(config.architectures[0], getattr(config, "vision_config", None) is not None)


def _load_processor(
    model_id: str, modality: str, trust_remote_code: bool, subfolder: str = "", prefer_composite: bool = True
):
    """Load a processor for ``model_id``.

    Prefer the composite ``AutoProcessor``: it bundles every sub-processor a model needs (e.g. a
    VLM-based image classifier like ShieldGemma2 needs the text side — a built-in prompt → ``input_ids``
    — alongside ``pixel_values``, which a bare image processor can't supply). Fall back to the
    modality-specific auto-class for models that register no composite processor — or, with
    ``prefer_composite=False``, when the composite loads but can't drive this modality's sample (a
    VisionEncoderDecoder processor, say, rejects an image-only call because it demands text).
    """
    if prefer_composite:
        with contextlib.suppress(Exception):
            return transformers.AutoProcessor.from_pretrained(
                model_id, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
    processor_class = getattr(transformers, _MODALITY_TO_PROCESSOR[modality])
    return processor_class.from_pretrained(model_id, subfolder=subfolder, trust_remote_code=trust_remote_code)


def _load_processor_and_build_sample(model_id, modality, model, trust_remote_code, subfolder=""):
    """Load a processor and build the tiny example batch for ``modality``, returning both.

    Retries once with the modality-specific processor when the composite one loads but can't drive the
    sample (a VisionEncoderDecoder demands text for an image-only call; a VLM on a text export has no
    tokenizer ``pad_token``).
    """
    for prefer_composite in (True, False):
        processor = _load_processor(model_id, modality, trust_remote_code, subfolder, prefer_composite)
        try:
            return processor, _build_sample_inputs(processor, modality, model)
        except (ValueError, TypeError, AttributeError):
            if not prefer_composite:
                raise


def _resolve_device(device):
    """Pick the export device: the requested one, else the first available accelerator, else CPU."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _to_device(inputs, device):
    """Move every tensor in an inputs mapping to ``device`` (non-tensors pass through)."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def _save_openvino_tokenizer(
    model_id: str, processor, output: Path, trust_remote_code: bool = False, subfolder: str = ""
) -> None:
    """Convert the tokenizer to OpenVINO IR (`openvino_tokenizer.xml` / `openvino_detokenizer.xml`) so
    OpenVINO GenAI (`LLMPipeline`, `VLMPipeline`, `WhisperPipeline`) can decode text from the export.

    The tokenizer is resolved from the processor when possible — the processor *is* the tokenizer for
    text models, or exposes it as `.tokenizer` (VLM/ASR processors) — and loaded separately otherwise
    (audio feature extractors carry no tokenizer). Best-effort: a failure only skips GenAI text decode.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = (
            processor
            if hasattr(processor, "convert_tokens_to_ids")
            else AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    try:
        export_tokenizer(tokenizer, output)
    except Exception as exc:  # noqa: BLE001 — tokenizer conversion is best-effort for GenAI
        logger.warning("Could not convert the tokenizer to OpenVINO (GenAI needs it): %s", exc)


def _try_save_generation_config(model, output: Path) -> None:
    """Best-effort save of the model's generation config next to the IR (needed by the OV runtimes)."""
    try:
        model.generation_config.save_pretrained(output)
    except Exception as exception:  # noqa: BLE001 — best-effort; a missing/invalid config only skips defaults
        logger.warning("Generation config not saved, saving failed with: %s", exception)


def export_openvino_hf(
    model_id: str,
    output: Union[str, Path],
    task: str = "auto",
    stateful: bool = True,
    fp16: bool = True,
    dynamic: bool = True,
    trust_remote_code: bool = False,
    device: Union[str, None] = None,
    subfolder: str = "",
) -> Path:
    """Export a Transformers model to OpenVINO IR through the Transformers (HF) exporter.

    Args:
        model_id: Hub id or local path of the model to export.
        subfolder: Subfolder within ``model_id`` holding the model + processor files (forwarded to every
            ``from_pretrained``), for repos that nest models under a path.
        output: Directory to write the OpenVINO IR, processor and config into.
        task: One of ``_TASK_TO_AUTO_MODEL`` or ``"auto"`` to infer it from the model config.
        stateful: Fuse the KV-cache into OpenVINO state (generative models only).
        fp16: Compress weights to fp16 when saving.
        dynamic: Mark input axes dynamic so the IR accepts variable batch/sequence lengths. Also drives
            a multi-token ``decode`` capture so the decode sequence axis stays dynamic.
        trust_remote_code: Forwarded to the model/processor loaders.
        device: Torch device the model is loaded and traced on (e.g. ``"cuda"``, ``"xpu"``, ``"cpu"``).
            ``None`` auto-selects the first available accelerator. Speeds up the real forward passes
            some decompositions run (VLM/seq2seq ``generate`` captures); OpenVINO conversion pulls the
            weights back to CPU for the IR.

    Returns:
        The output directory containing ``openvino_*.xml`` (one per exported component), the processor
        and the model config.
    """
    if task == "auto":
        task = _infer_task(model_id, trust_remote_code, subfolder=subfolder)
        logger.info("Inferred task %r", task)
    if task not in _TASK_TO_AUTO_MODEL:
        raise ValueError(f"Unsupported task {task!r}. Choose one of: {sorted(_TASK_TO_AUTO_MODEL)} (or 'auto').")
    auto_model_class_name, generative, modality = _TASK_TO_AUTO_MODEL[task]
    auto_model_class = getattr(transformers, auto_model_class_name)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device)
    logger.info("Loading %s (%s) for task %r [%s] on %s", model_id, auto_model_class_name, task, modality, device)
    model = auto_model_class.from_pretrained(
        model_id, subfolder=subfolder, dtype=torch.float32, trust_remote_code=trust_remote_code
    )
    model = model.eval().to(device)
    # The per-architecture example batch, moved to the export device for the real forward passes the
    # decompositions run (VLM/seq2seq `generate` captures).
    processor, sample_inputs = _load_processor_and_build_sample(
        model_id, modality, model, trust_remote_code, subfolder
    )
    sample_inputs = _to_device(sample_inputs, device)

    exporter = OpenVINOExporter()
    config = OpenVINOConfig(dynamic=dynamic, stateful=stateful and generative)

    # Each branch produces just the `components` mapping (name -> ov.Model); the shared tail below saves
    # them and, for generative models, the OpenVINO tokenizer + generation config.
    if generative and getattr(model.config, "is_encoder_decoder", False):
        # OpenVINO seq2seq layout: a stateless encoder + a stateful decoder, the pair
        # `OVModelForSeq2SeqLM` / `OVModelForSpeechSeq2Seq` load. Decomposed here (not via the
        # Transformers exporter) because the single-stateful-decoder shape is OpenVINO-specific.
        parts = decompose_encoder_decoder(model, sample_inputs)
        components = {
            "encoder_model": exporter.export(
                *parts["encoder"], config=OpenVINOConfig(dynamic=dynamic, stateful=False)
            ),
            "decoder_model": exporter.export(
                *parts["decoder"], config=OpenVINOConfig(dynamic=dynamic, stateful=stateful)
            ),
        }
    elif generative and modality == "multimodal":
        # VLM (llava-family): the text-embeddings + vision-embeddings + stateful language-model layout
        # `OVModelForVisualCausalLM` loads. Decomposed here (not via the Transformers exporter) because
        # the component split targets that OpenVINO runtime specifically.
        parts = decompose_vlm(model, sample_inputs)
        components = {
            name: exporter.export(
                module, inputs, config=OpenVINOConfig(dynamic=dynamic, stateful=stateful and part_stateful)
            )
            for name, (module, inputs, part_stateful) in parts.items()
        }
    elif generative and modality == "text":
        # OpenVINO text generation needs exactly ONE graph: the unified stateful multi-token `decode`
        # (dynamic sequence axis). Both OVModelForCausalLM and OpenVINO GenAI's LLMPipeline drive it for
        # the whole loop — empty state == prefill, then step-by-step decode, and its dynamic query axis
        # also covers chunked prefill / continuation. The separate `prefill` graph the decomposition
        # produces is never used by any OV runtime, so decompose (cheap) and export ONLY the `decode`
        # graph — tracing + converting `prefill` is the dominant, wasted export cost. (A static export,
        # dynamic=False, captures a single-token `decode` that can't prefill.)
        if not dynamic:
            logger.warning(
                "A static text-generation export (dynamic=False) produces a single-token decoder that "
                "OpenVINO GenAI can't drive (it can't prefill). Export with dynamic=True for GenAI."
            )
        submodel, subinputs = decompose_for_generation(model, sample_inputs, multi_token_decode=dynamic)["decode"]
        components = {"model": exporter.export(submodel, subinputs, config=config)}
    elif generative:
        # Other generative modalities: decompose and export every component.
        components = exporter.export_for_generation(model, sample_inputs, config=config, multi_token_decode=dynamic)
    else:
        components = {"model": exporter.export(model, sample_inputs, config=config)}

    for name, ov_model in components.items():
        path = output / f"openvino_{name}.xml"
        save_model(ov_model, str(path), compress_to_fp16=fp16)
        logger.info("Saved %s", path)

    if generative:
        # Generative models decode to text: save the OpenVINO tokenizer + generation config alongside the IR.
        _save_openvino_tokenizer(model_id, processor, output, trust_remote_code, subfolder=subfolder)
        _try_save_generation_config(model, output)

    processor.save_pretrained(output)
    # OpenVINO GenAI's image preprocessing reads the legacy `preprocessor_config.json`, which the
    # combined `processor.save_pretrained` no longer writes — save the image processor explicitly so
    # GenAI resizes to the model's own image size instead of a default.
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.save_pretrained(output)
    # anyres VLMs (llava_next) keep the row-separator embedding as a weight; the OV runtime has no
    # weights, so persist it into the config for `pack_image_features` to read back.
    image_newline = getattr(model, "image_newline", getattr(getattr(model, "model", None), "image_newline", None))
    if image_newline is not None:
        model.config.image_newline = image_newline.tolist()
    model.config.save_pretrained(output)
    logger.info("Export complete: %s", output)
    return output
