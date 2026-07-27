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
"""Tiny example batches fed to the exporter (and, for generative models, to ``model.generate``).

Only shapes and dtypes reach the traced graph — the values are fixed placeholders. Text/image/audio
use one generic builder; VLMs need a per-architecture sample because the image-token count and prompt
format vary by family, so they dispatch through ``_VLM_INPUT_BUILDERS``.
"""

from typing import Any, Callable

import numpy as np
from PIL import Image


# Example-input placeholders. Only shapes/dtypes reach the traced graph — the values are irrelevant —
# so they're fixed here rather than exposed as knobs (a single prompt doesn't fit image/audio anyway).
# Batch 2 keeps the batch axis dynamic (torch.export specializes size-0/1 axes to constants).
_EXAMPLE_PROMPT = "Hello, world!"
_EXAMPLE_BATCH_SIZE = 2


def _default_vlm_inputs(processor, model) -> dict:
    """Default VLM sample: an ``<image>`` prompt whose token expands to match the pixel features
    (works for llava-style processors). Kept off the runtime `preprocess_inputs` so the traced vision
    size stays the model's own — what the runtime feeds at inference."""
    image_token = getattr(processor, "image_token", None) or "<image>"
    images = [Image.new("RGB", (224, 224))] * _EXAMPLE_BATCH_SIZE
    text = [f"{image_token}\n{_EXAMPLE_PROMPT}"] * _EXAMPLE_BATCH_SIZE
    return dict(processor(text=text, images=images, return_tensors="pt"))


def _llava_next_inputs(processor, model) -> dict:
    """llava-next counts image tokens from the processor's ``patch_size`` /
    ``vision_feature_select_strategy`` / ``num_additional_image_tokens``; align those to the model so
    the token count matches the features it produces (real checkpoints already agree — this only fills
    the gaps left by under-specified processor configs)."""
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = model.config.vision_config.patch_size
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    # The patch grid excludes the vision tower's non-patch tokens (a CLS embedding), which the "default"
    # strategy drops; the processor offsets by that count, so set it to the tower's actual extra tokens
    # (num_positions - num_patches on the vision embeddings, found regardless of tower nesting).
    embeddings = next((m for m in model.modules() if hasattr(m, "num_positions") and hasattr(m, "num_patches")), None)
    if embeddings is not None:
        processor.num_additional_image_tokens = embeddings.num_positions - embeddings.num_patches
    return _default_vlm_inputs(processor, model)


def _got_ocr2_inputs(processor, model) -> dict:
    """GOT-OCR2 builds its OCR prompt (and the matching image-token count) from the image alone."""
    return dict(processor(Image.new("RGB", (64, 64)), return_tensors="pt"))


def _chat_template_inputs(processor, model, size: int = 64) -> dict:
    """Build the prompt via the processor's chat template with one ``size``x``size`` image — the reliable
    way to get the right image-token expansion for families whose bare image token isn't recognised in raw
    text (idefics3, smolvlm)."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _EXAMPLE_PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return dict(processor(text=[text], images=[Image.new("RGB", (size, size))], return_tensors="pt"))


def _qwen_vl_inputs(processor, model) -> dict:
    """qwen VLMs need the image side to be a whole number of patch*merge blocks, or the image-token count
    won't match the pixel features. Derive that block unit from the vision config (patch_size *
    spatial_merge_size) rather than hardcoding a size — it differs per family (28 for qwen2/2.5-VL, 32 for
    qwen3-VL); two blocks per side is a safe small sample."""
    return _chat_template_inputs(
        processor,
        model,
        size=2 * model.config.vision_config.patch_size * model.config.vision_config.spatial_merge_size,
    )


# model_type -> builder for the sample generate-inputs. Per architecture because the image-token count
# and prompt format are model-specific; defaults to `_default_vlm_inputs`.
_VLM_INPUT_BUILDERS: dict[str, Callable] = {
    "got_ocr2": _got_ocr2_inputs,
    "idefics3": _chat_template_inputs,
    "smolvlm": _chat_template_inputs,
    "qwen2_vl": _qwen_vl_inputs,
    "qwen2_5_vl": _qwen_vl_inputs,
    "qwen3_vl": _qwen_vl_inputs,
    "gemma3": _chat_template_inputs,
    "llava_next": _llava_next_inputs,
}


def _build_sample_inputs(processor, modality: str, model=None) -> dict[str, Any]:
    """Build a tiny example batch for ``modality`` — the only per-model preparation this path does.

    Values are fixed placeholders (a prompt / a blank image / a second of silence); only shapes and
    dtypes matter to the exporter. For generative models these double as ``model.generate`` kwargs.
    ``model`` is only needed for ``multimodal``, whose sample is per-architecture (image-token count and
    prompt format vary by family).
    """
    if modality == "multimodal":
        return _VLM_INPUT_BUILDERS.get(model.config.model_type, _default_vlm_inputs)(processor, model)

    if modality == "text":
        # Need a pad token to build a batch > 1; fall back through the usual specials.
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token or processor.bos_token or processor.unk_token
        return dict(processor([_EXAMPLE_PROMPT] * _EXAMPLE_BATCH_SIZE, return_tensors="pt", padding=True))

    if modality == "image":
        images = [Image.new("RGB", (224, 224))] * _EXAMPLE_BATCH_SIZE
        return dict(processor(images=images, return_tensors="pt"))

    if modality == "image-text":
        # CLIP-style dual encoder: a batch of candidate captions scored against a batch of images.
        images = [Image.new("RGB", (224, 224))] * _EXAMPLE_BATCH_SIZE
        text = ["a photo of a cat", "a photo of a dog"]
        return dict(processor(text=text, images=images, return_tensors="pt", padding=True))

    if modality == "audio":
        audio = [np.zeros(16000, dtype=np.float32)] * _EXAMPLE_BATCH_SIZE  # 1s of silence at 16 kHz
        return dict(processor(audio, sampling_rate=16000, return_tensors="pt"))

    raise ValueError(f"Unsupported modality {modality!r}.")
