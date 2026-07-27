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
"""Model-specific decomposers/wrappers for the experimental ``openvino-hf`` export path.

The default ("natural") decomposition lives in the Transformers exporter (single component for
plain models, prefill/decode for generative ones) and stays runtime-agnostic. This module holds the
OpenVINO-specific specializations that target a particular ``OVModel*`` runtime layout:

- ``decompose_encoder_decoder`` — a seq2seq model into a stateless encoder + a stateful decoder
  (``OVModelForSeq2SeqLM`` / ``OVModelForSpeechSeq2Seq``). Generic across encoder-decoder families.
- ``decompose_vlm`` — a VLM into the components ``OVModelForVisualCausalLM`` loads. Most families use
  the generic text-embeddings + single vision-embeddings + stateful language-model layout (the vision
  graph is per-architecture, looked up from ``_VLM_VISION_FUSERS``); families whose runtime expects a
  different split (qwen2_vl's two-stage vision + 3D M-RoPE) register a bespoke decomposer in
  ``_VLM_DECOMPOSERS``.

``decompose_encoder_decoder`` returns ``{component: (nn.Module, forward_inputs)}``; ``decompose_vlm``
returns ``{component: (nn.Module, forward_inputs, stateful)}`` keyed by ``openvino_<component>.xml``.
"""

import copy
import inspect
from typing import Any, Callable

import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput


def _capture_forward_pre_hook(store: list, module: torch.nn.Module):
    """Record each forward call's kwargs (positional args normalised, ``**kwargs`` flattened) so a
    captured dict can be passed straight to the exporter."""
    signature = inspect.signature(module.forward)

    def _hook(_module, args, kwargs):
        captured = {}
        for name, value in signature.bind(*args, **kwargs).arguments.items():
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                captured.update(copy.deepcopy(value))
            elif kind is not inspect.Parameter.VAR_POSITIONAL:
                captured[name] = copy.deepcopy(value)
        store.append(captured)

    return module.register_forward_pre_hook(_hook, with_kwargs=True)


# ── seq2seq (encoder-decoder) ────────────────────────────────────────────────


class _EncoderDecoderDecodeStep(torch.nn.Module):
    """Decode step (decoder stack + ``lm_head``) with canonical port names for encoder-decoder export.

    Bridges ``get_decoder()`` (canonical inputs but no ``logits``) and the full model (``logits`` but
    under ``decoder_input_ids``/``encoder_outputs``): canonical inputs in, ``logits`` + cache out.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, encoder_hidden_states, cache_position, encoder_attention_mask=None, past_key_values=None
    ):
        outputs = self.model(
            decoder_input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        return {"logits": outputs.logits, "past_key_values": outputs.past_key_values}


def decompose_encoder_decoder(model, inputs: dict[str, Any]) -> dict[str, tuple]:
    """Split an encoder-decoder model (T5, BART, Whisper, …) into ``encoder`` and ``decoder``.

    Runs a real ``model.generate``, captures the encoder forward and a decode-step forward, and wraps
    the decoder with [`_EncoderDecoderDecodeStep`]. `torch.export`'s Dim.AUTO specializes size-1 axes,
    so the capture is taken once the self-attention past length is >= 2 and the query is widened to 2
    (values don't affect the traced graph). Cross-attention is dropped from the captured cache so the
    trace recomputes it from ``encoder_hidden_states`` (a plain output, not growing state).
    """
    encoder, decoder = model.get_encoder(), model.get_decoder()
    encoder_calls, decoder_calls = [], []
    handles = [_capture_forward_pre_hook(encoder_calls, encoder), _capture_forward_pre_hook(decoder_calls, decoder)]
    try:
        model.generate(**copy.deepcopy(inputs), max_new_tokens=3, min_new_tokens=3, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    if not encoder_calls or len(decoder_calls) < 3:
        raise RuntimeError(
            f"decompose_encoder_decoder expected 1 encoder forward and >=3 decoder forwards during "
            f"generate() on {type(model).__name__}, but captured {len(encoder_calls)} and {len(decoder_calls)}."
        )

    decode = decoder_calls[-1]  # self-attention past length is >= 2 by this step
    past_length = decode["past_key_values"].self_attention_cache.get_seq_length()
    decoder_inputs = {
        "input_ids": decode["input_ids"].repeat(1, 2),
        "encoder_hidden_states": decode["encoder_hidden_states"],
        "past_key_values": EncoderDecoderCache(decode["past_key_values"].self_attention_cache, DynamicCache()),
        "cache_position": torch.arange(past_length, past_length + 2),
    }
    if decode.get("encoder_attention_mask") is not None:  # speech encoders (Whisper) have no padding mask
        decoder_inputs["encoder_attention_mask"] = decode["encoder_attention_mask"]

    return {
        "encoder": (encoder, dict(encoder_calls[0])),
        "decoder": (_EncoderDecoderDecodeStep(model), decoder_inputs),
    }


# ── VLM (image-text-to-text) ─────────────────────────────────────────────────


class _TextEmbeddings(torch.nn.Module):
    """``input_ids`` -> token embeddings (the ``text_embeddings_model`` OVModelForVisualCausalLM loads)."""

    def __init__(self, model):
        super().__init__()
        self.embeddings = model.get_input_embeddings()

    def forward(self, input_ids):
        return self.embeddings(input_ids)


class _VisionEmbeddings(torch.nn.Module):
    """``pixel_values`` (+ optional ``patch_attention_mask``) -> projected image features, via a
    per-architecture fuser. The optional mask covers split-image families (idefics3, smolvlm) whose
    vision tower needs a patch attention mask; single-image families ignore it."""

    def __init__(self, model, fuser: Callable):
        super().__init__()
        self.model = model
        self.fuser = fuser

    def forward(self, pixel_values, patch_attention_mask=None):
        if patch_attention_mask is None:
            return self.fuser(self.model, pixel_values)
        return self.fuser(self.model, pixel_values, patch_attention_mask)


class _LanguageModel(torch.nn.Module):
    """``inputs_embeds`` -> ``logits`` (the stateful ``language_model`` OVModelForVisualCausalLM loads)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return {"logits": outputs.logits, "past_key_values": outputs.past_key_values}


def _llava_vision_features(model, pixel_values):
    """llava vision tower + feature-layer select + projector -> ``[num_images, patches, hidden]``.

    Reimplemented directly (rather than calling ``get_image_features``) to return a single projected
    tensor and keep the ``num_images`` axis dynamic — ``get_image_features`` returns a per-image list
    that bakes the image count into the graph.
    """
    config = model.config
    base = getattr(model, "model", model)
    vision_outputs = base.vision_tower(pixel_values, output_hidden_states=True)
    selected = vision_outputs.hidden_states[config.vision_feature_layer]
    if config.vision_feature_select_strategy == "default":
        selected = selected[:, 1:]
    return base.multi_modal_projector(selected)


def _tower_projector_vision_features(model, pixel_values):
    """Vision tower -> projector on the tower's ``last_hidden_state`` (no feature-layer select). Covers
    families whose ``get_image_features`` is exactly ``projector(vision_tower(px).last_hidden_state)``
    (GOT-OCR2, gemma3)."""
    base = getattr(model, "model", model)
    last_hidden_state = base.vision_tower(pixel_values).last_hidden_state
    return base.multi_modal_projector(last_hidden_state)


def _idefics3_vision_features(model, pixel_values, patch_attention_mask):
    """idefics3 / smolvlm split-image vision model (needs a patch attention mask) -> connector."""
    base = getattr(model, "model", model)
    last_hidden_state = base.vision_model(
        pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
    ).last_hidden_state
    return base.connector(last_hidden_state)


# model_type -> fuser that turns pixel_values into projected image features. Extend per architecture.
# gemma3's vision is `_tower_projector_vision_features`, but its hybrid sliding-window language model
# doesn't yet export cleanly to a stateful decoder (same gap as mistral/mixtral/glm4), so it's not
# registered here.
_VLM_VISION_FUSERS: dict[str, Callable] = {
    "llava": _llava_vision_features,
    "llava_next": _llava_vision_features,
    "got_ocr2": _tower_projector_vision_features,
    "idefics3": _idefics3_vision_features,
    "smolvlm": _idefics3_vision_features,
}

# model_type -> attribute (on the model / its base) of the vision submodule whose forward inputs are
# captured for the vision component. Used by split-image families whose vision graph is fed the
# post-preprocessing tensors (reshaped pixel_values + patch attention mask), not the raw pixel_values.
_VLM_VISION_CAPTURE: dict[str, str] = {
    "idefics3": "vision_model",
    "smolvlm": "vision_model",
}


def decompose_vlm(model, inputs: dict[str, Any]) -> dict[str, tuple]:
    """Split a VLM into the components ``OVModelForVisualCausalLM`` loads, each as
    ``(nn.Module, forward_inputs, stateful)`` keyed by its ``openvino_<name>.xml`` file stem.

    Most families use the generic three-component layout (`_decompose_vlm_generic`): ``text_embeddings``
    + a single ``vision_embeddings`` + a stateful ``language_model``. Families whose runtime expects a
    different split (e.g. qwen2_vl's two-stage vision + 3D M-RoPE language model) register a bespoke
    decomposer in ``_VLM_DECOMPOSERS``. ``inputs`` is the per-architecture sample the caller built
    (mirroring ``decompose_encoder_decoder``, which also receives its inputs).
    """
    decomposer = _VLM_DECOMPOSERS.get(model.config.model_type, _decompose_vlm_generic)
    return decomposer(model, inputs)


def _decompose_vlm_generic(model, inputs: dict[str, Any]) -> dict[str, tuple]:
    """The llava-style three-component layout: text-embeddings + a single vision-embeddings graph +
    a stateful language model. The vision piece and its captured inputs are per-architecture
    (``_VLM_VISION_FUSERS`` / ``_VLM_VISION_CAPTURE``); the runtime merges vision into text embeddings
    itself, so that step is left outside the graphs."""
    model_type = model.config.model_type
    fuser = _VLM_VISION_FUSERS.get(model_type)
    if fuser is None:
        raise ValueError(
            f"The `openvino-hf` VLM path has no vision fuser for model_type {model_type!r}. "
            f"Registered: {sorted(_VLM_VISION_FUSERS)}. Use the `openvino` exporter for this model, "
            f"or add a fuser to `_VLM_VISION_FUSERS`."
        )

    # For split-image families the vision graph is fed the post-preprocessing tensors (reshaped
    # pixel_values + patch attention mask), so capture the vision submodule's forward during generate.
    decoder = model.get_decoder()
    decoder_calls, vision_calls = [], []
    handles = [_capture_forward_pre_hook(decoder_calls, decoder)]
    vision_attr = _VLM_VISION_CAPTURE.get(model_type)
    if vision_attr is not None:
        handles.append(_capture_forward_pre_hook(vision_calls, getattr(getattr(model, "model", model), vision_attr)))
    try:
        model.generate(**copy.deepcopy(inputs), max_new_tokens=3, min_new_tokens=3, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    if vision_attr is not None:
        vision_inputs = {k: vision_calls[0][k] for k in ("pixel_values", "patch_attention_mask")}
    else:
        pixel_values = inputs["pixel_values"]
        # anyres families (llava_next) stack tiles as [num_images, num_patches, C, H, W]; the vision
        # tower — and the runtime — see the tiles flattened to a single [total_patches, C, H, W] batch.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.flatten(0, 1)
        vision_inputs = {"pixel_values": pixel_values}

    decode = decoder_calls[-1]  # self-attention past length is >= 2 by this step
    past_length = decode["past_key_values"].get_seq_length()
    batch = decode["inputs_embeds"].shape[0]
    language_inputs = {
        "inputs_embeds": decode["inputs_embeds"].repeat(1, 2, 1),  # widen query to 2 (keeps the axis dynamic)
        "attention_mask": torch.ones(batch, past_length + 2, dtype=torch.long),
        "position_ids": torch.arange(past_length, past_length + 2).unsqueeze(0).expand(batch, -1),
        "past_key_values": decode["past_key_values"],
    }

    return {
        "text_embeddings_model": (_TextEmbeddings(model), {"input_ids": inputs["input_ids"]}, False),
        "vision_embeddings_model": (_VisionEmbeddings(model, fuser), vision_inputs, False),
        "language_model": (_LanguageModel(model), language_inputs, True),
    }


# ── qwen2_vl (grid_thw + 3D M-RoPE) ──────────────────────────────────────────


class _Qwen2VLPatchEmbed(torch.nn.Module):
    """``pixel_values`` -> patch-embedded hidden states (the ``vision_embeddings`` graph). The heavy
    transformer blocks live in the separate merger graph, matching the qwen2_vl runtime split."""

    def __init__(self, visual):
        super().__init__()
        self.patch_embed = visual.patch_embed

    def forward(self, pixel_values):
        return self.patch_embed(pixel_values)


class _Qwen2VLVisionMerger(torch.nn.Module):
    """Patch-embedded hidden states -> merged image features (the ``vision_embeddings_merger`` graph).

    Runs the vision transformer blocks + patch merger. The runtime computes ``rotary_pos_emb`` and a
    block-diagonal ``attention_mask`` from ``image_grid_thw`` in Python and feeds them in, so the block
    attention here takes a dense additive mask via SDPA instead of the model's ``cu_seqlens``
    variable-length path (which can't export: it does a data-dependent ``torch.split``)."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

        self._apply_rope = apply_rotary_pos_emb_vision

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        # Input is named ``hidden_states`` (patch-embedded tokens): OpenVINO GenAI's VLMPipeline feeds
        # the merger by that exact tensor name, and OVModelForVisualCausalLM maps to it too.
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for block in self.visual.blocks:
            hidden_states = self._run_block(block, hidden_states, attention_mask, position_embeddings)
        return self.visual.merger(hidden_states)

    def _run_block(self, block, hidden_states, attention_mask, position_embeddings):
        attn = block.attn
        seq_length = hidden_states.shape[0]
        query, key, value = (
            attn.qkv(block.norm1(hidden_states))
            .reshape(seq_length, 3, attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query, key = self._apply_rope(query, key, cos, sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask.unsqueeze(1), scale=attn.scaling
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        hidden_states = hidden_states + attn.proj(attn_output)
        hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
        return hidden_states


class _Qwen2VLLanguageModel(torch.nn.Module):
    """``inputs_embeds`` (+ 3D M-RoPE ``position_ids``) -> ``logits`` (the stateful ``language_model``).
    Wraps the text decoder + ``lm_head`` directly, not the full VLM forward (which would recompute the
    multimodal RoPE index and re-enter the vision path)."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.get_decoder()
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values=None):
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return {"logits": self.lm_head(outputs.last_hidden_state), "past_key_values": outputs.past_key_values}


def _decompose_qwen2_vl(model, inputs: dict[str, Any]) -> dict[str, tuple]:
    """qwen2_vl / qwen2_5_vl: two-stage vision (patch-embed + block/merger) and a 3D-M-RoPE language
    model, matching ``_OVQwen2VLForCausalLM``'s four-component layout."""
    visual = getattr(model, "model", model).visual

    decoder = model.get_decoder()
    decoder_calls = []
    handle = _capture_forward_pre_hook(decoder_calls, decoder)
    try:
        model.generate(**copy.deepcopy(inputs), max_new_tokens=3, min_new_tokens=3, use_cache=True)
    finally:
        handle.remove()

    decode = decoder_calls[-1]  # self-attention past length is >= 2 by this step
    past_length = decode["past_key_values"].get_seq_length()
    batch = decode["inputs_embeds"].shape[0]
    # 3-section M-RoPE ``position_ids`` [sections, batch, query], matching what the runtime feeds (the
    # rotary embedding uses 3 sections; a 4-row position_ids is reduced to 3 inside the decoder, so we
    # trace the 3-row form the runtime supplies directly). Query widened to 2 to keep the axis dynamic.
    language_inputs = {
        "inputs_embeds": decode["inputs_embeds"].repeat(1, 2, 1),
        "attention_mask": torch.ones(batch, past_length + 2, dtype=torch.long),
        "position_ids": torch.arange(past_length, past_length + 2).view(1, 1, 2).expand(3, batch, 2).contiguous(),
        "past_key_values": decode["past_key_values"],
    }

    # Vision: patch-embed produces the merger's hidden-states input; rotary_pos_emb + a full (single
    # image) attention mask are computed here for the trace and supplied by the runtime at inference.
    # Detach so the computed sample tensors are graph leaves the exporter can deepcopy.
    from transformers.models.qwen2_vl.modeling_qwen2_vl import get_vision_position_ids

    pixel_values = inputs["pixel_values"]
    with torch.no_grad():
        hidden_states = visual.patch_embed(pixel_values)
        vision_position_ids = get_vision_position_ids(inputs["image_grid_thw"], visual.spatial_merge_size)
        rotary_pos_emb = visual.rotary_pos_emb(vision_position_ids)
    seq_length = hidden_states.shape[0]
    merger_inputs = {
        "hidden_states": hidden_states,
        "attention_mask": torch.zeros(1, seq_length, seq_length),
        "rotary_pos_emb": rotary_pos_emb,
    }

    return {
        "text_embeddings_model": (_TextEmbeddings(model), {"input_ids": inputs["input_ids"]}, False),
        "vision_embeddings_model": (_Qwen2VLPatchEmbed(visual), {"pixel_values": pixel_values}, False),
        "vision_embeddings_merger_model": (_Qwen2VLVisionMerger(visual), merger_inputs, False),
        "language_model": (_Qwen2VLLanguageModel(model), language_inputs, True),
    }


# ── qwen2_5_vl (grid_thw + 3D M-RoPE + window attention) ─────────────────────


class _Qwen2_5_VLVisionMerger(_Qwen2VLVisionMerger):
    """qwen2_5_vl vision blocks + merger. Extends the qwen2_vl merger with window attention: patches are
    reordered into windows by ``window_index``, full-attention blocks (``fullatt_block_indexes``) use
    ``attention_mask`` while the rest use ``window_attention_mask``, and the merged output is reordered
    back. The runtime computes both masks + ``window_index`` from ``image_grid_thw`` and feeds them in."""

    def __init__(self, visual):
        super().__init__(visual)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

        self._apply_rope = apply_rotary_pos_emb_vision
        self._fullatt_indexes = list(visual.fullatt_block_indexes)
        self._merge_unit = visual.spatial_merge_unit

    def forward(self, hidden_states, attention_mask, window_attention_mask, window_index, rotary_pos_emb):
        # Reorder patches into windows (explicit static dims + a single -1 keep the seq axis dynamic).
        dim, rope_dim = hidden_states.shape[-1], rotary_pos_emb.shape[-1]
        hidden_states = hidden_states.reshape(-1, self._merge_unit, dim)[window_index].reshape(-1, dim)
        rotary_pos_emb = rotary_pos_emb.reshape(-1, self._merge_unit, rope_dim)[window_index].reshape(-1, rope_dim)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for index, block in enumerate(self.visual.blocks):
            mask = attention_mask if index in self._fullatt_indexes else window_attention_mask
            hidden_states = self._run_block(block, hidden_states, mask, position_embeddings)
        return self.visual.merger(hidden_states)[torch.argsort(window_index)]


def _decompose_qwen2_5_vl(model, inputs: dict[str, Any]) -> dict[str, tuple]:
    """qwen2_5_vl: like qwen2_vl but its vision merger uses window attention (see
    [`_Qwen2_5_VLVisionMerger`]); the language model and other components are identical."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import get_vision_position_ids, get_vision_window_index

    visual = getattr(model, "model", model).visual

    decoder = model.get_decoder()
    decoder_calls = []
    handle = _capture_forward_pre_hook(decoder_calls, decoder)
    try:
        model.generate(**copy.deepcopy(inputs), max_new_tokens=3, min_new_tokens=3, use_cache=True)
    finally:
        handle.remove()

    decode = decoder_calls[-1]  # self-attention past length is >= 2 by this step
    past_length = decode["past_key_values"].get_seq_length()
    batch = decode["inputs_embeds"].shape[0]
    language_inputs = {
        "inputs_embeds": decode["inputs_embeds"].repeat(1, 2, 1),
        "attention_mask": torch.ones(batch, past_length + 2, dtype=torch.long),
        "position_ids": torch.arange(past_length, past_length + 2).view(1, 1, 2).expand(3, batch, 2).contiguous(),
        "past_key_values": decode["past_key_values"],
    }

    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    with torch.no_grad():
        hidden_states = visual.patch_embed(pixel_values)
        rotary_pos_emb = visual.rotary_pos_emb(get_vision_position_ids(grid_thw, visual.spatial_merge_size))
        window_index, _ = get_vision_window_index(
            grid_thw,
            spatial_merge_size=visual.spatial_merge_size,
            window_size=visual.window_size,
            patch_size=visual.patch_size,
        )
    seq_length = hidden_states.shape[0]
    merger_inputs = {
        "hidden_states": hidden_states,
        "attention_mask": torch.zeros(1, seq_length, seq_length),
        "window_attention_mask": torch.zeros(1, seq_length, seq_length),
        "window_index": window_index,
        "rotary_pos_emb": rotary_pos_emb,
    }

    return {
        "text_embeddings_model": (_TextEmbeddings(model), {"input_ids": inputs["input_ids"]}, False),
        "vision_embeddings_model": (_Qwen2VLPatchEmbed(visual), {"pixel_values": pixel_values}, False),
        "vision_embeddings_merger_model": (_Qwen2_5_VLVisionMerger(visual), merger_inputs, False),
        "language_model": (_Qwen2VLLanguageModel(model), language_inputs, True),
    }


# ── qwen3_vl (grid_thw + 3D M-RoPE + deepstack) ──────────────────────────────


class _Qwen3VLVisionPos(torch.nn.Module):
    """Interpolated position-embedding lookup (the ``vision_embeddings_pos`` graph). The runtime builds
    the bilinear ``idx`` from ``image_grid_thw`` and does the weighted sum; this IR is just the lookup."""

    def __init__(self, visual):
        super().__init__()
        self.pos_embed = visual.pos_embed

    def forward(self, input):
        # Input named ``input`` to match the tensor name OpenVINO GenAI feeds this embedding-lookup IR.
        return self.pos_embed(input)


class _Qwen3VLVisionMerger(_Qwen2VLVisionMerger):
    """qwen3_vl vision blocks + merger, returning ``(image_embeds, deepstack_features)``. Same dense-mask
    block reimplementation as qwen2_vl (identical vision attention), plus the deepstack features collected
    at ``deepstack_visual_indexes`` and stacked — the runtime injects them into the language model."""

    def __init__(self, visual):
        super().__init__(visual)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

        self._apply_rope = apply_rotary_pos_emb_vision
        self._deepstack_indexes = list(visual.deepstack_visual_indexes)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        deepstack_features = []
        for index, block in enumerate(self.visual.blocks):
            hidden_states = self._run_block(block, hidden_states, attention_mask, position_embeddings)
            if index in self._deepstack_indexes:
                merger = self.visual.deepstack_merger_list[self._deepstack_indexes.index(index)]
                deepstack_features.append(merger(hidden_states))
        # Dict return names the two outputs (`last_hidden_state`, `deepstack_feature_lists`) so GenAI's
        # VLMPipeline finds them by name; OVModelForVisualCausalLM reads them positionally, order-preserved.
        return {
            "last_hidden_state": self.visual.merger(hidden_states),
            "deepstack_feature_lists": torch.stack(deepstack_features),
        }


class _Qwen3VLLanguageModel(torch.nn.Module):
    """qwen3_vl language model: like ``_Qwen2VLLanguageModel`` but also takes ``visual_pos_masks`` and
    stacked ``deepstack_visual_embeds`` the runtime injects at the deepstack layers (see the exporter's
    ``_patch_qwen3vl_deepstack``, which makes that injection export-safe)."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.get_decoder()
        self.lm_head = model.lm_head

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    ):
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return {"logits": self.lm_head(outputs.last_hidden_state), "past_key_values": outputs.past_key_values}


def _decompose_qwen3_vl(model, inputs: dict[str, Any]) -> dict[str, tuple]:
    """qwen3_vl: qwen2_vl's two-stage vision + 3D M-RoPE, plus a ``vision_embeddings_pos`` lookup and
    deepstack features (returned by the merger, injected into the language model)."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import get_vision_position_ids

    visual = getattr(model, "model", model).visual
    n_deepstack = len(visual.deepstack_visual_indexes)
    hidden_size = model.config.text_config.hidden_size

    decoder = model.get_decoder()
    decoder_calls, pos_calls = [], []
    handles = [
        _capture_forward_pre_hook(decoder_calls, decoder),
        _capture_forward_pre_hook(pos_calls, visual.pos_embed),
    ]
    try:
        model.generate(**copy.deepcopy(inputs), max_new_tokens=3, min_new_tokens=3, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    decode = decoder_calls[-1]  # self-attention past length is >= 2 by this step
    past_length = decode["past_key_values"].get_seq_length()
    batch = decode["inputs_embeds"].shape[0]
    # Deepstack features (all-True mask + full-width embeds keep the deepstack axes dynamic; the runtime
    # feeds real features at prefill and zero placeholders at decode). 3-section M-RoPE as in qwen2_vl.
    language_inputs = {
        "inputs_embeds": decode["inputs_embeds"].repeat(1, 2, 1),
        "attention_mask": torch.ones(batch, past_length + 2, dtype=torch.long),
        "position_ids": torch.arange(past_length, past_length + 2).view(1, 1, 2).expand(3, batch, 2).contiguous(),
        "past_key_values": decode["past_key_values"],
        "visual_pos_masks": torch.ones(batch, 2, dtype=torch.bool),
        "deepstack_visual_embeds": torch.zeros(n_deepstack, batch * 2, hidden_size),
    }

    pixel_values = inputs["pixel_values"]
    # int64 index (not the captured int32) so the pos-embed IR input matches the i64 tensor GenAI feeds.
    pos_input = next(iter(pos_calls[0].values())).long()
    with torch.no_grad():
        hidden_states = visual.patch_embed(pixel_values)
        rotary_pos_emb = visual.rotary_pos_emb(
            get_vision_position_ids(inputs["image_grid_thw"], visual.spatial_merge_size)
        )
    seq_length = hidden_states.shape[0]
    merger_inputs = {
        "hidden_states": hidden_states,
        "attention_mask": torch.zeros(1, seq_length, seq_length),
        "rotary_pos_emb": rotary_pos_emb,
    }

    return {
        "text_embeddings_model": (_TextEmbeddings(model), {"input_ids": inputs["input_ids"]}, False),
        "vision_embeddings_model": (_Qwen2VLPatchEmbed(visual), {"pixel_values": pixel_values}, False),
        "vision_embeddings_pos_model": (_Qwen3VLVisionPos(visual), {"input": pos_input}, False),
        "vision_embeddings_merger_model": (_Qwen3VLVisionMerger(visual), merger_inputs, False),
        "language_model": (_Qwen3VLLanguageModel(model), language_inputs, True),
    }


# ── gemma3 (bidirectional image-token attention via token_type_ids) ──────────


class _Gemma3LanguageModel(torch.nn.Module):
    """gemma3 language model that also takes ``token_type_ids``. Calling the full conditional-generation
    model (with ``inputs_embeds`` and no ``pixel_values``) lets it build gemma3's bidirectional
    image-token attention mask from ``token_type_ids`` natively — the runtime feeds ``token_type_ids``,
    all-text at decode and marking image spans at prefill."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids, token_type_ids, past_key_values=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return {"logits": outputs.logits, "past_key_values": outputs.past_key_values}


def _decompose_gemma3(model, inputs: dict[str, Any]) -> dict[str, tuple]:
    """gemma3: generic vision (tower + projector) + a language model that consumes ``token_type_ids``
    for bidirectional image-token attention (see [`_Gemma3LanguageModel`])."""
    decoder = model.get_decoder()
    decoder_calls = []
    handle = _capture_forward_pre_hook(decoder_calls, decoder)
    try:
        model.generate(**copy.deepcopy(inputs), max_new_tokens=3, min_new_tokens=3, use_cache=True)
    finally:
        handle.remove()

    decode = decoder_calls[-1]  # self-attention past length is >= 2 by this step
    past_length = decode["past_key_values"].get_seq_length()
    batch = decode["inputs_embeds"].shape[0]
    language_inputs = {
        "inputs_embeds": decode["inputs_embeds"].repeat(1, 2, 1),
        "attention_mask": torch.ones(batch, past_length + 2, dtype=torch.long),
        "position_ids": torch.arange(past_length, past_length + 2).unsqueeze(0).expand(batch, -1),
        "token_type_ids": torch.zeros(batch, 2, dtype=torch.long),
        "past_key_values": decode["past_key_values"],
    }

    return {
        "text_embeddings_model": (_TextEmbeddings(model), {"input_ids": inputs["input_ids"]}, False),
        "vision_embeddings_model": (
            _VisionEmbeddings(model, _tower_projector_vision_features),
            {"pixel_values": inputs["pixel_values"]},
            False,
        ),
        "language_model": (_Gemma3LanguageModel(model), language_inputs, True),
    }


_VLM_DECOMPOSERS: dict[str, Callable] = {
    "qwen2_vl": _decompose_qwen2_vl,
    "qwen2_5_vl": _decompose_qwen2_5_vl,
    "qwen3_vl": _decompose_qwen3_vl,
    "gemma3": _decompose_gemma3,
}
