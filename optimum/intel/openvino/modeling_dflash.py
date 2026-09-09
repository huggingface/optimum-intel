#  Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""OpenVINO runtime for DFlash speculative decoding.

DFlash pairs a large target model with a small block-diffusion "drafter": the drafter
reads the target's hidden states at a handful of layers plus the last decoded token, and
denoises a whole block of `block_size` masked tokens in one forward pass. The target then
verifies the block in a single forward, so several tokens can be committed per target
call. Greedy DFlash is lossless - the accepted prefix plus the bonus token always
reproduces the target's own greedy continuation.

The drafter ships without a token embedding and without an lm_head: it borrows the
target's. With an OpenVINO export those two live inside other IR files - the embedding
inside `openvino_text_embeddings_model.xml` (behind the embedding norm) and the lm_head
fused into `openvino_language_model.xml`. This module extracts each as a small standalone
`ov.Model` rather than requiring a different export or duplicating weights in float.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import openvino as ov
import torch
from openvino import opset13
from transformers import AutoModel, GenerationConfig, PretrainedConfig
from transformers.generation.candidate_generator import CandidateGenerator
from transformers.modeling_outputs import BaseModelOutputWithPast

from .modeling_base import OVBaseModel


if TYPE_CHECKING:
    from transformers.generation.logits_process import LogitsProcessorList


HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers"
DFLASH_HIDDEN_STATE_PREFIX = "dflash_hidden_states"


def is_dflash_draft_model(ov_model: ov.Model) -> bool:
    """Whether an IR was exported as a DFlash drafter."""
    return ov_model.has_rt_info(["dflash_mode"]) and ov_model.get_rt_info(["dflash_mode"]).value == "True"


def read_dflash_rt_info(ov_model: ov.Model) -> Dict[str, Any]:
    """Read the `dflash` RT-info block stamped onto a drafter export."""
    info = {}
    if ov_model.has_rt_info(["dflash", "mask_token_id"]):
        info["mask_token_id"] = int(ov_model.get_rt_info(["dflash", "mask_token_id"]).value)
    if ov_model.has_rt_info(["dflash", "block_size"]):
        info["block_size"] = int(ov_model.get_rt_info(["dflash", "block_size"]).value)
    if ov_model.has_rt_info(["dflash", "target_layer_ids"]):
        raw = ov_model.get_rt_info(["dflash", "target_layer_ids"]).value
        info["target_layer_ids"] = [int(part) for part in raw.split(",") if part != ""]
    return info


# --------------------------------------------------------------------------------------
# Subgraph extraction
# --------------------------------------------------------------------------------------


def _hidden_state_locators(ov_model: ov.Model) -> Dict[int, ov.Output]:
    """Resolve the `hidden_states_decoder_layers` RT-info annotation to graph outputs.

    The annotation is written at export time by `utils_annotations.add_hidden_states_rt_info`
    and survives weight compression, because it records friendly names rather than
    positions.
    """
    import json

    if not ov_model.has_rt_info([HIDDEN_STATES_RT_INFO_KEY]):
        raise ValueError(
            "The language model was exported without the `hidden_states_decoder_layers` annotation, "
            "which DFlash needs to tap the target's intermediate hidden states. Re-export the target "
            "with a recent optimum-intel."
        )
    annotation = json.loads(ov_model.get_rt_info([HIDDEN_STATES_RT_INFO_KEY]).value)
    by_name = {}
    for op in ov_model.get_ordered_ops():
        by_name.setdefault(op.get_friendly_name(), op)

    locators = {}
    for layer_id, locator in annotation["layers"].items():
        producer = by_name.get(locator["producer"])
        if producer is None:
            raise ValueError(f"Hidden-state producer {locator['producer']!r} is missing from the graph.")
        locators[int(layer_id)] = producer.output(int(locator["output_index"]))
    return locators


def add_hidden_state_outputs(ov_model: ov.Model, layer_ids: List[int]) -> List[str]:
    """Expose the target's hidden states at `layer_ids` as extra model outputs.

    Returns the tensor names of the added outputs, ordered like `layer_ids`. Adding
    outputs does not change the graph's computation, only what it hands back.
    """
    locators = _hidden_state_locators(ov_model)
    missing = [layer_id for layer_id in layer_ids if layer_id not in locators]
    if missing:
        raise ValueError(
            f"The target model exposes hidden states for layers {sorted(locators)}, "
            f"but the drafter asks for {missing}."
        )

    names = []
    for layer_id in layer_ids:
        name = f"{DFLASH_HIDDEN_STATE_PREFIX}.{layer_id}"
        names.append(name)
        if name in {n for out in ov_model.outputs for n in out.get_names()}:
            continue
        added = ov_model.add_outputs([locators[layer_id]])[0]
        # `add_outputs` keeps the producer's existing tensor names; add a stable one so the
        # output can be looked up by name regardless of how the graph was serialized.
        added.get_tensor().set_names(added.get_names() | {name})
    return names


def extract_raw_embedding_model(text_embeddings_model: ov.Model) -> ov.Model:
    """Build a lookup-only embedding model from an exported text-embeddings IR.

    Some architectures (MuseGlimmer among them) normalize or scale the embedding right
    after the lookup, and the exported IR includes that. A DFlash drafter needs the *raw*
    table - transformers spells this `F.embedding(ids, embed_tokens.weight)` - so the
    graph is cut at the Gather and everything downstream is dropped.
    """
    gathers = [op for op in text_embeddings_model.get_ordered_ops() if op.get_type_name() == "Gather"]
    if len(gathers) != 1:
        raise ValueError(f"Expected exactly one Gather in the text-embeddings model, found {len(gathers)}.")
    output = gathers[0].output(0)
    output.get_tensor().set_names(output.get_names() | {"inputs_embeds"})
    return ov.Model([output], text_embeddings_model.get_parameters(), "raw_text_embeddings")


def extract_lm_head_model(language_model: ov.Model, hidden_size: int) -> ov.Model:
    """Build a standalone lm_head model from an exported language model IR.

    The final entry of the `hidden_states_decoder_layers` annotation is, by construction,
    the hidden-width value consumed by the lm_head. Re-rooting the graph there and keeping
    only the `logits` result prunes everything else away, leaving the projection (plus any
    logit softcapping the architecture applies, which is monotonic and so does not disturb
    greedy verification).

    The graph is re-rooted on a clone, so the caller's `language_model` is left intact.
    """
    language_model = language_model.clone()
    locators = _hidden_state_locators(language_model)
    source = locators[max(locators)]

    results = [result for result in language_model.get_results() if "logits" in result.get_output_tensor(0).names]
    if len(results) != 1:
        raise ValueError("Expected exactly one `logits` result in the language model.")

    parameter = opset13.parameter(
        ov.PartialShape([-1, -1, hidden_size]), source.get_element_type(), name="hidden_states"
    )
    parameter.get_output_tensor(0).set_names({"hidden_states"})
    for consumer in list(source.get_target_inputs()):
        consumer.replace_source_output(parameter.output(0))
    return ov.Model(results, [parameter], "lm_head")


# --------------------------------------------------------------------------------------
# Stateful KV-cache proxy
# --------------------------------------------------------------------------------------


class OVStatefulCacheProxy:
    """Gives a stateful OpenVINO model the `crop()` API assisted decoding relies on.

    Speculative decoding feeds `candidate_length + 1` tokens in one target forward and
    then keeps only the accepted prefix, so the rejected tail has to leave the KV cache.
    A stateful OpenVINO model holds that cache in `VariableState`s rather than in tensors
    the caller passes around, but those states can be read back and written, so the tail
    can be trimmed in place.
    """

    def __init__(self, model_part):
        self._model_part = model_part

    def crop(self, max_length: int) -> None:
        """Trim the cache. Negative `max_length` drops that many trailing positions."""
        request = self._model_part.request
        if request is None:
            return
        past_length = self._model_part._past_length
        keep = past_length + max_length if max_length < 0 else min(max_length, past_length)
        if keep >= past_length:
            return
        if keep < 0:
            raise ValueError(f"Cannot crop {-max_length} positions from a cache of length {past_length}.")

        # Drop a fixed *count* from the tail rather than truncating to an absolute length:
        # a sliding-window layer's state is capped at its window, so it can be shorter than
        # the logical past length while still needing the same number of entries removed.
        dropped = past_length - keep
        for state in request.query_state():
            data = state.state.data
            # KV states are laid out [batch, heads, sequence, head_dim].
            if data.ndim != 4:
                continue
            state.state = ov.Tensor(np.ascontiguousarray(data[:, :, : max(data.shape[2] - dropped, 0), :]))
        self._model_part._past_length = keep

    def activate_past_recording(self) -> None:
        # Transformers asks a Cache to start recording so that `crop` can roll it back.
        # An OpenVINO variable state is always readable and writable, so nothing to arm.
        return

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._model_part._past_length

    def __len__(self) -> int:
        return self._model_part._past_length

    def __bool__(self) -> bool:
        # Truthiness decides "has a cache", which is true from the first forward onwards.
        return True


# --------------------------------------------------------------------------------------
# Drafter runtime
# --------------------------------------------------------------------------------------


class OVAssistantForCausalLM(OVBaseModel):
    """OpenVINO runtime for a DFlash draft ("assistant") model.

    Pass an instance as `assistant_model=` to a compatible OpenVINO target model's
    `generate()` to run speculative decoding:

    ```python
    from optimum.intel.openvino import OVAssistantForCausalLM, OVModelForVisualCausalLM

    model = OVModelForVisualCausalLM.from_pretrained("Muse-Glimmer-30B-ov")
    assistant = OVAssistantForCausalLM.from_pretrained("Muse-Glimmer-30B-assistant-ov")
    model.generate(**inputs, max_new_tokens=64, assistant_model=assistant)
    ```

    The exported IR takes the diffusion window's embeddings (`inputs_embeds`) and the
    target's concatenated hidden states (`hidden_states`) and returns `last_hidden_state`
    for the drafted positions. Its KV cache holds only the target context, so - unlike the
    target's - it never needs to be rolled back after a rejected block.
    """

    export_feature = "text-generation-with-past"
    auto_model_class = AutoModel

    def __init__(
        self,
        model: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = None,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
        quantization_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            model,
            config=config,
            device=device,
            dynamic_shapes=dynamic_shapes,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )
        rt_info = {}
        if not self._compile_only:
            if not is_dflash_draft_model(self.model):
                raise ValueError(
                    f"{model_save_dir} is not a DFlash draft model. Export one with "
                    "`optimum-cli export openvino --task text-generation-with-past`, from a "
                    "checkpoint whose architecture optimum-intel exports as a drafter."
                )
            rt_info = read_dflash_rt_info(self.model)

        # The RT-info block is authoritative (it travels with the IR), the config is the
        # fallback for drafters exported before it was stamped.
        self.block_size = rt_info.get("block_size") or getattr(config, "block_size", None)
        self.mask_token_id = rt_info.get("mask_token_id")
        if self.mask_token_id is None:
            self.mask_token_id = getattr(config, "mask_token_id", None)
        self.target_layer_ids = rt_info.get("target_layer_ids") or getattr(config, "target_layer_ids", None)
        if not self.block_size or self.mask_token_id is None or not self.target_layer_ids:
            raise ValueError(
                "Could not determine the DFlash parameters (block_size, mask_token_id, target_layer_ids) "
                f"of {model_save_dir}. Re-export the drafter with a recent optimum-intel."
            )
        self.request = None if not self._compile_only else self.model.create_infer_request()

    def can_generate(self) -> bool:
        # A drafter emits hidden states, not logits; it is never driven by `generate()`
        # directly, only through a target model's candidate generator.
        return False

    def _reshape(self, model: ov.Model, batch_size: int, sequence_length: int, height=None, width=None):
        # Every axis of the drafter is dynamic by construction, and the context, the
        # diffusion window and the attention mask all have different lengths, so there is
        # no single `sequence_length` to pin. Only the batch axis is set here; `beam_idx`
        # is rank-1 and has no sequence axis at all.
        shapes = {}
        for model_input in model.inputs:
            shape = model_input.get_partial_shape()
            shape[0] = batch_size
            shapes[model_input] = shape
        model.reshape(shapes)
        return model

    def compile(self):
        if self.request is None:
            super().compile()
            self.request = self.request.create_infer_request()

    def clear_requests(self):
        if not self._compile_only:
            self.request = None

    def reset_state(self):
        if self.request is not None:
            self.request.reset_state()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        self.compile()
        inputs = {
            "inputs_embeds": np.ascontiguousarray(inputs_embeds.numpy()),
            "hidden_states": np.ascontiguousarray(hidden_states.numpy()),
            "position_ids": np.ascontiguousarray(position_ids.numpy()),
            "attention_mask": np.ascontiguousarray(attention_mask.numpy()),
        }
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = np.arange(inputs_embeds.shape[0], dtype=np.int32)
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        last_hidden_state = torch.from_numpy(self.request.get_tensor("last_hidden_state").data).clone()
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# --------------------------------------------------------------------------------------
# Candidate generator
# --------------------------------------------------------------------------------------


class OVDFlashCandidateGenerator(CandidateGenerator):
    """DFlash candidate generator driving OpenVINO models.

    Adapted from `transformers.generation.candidate_generator.DFlashTokenCandidateGenerator`.
    The drafting logic is identical; what differs is where the two borrowed pieces of the
    target come from. Transformers reaches into the target's `nn.Module`s - `F.embedding`
    against the embedding weight, and a call to the lm_head - which an OpenVINO target does
    not have, because those weights live compressed inside its IR files. Here they are
    small `ov.Model`s extracted from the target's own exports (see
    `extract_raw_embedding_model` / `extract_lm_head_model`), so nothing is materialized in
    float and no second export is needed.

    The drafter's own KV cache lives inside its stateful IR, so there is no `DFlashCache`
    to crop here; the export persists only the target context, never the diffusion window.
    """

    requires_model_outputs: bool = True
    # The target must hand back the hidden states the drafter conditions on.
    model_kwargs_overrides: Dict[str, Any] = {}

    def __init__(
        self,
        assistant_model: OVAssistantForCausalLM,
        raw_embedding: "ov.CompiledModel",
        lm_head: "ov.CompiledModel",
        generation_config: GenerationConfig,
        logits_processor: Optional["LogitsProcessorList"] = None,
    ):
        self.assistant_model = assistant_model
        self.raw_embedding = raw_embedding
        self.lm_head = lm_head
        self.main_model_max_length = generation_config.max_length

        self.target_layer_ids = assistant_model.target_layer_ids
        self.block_size = assistant_model.block_size
        self.mask_token_id = assistant_model.mask_token_id
        self.noise_ids_mask = torch.tensor([self.mask_token_id] * (self.block_size - 1))[None, ...]

        self.do_sample = generation_config.do_sample
        self.logits_processor = logits_processor
        self.is_main_model_prefill = True
        assistant_model.reset_state()

    def get_candidates(
        self,
        input_ids: torch.LongTensor,
        model_kwargs: Dict[str, Any],
        model_outputs,
        is_first_iteration: bool,
        n_last_matches: int,
        **kwargs,
    ):
        # The first loop of `_assisted_decoding` runs the target so that its hidden states
        # exist for the drafter to condition on; nothing is drafted yet.
        if is_first_iteration:
            return input_ids, None

        max_new_tokens = min(int(self.block_size), self.main_model_max_length - input_ids.shape[1] - 1)
        if max_new_tokens <= 0:
            return input_ids, None

        hidden_states = getattr(model_outputs, "dflash_hidden_states", None)
        if hidden_states is None:
            raise ValueError("The target model did not return the hidden states DFlash needs.")

        # The target's last forward covered all candidates; keep only the accepted ones.
        num_last_main_model_tokens = n_last_matches + 1 if not self.is_main_model_prefill else input_ids.shape[1] - 1
        context_hidden_states = torch.cat(
            [hidden_states[layer_id][:, :num_last_main_model_tokens] for layer_id in self.target_layer_ids],
            dim=-1,
        )

        # `position_ids`/`attention_mask` cover the full sequence including the bonus token
        # the target just produced; the drafter sees the tokens the target processed plus
        # the diffusion window that follows them.
        position_ids = model_kwargs["position_ids"][:, -num_last_main_model_tokens - 1 : -1]
        attention_mask = model_kwargs["attention_mask"][:, :-1]

        # The window is the last decoded token ("anchor") followed by mask tokens.
        noise_ids = torch.cat([input_ids[:, -1:], self.noise_ids_mask.to(input_ids.device)], dim=-1)
        noise_embeds = torch.from_numpy(self.raw_embedding(np.ascontiguousarray(noise_ids.numpy()))[0]).clone()

        noise_position_ids = torch.arange(self.block_size, device=position_ids.device) + position_ids[..., -1:] + 1
        position_ids = torch.cat([position_ids, noise_position_ids], dim=-1)
        noise_attention_mask = torch.ones(
            attention_mask.shape[0], self.block_size, device=attention_mask.device, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask, noise_attention_mask], dim=-1)

        outputs = self.assistant_model(
            inputs_embeds=noise_embeds,
            hidden_states=context_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        self.is_main_model_prefill = False

        # The export already dropped the anchor position, so these states line up 1:1 with
        # the drafted tokens.
        candidate_logits = torch.from_numpy(
            self.lm_head(np.ascontiguousarray(outputs.last_hidden_state.numpy()))[0]
        ).clone()

        if self.logits_processor is not None and len(self.logits_processor) > 0:
            candidate_ids = input_ids
            for i in range(candidate_logits.shape[1]):
                next_token_logits = self.logits_processor(candidate_ids, candidate_logits[:, i, :].float())
                if self.do_sample:
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1, dtype=torch.float32)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                candidate_ids = torch.cat([candidate_ids, next_token], dim=-1)
        else:
            if self.do_sample:
                probs = torch.nn.functional.softmax(candidate_logits, dim=-1, dtype=torch.float32)
                # Assisted decoding is batch-size 1, and multinomial needs a 2D input.
                candidate_ids = torch.multinomial(probs.squeeze(0), num_samples=1)
            else:
                candidate_ids = candidate_logits.argmax(dim=-1)
            candidate_ids = torch.cat([input_ids, candidate_ids], dim=-1)

        return candidate_ids, candidate_logits

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        return


# --------------------------------------------------------------------------------------
# Target-side wiring
# --------------------------------------------------------------------------------------


class OVDFlashTargetMixin:
    """Adds DFlash speculative decoding to an OpenVINO target model.

    Mixed into models whose language part is an `OVModelWithEmbedForCausalLM`. Setup is
    lazy: nothing changes until `generate()` is actually called with an
    `OVAssistantForCausalLM`, so ordinary generation is untouched.
    """

    def _setup_dflash(self, assistant_model: OVAssistantForCausalLM) -> None:
        language_model = self.language_model
        layer_ids = list(assistant_model.target_layer_ids)
        if getattr(language_model, "_dflash_layer_ids", None) == layer_ids:
            return
        if language_model._compile_only:
            raise ValueError(
                "DFlash speculative decoding needs to add hidden-state outputs to the language model, "
                "which `compile_only=True` does not allow. Load the target without `compile_only`."
            )

        hidden_size = self.config.get_text_config().hidden_size
        language_model._dflash_hidden_state_names = add_hidden_state_outputs(language_model.model, layer_ids)
        language_model._dflash_layer_ids = layer_ids
        # The graph gained outputs, so the compiled model is stale.
        language_model.clear_requests()
        language_model.compile()

        core = ov.Core()
        ov_config = getattr(language_model, "ov_config", None) or {}
        self._dflash_lm_head = core.compile_model(
            extract_lm_head_model(language_model.model, hidden_size), language_model._device, ov_config
        )
        self._dflash_raw_embedding = core.compile_model(
            extract_raw_embedding_model(language_model.text_emb_model), language_model._device, ov_config
        )

    def generate(self, *args, **kwargs):
        # `speculation_type="dflash"` is what tells transformers' assisted decoding that
        # the drafter is a block-diffusion model rather than a small copy of the target
        # (and so that they need not share a tokenizer). It is implied by passing an
        # `OVAssistantForCausalLM`, so callers do not have to spell it out.
        assistant_model = kwargs.get("assistant_model")
        if isinstance(assistant_model, OVAssistantForCausalLM):
            kwargs.setdefault("speculation_type", "dflash")
            self._setup_dflash(assistant_model)

            # Assisted decoding insists on a Cache object it can roll back, and refuses to
            # start without one. The OpenVINO target keeps its KV cache in variable states
            # instead, so seed the proxy that exposes them; transformers leaves a
            # caller-provided cache alone. Seeding it also means the target's forward will
            # not see `past_key_values=None`, so reset the state here as that branch would.
            language_model = self.language_model
            language_model.compile()
            language_model.request.reset_state()
            language_model._past_length = 0
            language_model.next_beam_idx = np.arange(1, dtype=int)
            kwargs.setdefault("past_key_values", OVStatefulCacheProxy(language_model))
            assistant_model.reset_state()
        return super().generate(*args, **kwargs)

    def _get_candidate_generator(
        self,
        generation_config,
        input_ids,
        inputs_tensor,
        logits_processor,
        model_kwargs,
        assistant_model=None,
        **kwargs,
    ):
        if not isinstance(assistant_model, OVAssistantForCausalLM):
            return super()._get_candidate_generator(
                generation_config,
                input_ids,
                inputs_tensor,
                logits_processor,
                model_kwargs,
                assistant_model=assistant_model,
                **kwargs,
            )
        self._setup_dflash(assistant_model)
        return OVDFlashCandidateGenerator(
            assistant_model=assistant_model,
            raw_embedding=self._dflash_raw_embedding,
            lm_head=self._dflash_lm_head,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
