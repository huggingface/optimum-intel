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

A DFlash drafter reads the target's hidden states at a few layers and drafts a block of
`block_size` tokens in one forward pass; the target then verifies the block in one forward.

The drafter has no token embedding or lm_head of its own and uses the target's. The
embedding is the target's text-embeddings model; the lm_head lives inside the language
model IR, so this module extracts it as a small standalone `ov.Model`.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
import openvino as ov
import torch
from openvino import opset13
from transformers import AutoModel, GenerationConfig, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..utils.import_utils import is_transformers_version
from .modeling_base import OVBaseModel


if is_transformers_version(">=", "5.15"):
    from transformers.generation.candidate_generator import DFlashTokenCandidateGenerator
else:
    DFlashTokenCandidateGenerator = object

if TYPE_CHECKING:
    from transformers.generation.logits_process import LogitsProcessorList


HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers"
DFLASH_HIDDEN_STATE_PREFIX = "dflash_hidden_states"


def is_dflash_draft_model(ov_model: ov.Model) -> bool:
    """Whether an IR was exported as a DFlash drafter."""
    return ov_model.has_rt_info(["dflash_mode"]) and ov_model.get_rt_info(["dflash_mode"]).value == "True"


def dflash_draft_has_lm_head(ov_model: ov.Model) -> bool:
    """Whether a DFlash draft IR computes its own logits instead of reusing the target LM head at runtime."""
    return any("logits" in output.get_names() for output in ov_model.outputs)


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

    Each step of the exported IR runs the new rows [context; block]: `token_type_ids` marks context
    rows (0), which read the target's concatenated hidden states (`hidden_states`), and block rows (1),
    which read the diffusion window's embeddings (`inputs_embeds`). `last_hidden_state` covers every
    row. `forward` follows the call signature of transformers' `DFlashTokenCandidateGenerator`. The KV
    cache stores the block rows too, so `OVDFlashCache` crops them before the next step.
    """

    export_feature = "text-generation-with-past"
    auto_model_class = AutoModel

    def __init__(self, model: ov.Model, config: PretrainedConfig = None, **kwargs):
        super().__init__(model, config=config, **kwargs)
        if not self._compile_only and not is_dflash_draft_model(self.model):
            raise ValueError(
                f"{self.model_save_dir} is not a DFlash draft model. Export one with "
                "`optimum-cli export openvino --task text-generation-with-past`, from a "
                "checkpoint whose architecture optimum-intel exports as a drafter."
            )
        if not self._compile_only and "token_type_ids" not in self.input_names:
            raise ValueError(
                f"The DFlash draft model in {self.model_save_dir} has no `token_type_ids` input. "
                "It was exported by an older optimum-intel; re-export it with the current one."
            )
        if (
            not getattr(config, "block_size", None)
            or getattr(config, "mask_token_id", None) is None
            or not getattr(config, "target_layer_ids", None)
        ):
            raise ValueError(
                f"The config of {self.model_save_dir} lacks the DFlash parameters "
                "(block_size, mask_token_id, target_layer_ids)."
            )
        self.request = None if not self._compile_only else self.model.create_infer_request()
        self._past_length = 0

    def can_generate(self) -> bool:
        # A drafter emits hidden states, not logits; it is never driven by `generate()`
        # directly, only through a target model's candidate generator.
        return False

    def _reshape(self, model: ov.Model, batch_size: int, sequence_length: int, height=None, width=None):
        # Every axis is dynamic; only the batch axis is pinned.
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
        self._past_length = 0

    def forward(
        self,
        noise_embeds: torch.Tensor,
        context_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        # `past_key_values` is accepted for transformers' call signature only: the drafter's
        # KV cache lives in the IR's variable states.
        self.compile()
        batch_size, context_len = context_hidden_states.shape[:2]
        block_len = noise_embeds.shape[1]
        # Each row reads one of the two inputs; the other is zero there.
        inputs_embeds = torch.cat(
            [noise_embeds.new_zeros(batch_size, context_len, noise_embeds.shape[-1]), noise_embeds], 1
        )
        hidden_states = torch.cat(
            [
                context_hidden_states,
                context_hidden_states.new_zeros(batch_size, block_len, context_hidden_states.shape[-1]),
            ],
            1,
        )
        token_type_ids = torch.cat(
            [
                torch.zeros(batch_size, context_len, dtype=torch.long),
                torch.ones(batch_size, block_len, dtype=torch.long),
            ],
            1,
        )
        inputs = {
            "inputs_embeds": np.ascontiguousarray(inputs_embeds.numpy()),
            "hidden_states": np.ascontiguousarray(hidden_states.numpy()),
            "position_ids": np.ascontiguousarray(position_ids.numpy()),
            "attention_mask": np.ascontiguousarray(attention_mask.numpy()),
            "token_type_ids": token_type_ids.numpy(),
        }
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = np.arange(batch_size, dtype=np.int32)
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        self._past_length += context_len + block_len
        # The candidate generator expects the block rows only.
        last_hidden_state = torch.from_numpy(
            self.request.get_tensor("last_hidden_state").data[:, context_len:]
        ).clone()
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_state)


# --------------------------------------------------------------------------------------
# Candidate generator
# --------------------------------------------------------------------------------------


class OVDFlashTargetLayerWeight:
    """Stands in for the `weight` tensor of an `OVDFlashTargetLayer`.

    Transformers 5.15 uses the weight of the target's layers in two ways only: it reads the
    lm_head's `weight.device`, and embeds tokens with `F.embedding(ids, embedding.weight)`.
    PyTorch passes any torch call that receives an object defining `__torch_function__` to
    that method, so the embedding lookup runs the OpenVINO layer instead.
    """

    device = torch.device("cpu")

    def __init__(self, layer: "OVDFlashTargetLayer"):
        self._layer = layer

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is not torch.nn.functional.embedding:
            return NotImplemented
        input_ids, weight = args[0], args[1]
        return weight._layer(input_ids)


class OVDFlashTargetLayer:
    """A layer of the target (embedding lookup or lm_head) run by OpenVINO, called on torch tensors."""

    def __init__(self, run: Callable[[np.ndarray], np.ndarray]):
        self._run = run
        self.weight = OVDFlashTargetLayerWeight(self)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._run(np.ascontiguousarray(inputs.numpy()))).clone()


class OVDFlashCache(OVStatefulCacheProxy):
    """Replaces transformers' `DFlashCache` with the drafter's variable states.

    The candidate generator crops the previous block rows (`crop(-block_size)`) before each step.
    """

    def set_previous_accepted_tokens(self, number_of_accepted_tokens: int) -> None:
        # transformers needs this to size its masks; the OpenVINO drafter takes the full mask as input
        return


class OVDFlashCandidateGenerator(DFlashTokenCandidateGenerator):
    """`DFlashTokenCandidateGenerator` driving OpenVINO models.

    The drafter needs the target's raw embedding lookup, which is what the target's text-embeddings
    model returns, and its lm_head, which is extracted from the language model IR. Nothing is
    materialized in float and no second export is needed.
    """

    # The target exposes the drafter's hidden states as graph outputs (see `_setup_dflash`),
    # so it does not need `output_hidden_states=True`.
    model_kwargs_overrides: Dict[str, Any] = {}

    def __init__(
        self,
        assistant_model: OVAssistantForCausalLM,
        embed_tokens: Callable[[np.ndarray], np.ndarray],
        lm_head: ov.CompiledModel,
        generation_config: GenerationConfig,
        logits_processor: Optional["LogitsProcessorList"] = None,
    ):
        super().__init__(
            assistant_model=assistant_model,
            main_model_input_embeddings=OVDFlashTargetLayer(embed_tokens),
            main_model_output_embeddings=OVDFlashTargetLayer(lambda hidden_states: lm_head(hidden_states)[0]),
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
        self.cache = OVDFlashCache(assistant_model)
        assistant_model.reset_state()


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
        if is_transformers_version("<", "5.15"):
            raise ValueError("DFlash speculative decoding with OpenVINO models requires transformers >= 5.15.")
        language_model = self.language_model
        layer_ids = list(assistant_model.config.target_layer_ids)
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

    def generate(self, *args, **kwargs):
        assistant_model = kwargs.get("assistant_model")
        if isinstance(assistant_model, OVAssistantForCausalLM):
            # Selects transformers' DFlash candidate generator.
            kwargs.setdefault("speculation_type", "dflash")
            self._setup_dflash(assistant_model)

            # Assisted decoding needs a cache it can crop. Pass the proxy over the target's
            # variable states, and reset them as the target does when `past_key_values=None`.
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
            embed_tokens=self.language_model.embed_tokens,
            lm_head=self._dflash_lm_head,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
