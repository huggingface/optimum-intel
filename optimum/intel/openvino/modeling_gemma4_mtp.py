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
"""
OpenVINO modeling classes for Gemma 4 Multi-Token-Prediction (MTP) assisted
decoding.

Two classes are defined here:

* ``Gemma4AssistantOVForCausalLM`` (aliased as ``OVAssistantForCausalLM``) — the
  OpenVINO runtime backend for ``transformers.Gemma4AssistantForCausalLM``. The
  class name intentionally starts with ``Gemma4Assistant`` because
  ``transformers.generation.candidate_generator.SinglePositionMultiTokenCandidateGenerator``
  dispatches on ``assistant_model.__class__.__name__.startswith("Gemma4Assistant")``.

* ``OVGemma4ForCausalLM`` — a subclass of :class:`OVModelForCausalLM` that, in
  addition to ``logits``, surfaces ``hidden_states`` (last layer) and
  ``shared_kv_states`` (``full_attention`` and ``sliding_attention`` last-layer
  K/V) from the OpenVINO IR. These extra outputs are what
  ``transformers.generation.utils._assisted_decoding`` feeds to the MTP
  candidate generator.

IR contract
-----------

For ``OVGemma4ForCausalLM`` the OpenVINO IR is expected to expose the regular
``logits`` output plus the following extra outputs (added by the dedicated
exporter):

* ``mtp_last_hidden_state``         — shape ``(B, S, hidden)``
* ``mtp_full_attention_key``        — shape ``(B, num_kv_heads, S, head_dim)``
* ``mtp_full_attention_value``      — shape ``(B, num_kv_heads, S, head_dim)``
* ``mtp_sliding_attention_key``     — shape ``(B, num_kv_heads, S, head_dim)``
* ``mtp_sliding_attention_value``   — shape ``(B, num_kv_heads, S, head_dim)``

For ``Gemma4AssistantOVForCausalLM`` the IR is expected to expose:

Inputs
  * ``inputs_embeds``        — shape ``(B, 1, 2*backbone_hidden_size)``
  * ``position_ids``         — shape ``(B, 1)``
  * ``attention_mask``       — shape ``(B, S_kv)``
  * ``full_attention.key``   — shape ``(B, num_kv_heads, S_kv, head_dim)``
  * ``full_attention.value`` — shape ``(B, num_kv_heads, S_kv, head_dim)``
  * ``sliding_attention.key``   — shape ``(B, num_kv_heads, S_kv, head_dim)``
  * ``sliding_attention.value`` — shape ``(B, num_kv_heads, S_kv, head_dim)``

Outputs
  * ``logits``             — shape ``(B, 1, vocab_size)``
  * ``last_hidden_state``  — shape ``(B, 1, backbone_hidden_size)``

These contracts are produced by the exporter added alongside this module.

If the loaded IR does not match the contract, :meth:`forward` raises a
descriptive :class:`RuntimeError` so users can re-export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import openvino
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

from .modeling import OVModel
from .utils import ONNX_WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers.generation.configuration_utils import GenerationConfig


logger = logging.getLogger(__name__)


# Names of the extra outputs that the dedicated exporter must add to the
# Gemma4 target IR. Keep these in sync with the exporter
# (see ``Gemma4TextOpenVINOConfig.outputs`` and ``_Gemma4MTPOutput`` in
# optimum-intel's openvino exporter).  Dots are not valid in IR output names
# in this code path, so the exporter uses ``mtp_*`` flat names.
_TARGET_EXTRA_OUTPUTS = (
    "mtp_last_hidden_state",
    "mtp_full_attention_key",
    "mtp_full_attention_value",
    "mtp_sliding_attention_key",
    "mtp_sliding_attention_value",
)

# Names of the inputs / outputs of the assistant IR.
_ASSISTANT_REQUIRED_INPUTS = (
    "inputs_embeds",
    "position_ids",
    "full_attention.key",
    "full_attention.value",
    "sliding_attention.key",
    "sliding_attention.value",
)
_ASSISTANT_REQUIRED_OUTPUTS = ("logits", "last_hidden_state")


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass
class Gemma4OVCausalLMOutput(ModelOutput):
    """Output of :class:`OVGemma4ForCausalLM` carrying the extra MTP fields."""

    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    shared_kv_states: Optional[Dict[str, Tuple[torch.FloatTensor, torch.FloatTensor]]] = None


@dataclass
class Gemma4AssistantOVOutput(BaseModelOutput):
    """Output of :class:`Gemma4AssistantOVForCausalLM`."""

    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


# ---------------------------------------------------------------------------
# OVGemma4ForCausalLM — target subclass of OVModelForCausalLM
# ---------------------------------------------------------------------------


def _import_ov_causal_lm():
    """Late import to avoid circular import with :mod:`modeling_decoder`."""
    from .modeling_decoder import OVModelForCausalLM

    return OVModelForCausalLM


def _build_ov_gemma4_class():
    """Construct :class:`Gemma4OVForCausalLM` lazily.

    The class is defined inside a function because :class:`OVModelForCausalLM`
    is defined in :mod:`modeling_decoder` which imports *this* module at the
    end (to register the subclass). Defining the class lazily breaks the
    circular import.

    The class is named ``Gemma4OVForCausalLM`` (starting with ``"Gemma4"``) so
    that ``transformers.generation.utils._get_candidate_generator`` accepts it
    as a valid MTP target (gate: ``self.__class__.__name__.startswith(("Gemma4",
    "Gemma3n"))`` at ``generation/utils.py:991``).
    """
    OVModelForCausalLM = _import_ov_causal_lm()

    class Gemma4OVForCausalLM(OVModelForCausalLM):
        """OpenVINO Gemma 4 causal LM.

        Subclass of :class:`OVModelForCausalLM` whose forward additionally
        returns ``hidden_states`` (last layer) and ``shared_kv_states``, which
        are required by ``SinglePositionMultiTokenCandidateGenerator`` to draft
        candidates with a :class:`Gemma4AssistantOVForCausalLM` assistant.
        """

        def _has_mtp_outputs(self) -> bool:
            output_names = set()
            for out in self.model.outputs:
                output_names.update(out.get_names())
            return all(name in output_names for name in _TARGET_EXTRA_OUTPUTS)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
            token_type_ids=None,
            **kwargs,
        ):
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                **kwargs,
            )

            if not self._has_mtp_outputs():
                # Standard IR without MTP outputs: behave like the parent.
                return outputs

            # The OpenVINO request was just run by the parent forward(); reuse
            # the same inference result to pull the extra tensors out.
            request = self.request
            try:
                last_hidden_state = torch.from_numpy(
                    request.get_tensor("mtp_last_hidden_state").data.copy()
                ).to(self.device)
                shared_kv_states = {
                    "full_attention": (
                        torch.from_numpy(
                            request.get_tensor("mtp_full_attention_key").data.copy()
                        ).to(self.device),
                        torch.from_numpy(
                            request.get_tensor("mtp_full_attention_value").data.copy()
                        ).to(self.device),
                    ),
                    "sliding_attention": (
                        torch.from_numpy(
                            request.get_tensor("mtp_sliding_attention_key").data.copy()
                        ).to(self.device),
                        torch.from_numpy(
                            request.get_tensor("mtp_sliding_attention_value").data.copy()
                        ).to(self.device),
                    ),
                }
            except RuntimeError as exc:  # missing tensor
                raise RuntimeError(
                    "The loaded Gemma4 OpenVINO IR does not expose the MTP outputs "
                    f"({_TARGET_EXTRA_OUTPUTS!r}). Re-export the model with the "
                    "Gemma4 MTP exporter to enable assisted decoding."
                ) from exc

            return Gemma4OVCausalLMOutput(
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=(last_hidden_state,),  # tuple of one is enough — MTP only reads [-1]
                shared_kv_states=shared_kv_states,
            )

    return Gemma4OVForCausalLM


# Lazy singleton — built on first attribute access.
_OVGemma4ForCausalLM_cls: Optional[type] = None


def __getattr__(name: str):
    global _OVGemma4ForCausalLM_cls
    if name in ("Gemma4OVForCausalLM", "OVGemma4ForCausalLM"):
        if _OVGemma4ForCausalLM_cls is None:
            _OVGemma4ForCausalLM_cls = _build_ov_gemma4_class()
        return _OVGemma4ForCausalLM_cls
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Gemma4AssistantOVForCausalLM — the OV assistant model
# ---------------------------------------------------------------------------


class Gemma4AssistantOVForCausalLM(OVModel, GenerationMixin):
    """OpenVINO backend for ``transformers.Gemma4AssistantForCausalLM``.

    Implements just enough of the ``PreTrainedModel`` API for the
    ``SinglePositionMultiTokenCandidateGenerator`` from transformers to use
    this class as ``assistant_model``. Specifically it provides:

    * ``__call__``/``forward`` with the signature
      ``(inputs_embeds, attention_mask, position_ids, shared_kv_states, use_cache)``
      returning an object with ``logits`` and ``last_hidden_state`` attributes,
      matching :class:`transformers.models.gemma4_assistant.Gemma4AssistantOutput`.
    * ``device``, ``config``, ``generation_config`` attributes (inherited).
    * Class name starting with ``Gemma4Assistant`` (gated on by the candidate
      generator).
    """

    # Expose under the transformers AutoModel registry name so the candidate
    # generator's ``startswith("Gemma4Assistant")`` check passes.
    export_feature = "gemma4-assistant"
    base_model_prefix = "openvino_assistant_model"
    main_input_name = "inputs_embeds"

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        compile_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else cls._all_ov_model_paths["model"]
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=Path(model_id),
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        if not compile_only:
            model = cls.load_model(model_cache_path)
        else:
            model = cls._compile_model(
                model_cache_path,
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_cache_path.parent,
            )

        compile_model = kwargs.pop("compile", True)
        return cls(
            model=model,
            config=config,
            model_save_dir=model_cache_path.parent,
            compile=compile_model,
            compile_only=compile_only,
            **kwargs,
        )

    # -- forward -----------------------------------------------------------

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().numpy()

    def _check_io_contract(self) -> None:
        input_names: set = set()
        for inp in self.model.inputs:
            input_names.update(inp.get_names())
        missing_in = [n for n in _ASSISTANT_REQUIRED_INPUTS if n not in input_names]
        if missing_in:
            raise RuntimeError(
                "The loaded Gemma4Assistant OpenVINO IR is missing required inputs: "
                f"{missing_in!r}. Re-export the assistant model with the dedicated "
                "Gemma4 assistant exporter."
            )
        output_names: set = set()
        for out in self.model.outputs:
            output_names.update(out.get_names())
        missing_out = [n for n in _ASSISTANT_REQUIRED_OUTPUTS if n not in output_names]
        if missing_out:
            raise RuntimeError(
                "The loaded Gemma4Assistant OpenVINO IR is missing required outputs: "
                f"{missing_out!r}."
            )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        shared_kv_states: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,  # accepted but unused
        **kwargs: Any,
    ) -> Gemma4AssistantOVOutput:
        if shared_kv_states is None:
            raise ValueError("Gemma4AssistantOVForCausalLM.forward requires `shared_kv_states`.")

        self.compile()
        self._check_io_contract()

        full_k, full_v = shared_kv_states["full_attention"]
        slid_k, slid_v = shared_kv_states["sliding_attention"]

        inputs = {
            "inputs_embeds": self._to_numpy(inputs_embeds),
            "position_ids": self._to_numpy(position_ids),
            "full_attention.key": self._to_numpy(full_k),
            "full_attention.value": self._to_numpy(full_v),
            "sliding_attention.key": self._to_numpy(slid_k),
            "sliding_attention.value": self._to_numpy(slid_v),
        }
        if attention_mask is not None:
            inputs["attention_mask"] = self._to_numpy(attention_mask)

        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor("logits").data.copy()).to(self.device)
        last_hidden_state = torch.from_numpy(
            self.request.get_tensor("last_hidden_state").data.copy()
        ).to(self.device)

        return Gemma4AssistantOVOutput(logits=logits, last_hidden_state=last_hidden_state)


# User-facing alias.
OVAssistantForCausalLM = Gemma4AssistantOVForCausalLM


__all__ = [
    "OVAssistantForCausalLM",
    "Gemma4AssistantOVForCausalLM",
    "Gemma4OVCausalLMOutput",
    "Gemma4AssistantOVOutput",
    # ``OVGemma4ForCausalLM`` is exposed via module ``__getattr__``.
]
