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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import openvino
import torch
from transformers import AutoModel
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import ModelOutput

from ...exporters.openvino.stateful import model_has_state
from .modeling import MODEL_START_DOCSTRING
from .modeling_decoder import OVBaseDecoderModel


GUARD_OUTPUT_NAMES = (
    "risk_level_logits",
    "category_logits",
    "query_risk_level_logits",
    "query_category_logits",
)


@dataclass
class OVGuardOutput(ModelOutput):
    """
    Token level moderation scores. The `query_*` heads score the user turn, the other two the
    assistant turn, so a caller picks the pair matching the role of the tokens it just submitted.
    """

    risk_level_logits: torch.FloatTensor = None
    category_logits: torch.FloatTensor = None
    query_risk_level_logits: torch.FloatTensor = None
    query_category_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@add_start_docstrings(
    """
    OpenVINO Model for guard (moderation) decoders, which expose token level classification heads
    instead of a language modeling head, such as Qwen3Guard-Stream.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForGuard(OVBaseDecoderModel):
    export_feature = "feature-extraction"
    auto_model_class = AutoModel

    def __init__(self, model=None, config=None, **kwargs):
        # unlike other decoder families, guard also has a genuinely cache-less export, so the
        # requested use_cache must default to what the loaded model actually is, not to True
        kwargs.setdefault("use_cache", model_has_state(model))
        super().__init__(model, config, **kwargs)

    def reshape(self, batch_size: int, sequence_length: int):
        # unlike OVBaseDecoderModel, the stateless IR has no incompatible 1D inputs (e.g. beam_idx),
        # so it can still be reshaped to a static shape, which the NPU requires
        if self.stateful:
            return super().reshape(batch_size, sequence_length)
        if self._compile_only:
            raise ValueError(
                "`reshape()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        shape = openvino.PartialShape([batch_size, sequence_length])
        self.model.reshape(dict.fromkeys(self.model.inputs, shape))
        self.is_dynamic = batch_size == -1 and sequence_length == -1
        self.request = None
        return self

    def reset_state(self):
        """Forget the cached conversation prefix, so the next call starts a new stream."""
        if self.request is not None and self.stateful:
            self.request.reset_state()
        self.next_beam_idx = None
        self._past_length = 0

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        if self.stateful:
            return self._past_length
        if isinstance(past_key_values[0], (tuple, list)):
            return past_key_values[0][1].shape[-2]
        return past_key_values[1].shape[-2]

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> dict:
        batch_size = input_ids.shape[0]
        inputs = {}
        if not self.stateful:
            if past_key_values is not None:
                past_key_values = tuple(
                    past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                )
                inputs = dict(zip(self.key_value_input_names, past_key_values))
            elif self.use_cache:
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    shape[0] = batch_size
                    if shape[2].is_dynamic:
                        shape[2] = 0
                    else:
                        shape[1] = 0
                    inputs[input_name] = openvino.Tensor(
                        model_inputs.get_element_type(), [dim.get_length() for dim in shape]
                    )
        elif past_key_values is None:
            if self.request is not None:
                self.request.reset_state()
            self.next_beam_idx = np.arange(batch_size, dtype=int)
            self._past_length = 0

        past_len = self._get_past_length(past_key_values)
        inputs["input_ids"] = input_ids.cpu().numpy()
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = attention_mask.cpu().numpy()
            else:
                attention_mask = np.ones(
                    (input_ids.shape[0], input_ids.shape[1] + past_len), dtype=inputs["input_ids"].dtype
                )

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = position_ids.cpu().numpy()
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
            )

        return inputs

    def _inference(self, inputs):
        # self.request is an InferRequest, which unlike a CompiledModel is not callable
        try:
            self.request.start_async(inputs, share_inputs=True)
            self.request.wait()
        except Exception as exc:
            message = self._incompatible_inputs_warning(inputs)
            if message is not None:
                exc.args += (message,)
            raise
        return {name: self.request.get_tensor(name).data for name in GUARD_OUTPUT_NAMES if name in self.output_names}

    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        position_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> OVGuardOutput:
        self.compile()

        np_inputs = isinstance(input_ids, np.ndarray)
        input_ids = torch.as_tensor(input_ids)
        if attention_mask is not None:
            attention_mask = torch.as_tensor(attention_mask)
        if position_ids is not None:
            position_ids = torch.as_tensor(position_ids)

        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        raw_logits = self._inference(inputs)
        if np_inputs:
            logits = {name: value.copy() for name, value in raw_logits.items()}
        else:
            logits = {name: torch.from_numpy(value).clone().to(self.device) for name, value in raw_logits.items()}

        if self.stateful:
            self._past_length += input_ids.shape[1]
            # a marker so the next call continues the same stream, cf. OVModelForCausalLM
            past_key_values = ((),)

        return OVGuardOutput(past_key_values=past_key_values, **logits)
