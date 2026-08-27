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

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import openvino
import torch
from transformers import AutoModel
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import ModelOutput

from ...exporters.openvino.stateful import model_has_state
from .modeling import MODEL_START_DOCSTRING, OVModel
from .utils import ensure_numpy


logger = logging.getLogger(__name__)


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
class OVModelForGuard(OVModel):
    export_feature = "feature-extraction"
    auto_model_class = AutoModel

    def __init__(self, model=None, config=None, **kwargs):
        self.stateful = model_has_state(model)
        if self.stateful:
            # the KV cache is hidden in the model, which makes it dynamic by construction and adds a
            # 1D `beam_idx` input that the generic reshape cannot handle, cf. OVModelForCausalLM
            kwargs["dynamic_shapes"] = False

        super().__init__(model, config, **kwargs)
        self._past_length = 0

    def compile(self):
        super().compile()
        if isinstance(self.request, openvino.CompiledModel):
            # the KV cache lives in the infer request, so the compiled model cannot be called directly
            self.request = self.request.create_infer_request()

    def reset_state(self):
        """Forget the cached conversation prefix, so the next call starts a new stream."""
        if self.request is not None and self.stateful:
            self.request.reset_state()
        self._past_length = 0

    def _inference(self, inputs):
        # self.request is an InferRequest, which unlike a CompiledModel is not callable
        try:
            return self.request.infer(inputs)
        except Exception as exc:
            message = self._incompatible_inputs_warning(inputs)
            if message is not None:
                exc.args += (message,)
            raise

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
        input_ids = ensure_numpy(input_ids)
        batch_size, sequence_length = input_ids.shape

        if self.stateful and past_key_values is None:
            self.reset_state()
        past_length = self._past_length if self.stateful else 0

        if attention_mask is not None:
            attention_mask = ensure_numpy(attention_mask)
        else:
            attention_mask = np.ones((batch_size, past_length + sequence_length), dtype=input_ids.dtype)

        inputs = {"input_ids": input_ids}
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is None:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                position_ids = position_ids[:, past_length:]
            inputs["position_ids"] = ensure_numpy(position_ids)

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = np.arange(batch_size, dtype=int)

        outputs = self._inference(inputs)
        logits = {name: outputs[name] for name in GUARD_OUTPUT_NAMES if name in self.output_names}
        if not np_inputs:
            logits = {name: torch.from_numpy(value).to(self.device) for name, value in logits.items()}

        if self.stateful:
            self._past_length += sequence_length
            # a marker so that the next call knows it continues the same stream, cf. OVModelForCausalLM
            past_key_values = ((),)

        return OVGuardOutput(past_key_values=past_key_values, **logits)
