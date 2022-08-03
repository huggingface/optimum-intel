#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput

from .modeling_base import OVBaseModel


logger = logging.getLogger(__name__)


class OVModel(OVBaseModel):

    base_model_prefix = "ov_model"
    auto_model_class = AutoModel

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)
        self.device = torch.device("cpu") if self._device == "CPU" else torch.device("cuda")

    def to(self, device):
        self.device = device
        self._device = "CPU" if device == torch.device("cpu") else "GPU"
        self.request = self._create_infer_request()
        return self


class OVModelForSequenceClassification(OVModel):
    """
    Sequence Classification model for OpenVINO.
    """

    export_feature = "sequence-classification"
    auto_model_class = AutoModelForSequenceClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # Run inference
        outputs = self.request.infer(inputs)
        outputs = {key.get_any_name(): value for key, value in outputs.items()}
        logits = torch.from_numpy(outputs["logits"]).to(self.device)
        return SequenceClassifierOutput(logits=logits)


class OVModelForQuestionAnswering(OVModel):
    """
    Question Answering model for OpenVINO.
    """

    export_feature = "question-answering"
    auto_model_class = AutoModelForQuestionAnswering

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # Run inference
        outputs = self.request.infer(inputs)
        outputs = {key.get_any_name(): value for key, value in outputs.items()}
        start_logits = torch.from_numpy(outputs["start_logits"]).to(self.device)
        end_logits = torch.from_numpy(outputs["end_logits"]).to(self.device)
        return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)


class OVModelForTokenClassification(OVModel):
    """
    Token Classification model for OpenVINO.
    """

    export_feature = "token-classification"
    auto_model_class = AutoModelForTokenClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # Run inference
        outputs = self.request.infer(inputs)
        outputs = {key.get_any_name(): value for key, value in outputs.items()}
        logits = torch.from_numpy(outputs["logits"]).to(self.device)
        return TokenClassifierOutput(logits=logits)
