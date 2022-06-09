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
from typing import Optional, Union

import torch
from transformers import AutoModel, PreTrainedModel

from neural_compressor.experimental import Pruning, Quantization, common
from neural_compressor.experimental.scheduler import Scheduler

from .pruning import IncPruner
from .quantization import IncQuantizer


logger = logging.getLogger(__name__)


class IncOptimizer:

    TRANSFORMERS_AUTO_CLASS = AutoModel

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        quantizer: Optional[IncQuantizer] = None,
        pruner: Optional[IncPruner] = None,
    ):
        """
        Args:
            model (:obj:`Union[PreTrainedModel, torch.nn.Module]`):
                Model to quantize and/or prune.
            quantizer (:obj:`IncQuantizer`, `optional`):
                Quantization object which handles the quantization process.
            pruner (:obj:`IncPruner`, `optional`):
                Pruning object which handles the pruning process.
        """
        self.model = model
        self.scheduler = Scheduler()
        self.scheduler.model = common.Model(self.model)

        if pruner is not None and isinstance(pruner.pruner, Pruning):
            self.scheduler.append(pruner.pruner)

        if quantizer is not None and isinstance(quantizer.quantizer, Quantization):
            self.scheduler.append(quantizer.quantizer)

    def fit(self):
        # If no optimization, the original model is returned
        if len(self.scheduler.components) == 0:
            return self.model
        opt_model = self.scheduler()
        return opt_model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        quantizer: Optional[IncQuantizer] = None,
        pruner: Optional[IncPruner] = None,
    ):
        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(model_name_or_path)
        return cls(model, quantizer=quantizer, pruner=pruner)


class IncOptimizerForQuestionAnswering(IncOptimizer):
    from transformers import AutoModelForQuestionAnswering

    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncOptimizerForSequenceClassification(IncOptimizer):
    from transformers import AutoModelForSequenceClassification

    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class IncOptimizerForTokenClassification(IncOptimizer):
    from transformers import AutoModelForTokenClassification

    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


class IncOptimizerForMultipleChoice(IncOptimizer):
    from transformers import AutoModelForMultipleChoice

    TRANSFORMERS_AUTO_CLASS = AutoModelForMultipleChoice


class IncOptimizerForSeq2SeqLM(IncOptimizer):
    from transformers import AutoModelForSeq2SeqLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForSeq2SeqLM


class IncOptimizerForCausalLM(IncOptimizer):
    from transformers import AutoModelForCausalLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForCausalLM


class IncOptimizerForMaskedLM(IncOptimizer):
    from transformers import AutoModelForMaskedLM

    TRANSFORMERS_AUTO_CLASS = AutoModelForMaskedLM


class IncOptimizerForXLNetLM(IncOptimizer):
    from transformers import XLNetLMHeadModel

    TRANSFORMERS_AUTO_CLASS = XLNetLMHeadModel
