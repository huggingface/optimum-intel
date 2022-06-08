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

import neural_compressor
from neural_compressor.experimental import Pruning, Quantization, common
from neural_compressor.experimental.scheduler import Scheduler

from .quantization import IncQuantizationMode
from .utils import IncDataLoader, _cfgs_to_fx_cfgs


logger = logging.getLogger(__name__)


class IncOptimizer:

    TRANSFORMERS_AUTO_CLASS = AutoModel

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        eval_func=None,
        train_func=None,
        calib_dataloader=None,
        quantization_config=None,
        pruning_config=None,
    ):
        self.eval_func = eval_func
        self.train_func = train_func
        self.calib_dataloader = calib_dataloader
        self.quantization_config = quantization_config
        self.pruning_config = pruning_config
        self.model = model
        self.pruner = None

        self.scheduler = Scheduler()

        if quantization_config is not None:
            quantizer = self.get_quantizer()
            self.scheduler.append(quantizer)

        if pruning_config is not None:
            self.pruner = self.get_pruner()
            self.scheduler.append(self.pruner)

        self.scheduler.model = common.Model(self.model)

    def fit(self):
        # If no optimization, the original model is returned
        if len(self.scheduler.components) == 0:
            return self.model
        opt_model = self.scheduler()
        return opt_model

    def get_quantizer(self):
        approach = IncQuantizationMode(self.quantization_config.usr_cfg.quantization.approach)

        if self.quantization_config.usr_cfg.model.framework == "pytorch_fx":
            neural_compressor.adaptor.pytorch._cfgs_to_fx_cfgs = _cfgs_to_fx_cfgs

        quantizer = Quantization(self.quantization_config.config)

        if self.eval_func is None:
            raise ValueError("eval_func must be provided for quantization.")

        quantizer.eval_func = self.eval_func

        if approach == IncQuantizationMode.STATIC:
            if self.calib_dataloader is None:
                raise ValueError("calib_dataloader must be provided for static quantization.")
            quantizer._calib_dataloader = IncDataLoader.from_pytorch_dataloader(self.calib_dataloader)

        if approach == IncQuantizationMode.AWARE_TRAINING:
            if self.train_func is None:
                raise ValueError("train_func must be provided for quantization aware training.")
            quantizer.q_func = self.train_func

        return quantizer

    def get_pruner(self):
        if self.eval_func is None:
            raise ValueError("eval_func must be provided for pruning.")

        if self.train_func is None:
            raise ValueError("train_func must be provided for pruning.")

        pruner = Pruning(self.pruning_config.config)
        pruner.pruning_func = self.train_func
        pruner.eval_func = self.eval_func

        return pruner

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, kwargs):
        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(model_name_or_path)
        return cls(model, **kwargs)


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