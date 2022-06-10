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

import os
import logging
import torch
import yaml
from typing import Optional, Union
from transformers import AutoModel, PreTrainedModel
from neural_compressor.experimental import Pruning, Quantization, common
from neural_compressor.experimental.scheduler import Scheduler
from .utils import WEIGHTS_NAME
from .pruning import IncPruner
from .quantization import IncQuantizer
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForMultipleChoice, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, XLNetLMHeadModel
from .utils import CONFIG_NAME


logger = logging.getLogger(__name__)


class IncOptimizer:

    TRANSFORMERS_AUTO_CLASS = AutoModel

    def __init__(
        self,
        model: Union[PreTrainedModel],
        quantizer: Optional[IncQuantizer] = None,
        pruner: Optional[IncPruner] = None,
    ):
        """
        Arguments:
            model (:obj:`Union[PreTrainedModel]`):
                Model to quantize and/or prune.
            quantizer (:obj:`IncQuantizer`, `optional`):
                Quantization object which handles the quantization process.
            pruner (:obj:`IncPruner`, `optional`):
                Pruning object which handles the pruning process.
        """
        self.config = model.config
        self.scheduler = Scheduler()
        self.scheduler.model = common.Model(model)
        self.pruner = pruner
        self.model = None

        if pruner is not None:
            self.scheduler.append(pruner.pruner)

        if quantizer is not None:
            self.scheduler.append(quantizer.quantizer)
            self.config.torch_dtype = "int8"

    def fit(self):
        # If no optimization, the original model is returned
        if len(self.scheduler.components) == 0:
            logger.error("No optimization applied.`IncOptimizer` requires either a `quantizer` or `pruner` argument")
        self.model = self.scheduler()
        return self.model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        quantizer: Optional[IncQuantizer] = None,
        pruner: Optional[IncPruner] = None,
    ):
        """
        Arguments:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            quantizer (:obj:`IncQuantizer`, `optional`):
                Quantization object which handles the quantization process.
            pruner (:obj:`IncPruner`, `optional`):
                Pruning object which handles the pruning process.
        """
        model = cls.TRANSFORMERS_AUTO_CLASS.from_pretrained(model_name_or_path)
        return cls(model, quantizer=quantizer, pruner=pruner)

    def save_pretrained(self, save_directory: Optional[Union[str, os.PathLike]] = None):
        """
        Save the optimized model to a directory, so that it can be re-loaded using `from_pretrained()`.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if self.model is None:
            logger.error(f"The model was not optimized, please call the `fit` method before saving.")
            return

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict = self.model.model.state_dict()
        if hasattr(self.model, "tune_cfg"):
            state_dict["best_configure"] = self.model.tune_cfg
            with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
                yaml.dump(self.model.tune_cfg, f, default_flow_style=False)

        torch.save(state_dict, os.path.join(save_directory, WEIGHTS_NAME))
        logger.info(f"Model weights saved to {save_directory}")


class IncOptimizerForQuestionAnswering(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncOptimizerForSequenceClassification(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class IncOptimizerForTokenClassification(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


class IncOptimizerForMultipleChoice(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = AutoModelForMultipleChoice


class IncOptimizerForSeq2SeqLM(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = AutoModelForSeq2SeqLM


class IncOptimizerForCausalLM(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = AutoModelForCausalLM


class IncOptimizerForMaskedLM(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = AutoModelForMaskedLM


class IncOptimizerForXLNetLM(IncOptimizer):

    TRANSFORMERS_AUTO_CLASS = XLNetLMHeadModel
