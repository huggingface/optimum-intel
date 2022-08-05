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
import os
from typing import Callable, Optional, Union

import torch
from transformers import AutoModel, PreTrainedModel

from neural_compressor.experimental import Pruning, Quantization, common
from neural_compressor.experimental.scheduler import Scheduler

from .distillation import IncDistiller
from .pruning import IncPruner
from .quantization import IncQuantizationMode, IncQuantizer
from .utils import WEIGHTS_NAME


logger = logging.getLogger(__name__)


class IncOptimizer:

    TRANSFORMERS_AUTO_CLASS = AutoModel

    def __init__(
        self,
        model: PreTrainedModel,
        quantizer: Optional[IncQuantizer] = None,
        pruner: Optional[IncPruner] = None,
        distiller: Optional[IncDistiller] = None,
        one_shot_optimization: bool = True,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """
        Arguments:
            model (`PreTrainedModel`):
                Model to quantize and/or prune.
            quantizer (`IncQuantizer`, *optional*):
                Quantization object which handles the quantization process.
            pruner (`IncPruner`, *optional*):
                Pruning object which handles the pruning process.
            distiller (`IncDistiller`, *optional*):
                Distillation object which handles the distillation process.
            one_shot_optimization (`bool`, *optional*, defaults to True):
                Whether to apply the compression processes all together.
            eval_func (`Callable`, *optional*):
                Evaluation function to evaluate the tuning objective.
            train_func (`Callable`, *optional*):
                Training function which will be combined with pruning.
        """
        self.config = model.config
        self.scheduler = Scheduler()
        self.scheduler.model = common.Model(model)
        self._model = None
        self.apply_pruning = False
        self.apply_quantization = False
        self.one_shot_optimization = one_shot_optimization

        components = []
        if pruner is not None:
            components.append(pruner.pruning)
            self.apply_pruning = True

        criterion = None
        if distiller is not None:
            distiller.distillation.model = self.scheduler.model
            components.append(distiller.distillation)
            distiller.distillation.create_criterion()
            criterion = getattr(distiller.distillation, "criterion", None)

        if quantizer is not None:
            if quantizer.approach == IncQuantizationMode.AWARE_TRAINING:
                components.append(quantizer.quantization)
            self.apply_quantization = True
            self.config.torch_dtype = "int8"

        if one_shot_optimization and len(components) > 1:
            agent = self.scheduler.combine(*components)
            agent.train_func = train_func
            agent.eval_func = eval_func
            agent.criterion = criterion
            self.scheduler.append(agent)
        else:
            self.scheduler.append(*components)

        if self.apply_quantization and quantizer.approach != IncQuantizationMode.AWARE_TRAINING:
            self.scheduler.append(quantizer.quantization)

    def fit(self):
        # If no optimization, the original model is returned
        if len(self.scheduler.components) == 0:
            logger.error(
                "No optimization applied.`IncOptimizer` requires either a `quantizer`, a `pruner` or a `distiller` argument."
            )
        self._model = self.scheduler()
        return self.model

    @property
    def model(self):
        return self._model.model if self._model is not None else None

    def get_agent(self):
        return self.scheduler.components[0] if self.one_shot_optimization or self.apply_pruning else None

    def get_sparsity(self):
        sparsity = self._model.report_sparsity()
        return sparsity[-1]

    def save_pretrained(self, save_directory: Optional[Union[str, os.PathLike]] = None):
        """
        Save the optimized model as well as its corresponding configuration to a directory, so that it can be re-loaded.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if self._model is None:
            logger.error(f"The model was not optimized, please call the `fit` method before saving.")
            return

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict = self._model.model.state_dict()
        if hasattr(self._model, "q_config"):
            state_dict["best_configure"] = self._model.q_config
        elif hasattr(self._model, "tune_cfg"):
            state_dict["best_configure"] = self._model.tune_cfg
        torch.save(state_dict, os.path.join(save_directory, WEIGHTS_NAME))
        logger.info(f"Model weights saved to {save_directory}")
