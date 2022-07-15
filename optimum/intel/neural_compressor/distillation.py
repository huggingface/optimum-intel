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
from typing import Callable, Optional, Union

from transformers import PreTrainedModel

from neural_compressor.conf.config import Distillation_Conf
from neural_compressor.experimental import Distillation, common

from .configuration import IncDistillationConfig


logger = logging.getLogger(__name__)


class IncDistiller:
    def __init__(
        self,
        config: Union[str, IncDistillationConfig],
        teacher_model: PreTrainedModel,
        eval_func: Optional[Callable],
        train_func: Optional[Callable],
    ):
        """
        Arguments:
            config (`Union[str, IncDistillationConfig]`):
                Path to the YAML configuration file or an instance of the class :class:`IncDistillationConfig`, used to
                control the tuning behavior.
            teacher_model (`PreTrainedModel`):
                Teacher model.
            eval_func (`Callable`):
                Evaluation function to evaluate the tuning objective.
            train_func (`Callable`):
                Training function which will be combined with knowledge distillation.
        """

        self.config = config.config if isinstance(config, IncDistillationConfig) else Distillation_Conf(config)
        self.eval_func = eval_func
        self.train_func = train_func
        self.distillation = Distillation(self.config)
        self.distillation.train_func = self.train_func
        self.distillation.eval_func = self.eval_func
        self.distillation.teacher_model = common.Model(teacher_model)
