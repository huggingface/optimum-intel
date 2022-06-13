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
from enum import Enum
from typing import Callable, ClassVar, Optional, Union

from neural_compressor.conf.config import Pruning_Conf
from neural_compressor.experimental import Pruning, common

from .configuration import IncPruningConfig


logger = logging.getLogger(__name__)


class IncPruningMode(Enum):
    MAGNITUDE = "basic_magnitude"


SUPPORTED_PRUNING_MODE = set([approach.value for approach in IncPruningMode])


class IncPruner:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(
        self,
        config_path_or_obj: Union[str, IncPruningConfig],
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """
        Arguments:
            config_path_or_obj (:obj:`Union[str, IncPruningConfig]` ):
                Path to the YAML configuration file or an instance of the class :class:`IncPruningConfig`, used to
                control the tuning behavior.
            eval_func (:obj:`Callable`, `optional`):
                Evaluation function to evaluate the tuning objective.
            train_func (:obj:`Callable`, `optional`):
                Training function which will be combined with pruning.
        Returns:
            pruner: IncPruner object.
        """

        self.config = (
            config_path_or_obj.config
            if isinstance(config_path_or_obj, IncPruningConfig)
            else Pruning_Conf(config_path_or_obj)
        )
        self.eval_func = eval_func
        self.train_func = train_func
        self.pruner = Pruning(self.config)

        if self.eval_func is None:
            raise ValueError("eval_func must be provided for pruning.")

        if self.train_func is None:
            raise ValueError("train_func must be provided for pruning.")

        self.pruner.pruning_func = self.train_func
        self.pruner.eval_func = self.eval_func
