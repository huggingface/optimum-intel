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
from typing import Dict, Optional, Union

from neural_compressor.config import DistillationConfig, WeightPruningConfig, _BaseQuantizationConfig

from optimum.configuration_utils import BaseConfig

from ..utils.import_utils import _neural_compressor_version, _torch_version


_quantization_model = {
    "post_training_dynamic_quant": "dynamic",
    "post_training_static_quant": "static",
    "quant_aware_training": "aware_training",
    "post_training_weight_only": "weight_only",
}

logger = logging.getLogger(__name__)


class INCConfig(BaseConfig):
    CONFIG_NAME = "inc_config.json"
    FULL_CONFIGURATION_FILE = "inc_config.json"

    def __init__(
        self,
        quantization: Optional[Union[Dict, _BaseQuantizationConfig]] = None,
        pruning: Optional[Union[Dict, _BaseQuantizationConfig]] = None,
        distillation: Optional[Union[Dict, _BaseQuantizationConfig]] = None,
        **kwargs,
    ):
        super().__init__()
        self.torch_version = _torch_version
        self.neural_compressor_version = _neural_compressor_version
        self.quantization = self._create_quantization_config(quantization) or {}
        self.pruning = self._create_pruning_config(pruning) or {}
        self.distillation = self._create_distillation_config(distillation) or {}

    @staticmethod
    def _create_quantization_config(config: Union[Dict, _BaseQuantizationConfig]):
        # TODO : add activations_dtype and weights_dtype
        if isinstance(config, _BaseQuantizationConfig):
            approach = _quantization_model[config.approach]
            config = {
                "is_static": approach != "dynamic",
                "dataset_num_samples": config.calibration_sampling_size[0] if approach == "static" else None,
                # "approach" : approach,
            }
        return config

    @staticmethod
    def _create_pruning_config(config: Union[Dict, WeightPruningConfig]):
        if isinstance(config, WeightPruningConfig):
            weight_compression = config.weight_compression
            config = {
                "approach": weight_compression.pruning_type,
                "pattern": weight_compression.pattern,
                "sparsity": weight_compression.target_sparsity,
                # "operators": weight_compression.pruning_op_types,
                # "start_step": weight_compression.start_step,
                # "end_step": weight_compression.end_step,
                # "scope": weight_compression.pruning_scope,
                # "frequency": weight_compression.pruning_frequency,
            }
        return config

    @staticmethod
    def _create_distillation_config(config: Union[Dict, DistillationConfig]):
        if isinstance(config, DistillationConfig):
            criterion = getattr(config.criterion, "config", config.criterion)
            criterion = next(iter(criterion.values()))
            config = {
                "teacher_model_name_or_path": config.teacher_model.config._name_or_path,
                "temperature": criterion.temperature,
                # "loss_types": criterion.loss_types,
                # "loss_weights": criterion.loss_weights,
            }
        return config
