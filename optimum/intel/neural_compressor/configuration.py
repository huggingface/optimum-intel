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

from typing import Dict, Optional, Union

from neural_compressor.config import DistillationConfig, WeightPruningConfig, _BaseQuantizationConfig
from transformers.utils.quantization_config import QuantizationConfigMixin

from optimum.configuration_utils import BaseConfig

from ..utils.import_utils import _neural_compressor_version, _torch_version


_quantization_model = {
    "post_training_dynamic_quant": "dynamic",
    "post_training_static_quant": "static",
    "quant_aware_training": "aware_training",
    "post_training_weight_only": "weight_only",
}


class INCConfig(BaseConfig):
    CONFIG_NAME = "inc_config.json"
    FULL_CONFIGURATION_FILE = "inc_config.json"

    def __init__(
        self,
        quantization: Optional[Union[Dict, _BaseQuantizationConfig]] = None,
        pruning: Optional[Union[Dict, _BaseQuantizationConfig]] = None,
        distillation: Optional[Union[Dict, _BaseQuantizationConfig]] = None,
        save_onnx_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.torch_version = _torch_version
        self.neural_compressor_version = _neural_compressor_version
        self.quantization = self._create_quantization_config(quantization) or {}
        self.pruning = self._create_pruning_config(pruning) or {}
        self.distillation = self._create_distillation_config(distillation) or {}
        self.save_onnx_model = save_onnx_model

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


@dataclass
class INCWeightQuantizationConfig(QuantizationConfigMixin):
    """
    Args:
        bits (`int`, defaults to 8):
            The number of bits to quantize to.
        sym (`bool`, defaults to `False`):
            Whether to use symetric quantization.
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        group_size (`int`, *optional*):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        num_samples (`int`, *optional*):
            The maximum number of samples composing the calibration dataset.
        damp_percent (`float`, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.

    """

    def __init__(
        self,
        bits: int = 8,
        sym: bool = False,
        tokenizer: Optional[Any] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        group_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        damp_percent: float = 0.1,
        quant_method: str = "default",
        **kwargs,
    ):
        self.bits = bits
        self.sym = sym
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size or (-1 if bits == 8 else 128)
        self.num_samples = num_samples
        self.quant_method = quant_method
        self.damp_percent = damp_percent if self.quant_method == "gptq" else None

        self.post_init()

    def post_init(self):
        if self.group_size is not None and self.group_size != -1 and self.group_size <= 0:
            raise ValueError(f"`group_size` must be greater than 0 or equal to -1 got {self.group_size}")

        if self.bits not in [4, 8]:
            raise ValueError(f"Only support quantization to [4,8] bits but found {self.bits}")

        supported_methodologies = {"rtn", "gptq", "awq", "autoaround"}
        if self.quant_method not in supported_methodologies:
            raise ValueError(
                f"Unsupported quantization methodology {self.quant_method}, only {supported_methodologies} are currently supported"
            )
