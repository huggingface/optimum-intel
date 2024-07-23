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
from dataclasses import dataclass
from enum import Enum
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


class INCQuantizationMethod(str, Enum):
    RTN = "rtn"
    GPTQ = "gptq"
    AutoRound = "autoround"


@dataclass
class INCQuantizationConfigBase(QuantizationConfigMixin):
    """
    Base configuration class for quantization parameters
    """

    quant_method = INCQuantizationMethod.RTN

    def __init__(
        self,
        bits: int = 4,
        sym: bool = False,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            bits (`int`, defaults to 4):
                The number of bits to quantize to.
            sym (`bool`, defaults to `False`):
                Whether to use symmetric quantization.
            num_samples (`int`, *optional*):
                The maximum number of samples composing the calibration dataset.
            batch_size (`int`, *optional*):
                The batch size number of calibration dataset.
            seq_len (`int`, *optional*):
                The sequence length number of calibration dataset.
        """
        self.bits = bits
        self.sym = sym
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seq_len = seq_len


@dataclass
class INCWeightQuantizationConfig(INCQuantizationConfigBase):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum-intel` api for weight-only quantization with neural compressor.
    Args:
        bits (`int`, defaults to 4):
            The number of bits to quantize to.
        sym (`bool`, defaults to `False`):
            Whether to use symmetric quantization on the weights.
        num_samples (`int`, *optional*):
            The maximum number of samples composing the calibration dataset.
        batch_size (`int`, *optional*):
            The batch size number of calibration dataset.
        seq_len (`int`, *optional*):
            The sequence length number of calibration dataset.
        tokenizer (`str`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`str or List[str]`, *optional*):
            The dataset used for weight-only quantization with neural-compressor, the default is "NeelNanda/pile-10k".
        group_size (`int`, *optional*):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        use_layer_wise(`bool`, *optional*):
            Enable quantize model per layer. `model_name_or_path` parameter is necessary when used `use_layer_wise`, it indicates the model
            local folder name or the name on remote model hub.
        quant_method (`str or INCQuantizationMethod`, defaults of INCQuantizationMethod.RTN):
            Weight compression method to apply. Possible options:
                - "rtn": weight quantization based round to nearest will be applied.
                - "gptq": A new one-shot weight quantization method based on approximate second-order information,
                    that is both highly-accurate and highly efficient. The weights of each column are updated based on
                    the fixed-scale pseudo-quantization error and the inverse of the Hessian matrix calculated from the activations.
                    The updated columns sharing the same scale may generate a new max/min value,
                    so the scale needs to be saved for restoration.
                - "autoround": AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference.
                    It's tailored for a wide range of models and consistently delivers noticeable improvements,
                    often significantly outperforming SignRound with the cost of more tuning time for quantization.
    """

    def __init__(
        self,
        bits: int = 4,
        sym: bool = False,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        tokenizer: Optional[str] = None,
        dataset: Optional[Union[str]] = None,
        group_size: Optional[int] = None,
        scale_dtype: Optional[str] = None,
        compute_dtype: Optional[str] = None,
        use_layer_wise: Optional[bool] = False,
        quant_method: Union[str, INCQuantizationMethod] = INCQuantizationMethod.RTN,
        **kwargs,
    ):
        super().__init__(bits=bits, sym=sym, num_samples=num_samples, batch_size=batch_size, seq_len=seq_len)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.quant_method = INCQuantizationMethod(quant_method) if isinstance(quant_method, str) else quant_method
        self.group_size = group_size or (-1 if bits == 8 else 128)
        self.scale_dtype = scale_dtype
        self.compute_dtype = compute_dtype
        self.use_layer_wise = use_layer_wise

        # GPTQ parameters
        if self.quant_method == INCQuantizationMethod.GPTQ:
            self.desc_act = kwargs.get("desc_act", False)
            self.damp_percent = kwargs.get("damp_percent", 0.01)
            self.block_size = kwargs.get("block_size ", 128)
            self.static_groups = kwargs.get("static_groups", False)
            self.true_sequential = kwargs.get("true_sequential", False)
        # AutoRound parameters
        if self.quant_method == INCQuantizationMethod.AutoRound:
            self.lr_scheduler = kwargs.get("lr_scheduler", None)
            self.lr = kwargs.get("lr", 0)
            self.minmax_lr = kwargs.get("minmax_lr", None)
            self.enable_quanted_input = kwargs.get("enable_quanted_input", True)
            self.enable_minmax_tuning = kwargs.get("enable_minmax_tuning", True)
            self.quant_lm_head = kwargs.get("quant_lm_head", True)
        # Layerwise
        if self.use_layer_wise:
            self.model_name_or_path = kwargs.get("model_name_or_path", None)
            if self.model_name_or_path is None:
                raise ValueError("model_name_or_path is necessary for layer wise quantization.")
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.bits is not None and self.bits not in [4, 8]:
            raise ValueError(f"Only support quantization to [4,8] bits but found {self.bits}")
        elif self.bits is None:
            self.bits = 4

        if self.quant_method != INCQuantizationMethod.RTN:
            self.batch_size = 8 if self.batch_size is None else self.batch_size
            self.num_samples = 200 if self.num_samples is None else self.num_samples
            self.seq_len = 1024 if self.seq_len is None else self.seq_len
            self.dataset = "NeelNanda/pile-10k" if self.dataset is None else self.dataset

        if self.quant_method == INCQuantizationMethod.GPTQ:
            self.post_init_gptq()

    def post_init_cpu(self):
        if self.compute_dtype is not None and self.compute_dtype not in [
            "fp32",
            "bf16",
            "fp16",
            "int8",
        ]:
            raise ValueError("compute_dtype must be 'fp32', 'bf16', 'fp16', 'int8'.")
        elif self.compute_dtype is None:
            self.compute_dtype = "fp32"

        if self.scale_dtype is not None and self.scale_dtype not in [
            "fp32",
            "bf16",
            "fp16",
        ]:
            raise ValueError("scale_dtype must be 'fp32', 'bf16', 'fp16'.")
        elif self.scale_dtype is None:
            self.scale_dtype = "fp32"

    def post_init_xpu(self):
        if self.compute_dtype is not None and self.compute_dtype not in ["fp16"]:
            raise ValueError("compute_dtype must be 'fp16'.")
        elif self.compute_dtype is None:
            self.compute_dtype = "fp16"

        if self.scale_dtype is not None and self.scale_dtype not in ["fp16"]:
            raise ValueError("scale_dtype must be a string in 'fp16'")
        elif self.scale_dtype is None:
            self.scale_dtype = "fp16"

        if self.bits is not None and self.bits not in [4]:
            raise ValueError(f"Only support quantization to [4] bits but found {self.bits}")
        elif self.bits is None:
            self.bits = 4

        if not self.sym:
            raise ValueError("scheme: asym is not support, only support 'sym' now!")

    def post_init_gptq(self):
        r"""
        Safety checker that arguments are correct for GPTQ.
        """
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")
