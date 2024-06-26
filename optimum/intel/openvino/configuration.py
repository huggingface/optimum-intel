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
import copy
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PretrainedConfig
from transformers.utils.quantization_config import QuantizationConfigMixin

from optimum.configuration_utils import BaseConfig

from ..utils.import_utils import is_nncf_available


if is_nncf_available():
    import nncf

logger = logging.getLogger(__name__)

_DEFAULT_4BIT_CONFIGS = {
    "databricks/dolly-v2-3b": {"bits": 4, "sym": False, "group_size": 128, "ratio": 0.8},
    "EleutherAI/gpt-j-6b": {"bits": 4, "sym": False, "group_size": 64},
    "facebook/opt-6.7b": {"bits": 4, "sym": False, "group_size": 64, "ratio": 0.8},
    "bigscience/bloomz-7b1": {"bits": 4, "sym": False, "group_size": 32, "ratio": 0.6},
    "togethercomputer/RedPajama-INCITE-7B-Instruct": {"bits": 4, "sym": False, "group_size": 128},
    "HuggingFaceH4/zephyr-7b-beta": {
        "bits": 4,
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
        "dataset": "wikitext2",
        "awq": True,
    },
    "meta-llama/Llama-2-7b": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.7},
    "meta-llama/Llama-2-7b-chat": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.8},
    "meta-llama/Llama-2-13b-chat": {"bits": 4, "sym": True, "group_size": 64, "ratio": 0.8},
    "stabilityai/stablelm-3b-4e1t": {"bits": 4, "sym": True, "group_size": 64, "ratio": 0.8},
    "stablelm-epoch-3b-preview": {"bits": 4, "sym": True, "group_size": 64, "ratio": 0.8},
    "stabilityai/stablelm-zephyr-3b": {"bits": 4, "sym": False, "group_size": 128, "ratio": 1.0},
    "pansophic/rocket-3B": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.8},
    "THUDM/chatglm2-6b": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.72},
    "Qwen/Qwen-7B-Chat": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.6},
    "openlm-research/open_llama_3b": {"bits": 4, "sym": True, "group_size": 64, "all_layers": True},
    "tiiuae/falcon-7b": {"bits": 4, "sym": True, "group_size": 64, "all_layers": True},
    "psmathur/orca_mini_3b": {"bits": 4, "sym": True, "group_size": 64, "all_layers": True},
    "mistralai/Mixtral-8x7B-v0.1": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.8},
    "facebook/opt-2.7b": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.7},
    "togethercomputer/RedPajama-INCITE-Chat-3B-v1": {"bits": 4, "sym": False, "group_size": 128, "ratio": 0.8},
    "lmsys/vicuna-7b-v1.5": {"bits": 4, "sym": False, "group_size": 128, "ratio": 1.0},
    "stabilityai/stablelm-tuned-alpha-3b": {"bits": 4, "sym": False, "group_size": 128, "ratio": 0.8},
    "mistralai/Mistral-7B-v0.1": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.9},
    "baichuan-inc/Baichuan2-7B-Chat": {
        "bits": 4,
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
        "dataset": "wikitext2",
        "awq": True,
    },
}


class OVQuantizationMethod(str, Enum):
    DEFAULT = "default"
    HYBRID = "hybrid"
    AWQ = "awq"


@dataclass
class OVQuantizationConfigBase(QuantizationConfigMixin):
    """
    Base configuration class for quantization parameters
    """

    quant_method = OVQuantizationMethod.DEFAULT

    def __init__(
        self,
        bits: int = 8,
        sym: bool = False,
        ignored_scope: Optional[dict] = None,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            bits (`int`, defaults to 8):
                The number of bits to quantize to.
            sym (`bool`, defaults to `False`):
                Whether to use symmetric quantization.
            ignored_scope (`dict`, *optional*):
                An ignored scope that defines a list of model nodes to be ignored during quantization. Dictionary
                entries provided via this argument are used to create an instance of `nncf.IgnoredScope` class.
            num_samples (`int`, *optional*):
                The maximum number of samples composing the calibration dataset.
        """
        self.bits = bits
        self.sym = sym
        self.num_samples = num_samples

        if isinstance(ignored_scope, nncf.IgnoredScope):
            ignored_scope = ignored_scope.__dict__
        self.ignored_scope = ignored_scope

    def post_init(self):
        try:
            self.get_ignored_scope_instance()
        except Exception as e:
            raise ValueError(
                f"Can't create an `IgnoredScope` object from the provided ignored scope dict: {self.ignored_scope}.\n{e}"
            )
        if not (self.num_samples is None or isinstance(self.num_samples, int) and self.num_samples > 0):
            raise ValueError(f"`num_samples` is expected to be a positive integer, but found: {self.num_samples}")

    def get_ignored_scope_instance(self) -> "nncf.IgnoredScope":
        if self.ignored_scope is None:
            return nncf.IgnoredScope()
        return nncf.IgnoredScope(**copy.deepcopy(self.ignored_scope))


@dataclass
class OVWeightQuantizationConfig(OVQuantizationConfigBase):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum-intel` api for weight-only quantization with NNCF. For full model quantization please see
    OVQuantizationConfig.
    Args:
        bits (`int`, defaults to 8):
            The number of bits to quantize to.
        sym (`bool`, defaults to `False`):
            Whether to use symmetric quantization on the weights.
        group_size (`int`, *optional*):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        tokenizer (`str`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`str or List[str]`, *optional*):
            The dataset used for data-aware compression or quantization with NNCF. You can provide your own dataset
            in a list of strings or just use the one from the list ['wikitext2','c4','c4-new'] for language models
            or ['conceptual_captions','laion/220k-GPT4Vision-captions-from-LIVIS','laion/filtered-wit'] for diffusion models.
            Alternatively, you can provide data objects via `calibration_dataset` argument
            of `OVQuantizer.quantize()` method.
        ratio (`float`, defaults to 1.0):
            The ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to INT4_ASYM
            and the rest to INT8_ASYM).
        all_layers (`bool`, *optional*):
            Defines how many layers are compressed to 4-bits while the rest are kept in 8-bit precision.
        sensitivity_metric (`str`, *optional*):
            The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        ignored_scope (`dict`, *optional*):
            An ignored scope that defines the list of model nodes to be ignored during quantization. Dictionary
            entries provided via this argument are used to create an instance of `nncf.IgnoredScope` class.
        num_samples (`int`, *optional*):
            The maximum number of samples composing the calibration dataset.
        quant_method (`str or OVQuantizationMethod`, defaults of OVQuantizationMethod.DEFAULT):
            Weight compression method to apply. Possible options:
                - "default": default weight quantization will be applied.
                - "awq": compressed weights will be computed according to the Activation-Aware-Quantization (AWQ)
                  method. AWQ improves generation quality of INT4-compressed LLMs, but requires
                  additional time for tuning weights on a calibration dataset. To run AWQ, providing a dataset is
                  required. Note: it's possible that there will be no matching patterns in the model to apply AWQ, in
                  such case it will be skipped.
                - "hybrid": The hybrid mode involves the quantization of weights in MatMul and Embedding layers, and
                  activations of other layers, facilitating accuracy preservation post-optimization while reducing
                  the model size. Hybrid mode performs well when applied to a UNet model in diffusion pipelines.
        scale_estimation (`bool`, *optional*):
            Indicates whether to apply a scale estimation algorithm that minimizes the L2 error between the original and
            compressed layers. Providing a dataset is required to run scale estimation.
    """

    def __init__(
        self,
        bits: int = 8,
        sym: bool = False,
        group_size: Optional[int] = None,
        tokenizer: Optional[str] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        ratio: float = 1.0,
        all_layers: Optional[bool] = None,
        sensitivity_metric: Optional[str] = None,
        ignored_scope: Optional[dict] = None,
        num_samples: Optional[int] = None,
        quant_method: Union[str, OVQuantizationMethod] = OVQuantizationMethod.DEFAULT,
        scale_estimation: bool = None,
        **kwargs,
    ):
        super().__init__(bits=bits, sym=sym, ignored_scope=ignored_scope, num_samples=num_samples)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size or (-1 if bits == 8 else 128)
        self.ratio = ratio
        self.all_layers = all_layers
        self.sensitivity_metric = sensitivity_metric
        self.quant_method = OVQuantizationMethod(quant_method) if isinstance(quant_method, str) else quant_method
        self.scale_estimation = scale_estimation
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        super().post_init()
        if self.ratio is not None and not (0 <= self.ratio <= 1):
            raise ValueError("`ratio` must between 0 and 1.")
        if self.group_size is not None and self.group_size != -1 and self.group_size <= 0:
            raise ValueError("`group_size` must be greater than 0 or equal to -1")
        if not (self.dataset is None or isinstance(self.dataset, (str, list))):
            raise ValueError(
                f"Dataset must be a instance of either string or list of strings, but found {type(self.dataset)}. "
                f"If you wish to provide a custom dataset, please use the `OVQuantizer` instead."
            )
        if self.dataset is not None and isinstance(self.dataset, str):
            llm_datasets = ["wikitext2", "c4", "c4-new"]
            stable_diffusion_datasets = [
                "conceptual_captions",
                "laion/220k-GPT4Vision-captions-from-LIVIS",
                "laion/filtered-wit",
            ]
            if self.dataset not in llm_datasets + stable_diffusion_datasets:
                raise ValueError(
                    f"""You have entered a string value for dataset. You can only choose between
                    {llm_datasets} for LLLMs or {stable_diffusion_datasets} for diffusion models, but we found {self.dataset}"""
                )

        if self.bits not in [4, 8]:
            raise ValueError(f"Only support quantization to [4,8] bits but found {self.bits}")

        if self.bits == 8:
            if self.ratio != 1:
                raise ValueError(
                    f"For 8-bit quantization, `ratio` is expected to be set to 1.0, but was set to {self.ratio}"
                )
            if self.group_size != -1:
                raise ValueError(
                    f"For 8-bit quantization, `group_size` is expected to be set to -1, but was set to {self.group_size}"
                )

        if self.tokenizer is not None and not isinstance(self.tokenizer, str):
            raise ValueError(f"Tokenizer is expected to be a string, but found {self.tokenizer}")


@dataclass
class OVDynamicQuantizationConfig(OVWeightQuantizationConfig):
    def __init__(
        self,
        bits: int = 8,
        sym: bool = False,
        weights_group_size: Optional[int] = None,
        activations_group_size: int = 32,
        **kwargs,
    ):
        super().__init__(bits=bits, sym=sym, group_size=weights_group_size, **kwargs)
        self.activations_group_size = activations_group_size


@dataclass
class OVQuantizationConfig(OVQuantizationConfigBase):
    def __init__(
        self,
        bits: int = 8,
        sym: bool = False,
        ignored_scope: Optional[dict] = None,
        num_samples: Optional[int] = 300,
        model_type: str = "transformer",
        fast_bias_correction: bool = True,
        overflow_fix: str = "disable",
        **kwargs,
    ):
        """
        Configuration class containing parameters related to model quantization with NNCF. Compared to weight
        compression, during quantization both weights and activations are converted to lower precision.
        For weight-only model quantization please see OVWeightQuantizationConfig.
        Args:
            bits (`int`, defaults to 8):
                The number of bits to quantize to.
            sym (`bool`, defaults to `False`):
                Whether to use symmetric quantization on the activations. Symmetric quantization will be applied on the weights in any case.
            ignored_scope (`dict`, *optional*):
                An ignored scope that defines the list of model nodes to be ignored during quantization. Dictionary
                entries provided via this argument are used to create an instance of `nncf.IgnoredScope` class.
            num_samples (`int`, *optional*):
                The maximum number of samples composing the calibration dataset.
            model_type (`str`, defaults to "transformer"):
                Model type is needed to specify additional patterns in the model. Supported only `transformer` now.
            fast_bias_correction (`bool`, defaults to True):
                Whether to apply fast or full bias correction algorithm.
            overflow_fix (`str`, default to "disable"):
                Parameter for controlling overflow fix setting.
        """
        super().__init__(bits=bits, sym=sym, ignored_scope=ignored_scope, num_samples=num_samples)
        self.model_type = model_type
        self.fast_bias_correction = fast_bias_correction
        self.overflow_fix = overflow_fix
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        super().post_init()

        if self.bits != 8:
            raise ValueError(f"Only support 8-bit for static quantization but found {self.bits}")


def _check_default_4bit_configs(config: PretrainedConfig):
    return _DEFAULT_4BIT_CONFIGS.get(config.name_or_path, None)


class OVConfig(BaseConfig):
    CONFIG_NAME = "openvino_config.json"
    FULL_CONFIGURATION_FILE = "openvino_config.json"

    def __init__(
        self,
        input_info: Optional[List] = None,
        save_onnx_model: bool = False,
        quantization_config: Optional[Union[dict, OVQuantizationConfigBase]] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_info = input_info
        self.save_onnx_model = save_onnx_model
        self.optimum_version = kwargs.pop("optimum_version", None)
        if isinstance(quantization_config, dict):
            quantization_config = self._quantization_config_from_dict(quantization_config)
        self.quantization_config = quantization_config
        self.compression = kwargs.get(
            "compression", None
        )  # A field for backward-compatability of training-time compression parameters
        bits = self.quantization_config.bits if self.quantization_config else None
        self.dtype = "int" + str(bits) if isinstance(bits, int) else dtype

    def add_input_info(self, model_inputs: Dict, force_batch_one: bool = False):
        self.input_info = [
            {
                "sample_size": [1] + list(value.shape[1:]) if force_batch_one else list(value.shape),
                "type": "long" if value.dtype is torch.int64 else "float",
                "keyword": name,
            }
            for name, value in model_inputs.items()
        ]

    @staticmethod
    def _quantization_config_from_dict(quantization_config: dict) -> OVQuantizationConfigBase:
        wq_args = inspect.getfullargspec(OVWeightQuantizationConfig.__init__).args
        q_args = inspect.getfullargspec(OVQuantizationConfig.__init__).args
        weight_only = quantization_config.pop("weight_only", None)
        config_keys = quantization_config.keys()
        matches_wq_config_signature = all(arg_name in wq_args for arg_name in config_keys)
        matches_q_config_signature = all(arg_name in q_args for arg_name in config_keys)
        if matches_wq_config_signature == matches_q_config_signature:
            if weight_only is None:
                logger.warning(
                    "Can't determine type of OV quantization config. Please specify explicitly whether you intend to "
                    "run weight-only quantization or not with `weight_only` parameter. Creating an instance of "
                    "OVWeightQuantizationConfig."
                )
                return OVWeightQuantizationConfig.from_dict(quantization_config)
            matches_wq_config_signature = weight_only

        config_type = OVWeightQuantizationConfig if matches_wq_config_signature else OVQuantizationConfig
        return config_type.from_dict(quantization_config)

    def _to_dict_safe(self, to_diff_dict: bool = False) -> Dict[str, Any]:
        class ConfigStub:
            def to_dict(self):
                return None

            def to_diff_dict(self):
                return None

        if self.quantization_config is None:
            # Parent to_dict() implementation does not support quantization_config being None
            self_copy = copy.deepcopy(self)
            self_copy.quantization_config = ConfigStub()
            result = self_copy.to_diff_dict() if to_diff_dict else self_copy.to_dict()
        else:
            result = super().to_diff_dict() if to_diff_dict else super().to_dict()
        return result

    def to_dict(self) -> Dict[str, Any]:
        return self._to_dict_safe(to_diff_dict=False)

    def to_diff_dict(self) -> Dict[str, Any]:
        return self._to_dict_safe(to_diff_dict=True)
