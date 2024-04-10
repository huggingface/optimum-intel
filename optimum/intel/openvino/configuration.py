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
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import nncf
import torch
from nncf.quantization.advanced_parameters import OverflowFix
from transformers import PretrainedConfig
from transformers.utils.quantization_config import QuantizationConfigMixin, QuantizationMethod

from optimum.configuration_utils import BaseConfig


logger = logging.getLogger(__name__)

_DEFAULT_4BIT_CONFIGS = {
    "databricks/dolly-v2-3b": {"bits": 4, "sym": False, "group_size": 32, "ratio": 0.5},
    "EleutherAI/gpt-j-6b": {"bits": 4, "sym": False, "group_size": 64},
    "facebook/opt-6.7b": {"bits": 4, "sym": False, "group_size": 64, "ratio": 0.8},
    "bigscience/bloomz-7b1": {"bits": 4, "sym": False, "group_size": 32, "ratio": 0.6},
    "togethercomputer/RedPajama-INCITE-7B-Instruct": {"bits": 4, "sym": False, "group_size": 128},
    "HuggingFaceH4/zephyr-7b-beta": {"bits": 4, "sym": True, "group_size": 64, "ratio": 0.6},
    "meta-llama/Llama-2-7b": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.6},
    "meta-llama/Llama-2-7b-chat": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.8},
    "meta-llama/Llama-2-13b-chat": {"bits": 4, "sym": True, "group_size": 64, "ratio": 0.8},
    "stabilityai/stablelm-3b-4e1t": {"bits": 4, "sym": True, "group_size": 64, "ratio": 0.8},
    "stablelm-epoch-3b-preview": {"bits": 4, "sym": True, "group_size": 64, "ratio": 0.8},
    "stable-zephyr-3b-dpo": {"bits": 4, "sym": False, "group_size": 64, "ratio": 0.8},
    "pansophic/rocket-3B": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.8},
    "THUDM/chatglm2-6b": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.72},
    "Qwen/Qwen-7B-Chat": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.6},
    "openlm-research/open_llama_3b": {"bits": 4, "sym": True, "group_size": 64, "all_layers": True},
    "tiiuae/falcon-7b": {"bits": 4, "sym": True, "group_size": 64, "all_layers": True},
    "psmathur/orca_mini_3b": {"bits": 4, "sym": True, "group_size": 64, "all_layers": True},
    "mistralai/Mixtral-8x7B-v0.1": {"bits": 4, "sym": True, "group_size": 128, "ratio": 0.8},
}


class _replace_properties_values:
    """
    A context manager for temporarily overriding an object's properties
    """

    def __init__(self, obj, property_names, property_values):
        self.obj = obj
        self.property_names = property_names
        self.new_property_values = property_values
        self.old_property_values = [None] * len(property_names)
        for i, property_name in enumerate(self.property_names):
            self.old_property_values[i] = getattr(obj, property_name)

    def __enter__(self):
        for property_name, new_property_value in zip(self.property_names, self.new_property_values):
            setattr(self.obj, property_name, new_property_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for property_name, old_property_value in zip(self.property_names, self.old_property_values):
            setattr(self.obj, property_name, old_property_value)


def _is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False


@dataclass
class OVQuantizationConfigBase(QuantizationConfigMixin):
    """
    Base configuration class for quantization parameters
    """

    def __init__(
        self,
        dataset: Optional[Union[str, List[str], nncf.Dataset, datasets.Dataset]] = None,
        ignored_scope: Optional[Union[dict, nncf.IgnoredScope]] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset (`str or List[str] or nncf.Dataset or datasets.Dataset`, *optional*):
                 The dataset used for data-aware weight compression or quantization with NNCF.
            ignored_scope (`dict or nncf.IgnoredScope`, *optional*):
                An ignored scope that defines the list of model nodes to be ignored during quantization.
            num_samples (`int`, *optional*):
                The maximum number of samples composing the calibration dataset.
        """
        self.dataset = dataset
        if isinstance(ignored_scope, dict):
            ignored_scope = nncf.IgnoredScope(**ignored_scope)
        self.ignored_scope = ignored_scope
        self.num_samples = num_samples

    def post_init(self):
        if not (self.dataset is None or isinstance(self.dataset, (str, list, nncf.Dataset, datasets.Dataset))):
            raise ValueError(
                "Dataset must be a instance of either string, list of strings, nncf.Dataset or "
                f"dataset.Dataset, but found {type(self.dataset)}"
            )
        if not (self.ignored_scope is None or isinstance(self.ignored_scope, nncf.IgnoredScope)):
            raise ValueError(
                "Ignored scope must be a instance of either dict, or nncf.IgnoredScope but found "
                f"{type(self.dataset)}"
            )

    def _to_dict_without_properties(self, property_names: Union[List[str], Tuple[str]]) -> Dict[str, Any]:
        """
        Calls to_dict() with given properties overwritten with None. Useful for hiding non-serializable properties.
        """
        if len(property_names) == 0:
            return super().to_dict()
        with _replace_properties_values(self, property_names, [None] * len(property_names)):
            result = super().to_dict()
        return result

    def to_dict(self) -> Dict[str, Any]:
        properties_to_omit = [] if _is_serializable(self.dataset) else ["dataset"]
        if isinstance(self.ignored_scope, nncf.IgnoredScope):
            with _replace_properties_values(self, ["ignored_scope"], [self.ignored_scope.__dict__]):
                return self._to_dict_without_properties(properties_to_omit)
        return self._to_dict_without_properties(properties_to_omit)


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
        self.quantization_config = quantization_config
        self.compression = None  # A backward-compatability field for training-time compression parameters

        if isinstance(self.quantization_config, dict):
            # Config is loaded as dict during deserialization
            logger.info(
                "`quantization_config` was provided as a dict, in this form it can't be used for quantization. "
                "Please provide config as an instance of OVWeightQuantizationConfig or OVQuantizationConfig"
            )

        bits = (
            self.quantization_config.bits if isinstance(self.quantization_config, OVWeightQuantizationConfig) else None
        )
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

    def _to_dict_safe(self, to_diff_dict: bool = False) -> Dict[str, Any]:
        if self.quantization_config is None:
            # Parent to_dict() implementation does not support quantization_config being None
            with _replace_properties_values(self, ("quantization_config",), (OVQuantizationConfigBase(),)):
                result = super().to_diff_dict() if to_diff_dict else super().to_dict()
                del result["quantization_config"]
        else:
            result = super().to_diff_dict() if to_diff_dict else super().to_dict()
        return result

    def to_dict(self) -> Dict[str, Any]:
        return self._to_dict_safe(to_diff_dict=False)

    def to_diff_dict(self) -> Dict[str, Any]:
        return self._to_dict_safe(to_diff_dict=True)


class OVQuantizationMethod(str, Enum):
    DEFAULT = "default"


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
            Whether to use symmetric quantization.
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`str or List[str]`, *optional*):
            The dataset used for data-aware compression or quantization with NNCF. You can provide your own dataset
            in a list of strings or just use the one from the list ['wikitext','c4','c4-new','ptb','ptb-new'] for LLLMs
            or ['conceptual_captions','laion/220k-GPT4Vision-captions-from-LIVIS','laion/filtered-wit'] for diffusion models.
        ratio (`float`, defaults to 1.0):
            The ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to INT4_ASYM
            and the rest to INT8_ASYM).
        group_size (`int`, *optional*):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        all_layers (`bool`, *optional*):
            Defines how many layers are compressed to 4-bits while the rest are kept in 8-bit precision.
        sensitivity_metric (`str`, *optional*):
            The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        ignored_scope (`dict`, *optional*):
            An ignored scope that defined the list of model control flow graph nodes to be ignored during quantization.
        num_samples (`int`, *optional*):
            The maximum number of samples composing the calibration dataset.
        quant_method (`str`, defaults of OVQuantizationMethod.DEFAULT):
            Weight compression method to apply.
    """

    def __init__(
        self,
        bits: int = 8,
        sym: bool = False,
        tokenizer: Optional[Any] = None,
        dataset: Optional[Union[str, List[str], nncf.Dataset, datasets.Dataset]] = None,
        ratio: float = 1.0,
        group_size: Optional[int] = None,
        all_layers: Optional[bool] = None,
        sensitivity_metric: Optional[str] = None,
        ignored_scope: Optional[Union[dict, nncf.IgnoredScope]] = None,
        num_samples: Optional[int] = None,
        quant_method: Optional[Union[QuantizationMethod, OVQuantizationMethod]] = OVQuantizationMethod.DEFAULT,
        **kwargs,
    ):
        super().__init__(dataset, ignored_scope, num_samples)
        self.bits = bits
        self.sym = sym
        self.tokenizer = tokenizer
        self.group_size = group_size or (-1 if bits == 8 else 128)
        self.ratio = ratio
        self.all_layers = all_layers
        self.sensitivity_metric = sensitivity_metric
        self.quant_method = quant_method
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
        if self.dataset is not None and isinstance(self.dataset, str):
            llm_datasets = ["wikitext", "c4", "c4-new", "ptb", "ptb-new"]
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

    def to_dict(self) -> Dict[str, Any]:
        if not _is_serializable(self.tokenizer):
            return self._to_dict_without_properties(("tokenizer",))
        return super().to_dict()


@dataclass
class OVQuantizationConfig(OVQuantizationConfigBase):
    def __init__(
        self,
        dataset: Union[str, List[str], nncf.Dataset, datasets.Dataset],
        ignored_scope: Optional[Union[dict, nncf.IgnoredScope]] = None,
        num_samples: Optional[int] = 300,
        preset: nncf.QuantizationPreset = None,
        model_type: nncf.ModelType = nncf.ModelType.TRANSFORMER,
        fast_bias_correction: bool = True,
        overflow_fix: OverflowFix = OverflowFix.DISABLE,
        **kwargs,
    ):
        """
        Configuration class containing parameters related to model quantization with NNCF. Compared to weight
        compression, during quantization both weights and activations are converted to lower precision.
        For weight-only model quantization please see OVWeightQuantizationConfig.
        Args:
            dataset (`str or List[str] or nncf.Dataset or datasets.Dataset`):
                 A dataset used for quantization parameters calibration. Required parameter.
            ignored_scope (`dict or nncf.IgnoredScope`, *optional*):
                An ignored scope that defines the list of model nodes to be ignored during quantization.
            num_samples (`int`, *optional*):
                The maximum number of samples composing the calibration dataset.
            preset (`nncf.QuantizationPreset`, *optional*):
                A preset controls the quantization mode (symmetric and asymmetric).
                It can take the following values:
                - `performance`: Symmetric quantization of weights and activations.
                - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
                Default value is None. In this case, `mixed` preset is used for `transformer`
                model type otherwise `performance`.
            model_type (`nncf.ModelType`, defaults to nncf.ModelType.TRANSFORMER):
                Model type is needed to specify additional patterns in the model. Supported only `transformer` now.
            fast_bias_correction (`bool`, defaults to True):
                Whether to apply fast or full bias correction algorithm.
            overflow_fix (`nncf.OverflowFix`, default to OverflowFix.DISABLE):
                Parameter for controlling overflow fix setting.
        """
        super().__init__(dataset, ignored_scope, num_samples)
        self.preset = preset
        self.model_type = model_type
        self.fast_bias_correction = fast_bias_correction
        self.overflow_fix = overflow_fix
        self.post_init()

    def post_init(self):
        """
        Safety checker that arguments are correct
        """
        super().post_init()
        if self.dataset is None:
            raise ValueError(
                "`dataset` is needed to compute the activations range during the calibration step and was not provided."
                " In case you only want to apply quantization on the weights, please run weight-only quantization."
            )

    def to_dict(self) -> Dict[str, Any]:
        # TODO: remove code below once NNCF is updated to 2.10
        overflow_fix_value = None if self.overflow_fix is None else self.overflow_fix.value
        preset_value = None if self.preset is None else self.preset.value
        with _replace_properties_values(self, ("overflow_fix", "preset"), (overflow_fix_value, preset_value)):
            return super().to_dict()


def _check_default_4bit_configs(config: PretrainedConfig):
    return _DEFAULT_4BIT_CONFIGS.get(config.name_or_path, None)
