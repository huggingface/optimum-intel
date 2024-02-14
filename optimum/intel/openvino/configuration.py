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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PretrainedConfig
from transformers.utils.quantization_config import QuantizationConfigMixin

from optimum.configuration_utils import BaseConfig


DEFAULT_QUANTIZATION_CONFIG = {
    "algorithm": "quantization",
    "preset": "mixed",
    "overflow_fix": "disable",
    "initializer": {
        "range": {"num_init_samples": 300, "type": "mean_min_max"},
        "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
    },
    "scope_overrides": {"activations": {"{re}.*matmul_0": {"mode": "symmetric"}}},
    "ignored_scopes": [
        "{re}.*Embedding.*",
        "{re}.*add___.*",
        "{re}.*layer_norm_.*",
        "{re}.*matmul_1",
        "{re}.*__truediv__.*",
    ],
}

INT8_WEIGHT_COMPRESSION_CONFIG = {
    "algorithm": "quantization",
    "weights": {
        "mode": "symmetric",
        "bits": 8,
        "target_scopes": [
            "{re}.*Embedding.*",
            "{re}.*matmul_.*",
            "{re}.*addmm_.*",
            "{re}.*baddmm_.*",
            "{re}.*linear_.*",
        ],
        "ignored_scopes": [
            "{re}.*conv_*",
        ],
    },
    "activations": {
        "ignored_scopes": [
            "{re}.*add___.*",
            "{re}.*__radd___.*",
            "{re}.*layer_norm_.*",
            "{re}.*__truediv__.*",
            "{re}.*__mul___.*",
            "{re}.*__rmul___.*",
            "{re}.*tanh_.*",
            "{re}.*pow_.*",
            "{re}.*matmul_.*",
            "{re}.*addmm_.*",
            "{re}.*baddmm_.*",
            "{re}.*linear_.*",
            "{re}.*conv_.*",
        ],
    },
    "overflow_fix": "disable",
}


DEFAULT_4BIT_CONFIGS = {
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
}


class OVConfig(BaseConfig):
    CONFIG_NAME = "openvino_config.json"
    FULL_CONFIGURATION_FILE = "openvino_config.json"

    def __init__(
        self,
        compression: Union[List[Dict], Dict, None] = None,
        input_info: Optional[List] = None,
        save_onnx_model: bool = False,
        quantization_config: Optional[QuantizationConfigMixin] = None,
        **kwargs,
    ):
        super().__init__()
        self.compression = compression or DEFAULT_QUANTIZATION_CONFIG
        self.input_info = input_info
        self.save_onnx_model = save_onnx_model
        self._enable_standard_onnx_export_option()
        self.optimum_version = kwargs.pop("optimum_version", None)
        self.quantization_config = quantization_config

    def add_input_info(self, model_inputs: Dict, force_batch_one: bool = False):
        self.input_info = [
            {
                "sample_size": [1] + list(value.shape[1:]) if force_batch_one else list(value.shape),
                "type": "long" if value.dtype is torch.int64 else "float",
                "keyword": name,
            }
            for name, value in model_inputs.items()
        ]

    def save_pretrained(self, *args, **kwargs):
        if self.quantization_config is None:
            self.quantization_config = OVWeightQuantizationConfig()
        super().save_pretrained(*args, **kwargs)

    def _enable_standard_onnx_export_option(self):
        # This method depends on self.save_onnx_model.
        # save_onnx_model is defaulted to false so that the final model output is
        # in OpenVINO IR to realize performance benefit in OpenVINO runtime.
        # True value of save_onnx_model will save a model in onnx format.
        if (
            isinstance(self.compression, dict)
            and "algorithm" in self.compression
            and self.compression["algorithm"] == "quantization"
        ):
            self.compression["export_to_onnx_standard_ops"] = self.save_onnx_model
        elif isinstance(self.compression, list):
            for i, algo_config in enumerate(self.compression):
                if algo_config["algorithm"] == "quantization":
                    self.compression[i]["export_to_onnx_standard_ops"] = self.save_onnx_model


@dataclass
class OVWeightQuantizationConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum-intel` api for quantization with NNCF.

    Args:

        bits (`int`, defaults to 8):
            The number of bits to quantize to.
        sym (`bool`, *optional*, defaults to `False`):
            Whether to use symetric quantization.
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[List[str]]`, *optional*):
            The dataset used for data-aware compression. You can provide your own dataset in a list of string or just use the
            the one from the list ['wikitext2','c4','c4-new','ptb','ptb-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        ratio (`float`, *optional*, defaults to 1.0):
            The ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to INT4_ASYM
            and the rest to INT8_ASYM).
        all_layers (`bool`, *optional*):
            Defines how many layers are compressed to 4-bits while the rest are kept in 8-bit presicion.
        sensitivity_metric (`nncf.SensitivityMetric`, *optional*):
            The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        awq (`bool`, *optional*):
            Enables AWQ method to unify weight ranges and improve overall model accuracy.
        ignored_scope (`nncf.IgnoredScope`, *optional*):
            An ignored scope that defined the list of model control flow graph nodes to be ignored during quantization.

    """

    def __init__(
        self,
        bits: int = 8,
        sym: bool = False,
        tokenizer: Any = None,
        dataset: Optional[str] = None,
        ratio: Optional[float] = None,
        group_size: Optional[int] = None,
        all_layers: Optional[bool] = None,
        sensitivity_metric: Optional[str] = None,
        ignored_scope: Optional[dict] = None,
        **kwargs,
    ):
        self.bits = bits
        self.sym = sym
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.ratio = ratio
        self.all_layers = all_layers
        self.sensitivity_metric = sensitivity_metric
        self.ignored_scope = ignored_scope
        self.quant_method = "default"  # TODO : enable AWQ after nncf v2.9.0 release
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.ratio is not None and not (0 <= self.ratio <= 1):
            raise ValueError("damp_percent must between 0 and 1.")
        if self.group_size is not None and self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if self.dataset is not None and isinstance(self.dataset, str):
            if self.dataset not in ["wikitext2", "c4", "c4-new", "ptb", "ptb-new"]:
                raise ValueError(
                    f"""You have entered a string value for dataset. You can only choose between
                    ['wikitext2','c4','c4-new','ptb','ptb-new'], but we found {self.dataset}"""
                )

        if self.bits not in [4, 8]:
            raise ValueError(f"Only support quantization to [4,8] bits but found {self.bits}")


def _check_default_4bit_configs(config: PretrainedConfig):
    return DEFAULT_4BIT_CONFIGS.get(config.name_or_path, None)
