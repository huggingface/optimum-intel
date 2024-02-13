#  Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Any, Dict, Optional, Union

import nncf
from transformers import AutoTokenizer, PretrainedConfig
from transformers.utils.quantization_config import QuantizationConfigMixin


@dataclass
class OVWeightQuantizationConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum-intel` api for quantization with NNCF.

    Args:
        mode (`nncf.CompressWeightsMode`, *optional*, defaults to INT8_ASYM):
            The model defines the weight compressoin method (4-bit, 8-bit, etc.) available in nncf.compress_weights nncf.CompressWeightsMode.
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
        mode=None,
        tokenizer: Any = None,
        dataset: Optional[Union[nncf.Dataset, str]] = None,
        ratio: Optional[float] = None,
        group_size: Optional[int] = None,
        all_layers: Optional[bool] = None,
        sensitivity_metric: Optional[nncf.SensitivityMetric] = None,
        awq: Optional[bool] = None,
        ignored_scope: Optional[nncf.IgnoredScope] = None,
        **kwargs,
    ):
        self.mode = mode
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.ratio = ratio
        self.ignored_scope = ignored_scope
        self.all_layers = all_layers
        self.sensitivity_metric = sensitivity_metric
        self.awq = awq
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


DEFAULT_4BIT_CONFIGS = {
    "databricks/dolly-v2-3b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 32, "ratio": 0.5},
    "EleutherAI/gpt-j-6b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64},
    "facebook/opt-6.7b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64, "ratio": 0.8},
    "bigscience/bloomz-7b1": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 32, "ratio": 0.6},
    "togethercomputer/RedPajama-INCITE-7B-Instruct": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 128},
    "HuggingFaceH4/zephyr-7b-beta": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.6},
    "meta-llama/Llama-2-7b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.6},
    "meta-llama/Llama-2-7b-chat": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.8},
    "meta-llama/Llama-2-13b-chat": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "stabilityai/stablelm-3b-4e1t": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "stablelm-epoch-3b-preview": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "stable-zephyr-3b-dpo": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64, "ratio": 0.8},
    "pansophic/rocket-3B": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.8},
    "THUDM/chatglm2-6b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.72},
    "Qwen/Qwen-7B-Chat": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.6},
    "openlm-research/open_llama_3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers"=True},
    "tiiuae/falcon-7b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers"=True},
    "psmathur/orca_mini_3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers"=True},
}


def _check_default_4bit_configs(config: PretrainedConfig):
    return DEFAULT_4BIT_CONFIGS.get(config.name_or_path, None)


def compress_decoder_weights(model, quantization_config: Union[OVWeightQuantizationConfig, Dict] = None):
    quantization_config = (
        quantization_config if quantization_config is not None else _check_default_4bit_configs(model.config)
    )
    ov_model = model.model

    if quantization_config is not None:
        config = quantization_config
        if isinstance(config, Dict):
            config = OVWeightQuantizationConfig.from_dict(quantization_config)

        dataset = config.dataset
        if config.dataset is not None and isinstance(config.dataset, str):
            tokenizer = config.tokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
            elif isinstance(tokenizer, str):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer)

            from optimum.gptq.data import get_dataset, prepare_dataset

            dataset = get_dataset(config.dataset, tokenizer, seqlen=32)
            dataset = prepare_dataset(dataset)
            dataset = nncf.Dataset(dataset, lambda x: model.prepare_inputs(**x))

        model.model = nncf.compress_weights(
            ov_model,
            mode=config.mode,
            ratio=config.ratio,
            group_size=config.group_size,
            all_layers=config.all_layers,
            sensitivity_metric=config.sensitivity_metric,
            awq=config.awq,
            ignored_scope=config.ignored_scope,
            dataset=dataset,
        )
    else:  # Data-free weight-only quantization to asymmetric INT4
        model.model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT4_ASYM)
