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
import warnings
from collections import UserDict

import torch
from neural_compressor.utils.pytorch import load
from torch.utils.data import DataLoader

from ..utils.constant import WEIGHTS_NAME


logger = logging.getLogger(__name__)


CONFIG_NAME = "best_configure.yaml"
QUANTIZATION_CONFIG_NAME = "quantize_config.json"

IPEX_MINIMUM_VERSION = "2.4.0"
NEURAL_COMPRESSOR_MINIMUM_VERSION = "2.1.0"
NEURAL_COMPRESSOR_WEIGHT_ONLY_MINIMUM_VERSION = "2.3.0"

_HEAD_TO_AUTOMODELS = {
    "fill-mask": "INCModelForMaskedLM",
    "text-generation": "INCModelForCausalLM",
    "text2text-generation": "INCModelForSeq2SeqLM",
    "text-classification": "INCModelForSequenceClassification",
    "token-classification": "INCModelForTokenClassification",
    "question-answering": "INCModelForQuestionAnswering",
    "multiple-choice": "INCModelForMultipleChoice",
    "stable-diffusion": "INCStableDiffusionPipeline",
    "feature-extraction": "INCModel",
}


class INCDataLoader(DataLoader):
    use_label = True

    @classmethod
    def from_pytorch_dataloader(cls, dataloader: DataLoader, use_label: bool = True):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected a PyTorch DataLoader, got: {type(dataloader)}.")
        inc_dataloader = cls(dataloader.dataset)
        cls.use_label = use_label
        for key, value in dataloader.__dict__.items():
            inc_dataloader.__dict__[key] = value
        return inc_dataloader

    def __iter__(self):
        for input in super().__iter__():
            if not isinstance(input, (dict, tuple, list, UserDict)):
                raise TypeError(f"Model calibration cannot use input of type {type(input)}.")
            label = input.get("labels") if isinstance(input, dict) else None
            if self.use_label:
                yield input, label
            else:
                yield input


def load_quantized_model(checkpoint_dir_or_file: str, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """
    Returns the quantized model, which was quantized through neural_compressor.

    Arguments:
        checkpoint_dir_or_file (`str`):
            The path to the model checkpoint containing the quantization information.
        model (`torch.nn.Module`):
            The original FP32 model.
    """
    warnings.warn("This function has been depreciated and will be removed in optimum-intel v1.9.")
    if os.path.isdir(checkpoint_dir_or_file):
        checkpoint_dir_or_file = os.path.join(
            os.path.abspath(os.path.expanduser(checkpoint_dir_or_file)), WEIGHTS_NAME
        )

    return load(checkpoint_dir_or_file, model, **kwargs)
