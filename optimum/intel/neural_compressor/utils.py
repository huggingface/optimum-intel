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
from collections import UserDict
from typing import Dict

import torch
from packaging import version
from torch.utils.data import DataLoader

from neural_compressor.utils.pytorch import load


logger = logging.getLogger(__name__)


CONFIG_NAME = "best_configure.yaml"
WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"
ONNX_WEIGHTS_NAME = "model.onnx"
MIN_QDQ_ONNX_OPSET = 14

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_less_than_1_13 = parsed_torch_version_base < version.parse("1.13.0")


class INCDataLoader(DataLoader):
    @classmethod
    def from_pytorch_dataloader(cls, dataloader: DataLoader):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected a PyTorch DataLoader, got: {type(dataloader)}.")
        inc_dataloader = cls(dataloader.dataset)
        for key, value in dataloader.__dict__.items():
            inc_dataloader.__dict__[key] = value
        return inc_dataloader

    def __iter__(self):
        for input in super().__iter__():
            if not isinstance(input, (dict, tuple, list, UserDict)):
                raise TypeError(f"Model calibration cannot use input of type {type(input)}.")
            label = input.get("labels") if isinstance(input, dict) else None
            yield input, label


def _cfgs_to_fx_cfgs(op_cfgs: Dict, observer_type: str = "post_training_static_quant") -> Dict:
    """Inc function which convert a quantization config to a format that meets the requirements of torch.fx.

    Arguments:
        op_cfgs (`dict`):
            Dictionary of quantization configure for each op.
        observer_type (`str`):
            Specify observer type.
    Returns:
        fx_op_cfgs (`dict`):
            Dictionary of quantization configure that meets the requirements of torch.fx.
    """
    if not is_torch_less_than_1_13:
        from torch.ao.quantization import QConfigMapping

        fx_op_cfgs = QConfigMapping()
    else:
        fx_op_cfgs = dict()
        op_tuple_cfg_list = []
    for key, value in op_cfgs.items():
        if key == "default_qconfig":
            if not is_torch_less_than_1_13:
                fx_op_cfgs.set_global(value)
            else:
                fx_op_cfgs[""] = value
            continue
        if not is_torch_less_than_1_13:
            fx_op_cfgs.set_module_name(key, value)
        else:
            op_tuple = (key, value)
            op_tuple_cfg_list.append(op_tuple)

    if is_torch_less_than_1_13:
        fx_op_cfgs["module_name"] = op_tuple_cfg_list

    return fx_op_cfgs


def load_quantized_model(checkpoint_dir_or_file: str, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """
    Returns the quantized model, which was quantized through neural_compressor.

    Arguments:
        checkpoint_dir_or_file (`str`):
            The path to the model checkpoint containing the quantization information.
        model (`torch.nn.Module`):
            The original FP32 model.
    """
    if os.path.isdir(checkpoint_dir_or_file):
        checkpoint_dir_or_file = os.path.join(
            os.path.abspath(os.path.expanduser(checkpoint_dir_or_file)), WEIGHTS_NAME
        )

    return load(checkpoint_dir_or_file, model, **kwargs)
