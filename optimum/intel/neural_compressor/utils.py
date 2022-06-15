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
from collections import UserDict
from typing import Dict, List

from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


CONFIG_NAME = "best_configure.yaml"
WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class IncDataLoader(DataLoader):
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
    Args:
        op_cfgs (`dict`):
            Dictionary of quantization configure for each op.
        observer_type (`str`):
            Specify observer type.
    Returns:
        fx_op_cfgs (`dict`):
            Dictionary of quantization configure that meets the requirements of torch.fx.
    """
    fx_op_cfgs = dict()
    op_tuple_cfg_list = []
    for key, value in op_cfgs.items():
        if key == "default_qconfig":
            fx_op_cfgs[""] = value
            continue
        op_tuple = (key, value)
        op_tuple_cfg_list.append(op_tuple)
    fx_op_cfgs["module_name"] = op_tuple_cfg_list
    return fx_op_cfgs
