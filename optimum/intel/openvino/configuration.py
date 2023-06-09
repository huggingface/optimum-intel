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

from typing import Dict, List, Optional, Union

import torch

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
        "{re}.*Embedding*",
        "{re}.*__add___[0-1]",
        "{re}.*layer_norm_0",
        "{re}.*matmul_1",
        "{re}.*__truediv__*",
    ],
}


class OVConfig(BaseConfig):
    CONFIG_NAME = "openvino_config.json"
    FULL_CONFIGURATION_FILE = "openvino_config.json"

    def __init__(
        self,
        compression: Union[List[Dict], Dict, None] = None,
        input_info: Optional[List] = None,
        save_onnx_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.compression = compression or DEFAULT_QUANTIZATION_CONFIG
        self.input_info = input_info
        self.save_onnx_model = save_onnx_model
        self._enable_standard_onnx_export_option()
        self.optimum_version = kwargs.pop("optimum_version", None)

    def add_input_info(self, model_inputs: Dict, force_batch_one: bool = False):
        self.input_info = [
            {
                "sample_size": [1] + list(value.shape[1:]) if force_batch_one else list(value.shape),
                "type": "long" if value.dtype is torch.int64 else "float",
                "keyword": name,
            }
            for name, value in model_inputs.items()
        ]

    def _enable_standard_onnx_export_option(self):
        # This method depends on self.save_onnx_model.
        # save_onnx_model is defaulted to false so that the final model output is
        # in OpenVINO IR to realize performance benefit in OpenVINO runtime.
        # True value of save_onnx_model will save a model in onnx format.
        if isinstance(self.compression, dict) and self.compression["algorithm"] == "quantization":
            self.compression["export_to_onnx_standard_ops"] = self.save_onnx_model
        elif isinstance(self.compression, list):
            for i, algo_config in enumerate(self.compression):
                if algo_config["algorithm"] == "quantization":
                    self.compression[i]["export_to_onnx_standard_ops"] = self.save_onnx_model
