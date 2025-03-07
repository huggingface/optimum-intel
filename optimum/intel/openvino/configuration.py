# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

# ruff: noqa

from .quantization.configuration import (
    OVQuantizationMethod,
    OVQuantizationConfigBase,
    OVQuantizationConfig,
    OVWeightQuantizationConfig,
    OVDynamicQuantizationConfig,
    OVMixedQuantizationConfig,
    OVConfig,
    _DEFAULT_4BIT_CONFIG,
    _DEFAULT_4BIT_CONFIGS,
    get_default_int4_config,
    _check_default_4bit_configs,
)

logger = logging.getLogger(__name__)

logger.warning(
    "`optimum.intel.configuration` import path is deprecated and will be removed in optimum-intel v1.24. "
    "Please use `optimum.intel.quantization.configuration` instead."
)
