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


from optimum.intel.utils.import_utils import is_nncf_available

from .configuration import (
    OVConfig,
    OVDynamicQuantizationConfig,
    OVMixedQuantizationConfig,
    OVQuantizationConfig,
    OVQuantizationConfigBase,
    OVQuantizationMethod,
    OVWeightQuantizationConfig,
)


if is_nncf_available():
    # Quantization is possible only if nncf is installed
    from .quantizer import OVQuantizer
