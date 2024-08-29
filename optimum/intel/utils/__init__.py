#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

from .import_utils import (
    _neural_compressor_version,
    _torch_version,
    compare_versions,
    is_accelerate_available,
    is_diffusers_available,
    is_ipex_available,
    is_neural_compressor_available,
    is_neural_compressor_version,
    is_nncf_available,
    is_numa_available,
    is_openvino_available,
    is_sentence_transformers_available,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
)
