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

import logging as log

from optimum.intel.utils.import_utils import is_torch_version


def patch_model_with_bettertransformer(model):
    if is_torch_version("<", "2.0"):
        log.warn(
            "integration Scaled Dot Product Attention optimization supported only with torch > 2.0."
            "Usage model with stateful=True may be non-effective if model does not contain torch.functional.scaled_dot_product_attention"
            "It is recommended to upgrade PyTorch version for using stateful model or use stateful=False"
        )
    # model already has required SDPA implementation
    if getattr(model, "_supports_sdpa", False) and getattr(model.config, "_attn_implementation", "eager") == "sdpa":
        return model
    try:
        model = model.to_bettertransformer()
    except Exception as e:
        log.warn(
            f"Cannot apply model.to_bettertransformer because of the exception:\n{e}."
            " Usage model with stateful=True may be non-effective if model does not contain torch.functional.scaled_dot_product_attention"
        )
        return model

    return model
