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

from optimum.intel.utils.import_utils import (
    _torch_version,
    _transformers_version,
    is_torch_version,
    is_transformers_version,
)


def patch_model_with_bettertransformer(model):
    # check that the model has not yet been pathced
    if hasattr(model, "use_bettertransformer") and model.use_bettertransformer is True:
        return model

    if is_transformers_version("<", "4.36") or is_torch_version("<", "2.1.1"):
        COLOR_RED = "\033[1;31m"
        COLOR_RESET = "\033[0m"
        log.warn(
            COLOR_RED
            + "[WARNING] For good performance with stateful models, transformers>=4.36.2 and PyTorch>=2.1.1 are required. "
            f"This Python environment has Transformers {_transformers_version} and PyTorch {_torch_version}. "
            "Consider upgrading PyTorch and Transformers, for example by running "
            "`pip install --upgrade --upgrade-strategy eager optimum[openvino]`, and export the model again"
            + COLOR_RESET
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
