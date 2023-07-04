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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from transformers import PretrainedConfig
from transformers.file_utils import add_start_docstrings

from optimum.intel.generation import BaseModelForCausalLM

from .modeling_base import MODEL_START_DOCSTRING, INCBaseModel


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Neural-compressor Model with a causal language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MODEL_START_DOCSTRING,
)
class INCModelForCausalLM(INCBaseModel, BaseModelForCausalLM):
    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super(INCModelForCausalLM, self).__init__(
            model=model, config=config, model_save_dir=model_save_dir, use_cache=use_cache, **kwargs
        )
