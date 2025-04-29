#  Copyright 2024 The HuggingFace Team. All rights reserved.
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


from typing import Any, Dict, Optional

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer
from sentence_transformers.models.Transformer import _save_pretrained_wrapper
from sentence_transformers.util import import_from_string
from transformers import MT5Config, T5Config
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ..utils.import_utils import _sentence_transformers_version, is_sentence_transformers_version
from .modeling_base import IPEXModel


class IPEXTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        if is_sentence_transformers_version("<", "3.4"):
            raise ImportError(
                f"Backend: ipex requires sentence-transformers>=3.4 but found {_sentence_transformers_version}. "
                "You can install it with pip: `pip install --upgrade sentence-transformers`"
            )

        super().__init__(*args, **kwargs)
        self.backend = "ipex"

    def _load_model(self, model_name_or_path, config, cache_dir, backend, is_peft_model, **model_args) -> None:
        self._load_ipex_model(model_name_or_path, config, cache_dir, **model_args)

    def _load_ipex_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
        if isinstance(config, T5Config) or isinstance(config, MT5Config):
            raise ValueError("T5 models are not yet supported by the IPEX backend.")

        self.auto_model = IPEXModel.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            **model_args,
        )

        # Wrap the save_pretrained method to save the model in the correct subfolder
        self.auto_model._save_pretrained = _save_pretrained_wrapper(self.auto_model._save_pretrained, "ipex")


class IPEXSentenceTransformer(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backend = "ipex"

    def _load_module_class_from_ref(
        self,
        class_ref: str,
        model_name_or_path: str,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        if class_ref.startswith("sentence_transformers."):
            if class_ref == "sentence_transformers.models.Transformer":
                class_ref = "optimum.intel.ipex.modeling_sentence_transformers.IPEXTransformer"
            return import_from_string(class_ref)

        if trust_remote_code:
            code_revision = model_kwargs.pop("code_revision", None) if model_kwargs else None
            try:
                return get_class_from_dynamic_module(
                    class_ref,
                    model_name_or_path,
                    revision=revision,
                    code_revision=code_revision,
                )
            except OSError:
                # Ignore the error if the file does not exist, and fall back to the default import
                pass

        return import_from_string(class_ref)
