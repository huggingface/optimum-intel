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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig
from transformers.file_utils import add_start_docstrings

from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.modeling_base import OptimizedModel
from optimum.exporters.onnx import export_models, get_decoder_models_for_export

from ..utils.import_utils import is_transformers_version
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_PAST_NAME = "pytorch_model_with_past.bin"


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin

core = Core()

logger = logging.getLogger(__name__)

class INCModelForGeneration(OptimizedModel):
    export_feature = "causal-lm"
    auto_model_class = AutoModelForCausalLM

    def __init__(
        self,
        model,
        model_with_past = None,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.model = model
        self.model_with_past = model_with_past

        self.config = config
        self.model_save_dir = model_save_dir

        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        if isinstance(file_name, str):
            file_name = Path(file_name)
        return None

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))
        if self.model_with_past is not None:    
            torch.jit.save(self.model_with_past, os.path.join(save_directory, WEIGHTS_PAST_NAME))

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        # TODO : downloads and load the model subcomponent 
        return cls(model, model_with_past, config=config, model_save_dir=model_save_dir, **kwargs)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        task = cls.export_feature

        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }

        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
        model.config.return_dict = False
        signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.call)


        # onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        # onnx_config = onnx_config_class(model.config, use_past=use_cache)
        # models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)


        onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        onnx_config = onnx_config_class(model.config)
        dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
        model_inputs = {key: dummy_inputs[key] for key in signature.parameters if dummy_inputs.get(key, None) is not None}
        input_key_list = model_inputs.keys()
        inputs = tuple(model_inputs.values())

        traced_model = torch.jit.trace(model, example_inputs=inputs, strict=False)
        traced_model = torch.jit.freeze(traced_model.eval())
        # Only for IPEX int8 jit model, IPEX jit model need 2 iterations to convert model to int8 model
        for i in range(2):
            traced_model(*inputs)

        torch.jit.save(traced_model, os.path.join(save_dir_path, WEIGHTS_NAME))

        # TODO : add use_cache
        # if use_cache:
        # torch.jit.save(model_with_past, os.path.join(save_dir_path, WEIGHTS_PAST_NAME))

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        # TODO : add
        raise NotImplementedError

    def can_generate(self) -> bool:
        if isinstance(self, GenerationMixin):
            return True
        return False