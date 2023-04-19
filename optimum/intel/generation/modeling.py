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

import inspect
import logging
import os
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, add_start_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel, FROM_PRETRAINED_START_DOCSTRING

from ..utils.import_utils import is_torch_version, is_transformers_version


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)


class TracedModelForCausalLM(OptimizedModel, GenerationMixin):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    main_input_name = "input_ids"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.model = model
        self.config = config
        self.model_save_dir = model_save_dir
        self.preprocessors = kwargs.get("preprocessors", [])
        self.model_inputs = kwargs.get("model_inputs")
        self.use_cache = use_cache

        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config)

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        model = torch.jit.load(file_name)
        torch.jit.freeze(model.eval())
        return model

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        inputs = {
            "attention_mask": attention_mask,
        }

        if self.use_cache:
            if past_key_values is None:
                dummy_past_key_values = self.model_inputs["past_key_values"]
                nb_layer = len(dummy_past_key_values)
                nb_pkv = len(dummy_past_key_values[0])
                new_shape = list(dummy_past_key_values[0][0].shape)
                new_shape[2] = 0
                new_shape[0] = input_ids.shape[0]
                empty_tensor = torch.empty(size=new_shape)
                past_key_values = tuple(tuple(empty_tensor for _ in range(nb_pkv)) for _ in range(nb_layer))

            inputs["past_key_values"] = past_key_values

        inputs["input_ids"] = input_ids

        outputs = self.model(**inputs)

        return CausalLMOutputWithPast(logits=outputs[0], past_key_values=outputs[1] if self.use_cache else None)

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        config: Optional["PretrainedConfig"] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "OptimizedModel":
        """
        Returns:
            `OptimizedModel`: The loaded optimized model.
        """
        if isinstance(model_id, Path):
            model_id = model_id.as_posix()

        from_transformers = kwargs.pop("from_transformers", None)
        if from_transformers is not None:
            logger.warning(
                "The argument `from_transformers` is deprecated, and will be removed in optimum 2.0.  Use `export` instead"
            )
            export = from_transformers

        if isinstance(model_id, str) and len(model_id.split("@")) == 2:
            if revision is not None:
                logger.warning(
                    f"The argument `revision` was set to {revision} but will be ignored for {model_id.split('@')[1]}"
                )
            model_id, revision = model_id.split("@")

        if config is None:
            if os.path.isdir(os.path.join(model_id, subfolder)) and cls.config_name == CONFIG_NAME:
                if CONFIG_NAME in os.listdir(os.path.join(model_id, subfolder)):
                    config = AutoConfig.from_pretrained(os.path.join(model_id, subfolder, CONFIG_NAME))
                elif CONFIG_NAME in os.listdir(model_id):
                    config = AutoConfig.from_pretrained(os.path.join(model_id, CONFIG_NAME))
                    logger.info(
                        f"config.json not found in the specified subfolder {subfolder}. Using the top level config.json."
                    )
                else:
                    raise OSError(f"config.json not found in {model_id} local folder")
            else:
                config = cls._load_config(
                    model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                    force_download=force_download,
                    subfolder=subfolder,
                )
        elif isinstance(config, (str, os.PathLike)):
            config = cls._load_config(
                config,
                revision=revision,
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
                force_download=force_download,
                subfolder=subfolder,
            )

        if not export and trust_remote_code:
            logger.warning(
                "The argument `trust_remote_code` is to be used along with export=True. It will be ignored."
            )
        elif export and trust_remote_code is None:
            trust_remote_code = False

        from_pretrained_method = cls._from_transformers if export else cls._from_pretrained
        return from_pretrained_method(
            model_id=model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            use_auth_token=use_auth_token,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = WEIGHTS_NAME,
        local_files_only: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        if not config.torchscript:
            raise ValueError("The model is not the script model, please check it!")

        # Load the model from local directory
        if os.path.isdir(model_id):
            file_name = os.path.join(model_id, file_name)
            model = cls.load_model(file_name)
            model_save_dir = model_id
        # The model_id is the TorchScript model
        elif isinstance(model_id, torch.jit.RecursiveScriptModule):
            model = model_id
        # Download the model from the hub
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            model_save_dir = Path(model_cache_path).parent
            model = cls.load_model(model_cache_path)

        # IPEX jit model need 2 iterations to convert model to int8 model
        onnx_config_class = TasksManager.get_exporter_config_constructor(
            model_type=config.model_type, exporter="onnx", task=cls.export_feature
        )
        onnx_config = onnx_config_class(config, use_past=use_cache)
        model_inputs = onnx_config.generate_dummy_inputs(framework="pt")
        for i in range(2):
            model(**model_inputs)

        return cls(
            model,
            config=config,
            model_save_dir=model_save_dir,
            use_cache=use_cache,
            model_inputs=model_inputs,
            **kwargs,
        )

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
        if is_torch_version("<", "2.0.0"):
            raise ImportError("`torch>=2.0.0` is needed to trace your model")

        task = cls.export_feature
        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }

        if isinstance(model_id, str):
            model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
            model.config.return_dict = False
        elif isinstance(model_id, torch.nn.Module):
            model = model_id
            model.config = config
            model.config.return_dict = False
        else:
            raise ValueError("model_id should be path of model or model name in huggingface hub or a torch.nn.Module.")
        signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.call)
        onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        onnx_config = onnx_config_class(model.config, use_past=use_cache)
        dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
        model_inputs = {
            key: dummy_inputs[key] for key in signature.parameters if dummy_inputs.get(key, None) is not None
        }

        if use_cache:
            traced_model = torch.jit.trace(model, example_inputs=tuple(model_inputs.values()))
        else:
            traced_model = torch.jit.trace(model, example_kwarg_inputs=model_inputs)
        traced_model = torch.jit.freeze(traced_model.eval())
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
        config.torchscript = True

        return cls(
            traced_model,
            config=config,
            use_cache=use_cache,
            model_inputs=model_inputs,
            **kwargs,
        )

    def can_generate(self) -> bool:
        if isinstance(self, GenerationMixin):
            return True
        return False

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to(self, device: Union[torch.device, str, int]):
        self.model.to(device)
        return self

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = past_key_values or kwargs.get("past", None)
        if self.use_cache:
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_cache,
            "position_ids": None,
            "attention_mask": kwargs.get("attention_mask", None),
            "token_type_ids": None,
        }
