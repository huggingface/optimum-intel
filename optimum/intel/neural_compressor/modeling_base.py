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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from intel_extension_for_transformers.backends.neural_engine.compile import compile
from neural_compressor.model.torch_model import IPEXModel, PyTorchModel
from neural_compressor.utils.pytorch import load
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel

from ..utils.import_utils import _torch_version, is_torch_version, is_transformers_version
from .configuration import INCConfig
from .utils import ENGINE_MODEL_CONFIG, ENGINE_MODEL_NAME, WEIGHTS_NAME


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin

logger = logging.getLogger(__name__)

_SUPPORTED_DEVICES = {
    "CPU",
    "GPU",
    "AUTO",
    "AUTO:CPU,GPU",
    "AUTO:GPU,CPU",
    "MULTI",
    "MULTI:CPU,GPU",
    "MULTI:GPU,CPU",
}


_TOKENIZER_FOR_DOC = "AutoTokenizer"
_FEATURE_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

MODEL_START_DOCSTRING = r"""
    This model check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        model (`PyTorch model`): is the main class used to run inference.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is the Model configuration class with all the parameters of the model.
        device (`str`, defaults to `"cpu"`):
            The device type for which the model will be optimized for. The resulting compiled model will contains nodes specific to this device.
"""

INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`torch.Tensor`), *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        token_type_ids (`torch.Tensor`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 1 for tokens that are **sentence A**,
            - 0 for tokens that are **sentence B**.
            [What are token type IDs?](https://huggingface.co/docs/transformers/glossary#token-type-ids)
"""

IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.Tensor`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoFeatureExtractor`](https://huggingface.co/docs/transformers/autoclass_tutorial#autofeatureextractor).
"""


@add_start_docstrings(
    """
    Base INCBaseModel class.
    """,
)
class INCBaseModel(OptimizedModel):
    _AUTOMODELS_TO_TASKS = {cls_name: task for task, cls_name in TasksManager._TASKS_TO_AUTOMODELS.items()}
    base_model_prefix = "inc_model"
    auto_model_class = AutoModel
    export_feature = None

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.model = model
        self.config = config
        self.model_save_dir = model_save_dir
        self.preprocessors = kwargs.get("preprocessors", [])
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backend = getattr(config, "backend", None)

        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        model = torch.jit.load(file_name)
        torch.jit.freeze(model.eval())
        return model

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = WEIGHTS_NAME, **kwargs):
        if self.config.backend == "neural_engine":
            self.model.save(save_directory)
        elif self.config.torchscript:
            torch.jit.save(self.model, os.path.join(save_directory, file_name))
        elif isinstance(self.model, IPEXModel):
            self.model._model.save(os.path.join(save_directory, file_name))
        elif isinstance(self.model, PyTorchModel):
            state_dict = self.model._model.state_dict()

            if hasattr(self.model, "q_config"):
                state_dict["best_configure"] = self.model.q_config
            torch.save(state_dict, os.path.join(save_directory, file_name))
        else:
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(save_directory, file_name))
        logger.info(f"Model weights saved to {save_directory}")

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
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`, *optional*):
                The file name of the model to load. Overwrites the default file name and allows one to load the model
                with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        backend = getattr(config, "backend", None)
        if backend == "ipex":
            pass

        if config.torchscript or backend == "neural_engine":
            # Load the model from local directory
            if os.path.isdir(model_id):
                file_name = os.path.join(model_id, file_name)
                model_save_dir = model_id
            # Download the model from the hub
            else:
                if backend == "neural_engine":
                    file_name = hf_hub_download(
                        repo_id=model_id,
                        filename=ENGINE_MODEL_NAME,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    hf_hub_download(
                        repo_id=model_id,
                        filename=ENGINE_MODEL_CONFIG,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                else:
                    file_name = hf_hub_download(
                        repo_id=model_id,
                        filename=file_name,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                model_save_dir = Path(file_name).parent
            if backend == "neural_engine":
                model = compile(model_save_dir)
                config.backend = "neural_engine"
            else:
                try:
                    model = compile(file_name)
                    config.torchscript = False
                    config.backend = "neural_engine"
                except Exception as e:
                    logger.warning(e)
                    logger.info("Compile model with neural engine failed! Inference with original model.")
                    model = cls.load_model(file_name)
        else:
            model_save_dir = None
            model_kwargs = {
                "revision": revision,
                "use_auth_token": use_auth_token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "force_download": force_download,
                "torch_dtype": torch_dtype,
            }
            task = cls.export_feature
            if config.torch_dtype != "int8":
                model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
            else:
                state_dict_path = kwargs.get("state_dict_path", None)
                import copy

                from transformers.models.auto.auto_factory import _get_model_class
                from transformers.utils import TRANSFORMERS_CACHE, is_offline_mode

                model_class = _get_model_class(config, cls.auto_model_class._model_mapping)
                keys_to_ignore_on_load_unexpected = copy.deepcopy(
                    getattr(model_class, "_keys_to_ignore_on_load_unexpected", None)
                )
                keys_to_ignore_on_load_missing = copy.deepcopy(
                    getattr(model_class, "_keys_to_ignore_on_load_missing", None)
                )
                # Avoid unnecessary warnings resulting from quantized model initialization
                quantized_keys_to_ignore_on_load = [
                    r"zero_point",
                    r"scale",
                    r"packed_params",
                    r"constant",
                    r"module",
                    r"best_configure",
                    r"max_val",
                    r"min_val",
                    r"eps",
                    r"fake_quant_enabled",
                    r"observer_enabled",
                ]
                if keys_to_ignore_on_load_unexpected is None:
                    model_class._keys_to_ignore_on_load_unexpected = quantized_keys_to_ignore_on_load
                else:
                    model_class._keys_to_ignore_on_load_unexpected.extend(quantized_keys_to_ignore_on_load)
                missing_keys_to_ignore_on_load = [r"weight", r"bias"]
                if keys_to_ignore_on_load_missing is None:
                    model_class._keys_to_ignore_on_load_missing = missing_keys_to_ignore_on_load
                else:
                    model_class._keys_to_ignore_on_load_missing.extend(missing_keys_to_ignore_on_load)

                try:
                    model = model_class.from_pretrained(model_id, **kwargs)
                except AttributeError:
                    init_contexts = [no_init_weights(_enable=True)]
                    with ContextManagers(init_contexts):
                        model = model_class(config, **kwargs)

                model_class._keys_to_ignore_on_load_unexpected = keys_to_ignore_on_load_unexpected
                model_class._keys_to_ignore_on_load_missing = keys_to_ignore_on_load_missing

                if state_dict_path is None:
                    revision = model_kwargs.get("revision", None)
                    if os.path.isdir(model_id):
                        state_dict_path = os.path.join(model_id, file_name)
                    elif os.path.isfile(model_id):
                        state_dict_path = model_id
                    else:
                        local_files_only = False
                        if is_offline_mode():
                            logger.info("Offline mode: forcing local_files_only=True")
                            local_files_only = True
                        cache_dir = model_kwargs.get("cache_dir", None)
                        if cache_dir is None:
                            cache_dir = TRANSFORMERS_CACHE
                        if isinstance(cache_dir, Path):
                            cache_dir = str(cache_dir)
                        try:
                            state_dict_path = hf_hub_download(
                                repo_id=model_id,
                                filename=file_name,
                                revision=revision,
                                cache_dir=cache_dir,
                                local_files_only=local_files_only,
                            )
                        except EnvironmentError as err:
                            logger.error(err)
                            msg = (
                                f"Can't load config for '{model_id}'. Make sure that:\n\n"
                                f"-'{model_id}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                                f"-or '{model_id}' is a correct path to a directory containing a {file_name} file\n\n"
                            )

                            if revision is not None:
                                msg += (
                                    f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that "
                                    f"exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"
                                )

                            raise EnvironmentError(msg)

                msg = None
                try:
                    inc_config = INCConfig.from_pretrained(model_id)
                    if not is_torch_version("==", inc_config.torch_version):
                        msg = f"Quantized model was obtained with torch version {inc_config.torch_version} but {_torch_version} was found."
                        logger.warning(f"{msg}")
                except Exception:
                    logger.info("Couldn't verify torch version.")

                # Load the state dictionary of the model to verify whether the model is quantized or not
                state_dict = torch.load(state_dict_path, map_location="cpu")

                if "best_configure" in state_dict and state_dict["best_configure"] is not None:
                    try:
                        model = load(state_dict_path, model)
                    except Exception as e:
                        if msg is not None:
                            e.args += (msg,)
                        raise

            model.eval()

        return cls(
            model,
            config=config,
            model_save_dir=model_save_dir,
            use_cache=use_cache,
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
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into a TorchScript model using `torch.jit.trace`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            config (`PretrainedConfig`) :
                an object of PretrainedConfig.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
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
            "torch_dtype": torch_dtype,
        }

        if config.torch_dtype != "int8":
            model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
        else:
            file_name = kwargs.get("file_name", WEIGHTS_NAME)
            # Load the model from local directory
            if os.path.isdir(model_id):
                state_dict_path = os.path.join(model_id, file_name)
            # Download the model from the hub
            else:
                state_dict_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            # Load the state dictionary of the model to verify whether the model is quantized or not
            state_dict = torch.load(state_dict_path, map_location="cpu")
            if "best_configure" in state_dict and state_dict["best_configure"] is not None:
                subfolder = kwargs.get("subfolder", "")
                arg_framework = kwargs.get("framework", None)
                framework = TasksManager.determine_framework(model_id, subfolder=subfolder, framework=arg_framework)
                model_class = TasksManager.get_model_class_for_task(task, framework)
                init_contexts = [no_init_weights(_enable=True)]
                with ContextManagers(init_contexts):
                    model = model_class.from_pretrained(model_id, **model_kwargs)
                try:
                    model = load(state_dict_path, model)
                except Exception as e:
                    logger.error(e.args)
                    raise
            else:
                model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
        try:
            model.config.return_dict = False
            signature = (
                inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.call)
            )
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
        except Exception as e:
            logger.warning(f"Unexpected {e=}, {type(e)=}")
            logger.info("Can't trace the model, use original model now!")
            save_dir_path = model_id

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

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return cls._AUTOMODELS_TO_TASKS[auto_model_class.__name__]

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        if isinstance(self, GenerationMixin):
            return True
        return False

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def eval(self):
        self.model.eval()
