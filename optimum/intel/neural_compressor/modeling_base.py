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
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from intel_extension_for_transformers.backends.neural_engine.compile import compile
from neural_compressor.utils.pytorch import load
from transformers import AutoModel, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import no_init_weights
from transformers.utils import is_ipex_available
from transformers.utils.generic import ContextManagers

from optimum.exporters import TasksManager

from ..generation.modeling import jit_trace
from ..utils.import_utils import _torch_version, is_torch_version, is_transformers_version
from ..utils.modeling_utils import _prepare_attn_mask, _prepare_decoder_attention_mask
from .configuration import INCConfig
from .utils import ENGINE_MODEL_CONFIG, ENGINE_MODEL_NAME, WEIGHTS_NAME


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin

logger = logging.getLogger(__name__)


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
class INCBaseModel:
    _AUTOMODELS_TO_TASKS = {cls_name: task for task, cls_name in TasksManager._TASKS_TO_AUTOMODELS.items()}
    base_model_prefix = "inc_model"
    auto_model_class = AutoModel
    export_feature = None

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super(INCBaseModel, self).__init__(
            model=model, config=config, model_save_dir=model_save_dir, use_cache=use_cache, **kwargs
        )
        if getattr(self.config, "backend", None) == "ipex":
            if not is_ipex_available():
                raise ImportError(
                    "Intel PyTorch Extensions was not found."
                    "please make sure you've installed the package or run "
                    "pip install intel_extension_for_pytorch"
                )
            else:
                # Need import intel_extension_for_pytorch for ipex model
                import intel_extension_for_pytorch as ipex

                # Just to avoid to change by ruff.
                logger.info("intel_extension_for_pytorch version is " + ipex.__version__)

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = WEIGHTS_NAME, **kwargs):
        if getattr(self.config, "backend", None) == "neural_engine":
            self.model.save(save_directory)
        elif getattr(self.config, "torchscript", False):
            torch.jit.save(self.model, os.path.join(save_directory, file_name))
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

        if getattr(config, "torchscript", None) or backend == "neural_engine":
            # Load the model from local directory
            if os.path.isdir(model_id):
                if backend == "neural_engine":
                    file_name = model_id
                else:
                    file_name = os.path.join(model_id, file_name)
                    model_save_dir = model_id
            # Download the model from the hub
            else:
                if backend == "neural_engine":
                    model_file_names = {"model": ENGINE_MODEL_NAME, "config": ENGINE_MODEL_CONFIG}
                    try:
                        for name, file_name in model_file_names.items():
                            model_cache_path = hf_hub_download(
                                repo_id=model_id,
                                filename=file_name,
                                use_auth_token=use_auth_token,
                                revision=revision,
                                cache_dir=cache_dir,
                                force_download=force_download,
                                local_files_only=local_files_only,
                            )
                        file_name = Path(model_cache_path).parent
                    except Exception:
                        logger.warning(
                            f"The file names {ENGINE_MODEL_NAME} or {ENGINE_MODEL_CONFIG} was not found! Please check it!"
                        )
                        raise Exception
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
            if config.torch_dtype != "int8" and config.torch_dtype != torch.int8:
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
                    framework = TasksManager.determine_framework(
                        model_id, subfolder=subfolder, framework=arg_framework
                    )
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
                    raise Exception(
                        "Couldn't load quantized model correctly, "
                        "Please ensure the best_configure is in model state dict!"
                    )

                try:
                    inc_config = INCConfig.from_pretrained(model_id)
                    if not is_torch_version("==", inc_config.torch_version):
                        msg = f"Quantized model was obtained with torch version {inc_config.torch_version} but {_torch_version} was found."
                        logger.warning(f"{msg}")
                except Exception:
                    logger.info("Couldn't verify torch version.")

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

        if config.torch_dtype != "int8" and config.torch_dtype != torch.int8:
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
        if model.config.model_type == "bloom":
            model.transformer._prepare_attn_mask = _prepare_attn_mask

        if model.config.model_type == "llama":
            model.model._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        traced_model = jit_trace(model, task, use_cache)
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
        config.torchscript = True

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

    def eval(self):
        self.model.eval()
