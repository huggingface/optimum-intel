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
from transformers import PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.utils import is_ipex_available

from optimum.exporters import TasksManager

from ..generation.modeling import jit_trace
from ..utils.import_utils import is_torch_version
from ..utils.modeling_utils import _prepare_attn_mask, _prepare_decoder_attention_mask
from .quantization import INCModel
from .utils import WEIGHTS_NAME


logger = logging.getLogger(__name__)


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


@add_start_docstrings(
    """
    Base INCBaseModel class.
    """,
)
class INCBaseModel:
    base_model_prefix = "inc_model"

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

    def _save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        if getattr(self.config, "torchscript", False):
            torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))
        else:
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(save_directory, WEIGHTS_NAME))
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
        file_name: Optional[str] = None,
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
                with a different name. This argument will be deprecated in next release.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        if file_name is not None:
            logger.warning("The argument of `file_name` will be deprecated in next release.")
        else:
            file_name = WEIGHTS_NAME
        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }
        if getattr(config, "torchscript", None):
            # Load the model from local directory
            if os.path.isdir(model_id):
                file_name = os.path.join(model_id, file_name)
                model_save_dir = model_id
            # Download the model from the hub
            else:
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    **model_kwargs,
                )
                model_save_dir = Path(model_cache_path).parent
            model = cls.load_model(file_name)
        else:
            model_save_dir = None
            task = cls.export_feature
            if config.torch_dtype != "int8" and config.torch_dtype != torch.int8:
                model = TasksManager.get_model_from_task(task, model_id, torch_dtype=torch_dtype, **model_kwargs)
            else:
                INCModel.TRANSFORMERS_AUTO_CLASS = cls.auto_model_class
                model = INCModel.from_pretrained(model_id, q_model_name=file_name, **model_kwargs)

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
            file_name = kwargs.get("file_name", None)
            if file_name is not None:
                logger.warning("The argument of `file_name` will be deprecated in next release.")
            INCModel.TRANSFORMERS_AUTO_CLASS = cls.auto_model_class
            model = INCModel.from_pretrained(model_id, q_model_name=file_name, **model_kwargs)

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

    def eval(self):
        self.model.eval()
