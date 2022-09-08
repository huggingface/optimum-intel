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
from typing import Optional, Union

import transformers
from transformers import AutoConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings, default_cache_path
from transformers.onnx import FeaturesManager, export
from transformers.onnx.utils import get_preprocessor

import openvino
import openvino.runtime.passes as passes
from huggingface_hub import HfApi, hf_hub_download
from openvino.runtime import Core, Dimension
from optimum.modeling_base import OptimizedModel

from .utils import ONNX_WEIGHTS_NAME, OV_WEIGHTS_NAME


core = Core()

logger = logging.getLogger(__name__)

_SUPPORTED_DEVICES = {"CPU"}


@add_start_docstrings(
    """
    Base OVModel class.
    """,
)
class OVBaseModel(OptimizedModel):

    export_feature = None

    def __init__(self, model: openvino.runtime.Model, config: transformers.PretrainedConfig = None, **kwargs):
        self.config = config
        self.model = model
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self._device = kwargs.get("device", "CPU")
        # Ensure the selected device is supported by OpenVINO
        self._ensure_supported_device()
        self.ov_config = {"PERFORMANCE_HINT": "LATENCY"}
        self.request = self._create_infer_request(model)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(model.inputs)}

    @staticmethod
    def load_model(file_name: Union[str, Path], bin_file_name: Union[str, Path] = None):
        """
        Loads the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model weights.
            bin_file_name (`str` or `Path`):
                The path of the model binary file.

        """
        bin_file_name = str(bin_file_name) if bin_file_name is not None else bin_file_name
        return core.read_model(str(file_name), bin_file_name)

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            file_name(`str`, *optional*):
                The model file name to use when saving the model. Overwrites the default file names.
        """
        file_name = file_name if file_name is not None else OV_WEIGHTS_NAME
        dst_path = os.path.join(save_directory, file_name)
        pass_manager = passes.Manager()
        pass_manager.register_pass("Serialize", dst_path, dst_path.replace(".xml", ".bin"))
        pass_manager.run_passes(self.model)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`, *optional*):
                The file name of the model to load. Overwrites the default file name and allows one to load the model
                with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        from_onnx = kwargs.pop("from_onnx", False)
        local_files_only = kwargs.pop("local_files_only", False)
        config_dict = kwargs.pop("config", {})
        config = PretrainedConfig.from_dict(config_dict)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_WEIGHTS_NAME
        file_name = file_name or default_file_name

        # Load the model from local directory
        if os.path.isdir(model_id):
            file_name = os.path.join(model_id, file_name)
            bin_file_name = file_name.replace(".xml", ".bin") if not from_onnx else None
            model = cls.load_model(file_name, bin_file_name)
            kwargs["model_save_dir"] = model_id
        # Download the model from the hub
        else:
            model_file_names = [file_name]
            # If not ONNX then OpenVINO IR
            if not from_onnx:
                model_file_names.append(file_name.replace(".xml", ".bin"))
            file_names = []
            for file_name in model_file_names:
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                file_names.append(Path(model_cache_path).name)
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            bin_file_name = file_names[1] if not from_onnx else None
            model = cls.load_model(file_names[0], bin_file_name=bin_file_name)
        return cls(model, config=config, **kwargs)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        save_dir: Union[str, Path] = default_cache_path,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                Directory from which to load
            save_dir (`str` or `Path`):
                Directory where the exported ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        # Create a local directory to save the model
        save_dir = Path(save_dir).joinpath(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model_save_dir"] = save_dir

        # Get the task to load and export the model with the right topology if available else extract it from the hub
        if cls.export_feature is not None:
            task = cls.export_feature
        else:
            task = HfApi().model_info(model_id, revision=revision).pipeline_tag
            if task in ["sentiment-analysis", "text-classification", "zero-shot-classification"]:
                task = "sequence-classification"
            elif task in ["feature-extraction", "fill-mask"]:
                task = "default"

        # TODO: support private models
        preprocessor = get_preprocessor(model_id)
        # TODO: Add framework ["pt", "tf"]
        model = FeaturesManager.get_model_from_feature(task, model_id)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=task)
        onnx_config = model_onnx_config(model.config)

        # Export the model to the ONNX format
        export(
            preprocessor=preprocessor,
            model=model,
            config=onnx_config,
            opset=onnx_config.default_onnx_opset,
            output=save_dir.joinpath(ONNX_WEIGHTS_NAME),
        )
        kwargs["config"] = model.config.__dict__
        kwargs["from_onnx"] = True

        return cls._from_pretrained(save_dir, **kwargs)

    def _create_infer_request(self, model):
        compiled_model = core.compile_model(model, self._device, self.ov_config)
        return compiled_model.create_infer_request()

    def _ensure_supported_device(self, device: str = None):
        device = device if device is not None else self._device
        if device not in _SUPPORTED_DEVICES:
            raise ValueError(f"Unknown device: {device}. Expected one of {_SUPPORTED_DEVICES}.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError
