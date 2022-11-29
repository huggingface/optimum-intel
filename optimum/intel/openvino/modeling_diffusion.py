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

import numpy as np

import openvino
from huggingface_hub import HfApi, hf_hub_download
from openvino.offline_transformations import compress_model_transformation
from openvino.runtime import Core
from optimum.modeling_base import OptimizedModel

from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME


core = Core()

logger = logging.getLogger(__name__)


class OVModelForStableDiffusion:
    def __init__(self, model: openvino.runtime.Model, **kwargs):
        logger.info(
            "`optimum.intel.openvino.OVModelForStableDiffusion` is experimental and might change in the future."
        )
        self.model = model
        self._device = kwargs.get("device", "CPU")
        self.ov_config = {"PERFORMANCE_HINT": "LATENCY"}
        self.request = None

        # Reshape the U-NET model
        if "encoder_hidden_states" in [inputs.get_any_name() for inputs in self.model.inputs]:
            self.model = self._reshape(self.model)

        # disable_compilation = kwargs.get("disable_compilation", False)
        self.compile()

    def __call__(self, **kwargs):
        self.compile()
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        outputs = self.request.infer(inputs)
        return [value for value in outputs.values()]

    @staticmethod
    def load_model(file_name: Union[str, Path], bin_file_name: Optional[Union[str, Path]] = None):
        """
        Load the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model ONNX or XML file.
            bin_file_name (`str` or `Path`, *optional*):
                The path of the model binary file, for OpenVINO IR the weights file is expected to be in the same
                directory as the .xml file.
        """
        return core.read_model(file_name, bin_file_name)

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None):
        """
        Save the model to a directory.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            file_name(`str`, *optional*):
                The model file name to use when saving the model. Overwrites the default file names.
        """
        file_name = file_name if file_name is not None else OV_XML_FILE_NAME
        dst_path = os.path.join(save_directory, file_name)
        openvino.runtime.serialize(self.model, dst_path, dst_path.replace(".xml", ".bin"))

    def save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None):
        """
        Save the model to a directory so that it can be re-loaded using the
        [`~OVModelForStableDiffusion.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files. Will be created if it doesn't exist.
            file_name(`str`, *optional*):
                The model file name to use when saving the model. Overwrites the default file names.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Saves the model weights into a directory
        self._save_pretrained(save_directory, file_name)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        from_onnx: bool = False,
        **kwargs,
    ):
        """
        Load a model from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                Directory from which to load
            use_auth_token (`str` or `bool`, *optional*):
                Is needed to load models from a private repository
            revision (`str`, *optional*):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`str`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            file_name (`str`, *optional*):
                Overwrites the default model file name from `"openvino_model.bin"` to `file_name`.
                This allows you to load different model files from the same repository or directory.
            from_onnx (`bool`, *optional*, defaults to `False`):
                Wether or not to load the model from an ONNX or XML file.
        """
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        model_file_name = file_name or default_file_name
        # Load the model from local directory
        if os.path.isdir(model_id):
            model = core.read_model(os.path.join(model_id, model_file_name))
        # Download the model from the hub
        else:
            file_names = [model_file_name]
            if not from_onnx:
                file_names.append(model_file_name.replace(".xml", ".bin"))
            for file_name in file_names:
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                )
            model_cache_path = model_cache_path.replace(".bin", ".xml") if not from_onnx else model_cache_path
            model = core.read_model(model_cache_path)
        return cls(model=model, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        force_download: bool = True,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **model_kwargs,
    ):
        revision = None
        if len(str(model_id).split("@")) == 2:
            model_id, revision = model_id.split("@")

        return cls._from_pretrained(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            use_auth_token=use_auth_token,
            **model_kwargs,
        )

    def _reshape(
        self,
        model: openvino.runtime.Model,
        batch_size: int = -1,
        sequence_length: int = None,
        height: int = 64,
        width: int = 64,
        num_channels: int = 4,
    ):
        shapes = {}
        for inputs in model.inputs:
            if "sample" in inputs.get_names():
                dim_1 = num_channels
                dim_2 = height
                dim_3 = width
            elif "encoder_hidden_states" in inputs.get_names():
                dim_1 = sequence_length
                dim_2 = None
                dim_3 = None
            else:
                continue

            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size

            if dim_1 is not None:
                shapes[inputs][1] = dim_1
            if dim_2 is not None:
                shapes[inputs][2] = dim_2
            if dim_3 is not None:
                shapes[inputs][3] = dim_3
        model.reshape(shapes)
        return model

    def compile(self):
        if self.request is None:
            logger.info("Compiling the model and creating the inference request ...")
            compiled_model = core.compile_model(self.model, self._device, self.ov_config)
            self.request = compiled_model.create_infer_request()

    def to(self, device: str):
        self._device = device
        self.request = None
        return self
