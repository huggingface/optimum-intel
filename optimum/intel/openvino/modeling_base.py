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
from openvino.offline_transformations import compress_model_transformation
from openvino.runtime import Core, Dimension
from optimum.modeling_base import OptimizedModel

from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME


core = Core()

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


@add_start_docstrings(
    """
    Base OVModel class.
    """,
)
class OVBaseModel(OptimizedModel):

    export_feature = None

    def __init__(self, model: openvino.runtime.Model, config: transformers.PretrainedConfig = None, **kwargs):
        self.config = config
        self.model_save_dir = kwargs.get("model_save_dir")
        self.device = kwargs.get("device", "CPU")
        self.is_dynamic = kwargs.get("dynamic_shapes", True)
        self.ov_config = {"PERFORMANCE_HINT": "LATENCY"}
        cache_dir = Path(self.model_save_dir).joinpath("model_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.ov_config["CACHE_DIR"] = str(cache_dir)
        if "GPU" in self.device and self.is_dynamic:
            raise ValueError(
                "Support of dynamic shapes for GPU devices is not yet available. Set `dynamic_shapes` to `False` to continue."
            )
        if self.is_dynamic:
            height = -1 if self.export_feature == "image-classification" else None
            width = -1 if self.export_feature == "image-classification" else None
            model = self._reshape(model, -1, -1, height, width)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(model.inputs)}
        self.model = model
        self.request = None

    @staticmethod
    def load_model(file_name: Union[str, Path], bin_file_name: Optional[Union[str, Path]] = None):
        """
        Loads the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model ONNX or XML file.
            bin_file_name (`str` or `Path`, *optional*):
                The path of the model binary file, for OpenVINO IR the weights file is expected to be in the same
                directory as the .xml file if not provided.
        """
        return core.read_model(file_name, bin_file_name)

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
        file_name = file_name if file_name is not None else OV_XML_FILE_NAME
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
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
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
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
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
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, default to
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

    def _create_inference_request(self):
        if self.request is None:
            logger.info("Compiling the model and creating the inference request ...")
            compiled_model = core.compile_model(self.model, self.device, self.ov_config)
            self.request = compiled_model.create_infer_request()

    def _reshape(
        self,
        model: openvino.runtime.Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size
            shapes[inputs][1] = sequence_length
            if height is not None:
                shapes[inputs][2] = height
            if width is not None:
                shapes[inputs][3] = width
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int, height: int = None, width: int = None):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            sequence_length (`int`):
                The sequence length or number of channels.
            height (`int`, *optional*):
                The image height.
            width (`int`, *optional*):
                The image width.
        """
        self.is_dynamic = True if batch_size == -1 and sequence_length == -1 else False
        self.model = self._reshape(self.model, batch_size, sequence_length, height, width)
        self.request = None
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        compress_model_transformation(self.model)
        self.request = None
        return self

    def _ensure_supported_device(self, device: str = None):
        device = device if device is not None else self.device
        if device not in _SUPPORTED_DEVICES:
            raise ValueError(f"Unknown device: {device}. Expected one of {_SUPPORTED_DEVICES}.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError
