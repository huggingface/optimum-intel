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

import transformers
from transformers import AutoConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings, default_cache_path
from transformers.onnx.utils import get_preprocessor

import openvino
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation
from openvino.runtime import Core
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.modeling_base import OptimizedModel

from ..utils.import_utils import is_transformers_version
from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin

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
    _AUTOMODELS_TO_TASKS = {cls_name: task for task, cls_name in TasksManager._TASKS_TO_AUTOMODELS.items()}
    auto_model_class = None
    export_feature = None

    def __init__(
        self,
        model: openvino.runtime.Model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs
    ):
        self.config = config
        self.model_save_dir = model_save_dir
        self._device = kwargs.get("device", "CPU").upper()
        self.is_dynamic = kwargs.get("dynamic_shapes", True)
        self.ov_config = {"PERFORMANCE_HINT": "LATENCY"}
        self.preprocessors = kwargs.get("preprocessors", [])
        enable_compilation = kwargs.get("compile", True)
        if "GPU" in self._device and self.is_dynamic:
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
        if enable_compilation:
            self.compile()

        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        """
        Loads the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model ONNX or XML file.
        """
        if isinstance(file_name, str):
            file_name = Path(file_name)
        bin_file_name = file_name.with_suffix(".bin") if file_name.suffix == ".xml" else None

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
        openvino.runtime.serialize(self.model, dst_path, dst_path.replace(".xml", ".bin"))

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
        from_onnx: bool = False,
        local_files_only: bool = False,
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
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        # Load the model from local directory
        if os.path.isdir(model_id):
            file_name = os.path.join(model_id, file_name)
            if os.path.isfile(os.path.join(model_id, "ov_model.xml")):
                file_name = os.path.join(model_id, "ov_model.xml")
                logger.warning(
                    "The file names `ov_model.xml` and `ov_model.bin` will be soon deprecated."
                    "Make sure to rename your file to respectively `openvino_model.xml` and `openvino_model.bin`"
                )
            model = cls.load_model(file_name)
            model_save_dir = model_id
        # Download the model from the hub
        else:
            model_file_names = [file_name]
            # If not ONNX then OpenVINO IR
            if not from_onnx:
                model_file_names.append(file_name.replace(".xml", ".bin"))
            file_names = []
            try:
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
                    file_names.append(model_cache_path)
            except EntryNotFoundError:
                file_names = []
                model_file_names = ["ov_model.xml", "ov_model.bin"]
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
                    file_names.append(model_cache_path)
                logger.warning(
                    "The file names `ov_model.xml` and `ov_model.bin` will be soon deprecated."
                    "Make sure to rename your file to respectively `openvino_model.xml` and `openvino_model.bin`"
                )
            model_save_dir = Path(model_cache_path).parent
            model = cls.load_model(file_names[0])
        return cls(model, config=config, model_save_dir=model_save_dir, **kwargs)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        task: Optional[str] = None,
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
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        model = TasksManager.get_model_from_task(task, model_id)
        model_type = model.config.model_type.replace("_", "-")
        onnx_config_class = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model=model,
            task=task,
            model_name=model_id,
            model_type=model_type,
        )

        onnx_config = onnx_config_class(model.config)
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # Export the model to the ONNX format
        export(
            model=model,
            config=onnx_config,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output=save_dir_path / ONNX_WEIGHTS_NAME,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=True,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs,
        )

    def compile(self):
        if self.request is None:
            logger.info("Compiling the model and creating the inference request ...")
            # Only enable CACHE_DIR for GPU because CACHE_DIR fails with some INT8 models on CPU with 2022.3
            ov_config = self.ov_config.copy()
            if self._device == "GPU":
                cache_dir = Path(self.model_save_dir).joinpath("model_cache")
                ov_config["CACHE_DIR"] = str(cache_dir)
            compiled_model = core.compile_model(self.model, self._device, ov_config)
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
        apply_moc_transformations(self.model, cf=False)
        compress_model_transformation(self.model)
        self.request = None
        return self

    def _ensure_supported_device(self, device: str = None):
        device = device if device is not None else self._device
        if device not in _SUPPORTED_DEVICES:
            raise ValueError(f"Unknown device: {device}. Expected one of {_SUPPORTED_DEVICES}.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

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
