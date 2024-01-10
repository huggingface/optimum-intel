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
from tempfile import TemporaryDirectory, gettempdir
from typing import Dict, Optional, Union

import openvino
from huggingface_hub import hf_hub_download
from openvino import Core, convert_model
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation
from transformers import PretrainedConfig
from transformers.file_utils import add_start_docstrings

from optimum.exporters.onnx import OnnxConfig
from optimum.modeling_base import OptimizedModel

from ...exporters.openvino import export, main_export
from ..utils.import_utils import is_nncf_available, is_transformers_version
from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME, _print_compiled_model_properties


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin

core = Core()

logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Base OVModel class.
    """,
)
class OVBaseModel(OptimizedModel):
    auto_model_class = None
    export_feature = None

    def __init__(
        self,
        model: openvino.runtime.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.config = config
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = ov_config if ov_config is not None else {"PERFORMANCE_HINT": "LATENCY"}
        self.preprocessors = kwargs.get("preprocessors", [])
        enable_compilation = kwargs.get("compile", True)

        if self.is_dynamic:
            height = -1 if self.export_feature == "image-classification" else None
            width = -1 if self.export_feature == "image-classification" else None
            model = self._reshape(model, -1, -1, height, width)

        input_names = {}
        for idx, key in enumerate(model.inputs):
            names = tuple(key.get_names())
            input_names[next((name for name in names if "/" not in name), names[0])] = idx
        self.input_names = input_names

        output_names = {}
        for idx, key in enumerate(model.outputs):
            names = tuple(key.get_names())
            output_names[next((name for name in names if "/" not in name), names[0])] = idx
        self.output_names = output_names

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
    def load_model(file_name: Union[str, Path], load_in_8bit: bool = False):
        """
        Loads the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model ONNX or XML file.
        """

        def fix_op_names_duplicates(model: openvino.runtime.Model):
            names = set()
            for op in model.get_ops():
                friendly_name = op.get_friendly_name()
                while True:
                    if friendly_name not in names:
                        break
                    friendly_name += "_"
                names.add(friendly_name)
                op.set_friendly_name(friendly_name)
            return model

        if isinstance(file_name, str):
            file_name = Path(file_name)
        model = core.read_model(file_name) if not file_name.suffix == ".onnx" else convert_model(file_name)
        if file_name.suffix == ".onnx":
            model = fix_op_names_duplicates(model)  # should be called during model conversion to IR

        if load_in_8bit:
            if not is_nncf_available():
                raise ImportError(
                    "Quantization of the weights to int8 requires nncf, please install it with `pip install nncf`"
                )
            import nncf

            model = nncf.compress_weights(model)

        return model

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        dst_path = os.path.join(save_directory, OV_XML_FILE_NAME)
        openvino.save_model(self.model, dst_path, compress_to_fp16=False)

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
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
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

        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)
        return cls(model, config=config, model_save_dir=model_cache_path.parent, **kwargs)

    @staticmethod
    def _cached_file(
        model_path: Union[Path, str],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
    ):
        # locates a file in a local folder and repo, downloads and cache it if necessary.
        model_path = Path(model_path)
        if model_path.is_dir():
            model_cache_path = model_path / file_name
        else:
            file_name = Path(file_name)
            if file_name.suffix != "onnx":
                model_file_names = [file_name.with_suffix(".bin"), file_name]
            else:
                model_file_names = [file_name]
            for file_name in model_file_names:
                model_cache_path = hf_hub_download(
                    repo_id=model_path.as_posix(),
                    filename=file_name.as_posix(),
                    subfolder=subfolder,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            model_cache_path = Path(model_cache_path)

        return model_cache_path

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
        task: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: Optional[bool] = None,
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
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        compression_option = None
        if load_in_8bit is not None:
            compression_option = "int8" if load_in_8bit else "fp32"

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task or cls.export_feature,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            compression_option=compression_option,
        )

        config.save_pretrained(save_dir_path)
        return cls._from_pretrained(model_id=save_dir_path, config=config, load_in_8bit=False, **kwargs)

    @classmethod
    def _to_load(
        cls,
        model,
        config: PretrainedConfig,
        onnx_config: OnnxConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # Export the model to the ONNX format
        export(
            model=model,
            config=onnx_config,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output=save_dir_path / OV_XML_FILE_NAME,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=False,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs,
        )

    def compile(self):
        if self.request is None:
            logger.info(f"Compiling the model to {self._device} ...")
            ov_config = {**self.ov_config}
            if "CACHE_DIR" not in self.ov_config.keys() and not str(self.model_save_dir).startswith(gettempdir()):
                # Set default CACHE_DIR only if it is not set, and if the model is not in a temporary directory
                cache_dir = Path(self.model_save_dir).joinpath("model_cache")
                ov_config["CACHE_DIR"] = str(cache_dir)
                logger.info(f"Setting OpenVINO CACHE_DIR to {str(cache_dir)}")
            self.request = core.compile_model(self.model, self._device, ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                logger.info(f"{self._device} SUPPORTED_PROPERTIES:")
                _print_compiled_model_properties(self.request)

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
        Converts all the model weights to FP16
        """
        apply_moc_transformations(self.model, cf=False)
        compress_model_transformation(self.model)
        self.request = None
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        if isinstance(self, GenerationMixin):
            return True
        return False
