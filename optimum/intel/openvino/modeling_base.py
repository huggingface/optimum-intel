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
import copy
import logging
import os
import warnings
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, List, Optional, Union

import openvino
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino import CompiledModel, Core, Model, convert_model
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation
from transformers import GenerationConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.generation import GenerationMixin
from transformers.utils import is_offline_mode
from transformers.utils.hub import cached_file

from optimum.exporters.base import ExportConfig
from optimum.modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel

from ...exporters.openvino import export, main_export
from ..utils.import_utils import is_nncf_available, is_transformers_version
from ..utils.modeling_utils import _find_files_matching_pattern
from .configuration import OVConfig, OVDynamicQuantizationConfig, OVWeightQuantizationConfig
from .utils import (
    ONNX_WEIGHTS_NAME,
    OV_TO_PT_TYPE,
    OV_XML_FILE_NAME,
    TemporaryDirectory,
    _print_compiled_model_properties,
    model_has_dynamic_inputs,
)


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
    _supports_cache_class = False
    _library_name = "transformers"
    _xml_model_name = OV_XML_FILE_NAME
    _search_pattern = r"(.*)?openvino(.*)?\_(.*)?.xml$"

    def __init__(
        self,
        model: openvino.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        self.config = config
        self.name_or_path = getattr(config, "name_or_path", None)
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])
        self._compile_only = kwargs.get("compile_only", False)
        enable_compilation = kwargs.get("compile", True)

        if self._compile_only:
            if not enable_compilation:
                raise ValueError(
                    "`compile_only` mode does not support disabling compilation."
                    "Please provide `compile=True` if you want to use `compile_only=True` or set `compile_only=False`"
                )

            if not isinstance(model, CompiledModel):
                raise ValueError("`compile_only` expect that already compiled model will be provided")

            model_dynamic_shapes = model_has_dynamic_inputs(model)
            if dynamic_shapes ^ model_dynamic_shapes:
                raise ValueError(
                    f"Provided compiled model with {'dynamic' if model_dynamic_shapes else 'static'} shapes but requested to use {'dynamic' if dynamic_shapes else 'static'}. Please set `compile_only=False` or `dynamic_shapes`={model_dynamic_shapes}"
                )

        if self.is_dynamic and not self._compile_only:
            height = -1 if self.export_feature == "image-classification" else None
            width = -1 if self.export_feature == "image-classification" else None
            model = self._reshape(model, -1, -1, height, width)

        input_names = {}
        input_dtypes = {}
        for idx, key in enumerate(model.inputs):
            names = tuple(key.get_names())
            input_names[next((name for name in names if "/" not in name), names[0])] = idx
            input_dtypes[
                next((name for name in names if "/" not in name), names[0])
            ] = key.get_element_type().get_type_name()
        self.input_names = input_names
        self.input_dtypes = input_dtypes

        output_names = {}
        output_dtypes = {}
        for idx, key in enumerate(model.outputs):
            names = tuple(key.get_names())
            output_names[next((name for name in names if "/" not in name), names[0])] = idx
            output_dtypes[
                next((name for name in names if "/" not in name), names[0])
            ] = key.get_element_type().get_type_name()

        self.output_names = output_names
        self.output_dtypes = output_dtypes
        self.model = model
        self.request = None if not self._compile_only else self.model

        generation_config = kwargs.get("generation_config", None)
        if self.can_generate():
            self.generation_config = generation_config or GenerationConfig.from_model_config(config)

            if is_transformers_version(">=", "4.44.99"):
                # some model configs may have issues with loading without parameters initialization
                try:
                    misplaced_generation_parameters = self.config._get_non_default_generation_parameters()
                except (KeyError, TypeError):
                    misplaced_generation_parameters = {}
                if len(misplaced_generation_parameters) > 0:
                    logger.warning(
                        "Moving the following attributes in the config to the generation config: "
                        f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                        "generation parameters in the model config, as opposed to in the generation config.",
                    )
                    for param_name, param_value in misplaced_generation_parameters.items():
                        setattr(self.generation_config, param_name, param_value)
                        setattr(self.config, param_name, None)

        else:
            self.generation_config = None

        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()

        if not self._compile_only and enable_compilation:
            self.compile()

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (for torch compatibility).
        """
        return torch.device("cpu")

    def to(self, device: str):
        """
        Use the specified `device` for inference. For example: "cpu" or "gpu". `device` can
        be in upper or lower case. To speed up first inference, call `.compile()` after `.to()`.
        """
        if self._compile_only and isinstance(device, str):
            raise ValueError(
                "`to()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()
        else:
            logger.debug(f"device must be of type {str} but got {type(device)} instead")

        return self

    def clear_requests(self):
        self.request = None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        for dtype in self.input_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    @property
    def ov_submodels(self) -> Dict[str, openvino.Model]:
        return {submodel_name: getattr(self, submodel_name) for submodel_name in self._ov_submodel_names}

    @property
    def _ov_submodel_names(self) -> List[str]:
        """
        List of openvino submodel names. Used as keys for a dictionary returned by `.ov_submodels` property.
        """
        return ["model"]

    @staticmethod
    def load_model(
        file_name: Union[str, Path],
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
    ) -> openvino.Model:
        """
        Loads the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model ONNX or XML file.
            quantization_config (`OVWeightQuantizationConfig` or `Dict`, *optional*):
                Quantization config to apply after model is loaded.
        """

        def fix_op_names_duplicates(model: openvino.Model):
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
        model = (
            core.read_model(file_name.resolve(), file_name.with_suffix(".bin").resolve())
            if not file_name.suffix == ".onnx"
            else convert_model(file_name)
        )
        if file_name.suffix == ".onnx":
            model = fix_op_names_duplicates(model)  # should be called during model conversion to IR

        # TODO: remove this way of applying quantization; instead apply it after instance of OVModel* is loaded
        if quantization_config:
            if not is_nncf_available():
                raise ImportError(
                    "Quantization of the weights to int8 requires nncf, please install it with `pip install nncf`"
                )

            from optimum.intel.openvino.quantization import _weight_only_quantization

            if not isinstance(quantization_config, (dict, OVWeightQuantizationConfig)):
                raise TypeError(
                    f"Expected `quantization_config` to be either a dictionary or OVWeightQuantizationConfig object, got {type(quantization_config)}."
                )

            model = _weight_only_quantization(model, quantization_config)

        return model

    @staticmethod
    def _compile_model(
        model: Union[str, Path, Model],
        device: Optional[str] = None,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Union[str, Path] = None,
    ):
        if isinstance(model, str):
            model = Path(model)
        ov_config = ov_config or {}

        if model_save_dir is None and isinstance(model, Path):
            model_save_dir = model.parent
        if "CACHE_DIR" not in ov_config.keys() and (
            model_save_dir is not None
            and not str(model_save_dir).startswith(gettempdir())
            and (device is not None and "gpu" in device.lower())
        ):
            # Set default CACHE_DIR only if it is not set, if the model is not in a temporary directory, and device is GPU
            cache_dir = Path(model_save_dir).joinpath("model_cache")
            ov_config["CACHE_DIR"] = str(cache_dir)
            logger.info(f"Setting OpenVINO CACHE_DIR to {str(cache_dir)}")

        compiled_model = core.compile_model(model, device.upper() if device is not None else device, config=ov_config)
        if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
            _print_compiled_model_properties(compiled_model)
        return compiled_model

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """

        if self._compile_only:
            raise ValueError(
                "`save_pretrained()` is not supported with `compile_only=True` mode, to save your model please initialize your model with compile_only=False"
            )
        dst_path = os.path.join(save_directory, self._xml_model_name)
        openvino.save_model(self.model, dst_path, compress_to_fp16=False)
        generation_config = getattr(self, "generation_config", None)
        if generation_config is not None:
            try:
                generation_config.save_pretrained(save_directory)
            except Exception as exception:
                logger.warning(
                    f"The generation config will not be saved, saving failed with following error:\n{exception}"
                )

        self._save_openvino_config(save_directory)

    def _save_openvino_config(self, save_directory: Union[str, Path]):
        if self._openvino_config is not None:
            if not isinstance(self._openvino_config.quantization_config.dataset, (str, type(None))):
                self._openvino_config.quantization_config.dataset = None

            self._openvino_config.save_pretrained(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
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
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name (`str`, *optional*):
                The file name of the model to load. Overwrites the default file name and allows one to load the model
                with a different name.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            load_in_8bit (`bool`, *optional*, defaults to `False`):
                Whether or not to apply 8-bit weight quantization.
        """
        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        compile_only = kwargs.get("compile_only", False)

        quantization_config = cls._prepare_quantization_config(quantization_config, load_in_8bit)
        is_data_aware_quantization = quantization_config is not None and quantization_config.dataset is not None

        if not compile_only:
            ov_model = cls.load_model(
                model_cache_path, quantization_config=None if is_data_aware_quantization else quantization_config
            )
        else:
            ov_model = cls._compile_model(
                model_cache_path,
                kwargs.get("device"),
                kwargs.get("ov_config"),
                model_save_dir=model_cache_path.parent,
            )

        model = cls(
            ov_model,
            config=config,
            model_save_dir=model_cache_path.parent,
            quantization_config=quantization_config,
            **kwargs,
        )

        if is_data_aware_quantization:
            from optimum.intel import OVQuantizer

            quantizer = OVQuantizer(model)
            quantization_config_copy = copy.deepcopy(quantization_config)
            quantization_config_copy.tokenizer = quantization_config.tokenizer or model_id
            quantization_config_copy.processor = quantization_config.processor or model_id
            quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config_copy))

        return model

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        config: Optional[PretrainedConfig] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        _export = export
        try:
            if local_files_only:
                object_id = model_id.replace("/", "--")
                cached_model_dir = os.path.join(cache_dir, f"models--{object_id}")
                refs_file = os.path.join(os.path.join(cached_model_dir, "refs"), revision or "main")
                with open(refs_file) as f:
                    revision = f.read()
                model_dir = os.path.join(cached_model_dir, "snapshots", revision)
            else:
                model_dir = model_id

            ov_files = _find_files_matching_pattern(
                model_dir,
                pattern=cls._search_pattern if not kwargs.get("from_onnx", False) else "*.onnx",
                subfolder=subfolder,
                use_auth_token=token,
                revision=revision,
            )
            _export = len(ov_files) == 0
            if _export ^ export:
                if export:
                    logger.warning(
                        f"The model {model_id} was already converted to the OpenVINO IR but got `export=True`, the model will be converted to OpenVINO once again. "
                        "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
                    _export = True
                else:
                    logger.warning(
                        f"No OpenVINO files were found for {model_id}, setting `export=True` to convert the model to the OpenVINO IR. "
                        "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
        except Exception as exception:
            logger.warning(
                f"Could not infer whether the model was already converted or not to the OpenVINO IR, keeping `export={export}`.\n{exception}"
            )

        return super().from_pretrained(
            model_id,
            export=_export,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            subfolder=subfolder,
            config=config,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

    @staticmethod
    def _prepare_quantization_config(
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None, load_in_8bit: bool = False
    ):
        # Give default quantization config if not provided and load_in_8bit=True
        if not quantization_config and load_in_8bit:
            quantization_config = OVWeightQuantizationConfig(bits=8)
        elif isinstance(quantization_config, dict):
            quantization_config = OVConfig.quantization_config_from_dict(quantization_config)

        return quantization_config

    def _set_ov_config_parameters(self):
        if self.ov_config.get("PERFORMANCE_HINT") is None:
            self.ov_config["PERFORMANCE_HINT"] = "LATENCY"

        q_config = self._openvino_config.quantization_config if self._openvino_config else None
        if isinstance(q_config, OVDynamicQuantizationConfig):
            self.ov_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = str(q_config.activations_group_size)
            if self.can_generate() and "KV_CACHE_PRECISION" not in self.ov_config:
                self.ov_config["KV_CACHE_PRECISION"] = "u8"

    @staticmethod
    def _cached_file(
        model_path: Union[Path, str],
        token: Optional[Union[bool, str]] = None,
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
            model_cache_path = model_path / subfolder / file_name
        else:
            file_name = Path(file_name)
            if file_name.suffix != ".onnx":
                model_file_names = [file_name.with_suffix(".bin"), file_name]
            else:
                model_file_names = [file_name]
            for file_name in model_file_names:
                model_cache_path = Path(
                    cached_file(
                        model_path.as_posix(),
                        filename=file_name.as_posix(),
                        token=token,
                        revision=revision,
                        force_download=force_download,
                        cache_dir=cache_dir,
                        subfolder=subfolder,
                        local_files_only=local_files_only,
                    )
                )

        return model_cache_path

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
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
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting
        # would end-up removing the directory containing the underlying OpenVINO model
        cls._model_save_dir_tempdirectory_instance = save_dir

        compile_only = kwargs.pop("compile_only", False)
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        variant = kwargs.pop("variant", None)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task or cls.export_feature,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            ov_config=ov_config,
            library_name=cls._library_name,
            variant=variant,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            compile_only=compile_only,
            **kwargs,
        )

    @classmethod
    def _to_load(
        cls,
        model,
        config: PretrainedConfig,
        onnx_config: ExportConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        stateful: bool = False,
        **kwargs,
    ):
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        compile_only = kwargs.pop("compile_only", False)
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False

        # Export the model to the ONNX format
        export(
            model=model,
            config=onnx_config,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output=save_dir_path / OV_XML_FILE_NAME,
            stateful=stateful,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=False,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            compile_only=compile_only,
            **kwargs,
        )

    def compile(self):
        if self.request is None:
            ov_config = {**self.ov_config}
            logger.info(f"Compiling the model to {self._device} ...")
            self.request = self._compile_model(self.model, self._device, ov_config, self.model_save_dir)

    def _reshape(
        self,
        model: openvino.Model,
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
        if self._compile_only:
            raise ValueError(
                "`reshape()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        self.is_dynamic = True if batch_size == -1 and sequence_length == -1 else False
        self.model = self._reshape(self.model, batch_size, sequence_length, height, width)
        self.request = None
        return self

    def half(self):
        """
        Converts all the model weights to FP16
        """
        if self._compile_only:
            raise ValueError(
                "`half()` is not supported with `compile_only=True` mode, to use this option please initialize your model with compile_only=False"
            )
        apply_moc_transformations(self.model, cf=False)
        compress_model_transformation(self.model)
        self.request = None
        return self

    def eval(self):
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        return isinstance(self, GenerationMixin)

    def _inference(self, inputs):
        try:
            outputs = self.request(inputs)
        except Exception as e:
            invalid_inputs_msg = self._incompatible_inputs_warning(inputs)
            if invalid_inputs_msg is not None:
                e.args += (invalid_inputs_msg,)
            raise e
        return outputs

    def _incompatible_inputs_warning(self, inputs: Dict):
        expected_inputs_names = set(self.input_names.keys())
        inputs_names = set(inputs.keys())

        if expected_inputs_names != inputs_names:
            return f"Got unexpected inputs: expecting the following inputs {expected_inputs_names} but got {inputs_names}."

        for input_name in inputs:
            if inputs[input_name] is None:
                dtype = self.request.inputs[self.input_names[input_name]].get_element_type()
                return f"Got unexpected inputs: `{input_name}` set to {type(inputs[input_name])} while expected to be {dtype}."

        return None


class OVModelPart:
    def __init__(
        self,
        model: Model,
        parent_model: OVBaseModel,
        ov_config: Optional[Dict[str, str]] = None,
        model_name: str = "encoder",
        model_dir: str = None,
    ):
        self.model = model
        self.parent_model = parent_model
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtype = {
            inputs.get_any_name(): OV_TO_PT_TYPE[inputs.get_element_type().get_type_name()]
            for inputs in self.model.inputs
        }
        self.ov_config = ov_config or {**self.parent_model.ov_config}
        self.request = None if not self.parent_model._compile_only else self.model
        self._model_name = model_name
        self.config = self.parent_model.config
        self._model_dir = Path(model_dir or parent_model._model_save_dir)

    def _compile(self):
        if self.parent_model._compile_only and isinstance(self.model, CompiledModel):
            self.request = self.model
        if self.request is None:
            if (
                "CACHE_DIR" not in self.ov_config.keys()
                and not str(self._model_dir).startswith(gettempdir())
                and "GPU" in self._device
            ):
                self.ov_config["CACHE_DIR"] = os.path.join(self._model_dir, self._model_name, "model_cache")

            logger.info(f"Compiling the {self._model_name} to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, self.ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                _print_compiled_model_properties(self.request)

    @property
    def _device(self) -> str:
        return self.parent_model._device

    @property
    def device(self) -> torch.device:
        return self.parent_model.device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        for dtype in self.input_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def clear_requests(self):
        self.request = None
