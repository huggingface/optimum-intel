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

import collections.abc
import copy
import inspect
import logging
import os
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import nncf
import openvino
import torch
import transformers
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from nncf import CompressWeightsMode, SensitivityMetric
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters, OverflowFix
from nncf.torch import register_module
from nncf.torch.initialization import PTInitializingDataLoader
from openvino._offline_transformations import compress_quantize_weights_transformation
from openvino.runtime import Core, Tensor
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, DataCollator, PreTrainedModel, default_data_collator
from transformers.pytorch_utils import Conv1D
from transformers.utils import is_accelerate_available

from optimum.exporters.onnx.convert import check_dummy_inputs_are_allowed
from optimum.exporters.tasks import TasksManager
from optimum.quantization_base import OptimumQuantizer

from ...exporters.openvino import export, export_pytorch_via_onnx
from ...exporters.openvino.model_patcher import patch_model_with_bettertransformer
from ...exporters.openvino.stateful import ensure_export_task_support_stateful, ensure_stateful_is_available
from ..utils.constant import _TASK_ALIASES
from ..utils.import_utils import (
    DATASETS_IMPORT_ERROR,
    is_datasets_available,
    is_datasets_version,
    is_diffusers_available,
)
from ..utils.modeling_utils import get_model_device
from .configuration import OVConfig, OVQuantizationConfig, OVQuantizationMethod, OVWeightQuantizationConfig
from .modeling_base import OVBaseModel
from .utils import (
    MAX_ONNX_OPSET,
    MIN_ONNX_QDQ_OPSET,
    ONNX_WEIGHTS_NAME,
    OV_XML_FILE_NAME,
    PREDEFINED_SD_DATASETS,
)


if is_datasets_available():
    from datasets import Dataset

register_module(ignored_algorithms=[])(Conv1D)

core = Core()
logger = logging.getLogger(__name__)


class OVDataLoader(PTInitializingDataLoader):
    def get_inputs(self, dataloader_output) -> Tuple[Tuple, Dict]:
        return (), dataloader_output

    @property
    def batch_size(self):
        batch_size = self._data_loader.batch_size
        if is_accelerate_available():
            from accelerate.data_loader import DataLoaderStateMixin

            if batch_size is None and isinstance(self._data_loader, DataLoaderStateMixin):
                batch_size = self._data_loader.total_batch_size
        return batch_size


class InferRequestWrapper:
    """
    Wrapper class for OV InferRequest or CompiledModel objects that collects inputs which they were called with to
    a list.
    """

    def __init__(
        self,
        request: Union[openvino.InferRequest, openvino.CompiledModel],
        collected_inputs: List = None,
        apply_caching: bool = False,
    ):
        """
        Args:
            request (`Union[openvino.InferRequest, openvino.CompiledModel]`):
                Infer request instance to wrap. May also be an instance of CompiledModel.
            collected_inputs (`List`, *optional*):
                List where collected inputs will be stored. If None, an empty list will be created
                at self.collected_inputs.
            apply_caching (`bool`, defaults to False):
                Whether to apply data caching. May improve memory footprint, but results in slight performance overhead
                due to tensor hash computation.
        """
        self.request = request
        self.collected_inputs = [] if collected_inputs is None else collected_inputs
        self.apply_caching = apply_caching
        self.tensor_cache = {}

    def collect_inputs(self, inputs):
        if not self.apply_caching or not isinstance(inputs, dict):
            self.collected_inputs.append(copy.deepcopy(inputs))
            return

        copied_inputs = {}
        for k, v in inputs.items():
            data = v
            if isinstance(data, openvino.Tensor):
                data = data.data
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            data_hash = hash(data.tobytes())

            # Avoid data copying if tensor contains data encountered earlier
            if data_hash not in self.tensor_cache:
                self.tensor_cache[data_hash] = copy.deepcopy(v)
            copied_inputs[k] = self.tensor_cache[data_hash]
        self.collected_inputs.append(copied_inputs)

    def __call__(self, *args, **kwargs):
        # If __call__ is invoked then self.request must be an instance of CompiledModel
        signature = inspect.signature(self.request)
        bound_args = signature.bind(*args, **kwargs).arguments
        self.collect_inputs(bound_args["inputs"])
        return self.request(*args, **kwargs)

    def infer(self, inputs: Any = None, share_inputs: bool = False):
        self.collect_inputs(inputs)
        return self.request.infer(inputs, share_inputs)

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
        share_inputs: bool = False,
        *,
        shared_memory: Any = None,
    ):
        self.collect_inputs(inputs)
        self.request.infer(inputs, share_inputs, share_outputs=True)

    def wait(self):
        pass

    def get_tensor(self, name: str):
        return Tensor(self.request.results[name])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)


class OVQuantizer(OptimumQuantizer):
    """
    Handle the NNCF quantization process.
    """

    def __init__(self, model: transformers.PreTrainedModel, task: Optional[str] = None, seed: int = 42, **kwargs):
        """
        Args:
            model (`transformers.PreTrainedModel`):
                The [PreTrainedModel](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel) to quantize.
            task (`str`, defaults to None):
                The task defining the model topology used for the ONNX export.
            seed (`int`, defaults to 42):
                The random seed to use when shuffling the calibration dataset.
        """
        super().__init__()
        self.model = model
        self.task = task
        self.seed = seed
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())

    @property
    def input_names(self):
        logger.warning("The`input_names` attribute is deprecated and will be removed in v1.18.0")
        return None

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        calibration_dataset: Optional[Union["Dataset", nncf.Dataset, Iterable]] = None,
        save_directory: Optional[Union[str, Path]] = None,
        ov_config: OVConfig = None,
        file_name: Optional[str] = None,
        batch_size: int = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = True,
        weights_only: bool = None,
        **kwargs,
    ):
        """
        Quantize a model given the optimization specifications defined in `quantization_config`.

        Args:
            calibration_dataset (`datasets.Dataset` or `nncf.Dataset` or `Iterable`, *optional*):
                A collection of data samples to use for quantization calibration. Is optional for weight-only
                quantization and is required for full quantization.
            save_directory (`Union[str, Path]`, *optional*):
                The directory where the quantized model should be saved.
            ov_config (`OVConfig`, *optional*):
                The configuration containing the parameters related to quantization. If not provided, 8-bit symmetric
                weight-only quantization will be applied.
            file_name (`str`, *optional*):
                The model file name to use when saving the model. Overwrites the default file name `"model.onnx"`.
            batch_size (`int`, defaults to 1):
                The number of calibration samples to load per batch.
            data_collator (`DataCollator`, *optional*):
                The function to use to form a batch from a list of elements of the calibration dataset.
            remove_unused_columns (`bool`, defaults to `True`):
                Whether to remove the columns unused by the model forward method.
            weights_only (`bool`, *optional*):
                Being deprecated.
                Compress weights to integer precision (8-bit by default) while keeping activations
                floating-point. Fits best for LLM footprint reduction and performance acceleration.

        Examples:
        ```python
        >>> from optimum.intel import OVQuantizer, OVModelForCausalLM
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
        >>> quantizer = OVQuantizer.from_pretrained(model, task="text-generation")
        >>> ov_config = OVConfig(quantization_config=OVWeightQuantizationConfig())
        >>> quantizer.quantize(ov_config=ov_config, save_directory="./quantized_model")
        >>> optimized_model = OVModelForCausalLM.from_pretrained("./quantized_model")
        ```

        ```python
        >>> from optimum.intel import OVQuantizer, OVModelForSequenceClassification
        >>> from transformers import AutoModelForSequenceClassification
        >>> model = OVModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", export=True)
        >>> # or
        >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        >>> quantizer = OVQuantizer.from_pretrained(model, task="text-classification")
        >>> ov_config = OVConfig(quantization_config=OVQuantizationConfig())
        >>> quantizer.quantize(calibration_dataset=dataset, ov_config=ov_config, save_directory="./quantized_model")
        >>> optimized_model = OVModelForSequenceClassification.from_pretrained("./quantized_model")
        ```
        """
        # TODO: deprecate weights_only argument
        if weights_only is not None:
            logger.warning(
                "`weights_only` argument is deprecated and will be removed in v1.18.0. In the future please provide `ov_config.quantization_config` "
                "as an instance of `OVWeightQuantizationConfig` for weight-only compression or as an instance of `OVQuantizationConfig` for full model quantization."
            )

        if ov_config is None:
            ov_config = OVConfig()
        if not isinstance(ov_config, OVConfig):
            raise TypeError(f"`ov_config` should be an `OVConfig`, but got: {type(ov_config)} instead.")
        quantization_config = ov_config.quantization_config
        if quantization_config is None:
            if (weights_only is None or weights_only is True) and calibration_dataset is None:
                if weights_only is None:
                    logger.info(
                        "`quantization_config` was not provided, 8-bit asymmetric weight quantization will be applied."
                    )
                ov_config.quantization_config = OVWeightQuantizationConfig(bits=8)
            else:
                logger.warning(
                    "`quantization_config` was not provided, but calibration dataset was provided, assuming full "
                    "model quantization is intended. In the future, please provide `quantization_config` as an "
                    "instance of OVQuantizationConfig."
                )
                ov_config.quantization_config = OVQuantizationConfig()

        if isinstance(self.model, OVBaseModel):
            self._quantize_ovbasemodel(
                ov_config,
                save_directory,
                calibration_dataset,
                batch_size,
                data_collator,
                remove_unused_columns,
                **kwargs,
            )

        elif isinstance(self.model, torch.nn.Module):
            logger.warning(
                "The support of `torch.nn.Module` will be deprecated in a future release of optimum-intel, please use the corresponding `OVModelForXxx` class to load you model."
                "To convert a PyTorch model to OpenVINO, you can set `export=True` when loading your model as `OVModelForXxx.from_pretrained(..., export=True)`"
            )
            self._quantize_torchmodel(
                ov_config,
                save_directory,
                calibration_dataset,
                file_name,
                batch_size,
                data_collator,
                remove_unused_columns,
                **kwargs,
            )
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

    def _quantize_ovbasemodel(
        self,
        ov_config: OVConfig,
        save_directory: Union[str, Path] = None,
        calibration_dataset: Optional[Union["Dataset", nncf.Dataset, Iterable]] = None,
        batch_size: int = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = True,
        **kwargs,
    ):
        if is_diffusers_available():
            from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionPipelineBase

        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
        quantization_config = ov_config.quantization_config

        if calibration_dataset is not None:
            # Process custom calibration dataset

            if is_diffusers_available() and isinstance(self.model, OVStableDiffusionPipelineBase):
                calibration_dataset = self._prepare_unet_dataset(
                    quantization_config.num_samples, dataset=calibration_dataset
                )
            elif is_datasets_available() and isinstance(calibration_dataset, Dataset):
                calibration_dataloader = self._get_calibration_dataloader(
                    calibration_dataset=calibration_dataset,
                    batch_size=batch_size,
                    remove_unused_columns=remove_unused_columns,
                    data_collator=data_collator,
                )
                if self.model.export_feature == "text-generation" and self.model.use_cache:
                    calibration_dataset = self._prepare_text_generation_dataset(
                        quantization_config, calibration_dataloader
                    )
                else:
                    calibration_dataset = nncf.Dataset(calibration_dataloader)
            elif isinstance(calibration_dataset, collections.abc.Iterable):
                calibration_dataset = nncf.Dataset(calibration_dataset)
            elif not isinstance(calibration_dataset, nncf.Dataset):
                raise ValueError(
                    "`calibration_dataset` must be either an `Iterable` object or an instance of "
                    f"`nncf.Dataset` or `datasets.Dataset`. Found: {type(calibration_dataset)}."
                )

        if isinstance(quantization_config, OVWeightQuantizationConfig):
            if quantization_config.dataset is not None and calibration_dataset is not None:
                logger.info(
                    "Both `quantization_config.dataset` and `calibration_dataset` were provided for weight only "
                    "quantization. Will rely on `calibration_dataset`."
                )

            if calibration_dataset is None and isinstance(quantization_config.dataset, str):
                from optimum.intel import OVModelForCausalLM

                if isinstance(self.model, OVModelForCausalLM):
                    calibration_dataset = self._prepare_builtin_dataset(quantization_config)
                elif is_diffusers_available() and isinstance(self.model, OVStableDiffusionPipelineBase):
                    calibration_dataset = self._prepare_unet_dataset(
                        quantization_config.num_samples, dataset_name=quantization_config.dataset
                    )
                else:
                    raise ValueError(
                        f"Can't create weight compression calibration dataset from string for {type(self.model)}"
                    )

            if quantization_config.quant_method == OVQuantizationMethod.HYBRID:
                if calibration_dataset is None:
                    raise ValueError("Calibration dataset is required to run hybrid quantization.")
                if is_diffusers_available() and isinstance(self.model, OVStableDiffusionPipelineBase):
                    # Apply weight-only quantization to all SD submodels except UNet
                    quantization_config_copy = copy.deepcopy(quantization_config)
                    quantization_config_copy.dataset = None
                    quantization_config_copy.quant_method = OVQuantizationMethod.DEFAULT
                    for sd_submodel_name in ["vae_encoder", "vae_decoder", "text_encoder", "text_encoder_2"]:
                        sd_submodel = getattr(self.model, sd_submodel_name)
                        if sd_submodel is not None:
                            _weight_only_quantization(sd_submodel.model, quantization_config_copy)

                    # Apply hybrid quantization to UNet
                    self.model.unet.model = _hybrid_quantization(
                        self.model.unet.model, quantization_config, calibration_dataset
                    )
                else:
                    # The model may be for example OVModelForImageClassification, OVModelForAudioClassification, etc.
                    self.model.model = _hybrid_quantization(self.model.model, quantization_config, calibration_dataset)
            else:
                _weight_only_quantization(self.model.model, quantization_config, calibration_dataset)
            if save_directory is not None:
                self.model.save_pretrained(save_directory)
                ov_config.save_pretrained(save_directory)
            return

        if not isinstance(quantization_config, OVQuantizationConfig):
            raise ValueError(f"Unsupported type of quantization config: {type(quantization_config)}")

        if calibration_dataset is None:
            raise ValueError("Calibration dataset is required to run quantization.")

        # Actual model quantization
        quantized_model = nncf.quantize(
            self.model.model,
            calibration_dataset,
            subset_size=quantization_config.num_samples,
            ignored_scope=quantization_config.get_ignored_scope_instance(),
            model_type=nncf.ModelType(quantization_config.model_type),
            preset=nncf.QuantizationPreset.PERFORMANCE if quantization_config.sym else nncf.QuantizationPreset.MIXED,
            fast_bias_correction=quantization_config.fast_bias_correction,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                overflow_fix=OverflowFix(quantization_config.overflow_fix)
            ),
            **kwargs,
        )

        self.model.model = quantized_model
        if save_directory is not None:
            self.model.save_pretrained(save_directory)
            ov_config.save_pretrained(save_directory)

    def _quantize_torchmodel(
        self,
        ov_config: OVConfig,
        save_directory: Union[str, Path],
        calibration_dataset: Optional[Union["Dataset", nncf.Dataset, Iterable]] = None,
        file_name: Optional[str] = None,
        batch_size: int = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = True,
        **kwargs,
    ):
        if save_directory is None:
            # TODO : can be set to self.model.config.name_or_path for OVModels when not provided
            raise ValueError("`save_directory` needs to be specified")

        self._set_task()
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        ov_file_name = file_name if file_name is not None else OV_XML_FILE_NAME
        output_path = save_directory.joinpath(ov_file_name)
        output_path = output_path.with_suffix(".xml").as_posix()

        model_type = self.model.config.model_type.replace("_", "-")
        onnx_config_class = TasksManager.get_exporter_config_constructor(
            exporter="openvino",
            model=self.model,
            task=self.task,
            model_type=model_type,
        )

        save_onnx_model = ov_config.save_onnx_model
        onnx_file_name = (
            ONNX_WEIGHTS_NAME if file_name is None and save_onnx_model else Path(ov_file_name).with_suffix(".onnx")
        )

        task = self.task
        model = self.model
        self.model.config.save_pretrained(save_directory)
        if task.startswith("text-generation"):
            onnx_config = onnx_config_class(
                model.config, use_past=model.config.use_cache, use_past_in_inputs=model.config.use_cache
            )
            if model.config.use_cache:
                task = "text-generation-with-past"
        else:
            onnx_config = onnx_config_class(model.config)

        stateful = ensure_stateful_is_available() and ensure_export_task_support_stateful(task)

        quantization_config = ov_config.quantization_config
        if isinstance(quantization_config, OVWeightQuantizationConfig):
            if stateful:
                # patch model before weight compression
                model = patch_model_with_bettertransformer(model)

            dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
            device = get_model_device(model)
            dummy_inputs = tree_map(
                lambda value: value.to(device) if isinstance(value, torch.Tensor) else value, dummy_inputs
            )
            check_dummy_inputs_are_allowed(model, dummy_inputs)

            nncf.compress_weights(model, dataset=nncf.Dataset([dummy_inputs]))
        else:
            if not isinstance(quantization_config, OVQuantizationConfig):
                raise ValueError(f"Unsupported type of quantization config: {type(quantization_config)}")
            if stateful:
                logger.warn(
                    "Quantization algorithm does not support optimized stateful models. "
                    "The original model without optimization will be quantized and exported."
                )
                stateful = False

            if isinstance(calibration_dataset, nncf.Dataset):
                quantization_dataset = calibration_dataset
            elif isinstance(calibration_dataset, Dataset):
                calibration_dataloader = self._get_calibration_dataloader(
                    calibration_dataset=calibration_dataset,
                    batch_size=batch_size,
                    remove_unused_columns=remove_unused_columns,
                    data_collator=data_collator,
                )
                quantization_dataset = nncf.Dataset(calibration_dataloader)
            else:
                if calibration_dataset is None:
                    raise ValueError("Calibration dataset is required to run quantization.")
                quantization_dataset = nncf.Dataset(calibration_dataset)
            model = nncf.quantize(
                model,
                quantization_dataset,
                subset_size=quantization_config.num_samples,
                ignored_scope=quantization_config.get_ignored_scope_instance(),
                model_type=nncf.ModelType(quantization_config.model_type),
                preset=(
                    nncf.QuantizationPreset.PERFORMANCE if quantization_config.sym else nncf.QuantizationPreset.MIXED
                ),
                fast_bias_correction=quantization_config.fast_bias_correction,
                advanced_parameters=nncf.AdvancedQuantizationParameters(
                    overflow_fix=OverflowFix(quantization_config.overflow_fix)
                ),
                **kwargs,
            )

        model_path = save_directory / (onnx_file_name if save_onnx_model else ov_file_name)
        onnx_path = save_directory / onnx_file_name
        export_fn = export if not save_onnx_model else export_pytorch_via_onnx
        opset = min(onnx_config.DEFAULT_ONNX_OPSET, MAX_ONNX_OPSET)
        opset = max(opset, MIN_ONNX_QDQ_OPSET)
        export_kwargs = {}
        if not save_onnx_model:
            export_kwargs = {"stateful": stateful}

        _, _, is_onnx = export_fn(model=model, config=onnx_config, output=model_path, opset=opset, **export_kwargs)
        if is_onnx:
            # Load and save the compressed model
            model = core.read_model(onnx_path)
            # Model required second saving for appling weights compression transformations
            self._save_pretrained(model, output_path)
            # if onnx conversion happens as fallback for pytorch conversion, remove onnx model
            if not save_onnx_model:
                os.remove(onnx_path)
                try:
                    os.remove(f"{onnx_path}_data")
                except FileNotFoundError:
                    pass

        ov_config.save_pretrained(save_directory)

    @staticmethod
    def _save_pretrained(model: openvino.runtime.Model, output_path: str):
        compress_quantize_weights_transformation(model)
        openvino.save_model(model, output_path, compress_to_fp16=False)

    def _set_task(self):
        if self.task is None:
            self.task = TasksManager.infer_task_from_model(self.model.config._name_or_path)
            if self.task is None:
                raise ValueError(
                    "The task defining the model topology could not be extracted and needs to be specified for the ONNX export."
                )

        self.task = _TASK_ALIASES.get(self.task, self.task)

        if self.task == "text2text-generation":
            raise ValueError("Seq2Seq models are currently not supported for post-training static quantization.")

        if self.task == "image-to-text":
            raise ValueError("Image2Text models are currently not supported for post-training static quantization.")

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "train",
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        trust_remote_code: bool = False,
    ) -> "Dataset":
        """
        Create the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                in generic formats and optionally a dataset script, if it requires some code to read the data files.
            num_samples (`int`, defaults to 100):
                The maximum number of samples composing the calibration dataset.
            dataset_config_name (`str`, *optional*):
                The name of the dataset configuration.
            dataset_split (`str`, defaults to `"train"`):
                Which split of the dataset to use to perform the calibration step.
            preprocess_function (`Callable`, *optional*):
                Processing function to apply to each example after loading dataset.
            preprocess_batch (`bool`, defaults to `True`):
                Whether the `preprocess_function` should be batched.
            use_auth_token (Optional[Union[bool, str]], defaults to `None`):
                Deprecated. Please use `token` instead.
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            cache_dir (`str`, *optional*):
                Caching directory for a calibration dataset.
            trust_remote_code (`bool`, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if not is_datasets_available():
            raise ValueError(DATASETS_IMPORT_ERROR.format("OVQuantizer.get_calibration_dataset"))

        from datasets import load_dataset

        datasets_kwargs = {"name": dataset_config_name, "split": dataset_split, "token": token, "cache_dir": cache_dir}
        if is_datasets_version(">=", "2.20.0"):
            datasets_kwargs["trust_remote_code"] = trust_remote_code

        calibration_dataset = load_dataset(dataset_name, **datasets_kwargs)

        if num_samples is not None:
            num_samples = min(num_samples, len(calibration_dataset))
            calibration_dataset = calibration_dataset.shuffle(seed=self.seed).select(range(num_samples))

        if preprocess_function is not None:
            calibration_dataset = calibration_dataset.map(preprocess_function, batched=preprocess_batch)

        return calibration_dataset

    def _get_calibration_dataloader(
        self,
        calibration_dataset: "Dataset",
        batch_size: int,
        remove_unused_columns: bool,
        data_collator: Optional[DataCollator] = None,
    ) -> OVDataLoader:
        data_collator = data_collator if data_collator is not None else default_data_collator

        if not is_datasets_available() or not isinstance(calibration_dataset, Dataset):
            logger.warning(
                "`remove_unused_columns` set to `False` as calibration_dataset is not an instance of `datasets.Dataset`"
            )
            remove_unused_columns = False

        if remove_unused_columns:
            calibration_dataset = self._remove_unused_columns(calibration_dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        sampler = RandomSampler(calibration_dataset, generator=generator)
        calibration_dataloader = DataLoader(
            calibration_dataset, batch_size=batch_size, sampler=sampler, collate_fn=data_collator, drop_last=False
        )
        return OVDataLoader(calibration_dataloader)

    def _remove_unused_columns(self, dataset: "Dataset"):
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)

    def _prepare_builtin_dataset(self, quantization_config: OVWeightQuantizationConfig):
        from optimum.gptq.data import get_dataset, prepare_dataset

        tokenizer = AutoTokenizer.from_pretrained(quantization_config.tokenizer)
        nsamples = quantization_config.num_samples if quantization_config.num_samples else 128
        calibration_dataset = get_dataset(quantization_config.dataset, tokenizer, seqlen=32, nsamples=nsamples)
        calibration_dataset = prepare_dataset(calibration_dataset)
        calibration_dataset = nncf.Dataset(calibration_dataset, lambda x: self.model.prepare_inputs(**x))

        return calibration_dataset

    def _prepare_text_generation_dataset(
        self, quantization_config: OVQuantizationConfig, calibration_dataloader: OVDataLoader
    ) -> nncf.Dataset:
        # Prefetch past_key_values
        self.model.update_pkv_precision(True)
        self.model.compile()
        collected_inputs = []

        num_samples = quantization_config.num_samples or 200

        self.model.request = InferRequestWrapper(self.model.request, collected_inputs)
        try:
            for data in calibration_dataloader:
                self.model.generate(**data, max_new_tokens=1)
                if len(collected_inputs) >= num_samples:
                    break
        finally:
            self.model.request = self.model.request.request
        calibration_dataset = nncf.Dataset(collected_inputs)

        return calibration_dataset

    def _prepare_unet_dataset(
        self,
        num_samples: Optional[int] = None,
        dataset_name: Optional[str] = None,
        dataset: Optional[Union[Iterable, "Dataset"]] = None,
    ) -> nncf.Dataset:
        self.model.compile()

        size = self.model.unet.config.get("sample_size", 64) * self.model.vae_scale_factor
        height, width = 2 * (min(size, 512),)
        num_samples = num_samples or 200

        if dataset is not None:
            if isinstance(dataset, nncf.Dataset):
                return dataset
            if is_datasets_available() and isinstance(dataset, Dataset):
                dataset = dataset.select_columns(["caption"])

            def transform_fn(data_item):
                return data_item if isinstance(data_item, (list, dict)) else [data_item]

        elif isinstance(dataset_name, str):
            available_datasets = PREDEFINED_SD_DATASETS.keys()
            if dataset_name not in available_datasets:
                raise ValueError(
                    f"""You have entered a string value for dataset. You can only choose between
                    {list(available_datasets)}, but the {dataset_name} was found"""
                )

            from datasets import load_dataset

            dataset_metadata = PREDEFINED_SD_DATASETS[dataset_name]
            datasets_kwargs = {"split": dataset_metadata["split"], "streaming": True}
            dataset = load_dataset(dataset_name, **datasets_kwargs).shuffle(seed=self.seed)

            input_names = dataset_metadata["inputs"]
            dataset = dataset.select_columns(list(input_names.values()))

            def transform_fn(data_item):
                return {inp_name: data_item[column] for inp_name, column in input_names.items()}

        else:
            raise ValueError(
                "For UNet inputs collection either quantization_config.dataset or custom "
                "calibration_dataset must be provided."
            )

        calibration_data = []
        try:
            self.model.unet.request = InferRequestWrapper(self.model.unet.request, calibration_data)

            for inputs in dataset:
                inputs = transform_fn(inputs)
                if isinstance(inputs, dict):
                    self.model(**inputs, height=height, width=width)
                else:
                    self.model(*inputs, height=height, width=width)
                if len(calibration_data) >= num_samples:
                    break
        finally:
            self.model.unet.request = self.model.unet.request.request

        calibration_dataset = nncf.Dataset(calibration_data[:num_samples])
        return calibration_dataset


def _weight_only_quantization(
    model: openvino.runtime.Model,
    quantization_config: Union[OVWeightQuantizationConfig, Dict],
    calibration_dataset: Optional[Union[nncf.Dataset, Iterable]] = None,
) -> openvino.runtime.Model:
    config = quantization_config
    if isinstance(config, dict):
        config = OVWeightQuantizationConfig.from_dict(quantization_config)

    dataset = None
    if calibration_dataset is not None:
        if is_datasets_available() and isinstance(calibration_dataset, Dataset):
            raise ValueError(
                "Providing calibration dataset as an instance of `datasets.Dataset` for OV weight-only "
                "quantization is not supported. Please provide it as `nncf.Dataset` or as iterable of "
                "model inputs."
            )
        elif isinstance(calibration_dataset, nncf.Dataset):
            dataset = calibration_dataset
        else:
            dataset = nncf.Dataset(calibration_dataset)

    sensitivity_metric = None
    if isinstance(config.sensitivity_metric, str):
        sensitivity_metric = getattr(SensitivityMetric, config.sensitivity_metric.upper())

    if config.bits == 8:
        mode = CompressWeightsMode.INT8_SYM if config.sym else CompressWeightsMode.INT8_ASYM
    else:
        mode = CompressWeightsMode.INT4_SYM if config.sym else CompressWeightsMode.INT4_ASYM

    return nncf.compress_weights(
        model,
        mode=mode,
        ratio=config.ratio,
        group_size=config.group_size,
        all_layers=config.all_layers,
        sensitivity_metric=sensitivity_metric,
        awq=getattr(config.quant_method, "name", "") == "AWQ" or None,
        ignored_scope=config.get_ignored_scope_instance(),
        dataset=dataset,
        subset_size=config.num_samples if config.num_samples else 128,
        scale_estimation=config.scale_estimation,
    )


def _get_operation_const_op(operation, const_port_id: int):
    node = operation.input_value(const_port_id).get_node()
    queue = deque([node])
    constant_node = None
    allowed_propagation_types_list = ["Convert", "FakeQuantize", "Reshape"]

    while len(queue) != 0:
        curr_node = queue.popleft()
        if curr_node.get_type_name() == "Constant":
            constant_node = curr_node
            break
        if len(curr_node.inputs()) == 0:
            break
        if curr_node.get_type_name() in allowed_propagation_types_list:
            queue.append(curr_node.input_value(0).get_node())

    return constant_node


def _is_embedding(node) -> bool:
    allowed_types_list = ["f16", "f32", "f64"]
    const_port_id = 0
    input_tensor = node.input_value(const_port_id)
    if input_tensor.get_element_type().get_type_name() in allowed_types_list:
        const_node = _get_operation_const_op(node, const_port_id)
        if const_node is not None:
            return True

    return False


def _collect_ops_with_weights(model):
    ops_with_weights = []
    for op in model.get_ops():
        if op.get_type_name() == "MatMul":
            constant_node_0 = _get_operation_const_op(op, const_port_id=0)
            constant_node_1 = _get_operation_const_op(op, const_port_id=1)
            if constant_node_0 or constant_node_1:
                ops_with_weights.append(op.get_friendly_name())
        if op.get_type_name() == "Gather" and _is_embedding(op):
            ops_with_weights.append(op.get_friendly_name())

    return ops_with_weights


def _hybrid_quantization(
    model: openvino.runtime.Model, quantization_config: OVWeightQuantizationConfig, dataset: nncf.Dataset
) -> openvino.runtime.Model:
    """
    Quantize a model in hybrid mode with NNCF which means that we quantize:
    weights of MatMul and Embedding layers and activations of other layers.
    The optimization specifications defined in `quantization_config`.

    Args:
        model (`openvino.runtime.Model`):
            The OpenVINO Runtime model for applying hybrid quantization.
        quantization_config (`OVWeightQuantizationConfig`):
            The configuration containing the parameters related to quantization.
        dataset (`nncf.Dataset`):
            The dataset used for hybrid quantization.
    Returns:
        The OpenVINO Runtime model with applied hybrid quantization.
    """
    ops_to_compress = _collect_ops_with_weights(model)

    wc_config = copy.deepcopy(quantization_config)
    wc_config.ignored_scope = wc_config.ignored_scope or {}
    wc_config.ignored_scope["types"] = wc_config.ignored_scope.get("types", []) + ["Convolution"]
    compressed_model = _weight_only_quantization(model, wc_config)

    ptq_ignored_scope = quantization_config.get_ignored_scope_instance()
    ptq_ignored_scope.names += ops_to_compress
    subset_size = quantization_config.num_samples if quantization_config.num_samples else 200
    quantized_model = nncf.quantize(
        model=compressed_model,
        calibration_dataset=dataset,
        model_type=nncf.ModelType.TRANSFORMER,
        ignored_scope=ptq_ignored_scope,
        # SQ algo should be disabled for MatMul nodes because their weights are already compressed
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=-1)
        ),
        subset_size=subset_size,
    )
    return quantized_model
