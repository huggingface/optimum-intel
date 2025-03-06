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
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import nncf
import openvino
import torch
import transformers
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.torch import register_module
from openvino._offline_transformations import compress_quantize_weights_transformation
from openvino.runtime import Core
from torch.utils._pytree import tree_map
from transformers import DataCollator, PreTrainedModel
from transformers.pytorch_utils import Conv1D

from optimum.exporters.tasks import TasksManager
from optimum.quantization_base import OptimumQuantizer

from ....exporters.openvino import export, export_pytorch_via_onnx
from ....exporters.openvino.model_patcher import patch_model_with_bettertransformer
from ....exporters.openvino.stateful import ensure_export_task_support_stateful, ensure_stateful_is_available
from ...utils.constant import _TASK_ALIASES
from ...utils.import_utils import (
    is_datasets_available,
    is_diffusers_available,
)
from ...utils.modeling_utils import get_model_device
from ..utils import (
    MAX_ONNX_OPSET,
    MIN_ONNX_QDQ_OPSET,
    ONNX_WEIGHTS_NAME,
    OV_XML_FILE_NAME,
)
from .calibration_dataset_builder import OVCalibrationDatasetBuilder
from .configuration import (
    OVConfig,
    OVMixedQuantizationConfig,
    OVQuantizationConfig,
    OVQuantizationMethod,
    OVWeightQuantizationConfig,
)


if is_datasets_available():
    from datasets import Dataset

register_module(ignored_algorithms=[])(Conv1D)

core = Core()
logger = logging.getLogger(__name__)


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
        self.dataset_builder = OVCalibrationDatasetBuilder(model, seed)

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        calibration_dataset: Optional[Union["Dataset", nncf.Dataset, Dict[str, nncf.Dataset], List]] = None,
        save_directory: Optional[Union[str, Path]] = None,
        ov_config: OVConfig = None,
        file_name: Optional[str] = None,
        batch_size: int = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
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
            remove_unused_columns (`bool`, defaults to `False`):
                Whether to remove the columns unused by the model forward method.

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
        if remove_unused_columns:
            logger.warning("`remove_unused_columns` is deprecated and will be removed in optimum-intel v1.24.")

        if isinstance(calibration_dataset, list):
            logger.warning(
                "Providing calibration dataset as a list is deprecated and will be removed in optimum-intel v1.24. "
                "Please provide it as `datasets.Dataset` or as dictionary of `nncf.Dataset` instances."
            )

        if ov_config is None:
            ov_config = OVConfig()
        if not isinstance(ov_config, OVConfig):
            raise TypeError(f"`ov_config` should be an `OVConfig`, but got: {type(ov_config)} instead.")
        if ov_config.quantization_config is None:
            logger.warning(
                "`quantization_config` was not provided. In the future, please provide `quantization_config`"
            )
            if calibration_dataset is None:
                logger.warning("Calibration dataset was not provided, assuming weight only quantization.")
                ov_config.quantization_config = OVWeightQuantizationConfig(bits=8)
            else:
                logger.warning("Calibration dataset was provided, assuming static quantization.")
                ov_config.quantization_config = OVQuantizationConfig()

        quantization_config = ov_config.quantization_config
        if quantization_config.dataset is not None and calibration_dataset is not None:
            logger.info(
                "Both `quantization_config.dataset` and `calibration_dataset` were provided for weight only "
                "quantization. Will rely on `calibration_dataset`."
            )

        if is_diffusers_available():
            from optimum.intel.openvino.modeling_diffusion import OVDiffusionPipeline

        if calibration_dataset is not None and isinstance(calibration_dataset, nncf.Dataset):
            logger.info("Assuming nncf calibration dataset is provided for a pipeline component named `model`.")
            calibration_dataset = {"model": calibration_dataset}
        if calibration_dataset is not None and not isinstance(calibration_dataset, dict):
            # Process custom calibration dataset
            if (
                is_diffusers_available()
                and isinstance(self.model, OVDiffusionPipeline)
                and is_datasets_available()
                and isinstance(calibration_dataset, Dataset)
                and "caption" in calibration_dataset.column_names
            ):
                logger.warning(
                    "Assuming `caption` column should be used for calibration. This behavior will be deprecated in "
                    "optimum-intel v1.24. Please filter the required columns before passing the dataset."
                )
                calibration_dataset = calibration_dataset.select_columns(["caption"])

            if (
                is_diffusers_available()
                and isinstance(self.model, OVDiffusionPipeline)
                and isinstance(calibration_dataset, list)
                and all(isinstance(it, str) for it in calibration_dataset)
            ):
                # To be deprecated
                if quantization_config.dataset is not None:
                    raise ValueError(
                        "Both `calibration_dataset` and `quantization_config.dataset` are provided and the latter is "
                        "a list of strings. This behavior is ambiguous."
                    )
                logger.warning(
                    "Providing calibration dataset for diffusion models a list of string will be deprecated "
                    "in optimum-intel v1.24. Please provide the list inside `quantization_config.dataset`"
                    "property instead."
                )
                quantization_config.dataset = calibration_dataset

            calibration_dataset = self.dataset_builder.build_from_dataset(
                quantization_config, calibration_dataset, batch_size, data_collator, remove_unused_columns
            )

        from ..modeling_base import OVBaseModel

        if isinstance(self.model, OVBaseModel):
            if self.model._compile_only:
                raise ValueError(
                    "Quantization for `compile_only` model is not supported. Please load model with `compile_only=False`"
                )
            self._quantize_ovbasemodel(
                ov_config,
                save_directory,
                calibration_dataset,
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
                **kwargs,
            )
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

    def _quantize_ovbasemodel(
        self,
        ov_config: OVConfig,
        save_directory: Union[str, Path] = None,
        calibration_datasets: Optional[Dict[str, nncf.Dataset]] = None,
        **kwargs,
    ):
        from optimum.intel.openvino.modeling_seq2seq import _OVModelForWhisper
        from optimum.intel.openvino.modeling_visual_language import OVModelForVisualCausalLM

        if is_diffusers_available():
            from optimum.intel.openvino.modeling_diffusion import OVDiffusionPipeline

        quantization_config = ov_config.quantization_config
        if calibration_datasets is None and quantization_config.dataset is not None:
            calibration_datasets = self.dataset_builder.build_from_quantization_config(quantization_config)

        if calibration_datasets is not None:
            if not isinstance(calibration_datasets, dict) or any(
                not isinstance(v, nncf.Dataset) for v in calibration_datasets.values()
            ):
                raise ValueError("`calibration_datasets` should be a dictionary of `nncf.Dataset` instances.")
            if len(calibration_datasets) == 0:
                raise ValueError("`calibration_datasets` should contain at least one element.")

        if (
            isinstance(quantization_config, OVWeightQuantizationConfig)
            and quantization_config.quant_method != OVQuantizationMethod.HYBRID
        ):
            #
            # Regular (non-hybrid) weight-only quantization
            #
            if is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
                for submodel in self.model.ov_submodels.values():
                    _weight_only_quantization(submodel, quantization_config, **kwargs)
            elif isinstance(self.model, OVModelForVisualCausalLM):
                for submodel_name, submodel in self.model.ov_submodels.items():
                    if submodel_name == "lm_model":
                        calibration_dataset = calibration_datasets.get("lm_model") if calibration_datasets else None
                        _weight_only_quantization(submodel, quantization_config, calibration_dataset, **kwargs)
                    else:
                        _weight_only_quantization(submodel, OVWeightQuantizationConfig(bits=8, sym=True), **kwargs)
            else:
                calibration_dataset = calibration_datasets.get("model") if calibration_datasets else None
                _weight_only_quantization(self.model.model, quantization_config, calibration_dataset, **kwargs)
        else:
            #
            # Some type of data-aware quantization
            #

            if calibration_datasets is None:
                raise ValueError("Calibration dataset is required to run data-aware quantization.")
            if (
                not (
                    is_diffusers_available()
                    and isinstance(self.model, OVDiffusionPipeline)
                    or isinstance(self.model, _OVModelForWhisper)
                )
                and "model" not in calibration_datasets
            ):
                raise RuntimeError("Calibration datasets should contain a key 'model' with a dataset.")

            if (
                isinstance(quantization_config, OVWeightQuantizationConfig)
                and quantization_config.quant_method == OVQuantizationMethod.HYBRID
            ):
                #
                # Hybrid quantization
                #
                if is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
                    if len(calibration_datasets) > 1:
                        raise ValueError("Calibration datasets for Diffusion models should contain only one value.")
                    # Apply hybrid quantization to diffusion model
                    diffusion_model_name, calibration_dataset = next(iter(calibration_datasets.items()))
                    diffusion_model = getattr(self.model, diffusion_model_name).model
                    getattr(self.model, diffusion_model_name).model = _hybrid_quantization(
                        diffusion_model, quantization_config, calibration_dataset, **kwargs
                    )

                    # Apply weight-only quantization to all SD submodels except UNet/Transformer
                    quantization_config_copy = quantization_config.clone()
                    quantization_config_copy.dataset = None
                    quantization_config_copy.quant_method = OVQuantizationMethod.DEFAULT
                    sub_models = [v for (k, v) in self.model.ov_submodels.items() if k != diffusion_model_name]
                    for sub_model in sub_models:
                        _weight_only_quantization(sub_model, quantization_config_copy, **kwargs)
                else:
                    # The model may be for example OVModelForImageClassification, OVModelForAudioClassification, etc.
                    self.model.model = _hybrid_quantization(
                        self.model.model, quantization_config, calibration_datasets["model"], **kwargs
                    )
            elif isinstance(quantization_config, OVQuantizationConfig):
                #
                # Full quantization
                #
                if isinstance(self.model, _OVModelForWhisper):
                    _quantize_whisper_model(self.model, quantization_config, calibration_datasets, **kwargs)
                else:
                    self.model.model = _full_quantization(
                        self.model.model, quantization_config, calibration_datasets["model"], **kwargs
                    )
            elif isinstance(quantization_config, OVMixedQuantizationConfig):
                #
                # Mixed quantization
                #
                self.model.model = _mixed_quantization(
                    self.model.model, quantization_config, calibration_datasets["model"], **kwargs
                )
            else:
                raise ValueError(f"Unsupported type of quantization config: {type(quantization_config)}")

        self.model.clear_requests()

        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_directory)
            ov_config.save_pretrained(save_directory)

    def _quantize_torchmodel(
        self,
        ov_config: OVConfig,
        save_directory: Union[str, Path],
        calibration_datasets: Optional[Dict[str, nncf.Dataset]] = None,
        file_name: Optional[str] = None,
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
            from optimum.exporters.utils import check_dummy_inputs_are_allowed

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

            if calibration_datasets is None:
                raise ValueError("Calibration dataset is required to run quantization.")
            if "model" not in calibration_datasets:
                raise RuntimeError("Calibration datasets should contain a key 'model' with a dataset.")
            model = nncf.quantize(
                model,
                calibration_datasets["model"],
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
        num_samples: Optional[int] = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "train",
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        token: Optional[Union[bool, str]] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        trust_remote_code: bool = False,
        streaming: bool = False,
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
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            cache_dir (`str`, *optional*):
                Caching directory for a calibration dataset.
            trust_remote_code (`bool`, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            streaming (`bool`, defaults to `False`):
                Whether to load dataset in streaming mode.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """

        # TODO: consider in the future for this method to return nncf.Datasets from either datasets.Dataset instance or its name as input.
        #  This way OVQuantizer.quantize() will accept fully ready nncf.Dataset instances and `batch_size` and `data_collator` arguments can be removed.
        #  Example usage in such scenario:
        #  ```
        #  calibration_dataset: Dict[str, nncf.Dataset] = ov_quantizer.get_calibration_dataset(ov_config, dataset_name, ..., batch_size, data_collator)
        #  ov_quantizer.quantize(calibration_dataset, ov_config)
        #  ```

        return self.dataset_builder.load_dataset(
            dataset_name,
            num_samples,
            dataset_config_name,
            dataset_split,
            preprocess_function,
            preprocess_batch,
            token,
            cache_dir,
            trust_remote_code,
            streaming,
        )


def _quantize_whisper_model(
    model, quantization_config: OVQuantizationConfig, calibration_dataset: Dict[str, nncf.Dataset], **kwargs
):
    for submodel_name, submodel in model.ov_submodels.items():
        config = quantization_config.clone()
        # quantization_config.num_samples of audio samples result in more actual model inputs
        config.num_samples = calibration_dataset[submodel_name].get_length()
        quantized_model = _full_quantization(submodel, config, calibration_dataset[submodel_name], **kwargs)
        setattr(model, submodel_name, quantized_model)
        getattr(model, "_".join(submodel_name.split("_")[:-1])).model = quantized_model


def _weight_only_quantization(
    model: openvino.runtime.Model,
    quantization_config: Union[OVWeightQuantizationConfig, Dict],
    calibration_dataset: Optional[Union[nncf.Dataset, Iterable]] = None,
    **kwargs,
) -> openvino.runtime.Model:
    _verify_not_optimized(model)
    config = quantization_config
    if isinstance(config, dict):
        config = OVWeightQuantizationConfig.from_dict(quantization_config)

    if not isinstance(config, OVWeightQuantizationConfig):
        raise ValueError(
            f"Expected quantization config to be an instance of `OVWeightQuantizationConfig`, but got {type(config)}."
        )

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

    wc_kwargs = config.to_nncf_dict()

    # Arguments provided in kwargs override the ones from the config
    kwargs_intersection = set(wc_kwargs.keys()) & set(kwargs.keys())
    if kwargs_intersection:
        logger.warning(
            f"The following nncf.compress_weights() arguments from the OVWeightQuantizationConfig will be overridden "
            f"by the ones given in _weight_only_quantization call kwargs: {kwargs_intersection}."
        )
    wc_kwargs.update(kwargs)
    wc_kwargs.pop("weight_only", None)

    compressed_model = nncf.compress_weights(
        model,
        dataset=dataset,
        **wc_kwargs,
    )

    _remove_f16_kv_cache_precision_flag(compressed_model)

    return compressed_model


def _full_quantization(
    model: openvino.runtime.Model,
    quantization_config: OVQuantizationConfig,
    calibration_dataset: nncf.Dataset,
    verify_not_optimized: bool = True,
    **kwargs,
):
    if not isinstance(quantization_config, OVQuantizationConfig):
        raise ValueError(
            f"Expected quantization config to be an instance of `OVQuantizationConfig`, but got {type(quantization_config)}."
        )

    if verify_not_optimized:
        _verify_not_optimized(model)

    q_kwargs = quantization_config.to_nncf_dict()

    # Arguments provided in kwargs override the ones from the config
    kwargs_intersection = set(q_kwargs.keys()) & set(kwargs.keys())
    if kwargs_intersection:
        logger.warning(
            f"The following nncf.quantize() arguments from the OVQuantizationConfig will be overridden "
            f"by the ones given in _full_quantization call kwargs: {kwargs_intersection}."
        )
    q_kwargs.update(kwargs)
    q_kwargs.pop("weight_only", None)

    quantized_model = nncf.quantize(model, calibration_dataset=calibration_dataset, **q_kwargs)

    _remove_f16_kv_cache_precision_flag(quantized_model)

    return quantized_model


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
    model: openvino.runtime.Model, quantization_config: OVWeightQuantizationConfig, dataset: nncf.Dataset, **kwargs
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

    wc_config = quantization_config.clone()
    wc_config.ignored_scope = {}
    if any(op.get_type_name() == "Convolution" for op in model.get_ops()):
        wc_config.ignored_scope["types"] = ["Convolution"]

    q_config_ignored_scope = {"names": _collect_ops_with_weights(model)}
    q_config = OVQuantizationConfig(
        ignored_scope=q_config_ignored_scope,
        num_samples=quantization_config.num_samples or 200,
        smooth_quant_alpha=-1,
        **kwargs,
    )

    mixed_quantization_config = OVMixedQuantizationConfig(
        weight_quantization_config=wc_config,
        full_quantization_config=q_config,
        ignored_scope=quantization_config.ignored_scope,
        **kwargs,
    )

    return _mixed_quantization(model, mixed_quantization_config, dataset, **kwargs)


def _mixed_quantization(
    model: openvino.Model,
    quantization_config: OVMixedQuantizationConfig,
    dataset: nncf.Dataset,
    **kwargs,
) -> openvino.Model:
    """
    Perform mixed precision quantization where we separately quantize:
        (1) weights of weighted layers to the precision given in the `quantization_config.weight_quantization_config`, and
        (2) weights and activations of other possible layers; precision is given in the `quantization_config.full_quantization_config`.

    By default, weights of all weighted layers are quantized in the first step. In the second step activations of
    weighted and non-weighted layers are quantized. If some layers are instructed to be ignored in the first step
    with `weight_quantization_config.ignored_scope` parameter, both weights and activations of these layers are
    quantized to the precision given in the `full_quantization_config`.

    Args:
        model (`openvino.runtime.Model`):
            The OpenVINO Runtime model for applying quantization.
        quantization_config (`OVMixedQuantizationConfig`):
            The configuration containing the parameters related to quantization.
        dataset (`nncf.Dataset`):
            The dataset used for quantization.
    Returns:
        The OpenVINO Runtime model with applied quantization.
    """

    def merge_ignored_scopes(
        ignored_scope_1: Union[Dict[str, List[str]], None], ignored_scope_2: Union[Dict[str, List[str]], None]
    ) -> Dict[str, List[str]]:
        if ignored_scope_1 is None:
            return copy.deepcopy(ignored_scope_2) if ignored_scope_2 is not None else None
        if ignored_scope_2 is None:
            return copy.deepcopy(ignored_scope_1)
        merged_ignored_scope = {}
        for key in set(ignored_scope_1) | set(ignored_scope_2):
            merged_ignored_scope[key] = list(set(ignored_scope_1.get(key, []) + ignored_scope_2.get(key, [])))
        return merged_ignored_scope

    wc_config = quantization_config.weight_quantization_config.clone()
    wc_config.ignored_scope = merge_ignored_scopes(wc_config.ignored_scope, quantization_config.ignored_scope)
    wc_dataset = dataset if wc_config.bits != 8 else None
    compressed_model = _weight_only_quantization(model, wc_config, wc_dataset, **kwargs)

    q_config = quantization_config.full_quantization_config.clone()
    q_config.ignored_scope = merge_ignored_scopes(q_config.ignored_scope, quantization_config.ignored_scope)
    quantized_model = _full_quantization(compressed_model, q_config, dataset, verify_not_optimized=False, **kwargs)

    return quantized_model


def _verify_not_optimized(ov_model):
    message_template = (
        "Cannot apply optimization to the model because it was already optimized with the following config: {}. "
        "To avoid this issue, check that you set load_in_8bit=False or not using quantization_config at export in the .from_pretrained(), "
        "or explicitly specify weight format with --weight_format fp16/fp32 when using CLI."
    )

    rt_info = ov_model.get_rt_info()
    if "nncf" in rt_info:
        model_weight_compression_config = rt_info["nncf"].get("weight_compression", None)
        model_quantization_config = rt_info["nncf"].get("quantization", None)
        if model_weight_compression_config is not None:
            raise RuntimeError(message_template.format(model_weight_compression_config))
        elif model_quantization_config is not None:
            raise RuntimeError(message_template.format(model_quantization_config))


def _remove_f16_kv_cache_precision_flag(model: openvino.Model) -> openvino.Model:
    # Remove the KV cache compression disabling flag from the model
    if model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]):
        prev_rt_info = model.get_rt_info("runtime_options").value
        if prev_rt_info["KV_CACHE_PRECISION"] == "f16":
            prev_rt_info.pop("KV_CACHE_PRECISION")
            model.set_rt_info(prev_rt_info, "runtime_options")
    return model
