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
import inspect
import logging
import types
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from datasets import Dataset, load_dataset
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.model.torch_model import IPEXModel, PyTorchModel
from neural_compressor.quantization import fit
from neural_compressor.transformers import GPTQConfig, RtnConfig
from neural_compressor.transformers.quantization import convert_to_quantized_model, save_low_bit
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    DataCollator,
    PretrainedConfig,
    PreTrainedModel,
    default_data_collator,
)

from optimum.exporters import TasksManager
from optimum.quantization_base import OptimumQuantizer

from ..utils.constant import _TASK_ALIASES, WEIGHTS_NAME
from ..utils.import_utils import (
    _ipex_version,
    _neural_compressor_version,
    is_ipex_version,
    is_neural_compressor_version,
)
from .configuration import INCConfig
from .utils import (
    IPEX_MINIMUM_VERSION,
    NEURAL_COMPRESSOR_MINIMUM_VERSION,
    INCDataLoader,
)


logger = logging.getLogger(__name__)


if is_neural_compressor_version("<", NEURAL_COMPRESSOR_MINIMUM_VERSION):
    raise ImportError(
        f"Found an incompatible version of neural-compressor. Found version {_neural_compressor_version}, "
        f"but only version {NEURAL_COMPRESSOR_MINIMUM_VERSION} or higher is supported."
    )


class INCQuantizationMode(Enum):
    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"
    WEIGHT_ONLY = "post_training_weight_only"


SUPPORTED_QUANT_MODE = {approach.value for approach in INCQuantizationMode}


class INCQuantizer(OptimumQuantizer):
    """
    Handle the Neural Compressor quantization process.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        eval_fn: Optional[Callable[[PreTrainedModel], int]] = None,
        calibration_fn: Optional[Callable[[PreTrainedModel], int]] = None,
        task: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Args:
            model (`torch.nn.Module`):
                The model to quantize.
            eval_fn (`Callable[[PreTrainedModel], int]`, defaults to None):
                The evaluation function to use for the accuracy driven strategy of the quantization process.
                The accuracy driven strategy will be enabled only if `eval_fn` is provided.
            task (`str`, defaults to None):
                The task defining the model topology. Will try to infer it from model if not provided.
            seed (`int`, defaults to 42):
                The random seed to use when shuffling the calibration dataset.
        """
        super().__init__()

        self._original_model = model
        self.eval_fn = eval_fn if eval_fn is not None else lambda model: 1
        self.calibration_fn = calibration_fn
        self.task = task
        self.seed = seed
        signature = inspect.signature(self._original_model.forward)
        self._signature_columns = list(signature.parameters.keys())
        self.input_names = None
        self._quantized_model = None

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        quantization_config: Union["PostTrainingQuantConfig"],
        save_directory: Union[str, Path],
        calibration_dataset: Dataset = None,
        batch_size: int = 8,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = True,
        file_name: str = None,
        **kwargs,
    ):
        """
        Quantize a model given the optimization specifications defined in `quantization_config`.

        Args:
            quantization_config (`Union[PostTrainingQuantConfig]`):
                The configuration containing the parameters related to quantization.
            save_directory (`Union[str, Path]`):
                The directory where the quantized model should be saved.
            calibration_dataset (`datasets.Dataset`, defaults to `None`):
                The dataset to use for the calibration step, needed for post-training static quantization.
            batch_size (`int`, defaults to 8):
                The number of calibration samples to load per batch.
            data_collator (`DataCollator`, defaults to `None`):
                The function to use to form a batch from a list of elements of the calibration dataset.
            remove_unused_columns (`bool`, defaults to `True`):
                Whether or not to remove the columns unused by the model forward method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        calibration_dataloader = None
        default_name = WEIGHTS_NAME
        self._set_task()

        if kwargs.pop("weight_only", None) is not None:
            logger.warning(
                "`weight_only` is deprecated. Use `quantization_config` instead to specify which methodology and quantization pamraters to apply."
            )

        if (
            isinstance(quantization_config, PostTrainingQuantConfig)
            and quantization_config.backend == "ipex"
            and is_ipex_version("<", IPEX_MINIMUM_VERSION)
            and "generation" in self.task
        ):
            raise ImportError(
                f"Found an incompatible version of intel-extension-for-pytorch. Found version {_ipex_version}, "
                f"but only version {IPEX_MINIMUM_VERSION} or higher is supported."
            )

        if INCQuantizationMode(quantization_config.approach) == INCQuantizationMode.STATIC:
            # Since PyTorch fx trace does not really require an example_inputs, only need calibration_dataset or calibration_fn here.
            if calibration_dataset is None and self.calibration_fn is None:
                raise ValueError(
                    "Post-training static quantization needs a calibration dataset or a calibration_function."
                )
            if calibration_dataset is not None:
                quantization_config.calibration_sampling_size = len(calibration_dataset)
                calibration_dataloader = self._get_calibration_dataloader(
                    calibration_dataset=calibration_dataset,
                    batch_size=batch_size,
                    remove_unused_columns=remove_unused_columns,
                    data_collator=data_collator,
                )

        if isinstance(self._original_model.config, PretrainedConfig):
            self._original_model.config.backend = quantization_config.backend

        compressed_model = fit(
            self._original_model,
            conf=quantization_config,
            calib_dataloader=calibration_dataloader,
            eval_func=self.eval_fn,
            calib_func=self.calibration_fn,
        )

        if not hasattr(compressed_model, "_model") or compressed_model._model is None:
            raise RuntimeError("Calling `neural_compressor.fit` returned unexpected results")

        if isinstance(self._original_model.config, PretrainedConfig):
            # If backend is IPEX, then the quantized model is JIT model which will drop the config attribute,
            # so need set config from original_model.
            model_config = copy.deepcopy(self._original_model.config)
            model_config.torch_dtype = "int8"
            if isinstance(compressed_model, IPEXModel):
                model_config.torchscript = True
                model_config.backend = "ipex"
            model_config.save_pretrained(save_directory)

        self._quantized_model = compressed_model._model

        output_path = save_directory.joinpath(file_name or default_name)
        # Save the quantized model
        self._save_pretrained(compressed_model, output_path)
        quantization_config = INCConfig(quantization=quantization_config)
        quantization_config.save_pretrained(save_directory)

    @staticmethod
    def _save_pretrained(model: Union[PyTorchModel, IPEXModel], output_path: str):
        if isinstance(model, IPEXModel):
            model._model.save(output_path)
        else:
            state_dict = model._model.state_dict()
            if hasattr(model, "q_config"):
                state_dict["best_configure"] = model.q_config
            torch.save(state_dict, output_path)

        logger.info(f"Model weights saved to {output_path}")

    def _set_task(self):
        if self.task is None:
            try:
                # using the actual model has better chances of success
                # since using the model path does not work with local models
                self.task = TasksManager.infer_task_from_model(self._original_model)
            except Exception as e:
                self.task = "default"
                logger.warning(
                    f"The task could not be automatically inferred and will be set to {self.task}. "
                    f"Please provide the task argument with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                )

        self.task = _TASK_ALIASES.get(self.task, self.task)

        if self.task == "text2text-generation":
            raise ValueError("Seq2Seq models are currently not supported for post-training static quantization.")

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
    ) -> Dataset:
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

        calibration_dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
            token=token,
        )

        if num_samples is not None:
            num_samples = min(num_samples, len(calibration_dataset))
            calibration_dataset = calibration_dataset.shuffle(seed=self.seed).select(range(num_samples))

        if preprocess_function is not None:
            calibration_dataset = calibration_dataset.map(preprocess_function, batched=preprocess_batch)

        return calibration_dataset

    def _get_calibration_dataloader(
        self,
        calibration_dataset: Dataset,
        batch_size: int,
        remove_unused_columns: bool,
        data_collator: Optional[DataCollator] = None,
        use_label: Optional[bool] = True,
    ) -> INCDataLoader:
        data_collator = data_collator if data_collator is not None else default_data_collator
        if remove_unused_columns:
            calibration_dataset = self._remove_unused_columns(calibration_dataset)
        self.input_names = getattr(calibration_dataset, "column_names", None)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        sampler = RandomSampler(calibration_dataset, generator=generator)
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=False,
        )

        return INCDataLoader.from_pytorch_dataloader(calibration_dataloader, use_label)

    def _remove_unused_columns(self, dataset: Dataset):
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)


def _weight_only_quantization(
    model_class,
    model_id: Union[str, Path],
    quantization_config: Union[RtnConfig, GPTQConfig],
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    force_download: bool = False,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    local_files_only: bool = False,
    subfolder: str = "",
    trust_remote_code: bool = False,
    **kwargs,
):
    device_map = kwargs.get("device_map", None)
    if device_map is None:
        device_map = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    else:
        device_map = device_map.type if isinstance(device_map, torch.device) else device_map

    use_xpu = device_map == torch.device("xpu") or device_map == "xpu"
    if use_xpu and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise AssertionError("There is no xpu device in this system!")

    if is_neural_compressor_version("<=", "3.0"):
        raise AssertionError("Please install neural_compressor > v3.0")
    if is_ipex_version("<", "2.3.1") and use_xpu:
        raise AssertionError("Please install intel_extension_for_pytorch >= v2.3.1.")

    loading_kwargs = {
        "subfolder": subfolder,
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "local_files_only": local_files_only,
        "force_download": force_download,
        "trust_remote_code": trust_remote_code,
    }

    low_cpu_mem_usage = True

    if getattr(quantization_config, "use_layer_wise", False):
        if is_neural_compressor_version(">=", "3.2"):
            from neural_compressor.torch import load_empty_model

            model = load_empty_model(model_id, cls=model_class, **loading_kwargs)
        else:
            raise ValueError("INC version must be >= 3.2 when use_layer_wise is set to True in quantization_config.")
    else:
        model = model_class.from_pretrained(model_id, low_cpu_mem_usage=low_cpu_mem_usage, **loading_kwargs)

    if use_xpu:
        quantization_config.update(**{"device": "xpu"})
        quantization_config.post_init_xpu()
    else:
        quantization_config.post_init_cpu()

    model.config.update({"low_cpu_mem_usage": low_cpu_mem_usage})
    model.eval()

    if (not torch.cuda.is_available() or device_map == "cpu") and model.config.model_type == "chatglm":
        model = model.float()

    model = convert_to_quantized_model(model, quantization_config, device=device_map)
    quantization_config.remove_redundant_parameters()
    model.config.quantization_config = quantization_config
    # add quantization_config and save_low_bit to pretrained model dynamically
    model.device_map = device_map
    model.quantization_config = quantization_config
    model.save_pretrained = types.MethodType(save_low_bit, model)

    return model
