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

import inspect
import io
import logging
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import transformers
from datasets import Dataset, load_dataset
from torch.onnx import export as onnx_export
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollator, PreTrainedModel, default_data_collator

import openvino
from huggingface_hub import HfApi
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.nncf_network import NNCFNetwork
from openvino._offline_transformations import compress_quantize_weights_transformation
from openvino.runtime import Core
from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfig
from optimum.quantization_base import OptimumQuantizer

from .configuration import OVConfig
from .utils import (
    MAX_ONNX_OPSET,
    MAX_ONNX_OPSET_2022_2_0,
    MIN_ONNX_QDQ_OPSET,
    ONNX_WEIGHTS_NAME,
    OV_XML_FILE_NAME,
    use_external_data_format,
)


core = Core()
logger = logging.getLogger(__name__)


class OVDataLoader(PTInitializingDataLoader):
    def get_inputs(self, dataloader_output) -> Tuple[Tuple, Dict]:
        return (), dataloader_output


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
        feature = kwargs.pop("feature", None)
        if feature is not None:
            logger.warning("`feature` is deprecated and will be removed in a future version. Use `task` instead.")
            if task is not None and task != feature:
                logger.warning(
                    f"Both `feature` and `task` were specified. {task} will be used to define the model topology for the model ONNX export."
                )
        self.task = task or feature
        self.seed = seed
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        self._export_input_names = [
            column for column in self._signature_columns if column not in {"label", "labels", "label_ids"}
        ]
        self.input_names = None

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        calibration_dataset: Dataset,
        save_directory: Union[str, Path],
        quantization_config: OVConfig = None,
        file_name: Optional[str] = None,
        batch_size: int = 8,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = True,
    ):
        """
        Quantize a model given the optimization specifications defined in `quantization_config`.

        Args:
            calibration_dataset (`datasets.Dataset`):
                The dataset to use for the calibration step.
            save_directory (`Union[str, Path]`):
                The directory where the quantized model should be saved.
            quantization_config (`OVConfig`, *optional*):
                The configuration containing the parameters related to quantization.
            file_name (`str`, *optional*):
                The model file name to use when saving the model. Overwrites the default file name `"model.onnx"`.
            batch_size (`int`, defaults to 8):
                The number of calibration samples to load per batch.
            data_collator (`DataCollator`, *optional*):
                The function to use to form a batch from a list of elements of the calibration dataset.
            remove_unused_columns (`bool`, defaults to `True`):
                Whether or not to remove the columns unused by the model forward method.

        Example:
        ```python
        >>> from optimum.intel.openvino import OVQuantizer, OVModelForSequenceClassification
        >>> from transformers import AutoModelForSequenceClassification
        >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        >>> OVQuantizer.from_pretrained(model, task="text-classification")
        >>> quantizer.quantize(calibration_dataset=calibration_dataset, save_directory="./quantized_model")
        >>> optimized_model = OVModelForSequenceClassification.from_pretrained("./quantized_model")
        ```
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        file_name = file_name if file_name is not None else OV_XML_FILE_NAME
        output_path = save_directory.joinpath(file_name)
        output_path = output_path.with_suffix(".xml").as_posix()
        calibration_dataloader = self._get_calibration_dataloader(
            calibration_dataset=calibration_dataset,
            batch_size=batch_size,
            remove_unused_columns=remove_unused_columns,
            data_collator=data_collator,
        )
        model_inputs = next(iter(calibration_dataloader))
        if quantization_config is None:
            logger.info(
                "No configuration describing the quantization process was provided, a default OVConfig will be generated."
            )
            quantization_config = OVConfig()
        quantization_config.add_input_info(model_inputs)
        nncf_config = NNCFConfig.from_dict(quantization_config.__dict__)
        nncf_config = register_default_init_args(nncf_config, calibration_dataloader)
        controller, compressed_model = create_compressed_model(
            self.model, nncf_config, wrap_inputs_fn=wrap_nncf_model_inputs_with_objwalk
        )
        controller.prepare_for_export()

        self._set_task()

        self.model.config.save_pretrained(save_directory)
        model_type = self.model.config.model_type.replace("_", "-")
        onnx_config_class = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model=self.model,
            task=self.task,
            model_type=model_type,
        )
        onnx_config = onnx_config_class(self.model.config)
        compressed_model.eval()
        num_parameters = compressed_model.num_parameters()
        save_as_external_data = use_external_data_format(num_parameters) or quantization_config.save_onnx_model
        f = io.BytesIO() if not save_as_external_data else save_directory / ONNX_WEIGHTS_NAME

        # Export the compressed model to the ONNX format
        self._onnx_export(compressed_model, onnx_config, model_inputs, quantization_config, f)

        # Load and save the compressed model
        model = core.read_model(f) if save_as_external_data else core.read_model(f.getvalue(), b"")
        self._save_pretrained(model, output_path)
        quantization_config.save_pretrained(save_directory)

    @staticmethod
    def _save_pretrained(model: openvino.runtime.Model, output_path: str):
        compress_quantize_weights_transformation(model)
        openvino.runtime.serialize(model, output_path, output_path.replace(".xml", ".bin"))

    def _onnx_export(
        self,
        model: NNCFNetwork,
        config: OnnxConfig,
        model_inputs: Dict,
        ov_config: OVConfig,
        f: Union[str, io.BytesIO],
    ):
        opset = min(config.DEFAULT_ONNX_OPSET, MAX_ONNX_OPSET)
        opset = opset if not ov_config.save_onnx_model else max(opset, MIN_ONNX_QDQ_OPSET)
        model_inputs = dict((k, v.to(model.device)) for k, v in model_inputs.items())
        # Create ordered inputs for the ONNX export of NNCFNetwork as keyword arguments are currently not supported
        inputs = tuple([model_inputs.pop(key, None) for key in self._export_input_names if len(model_inputs) != 0])

        with torch.no_grad():
            # Disable node additions to be exported in the graph
            model.disable_dynamic_graph_building()
            onnx_export(
                model,
                inputs,
                f=f,
                input_names=list(config.inputs.keys()),
                output_names=list(config.outputs.keys()),
                dynamic_axes={name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())},
                do_constant_folding=True,
                opset_version=opset,
            )
            model.enable_dynamic_graph_building()

    def _set_task(self):
        if self.task is None:
            self.task = HfApi().model_info(self.model.config._name_or_path).pipeline_tag
            if self.task in ["sentiment-analysis", "text-classification", "zero-shot-classification"]:
                self.task = "sequence-classification"
            elif self.task in ["feature-extraction", "fill-mask"]:
                self.task = "default"
            elif self.task == "text-generation":
                self.task = "causal-lm"
            elif self.task is None:
                raise ValueError(
                    "The task defining the model topology could not be extracted and needs to be specified for the ONNX export."
                )
        if self.task in ["seq2seq-lm", "translation", "summarization"]:
            raise ValueError(f"Seq2Seq models are currently not supported for post-training static quantization.")

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "train",
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        use_auth_token: bool = False,
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
            use_auth_token (`bool`, defaults to `False`):
                Whether to use the token generated when running `transformers-cli login`.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """
        calibration_dataset = load_dataset(
            dataset_name, name=dataset_config_name, split=dataset_split, use_auth_token=use_auth_token
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
    ) -> OVDataLoader:
        data_collator = data_collator if data_collator is not None else default_data_collator
        if remove_unused_columns:
            calibration_dataset = self._remove_unused_columns(calibration_dataset)
        self.input_names = calibration_dataset.column_names
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        sampler = RandomSampler(calibration_dataset, generator=generator)
        calibration_dataloader = DataLoader(
            calibration_dataset, batch_size=batch_size, sampler=sampler, collate_fn=data_collator, drop_last=False
        )
        return OVDataLoader(calibration_dataloader)

    def _remove_unused_columns(self, dataset: Dataset):
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)
