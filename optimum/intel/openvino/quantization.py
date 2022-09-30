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
from transformers import PreTrainedModel, default_data_collator
from transformers.onnx import FeaturesManager, OnnxConfig

import openvino
import openvino.runtime.passes as passes
from huggingface_hub import HfApi
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.nncf_network import NNCFNetwork
from openvino.runtime import Core
from optimum.quantization_base import OptimumQuantizer

from .nncf_config import get_config_with_input_info
from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME


MAX_ONNX_OPSET = 10


core = Core()

logger = logging.getLogger(__name__)


class OVDataLoader(PTInitializingDataLoader):
    def get_inputs(self, dataloader_output) -> Tuple[Tuple, Dict]:
        return (), dataloader_output


class OVQuantizer(OptimumQuantizer):
    """
    Handle the NNCF quantization process.
    """

    def __init__(self, model: transformers.PreTrainedModel, **kwargs):
        """
        Args:
            model (`transformers.PreTrainedModel`):
                The [PreTrainedModel](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel) to quantize.
            seed (`int`, defaults to 42):
                The random seed to use when shuffling the calibration dataset.
        """
        super().__init__()
        self.model = model
        self.seed = kwargs.pop("seed", 42)
        self.feature = kwargs.pop("feature", None)
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        self.input_names = None

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        quantization_config: NNCFConfig,
        calibration_dataset: Dataset,
        save_directory: Union[str, Path],
        file_name: Optional[str] = None,
        batch_size: int = 8,
    ):
        """
        Quantize a model given the optimization specifications defined in `quantization_config`.

        Args:
            quantization_config (`NNCFConfig`):
                The configuration containing the parameters related to quantization.
            calibration_dataset (`datasets.Dataset`):
                The dataset to use for the calibration step.
            save_directory (`Union[str, Path]`):
                The directory where the quantized model should be saved.
            file_name (`str`, *optional*):
                The model file name to use when saving the model. Overwrites the default file name `"model.onnx"`.
            batch_size (`int`, defaults to 8):
                The number of calibration samples to load per batch.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        file_name = file_name if file_name is not None else OV_XML_FILE_NAME
        output_path = save_directory.joinpath(file_name)
        output_path = output_path.with_suffix(".xml").as_posix()
        calibration_dataloader = self._get_calibration_dataloader(calibration_dataset, batch_size)
        model_inputs = next(iter(calibration_dataloader))
        nncf_config = get_config_with_input_info(quantization_config, model_inputs)
        nncf_config = register_default_init_args(nncf_config, calibration_dataloader)
        controller, compressed_model = create_compressed_model(
            self.model, nncf_config, wrap_inputs_fn=wrap_nncf_model_inputs_with_objwalk
        )
        controller.prepare_for_export()

        self._set_feature()

        self.model.config.save_pretrained(save_directory)
        model_type = self.model.config.model_type.replace("_", "-")
        onnx_config_cls = FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][self.feature]
        onnx_config = onnx_config_cls(self.model.config)
        compressed_model.eval()
        use_external_data_format = onnx_config.use_external_data_format(compressed_model.num_parameters())
        f = io.BytesIO() if not use_external_data_format else output_path.replace(".xml", ".onnx")

        # Export the compressed model to the ONNX format
        self._onnx_export(compressed_model, onnx_config, model_inputs, f)

        # Load and save the compressed model
        model = core.read_model(f) if use_external_data_format else core.read_model(f.getvalue(), b"")
        self._save_pretrained(model, output_path)

    @staticmethod
    def _save_pretrained(model: openvino.runtime.Model, output_path: str):
        pass_manager = passes.Manager()
        pass_manager.register_pass("Serialize", output_path, output_path.replace(".xml", ".bin"))
        pass_manager.run_passes(model)

    @staticmethod
    def _onnx_export(model: torch.nn.Module, config: OnnxConfig, model_inputs: Dict, f: Union[str, io.BytesIO]):
        # if onnx_config.default_onnx_opset > MAX_ONNX_OPSET:
        if config.default_onnx_opset > 11:
            logger.warning(
                f"The minimal ONNX opset for the given model architecture is {config.default_onnx_opset}, currently "
                f"OpenVINO only supports opset inferior or equal to {MAX_ONNX_OPSET} which could result in "
                "export issue."
            )
        with torch.no_grad():
            # Disable node additions to be exported in the graph
            model.disable_dynamic_graph_building()
            onnx_export(
                model,
                tuple(model_inputs.values()),
                f=f,
                input_names=list(model_inputs.keys()),
                output_names=list(config.outputs.keys()),
                dynamic_axes={name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())},
                do_constant_folding=True,
                opset_version=10,
            )
            model.enable_dynamic_graph_building()

    def _set_feature(self):
        if self.feature is None:
            self.feature = HfApi().model_info(self.model.config._name_or_path).pipeline_tag
            if self.feature in ["sentiment-analysis", "text-classification", "zero-shot-classification"]:
                self.feature = "sequence-classification"
            elif self.feature in ["feature-extraction", "fill-mask"]:
                self.feature = "default"
            elif self.feature is None:
                raise ValueError("The feature could not be extracted and needs to be specified for the ONNX export.")

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
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
            use_auth_token=use_auth_token,
        )

        if num_samples is not None:
            num_samples = min(num_samples, len(calibration_dataset))
            calibration_dataset = calibration_dataset.shuffle(seed=self.seed).select(range(num_samples))

        if preprocess_function is not None:
            calibration_dataset = calibration_dataset.map(preprocess_function, batched=preprocess_batch)

        return calibration_dataset

    def _get_calibration_dataloader(self, calibration_dataset: Dataset, batch_size: int) -> OVDataLoader:
        calibration_dataset = self._remove_unused_columns(calibration_dataset)
        self.input_names = calibration_dataset.column_names
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        sampler = RandomSampler(calibration_dataset, generator=generator)
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=default_data_collator,
            drop_last=False,
        )
        return OVDataLoader(calibration_dataloader)

    def _remove_unused_columns(self, dataset: Dataset):
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)
