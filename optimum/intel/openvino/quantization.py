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
import dataclasses
import inspect
import logging
from collections import UserDict, deque
from contextlib import contextmanager
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import datasets
import nncf
import numpy as np
import openvino
import requests
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from nncf.torch import register_module
from nncf.torch.initialization import PTInitializingDataLoader
from openvino import Core, Tensor
from openvino._offline_transformations import compress_quantize_weights_transformation
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, DataCollator, default_data_collator
from transformers.pytorch_utils import Conv1D
from transformers.utils import is_accelerate_available

from optimum.exporters.tasks import TasksManager
from optimum.intel.openvino.modeling_sam import OVSamPromptEncoder, OVSamVisionEncoder
from optimum.quantization_base import OptimumQuantizer
from optimum.utils.logging import warn_once

from ..utils.constant import _TASK_ALIASES
from ..utils.import_utils import (
    DATASETS_IMPORT_ERROR,
    _nncf_version,
    is_datasets_available,
    is_diffusers_available,
    is_nncf_version,
    is_sentence_transformers_available,
)
from .configuration import (
    OVConfig,
    OVMixedQuantizationConfig,
    OVPipelineQuantizationConfig,
    OVQuantizationConfig,
    OVQuantizationConfigBase,
    OVQuantizationMethod,
    OVWeightQuantizationConfig,
)
from .modeling import OVModel, OVModelForFeatureExtraction, OVModelForMaskedLM, OVModelForZeroShotImageClassification
from .modeling_base import OVBaseModel
from .modeling_decoder import OVBaseDecoderModel, OVModelForCausalLM
from .modeling_sam import OVSamModel
from .modeling_seq2seq import OVDecoder, OVEncoder, OVModelForSeq2SeqLM, _OVModelForWhisper
from .modeling_visual_language import OVModelForVisualCausalLM, OVVisionEmbedding
from .utils import (
    PREDEFINED_LANGUAGE_DATASETS,
    PREDEFINED_SAM_DATASETS,
    PREDEFINED_SD_DATASETS,
    PREDEFINED_SPEECH_TO_TEXT_DATASETS,
    PREDEFINED_TEXT_IMAGE_ENCODER_DATASETS,
    PREDEFINED_VISUAL_LM_DATASETS,
)


if is_diffusers_available():
    from optimum.intel.openvino.modeling_diffusion import OVDiffusionPipeline

if is_datasets_available():
    from datasets import Dataset

if is_sentence_transformers_available():
    from .modeling_sentence_transformers import OVSentenceTransformer

register_module(ignored_algorithms=[])(Conv1D)

core = Core()
logger = logging.getLogger(__name__)


class OVCalibrationDataset(UserDict):
    """
    A class to store calibration datasets for quantization with NNCF. Contains an instance of `nncf.Dataset` for each
    pipeline model component. For example, for a sequence-to-sequence pipeline with `encoder_model` and `decoder_model`
    components, the dictionary should contain two keys: `encoder_model` and `decoder_model`.
    """

    def __init__(self, calibration_dataset: Union[nncf.Dataset, Dict[str, nncf.Dataset]]):
        """
        Args:
            calibration_dataset (`Union[nncf.Dataset, Dict[str, nncf.Dataset]]`):
                The calibration dataset to store. Can be a single `nncf.Dataset` instance or a dictionary containing
                `nncf.Dataset` instances for each model component. In the first case it is assumed that the dataset
                corresponds to a pipeline component named "model".
        """
        if isinstance(calibration_dataset, nncf.Dataset):
            calibration_dataset = {"model": calibration_dataset}
        super().__init__(calibration_dataset)

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError


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
        inference_result_mock: Any = None,
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
            inference_result_mock (`Any`, *optional*):
                If provided, the target request won't be executed and this value will be returned instead resulting in
                faster inputs collection. This is useful when the actual model inference can be skipped, and it
                should not be provided for models which depend on previous inference results, e.g. encoder-decoder pipelines.
        """
        self.request = request
        self.collected_inputs = [] if collected_inputs is None else collected_inputs
        self.apply_caching = apply_caching
        self.inference_result_mock = inference_result_mock
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
            self.tensor_cache.setdefault(k, {})
            if data_hash not in self.tensor_cache[k]:
                self.tensor_cache[k][data_hash] = copy.deepcopy(v)
            copied_inputs[k] = self.tensor_cache[k][data_hash]
        self.collected_inputs.append(copied_inputs)

    def __call__(self, *args, **kwargs):
        # If __call__ is invoked then self.request must be an instance of CompiledModel
        signature = inspect.signature(self.request)
        bound_args = signature.bind(*args, **kwargs).arguments
        self.collect_inputs(bound_args["inputs"])
        if self.inference_result_mock is None:
            return self.request(*args, **kwargs)
        return self.inference_result_mock

    def infer(self, inputs: Any = None, share_inputs: bool = False):
        self.collect_inputs(inputs)
        if self.inference_result_mock is None:
            return self.request.infer(inputs, share_inputs)
        return self.inference_result_mock

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
        share_inputs: bool = False,
        *,
        shared_memory: Any = None,
    ):
        self.collect_inputs(inputs)
        if self.inference_result_mock is None:
            self.request.infer(inputs, share_inputs, share_outputs=True)
        return self.inference_result_mock

    def wait(self):
        pass

    def get_tensor(self, name: str):
        return Tensor(self.request.results[name])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)


class OVCalibrationDatasetBuilder:
    """
    A class to build calibration datasets for quantization with NNCF.

    Allows to build a calibration dataset from:
        - a `datasets.Dataset` object;
        - a name of the dataset from `datasets`;
        - a quantization config object containing dataset specification.

    Returns calibration dataset as an instance of `OVCalibrationDataset` containing an `nncf.Dataset` for each model component.
    For example, for a sequence-to-sequence model with `encoder_model` and `decoder_model` components, the dictionary
    will contain two keys: `encoder_model` and `decoder_model`.
    """

    def __init__(self, model: OVModel, seed: int = 42):
        """

        Args:
            model (`OVModel`):
                The model to build calibration dataset for.
            seed (`int`, defaults to 42):
                Random seed to use for reproducibility.
        """
        self.model = model
        self.seed = seed
        # TODO: deprecate "signature_columns": model.forward() may not be the method which is called during inference,
        #  for example there is model.generate()
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())

    def build_from_quantization_config(self, config: OVQuantizationConfigBase) -> OVCalibrationDataset:
        """
        Builds a calibration dataset from a quantization config object. Namely, `quantization_config.dataset` property
        is used to infer dataset name.

        Args:
            config (`OVQuantizationConfigBase`):
                The quantization configuration object.
        Returns:
            A calibration dataset as an instance of `OVCalibrationDataset` containing an `nncf.Dataset` for each model component.
        """

        if config.dataset is None:
            raise ValueError("Please provide a dataset for calibration.")

        if isinstance(self.model, OVModelForCausalLM):
            return self._prepare_causal_lm_calibration_data(config)
        elif isinstance(
            self.model,
            (OVModelForVisualCausalLM, _OVModelForWhisper, OVModelForZeroShotImageClassification, OVSamModel),
        ):
            if config.processor is None:
                raise ValueError(
                    "`processor` must be specified in order to run data-aware quantization. Please provide it as a"
                    "model id, or a path to a directory containing all the required configuration files."
                )

            if isinstance(self.model, OVModelForVisualCausalLM):
                dataset_metadata = PREDEFINED_VISUAL_LM_DATASETS[config.dataset]
                return self.build_from_dataset_name(
                    config,
                    dataset_metadata["id"],
                    num_samples=config.num_samples,
                    dataset_split=dataset_metadata["split"],
                )
            elif isinstance(self.model, _OVModelForWhisper):
                dataset_metadata = PREDEFINED_SPEECH_TO_TEXT_DATASETS[config.dataset]
                return self.build_from_dataset_name(
                    config,
                    dataset_metadata["id"],
                    num_samples=config.num_samples,  # This is an upper bound on how many audios are needed
                    dataset_split=dataset_metadata["split"],
                    streaming=dataset_metadata["streaming"],
                    data_dir=dataset_metadata.get("data_dir", None),
                    revision=dataset_metadata.get("revision", None),
                )
            elif isinstance(self.model, OVModelForZeroShotImageClassification):
                dataset_metadata = PREDEFINED_TEXT_IMAGE_ENCODER_DATASETS[config.dataset]
                return self.build_from_dataset_name(
                    config,
                    dataset_metadata["id"],
                    num_samples=None,
                    dataset_split=dataset_metadata["split"],
                    streaming=dataset_metadata["streaming"],
                )
            elif isinstance(self.model, OVSamModel):
                dataset_metadata = PREDEFINED_SAM_DATASETS[config.dataset]
                return self.build_from_dataset_name(
                    config,
                    dataset_metadata["id"],
                    dataset_split=dataset_metadata["split"],
                    streaming=dataset_metadata["streaming"],
                )
            else:
                raise Exception
        elif is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
            if isinstance(config.dataset, str):
                dataset_name = config.dataset
                dataset_metadata = PREDEFINED_SD_DATASETS[dataset_name]

                dataset = self.load_dataset(
                    dataset_name,
                    num_samples=config.num_samples,  # This is an upper bound on how many prompts are needed
                    dataset_split=dataset_metadata["split"],
                    streaming=dataset_metadata["streaming"],
                )
            elif isinstance(config.dataset, list) and all(isinstance(it, str) for it in config.dataset):
                dataset = config.dataset
            else:
                raise RuntimeError(
                    "Please provide dataset as one of the accepted dataset labels or as a list of string prompts."
                )

            return self.build_from_dataset(config, dataset)
        elif (
            isinstance(self.model, (OVModelForFeatureExtraction, OVModelForMaskedLM, OVModelForSeq2SeqLM))
            or is_sentence_transformers_available()
            and isinstance(self.model, OVSentenceTransformer)
        ):
            if isinstance(config.dataset, str):
                dataset_metadata = PREDEFINED_LANGUAGE_DATASETS[config.dataset]
                dataset = self.load_dataset(
                    dataset_metadata["id"],
                    num_samples=None,
                    dataset_config_name=dataset_metadata["name"],
                    dataset_split=dataset_metadata["split"],
                    streaming=dataset_metadata["streaming"],
                )
            elif isinstance(config.dataset, list) and all(isinstance(it, str) for it in config.dataset):
                dataset = datasets.Dataset.from_list([{"text": it} for it in config.dataset])
            else:
                raise ValueError(
                    "Please provide dataset as one of the accepted dataset labels or as a list of strings."
                )
            return self.build_from_dataset(config, dataset)
        else:
            raise RuntimeError("Unsupported model type for calibration dataset collection.")

    def build_from_dataset_name(
        self,
        quantization_config: OVQuantizationConfigBase,
        dataset_name: str,
        num_samples: Optional[int] = None,
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "train",
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        token: Optional[Union[bool, str]] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        streaming: bool = False,
        batch_size: Optional[int] = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
        **dataset_kwargs,
    ) -> OVCalibrationDataset:
        """
        Create the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            quantization_config (`OVQuantizationConfigBase`):
                The quantization configuration object.
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                in generic formats and optionally a dataset script, if it requires some code to read the data files.
            dataset_config_name (`str`, *optional*):
                The name of the dataset configuration.
            num_samples (`int`, *optional*):
                The maximum number of samples composing the calibration dataset.
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
            streaming (`bool`, defaults to `False`):
                Whether to load dataset in streaming mode.
            batch_size (`int`, defaults to 1):
                The number of calibration samples to load per batch.
            data_collator (`DataCollator`, *optional*):
                The function to use to form a batch from a list of elements of the calibration dataset.
            remove_unused_columns (`bool`, defaults to `False`):
                Whether to remove the columns unused by the model forward method.
        Returns:
            A calibration dataset as an instance of `OVCalibrationDataset` containing an `nncf.Dataset` for each model component.
        """

        if remove_unused_columns:
            logger.warning("`remove_unused_columns` is deprecated and will be removed in optimum-intel v1.25.")

        dataset = self.load_dataset(
            dataset_name,
            num_samples,
            dataset_config_name,
            dataset_split,
            preprocess_function,
            preprocess_batch,
            token,
            cache_dir,
            streaming,
            **dataset_kwargs,
        )

        return self.build_from_dataset(quantization_config, dataset, batch_size, data_collator, remove_unused_columns)

    def build_from_dataset(
        self,
        quantization_config: OVQuantizationConfigBase,
        dataset: Union["Dataset", List],
        batch_size: Optional[int] = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
    ) -> OVCalibrationDataset:
        """

        Args:
            quantization_config (`OVQuantizationConfigBase`):
                The quantization configuration object.
            dataset (`Union[datasets.Dataset, List]`):
                The dataset to collect calibration data from.
            batch_size (`int`, defaults to 1):
                The number of calibration samples to load per batch. Not always used.
            data_collator (`DataCollator`, *optional*):
                The function to use to form a batch from a list of elements of the calibration dataset. Not always used.
            remove_unused_columns (`bool`, defaults to `False`):
                Whether to remove the columns unused by the model forward method. Not always used.
        Returns:
            A calibration dataset as an instance of `OVCalibrationDataset` containing an `nncf.Dataset` for each model component.
        """

        if isinstance(dataset, list):
            logger.warning(
                "Providing dataset as a list is deprecated and will be removed in optimum-intel v1.25. "
                "Please provide it as `datasets.Dataset`."
            )

        if (
            isinstance(
                self.model,
                (
                    OVModelForVisualCausalLM,
                    _OVModelForWhisper,
                    OVModelForFeatureExtraction,
                    OVModelForMaskedLM,
                    OVModelForZeroShotImageClassification,
                    OVModelForSeq2SeqLM,
                    OVSamModel,
                ),
            )
            or is_diffusers_available()
            and isinstance(self.model, OVDiffusionPipeline)
            or is_sentence_transformers_available()
            and isinstance(self.model, OVSentenceTransformer)
        ):
            # Prepare from raw dataset avoiding dataloader creation
            if batch_size != 1 or data_collator is not None or remove_unused_columns:
                logger.warning(
                    "`batch_size`, `data_collator` and `remove_unused_columns` are not supported for this type of model."
                )

            if isinstance(self.model, OVModelForVisualCausalLM):
                return self._prepare_visual_causal_lm_calibration_data(quantization_config, dataset)
            elif isinstance(self.model, _OVModelForWhisper):
                return self._prepare_speech_to_text_calibration_data(quantization_config, dataset)
            elif isinstance(self.model, OVModelForSeq2SeqLM):
                return self._prepare_text_to_text_calibration_data(quantization_config, dataset)
            elif is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
                return self._prepare_diffusion_calibration_data(quantization_config, dataset)
            elif (
                isinstance(self.model, (OVModelForFeatureExtraction, OVModelForMaskedLM))
                or is_sentence_transformers_available()
                and isinstance(self.model, OVSentenceTransformer)
            ):
                return self._prepare_text_encoder_model_calibration_data(quantization_config, dataset)
            elif isinstance(self.model, OVModelForZeroShotImageClassification):
                return self._prepare_text_image_encoder_model_calibration_data(quantization_config, dataset)
            elif isinstance(self.model, OVSamModel):
                return self._prepare_sam_dataset(quantization_config, dataset)
            else:
                raise RuntimeError("Unsupported model type for calibration dataset collection.")
        else:
            # Prepare from dataloader
            # Setting `remove_unused_columns=True` until it is not deprecated
            dataloader = self._get_calibration_dataloader(
                dataset, batch_size, data_collator, remove_unused_columns=True
            )
            if isinstance(self.model, OVBaseDecoderModel):
                return self._prepare_decoder_calibration_data(quantization_config, dataloader)
            else:
                # Assuming this is the torch model quantization scenario
                return OVCalibrationDataset({"model": nncf.Dataset(dataloader)})

    def load_dataset(
        self,
        dataset_name: str,
        num_samples: Optional[int] = None,
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "train",
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        token: Optional[Union[bool, str]] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        streaming: bool = False,
        **dataset_kwargs,
    ) -> "Dataset":
        """
        Create the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                in generic formats and optionally a dataset script, if it requires some code to read the data files.
            num_samples (`int`, *optional*):
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
            streaming (`bool`, defaults to `False`):
                Whether to load dataset in streaming mode.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """
        if not is_datasets_available():
            raise ValueError(DATASETS_IMPORT_ERROR.format("OVCalibrationDatasetBuilder.load_dataset"))

        from datasets import load_dataset

        datasets_kwargs = {
            **dataset_kwargs,
            "name": dataset_config_name,
            "split": dataset_split,
            "token": token,
            "cache_dir": cache_dir,
            "streaming": streaming,
        }

        dataset = load_dataset(dataset_name, **datasets_kwargs)
        dataset = dataset.shuffle(seed=self.seed)
        if num_samples is not None:
            dataset = dataset.take(num_samples)

        if preprocess_function is not None:
            dataset = dataset.map(preprocess_function, batched=preprocess_batch)

        return dataset

    def _get_calibration_dataloader(
        self,
        dataset: Union["Dataset", List],
        batch_size: Optional[int] = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
    ) -> OVDataLoader:
        """
        Wrap dataset into a dataloader.
        """
        if remove_unused_columns:
            logger.warning("`remove_unused_columns` is deprecated and will be removed in optimum-intel v1.25.")

        if not is_datasets_available():
            raise ValueError(DATASETS_IMPORT_ERROR.format("OVCalibrationDatasetBuilder._get_calibration_dataloader"))

        from datasets import Dataset, IterableDataset

        data_collator = data_collator or default_data_collator

        if remove_unused_columns and isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        sampler = None
        if not isinstance(dataset, IterableDataset):
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            sampler = RandomSampler(dataset, generator=generator)

        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=data_collator, drop_last=False
        )
        return OVDataLoader(dataloader)

    def _prepare_decoder_calibration_data(
        self, quantization_config: OVQuantizationConfigBase, dataloader: OVDataLoader
    ) -> OVCalibrationDataset:
        """
        Prepares calibration data by collecting model inputs during inference.
        """
        # Prefetch past_key_values
        self.model.update_pkv_precision(True)
        self.model.compile()
        collected_inputs = []

        num_samples = quantization_config.num_samples or 200
        self.model.request = InferRequestWrapper(self.model.request, collected_inputs)
        try:
            for data in tqdm(dataloader, desc="Collecting calibration data", total=num_samples):
                if len(collected_inputs) > num_samples:
                    break
                self.model.generate(**data, max_new_tokens=1)
        finally:
            self.model.request = self.model.request.request

        return OVCalibrationDataset(nncf.Dataset(collected_inputs))

    def _prepare_causal_lm_calibration_data(
        self, config: OVQuantizationConfigBase, seqlen: int = 32
    ) -> OVCalibrationDataset:
        """
        Prepares calibration data for causal language models. Relies on `optimum.gptq.data` module.
        """
        from optimum.gptq.data import get_dataset, prepare_dataset

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=config.trust_remote_code)
        nsamples = config.num_samples if config.num_samples else 128
        if isinstance(config.dataset, str):
            if config.dataset == "auto":
                generated_data = nncf.data.generate_text_data(self.model, tokenizer, dataset_size=nsamples)
                calibration_dataset = [tokenizer(text, return_tensors="pt") for text in generated_data]
            else:
                calibration_dataset = get_dataset(config.dataset, tokenizer, seqlen=seqlen, nsamples=nsamples)
        elif isinstance(config.dataset, list) and all(isinstance(it, str) for it in config.dataset):
            calibration_dataset = [tokenizer(text, return_tensors="pt") for text in config.dataset[:nsamples]]
        else:
            raise ValueError("Please provide dataset as one of the accepted dataset labels or as a list of strings.")
        calibration_dataset = prepare_dataset(calibration_dataset)
        calibration_dataset = nncf.Dataset(calibration_dataset, lambda x: self.model.prepare_inputs(**x))

        return OVCalibrationDataset(calibration_dataset)

    def _prepare_visual_causal_lm_calibration_data(
        self,
        config: OVQuantizationConfigBase,
        dataset: "Dataset",
        max_image_size: Optional[int] = 600,
    ) -> OVCalibrationDataset:
        """
        Prepares calibration data for VLM pipelines.
        Currently, collects data only for a language model component.
        """
        processor = AutoProcessor.from_pretrained(config.processor, trust_remote_code=config.trust_remote_code)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=config.trust_remote_code)
            tokenizer_error = None
        except Exception as tokenizer_error:  # noqa: F841
            tokenizer = None

        dataset_metadata = PREDEFINED_VISUAL_LM_DATASETS[config.dataset]

        collected_inputs: Dict[str, List[Dict[str, Any]]] = {"lm_model": []}
        # Collect vision embeddings calibration data by using InferRequestWrapper
        vision_embedding_components = []
        for ov_component_name, ov_component in self.model.components.items():
            if not isinstance(ov_component, OVVisionEmbedding):
                continue
            vision_embedding_components.append(ov_component)
            submodel_name = f"{ov_component_name}_model"
            collected_inputs[submodel_name] = []
            ov_component._compile()
            ov_component.request = InferRequestWrapper(ov_component.request, collected_inputs[submodel_name])

        try:
            num_samples = config.num_samples or 32
            for item in tqdm(dataset, desc="Collecting calibration dataset", total=num_samples):
                if len(collected_inputs["lm_model"]) >= num_samples:
                    break

                instruction = item[dataset_metadata["inputs"]["instruction"]]
                image_url = item[dataset_metadata["inputs"]["image_url"]]
                image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
                if max_image_size is not None:
                    # To avoid large images, resize them keeping the aspect ratio
                    scale_factor = max(image.size[0] / max_image_size, image.size[1] / max_image_size)
                    if scale_factor > 1:
                        new_size = (int(image.size[0] / scale_factor), int(image.size[1] / scale_factor))
                        image = image.resize(new_size)

                try:
                    inputs = self.model.preprocess_inputs(
                        text=instruction,
                        image=image,
                        processor=processor,
                        tokenizer=tokenizer,
                        config=self.model.config,
                    )
                except ValueError as value_error:
                    if "Tokenizer is required." in str(value_error) and tokenizer_error is not None:
                        raise tokenizer_error
                    raise value_error

                inputs_embeds, attention_mask, position_ids = self.model.get_multimodal_embeddings(**inputs)

                language_model_inputs = self.model.language_model.prepare_inputs(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                )

                collected_inputs["lm_model"].append(language_model_inputs)

            # If an input dict contains `pixel_values` key and its batch size is greater than 1, we split the data
            # into multiple single-batch dicts below. This lowers peak RAM consumption during quantization calibration.
            for submodel_name in collected_inputs:
                single_batch_collected_inputs = []
                for input_dict in collected_inputs[submodel_name]:
                    # We expect 'pixel_values' to be a 4D tensor: [batch, channel, height, width].
                    # This is standard for batches of images in vision models.
                    if (
                        "pixel_values" in input_dict
                        and isinstance(input_dict["pixel_values"], torch.Tensor)
                        and input_dict["pixel_values"].dim() == 4
                        and input_dict["pixel_values"].shape[0] > 1
                    ):
                        if is_nncf_version("<=", "2.18"):
                            # TODO (Nikita): Remove once NNCF 2.19 is released.
                            warn_once(
                                logger,
                                "If you are facing RAM OOM issues, please update to the latest NNCF develop version.",
                            )
                        batch_size = input_dict["pixel_values"].shape[0]
                        for i in range(batch_size):
                            single_batch_input_dict = {}
                            for input_name, input_value in input_dict.items():
                                if not isinstance(input_value, torch.Tensor):
                                    raise TypeError(
                                        f"Expected a torch.Tensor instance for input '{input_name}', "
                                        f"but got {type(input_value)}."
                                    )
                                if input_value.shape[0] != batch_size:
                                    raise ValueError(
                                        f"Expected a tensor with batch size {batch_size} for input '{input_name}', "
                                        f"but got shape {input_value.shape}."
                                    )
                                single_batch_input_dict[input_name] = input_value[i : i + 1]
                            single_batch_collected_inputs.append(single_batch_input_dict)
                    else:
                        single_batch_collected_inputs.append(input_dict)
                collected_inputs[submodel_name] = single_batch_collected_inputs
        finally:
            for ov_component in vision_embedding_components:
                ov_component.request = ov_component.request.request

        for submodel_name in collected_inputs:
            collected_inputs[submodel_name] = nncf.Dataset(collected_inputs[submodel_name])

        return OVCalibrationDataset(collected_inputs)

    def _prepare_speech_to_text_calibration_data(
        self, config: OVQuantizationConfigBase, dataset: "Dataset"
    ) -> OVCalibrationDataset:
        """
        Prepares calibration data for speech-to-text pipelines by inferring it on a dataset and collecting incurred inputs.
        """

        models: Dict[str, Union[OVEncoder, OVDecoder]] = {}
        collected_inputs: Dict[str, List[Dict[str, Any]]] = {}
        for submodel_name in self.model._ov_submodel_names:
            ov_component: Union[OVEncoder, OVDecoder] = getattr(self.model, submodel_name)
            models[submodel_name] = ov_component
            collected_inputs[submodel_name] = []
            ov_component._compile()
            ov_component.request = InferRequestWrapper(
                ov_component.request, collected_inputs[submodel_name], apply_caching=True
            )
        try:
            processor = AutoProcessor.from_pretrained(config.processor, trust_remote_code=config.trust_remote_code)

            # Download audio inputs beforehand to avoid possible connection issues
            num_samples = config.num_samples or 32
            dataset = list(tqdm(dataset.take(num_samples), desc="Downloading audio inputs", total=num_samples))

            for item in tqdm(dataset, desc="Collecting calibration data"):
                audio = item["audio"]["array"]
                sampling_rate = item["audio"]["sampling_rate"]
                input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
                self.model.generate(input_features)
        finally:
            for model in models.values():
                model.request = model.request.request

        for model_name in collected_inputs:
            collected_inputs[model_name] = nncf.Dataset(collected_inputs[model_name])

        return OVCalibrationDataset(collected_inputs)

    def _prepare_text_to_text_calibration_data(
        self,
        config: OVQuantizationConfigBase,
        dataset: "Dataset",
        seq_len: int = 128,
    ) -> OVCalibrationDataset:
        """
        Prepares calibration data for text-to-text pipelines by inferring it on a dataset and collecting incurred inputs.
        """

        models: Dict[str, Union[OVEncoder, OVDecoder]] = {}
        collected_inputs: Dict[str, List[Dict[str, Any]]] = {}
        for submodel_name in self.model._ov_submodel_names:
            ov_component: Union[OVEncoder, OVDecoder] = getattr(self.model, submodel_name)
            models[submodel_name] = ov_component
            collected_inputs[submodel_name] = []
            ov_component._compile()
            ov_component.request = InferRequestWrapper(
                ov_component.request, collected_inputs[submodel_name], apply_caching=True
            )
        try:

            def get_tokenizer():
                if config.tokenizer is None:
                    raise ValueError("Please provide tokenizer for calibration via quantization_config.tokenizer.")
                return AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=config.trust_remote_code)

            num_samples = config.num_samples or 128
            dataset = list(tqdm(dataset.take(num_samples), desc="Downloading dataset", total=num_samples))

            tokenizer = None
            for item in tqdm(dataset, desc="Collecting calibration data"):
                if "input_ids" in item:
                    # Assuming that dataset contains already preprocessed text
                    inputs = self._wrap_sample_as_array(item, add_batch_dim=True)
                else:
                    tokenizer = tokenizer or get_tokenizer()
                    inputs = tokenizer(item["text"], truncation=True, max_length=seq_len, return_tensors="pt")

                self.model.generate(**inputs, max_new_tokens=seq_len)
        finally:
            for model in models.values():
                model.request = model.request.request

        for model_name in collected_inputs:
            collected_inputs[model_name] = nncf.Dataset(collected_inputs[model_name])

        return OVCalibrationDataset(collected_inputs)

    def _prepare_diffusion_calibration_data(
        self, config: OVQuantizationConfigBase, dataset: Union[List, "Dataset"]
    ) -> OVCalibrationDataset:
        """
        Prepares calibration data for diffusion models by inferring it on a dataset. Currently, collects data only for
        a vision diffusion component.
        """
        self.model.compile()

        diffuser_model_name = "unet" if self.model.unet is not None else "transformer"
        diffuser = getattr(self.model, diffuser_model_name)

        size = diffuser.config.get("sample_size", 64) * self.model.vae_scale_factor
        height, width = 2 * (min(size, 512),)

        num_samples = config.num_samples or 200
        calibration_data = []
        try:
            self.disable_progress_bar(disable=True)
            diffuser.request = InferRequestWrapper(diffuser.request, calibration_data)

            pbar = tqdm(total=num_samples, desc="Collecting calibration data")
            for item in dataset:
                prompt = (
                    item[PREDEFINED_SD_DATASETS[config.dataset]["prompt_column_name"]]
                    if isinstance(item, dict)
                    else item
                )
                self.model(prompt, height=height, width=width)
                pbar.update(min(num_samples, len(calibration_data)) - pbar.n)
                if len(calibration_data) >= num_samples:
                    calibration_data = calibration_data[:num_samples]
                    break
        finally:
            diffuser.request = diffuser.request.request
            self.disable_progress_bar(disable=False)

        return OVCalibrationDataset({diffuser_model_name: nncf.Dataset(calibration_data[:num_samples])})

    def _remove_unused_columns(self, dataset: "Dataset"):
        # TODO: deprecate because model.forward() may not be the method which is called during inference,
        #  for example there is model.generate()
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)

    def disable_progress_bar(self, disable: bool = True) -> None:
        if not hasattr(self.model, "_progress_bar_config"):
            self.model._progress_bar_config = {"disable": disable}
        else:
            self.model._progress_bar_config["disable"] = disable

    def _prepare_text_encoder_model_calibration_data(
        self,
        quantization_config: OVQuantizationConfigBase,
        dataset: "Dataset",
        seq_len: int = 128,
    ) -> OVCalibrationDataset:
        """
        Prepares calibration data for text-encoder-like models.
        Supports OVModelForFeatureExtraction, OVModelForMaskedLM and OVSentenceTransformer.
        """

        def get_tokenizer():
            if is_sentence_transformers_available() and isinstance(self.model, OVSentenceTransformer):
                return self.model.tokenizer
            else:
                if quantization_config.tokenizer is None:
                    raise ValueError("Please provide tokenizer for calibration via quantization_config.tokenizer.")
                tokenizer = AutoTokenizer.from_pretrained(
                    quantization_config.tokenizer, trust_remote_code=quantization_config.trust_remote_code
                )
            return tokenizer

        self.model.compile()

        num_samples = quantization_config.num_samples or 128
        calibration_data = []
        try:
            inference_result_mock = {}
            if isinstance(self.model, OVModelForFeatureExtraction):
                inference_result_mock["last_hidden_state"] = np.empty((1,), np.float32)
            elif isinstance(self.model, OVModelForMaskedLM):
                inference_result_mock["logits"] = np.empty((1,), np.float32)
            elif is_sentence_transformers_available() and isinstance(self.model, OVSentenceTransformer):
                inference_result_mock["token_embeddings"] = np.empty((1,), np.float32)
                inference_result_mock["sentence_embedding"] = np.empty((1,), np.float32)
            else:
                raise RuntimeError(
                    f"Unsupported model type {type(self.model).__name__} for calibration dataset collection."
                )

            self.model.request = InferRequestWrapper(
                self.model.request,
                calibration_data,
                inference_result_mock=inference_result_mock,
            )

            max_position_embeddings = getattr(self.model.config, "max_position_embeddings", None)
            if max_position_embeddings is not None and max_position_embeddings > 0:
                seq_len = min(seq_len, max_position_embeddings)

            random_positions = None
            if isinstance(self.model, OVModelForMaskedLM):
                with numpy_seed(self.seed):
                    random_positions = np.random.randint(0, seq_len, num_samples)

            tokenizer = None
            pbar = tqdm(total=num_samples, desc="Collecting calibration data")
            for item in dataset:
                if "input_ids" in item:
                    # Assuming that dataset contains already preprocessed text
                    inputs = self._wrap_sample_as_array(item, add_batch_dim=True)
                else:
                    tokenizer = tokenizer or get_tokenizer()
                    inputs = tokenizer(item["text"], truncation=True, max_length=seq_len, return_tensors="np")

                    if inputs["input_ids"].shape[1] < seq_len:
                        continue

                    if isinstance(self.model, OVModelForMaskedLM):
                        # Replace a random token with a mask token
                        inputs["input_ids"][0, random_positions[len(calibration_data)]] = tokenizer.mask_token_id

                self.model(inputs) if is_sentence_transformers_available() and isinstance(
                    self.model, OVSentenceTransformer
                ) else self.model(**inputs)

                pbar.update(min(num_samples, len(calibration_data)) - pbar.n)
                if len(calibration_data) >= num_samples:
                    break
        finally:
            self.model.request = self.model.request.request

        return OVCalibrationDataset({"model": nncf.Dataset(calibration_data)})

    def _prepare_text_image_encoder_model_calibration_data(
        self,
        quantization_config: OVQuantizationConfigBase,
        dataset: "Dataset",
        seq_len: int = 128,
    ) -> OVCalibrationDataset:
        self.model.compile()

        def get_processor():
            processor = AutoProcessor.from_pretrained(
                quantization_config.processor, trust_remote_code=quantization_config.trust_remote_code
            )
            return processor

        max_position_embeddings = getattr(self.model.config, "max_position_embeddings", None)
        if max_position_embeddings is not None and max_position_embeddings > 0:
            seq_len = min(seq_len, max_position_embeddings)

        num_samples = quantization_config.num_samples or 128
        calibration_data = []
        try:
            inference_result_mock = {
                "logits_per_image": np.empty((1,), np.float32),
                "logits_per_text": np.empty((1,), np.float32),
                "text_embeds": np.empty((1,), np.float32),
                "image_embeds": np.empty((1,), np.float32),
            }

            self.model.request = InferRequestWrapper(
                self.model.request,
                calibration_data,
                inference_result_mock=inference_result_mock,
            )

            processor = None
            pbar = tqdm(total=num_samples, desc="Collecting calibration data")
            for item in dataset:
                if "input_ids" in item:
                    # Assuming that dataset contains already preprocessed text
                    inputs = self._wrap_sample_as_array(item, add_batch_dim=True)
                else:
                    dataset_metadata = PREDEFINED_TEXT_IMAGE_ENCODER_DATASETS[quantization_config.dataset]
                    try:
                        response = requests.get(item[dataset_metadata["image_column_name"]], timeout=5)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                    except Exception:
                        continue
                    processor = processor or get_processor()
                    inputs = processor(
                        text=item[dataset_metadata["text_column_name"]],
                        images=image.convert("RGB"),
                        return_tensors="pt",
                        padding=True,
                    )
                    if inputs["input_ids"].shape[1] > seq_len:
                        inputs["input_ids"] = inputs["input_ids"][:, :seq_len]

                self.model(**inputs)

                pbar.update(min(num_samples, len(calibration_data)) - pbar.n)
                if len(calibration_data) >= num_samples:
                    break
        finally:
            self.model.request = self.model.request.request

        return OVCalibrationDataset({"model": nncf.Dataset(calibration_data)})

    def _prepare_sam_dataset(self, config: OVQuantizationConfigBase, dataset: "Dataset") -> OVCalibrationDataset:
        models: Dict[str, Union[OVSamVisionEncoder, OVSamPromptEncoder]] = {}
        collected_inputs: Dict[str, List[Dict[str, Any]]] = {}
        for submodel_name in self.model._ov_submodel_names:
            ov_component: Union[OVSamVisionEncoder, OVSamPromptEncoder] = getattr(self.model, submodel_name)
            models[submodel_name] = ov_component
            collected_inputs[submodel_name] = []
            ov_component._compile()
            ov_component.request = InferRequestWrapper(ov_component.request, collected_inputs[submodel_name])

        # We can avoid inferring the whole model if dataset is required only for the vision encoder model.
        collect_only_for_vision_encoder = (
            isinstance(config, OVPipelineQuantizationConfig)
            and len(config.quantization_configs) == 1
            and "vision_encoder" in config.quantization_configs
        )

        try:
            processor = AutoProcessor.from_pretrained(config.processor, trust_remote_code=config.trust_remote_code)

            num_samples = config.num_samples or 128
            for item in tqdm(islice(dataset, num_samples), total=num_samples, desc="Collecting calibration data"):
                inputs = processor(item["image"], input_points=[[[0, 0]]], return_tensors="pt")
                if collect_only_for_vision_encoder:
                    collected_inputs["vision_encoder"].append({"pixel_values": inputs["pixel_values"]})
                else:
                    self.model(**inputs)
        finally:
            for model in models.values():
                model.request = model.request.request

        if collect_only_for_vision_encoder:
            del collected_inputs["prompt_encoder_mask_decoder"]

        for model_name in collected_inputs:
            collected_inputs[model_name] = nncf.Dataset(collected_inputs[model_name])

        return OVCalibrationDataset(collected_inputs)

    @staticmethod
    def _wrap_sample_as_array(
        sample: Dict[str, Any], add_batch_dim: bool = False
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Converts a sample to a dictionary of numpy arrays or torch tensors.
        """
        results = {}
        for k, v in sample.items():
            v_as_array = v if isinstance(v, (torch.Tensor, np.ndarray)) else np.array(v)
            if add_batch_dim:
                v_as_array = v_as_array[None]
            results[k] = v_as_array
        return results


class OVQuantizer(OptimumQuantizer):
    """
    Handle the NNCF quantization process.
    """

    def __init__(self, model: OVModel, task: Optional[str] = None, seed: int = 42, **kwargs):
        """
        Args:
            model (`OVModel`):
                The [OVModel](https://huggingface.co/docs/optimum-intel/en/openvino/reference) to quantize.
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
    def from_pretrained(cls, model: OVModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        calibration_dataset: Optional[
            Union[OVCalibrationDataset, "Dataset", nncf.Dataset, Union[str, nncf.Dataset], List]
        ] = None,
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
            logger.warning("`remove_unused_columns` is deprecated and will be removed in optimum-intel v1.25.")

        if isinstance(calibration_dataset, list):
            logger.warning(
                "Providing calibration dataset as a list is deprecated and will be removed in optimum-intel v1.25. "
                "Please provide it as `datasets.Dataset` or as dictionary of `nncf.Dataset` instances."
            )

        if calibration_dataset is not None and isinstance(calibration_dataset, (dict, nncf.Dataset)):
            calibration_dataset = OVCalibrationDataset(calibration_dataset)

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

        if calibration_dataset is not None and not isinstance(calibration_dataset, OVCalibrationDataset):
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
                    "optimum-intel v1.25. Please filter out the unnecessary columns before passing the dataset."
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
                    "in optimum-intel v1.25. Please provide the list inside `quantization_config.dataset`"
                    "property instead."
                )
                quantization_config.dataset = calibration_dataset
                calibration_dataset = None
            else:
                calibration_dataset = self.dataset_builder.build_from_dataset(
                    quantization_config, calibration_dataset, batch_size, data_collator, remove_unused_columns
                )

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
            raise TypeError(
                "The support of `torch.nn.Module` is deprecated, please use the corresponding `OVModelForXxx` class to load and export your model to the OpenVINO IR format."
            )
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

    def _quantize_ovbasemodel(
        self,
        ov_config: OVConfig,
        save_directory: Union[str, Path] = None,
        calibration_dataset: Optional[OVCalibrationDataset] = None,
        **kwargs,
    ):
        quantization_config = ov_config.quantization_config
        dataset_was_built_from_config = False
        if calibration_dataset is None and quantization_config.dataset is not None:
            dataset_was_built_from_config = True
            calibration_dataset = self.dataset_builder.build_from_quantization_config(quantization_config)

        quantization_configs = {}
        default_config = None
        if (
            isinstance(quantization_config, OVWeightQuantizationConfig)
            and quantization_config.quant_method != OVQuantizationMethod.HYBRID
        ):
            #
            # Regular (non-hybrid) weight-only quantization
            #
            if isinstance(self.model, OVModelForVisualCausalLM):
                quantization_configs["lm_model"] = quantization_config
                default_config = OVWeightQuantizationConfig(bits=8, sym=True)
            else:
                default_config = quantization_config
        else:
            #
            # Hybrid/Full/Mixed quantization
            #

            if calibration_dataset is None:
                raise ValueError("Calibration dataset is required to run data-aware quantization.")

            if (
                isinstance(quantization_config, OVWeightQuantizationConfig)
                and quantization_config.quant_method == OVQuantizationMethod.HYBRID
            ):
                #
                # Hybrid quantization
                #
                if is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
                    if len(calibration_dataset) > 1:
                        raise ValueError("Calibration datasets for Diffusion models should contain only one value.")

                    # Apply hybrid quantization to diffusion model
                    diffusion_model_name = next(iter(calibration_dataset))
                    diffusion_model = getattr(self.model, diffusion_model_name).model
                    quantization_configs[diffusion_model_name] = _get_hybrid_mixed_quantization_config(
                        diffusion_model, quantization_config, **kwargs
                    )

                    # Apply weight-only quantization to all SD submodels except UNet/Transformer
                    quantization_config_copy = quantization_config.clone()
                    quantization_config_copy.dataset = None
                    quantization_config_copy.quant_method = OVQuantizationMethod.DEFAULT
                    default_config = quantization_config_copy
                else:
                    # The model may be for example OVModelForImageClassification, OVModelForAudioClassification, etc.
                    quantization_configs["model"] = quantization_config
            elif isinstance(quantization_config, OVQuantizationConfig):
                #
                # Full quantization
                #
                if is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
                    diffusion_model_name = next(iter(calibration_dataset))
                    quantization_configs[diffusion_model_name] = quantization_config
                    default_config = OVWeightQuantizationConfig(bits=8)
                elif isinstance(self.model, OVModelForVisualCausalLM):
                    quantization_configs["lm_model"] = quantization_config
                    vision_embedding_submodel_names = [
                        f"{name}_model"
                        for name, component in self.model.components.items()
                        if isinstance(component, OVVisionEmbedding)
                    ]
                    for submodel_name in vision_embedding_submodel_names:
                        quantization_configs[submodel_name] = quantization_config
                    default_config = OVWeightQuantizationConfig(bits=8, sym=True)
                else:
                    default_config = quantization_config
            elif isinstance(quantization_config, OVMixedQuantizationConfig):
                #
                # Mixed quantization
                #
                if is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
                    raise NotImplementedError("Mixed precision quantization isn't supported for diffusers.")

                default_config = quantization_config
            elif not isinstance(quantization_config, OVPipelineQuantizationConfig):
                raise ValueError(f"Unsupported type of quantization config: {type(quantization_config)}")

        pipeline_quantization_config = (
            quantization_config
            if isinstance(quantization_config, OVPipelineQuantizationConfig)
            else OVPipelineQuantizationConfig(quantization_configs, default_config=default_config)
        )

        for submodel_name in self.model.ov_submodels:
            config = pipeline_quantization_config.quantization_configs.get(
                submodel_name, pipeline_quantization_config.default_config
            )
            if config is None:
                continue
            submodel = self.model.ov_submodels[submodel_name]
            nncf_dataset = calibration_dataset.get(submodel_name, None) if calibration_dataset else None

            if isinstance(config, OVWeightQuantizationConfig) and config.quant_method == OVQuantizationMethod.HYBRID:
                config = _get_hybrid_mixed_quantization_config(submodel, config, **kwargs)

            if dataset_was_built_from_config and nncf_dataset is not None and nncf_dataset.get_length() is not None:
                # For datasets built from the quantization config, override num_samples per submodel
                config = config.clone()
                config.num_samples = nncf_dataset.get_length()

            if isinstance(config, OVWeightQuantizationConfig):
                if config.bits == 8:
                    # 8-bit weight only data-aware quantization is not supported
                    nncf_dataset = None
                # Weight only quantization is performed in-place
                _weight_only_quantization(submodel, config, nncf_dataset, **kwargs)
            elif isinstance(config, (OVQuantizationConfig, OVMixedQuantizationConfig)):
                if nncf_dataset is None:
                    raise ValueError(
                        f"Calibration dataset for submodel {submodel_name} is required to run quantization."
                    )
                if isinstance(config, OVQuantizationConfig):
                    quantized_model = _full_quantization(submodel, config, nncf_dataset, **kwargs)
                else:
                    quantized_model = _mixed_quantization(submodel, config, nncf_dataset, **kwargs)

                # Replace the original model with the quantized model
                if isinstance(self.model, OVModelForVisualCausalLM):
                    # Special handling of submodels in OVModelForVisualCausalLM
                    # TODO (nikita.savelyevv): Implement a proper fix including other model types
                    if submodel_name == "lm_model":
                        self.model.language_model.model = quantized_model
                    elif submodel_name == "text_embeddings_model":
                        self.model.language_model.text_emb_model = quantized_model
                    elif submodel_name == "vision_embeddings_model":
                        self.model.vision_embeddings.model = quantized_model
                if isinstance(getattr(self.model, submodel_name), openvino.Model):
                    setattr(self.model, submodel_name, quantized_model)
                elif isinstance(getattr(getattr(self.model, submodel_name), "model"), openvino.Model):
                    setattr(getattr(self.model, submodel_name), "model", quantized_model)
                else:
                    raise RuntimeError("Can't locate OpenVINO model to replace it with the quantized one.")
            else:
                raise ValueError(f"Unsupported type of quantization config: {type(config)}.")

        self.model.clear_requests()

        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_directory)
            ov_config.save_pretrained(save_directory)

    @staticmethod
    def _save_pretrained(model: openvino.Model, output_path: str):
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
        streaming: bool = False,
        **dataset_kwargs,
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
            streaming (`bool`, defaults to `False`):
                Whether to load dataset in streaming mode.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """

        # TODO: consider in the future for this method to return OVCalibrationDataset instance from either datasets.Dataset instance or its name as input.
        #  This way OVQuantizer.quantize() will accept fully ready OVCalibrationDataset instance and `batch_size` and `data_collator` arguments can be removed.
        #  Example usage in such scenario:
        #  ```
        #  calibration_dataset: OVCalibrationDataset = ov_quantizer.get_calibration_dataset(ov_config, dataset_name, ..., batch_size, data_collator)
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
            streaming,
            **dataset_kwargs,
        )


def _weight_only_quantization(
    model: openvino.Model,
    quantization_config: Union[OVWeightQuantizationConfig, Dict],
    calibration_dataset: Optional[Union[nncf.Dataset, Iterable]] = None,
    **kwargs,
) -> openvino.Model:
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
            # This already should not be used, deprecation warning is added just in case
            logger.warning("Providing calibration dataset as an iterable will be deprecated in optimum-intel v1.25.")
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

    advanced_parameters = wc_kwargs.get("advanced_parameters")
    if advanced_parameters is not None and advanced_parameters.statistics_path is not None and dataset is None:
        # Graceful handling of unnecessary statistics_path
        wc_kwargs["advanced_parameters"] = dataclasses.replace(advanced_parameters, statistics_path=None)

    compressed_model = nncf.compress_weights(
        model,
        dataset=dataset,
        **wc_kwargs,
    )

    _remove_f16_kv_cache_precision_flag(compressed_model)
    _add_nncf_version_flag(compressed_model)

    return compressed_model


def _full_quantization(
    model: openvino.Model,
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
    _add_nncf_version_flag(quantized_model)

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


def _get_hybrid_mixed_quantization_config(
    model: openvino.Model,
    quantization_config: OVWeightQuantizationConfig,
    **kwargs,
) -> OVMixedQuantizationConfig:
    """
    Returns mixed quantization config representing hybrid quantization

    Args:
        model (`openvino.Model`):
            The OpenVINO model for applying quantization.
        quantization_config (`OVWeightQuantizationConfig`):
            The configuration containing the parameters related to quantization.
    Returns:
        The mixed quantization config representing hybrid quantization.
    """

    # Hybrid quantization means that we quantize
    #  (1) weights of MatMul and Embedding layers
    #  (2) activations of other layers.

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

    return mixed_quantization_config


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
        model (`openvino.Model`):
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


def _add_nncf_version_flag(model: openvino.Model) -> openvino.Model:
    model.set_rt_info(_nncf_version, ["optimum", "nncf_version"])
    return model


@contextmanager
def numpy_seed(seed: int):
    """
    Context manager to set the numpy random seed.
    """
    old_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(old_state)
