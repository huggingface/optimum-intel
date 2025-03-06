# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import logging
from collections import UserDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import nncf
import openvino
import requests
import torch
import transformers
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from nncf.torch.initialization import PTInitializingDataLoader
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, DataCollator, default_data_collator

from optimum.intel.openvino.utils import (
    PREDEFINED_DIFFUSION_DATASETS,
    PREDEFINED_SPEECH_TO_TEXT_DATASETS,
    PREDEFINED_VISUAL_LM_DATASETS,
)
from optimum.intel.utils.import_utils import (
    DATASETS_IMPORT_ERROR,
    is_accelerate_available,
    is_datasets_available,
    is_datasets_version,
    is_diffusers_available,
)

from .configuration import OVQuantizationConfigBase


if is_datasets_available():
    from datasets import Dataset

logger = logging.getLogger(__name__)


class CalibrationDataset(UserDict):
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
        return openvino.Tensor(self.request.results[name])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)


class OVCalibrationDatasetBuilder:
    """
    A class to build calibration datasets for quantization with NNCF.

    Allows to build a calibration dataset from:
        - a `datasets.Dataset` object
        - a name of the dataset from `datasets`
        - a quantization config object containing dataset specification

    Returns calibration dataset as an instance of `CalibrationDataset` containing an `nncf.Dataset` for each model component.
    For example, for a sequence-to-sequence model with `encoder_model` and `decoder_model` components, the dictionary
    will contain two keys: `encoder_model` and `decoder_model`.
    """

    def __init__(self, model: transformers.PreTrainedModel, seed: int = 42):
        """

        Args:
            model (`transformers.PreTrainedModel`):
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

    def build_from_dataset(
        self,
        quantization_config: OVQuantizationConfigBase,
        dataset: Union["Dataset", List],
        batch_size: Optional[int] = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
    ) -> CalibrationDataset:
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
            A calibration dataset as an instance of `CalibrationDataset` containing an `nncf.Dataset` for each model component.
        """
        from optimum.intel import OVModelForVisualCausalLM
        from optimum.intel.openvino.modeling_decoder import OVBaseDecoderModel
        from optimum.intel.openvino.modeling_seq2seq import _OVModelForWhisper

        if is_diffusers_available():
            from optimum.intel.openvino.modeling_diffusion import OVDiffusionPipeline

        if isinstance(dataset, list):
            logger.warning(
                "Providing dataset as a list is deprecated and will be removed in optimum-intel v1.25. "
                "Please provide it as `datasets.Dataset`."
            )

        if isinstance(self.model, (OVModelForVisualCausalLM, _OVModelForWhisper)) or (
            is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline)
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
            elif is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
                return self._prepare_diffusion_calibration_data(quantization_config, dataset)
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
                return CalibrationDataset({"model": nncf.Dataset(dataloader)})

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
        trust_remote_code: bool = False,
        streaming: bool = False,
        batch_size: Optional[int] = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
    ) -> CalibrationDataset:
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
            trust_remote_code (`bool`, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            streaming (`bool`, defaults to `False`):
                Whether to load dataset in streaming mode.
            batch_size (`int`, defaults to 1):
                The number of calibration samples to load per batch.
            data_collator (`DataCollator`, *optional*):
                The function to use to form a batch from a list of elements of the calibration dataset.
            remove_unused_columns (`bool`, defaults to `False`):
                Whether to remove the columns unused by the model forward method.
        Returns:
            A calibration dataset as an instance of `CalibrationDataset` containing an `nncf.Dataset` for each model component.
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
            trust_remote_code,
            streaming,
        )

        return self.build_from_dataset(quantization_config, dataset, batch_size, data_collator, remove_unused_columns)

    def build_from_quantization_config(self, config: OVQuantizationConfigBase) -> CalibrationDataset:
        """
        Builds a calibration dataset from a quantization config object. Namely, `quantization_config.dataset` property
        is used to infer dataset name.

        Args:
            config (`OVQuantizationConfigBase`):
                The quantization configuration object.
        Returns:
            A calibration dataset as an instance of `CalibrationDataset` containing an `nncf.Dataset` for each model component.
        """
        from optimum.intel import OVModelForCausalLM, OVModelForVisualCausalLM
        from optimum.intel.openvino.modeling_seq2seq import _OVModelForWhisper

        if is_diffusers_available():
            from optimum.intel.openvino.modeling_diffusion import OVDiffusionPipeline

        if config.dataset is None:
            raise ValueError("Please provide a dataset for calibration.")

        if isinstance(self.model, OVModelForCausalLM):
            return self._prepare_causal_lm_calibration_data(config)
        elif isinstance(self.model, (OVModelForVisualCausalLM, _OVModelForWhisper)):
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
                    trust_remote_code=config.trust_remote_code,
                )
            elif isinstance(self.model, _OVModelForWhisper):
                dataset_metadata = PREDEFINED_SPEECH_TO_TEXT_DATASETS[config.dataset]
                return self.build_from_dataset_name(
                    config,
                    dataset_metadata["id"],
                    num_samples=config.num_samples,  # This is an upper bound on how many audios are needed
                    dataset_config_name=dataset_metadata["name"],
                    dataset_split=dataset_metadata["split"],
                    trust_remote_code=config.trust_remote_code,
                    streaming=dataset_metadata["streaming"],
                )
            else:
                raise Exception
        elif is_diffusers_available() and isinstance(self.model, OVDiffusionPipeline):
            if isinstance(config.dataset, str):
                dataset_name = config.dataset
                dataset_metadata = PREDEFINED_DIFFUSION_DATASETS[dataset_name]

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
        trust_remote_code: bool = False,
        streaming: bool = False,
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
            trust_remote_code (`bool`, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            streaming (`bool`, defaults to `False`):
                Whether to load dataset in streaming mode.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """
        if not is_datasets_available():
            raise ValueError(DATASETS_IMPORT_ERROR.format("OVCalibrationDatasetBuilder.load_dataset"))

        from datasets import load_dataset

        datasets_kwargs = {
            "name": dataset_config_name,
            "split": dataset_split,
            "token": token,
            "cache_dir": cache_dir,
            "streaming": streaming,
        }
        if is_datasets_version(">=", "2.20.0"):
            datasets_kwargs["trust_remote_code"] = trust_remote_code

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
    ) -> CalibrationDataset:
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
                self.model.generate(**data, max_new_tokens=1)
                if len(collected_inputs) >= num_samples:
                    break
        finally:
            self.model.request = self.model.request.request

        return CalibrationDataset(nncf.Dataset(collected_inputs))

    def _prepare_causal_lm_calibration_data(
        self, config: OVQuantizationConfigBase, seqlen: int = 32
    ) -> CalibrationDataset:
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

        return CalibrationDataset(calibration_dataset)

    def _prepare_visual_causal_lm_calibration_data(
        self, config: OVQuantizationConfigBase, dataset: "Dataset"
    ) -> CalibrationDataset:
        """
        Prepares calibration data for VLM pipelines. Currently, collects data only for a language model component.
        """
        processor = AutoProcessor.from_pretrained(config.processor, trust_remote_code=config.trust_remote_code)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=config.trust_remote_code)
            tokenizer_error = None
        except Exception as tokenizer_error:  # noqa: F841
            tokenizer = None

        dataset_metadata = PREDEFINED_VISUAL_LM_DATASETS[config.dataset]

        calibration_data = []
        num_samples = config.num_samples or 32
        for item in tqdm(dataset, desc="Collecting calibration dataset", total=num_samples):
            instruction = item[dataset_metadata["inputs"]["instruction"]]
            image_url = item[dataset_metadata["inputs"]["image_url"]]
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

            try:
                inputs = self.model.preprocess_inputs(
                    text=instruction, image=image, processor=processor, tokenizer=tokenizer, config=self.model.config
                )
            except ValueError as value_error:
                if "Tokenizer is required." in str(value_error) and tokenizer_error is not None:
                    raise tokenizer_error
                raise value_error

            input_ids = inputs.get("input_ids")
            position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to(input_ids.device)

            inputs_embeds, attention_mask, position_ids = self.model.get_multimodal_embeddings(
                **inputs,
                position_ids=position_ids,
            )

            language_model_inputs = self.model.language_model.prepare_inputs(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )

            calibration_data.append(language_model_inputs)

            if len(calibration_data) >= num_samples:
                break

        return CalibrationDataset({"lm_model": nncf.Dataset(calibration_data)})

    def _prepare_speech_to_text_calibration_data(
        self, config: OVQuantizationConfigBase, dataset: "Dataset"
    ) -> CalibrationDataset:
        """
        Prepares calibration data for speech-to-text pipelines by inferring it on a dataset and collecting incurred inputs.
        """
        from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder

        models: Dict[str, Union[OVEncoder, OVDecoder]] = {}
        collected_inputs: Dict[str, List[Dict[str, Any]]] = {}
        for submodel_name in self.model.ov_submodels:
            ov_component_name = "_".join(submodel_name.split("_")[:-1])  # e.g. "encoder_model" -> "encoder"
            ov_component: Union[OVEncoder, OVDecoder] = getattr(self.model, ov_component_name)
            models[ov_component_name] = ov_component
            collected_inputs[ov_component_name] = []
            ov_component._compile()
            ov_component.request = InferRequestWrapper(
                ov_component.request, collected_inputs[ov_component_name], apply_caching=True
            )

        try:
            processor = AutoProcessor.from_pretrained(config.processor, trust_remote_code=config.trust_remote_code)

            # Download audio inputs beforehand to avoid possible connection issues
            num_samples = config.num_samples or 32
            downloaded_dataset = list(tqdm(dataset, desc="Downloading audio inputs", total=num_samples))

            for item in tqdm(downloaded_dataset, desc="Collecting calibration data"):
                audio = item["audio"]["array"]
                sampling_rate = item["audio"]["sampling_rate"]
                input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
                self.model.generate(input_features)
        finally:
            for model in models.values():
                model.request = model.request.request

        calibration_data = {}
        for model_name, model_data in collected_inputs.items():
            calibration_data[f"{model_name}_model"] = nncf.Dataset(model_data)
        return CalibrationDataset(calibration_data)

    def _prepare_diffusion_calibration_data(
        self, config: OVQuantizationConfigBase, dataset: "Dataset"
    ) -> CalibrationDataset:
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
            diffuser.request = InferRequestWrapper(diffuser.request, calibration_data)

            for item in tqdm(dataset, desc="Collecting calibration data"):
                prompt = (
                    item[PREDEFINED_DIFFUSION_DATASETS[config.dataset]["prompt_column_name"]]
                    if isinstance(item, dict)
                    else item
                )
                self.model(prompt, height=height, width=width)
                if len(calibration_data) >= num_samples:
                    break
        finally:
            diffuser.request = diffuser.request.request

        return CalibrationDataset({diffuser_model_name: nncf.Dataset(calibration_data[:num_samples])})

    def _remove_unused_columns(self, dataset: "Dataset"):
        # TODO: deprecate because model.forward() may not be the method which is called during inference,
        #  for example there is model.generate()
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)
