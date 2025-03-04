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
import warnings
from typing import Union, List, Any, Tuple, Dict, Optional, Iterable, Callable, Sized

import nncf
import openvino
import requests
import torch
import transformers
from PIL.Image import Image
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from nncf.torch.initialization import PTInitializingDataLoader
from torch.utils.data import RandomSampler, DataLoader
from tqdm import tqdm
from transformers import DataCollator, default_data_collator, AutoTokenizer, AutoProcessor

from optimum.intel import is_accelerate_available, OVModelForCausalLM, OVModelForVisualCausalLM, \
    OVModelForSpeechSeq2Seq, OVDiffusionPipeline
from optimum.intel.openvino.modeling_decoder import OVBaseDecoderModel
from optimum.intel.openvino.quantization import OVQuantizationConfigBase
from optimum.intel.openvino.utils import PREDEFINED_VISUAL_LM_DATASETS, PREDEFINED_SPEECH_TO_TEXT_DATASETS, \
    PREDEFINED_DIFFUSION_DATASETS
from optimum.intel.utils.import_utils import is_datasets_available, DATASETS_IMPORT_ERROR, is_datasets_version

if is_datasets_available():
    from datasets import Dataset

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
    def __init__(self, model: transformers.PreTrainedModel, seed: int = 42, **kwargs):
        self.model = model
        self.seed = seed
        # TODO: deprecate because model.forward() may not be the method which is called during inference, for example there is model.generate()
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())

    def build_from_dataset(
        self,
        quantization_config: OVQuantizationConfigBase,
        dataset: Union["Dataset", Sized],
        batch_size: Optional[int] = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
    ) -> Dict[str, nncf.Dataset]:
        # TODO: deprecate remove_unused_columns ?
        
        dataloader = self._get_calibration_dataloader(dataset, batch_size, data_collator, remove_unused_columns)

        if isinstance(self.model, OVBaseDecoderModel):
            return self._prepare_decoder_calibration_data(quantization_config, dataloader)
        elif isinstance(self.model, OVModelForVisualCausalLM):
            return self._prepare_visual_causal_lm_calibration_data(quantization_config, dataloader)
        elif isinstance(self.model, OVModelForSpeechSeq2Seq):
            return self._prepare_speech_to_text_calibration_data(quantization_config, dataloader)
        elif isinstance(self.model, OVDiffusionPipeline):
            return self._prepare_diffusion_calibration_data(quantization_config, dataloader)
        else:
            raise Exception

    def build_from_dataset_name(
        self,
        quantization_config: OVQuantizationConfigBase,
        dataset_name: str,
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
    ) -> Dict[str, nncf.Dataset]:
        """
        Create the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                in generic formats and optionally a dataset script, if it requires some code to read the data files.
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
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """
        # TODO: deprecate remove_unused_columns ?
        
        dataset = self._load_dataset(
            dataset_name,
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

    def build_from_quantization_config(self, config: OVQuantizationConfigBase) -> Dict[str, nncf.Dataset]:
        if isinstance(self, OVModelForCausalLM):
            return self._prepare_causal_lm_calibration_data(self, config)
        elif isinstance(self, (OVModelForVisualCausalLM, OVModelForSpeechSeq2Seq)):
            if config.processor is None:
                raise ValueError(
                    "`processor` must be specified in order to run data-aware quantization. Please provide it as a"
                    "model id, or a path to a directory containing all the required configuration files."
                )

            trc = config.trust_remote_code
            processor = AutoProcessor.from_pretrained(config.processor, trust_remote_code=trc)
            if isinstance(self, OVModelForVisualCausalLM):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=trc)
                    tokenizer_error = None
                except Exception as tokenizer_error:  # noqa: F841
                    tokenizer = None

                dataset_metadata = PREDEFINED_VISUAL_LM_DATASETS[config.dataset]

                def preprocess_function(item):
                    inputs_metadata = dataset_metadata["inputs"]
                    instruction = item[inputs_metadata["instruction"]]
                    image_url = item[inputs_metadata["image_url"]]

                    image = Image.open(requests.get(image_url, stream=True).raw)

                    try:
                        inputs = self.model.preprocess_inputs(
                            text=instruction, image=image, processor=processor, tokenizer=tokenizer,
                            config=self.model.config
                        )
                    except ValueError as value_error:
                        if "Tokenizer is required." in str(value_error) and tokenizer_error is not None:
                            raise tokenizer_error
                        raise value_error

                    return inputs

                return self.build_from_dataset_name(
                    config,
                    config.dataset,
                    dataset_split=dataset_metadata["split"],
                    preprocess_function=preprocess_function,
                    trust_remote_code=trc,
                )
            elif isinstance(self.model, OVModelForSpeechSeq2Seq):
                dataset_metadata = PREDEFINED_SPEECH_TO_TEXT_DATASETS[config.dataset]

                def preprocess_function(item):
                    audio = item
                    for key_name in dataset_metadata["inputs"]["audio"]:
                        audio = audio[key_name]

                    sampling_rate = item
                    for key_name in dataset_metadata["inputs"]["sampling_rate"]:
                        sampling_rate = sampling_rate[key_name]

                    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

                    return input_features

                return self.build_from_dataset_name(
                    config,
                    dataset_metadata["id"],
                    dataset_metadata["name"],
                    dataset_metadata["split"],
                    preprocess_function=preprocess_function,
                    trust_remote_code=trc,
                    streaming=dataset_metadata["streaming"],
                )
            else:
                raise Exception
        elif isinstance(self, OVDiffusionPipeline):
            dataset = config.dataset
            if isinstance(dataset, str):
                dataset_name = dataset
                dataset_metadata = PREDEFINED_DIFFUSION_DATASETS[dataset_name]

                def preprocess_function(item):
                    return {inp_name: item[column] for inp_name, column in dataset_metadata["inputs"].items()}

                dataset = self._load_dataset(
                    dataset_name,
                    dataset_split=dataset_metadata["split"],
                    preprocess_function=preprocess_function,
                    streaming=dataset_metadata["streaming"],
                )
            elif not(isinstance(dataset, list) and all(isinstance(it, str) for it in dataset)):
                raise Exception

            return self.build_from_dataset(config, dataset)

    def _load_dataset(
        self,
        dataset_name: str,
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
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """
        # TODO: deprecate remove_unused_columns ?
        if not is_datasets_available():
            # TODO: update name
            raise ValueError(DATASETS_IMPORT_ERROR.format("OVQuantizer.get_calibration_dataset"))

        from datasets import load_dataset

        datasets_kwargs = {"name": dataset_config_name, "split": dataset_split, "token": token, "cache_dir": cache_dir, "streaming": streaming}
        if is_datasets_version(">=", "2.20.0"):
            datasets_kwargs["trust_remote_code"] = trust_remote_code

        dataset = load_dataset(dataset_name, **datasets_kwargs)
        dataset = dataset.shuffle(seed=self.seed)

        if preprocess_function is not None:
            dataset = dataset.map(preprocess_function, batched=preprocess_batch)

        return dataset

    def _get_calibration_dataloader(
        self,
        dataset: Union["Dataset", Sized],
        batch_size: Optional[int] = 1,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = False,
    ) -> OVDataLoader:
        if not is_datasets_available():
            # TODO: update name
            raise ValueError(DATASETS_IMPORT_ERROR.format("OVQuantizer.get_calibration_dataset"))
        
        from datasets import Dataset

        data_collator = data_collator if data_collator is not None else default_data_collator

        if remove_unused_columns and isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        sampler = RandomSampler(dataset, generator=generator)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, collate_fn=data_collator, drop_last=False
        )
        return OVDataLoader(dataloader)

    def _remove_unused_columns(self, dataset: "Dataset"):
        # TODO: deprecate because model.forward() may not be the method which is called during inference, for example there is model.generate()
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)

    def _prepare_decoder_calibration_data(
        self, quantization_config: OVQuantizationConfigBase, dataloader: OVDataLoader
    ) -> Dict[str, nncf.Dataset]:
        # Prefetch past_key_values
        self.model.update_pkv_precision(True)
        self.model.compile()
        collected_inputs = []

        num_samples = quantization_config.num_samples or 200
        self.model.request = InferRequestWrapper(self.model.request, collected_inputs)
        try:
            for data in tqdm(dataloader, desc="Collecting calibration data"):
                self.model.generate(**data, max_new_tokens=1)
                if len(collected_inputs) >= num_samples:
                    break
        finally:
            self.model.request = self.model.request.request

        return {"model": nncf.Dataset(collected_inputs)}

    def _prepare_causal_lm_calibration_data(self, config: OVQuantizationConfigBase, seqlen: Optional[int] = 32) -> Dict[str, nncf.Dataset]:
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

        return {"model": calibration_dataset}

    def _prepare_visual_causal_lm_calibration_data(self, quantization_config: OVQuantizationConfigBase, dataloader: OVDataLoader) -> Dict[str, nncf.Dataset]:
        calibration_data = []
        num_samples = quantization_config.num_samples or 32
        for inputs in tqdm(dataloader, desc="Collecting calibration dataset", total=num_samples):
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

        return {"language_model": nncf.Dataset(calibration_data)}

    def _prepare_speech_to_text_calibration_data(self, quantization_config: OVQuantizationConfigBase, dataloader: OVDataLoader) -> Dict[str, nncf.Dataset]:
        encoder_calibration_data = []
        encoder_model = self.model.encoder
        encoder_model._compile()
        encoder_model.request = InferRequestWrapper(
            encoder_model.request, encoder_calibration_data, apply_caching=True
        )

        decoder_calibration_data = []
        decoder_model = self.model.decoder
        decoder_model._compile()
        decoder_model.request = InferRequestWrapper(
            decoder_model.request, decoder_calibration_data, apply_caching=True
        )

        decoder_w_p_model = None
        decoder_w_p_calibration_data = []
        if self.model.decoder_with_past_model is not None:
            decoder_w_p_model = self.model.decoder_with_past
            decoder_w_p_model._compile()
            decoder_w_p_model.request = InferRequestWrapper(
                decoder_w_p_model.request, decoder_w_p_calibration_data, apply_caching=True
            )

        try:
            # Download audio inputs beforehand to avoid possible connection issues
            num_samples = quantization_config.num_samples or 32
            audio_inputs = list(tqdm(dataloader, desc="Downloading audio inputs", total=num_samples))

            for input_features in tqdm(audio_inputs, desc="Collecting calibration data"):
                self.model.generate(input_features)
        finally:
            encoder_model.request = encoder_model.request.request
            decoder_model.request = decoder_model.request.request
            if decoder_w_p_model is not None:
                decoder_w_p_model.request = decoder_w_p_model.request.request

        datasets = {
            "encoder_model": nncf.Dataset(encoder_calibration_data),
            "decoder_model": nncf.Dataset(decoder_calibration_data),
        }
        if len(decoder_w_p_calibration_data) > 0:
            datasets["decoder_with_past_model"] = nncf.Dataset(decoder_w_p_calibration_data)
        return datasets

    def _prepare_diffusion_calibration_data(
        self, quantization_config: OVQuantizationConfigBase, dataloader: OVDataLoader
    ) -> Dict[str, nncf.Dataset]:
        self.model.compile()

        diffuser_model_name = "unet" if self.model.unet is not None else "transformer"
        diffuser = getattr(self, diffuser_model_name)

        size = diffuser.config.get("sample_size", 64) * self.model.vae_scale_factor
        height, width = 2 * (min(size, 512),)

        # TODO: move the logic below to ov_quantizer
        # if dataset is not None:
        #     if isinstance(dataset, nncf.Dataset):
        #         return dataset
        #     if is_datasets_available() and isinstance(dataset, Dataset):
        #         dataset = dataset.select_columns(["caption"])
        #
        #     def transform_fn(data_item):
        #         return data_item if isinstance(data_item, (list, dict)) else [data_item]

        num_samples = quantization_config.num_samples or 200
        calibration_data = []
        try:
            diffuser.request = InferRequestWrapper(diffuser.request, calibration_data)

            for inputs in tqdm(dataloader, desc="Collecting calibration data"):
                if isinstance(inputs, dict):
                    self.model(**inputs, height=height, width=width)
                elif isinstance(inputs, str):
                    self.model(inputs, height=height, width=width)
                else:
                    self.model(*inputs, height=height, width=width)
                if len(calibration_data) >= num_samples:
                    break
        finally:
            diffuser.request = diffuser.request.request

        return {diffuser_model_name: nncf.Dataset(calibration_data[:num_samples])}
