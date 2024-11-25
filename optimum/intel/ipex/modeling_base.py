#  Copyright 2024 The HuggingFace Team. All rights reserved.
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
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Union

import torch
import transformers
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
    is_torch_xpu_available,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.candidate_generator import _crop_past_key_values
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.auto.auto_factory import _get_model_class as get_model_class
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager
from optimum.exporters.tasks import make_backend_config_constructor_for_task
from optimum.modeling_base import OptimizedModel
from optimum.utils import NormalizedConfigManager

from ...exporters.ipex.cache_utils import IPEXPagedCache
from ...exporters.ipex.model_config import ipex_onnx_config
from ...exporters.ipex.model_patcher import (
    _IPEX_EXPORTED_GENERATION_TASKS,
    _IPEX_MINIMUM_VERSION_FOR_PATCHING,
    _patch_model,
)
from ..generation.modeling import get_float_type
from ..utils.constant import _TASK_ALIASES
from ..utils.import_utils import is_ipex_version, is_transformers_version
from ..utils.modeling_utils import recursive_to_device


logger = logging.getLogger(__name__)


_IPEX_SUPPORT_MODEL_TYPES = ("llama", "bert", "vit", "falcon", "gpt2")
_IPEX_EXPORTED_GENERATION_METHODS = ("sample", "greedy_search", "beam_sample", "beam_search", "assisted_generation")


def _is_patched_with_ipex(model, task, use_cache: bool = True):
    if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
        return False
    if not use_cache and task in _IPEX_EXPORTED_GENERATION_TASKS:
        return False
    return model.config.model_type in _IPEX_SUPPORT_MODEL_TYPES


def _prepare_inputs_for_ipex_model(model, task, use_cache):
    task = _TASK_ALIASES.get(task, task)
    signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.__call__)
    if _is_patched_with_ipex(model, task, use_cache) and model.config.model_type in ipex_onnx_config:
        onnx_config_class = make_backend_config_constructor_for_task(
            ipex_onnx_config[model.config.model_type], task=task
        )
    else:
        onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
    float_dtype = get_float_type(model.dtype)
    if "text-generation" in task:
        onnx_config = onnx_config_class(
            model.config, use_past=use_cache, use_past_in_inputs=use_cache, float_dtype=float_dtype
        )
    else:
        onnx_config = onnx_config_class(model.config)

    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")

    # Check attention_mask shape
    if _is_patched_with_ipex(model, task, use_cache) and model.config.model_type in ipex_onnx_config:
        past_len = dummy_inputs["past_key_values"][0][0].shape[-2]
        input_len = dummy_inputs["input_ids"].shape[-1]
        attention_len = dummy_inputs["attention_mask"].shape[-1]
        if attention_len != input_len + past_len:
            dummy_inputs["attention_mask"] = torch.ones([dummy_inputs["input_ids"].shape[0], input_len + past_len]).to(
                dummy_inputs["input_ids"].dtype
            )

    return {key: dummy_inputs[key] for key in signature.parameters if dummy_inputs.get(key, None) is not None}


class IPEXModel(OptimizedModel):
    auto_model_class = AutoModel
    export_feature = "feature-extraction"
    base_model_prefix = "ipex_model"
    main_input_name = "input_ids"
    output_name = "last_hidden_state"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        export: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        warmup: bool = True,
        **kwargs,
    ):
        if is_torch_xpu_available(check_device=True):
            self._device = torch.device("xpu:0")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        config = config or model.config

        OptimizedModel.__init__(self, model=model, config=config)

        self.model.to(self._device)
        self._dtype = self.model.dtype if self.model.dtype is not None else torch.float32
        self.use_cache = kwargs.get("use_cache", False)
        self.model_save_dir = model_save_dir
        self._add_patch = _is_patched_with_ipex(model, self.export_feature, self.use_cache)

        self.input_names = set(inspect.signature(model.forward).parameters)

        if self._add_patch:
            model = _patch_model(model)
        # Registers the IPEXModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)
        if warmup:
            self._init_warmup()

    @classmethod
    def _from_transformers(cls, *args, **kwargs):
        return cls._from_pretrained(*args, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Union[str, Path] = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        trust_remote_code: bool = False,
        file_name: Optional[str] = WEIGHTS_NAME,
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
            use_auth_token (Optional[Union[bool, str]], defaults to `None`):
                Deprecated. Please use `token` instead.
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            subfolder (`str`, *optional*)
                In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can specify the folder name here.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            torch_dtype (`Optional[Union[str, "torch.dtype"]]`, *optional*)
                float16 or bfloat16 or float32: load in a specified dtype, ignoring the model config.torch_dtype if one exists. If not specified, the model will get loaded in float32.
            trust_remote_code (`bool`, *optional*)
                Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the model repository.
            file_name (`str`, *optional*):
                The file name of the model to load. Overwrites the default file name and allows one to load the model
                with a different name.
        """
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "Both the arguments `use_auth_token` and `token` were specified, which is not supported. Please specify only `token`."
                )
            token = use_auth_token

        commit_hash = kwargs.pop("_commit_hash", None)

        model_kwargs = {
            "revision": revision,
            "token": token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }

        task = cls.export_feature
        model = TasksManager.get_model_from_task(
            task,
            model_id,
            library_name="transformers",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            _commit_hash=commit_hash,
            **model_kwargs,
        )
        config = model.config
        return cls(model, config=config, export=True, **kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        if getattr(self.config, "torchscript", None):
            output_path = os.path.join(save_directory, WEIGHTS_NAME)
            torch.jit.save(self.model, output_path)
        else:
            logger.warning("The module is not a torchscript model, will be treated as a transformers model.")
            self.model.save_pretrained(save_directory, safe_serialization=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids

        if "position_ids" in self.input_names:
            inputs["position_ids"] = position_ids

        outputs = self._call_model(**inputs)
        if isinstance(outputs, dict):
            model_output = ModelOutput(**outputs)
        else:
            model_output = ModelOutput()
            model_output[self.output_name] = outputs[0]
        return model_output

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def model_dtype(self):
        logger.warning(
            "access to the `model_dtype` attribute is deprecated and will be removed after v1.18.0, please use `_dtype` instead."
        )
        return self._dtype

    @property
    def add_patch(self) -> bool:
        return self._add_patch

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def can_generate(self):
        return isinstance(self, GenerationMixin)

    def _call_model(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out

    def _init_warmup(self):
        # warmup, the first 2 forwards of an IPEX model include some preprocessing steps and
        # the results of the compute are unpredictable
        # TODO : add warmup for IPEX exported model
        if not self._add_patch:
            # use_cache = "past_key_values" in self.input_names
            dummy_inputs = _prepare_inputs_for_ipex_model(self, self.export_feature, self.use_cache)
            if self._device.type != "cpu":
                dummy_inputs = recursive_to_device(value=dummy_inputs, device=self._device)
            for _ in range(2):
                self(**dummy_inputs)


class IPEXModelForSequenceClassification(IPEXModel):
    auto_model_class = AutoModelForSequenceClassification
    export_feature = "text-classification"
    output_name = "logits"


class IPEXModelForTokenClassification(IPEXModel):
    auto_model_class = AutoModelForTokenClassification
    export_feature = "token-classification"
    output_name = "logits"


class IPEXModelForMaskedLM(IPEXModel):
    auto_model_class = AutoModelForMaskedLM
    export_feature = "fill-mask"
    output_name = "logits"


class IPEXModelForImageClassification(IPEXModel):
    auto_model_class = AutoModelForImageClassification
    export_feature = "image-classification"

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        inputs = {
            "pixel_values": pixel_values,
        }

        outputs = self._call_model(**inputs)
        return ModelOutput(**outputs) if isinstance(outputs, dict) else ModelOutput(logits=outputs[0])


class IPEXModelForAudioClassification(IPEXModel):
    auto_model_class = AutoModelForAudioClassification
    export_feature = "audio-classification"

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        inputs = {
            "input_values": input_values,
        }

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        outputs = self._call_model(**inputs)
        return ModelOutput(**outputs) if isinstance(outputs, dict) else ModelOutput(logits=outputs[0])


class IPEXModelForQuestionAnswering(IPEXModel):
    auto_model_class = AutoModelForQuestionAnswering
    export_feature = "question-answering"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids

        outputs = self._call_model(**inputs)
        start_logits = outputs["start_logits"] if isinstance(outputs, dict) else outputs[0]
        end_logits = outputs["end_logits"] if isinstance(outputs, dict) else outputs[1]
        return ModelOutput(start_logits=start_logits, end_logits=end_logits)


class IPEXModelForCausalLM(IPEXModel, GenerationMixin):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    _supports_cache_class = False
    _is_stateful = False

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        export: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        warmup: bool = True,
        **kwargs,
    ):
        # Perform the initial warmup at the end of __init__
        super().__init__(
            model, config, export=export, model_save_dir=model_save_dir, warmup=False, use_cache=use_cache
        )
        if self._add_patch:
            self._supports_cache_class = True
        GenerationMixin.__init__(self)

        model_type = self.config.model_type.replace("_", "-")
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(model_type)(self.config)

        self.config.is_decoder = True
        self.config.is_encoder_decoder = False

        self.generation_config = GenerationConfig.from_model_config(self.config)
        try:
            self.model_cls = get_class_from_dynamic_module(
                self.config.auto_map["AutoModelForCausalLM"], model_save_dir
            )
        except AttributeError:
            self.model_cls = get_model_class(self.config, AutoModelForCausalLM._model_mapping)

        if is_transformers_version(">=", "4.38.0") and model_type in {
            "llama",
            "phi",
            "persimmon",
            "mistral",
            "falcon",
            "gpt2",
        }:
            self.prepare_inputs_for_generation = _ipex_prepare_inputs_for_generation
        else:
            self.prepare_inputs_for_generation = self.model_cls.prepare_inputs_for_generation.__get__(self)

        if hasattr(self.model_cls, "_convert_to_standard_cache"):
            self._convert_to_standard_cache = self.model_cls._convert_to_standard_cache
        if hasattr(self.model_cls, "_convert_to_bloom_cache"):
            self._convert_to_bloom_cache = self.model_cls._convert_to_bloom_cache
        if warmup:
            self._init_warmup()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1. Prepare model inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "position_ids" in self.input_names or not self.input_names:
            inputs["position_ids"] = position_ids

        if self.use_cache:
            if past_key_values is None and self._add_patch:
                max_length = self.config.max_length + input_ids.shape[1]
                batch_size = input_ids.shape[0]
                past_key_values = IPEXPagedCache(
                    self.config, batch_size, max_length, input_ids.device, dtype=self.dtype
                )
            inputs["past_key_values"] = past_key_values

        # 2. Model forward
        outputs = self._call_model(**inputs)

        # 3. Process model outputs
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            past_key_values = outputs[1] if self.use_cache else None
        else:
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"] if self.use_cache else None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        generation_method = generation_config.get_generation_mode().value
        if generation_method not in _IPEX_EXPORTED_GENERATION_METHODS:
            raise ValueError(
                f"The generation method {generation_method} is not supported for IPEXModelForCausalLM for now, support methods are {_IPEX_EXPORTED_GENERATION_METHODS}"
            )

        return generation_config, model_kwargs

    def _reorder_cache(self, *args, **kwargs):
        return self.model._reorder_cache(*args, **kwargs)

    def generate(self, *args, **kwargs):
        new_kwargs = copy.deepcopy(kwargs)
        if is_ipex_version("<", "2.4.0") and self._add_patch and new_kwargs.get("assistant_model", None):
            raise ValueError(
                f"Assisted decoding is not supported for patched models if ipex < 2.4, support methods are {_IPEX_EXPORTED_GENERATION_METHODS}"
            )
        # Patch functions to support paged cache
        if self._add_patch:
            transformers.generation.utils.NEED_SETUP_CACHE_CLASSES_MAPPING["paged"] = IPEXPagedCache
            self.generation_config.cache_implementation = "paged"
            if is_transformers_version(">=", "4.45.0"):
                if "paged" not in transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS:
                    transformers.generation.configuration_utils.ALL_CACHE_IMPLEMENTATIONS.append("paged")
            if new_kwargs.get("generation_config", None):
                new_kwargs["generation_config"].cache_implementation = "paged"

        if self._add_patch and new_kwargs.get("assistant_model", None):
            transformers.generation.utils._crop_past_key_values = _ipex_crop_past_key_values
        elif self._add_patch:
            transformers.generation.candidate_generator._crop_past_key_values = _ipex_crop_past_key_values

        try:
            result = super().generate(*args, **new_kwargs)
        except Exception as e:
            transformers.generation.utils._crop_past_key_values = _crop_past_key_values
            transformers.generation.candidate_generator._crop_past_key_values = _crop_past_key_values
            raise e

        if self._add_patch and new_kwargs.get("assistant_model", None):
            transformers.generation.utils._crop_past_key_values = _crop_past_key_values
            transformers.generation.candidate_generator._crop_past_key_values = _crop_past_key_values

        return result


def _ipex_prepare_inputs_for_generation(
    input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    from transformers.cache_utils import Cache

    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            past_length = cache_length = past_key_values.get_seq_length()
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def _ipex_crop_past_key_values(model, past_key_values, max_length):
    if isinstance(model, IPEXModel) and _is_patched_with_ipex(model, "text-generation"):
        if isinstance(past_key_values, IPEXPagedCache):
            # .crop is an inplace op, returns None
            past_key_values = past_key_values.crop(max_length)
            return past_key_values
        else:
            raise ValueError("only support IPEXPagedCache input now")
    return _crop_past_key_values(model, past_key_values, max_length)
