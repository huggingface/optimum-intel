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


import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import intel_extension_for_pytorch as ipex
import torch
from huggingface_hub import hf_hub_download
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
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.auto.auto_factory import _get_model_class as get_model_class
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel
from optimum.utils import NormalizedConfigManager

from ..generation.modeling import jit_trace, prepare_jit_inputs
from ..utils.import_utils import is_torch_version
from ..utils.modeling_utils import MULTI_QUERY_ATTN_MODELS, patch_decoder_attention_mask


logger = logging.getLogger(__name__)


class IPEXModel(OptimizedModel):
    auto_model_class = AutoModel
    export_feature = "feature-extraction"
    base_model_prefix = "ipex_model"
    main_input_name = "input_ids"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        warmup: bool = True,
        **kwargs,
    ):
        OptimizedModel.__init__(self, model=model, config=config)
        # To do: add XPU support
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = self.config.torch_dtype if self.config.torch_dtype is not None else torch.float32
        self.model.to(self._device)
        self.model_save_dir = model_save_dir

        self.input_names = {
            inputs.debugName().split(".")[0] for inputs in model.graph.inputs() if inputs.debugName() != "self"
        }
        # Registers the IPEXModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)
        if warmup:
            self._init_warmup()

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        trust_remote_code: bool = False,
    ):
        if is_torch_version("<", "2.1.0"):
            raise ImportError("`torch>=2.0.0` is needed to trace your model")

        task = cls.export_feature
        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }

        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
        model = patch_decoder_attention_mask(model)
        model = ipex.optimize(model, dtype=torch_dtype, level="O1", auto_kernel_selection=True)
        traced_model = jit_trace(model, task, use_cache)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
        config.torchscript = True

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_cache=use_cache,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = WEIGHTS_NAME,
        local_files_only: bool = False,
        subfolder: str = "",
        **kwargs,
    ):
        # Load the model from local directory
        if os.path.isdir(model_id):
            model_cache_path = os.path.join(model_id, file_name)
            model_save_dir = model_id
        # Download the model from the hub
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                subfolder=subfolder,
            )
            model_save_dir = Path(model_cache_path).parent

        model = torch.jit.load(model_cache_path)
        torch.jit.freeze(model.eval())

        return cls(model, config=config, model_save_dir=model_save_dir, **kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        output_path = os.path.join(save_directory, WEIGHTS_NAME)
        torch.jit.save(self.model, output_path)

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
        return ModelOutput(**outputs) if isinstance(outputs, dict) else ModelOutput(logits=outputs[0])

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def can_generate(self):
        return isinstance(self, GenerationMixin)

    def _call_model(self, *args, **kwargs):
        try:
            with torch.autocast(self.device.type, self.dtype):
                out = self.model(*args, **kwargs)
        except RuntimeError:
            out = self.model(*args, **kwargs)
        return out

    def _init_warmup(self):
        # warmup, the first 2 forwards of an IPEX model include some preprocessing steps and
        # the results of the compute are unpredictable
        use_cache = "past_key_values" in self.input_names
        dummy_inputs = prepare_jit_inputs(self, self.export_feature, use_cache)
        for _ in range(2):
            self(**dummy_inputs)


class IPEXModelForSequenceClassification(IPEXModel):
    auto_model_class = AutoModelForSequenceClassification
    export_feature = "text-classification"


class IPEXModelForTokenClassification(IPEXModel):
    auto_model_class = AutoModelForTokenClassification
    export_feature = "token-classification"


class IPEXModelForMaskedLM(IPEXModel):
    auto_model_class = AutoModelForMaskedLM
    export_feature = "fill-mask"


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

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        warmup: bool = True,
        **kwargs,
    ):
        # Perform the initial warmup at the end of __init__
        super().__init__(model, config, model_save_dir=model_save_dir, warmup=False)

        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.model_dtype = kwargs.get("model_dtype", self.dtype)
        self.use_cache = "past_key_values" in self.input_names

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`. "
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )
        config.is_decoder = True
        config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(config)
        try:
            self.model_cls = get_class_from_dynamic_module(
                self.config.auto_map["AutoModelForCausalLM"], model_save_dir
            )
        except AttributeError:
            self.model_cls = get_model_class(self.config, AutoModelForCausalLM._model_mapping)
        self._reorder_cache = self.model_cls._reorder_cache.__get__(self)
        self.prepare_inputs_for_generation = self.model_cls.prepare_inputs_for_generation.__get__(self)
        if hasattr(self.model_cls, "_convert_to_standard_cache"):
            self._convert_to_standard_cache = self.model_cls._convert_to_standard_cache
        if hasattr(self.model_cls, "_convert_to_bloom_cache"):
            self._convert_to_bloom_cache = self.model_cls._convert_to_bloom_cache
        if warmup:
            self._init_warmup()

    def _prepare_past_key_values(self, input_ids):
        model_type = self.config.model_type.replace("_", "-")
        nb_pkv = 2
        num_layers = self.normalized_config.num_layers
        d_k = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        batch_size = input_ids.shape[0]

        if model_type in {"mistral", "llama"}:
            num_attention_heads = self.normalized_config.num_key_value_heads
        else:
            num_attention_heads = self.normalized_config.num_attention_heads

        if model_type == "bloom":
            shape_key = (batch_size * num_attention_heads, d_k, 0)
            shape_value = (batch_size * num_attention_heads, 0, d_k)
            key = torch.empty(size=shape_key, dtype=self.model_dtype, device=self._device)
            value = torch.empty(size=shape_value, dtype=self.model_dtype, device=self._device)
            past_key_values = tuple(
                tuple(key if idx % 2 == 0 else value for idx in range(nb_pkv)) for _ in range(num_layers)
            )
        elif model_type.replace("-", "_") in MULTI_QUERY_ATTN_MODELS:
            shape = (batch_size, 0, d_k * 2)
            pkv = torch.empty(size=shape, dtype=self.model_dtype, device=self._device)
            past_key_values = tuple(pkv for _ in range(num_layers))
        else:
            shape = (batch_size, num_attention_heads, 0, d_k)
            pkv = torch.empty(size=shape, dtype=self.model_dtype, device=self._device)
            past_key_values = tuple(tuple(pkv for _ in range(nb_pkv)) for _ in range(num_layers))

        return past_key_values

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

        if "position_ids" in self.input_names and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        if "position_ids" in self.input_names or not self.input_names:
            inputs["position_ids"] = position_ids

        if self.use_cache:
            if past_key_values is None:
                past_key_values = self._prepare_past_key_values(input_ids)

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
