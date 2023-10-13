#  Copyright 2023 The HuggingFace Team. All rights reserved.
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
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel
from optimum.utils import NormalizedConfigManager

from ..utils.constant import _TASK_ALIASES
from ..utils.import_utils import is_torch_version, is_transformers_version
from ..utils.modeling_utils import _prepare_attn_mask, _prepare_decoder_attention_mask


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)


def prepare_jit_inputs(model: PreTrainedModel, task: str, use_cache: bool = False):
    task = _TASK_ALIASES.get(task, task)
    signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.__call__)
    onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
    onnx_config = onnx_config_class(model.config)
    if task == "text-generation" and use_cache:
        onnx_config = onnx_config_class(model.config, use_past=True, use_past_in_inputs=True)
    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
    model_inputs = {key: dummy_inputs[key] for key in signature.parameters if dummy_inputs.get(key, None) is not None}
    if task == "text-generation" and use_cache and model.config.model_type != "gpt_bigcode":
        # WA jit.trace issue of model like llama in https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L464, or else, generation output will be incorrect
        pkv = []
        for i in range(len(model_inputs["past_key_values"])):
            pkv.append([])
            for j in range(len(model_inputs["past_key_values"][0])):
                pkv[i].append(model_inputs["past_key_values"][i][j].to(model.dtype))
            pkv[i] = tuple(pkv[i])
        model_inputs["past_key_values"] = tuple(pkv)
        i = model_inputs["input_ids"]
        a = model_inputs["attention_mask"]
        model_inputs["input_ids"] = torch.cat([torch.zeros(i.shape[0], 1), i], -1).to(i.dtype)
        model_inputs["attention_mask"] = torch.cat([torch.zeros(a.shape[0], 1), a], -1).to(a.dtype)
    return model_inputs


def jit_trace(model: PreTrainedModel, task: str, use_cache: bool = False):
    model_inputs = prepare_jit_inputs(model, task, use_cache)
    # check if the model_inputs is correct.
    model(**model_inputs)
    torch._C._jit_set_texpr_fuser_enabled(False)
    if "past_key_values" in model_inputs.keys():
        model.config.return_dict = False
        if is_torch_version(">", "2.0.1"):
            traced_model = torch.jit.trace(model, example_kwarg_inputs=model_inputs, strict=False)
        else:
            traced_model = torch.jit.trace(model, example_inputs=tuple(model_inputs.values()), strict=False)
    else:
        if is_torch_version(">=", "2.0.0"):
            traced_model = torch.jit.trace(model, example_kwarg_inputs=model_inputs, strict=False)
        else:
            traced_model = torch.jit.trace(model, example_inputs=tuple(model_inputs.values()), strict=False)
    traced_model = torch.jit.freeze(traced_model.eval())
    traced_model(**model_inputs)
    traced_model(**model_inputs)

    return traced_model


class PreTrainedModel(OptimizedModel):
    pass


class BaseModelForCausalLM(PreTrainedModel, GenerationMixin):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    main_input_name = "input_ids"
    base_model_prefix = "torch_script_model"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super(BaseModelForCausalLM, self).__init__(model=model, config=config)
        self.model_save_dir = model_save_dir
        self.preprocessors = kwargs.get("preprocessors", [])
        self.use_cache = use_cache
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.model_dtype = kwargs.get("model_dtype", None)

        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config)

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    def can_generate(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        model = torch.jit.load(file_name)
        torch.jit.freeze(model.eval())
        return model

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        past_key_values = past_key_values or kwargs.get("past", None)

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # `past_key_values` may be in the stardard format (e.g. in contrastive search), converts to bloom's format if needed
        if past_key_values is not None and self.config.model_type == "bloom":
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": self.use_cache,
            "position_ids": None,
            "attention_mask": kwargs.get("attention_mask", None),
            "token_type_ids": None,
        }

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.config.model_type == "bloom":
            return self._reorder_cache_bloom(past_key_values, beam_idx)

        # from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    # Copied from transformers.models.bloom.modeling_bloom.BloomForCausalLM._reorder_cache
    def _reorder_cache_bloom(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called for bloom architecture.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        standardized_past = self._convert_to_standard_cache(past_key_values, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device)
            for layer_past in past_key_values
            for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)

    # Copied from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_bloom_cache
    @staticmethod
    def _convert_to_bloom_cache(past_key_value: Tuple[Tuple[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    # Adapted from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_standard_cache
    def _convert_to_standard_cache(
        self, past_key_value: Tuple[Tuple[torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size, num_heads, ...]))
        """
        if self.config.model_type != "bloom":
            return past_key_value

        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.use_cache:
            if past_key_values is None:
                nb_pkv = 2
                num_layers = self.normalized_config.num_layers
                num_attention_heads = self.normalized_config.num_attention_heads
                num_key_value_heads = num_attention_heads
                if hasattr(self.normalized_config, "num_key_value_heads"):
                    num_key_value_heads = self.normalized_config.num_key_value_heads
                hidden_size = self.normalized_config.hidden_size
                d_k = hidden_size // num_attention_heads
                if self.config.model_type == "gpt_bigcode":
                    new_shape = [input_ids.shape[0], 0, d_k * 2]
                    empty_tensor = torch.empty(size=new_shape)
                    if self.model_dtype is not None:
                        empty_tensor = empty_tensor.to(self.model_dtype)
                    past_key_values = tuple([empty_tensor] * num_layers)
                elif self.config.model_type != "bloom":
                    new_shape = [input_ids.shape[0], num_key_value_heads, 0, d_k]
                    empty_tensor = torch.empty(size=new_shape)
                    if self.model_dtype is not None:
                        empty_tensor = empty_tensor.to(self.model_dtype)
                    pkv = tuple(empty_tensor for _ in range(nb_pkv))
                else:
                    pkv = ()
                    for nb_pkv in range(nb_pkv):
                        if nb_pkv % 2 == 0:
                            new_shape = [input_ids.shape[0] * num_key_value_heads, d_k, 0]
                        else:
                            new_shape = [input_ids.shape[0] * num_key_value_heads, 0, d_k]
                        empty_tensor = torch.empty(size=new_shape)
                        if self.model_dtype is not None:
                            empty_tensor = empty_tensor.to(self.model_dtype)
                        pkv = pkv + (empty_tensor,)
                if past_key_values is None:
                    past_key_values = tuple(tuple(pkv) for _ in range(num_layers))

            inputs["past_key_values"] = past_key_values
        outputs = self.model(**inputs)

        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            past_key_values = outputs[1] if self.use_cache else None
        else:
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"] if self.use_cache else None
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class TSModelForCausalLM(BaseModelForCausalLM):
    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super(TSModelForCausalLM, self).__init__(
            model=model, config=config, model_save_dir=model_save_dir, use_cache=use_cache, **kwargs
        )
        self.model.to(self._device)

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
        use_cache: bool = True,
        **kwargs,
    ):
        if not getattr(config, "torchscript", False):
            raise ValueError("`torchscript` should be set to True to load TorchScript model")

        # Load the model from local directory
        if os.path.isdir(model_id):
            file_name = os.path.join(model_id, file_name)
            model = cls.load_model(file_name)
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
            )
            model_save_dir = Path(model_cache_path).parent
            model = cls.load_model(model_cache_path)

        return cls(
            model,
            config=config,
            model_save_dir=model_save_dir,
            use_cache=use_cache,
            **kwargs,
        )

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
        **kwargs,
    ):
        if is_torch_version("<", "2.0.0"):
            raise ImportError("`torch>=2.0.0` is needed to trace your model")

        task = cls.export_feature
        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
            "use_cache": use_cache,
            "torch_dtype": torch_dtype,
        }

        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)

        if model.config.model_type == "bloom":
            model.transformer._prepare_attn_mask = _prepare_attn_mask

        if model.config.model_type == "llama":
            model.model._prepare_decoder_attention_mask = _prepare_decoder_attention_mask

        traced_model = jit_trace(model, task, use_cache)
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
        config.torchscript = True

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs,
        )
