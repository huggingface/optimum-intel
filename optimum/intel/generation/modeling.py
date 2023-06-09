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
from typing import Any, Dict, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from packaging.version import parse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel
from optimum.utils import NormalizedConfigManager

from ..utils.constant import _TASK_ALIASES
from ..utils.import_utils import is_torch_version, is_transformers_version


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)


def prepare_jit_inputs(
    model: PreTrainedModel,
    task: str,
    use_cache: bool = False,
    sequence_length=32,
    encoder_sequence_length=32,
    decoder_sequence_length=2,
):
    task = _TASK_ALIASES.get(task, task)
    signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.__call__)
    onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
    onnx_config = onnx_config_class(model.config)
    if use_cache and "generation" in task:
        onnx_config = onnx_config_class(model.config, use_past=True)
    # Models with encoder and decoder need to trace seperately
    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt", sequence_length=sequence_length)
    dummy_inputs_2 = {}
    # The length of input_ids in dummy_inputs is always 1 if use_past, see https://github.com/huggingface/optimum/blob/main/optimum/exporters/onnx/base.py#L558-L571
    if task == "text-generation" and use_cache:
        # WA jit.trace issue of model like llama in https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L463-L464,
        # or else, generation output will be incorrect
        pkv = []
        for i in range(len(dummy_inputs["past_key_values"])):
            pkv.append([])
            for j in range(len(dummy_inputs["past_key_values"][0])):
                pkv[i].append(dummy_inputs["past_key_values"][i][j].to(model.dtype))
            pkv[i] = tuple(pkv[i])
        dummy_inputs["past_key_values"] = tuple(pkv)
        dummy_inputs["input_ids"] = dummy_inputs["input_ids"].repeat_interleave(decoder_sequence_length, dim=-1)
        dummy_inputs["attention_mask"] = torch.cat(
            [torch.zeros(dummy_inputs["attention_mask"].shape[0], 1), dummy_inputs["attention_mask"]], -1
        ).to(dummy_inputs["attention_mask"].dtype)
    elif task == "text2text-generation":
        # dummy_inputs is for decoder, dummy_inputs_2 is for encoder
        dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt", sequence_length=1)
        encoder_inputs = dummy_inputs.pop("input_ids", None).repeat_interleave(encoder_sequence_length, dim=-1)
        dummy_inputs["attention_mask"] = torch.ones_like(encoder_inputs, dtype=torch.int64)
        encoder_outputs = torch.ones(
            [encoder_inputs.shape[0], encoder_inputs.shape[1], model.config.d_model], dtype=model.dtype
        )
        dummy_inputs["encoder_outputs"] = (encoder_outputs,)
        if use_cache:
            pkv = []
            for i in range(len(dummy_inputs["past_key_values"])):
                pkv.append([])
                for j in range(len(dummy_inputs["past_key_values"][0])):
                    pkv[i].append(dummy_inputs["past_key_values"][i][j].to(model.dtype))
                pkv[i] = tuple(pkv[i])
            dummy_inputs["past_key_values"] = tuple(pkv)

        dummy_inputs_2["input_ids"] = encoder_inputs

    dummy_inputs = {key: dummy_inputs[key] for key in signature.parameters if dummy_inputs.get(key, None) is not None}
    if dummy_inputs_2:
        dummy_inputs_2 = {
            key: dummy_inputs_2[key] for key in signature.parameters if dummy_inputs_2.get(key, None) is not None
        }

    return dummy_inputs, dummy_inputs_2


def trace_model(model: PreTrainedModel, model_inputs: dict):
    if ("past_key_values" in model_inputs.keys() and is_torch_version(">", "2.0.1")) or (
        "past_key_values" not in model_inputs.keys() and is_torch_version(">=", "2.0.0")
    ):
        traced_model = torch.jit.trace(model, example_kwarg_inputs=model_inputs, strict=False)
    else:
        traced_model = torch.jit.trace(model, example_inputs=tuple(model_inputs.values()), strict=False)
        logger.warning("Tuple jit inputs may cause unexpected error, please update your pytorch version.")
    traced_model = torch.jit.freeze(traced_model.eval())
    traced_model(**model_inputs)
    traced_model(**model_inputs)
    return traced_model


def jit_trace(model: PreTrainedModel, task: str, use_cache: bool = False):
    model_inputs, model_inputs_2 = prepare_jit_inputs(model, task, use_cache)
    if task == "text2text-generation" and parse(parse(torch.__version__).base_version) < parse("2.1.0"):
        logger.warning("Current torch version cause unexpected error, return the rager mode model instead.")
        return model, model.get_encoder()

    torch._C._jit_set_texpr_fuser_enabled(False)

    if "past_key_values" in model_inputs.keys():
        model.config.return_dict = False

    traced_model = trace_model(model, model_inputs)
    traced_model_2 = None
    if model_inputs_2:
        traced_model_2 = trace_model(model.get_encoder(), model_inputs_2)

    if traced_model_2 is not None:
        return traced_model, traced_model_2
    else:
        return traced_model


class TSModelForCausalLM(OptimizedModel, GenerationMixin):
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
        self.model = model
        self.config = config
        self.model_save_dir = model_save_dir
        self.preprocessors = kwargs.get("preprocessors", [])
        self.use_cache = use_cache
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.model_dtype = kwargs.get("model_dtype", None)
        self.generation_config = GenerationConfig.from_model_config(config)

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        model = torch.jit.load(file_name)
        torch.jit.freeze(model.eval())
        return model

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.use_cache:
            if past_key_values is None:
                nb_pkv = 2
                num_layers = self.normalized_config.num_layers
                num_attention_heads = self.normalized_config.num_attention_heads
                hidden_size = self.normalized_config.hidden_size
                d_k = hidden_size // num_attention_heads

                if self.config.model_type != "bloom":
                    new_shape = [input_ids.shape[0], num_attention_heads, 0, d_k]
                    empty_tensor = torch.empty(size=new_shape)
                    if self.model_dtype is not None:
                        empty_tensor = empty_tensor.to(self.model_dtype)
                    past_key_values = tuple(tuple(empty_tensor for _ in range(nb_pkv)) for _ in range(num_layers))
                    pkv = tuple(empty_tensor for _ in range(nb_pkv))
                else:
                    pkv = ()
                    for nb_pkv in range(nb_pkv):
                        if nb_pkv % 2 == 0:
                            new_shape = [input_ids.shape[0] * num_attention_heads, d_k, 0]
                        else:
                            new_shape = [input_ids.shape[0] * num_attention_heads, 0, d_k]
                        empty_tensor = torch.empty(size=new_shape)
                        if self.model_dtype is not None:
                            empty_tensor = empty_tensor.to(self.model_dtype)
                        pkv = pkv + (empty_tensor,)
                past_key_values = tuple(tuple(pkv) for _ in range(num_layers))

            inputs["past_key_values"] = past_key_values
        outputs = self.model(**inputs)

        if isinstance(outputs, tuple):
            outputs = CausalLMOutputWithPast(logits=outputs[0], past_key_values=outputs[1] if self.use_cache else None)
        else:
            outputs = CausalLMOutputWithPast(
                logits=outputs["logits"], past_key_values=outputs["past_key_values"] if self.use_cache else None
            )

        return outputs

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

        # IPEX jit model need 2 iterations to convert model to int8 model
        onnx_config_class = TasksManager.get_exporter_config_constructor(
            model_type=config.model_type.replace("_", "-"),
            exporter="onnx",
            task=cls.export_feature,
        )
        onnx_config = onnx_config_class(config, use_past=use_cache)
        model_inputs = onnx_config.generate_dummy_inputs(framework="pt")
        for i in range(2):
            model(**model_inputs)

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

    def can_generate(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

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


class TSModelForSeq2SeqLM(OptimizedModel, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    export_feature = "text2text-generation"
    main_input_name = "input_ids"
    base_model_prefix = "torch_script_model"

    def __init__(
        self,
        decoder,
        encoder,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        model_save_dir_2: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.decoder = decoder
        self.encoder = encoder
        self.config = config
        self.model_save_dir = model_save_dir
        self.model_save_dir_2 = model_save_dir_2
        self.preprocessors = kwargs.get("preprocessors", [])
        self.use_cache = use_cache
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.decoder.to(self._device)
        self.encoder.to(self._device)
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.model_dtype = kwargs.get("model_dtype", None)
        self.decoder_start_token_id = None
        if kwargs.get("decoder_start_token_id", None) is not None:
            self.decoder_start_token_id = kwargs.get("decoder_start_token_id", None)
        elif config.decoder_start_token_id is not None:
            self.decoder_start_token_id = config.decoder_start_token_id
        elif config.bos_token_id is not None:
            self.decoder_start_token_id = config.bos_token_id

        if self.decoder_start_token_id is None:
            logger.warning("Please assign the decoder_start_token_id, it may cause unexpected errors.")

        self.generation_config = GenerationConfig.from_model_config(config)

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    @staticmethod
    def save_model(model: Union[PreTrainedModel, torch.jit.RecursiveScriptModule], file_name: Union[str, Path]):
        if isinstance(model, torch.jit.RecursiveScriptModule):
            torch.jit.save(model, file_name)
        else:
            torch.save(model, file_name)
        return model

    @staticmethod
    def load_model(file_name: Union[str, Path], is_jit: bool):
        if is_jit:
            model = torch.jit.load(file_name)
            torch.jit.freeze(model.eval())
        else:
            model = torch.load(file_name)
        return model

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        save_directory_2: Union[str, Path],
        file_name: Optional[str] = None,
        is_jit: bool = None,
        **kwargs,
    ):
        if is_jit:
            torch.jit.save(self.decoder, os.path.join(save_directory, WEIGHTS_NAME))
            torch.jit.save(self.encoder, os.path.join(save_directory_2, WEIGHTS_NAME))
        else:
            torch.save(self.decoder, os.path.join(save_directory, WEIGHTS_NAME))
            torch.save(self.encoder, os.path.join(save_directory_2, WEIGHTS_NAME))

    @classmethod
    def _load_model(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = WEIGHTS_NAME,
        local_files_only: bool = False,
        is_jit: bool = False,
        **kwargs,
    ):
        if os.path.isdir(model_id):
            model = cls.load_model(os.path.join(model_id, file_name), is_jit)
            model_save_dir = model_id
        else:
            # Download the model from the hub
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
            model = cls.load_model(model_cache_path, is_jit)

        return model, model_save_dir

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        model_id_2: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = WEIGHTS_NAME,
        local_files_only: bool = False,
        use_cache: bool = True,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = torch.float32,
        is_jit: bool = None,
        **kwargs,
    ):
        if not getattr(config, "torchscript", False):
            raise ValueError("`torchscript` should be set to True to load TorchScript model")

        onnx_config_class = TasksManager.get_exporter_config_constructor(
            model_type=config.model_type.replace("_", "-"),
            exporter="onnx",
            task=cls.export_feature,
        )
        onnx_config = onnx_config_class(config, use_past=use_cache)
        dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt", sequence_length=1)
        dummy_inputs_2 = {}
        encoder_inputs = dummy_inputs.pop("input_ids", None).repeat_interleave(5, dim=-1)
        dummy_inputs["attention_mask"] = torch.ones_like(encoder_inputs, dtype=torch.int64)
        encoder_outputs = torch.ones(
            [encoder_inputs.shape[0], encoder_inputs.shape[1], config.d_model], dtype=torch_dtype
        )
        dummy_inputs["encoder_outputs"] = (encoder_outputs,)
        if use_cache:
            pkv = []
            for i in range(len(dummy_inputs["past_key_values"])):
                pkv.append([])
                for j in range(len(dummy_inputs["past_key_values"][0])):
                    pkv[i].append(dummy_inputs["past_key_values"][i][j].to(torch_dtype))
                pkv[i] = tuple(pkv[i])
            dummy_inputs["past_key_values"] = tuple(pkv)
        dummy_inputs_2["input_ids"] = encoder_inputs

        # Load the model from local directory
        decoder, model_save_dir_1 = cls._load_model(
            model_id=model_id,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            local_files_only=local_files_only,
            is_jit=is_jit,
        )
        if is_jit:
            decoder(**dummy_inputs)
            decoder(**dummy_inputs)

        encoder, model_save_dir_2 = cls._load_model(
            model_id=model_id_2,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            local_files_only=local_files_only,
            is_jit=is_jit,
        )
        if is_jit:
            encoder(**dummy_inputs_2)
            encoder(**dummy_inputs_2)

        return cls(
            decoder=decoder,
            encoder=encoder,
            config=config,
            use_cache=use_cache,
            model_dtype=torch_dtype,
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
        traced_model, traced_model_2 = jit_trace(model, task, use_cache)
        save_dir = TemporaryDirectory()
        save_dir_2 = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        save_dir_path_2 = Path(save_dir_2.name)
        cls.save_model(traced_model, save_dir_path / WEIGHTS_NAME)
        cls.save_model(traced_model_2, save_dir_path_2 / WEIGHTS_NAME)
        config.torchscript = True

        is_jit = None
        if isinstance(traced_model, torch.jit.RecursiveScriptModule):
            is_jit = True

        return cls._from_pretrained(
            model_id=save_dir_path,
            model_id_2=save_dir_path_2,
            config=config,
            use_cache=use_cache,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            is_jit=is_jit,
            **kwargs,
        )

    def can_generate(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.decoder.to(self._device)
        self.encoder.to(self._device)
        return self

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        decoder_input_ids: torch.LongTensor = None,
        encoder_outputs: Tuple[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if input_ids is not None:
            encoder_outputs = self.encoder(input_ids=input_ids)

        if not isinstance(encoder_outputs, tuple):
            encoder_outputs = (encoder_outputs["last_hidden_state"],)

        if decoder_input_ids is None:
            decoder_input_ids = (
                torch.ones([encoder_outputs[0].shape[0], 1], dtype=torch.int64) * self.decoder_start_token_id
            )

        if attention_mask is None:
            attention_mask = torch.ones([encoder_outputs[0].shape[0], encoder_outputs[0].shape[-2]], dtype=torch.int64)

        inputs = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

        if self.use_cache:
            if past_key_values is None:
                nb_pkv = 4
                num_layers = self.normalized_config.num_layers
                num_attention_heads = self.normalized_config.num_attention_heads
                hidden_size = self.normalized_config.hidden_size
                d_k = hidden_size // num_attention_heads

                new_shape = [decoder_input_ids.shape[0], num_attention_heads, 0, d_k]
                empty_tensor = torch.empty(size=new_shape)
                if self.model_dtype is not None:
                    empty_tensor = empty_tensor.to(self.model_dtype)
                past_key_values = tuple(tuple(empty_tensor for _ in range(nb_pkv)) for _ in range(num_layers))
                pkv = tuple(empty_tensor for _ in range(nb_pkv))

                past_key_values = tuple(tuple(pkv) for _ in range(num_layers))

            inputs["past_key_values"] = past_key_values

        decoder_outputs = self.decoder(**inputs)

        if isinstance(decoder_outputs, tuple):
            logits = decoder_outputs[0]
            encoder_last_hidden_state = decoder_outputs[2] if self.use_cache else decoder_outputs[1]
            past_key_values = decoder_outputs[1] if self.use_cache else None
        else:
            logits = decoder_outputs["logits"]
            encoder_last_hidden_state = decoder_outputs["encoder_last_hidden_state"]
            past_key_values = decoder_outputs["past_key_values"] if self.use_cache else None

        outputs = Seq2SeqLMOutput(
            logits=logits, encoder_last_hidden_state=encoder_last_hidden_state, past_key_values=past_key_values
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs[0].size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        model_kwargs["encoder_outputs"] = self.get_encoder()(input_ids=inputs_tensor)

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            if isinstance(model_kwargs["encoder_outputs"], tuple):
                encoder_outputs_list = []
                for i in range(len(model_kwargs["encoder_outputs"])):
                    encoder_outputs_list.append(
                        model_kwargs["encoder_outputs"][i].repeat_interleave(expand_size, dim=0)
                    )
                model_kwargs["encoder_outputs"] = tuple(encoder_outputs_list)
            else:
                model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs
