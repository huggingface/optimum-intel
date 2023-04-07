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

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Union

import numpy as np
import openvino
import torch
from openvino.runtime import Core, Tensor
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast

from optimum.exporters import TasksManager
from optimum.exporters.onnx import export

from ..utils.import_utils import is_transformers_version
from .modeling_base import OVBaseModel
from .utils import ONNX_WEIGHTS_NAME


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)

core = Core()


def _contiguous_helper(tensor: np.ndarray) -> np.ndarray:
    return tensor if tensor.flags["C_CONTIGUOUS"] else np.ascontiguousarray(tensor)


@add_start_docstrings(
    """
    Base OVBaseDecoderModel class.
    """,
)
class OVBaseDecoderModel(OVBaseModel):
    def __init__(
        self,
        model: openvino.runtime.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.config = config
        self.use_cache = any("past_key_values" in key.get_any_name() for key in model.inputs)
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = ov_config if ov_config is not None else {}
        self.preprocessors = kwargs.get("preprocessors", [])
        if self.is_dynamic:
            model = self._reshape(model, -1, -1)
        self.model = model
        self.device = torch.device("cpu")
        self.main_input_name = "input_ids"
        enable_compilation = kwargs.get("compile", True)
        self.decoder = OVDecoder(self.model, self._device, self.ov_config, self.use_cache)

        if enable_compilation:
            self.compile()

        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config)

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    def compile(self):
        self.decoder._create_inference_request()

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
        task: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        model_file_name = ONNX_WEIGHTS_NAME

        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }
        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
        onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        onnx_config = onnx_config_constructor(model.config, use_past=use_cache)

        # Export the model to the ONNX format
        export(model=model, config=onnx_config, output=save_dir_path / model_file_name)

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=True,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=model_file_name,
            local_files_only=local_files_only,
            **kwargs,
        )

    def _reshape(self, model: openvino.runtime.Model, batch_size: int, sequence_length: int, is_decoder=True):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            if inputs.get_any_name().startswith("past_key_values"):
                shapes[inputs][2] = -1
            else:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int):
        logger.warning("Static shapes currently not supported.")
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class OVModelForCausalLM(OVBaseDecoderModel, GenerationMixin):
    export_feature = "causal-lm"
    auto_model_class = AutoModelForCausalLM

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        outputs = self.decoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        return CausalLMOutputWithPast(logits=outputs.logits, past_key_values=outputs.past_key_values)

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values or kwargs.get("past", None),
            "use_cache": self.use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True


class OVDecoder:
    def __init__(self, model: openvino.runtime.Model, device: str, ov_config: Dict, use_cache: bool):
        self.model = model
        self._device = device
        self.device = torch.device("cpu")
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        self.key_value_output_names = [key for key in self.output_names if "present" in key]
        self.use_cache = use_cache
        self.num_pkv = 2
        self.ov_config = ov_config
        self.request = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> CausalLMOutputWithPast:
        self._create_inference_request()

        inputs = {}
        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                _contiguous_helper(np.array(past_key_value))
                for pkv_per_layer in past_key_values
                for past_key_value in pkv_per_layer
            )
            # Add the past_key_values to the decoder inputs
            inputs = {
                input_name: Tensor(past_key_value, shared_memory=True)
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values)
            }

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0]
                shape[2] = shape_input_ids[1]
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs["input_ids"] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names and attention_mask is not None:
            inputs["attention_mask"] = np.array(attention_mask)

        # Run inference
        self.request.start_async(inputs)
        self.request.wait()

        outputs = {
            key.get_any_name(): value.data for key, value in zip(self.request.model_outputs, self.request.outputs)
        }
        logits = torch.from_numpy(outputs["logits"]).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(
                torch.from_numpy(outputs[key]).to(self.device) for key in self.key_value_output_names
            )
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _create_inference_request(self):
        # TODO : remove
        if self.request is None:
            logger.info("Compiling the decoder and creating the inference request ...")
            compiled_model = core.compile_model(self.model, self._device, self.ov_config)
            self.request = compiled_model.create_infer_request()
