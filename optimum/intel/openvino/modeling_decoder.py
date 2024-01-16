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
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Union

import numpy as np
import openvino
import torch
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Tensor, Type
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast

from optimum.utils import NormalizedConfigManager

from ...exporters.openvino import ensure_stateful_is_available, main_export, patch_stateful
from ...exporters.openvino.stateful import model_has_state
from ..utils.import_utils import is_transformers_version
from ..utils.modeling_utils import MULTI_QUERY_ATTN_MODELS
from .modeling import _TOKENIZER_FOR_DOC, INPUTS_DOCSTRING, MODEL_START_DOCSTRING, OVModel
from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME, STR_TO_OV_TYPE


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin


logger = logging.getLogger(__name__)

core = Core()


TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> inputs = tokenizer("I love this story because", return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)
    ```
    Example using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    >>> text = "I love this story because"
    >>> gen = gen_pipeline(text)
    ```
"""

_SUPPORTED_ARCHITECTURES = {
    "bart",
    "blenderbot",
    "blenderbot-small",
    "bloom",
    "codegen",
    "gpt2",
    "gpt-bigcode",
    "gpt-neo",
    "gpt-neox",
    "llama",
    "marian",
    "opt",
    "pegasus",
}


@add_start_docstrings(
    """
    Base OVBaseDecoderModel class.
    """,
)
class OVBaseDecoderModel(OVModel):
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
        if not dynamic_shapes:
            raise ValueError(
                "`dynamic_shapes` was set to `False` but static shapes are not supported for causal language model. Please set `dynamic_shapes=True`."
            )

        enable_compilation = kwargs.get("compile", True)
        kwargs["compile"] = False  # avoid extra compilation in the base class

        super().__init__(
            model,
            config,
            device=device,
            dynamic_shapes=False,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            **kwargs,
        )

        self.is_dynamic = dynamic_shapes
        use_cache = kwargs.pop("use_cache", True)
        model_has_sinks = model_has_state(self.model)
        self.use_cache = any("past_key_values" in key.get_any_name() for key in model.inputs) or model_has_sinks
        stateful = kwargs.pop("stateful", None)  # stateful model only if it is converted with stateful=True
        self.stateful = model_has_sinks
        self.main_input_name = "input_ids"
        self.num_pkv = 2
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        self.key_value_output_names = [key for key in self.output_names if "present" in key]
        self._original_model = self.model.clone()  # keep original model for serialization
        self._pkv_precision = Type.f32
        self.next_beam_idx = None
        self.update_pkv_precision()
        if self.is_dynamic:
            self.model = self._reshape(self.model, -1, -1)
        is_stateful_supported = ensure_stateful_is_available(warn=False)

        if self.use_cache and not self.stateful:
            logger.warn(
                "Provided model does not contain state. It may lead to sub-optimal performance."
                "Please reexport model with updated OpenVINO version >= 2023.3.0 calling the `from_pretrained` method with original model "
                "and `export=True` parameter"
            )

        if self.stateful:
            if stateful is None:
                stateful = is_stateful_supported
            if model_has_sinks and not is_stateful_supported:
                raise ValueError(
                    "Loaded stateful model, while OpenVINO runtime version does not support stateful model inference. "
                    "Please update OpenVINO version >= 2023.3.0 "
                    "or export the original model once again with `stateful=False` when calling the `from_pretrained` method."
                    "To export your model, simply set `export=True`."
                )

        def raise_error(model_prop, user_prop, name):
            raise ValueError(
                f"`{name}` was set to `{user_prop}` but the loaded model only supports `{name}={model_prop}`. "
                f"Please load your current model with `{name}={model_prop}` or export the original model "
                f"once again with `{name}={user_prop}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )

        if stateful is not None and stateful ^ self.stateful:
            # We cannot transform stateful model to stateless
            raise_error(self.stateful, stateful, "stateful")

        if use_cache ^ self.use_cache:
            raise_error(self.use_cache, use_cache, "use_cache")

        if enable_compilation:
            self.compile()

    def update_pkv_precision(self, force_fp32=False):
        if not self.use_cache or self.stateful:
            return

        pkv_precision = Type.f32
        if not force_fp32:
            device = self._device.upper()
            try:
                if "INFERENCE_PRECISION_HINT" in core.get_property(device, "SUPPORTED_PROPERTIES"):
                    pkv_precision = core.get_property(device, "INFERENCE_PRECISION_HINT")
            except RuntimeError:  # use default precision when get_property fails, e.g. when device is "AUTO:GPU"
                pass

            # ov_config["INFERENCE_PRECISION_HINT"] may override the prefer precision
            if self.ov_config:
                inference_precision_hint = self.ov_config.get("INFERENCE_PRECISION_HINT", "")
                if inference_precision_hint in STR_TO_OV_TYPE:
                    pkv_precision = STR_TO_OV_TYPE[inference_precision_hint]

            ppp = PrePostProcessor(self.model)
            for key in self.model.inputs:
                if "past_key_values" in key.get_any_name() and pkv_precision != key.get_element_type():
                    ppp.input(key.get_any_name()).tensor().set_element_type(pkv_precision)
            for key in self.model.outputs:
                if "present" in key.get_any_name() and pkv_precision != key.get_element_type():
                    ppp.output(key.get_any_name()).tensor().set_element_type(pkv_precision)

            self.model = ppp.build()
            self._pkv_precision = pkv_precision
        else:
            if hasattr(self, "_pkv_precision") and self._pkv_precision != Type.f32:
                self._pkv_precision = Type.f32
                self.model = self._original_model.clone()
                if self.is_dynamic:
                    self.model = self._reshape(self.model, -1, -1)
                self.request = None

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        model_to_save = self.model if self._pkv_precision == Type.f32 else self._original_model
        dst_path = os.path.join(save_directory, OV_XML_FILE_NAME)
        openvino.save_model(model_to_save, dst_path, compress_to_fp16=False)

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
        trust_remote_code: bool = False,
        load_in_8bit: Optional[bool] = None,
        **kwargs,
    ):
        if config.model_type.replace("_", "-") not in _SUPPORTED_ARCHITECTURES:
            logger.warning(
                f"This architecture : {config.model_type} was not validated, only :{', '.join(_SUPPORTED_ARCHITECTURES)} architectures were "
                "validated, use at your own risk."
            )
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        if task is None:
            task = cls.export_feature

            if use_cache:
                task = task + "-with-past"

        compression_option = None
        if load_in_8bit is not None:
            compression_option = "int8" if load_in_8bit else "fp32"
        stateful = kwargs.pop("stateful", ensure_stateful_is_available(warn=False) and use_cache)
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            compression_option=compression_option,
            stateful=stateful,
        )

        config.is_decoder = True
        config.is_encoder_decoder = False
        config.save_pretrained(save_dir_path)
        return cls._from_pretrained(
            model_id=save_dir_path, config=config, use_cache=use_cache, load_in_8bit=False, stateful=None, **kwargs
        )

    def _reshape(
        self,
        model: openvino.runtime.Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        if height is not None:
            logger.warning(f"`height` set to `{height}` will be ignored during reshaping operation.")

        if width is not None:
            logger.warning(f"`width` set to `{width}` will be ignored during reshaping operation.")

        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            input_name = inputs.get_any_name()
            if input_name.startswith("past_key_values"):
                if len(inputs.partial_shape) == 3 and input_name.endswith("value"):
                    shapes[inputs][1] = -1
                else:
                    shapes[inputs][2] = -1
            elif input_name.startswith("beam_idx"):
                shapes[inputs][0] = -1
            else:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int):
        logger.warning("Static shapes are not supported for causal language model.")
        return self

    def compile(self):
        if self.request is None:
            super().compile()
            self.request = self.request.create_infer_request()

    def _make_stateful(self):
        patch_stateful(self.config, self.model)
        self.stateful = True


@add_start_docstrings(
    """
    OpenVINO Model with a causal language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForCausalLM(OVBaseDecoderModel, GenerationMixin):
    export_feature = "text-generation"
    auto_model_class = AutoModelForCausalLM

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForCausalLM",
            checkpoint="gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()
        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        batch_size = input_ids.shape[0]
        if self.config.model_type == "bloom":
            batch_size *= self.normalized_config.num_attention_heads

        inputs = {}
        past_len = 0
        if not self.stateful:
            if past_key_values is not None:
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS:
                    past_len = past_key_values[0][1].shape[-2]
                    if self._pkv_precision == Type.bf16:
                        # numpy does not support bf16, pretending f16, should change to bf16
                        past_key_values = tuple(
                            Tensor(past_key_value, past_key_value.shape, Type.bf16)
                            for pkv_per_layer in past_key_values
                            for past_key_value in pkv_per_layer
                        )
                    else:
                        # Flatten the past_key_values
                        past_key_values = tuple(
                            past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                        )
                else:
                    past_len = past_key_values[0].shape[-2]

                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))

            # Create empty past_key_values for decoder_with_past first generation step
            elif self.use_cache:
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    if self.config.model_type == "chatglm":
                        shape[0] = 0
                        shape[1] = batch_size
                    else:
                        shape[0] = batch_size
                        if shape[2].is_dynamic:
                            shape[2] = 0
                        else:
                            shape[1] = 0
                    inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())
        else:
            # past_key_values are not used explicitly, instead they are handled inside the model
            if past_key_values is None:
                # Need a marker to differentiate the first generate iteration from the others in
                # the first condition at the function beginning above.
                # It should be something that is not None and it should be True when converted to Boolean.
                past_key_values = ((),)
                # This is the first iteration in a sequence, reset all states
                self.request.reset_state()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.arange(batch_size, dtype=int)

        inputs["input_ids"] = np.array(input_ids)
        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones(
                    (input_ids.shape[0], input_ids.shape[1] + past_len), dtype=inputs["input_ids"].dtype
                )

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = np.expand_dims(position_ids[:, -1], axis=-1)

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
            )

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

        if not self.stateful:
            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
                past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS:
                    # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
                    past_key_values = tuple(
                        past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
                    )
            else:
                past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.stateful:
            # TODO: Apply it differently based on model type
            # TODO: At least for bloom we need to replicate values for each attention head
            self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
            return past_key_values
        else:
            return tuple(
                tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past) for layer_past in past_key_values
            )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        model = cls.load_model(model_cache_path, load_in_8bit=load_in_8bit)

        model_type = config.model_type.replace("_", "-")
        if model_type == "bloom":
            init_cls = OVBloomForCausalLM
        elif model_type == "mpt":
            init_cls = OVMPTForCausalLM
        elif model_type == "opt":
            init_cls = OVOPTForCausalLM
        elif model_type == "gpt-bigcode":
            init_cls = OVGPTBigCodeForCausalLM
        else:
            init_cls = cls

        return init_cls(model=model, config=config, model_save_dir=model_cache_path.parent, **kwargs)


class OVBloomForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        # only last token for input_ids if past is not None
        if past_key_values and not self.stateful:
            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }

    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called for bloom architecture.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.stateful:
            beam_idx = np.array(beam_idx)
            batch_size = beam_idx.shape[0]
            indices = np.array(range(batch_size * self.normalized_config.num_attention_heads))
            indices = indices.reshape([batch_size, self.normalized_config.num_attention_heads])
            self.next_beam_idx = np.take(indices, beam_idx, 0).flatten()
            return past_key_values
        else:
            standardized_past = self._convert_to_standard_cache(past_key_values, batch_size=len(beam_idx))
            reordered_past = tuple(
                (
                    np.take(layer_past[0], beam_idx, 0),
                    np.take(layer_past[1], beam_idx, 0),
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
                layer_past[0].reshape((batch_size_times_num_heads, head_dim, seq_length)),
                layer_past[1].reshape((batch_size_times_num_heads, seq_length, head_dim)),
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
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape((batch_size, num_heads, head_dim, seq_length)),
                layer_past[1].reshape((batch_size, num_heads, seq_length, head_dim)),
            )
            for layer_past in past_key_value
        )


class OVOPTForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }


class OVMPTForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
        }


class OVGPTBigCodeForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        if self.stateful:
            self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
            return past_key_values
        else:
            return tuple(np.take(layer_past, beam_idx, 0) for layer_past in past_key_values)
