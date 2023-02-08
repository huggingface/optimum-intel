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
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

import openvino
from openvino.runtime import Core, Tensor

from ..utils.import_utils import is_transformers_version
from .modeling_base_seq2seq import OVBaseModelForSeq2SeqLM


if is_transformers_version("<", "4.25.0"):
    from transformers.generation_utils import GenerationMixin
else:
    from transformers.generation import GenerationMixin

core = Core()

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "AutoTokenizer"

INPUTS_DOCSTRING = r"""
    Arguments:
        encoder (`openvino.runtime.Model`):
            The OpenVINO Runtime model associated to the encoder.
        decoder (`openvino.runtime.Model`):
            The OpenVINO Runtime model associated to the decoder.
        decoder_with_past (`openvino.runtime.Model`):
            The OpenVINO Runtime model associated  to the decoder with past key values.
        config (`transformers.PretrainedConfig`):
            [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is an instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
"""

ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
"""


DECODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SEQ2SEQ_MODEL_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""


TRANSLATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel.openvino import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> inputs = tokenizer(text, return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel.openvino import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> outputs = pipe(text)
    ```
"""


def _contiguous_helper(tensor: np.ndarray) -> np.ndarray:
    return tensor if tensor.flags["C_CONTIGUOUS"] else np.ascontiguousarray(tensor)


@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for OpenVINO inference.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForSeq2SeqLM(OVBaseModelForSeq2SeqLM, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM

    def __init__(
        self,
        encoder: openvino.runtime.Model,
        decoder: openvino.runtime.Model,
        decoder_with_past: openvino.runtime.Model = None,
        config: transformers.PretrainedConfig = None,
        **kwargs
    ):
        super().__init__(
            encoder=encoder, decoder=decoder, decoder_with_past=decoder_with_past, config=config, **kwargs
        )
        self.device = torch.device("cpu")
        self.main_input_name = "input_ids"
        self.decoder_with_past = None
        enable_compilation = kwargs.get("compile", True)
        encoder_cache_dir = Path(self.model_save_dir).joinpath("encoder_cache")
        encoder_cache_dir.mkdir(parents=True, exist_ok=True)
        ov_encoder_config = {**self.ov_config, "CACHE_DIR": str(encoder_cache_dir)}
        self.encoder = OVEncoder(self.encoder_model, self._device, ov_encoder_config)
        decoder_cache_dir = Path(self.model_save_dir).joinpath("decoder_cache")
        decoder_cache_dir.mkdir(parents=True, exist_ok=True)
        ov_decoder_config = {**self.ov_config, "CACHE_DIR": str(decoder_cache_dir)}
        self.decoder = OVDecoder(self.decoder_model, self._device, ov_decoder_config)
        if self.use_cache:
            decoder_past_cache_dir = Path(self.model_save_dir).joinpath("decoder_past_cache")
            decoder_past_cache_dir.mkdir(parents=True, exist_ok=True)
            ov_decoder_past_config = {**self.ov_config, "CACHE_DIR": str(decoder_past_cache_dir)}
            self.decoder_with_past = OVDecoder(self.decoder_with_past_model, self._device, ov_decoder_past_config)
        if enable_compilation:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    def to(self, device: str):
        self._device = device.upper()
        self.encoder._device = self._device
        self.decoder._device = self._device
        if self.use_cache:
            self.decoder_with_past._device = self._device
        self.clear_requests()
        return self

    @add_start_docstrings_to_model_forward(
        SEQ2SEQ_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TRANSLATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForSeq2SeqLM",
            checkpoint="t5-small",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Decode
        if past_key_values is None or self.decoder_with_past is None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids[:, -1:],  # Cut decoder_input_ids if past is used
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )

        return Seq2SeqLMOutput(logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ) -> Dict:
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values or kwargs.get("past", None),
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self):
        return self.encoder

    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        reordered_past = ()
        for layer_past in past:
            # Cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def reshape(self, batch_size: int, sequence_length: int):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            sequence_length (`int`):
                The sequence length.
        """
        super().reshape(batch_size, sequence_length)
        self.clear_requests()
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        super().half()
        self.clear_requests()
        return self

    def clear_requests(self):
        self.encoder.request = None
        self.decoder.request = None
        if self.use_cache:
            self.decoder_with_past.request = None

    def compile(self):
        self.encoder._create_inference_request()
        self.decoder._create_inference_request()
        if self.use_cache:
            self.decoder_with_past._create_inference_request()


class OVEncoder:
    """
    Encoder model for OpenVINO inference.

    Arguments:
        request (`openvino.runtime.ie_api.InferRequest`):
            The OpenVINO inference request associated to the encoder.
    """

    def __init__(self, model: openvino.runtime.Model, device: str, ov_config: Dict):
        self.model = model
        self._device = device
        self.device = torch.device("cpu")
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.main_input_name = "input_ids"
        self.ov_config = ov_config
        self.request = None

    @add_start_docstrings_to_model_forward(ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ) -> BaseModelOutput:
        self._create_inference_request()

        # Check if inputs are c-like, if not - convert them.
        input_ids = _contiguous_helper(input_ids.numpy())

        inputs = {
            "input_ids": Tensor(input_ids, shared_memory=True),
        }

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names:
            attention_mask = _contiguous_helper(attention_mask.numpy())
            inputs["attention_mask"] = Tensor(attention_mask, shared_memory=True)

        # Run inference
        self.request.start_async(inputs)
        self.request.wait()

        last_hidden_state = torch.from_numpy(self.request.get_tensor("last_hidden_state").data).to(self.device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _create_inference_request(self):
        if self.request is None:
            logger.info("Compiling the encoder and creating the inference request ...")
            compiled_model = core.compile_model(self.model, self._device, self.ov_config)
            self.request = compiled_model.create_infer_request()


class OVDecoder:
    """
    Decoder model for OpenVINO inference.

    Arguments:
        request (`openvino.runtime.ie_api.InferRequest`):
            The OpenVINO inference request associated to the decoder.
        device (`torch.device`):
            The device type used by this process.
    """

    def __init__(self, model: openvino.runtime.Model, device: str, ov_config: Dict):
        self.model = model
        self._device = device
        self.device = torch.device("cpu")
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        self.ov_config = ov_config
        self.request = None

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Seq2SeqLMOutput:
        self._create_inference_request()

        inputs = {}

        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = [
                _contiguous_helper(past_key_value.numpy())
                for pkv_per_layer in past_key_values
                for past_key_value in pkv_per_layer
            ]
            # Add the past_key_values to the decoder inputs
            inputs = {
                input_name: Tensor(past_key_value, shared_memory=True)
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values)
            }

        # Check if inputs are c-like, if not - convert them.
        input_ids = _contiguous_helper(input_ids.numpy())
        inputs["input_ids"] = Tensor(input_ids, shared_memory=True)

        # Add the encoder_attention_mask inputs when needed
        if "encoder_attention_mask" in self.input_names and encoder_attention_mask is not None:
            encoder_attention_mask = _contiguous_helper(encoder_attention_mask.numpy())
            inputs["encoder_attention_mask"] = Tensor(encoder_attention_mask, shared_memory=True)

        # Add the encoder_hidden_states inputs when needed
        if "encoder_hidden_states" in self.input_names and encoder_hidden_states is not None:
            encoder_hidden_states = _contiguous_helper(encoder_hidden_states.numpy())
            inputs["encoder_hidden_states"] = Tensor(encoder_hidden_states, shared_memory=True)

        # Run inference
        self.request.start_async(inputs)
        self.request.wait()

        outputs = {
            key.get_any_name(): value.data for key, value in zip(self.request.model_outputs, self.request.outputs)
        }

        # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
        # self-attention layer and 2 to the cross-attention layer)
        past_key_values = tuple(
            torch.from_numpy(outputs[key]).to(self.device)
            for key in outputs
            if ("key_values" in key or "present" in key)
        )

        # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
        # cross-attention per decoder layer
        num_pkv = 4
        past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))

        logits = torch.from_numpy(outputs["logits"]).to(self.device)

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _create_inference_request(self):
        if self.request is None:
            logger.info("Compiling the decoder and creating the inference request ...")
            compiled_model = core.compile_model(self.model, self._device, self.ov_config)
            self.request = compiled_model.create_infer_request()
