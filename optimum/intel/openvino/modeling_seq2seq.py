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
from tempfile import gettempdir
from typing import Dict, Optional, Tuple

import numpy as np
import openvino
import torch
import transformers
from openvino.runtime import Core
from transformers import AutoConfig, AutoModelForSeq2SeqLM, Pix2StructForConditionalGeneration
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

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
    >>> from optimum.intel import {model_class}

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
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> outputs = pipe(text)
    ```
"""

PIX2STRUCT_MODEL_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, seq_length, hidden_size)`):
            Flattened pixel patches. the `hidden_size` is obtained by the following formula: `hidden_size` =
            `num_channels` * `patch_size` * `patch_size`
            The process of flattening the pixel patches is done by `Pix2StructProcessor`.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.
            Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""
_PROCESSOR_FOR_DOC = "AutoProcessor"

PIX2STRUCT_EXAMPLE = r"""
    Example of pix2struct:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}
    >>> from PIL import Image
    >>> import requests

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
    >>> inputs = processor(images=image, text=question, return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = processor.batch_decode(gen_tokens, skip_special_tokens=True)
    ```
"""


@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for OpenVINO inference.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForSeq2SeqLM(OVBaseModelForSeq2SeqLM, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"
    export_feature = "text2text-generation"

    def __init__(
        self,
        encoder: openvino.runtime.Model,
        decoder: openvino.runtime.Model,
        decoder_with_past: openvino.runtime.Model = None,
        config: transformers.PretrainedConfig = None,
        **kwargs,
    ):
        super().__init__(
            encoder=encoder, decoder=decoder, decoder_with_past=decoder_with_past, config=config, **kwargs
        )
        self.device = torch.device("cpu")
        self.decoder_with_past = None
        enable_compilation = kwargs.get("compile", True)
        encoder_cache_dir = Path(self.model_save_dir).joinpath("encoder_cache")
        ov_encoder_config = {**self.ov_config}

        if "CACHE_DIR" not in ov_encoder_config.keys() and not str(self.model_save_dir).startswith(gettempdir()):
            ov_encoder_config["CACHE_DIR"] = str(encoder_cache_dir)

        self.encoder = OVEncoder(
            self.encoder_model, self._device, ov_encoder_config, main_input_name=self.main_input_name
        )

        decoder_cache_dir = Path(self.model_save_dir).joinpath("decoder_cache")
        ov_decoder_config = {**self.ov_config}

        if "CACHE_DIR" not in ov_decoder_config.keys() and not str(self.model_save_dir).startswith(gettempdir()):
            ov_decoder_config["CACHE_DIR"] = str(decoder_cache_dir)

        self.decoder = OVDecoder(self.decoder_model, self._device, ov_decoder_config)

        if self.use_cache:
            decoder_past_cache_dir = Path(self.model_save_dir).joinpath("decoder_past_cache")
            ov_decoder_past_config = {**self.ov_config}

            if "CACHE_DIR" not in ov_decoder_past_config.keys() and not str(self.model_save_dir).startswith(
                gettempdir()
            ):
                ov_decoder_past_config["CACHE_DIR"] = str(decoder_past_cache_dir)

            self.decoder_with_past = OVDecoder(self.decoder_with_past_model, self._device, ov_decoder_past_config)
        if enable_compilation:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        try:
            self.auto_model_class.register(AutoConfig, self.__class__)
        except AttributeError:
            pass

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
            checkpoint="echarlaix/t5-small-openvino",
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
        **kwargs,
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
                tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past[:2]) + layer_past[2:],
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
        self.encoder._compile()
        self.decoder._compile()
        if self.use_cache:
            self.decoder_with_past._compile()


class OVEncoder:
    """
    Encoder model for OpenVINO inference.

    Arguments:
        request (`openvino.runtime.ie_api.InferRequest`):
            The OpenVINO inference request associated to the encoder.
    """

    def __init__(self, model: openvino.runtime.Model, device: str, ov_config: Dict, main_input_name="input_ids"):
        self.model = model
        self._device = device
        self.device = torch.device("cpu")
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.main_input_name = main_input_name
        self.ov_config = ov_config
        self.request = None

    @add_start_docstrings_to_model_forward(ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ) -> BaseModelOutput:
        self._compile()

        # Model inputs
        inputs = {self.main_input_name: input_ids if input_ids is not None else kwargs.get(self.main_input_name)}

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        # Run inference
        last_hidden_state = torch.from_numpy(self.request(inputs, shared_memory=True)["last_hidden_state"]).to(
            self.device
        )

        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _compile(self):
        if self.request is None:
            logger.info(f"Compiling the encoder to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, self.ov_config)


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
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.key_value_output_names = [key for key in self.output_names if "key_values" in key or "present" in key]
        is_legacy = any("past_key_values" in key.get_any_name() for key in self.model.outputs)

        if len(self.key_value_input_names) > 0 and not is_legacy:
            self.use_past = True
            self.num_pkv = 2
        else:
            self.use_past = False
            self.num_pkv = 4

        self.ov_config = ov_config
        self.request = None

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        self._compile()
        # Model inputs
        inputs = {}

        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )

            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        inputs["input_ids"] = input_ids

        # Add the encoder_attention_mask inputs when needed
        if "encoder_attention_mask" in self.input_names and encoder_attention_mask is not None:
            inputs["encoder_attention_mask"] = encoder_attention_mask

        # Add the encoder_hidden_states inputs when needed
        if "encoder_hidden_states" in self.input_names and encoder_hidden_states is not None:
            inputs["encoder_hidden_states"] = encoder_hidden_states

        if "decoder_attention_mask" in self.input_names and decoder_attention_mask is not None:
            inputs["decoder_attention_mask"] = decoder_attention_mask
        # Run inference
        self.request.start_async(inputs, shared_memory=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

        # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
        # self-attention layer and 2 to the cross-attention layer)
        out_past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)

        # Tuple of tuple of length `n_layers`, with each tuple of length equal to:
        # * 4 for the decoder without cache (k/v of self-attention + k/v of cross-attention)
        # * 2 for the decoder with cache (k/v of self-attention as cross-attention cache is constant)
        if self.use_past is False:
            out_past_key_values = tuple(
                out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
            )
        else:
            # grab the cross attention key/values from the inputs
            out_past_key_values = tuple(
                out_past_key_values[i : i + self.num_pkv] + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                for i in range(0, len(out_past_key_values), self.num_pkv)
            )

        return Seq2SeqLMOutput(logits=logits, past_key_values=out_past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _compile(self):
        if self.request is None:
            logger.info(f"Compiling the decoder to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, self.ov_config).create_infer_request()


@add_start_docstrings(
    """
    Pix2Struct model with a language modeling head for OpenVINO inference.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForPix2Struct(OVModelForSeq2SeqLM):
    auto_model_class = Pix2StructForConditionalGeneration
    main_input_name = "flattened_patches"
    export_feature = "image-to-text"

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @add_start_docstrings_to_model_forward(
        PIX2STRUCT_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + PIX2STRUCT_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="OVModelForPix2Struct",
            checkpoint="google/pix2struct-ai2d-base",
        )
    )
    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=flattened_patches,
                attention_mask=attention_mask,
            )

        # Decode
        if past_key_values is None or self.use_cache is False:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids[:, -1:],  # Cut decoder_input_ids if past is used
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def _reshape(self, model: openvino.runtime.Model, batch_size: int, sequence_length: int, is_decoder=True):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size if not is_decoder else -1
            if is_decoder:
                if inputs.get_any_name().startswith("past_key_values"):
                    shapes[inputs][2] = -1
                elif not inputs.get_any_name().startswith("encoder"):
                    shapes[inputs][1] = -1
        model.reshape(shapes)
        return model
