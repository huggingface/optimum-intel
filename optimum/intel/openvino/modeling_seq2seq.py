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
from typing import Dict, Optional, Tuple

import torch
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

import openvino

from .modeling_base_seq2seq import OVBaseModelForSeq2SeqLM


logger = logging.getLogger(__name__)


class OVModelForSeq2SeqLM(OVBaseModelForSeq2SeqLM, GenerationMixin):
    """
    Sequence-to-sequence model with a language modeling head for OpenVINO inference.
    """

    auto_model_class = AutoModelForSeq2SeqLM

    def __init__(
        self,
        encoder: openvino.pyopenvino.Model,
        decoder: openvino.pyopenvino.Model,
        decoder_with_past: openvino.pyopenvino.Model = None,
        config: transformers.PretrainedConfig = None,
        **kwargs
    ):
        super().__init__(encoder, decoder, decoder_with_past, config, **kwargs)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.encoder = OVEncoder(self.encoder_request, self.device)
        self.decoder = OVDecoder(self.decoder_request, self.device, self.decoder_input_names)
        self.decoder_with_past = (
            OVDecoder(self.decoder_with_past_request, self.device, self.decoder_with_past_input_names)
            if self.use_cache
            else None
        )
        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    def to(self, device: str):
        # Ensure the selected device is supported by OpenVINO
        self._ensure_supported_device(device)
        self._device = device
        self.encoder_request = self._create_infer_request(self.encoder_model)
        self.encoder.request = self.encoder_request
        self.decoder_request = self._create_infer_request(self.decoder_model)
        self.decoder.request = self.decoder_request
        if self.use_cache:
            self.decoder_with_past_request = self._create_infer_request(self.decoder_with_past_model)
            self.decoder_with_past.request = self.decoder_with_past_request
        return self

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
        past=None,
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
            "past_key_values": past,
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


class OVEncoder:
    def __init__(self, request, device: torch.device):
        self.request = request
        self.device = device
        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutput:

        inputs = {
            "input_ids": input_ids,
        }
        # Add the attention_mask inputs when needed
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        # Run inference
        outputs = self.request.infer(inputs)
        outputs = {key.get_any_name(): value for key, value in outputs.items()}
        last_hidden_state = torch.from_numpy(outputs["last_hidden_state"]).to(self.device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OVDecoder:
    def __init__(self, request, device: torch.device, input_names):
        self.request = request
        self.device = device
        self.input_names = input_names
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Seq2SeqLMOutput:

        inputs = {
            "input_ids": input_ids,
            "encoder_attention_mask": encoder_attention_mask,
        }

        # Add the encoder_hidden_states inputs when needed
        if "encoder_hidden_states" in self.input_names:
            inputs["encoder_hidden_states"] = encoder_hidden_states

        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = [past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer]
            # Add the past_key_values to the decoder inputs
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                inputs[input_name] = past_key_value

        # Run inference
        outputs = self.request.infer(inputs)
        outputs = {key.get_any_name(): value for key, value in outputs.items()}

        # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
        # self-attention layer and 2 to the cross-attention layer)
        past_key_values = tuple(
            torch.from_numpy(outputs[key]).to(self.device) for key in outputs if "key_values" in key
        )

        # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
        # cross-attention per decoder layer
        num_pkv = 4
        past_key_values = tuple(past_key_values[i : i + num_pkv] for i in range(0, len(past_key_values), num_pkv))

        logits = torch.from_numpy(outputs["logits"]).to(self.device)

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
