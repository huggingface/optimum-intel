#  Copyright 2025 The HuggingFace Team. All rights reserved.
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
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import openvino as ov
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from torch import nn
from transformers import AutoConfig, GenerationConfig, PretrainedConfig, SpeechT5ForTextToSpeech
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import ModelOutput

from .configuration import OVConfig, OVWeightQuantizationConfig
from .modeling_base import OVBaseModel, OVModelPart
from .modeling_seq2seq import (
    _PROCESSOR_FOR_DOC,
    AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE,
    INPUTS_DOCSTRING,
    SPEECH_SEQ2SEQ_MODEL_DOCSTRING,
    OVModelForSeq2SeqLM,
)
from .utils import (
    TemporaryDirectory,
)


logger = logging.getLogger(__name__)


class OVTextToSpeechEncoder(OVModelPart):
    _model_name = "encoder_model"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.hidden_states_output_names = []
        self._main_input = list(self.input_names.keys())[0]

    def forward(self, input_ids, **kwargs):
        self._compile()
        inputs = {self._main_input: input_ids}
        result = self.request(inputs)
        last_hidden_state = torch.from_numpy(result[0])
        encoder_attention_mask = torch.from_numpy(result[1])
        return ModelOutput(last_hidden_state=last_hidden_state, encoder_attention_mask=encoder_attention_mask)


class OVTextToSpeechDecoder(OVModelPart):
    _model_name = "decoder_model"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.hidden_states_output_names = []
        if len(self.model.outputs) > 2:
            self.hidden_states_output_names = [
                key.get_any_name() for key in self.model.outputs[2:] if "hidden_states" in key.get_any_name()
            ]

    def forward(self, inputs_embeds, speaker_embeddings, encoder_last_hidden_state, encoder_attention_mask, **kwargs):
        self._compile()
        bsz = inputs_embeds.size(0)

        inputs = {
            "inputs_embeds": inputs_embeds,
            "speaker_embeddings": speaker_embeddings,
            "encoder_hidden_states": encoder_last_hidden_state,
            "encoder_attention_mask": encoder_attention_mask,
            "beam_idx": np.arange(bsz, dtype=np.int32),
        }
        result = self.request(inputs)
        output_sequence_out = torch.from_numpy(result[0])
        spectrum = torch.from_numpy(result[1])
        prob = torch.from_numpy(result[2])
        return ModelOutput(output_sequence_out=output_sequence_out, spectrum=spectrum, prob=prob)


class OVTextToSpeechPostNet(OVModelPart):
    _model_name = "postnet_model"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.hidden_states_output_names = []
        if len(self.model.outputs) > 2:
            self.hidden_states_output_names = [
                key.get_any_name() for key in self.model.outputs[2:] if "hidden_states" in key.get_any_name()
            ]

    def forward(self, spectrograms, **kwargs):
        self._compile()
        inputs = {
            "raw_spectrogram": spectrograms,
        }
        result = self.request(inputs)
        postnet_spectrogram = torch.from_numpy(result[0])
        return ModelOutput(postnet_spectrogram=postnet_spectrogram)


class OVTextToSpeechVocoder(OVModelPart):
    _model_name = "vocoder_model"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.hidden_states_output_names = []
        if len(self.model.outputs) > 2:
            self.hidden_states_output_names = [
                key.get_any_name() for key in self.model.outputs[2:] if "hidden_states" in key.get_any_name()
            ]

    def forward(self, spectrogram, **kwargs):
        self._compile()
        inputs = {
            "spectrogram": spectrogram,
        }
        result = self.request(inputs)
        waveform = torch.from_numpy(result[0])
        return ModelOutput(waveform=waveform)


@add_start_docstrings(
    """
    text-to-audio models
    """,
    INPUTS_DOCSTRING,
)
class OVModelForTextToSpeechSeq2Seq(OVModelForSeq2SeqLM):
    auto_model_class = SpeechT5ForTextToSpeech
    main_input_name = "input_ids"
    export_feature = "text-to-audio"
    _supports_cache_class = True
    OV_ENCODER_MODEL_NAME = "openvino_encoder_model.xml"
    OV_DECODER_MODEL_NAME = "openvino_decoder_model.xml"
    OV_POSTNET_MODEL_NAME = "openvino_postnet.xml"
    OV_VOCODER_MODEL_NAME = "openvino_vocoder.xml"

    def __init__(
        self,
        encoder_model: ov.Model,
        decoder_model: ov.Model,
        postnet_model: ov.Model,
        vocoder_model: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        self.config = config
        self.use_cache = kwargs.get("use_cache", True)
        self._model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.postnet_model = postnet_model
        self.vocoder_model = vocoder_model

        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self._compile_only = kwargs.get("compile_only", False)

        enable_compilation = kwargs.get("compile", True)
        self.generation_config = kwargs.get("generation_config", GenerationConfig.from_model_config(config))
        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()
        self.encoder_model = OVTextToSpeechEncoder(self.encoder_model, self)
        self.decoder_model = OVTextToSpeechDecoder(self.decoder_model, self)
        self.postnet_model = OVTextToSpeechPostNet(self.postnet_model, self)
        self.vocoder_model = OVTextToSpeechVocoder(self.vocoder_model, self)

        if enable_compilation and not self._compile_only:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        try:
            self.auto_model_class.register(AutoConfig, self.__class__)
        except AttributeError:
            pass

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please intialize model without this option"
            )

        for _, component in self.components.items():
            component.clear_requests()

    def compile(self):
        for _, component in self.components.items():
            if isinstance(component, OVModelPart):
                component._compile()
            else:
                component.compile()

    @property
    def _component_names(self):
        component_names = ["encoder_model", "decoder_model", "postnet_model", "vocoder_model"]
        return component_names

    @property
    def components(self):
        return {component_name: getattr(self, component_name) for component_name in self._component_names}

    @property
    def _submodel_names(self):
        model_names = ["encoder_model", "decoder_model", "postnet_model", "vocoder_model"]
        return model_names

    @property
    def submodels(self):
        return {submodel_name: getattr(self, submodel_name) for submodel_name in self._submodel_names}

    @property
    def ov_submodels(self) -> Dict[str, ov.runtime.Model]:
        return {submodel_name: getattr(self, submodel_name).model for submodel_name in self._submodel_names}

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        src_models = self.submodels
        dst_file_names = {
            "encoder_model": self.OV_ENCODER_MODEL_NAME,
            "decoder_model": self.OV_DECODER_MODEL_NAME,
            "postnet_model": self.OV_POSTNET_MODEL_NAME,
            "vocoder_model": self.OV_VOCODER_MODEL_NAME,
        }

        for name in self._submodel_names:
            model = src_models[name].model
            dst_file_name = dst_file_names[name]
            dst_path = os.path.join(save_directory, dst_file_name)
            ov.save_model(model, dst_path, compress_to_fp16=False)

        self._save_openvino_config(save_directory)
        if self.generation_config is not None:
            try:
                self.generation_config.save_pretrained(save_directory)
            except Exception as exception:
                logger.warning(
                    f"The generation config will not be saved, saving failed with following error:\n{exception}"
                )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        if decoder_attention_mask is None and decoder_input_ids is not None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids).to(decoder_input_ids.device)

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @add_start_docstrings_to_model_forward(
        SPEECH_SEQ2SEQ_MODEL_DOCSTRING
        + AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="OVModelForTextToSpeechSeq2Seq",
            checkpoint="microsoft/speecht5_tts",
        )
    )
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        return super().forward(
            input_ids=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        model_file_names = {
            "encoder_model": cls.OV_ENCODER_MODEL_NAME,
            "encoder_model_bin": cls.OV_ENCODER_MODEL_NAME.replace(".xml", ".bin"),
            "decoder_model": cls.OV_DECODER_MODEL_NAME,
            "decoder_model_bin": cls.OV_DECODER_MODEL_NAME.replace(".xml", ".bin"),
            "postnet_model": cls.OV_POSTNET_MODEL_NAME,
            "postnet_model_bin": cls.OV_POSTNET_MODEL_NAME.replace(".xml", ".bin"),
            "vocoder_model": cls.OV_VOCODER_MODEL_NAME,
            "vocoder_model_bin": cls.OV_VOCODER_MODEL_NAME.replace(".xml", ".bin"),
        }

        compile_only = kwargs.get("compile_only", False)
        if os.path.isdir(model_id):
            # Load model from a local directory
            model_save_dir = Path(model_id)
            file_names = {k: os.path.join(model_id, model_file_names[k]) for k in model_file_names}
        else:
            file_names = {}
            for name, file_name in model_file_names.items():
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                file_names[name] = model_cache_path
            model_save_dir = Path(model_cache_path).parent
        if not compile_only:
            encoder_model = OVBaseModel.load_model(file_names["encoder_model"])
            decoder_model = OVBaseModel.load_model(file_names["decoder_model"])
            postnet_model = OVBaseModel.load_model(file_names["postnet_model"])
            vocoder_model = OVBaseModel.load_model(file_names["vocoder_model"])
        else:
            encoder_model = OVBaseModel._compile_model(
                file_names["encoder_model"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
            decoder_model = OVBaseModel._compile_model(
                file_names["decoder_model"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
            postnet_model = OVBaseModel._compile_model(
                file_names["postnet_model"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
            vocoder_model = OVBaseModel._compile_model(
                file_names["vocoder_model"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            kwargs["generation_config"] = generation_config
        except Exception:
            pass

        quantization_config = OVBaseModel._prepare_quantization_config(quantization_config, load_in_8bit)
        to_quantize = not compile_only and quantization_config is not None
        if to_quantize:
            kwargs["compile"] = False

        model = OVModelForTextToSpeechSeq2Seq(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            postnet_model=postnet_model,
            vocoder_model=vocoder_model,
            config=config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )

        if to_quantize:
            from optimum.intel.openvino.quantization import OVQuantizer

            quantization_config_copy = copy.deepcopy(quantization_config)
            quantization_config_copy.tokenizer = quantization_config.tokenizer or model_id
            OVQuantizer(model).quantize(ov_config=OVConfig(quantization_config=quantization_config_copy))

        return model

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[nn.Module] = None,
        output_cross_attentions: bool = False,
        return_output_lengths: bool = False,
        **kwargs,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        if speaker_embeddings is None:
            raise ValueError(
                """`speaker_embeddings` must be specified. For example, you can use a speaker embeddings by following
                        the code snippet provided in this link:
                        https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors
                        """
            )
        input_values = input_ids

        if attention_mask is None:
            encoder_attention_mask = 1 - (input_values == self.config.pad_token_id).int()
        else:
            encoder_attention_mask = attention_mask

        bsz = input_values.size(0)

        encoder_out = self.encoder_model(input_values)

        encoder_last_hidden_state = encoder_out.last_hidden_state
        encoder_attention_mask = encoder_out.encoder_attention_mask

        maxlen = int(encoder_last_hidden_state.size(1) * maxlenratio / self.config.reduction_factor)
        minlen = int(encoder_last_hidden_state.size(1) * minlenratio / self.config.reduction_factor)

        # Start the output sequence with a mel spectrum that is all zeros.
        output_sequence = encoder_last_hidden_state.new_zeros(bsz, 1, self.config.num_mel_bins)

        spectrogram = []
        cross_attentions = []
        idx = 0
        result_spectrogram = {}

        while True:
            idx += 1

            decoder_out = self.decoder_model(
                inputs_embeds=output_sequence,
                speaker_embeddings=speaker_embeddings,
                encoder_last_hidden_state=encoder_last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
            )

            # if output_cross_attentions:
            #    cross_attentions.append(torch.cat(decoder_out.cross_attentions, dim=0))

            spectrum = decoder_out.spectrum
            spectrogram.append(spectrum)

            output_sequence = decoder_out.output_sequence_out
            prob = decoder_out.prob

            if idx < minlen:
                continue
            else:
                # If the generation loop is less than maximum length time, check the ones in the batch that have met
                # the prob threshold. Otherwise, assume all have met thresholds and fill other spectrograms for the batch.
                if idx < maxlen:
                    meet_thresholds = torch.sum(prob, dim=-1) >= threshold
                    meet_indexes = torch.where(meet_thresholds)[0].tolist()
                else:
                    meet_indexes = range(len(prob))
                meet_indexes = [i for i in meet_indexes if i not in result_spectrogram]
                if len(meet_indexes) > 0:
                    spectrograms = torch.stack(spectrogram)
                    spectrograms = self.postnet_model(spectrograms)
                    spectrograms = spectrograms.postnet_spectrogram

                    for meet_index in meet_indexes:
                        result_spectrogram[meet_index] = spectrograms[meet_index]
                if len(result_spectrogram) >= bsz:
                    break
        spectrograms = [result_spectrogram[i] for i in range(len(result_spectrogram))]
        if not return_output_lengths:
            spectrogram = (
                spectrograms[0].unsqueeze(0)
                if bsz == 1
                else torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
            )
            if self.vocoder_model is not None:
                outputs = self.vocoder_model(spectrogram)
                outputs = outputs.waveform
            else:
                outputs = spectrogram
            if output_cross_attentions:
                cross_attentions = torch.cat(cross_attentions, dim=2)
                if bsz > 1:
                    cross_attentions = cross_attentions.view(
                        bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
                    )
                outputs = (outputs, cross_attentions)
        else:
            # batched return values should also include the spectrogram/waveform lengths
            spectrogram_lengths = []
            for i in range(bsz):
                spectrogram_lengths.append(spectrograms[i].size(0))
            if vocoder is None:
                spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
                outputs = (spectrograms, spectrogram_lengths)
            else:
                waveforms = []
                spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
                waveforms = vocoder(spectrograms)
                waveform_lengths = [int(waveforms.size(1) / max(spectrogram_lengths)) * i for i in spectrogram_lengths]
                outputs = (waveforms, waveform_lengths)
            if output_cross_attentions:
                cross_attentions = torch.cat(cross_attentions, dim=2)
                cross_attentions = cross_attentions.view(
                    bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
                )
                outputs = (*outputs, cross_attentions)
        return outputs
