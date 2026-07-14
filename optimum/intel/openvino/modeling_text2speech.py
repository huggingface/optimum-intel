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

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTextToSpectrogram,
    GenerationConfig,
    PretrainedConfig,
)
from transformers.file_utils import add_start_docstrings
from transformers.utils import ModelOutput

from ...exporters.openvino.stateful import model_has_state
from . import OV_DECODER_NAME, OV_ENCODER_NAME
from .configuration import OVConfig, OVWeightQuantizationConfig
from .modeling_base import OVBaseModel, OVModelPart
from .modeling_seq2seq import (
    INPUTS_DOCSTRING,
    OVModelForSeq2SeqLM,
)
from .utils import TemporaryDirectory, classproperty


logger = logging.getLogger(__name__)


class OVTextToSpeechEncoder(OVModelPart):
    _model_name = "encoder"

    def __init__(self, model: openvino.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.hidden_states_output_names = []
        self._main_input = list(self.input_names.keys())[0]

    def forward(self, input_ids, **kwargs):
        self.compile()
        inputs = {self._main_input: input_ids}
        result = self.request(inputs)
        last_hidden_state = torch.from_numpy(result[0])
        encoder_attention_mask = torch.from_numpy(result[1])
        return ModelOutput(last_hidden_state=last_hidden_state, encoder_attention_mask=encoder_attention_mask)


class OVTextToSpeechDecoder(OVModelPart):
    _model_name = "decoder"

    def __init__(self, model: openvino.Model, parent_model: OVBaseModel) -> None:
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
        self.compile()
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

    def reset_state(self) -> None:
        if self.request:
            self.request.reset_state()


class OVTextToSpeechPostNet(OVModelPart):
    _model_name = "postnet"

    def __init__(self, model: openvino.Model, parent_model: OVBaseModel) -> None:
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
        self.compile()
        inputs = {
            "raw_spectrogram": spectrograms,
        }
        result = self.request(inputs)
        postnet_spectrogram = torch.from_numpy(result[0])
        return ModelOutput(postnet_spectrogram=postnet_spectrogram)


class OVTextToSpeechVocoder(OVModelPart):
    _model_name = "vocoder"

    def __init__(self, model: openvino.Model, parent_model: OVBaseModel) -> None:
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
        self.compile()
        inputs = {
            "spectrogram": spectrogram,
        }
        result = self.request(inputs)
        waveform = torch.from_numpy(result[0])
        return ModelOutput(waveform=waveform)


@add_start_docstrings(
    """
    This class provides interface to export and infer text-to-speech models using OpenVINO.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForTextToSpeechSeq2Seq(OVModelForSeq2SeqLM):
    auto_model_class = AutoModelForTextToSpectrogram
    export_feature = "text-to-audio"

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        # Chatterbox models do not ship a config.json recognizable by AutoConfig; their
        # metadata lives in `chatterbox_config.json`. When such a model is detected we
        # bypass the generic library/config inference (which requires a config.json) and
        # route directly to the Chatterbox implementation.
        if kwargs.get("config") is None and not kwargs.get("export", False):
            config = _try_load_chatterbox_config(
                model_id,
                cache_dir=kwargs.get("cache_dir", HUGGINGFACE_HUB_CACHE),
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            if config is not None:
                return _OVModelForChatterboxTextToSpeech._from_pretrained(model_id, config, **kwargs)

        # For Kokoro models, load config via PretrainedConfig since AutoConfig
        # does not recognize the "kokoro" model_type.
        if kwargs.get("config") is None:
            try:
                config = PretrainedConfig.from_pretrained(
                    model_id,
                    cache_dir=kwargs.get("cache_dir", HUGGINGFACE_HUB_CACHE),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                )
                # Detect Kokoro models that lack model_type by checking for
                # characteristic config keys (same heuristic used by CLI export).
                if not getattr(config, "model_type", None):
                    if hasattr(config, "istftnet") and hasattr(config, "plbert"):
                        config.model_type = "kokoro"
                        config.export_model_type = "kokoro"
                if getattr(config, "model_type", None) == "kokoro":
                    kwargs["config"] = config
            except Exception as e:
                logger.warning(f"Could not pre-load config for Kokoro detection: {e}")
        return super().from_pretrained(model_id, **kwargs)

    @classmethod
    def _export(cls, model_id, config, use_cache=False, **kwargs):
        return super()._export(model_id, config, use_cache=use_cache, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        **kwargs,
    ):
        if getattr(config, "model_type", None) == "kokoro":
            return _OVModelForKokoroTextToSpeech._from_pretrained(model_id, config, **kwargs)
        elif getattr(config, "model_type", None) == "chatterbox":
            return _OVModelForChatterboxTextToSpeech._from_pretrained(model_id, config, **kwargs)
        elif getattr(config, "architectures", None) and "SpeechT5ForTextToSpeech" in config.architectures:
            return _OVModelForSpeechT5ForTextToSpeech._from_pretrained(model_id, config, **kwargs)
        else:
            raise ValueError(
                f"{getattr(config, 'architectures', None)} are not supported text-to-audio model using OpenVINO"
            )

    def reshape(self, *args, **kwargs):
        logger.warning("Static shapes are not supported for this model.")
        return self

    def preprocess_input(self, text: str, **kwargs) -> dict:
        """
        Preprocess a text string into model inputs (input_ids and other required tensors).

        Args:
            text: The input text to synthesize.
            **kwargs: Model-specific arguments (e.g., voice, speed, lang_code for Kokoro).

        Returns:
            Dictionary with model inputs ready for `generate()`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `preprocess_input`. "
            "Use the appropriate model-specific subclass."
        )


class _OVModelForSpeechT5ForTextToSpeech(OVModelForTextToSpeechSeq2Seq):
    """
    This class implements an own generate method since we split the pipeline more compact
    to have encoder, decoder, postnet, and vocoder
    """

    @classproperty
    def _all_ov_model_paths(cls) -> Dict[str, str]:
        return {
            "encoder": OV_ENCODER_NAME,
            "decoder": OV_DECODER_NAME,
            "postnet": "openvino_postnet.xml",
            "vocoder": "openvino_vocoder.xml",
        }

    main_input_name = "input_ids"
    _supports_cache_class = True

    def __init__(
        self,
        encoder: openvino.Model,
        decoder: openvino.Model,
        postnet: openvino.Model,
        vocoder: openvino.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = None,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        if dynamic_shapes is not None:
            logger.warning(
                f"`dynamic_shapes` was set to {dynamic_shapes}, but this value will be ignored as only dynamic shapes are supported."
            )

        self.config = config
        self.use_cache = model_has_state(decoder)
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = True
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])

        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self._compile_only = kwargs.get("compile_only", False)

        enable_compilation = kwargs.get("compile", True)
        self.generation_config = kwargs.get("generation_config", GenerationConfig.from_model_config(config))
        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()
        self.encoder = OVTextToSpeechEncoder(encoder, self)
        self.decoder = OVTextToSpeechDecoder(decoder, self)
        self.postnet = OVTextToSpeechPostNet(postnet, self)
        self.vocoder = OVTextToSpeechVocoder(vocoder, self)

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
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        for component in self.components.values():
            component.clear_requests()

    def compile(self):
        for component in self.components.values():
            component.compile()

    @property
    def _component_names(self) -> List[str]:
        return ["encoder", "decoder", "postnet", "vocoder"]

    @property
    def _ov_model_names(self) -> List[str]:
        return self._component_names

    @property
    def ov_models(self) -> Dict[str, openvino.Model]:
        return {name: getattr(component, "model") for name, component in self.components.items()}

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        device = kwargs.pop("device", "CPU")
        dynamic_shapes = kwargs.pop("dynamic_shapes", None)
        ov_config = kwargs.pop("ov_config", None)
        generation_config = kwargs.pop("generation_config", None)
        preprocessors = kwargs.pop("preprocessors", [])
        compile_only = kwargs.pop("compile_only", False)
        enable_compilation = kwargs.pop("compile", True)

        model_file_names = cls._all_ov_model_paths.copy()
        for k in tuple(model_file_names):
            model_file_names[f"{k}_bin"] = model_file_names[k].replace(".xml", ".bin")

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
            encoder_model = OVBaseModel.load_model(file_names["encoder"])
            decoder_model = OVBaseModel.load_model(file_names["decoder"])
            postnet_model = OVBaseModel.load_model(file_names["postnet"])
            vocoder_model = OVBaseModel.load_model(file_names["vocoder"])
        else:
            encoder_model = OVBaseModel._compile_model(
                file_names["encoder"],
                device,
                ov_config,
                model_save_dir,
            )
            decoder_model = OVBaseModel._compile_model(
                file_names["decoder"],
                device,
                ov_config,
                model_save_dir,
            )
            postnet_model = OVBaseModel._compile_model(
                file_names["postnet"],
                device,
                ov_config,
                model_save_dir,
            )
            vocoder_model = OVBaseModel._compile_model(
                file_names["vocoder"],
                device,
                ov_config,
                model_save_dir,
            )
        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except Exception:
                pass

        quantization_config = quantization_config or (OVWeightQuantizationConfig(bits=8) if load_in_8bit else None)
        model = _OVModelForSpeechT5ForTextToSpeech(
            encoder=encoder_model,
            decoder=decoder_model,
            postnet=postnet_model,
            vocoder=vocoder_model,
            config=config,
            device=device,
            dynamic_shapes=dynamic_shapes,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            preprocessors=preprocessors,
            compile_only=compile_only,
            compile=enable_compilation and not quantization_config,
            generation_config=generation_config,
        )

        if quantization_config:
            if hasattr(config, "name_or_path"):
                model_id = config.name_or_path
            else:
                logger.warning(
                    "`model_id` could not be determined from the config. In the case there are default quantization "
                    "configurations for this model, they will not be applied."
                )
            quantization_config = cls._resolve_default_quantization_config(model_id, quantization_config)
            model._apply_quantization(
                quantization_config, compile_only, enable_compilation, model_id, trust_remote_code
            )

        return model

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/speecht5/modeling_speecht5.py#L2464
    # some decoder parts (prenet, wrapper_decoder, and feat_out) are combined into the single piece decoder
    # Finally, we split the pipeline into four parts: encoder, decoder, postnet, and vocoder
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

        encoder_out = self.encoder(input_values)

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

        # clean-up decoder states for new generation
        self.decoder.reset_state()

        while True:
            idx += 1

            decoder_out = self.decoder(
                inputs_embeds=output_sequence,
                speaker_embeddings=speaker_embeddings,
                encoder_last_hidden_state=encoder_last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
            )

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
                    spectrograms = self.postnet(spectrograms)
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
            if self.vocoder is not None:
                outputs = self.vocoder(spectrogram)
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


class _OVModelForKokoroTextToSpeech(OVBaseModel):
    """
    OpenVINO inference model for Kokoro TTS.

    Kokoro is a single-model architecture with inputs (input_ids, ref_s, speed) and
    outputs (waveform, phonemes). Voice embeddings are stored as .bin files in a voices/ subdirectory.
    """

    export_feature = "text-to-audio"
    auto_model_class = AutoModelForTextToSpectrogram

    def __init__(self, model: openvino.Model, config: PretrainedConfig = None, **kwargs):
        # Kokoro model does not support dynamic shapes due to Squeeze op limitations,
        # so we skip the automatic reshape to dynamic shapes.
        kwargs.setdefault("dynamic_shapes", False)
        super().__init__(model, config, **kwargs)
        self._voices = {}
        self._voices_dir = None

    def _reshape(self, model, batch_size, sequence_length, height=None, width=None):
        # Kokoro has inputs with different ranks (speed is 1D), so only reshape
        # dimensions that exist in each input.
        shapes = {}
        for inp in model.inputs:
            shape = inp.get_partial_shape()
            if len(shape) >= 1:
                shape[0] = batch_size
            if len(shape) >= 2:
                shape[1] = sequence_length
            shapes[inp] = shape
        model.reshape(shapes)
        return model

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        model = super()._from_pretrained(
            model_id,
            config=config,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        # Locate voices directory
        if model.model_save_dir is not None:
            voices_dir = Path(model.model_save_dir) / "voices"
            if voices_dir.is_dir():
                model._voices_dir = voices_dir
        return model

    def _load_voice(self, voice_name: str) -> np.ndarray:
        """Load a voice embedding by name, caching results."""
        if voice_name in self._voices:
            return self._voices[voice_name]

        if self._voices_dir is None:
            raise FileNotFoundError("No voices directory found in model directory.")

        voice_path = self._voices_dir / f"{voice_name}.bin"
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice '{voice_name}' not found at {voice_path}. "
                f"Available voices: {[f.stem for f in self._voices_dir.glob('*.bin')]}"
            )

        voice_data = np.fromfile(voice_path, dtype=np.float32)
        self._voices[voice_name] = voice_data
        return voice_data

    @property
    def available_voices(self) -> List[str]:
        """Returns list of available voice names."""
        if self._voices_dir is None or not self._voices_dir.is_dir():
            return []
        return sorted(f.stem for f in self._voices_dir.glob("*.bin"))

    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        ref_s: Union[torch.Tensor, np.ndarray],
        speed: Union[torch.Tensor, np.ndarray, float],
        **kwargs,
    ) -> ModelOutput:
        """
        Run inference on the Kokoro model.

        Args:
            input_ids: Token IDs of shape [batch_size, sequence_length].
            ref_s: Voice style embedding of shape [batch_size, style_dim].
            speed: Speed factor, scalar or array.

        Returns:
            ModelOutput with `waveform` and `phonemes`.
        """
        self.compile()

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.numpy()
        if isinstance(ref_s, torch.Tensor):
            ref_s = ref_s.numpy()
        if isinstance(speed, (int, float)):
            speed = np.array([speed], dtype=np.float32)
        elif isinstance(speed, torch.Tensor):
            speed = speed.numpy()

        inputs = {
            "input_ids": input_ids,
            "ref_s": ref_s,
            "speed": speed,
        }

        outputs = self._inference(inputs)
        waveform = torch.from_numpy(outputs[0])
        phonemes = torch.from_numpy(outputs[1])
        return ModelOutput(waveform=waveform, phonemes=phonemes)

    def generate(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        voice: Optional[str] = None,
        ref_s: Optional[Union[torch.Tensor, np.ndarray]] = None,
        speed: float = 1.0,
        segments: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Generate audio waveform from token IDs or preprocessed segments.

        Args:
            input_ids: Token IDs of shape [batch_size, sequence_length].
            voice: Name of a voice preset (e.g., "af_heart"). Ignored if ref_s is provided.
            ref_s: Voice style embedding. If None, loaded from voice preset.
            speed: Speed factor (default 1.0).
            segments: Optional list produced by ``preprocess_input`` for chunked
                long-text/multilingual synthesis. If provided, each segment is
                synthesized and the resulting waveforms are concatenated.

        Returns:
            Audio waveform tensor.
        """
        if segments is not None:
            waveforms = []
            for segment in segments:
                segment_result = self.forward(
                    input_ids=segment["input_ids"],
                    ref_s=segment["ref_s"],
                    speed=segment.get("speed", speed),
                )
                waveforms.append(segment_result.waveform)
            if not waveforms:
                raise ValueError("No valid segments were provided for Kokoro generation.")
            return torch.cat(waveforms, dim=-1)

        if input_ids is None:
            raise ValueError("`input_ids` must be provided when `segments` are not supplied.")

        if ref_s is None:
            if voice is None:
                voice = "af_heart"
            voice_data = self._load_voice(voice)
            ref_s = voice_data.reshape(1, -1)

        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        elif isinstance(input_ids, np.ndarray):
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)

        if isinstance(ref_s, np.ndarray) and ref_s.ndim == 1:
            ref_s = ref_s.reshape(1, -1)

        result = self.forward(input_ids=input_ids, ref_s=ref_s, speed=speed)
        return result.waveform

    def reshape(self, *args, **kwargs):
        logger.warning("Static shapes are not supported for Kokoro model.")
        return self

    def can_generate(self) -> bool:
        return True

    def preprocess_input(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
        split_pattern: Optional[str] = r"\n+",
        speaker_embedding: Optional[Union["openvino.Tensor", torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ) -> dict:
        """
        Preprocess a text string into model inputs for Kokoro TTS.

        Uses the ``kokoro`` and ``misaki`` packages for grapheme-to-phoneme
        conversion and phoneme tokenization.

        Args:
            text: The input text to synthesize.
            voice: Name of a voice preset (e.g., ``"af_heart"``). Ignored if
                   ``speaker_embedding`` is provided.
            speed: Speed factor (default 1.0).
            lang_code: Language code for G2P (default ``"a"`` for American English).
            speaker_embedding: Pre-selected speaker/style embedding. Accepts an
                ``openvino.Tensor``, ``torch.Tensor``, or ``numpy.ndarray`` of shape
                ``[style_dim]`` or ``[1, style_dim]``. When provided, the ``voice``
                argument is ignored and no voice-pack indexing is performed. This
                mirrors the ``speaker_embedding`` argument of
                ``openvino_genai.Text2SpeechPipeline.generate()``.

        Returns:
            Dictionary with either:
            - ``segments`` for multi-chunk inputs, or
            - ``input_ids``/``ref_s``/``speed`` plus ``segments`` for single-chunk inputs.

        Note:
            Chunking and language-specific G2P are delegated to ``KPipeline.__call__``
            (quiet mode, ``model=False``), so this wrapper does not duplicate
            Kokoro chunking/G2P internals.
        """
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "The `kokoro` and `misaki` packages are required for text preprocessing. "
                "Install them with: pip install kokoro misaki[en]"
            )

        vocab = getattr(self.config, "vocab", None)
        if vocab is None:
            raise ValueError("Model config does not contain 'vocab'. Cannot tokenize phonemes.")

        pipeline = KPipeline(lang_code=lang_code, model=False)
        segments = list(pipeline(text=text, split_pattern=split_pattern))
        if not segments:
            raise ValueError(f"G2P produced no phoneme segments for input text: {text!r}")

        if speaker_embedding is not None:
            # Convert to numpy regardless of source type
            if hasattr(speaker_embedding, "data"):  # openvino.Tensor
                shape = (
                    tuple(speaker_embedding.get_shape())
                    if hasattr(speaker_embedding, "get_shape")
                    else tuple(speaker_embedding.shape)
                )
                speaker_embedding_data = np.array(speaker_embedding.data, dtype=np.float32).reshape(shape)
            elif isinstance(speaker_embedding, torch.Tensor):
                speaker_embedding_data = speaker_embedding.detach().cpu().numpy()
            else:
                speaker_embedding_data = np.asarray(speaker_embedding, dtype=np.float32)
        else:
            speaker_embedding_data = None
            voice_pack = pipeline.load_voice(voice)

        preprocessed_segments = []
        for segment in segments:
            phonemes = segment.phonemes
            if not phonemes:
                continue

            # Tokenize: phoneme string -> token IDs (with BOS/EOS)
            token_ids = [vocab.get(p) for p in phonemes]
            token_ids = [i for i in token_ids if i is not None]
            input_ids = torch.LongTensor([[0, *token_ids, 0]])

            if speaker_embedding_data is not None:
                if speaker_embedding_data.ndim == 3:
                    idx = min(len(phonemes) - 1, speaker_embedding_data.shape[0] - 1)
                    ref_s = speaker_embedding_data[idx]  # -> [1, style_dim]
                elif speaker_embedding_data.ndim == 1:
                    ref_s = speaker_embedding_data.reshape(1, -1)
                else:
                    ref_s = speaker_embedding_data
            else:
                # Voice packs have one embedding per phoneme-sequence length.
                ref_s = voice_pack[min(len(phonemes) - 1, voice_pack.shape[0] - 1)]

            preprocessed_segments.append(
                {
                    "input_ids": input_ids,
                    "ref_s": ref_s,
                    "speed": speed,
                    "phonemes": phonemes,
                    "graphemes": segment.graphemes,
                }
            )

        if not preprocessed_segments:
            raise ValueError(f"No valid phoneme segments were produced for input text: {text!r}")

        if len(preprocessed_segments) == 1:
            single = preprocessed_segments[0]
            return {
                "input_ids": single["input_ids"],
                "ref_s": single["ref_s"],
                "speed": single["speed"],
                "segments": preprocessed_segments,
            }

        return {
            "segments": preprocessed_segments,
            "speed": speed,
        }


def _try_load_chatterbox_config(
    model_id, cache_dir=HUGGINGFACE_HUB_CACHE, token=None, revision=None, local_files_only=False
):
    """Load the Chatterbox inference config from an exported model directory or hub repo.

    Returns a ``PretrainedConfig`` with ``model_type == "chatterbox"`` if a
    ``chatterbox_config.json`` is found, otherwise ``None``.
    """
    config_path = None
    model_path = Path(model_id)
    if model_path.is_dir():
        candidate = model_path / "chatterbox_config.json"
        if candidate.is_file():
            config_path = str(candidate)
    else:
        try:
            config_path = hf_hub_download(
                repo_id=str(model_id),
                filename="chatterbox_config.json",
                cache_dir=cache_dir,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
        except Exception:
            config_path = None

    if config_path is None:
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = PretrainedConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    config.model_type = "chatterbox"
    config.export_model_type = "chatterbox"
    return config


def _chatterbox_punc_norm(text: str) -> str:
    """Light text normalization matching the original Chatterbox front-end."""
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text


class _OVModelForChatterboxTextToSpeech(OVBaseModel):
    """OpenVINO inference for the ResembleAI Chatterbox TTS model.

    The pipeline runs three exported submodels with thin PyTorch glue:

    * ``t3``      -- stateful autoregressive Llama decoder that consumes ``inputs_embeds``
      and produces speech-token logits. Embedding tables and the built-in voice
      conditioning prefix are stored as assets and applied in Python.
    * ``flow``    -- whole S3Gen flow (token -> mel) with the diffusion noise as an input.
    * ``hifigan`` -- vocoder (mel -> waveform).
    """

    export_feature = "text-to-audio"
    auto_model_class = AutoModelForTextToSpectrogram

    SPEECH_VOCAB_SIZE = 6561

    def __init__(self, t3, flow, hifigan, config: PretrainedConfig = None, **kwargs):
        self.config = config
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self._device = kwargs.get("device", "CPU").upper()
        self.ov_config = kwargs.get("ov_config") or {}
        self.is_dynamic = True
        self._compile_only = kwargs.get("compile_only", False)
        self.generation_config = kwargs.get("generation_config", None)
        self._openvino_config = None

        self._t3_model = t3
        self._flow_model = flow
        self._hifigan_model = hifigan
        self.request_t3 = None
        self.request_flow = None
        self.request_hifigan = None

        # Assets and tokenizer
        self._assets = kwargs.get("assets", {})
        self._tokenizer = kwargs.get("tokenizer", None)
        self.preprocessors = kwargs.get("preprocessors", [])
        self.multilingual = bool(getattr(config, "multilingual", False))

        if kwargs.get("compile", True) and not self._compile_only:
            self.compile()

    @staticmethod
    def _core_for_models():
        return openvino.Core()

    @staticmethod
    def _load_tokenizer(tokenizer_path, multilingual: bool = False):
        """Load the Chatterbox text front-end.

        Prefers the original ``MTLTokenizer``/``EnTokenizer`` classes (they implement
        language-specific normalization and the ``[lang]`` token prefix), and falls back
        to a raw ``tokenizers.Tokenizer`` if the ``chatterbox`` package is unavailable.
        """
        if tokenizer_path is None or not os.path.isfile(tokenizer_path):
            logger.warning("Chatterbox tokenizer file not found; text preprocessing will be unavailable.")
            return None
        try:
            if multilingual:
                from chatterbox.models.tokenizers import MTLTokenizer

                return MTLTokenizer(str(tokenizer_path))
            from chatterbox.models.tokenizers import EnTokenizer

            return EnTokenizer(str(tokenizer_path))
        except Exception as e:
            logger.warning(
                f"Could not load the Chatterbox tokenizer class ({e}). Falling back to a raw tokenizer; "
                "language-specific preprocessing will not be applied."
            )
            try:
                from tokenizers import Tokenizer

                return Tokenizer.from_file(str(tokenizer_path))
            except Exception as e2:
                logger.warning(f"Could not load Chatterbox tokenizer: {e2}")
                return None

    def _save_config(self, save_directory):
        # The Chatterbox metadata is persisted as `chatterbox_config.json` rather than the
        # standard config.json (handled in `_save_pretrained`).
        pass

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """Persist the OpenVINO submodels and the Chatterbox assets to ``save_directory``."""
        import shutil

        save_directory = Path(save_directory)
        models = {
            "openvino_t3.xml": self._t3_model,
            "openvino_flow.xml": self._flow_model,
            "openvino_hifigan.xml": self._hifigan_model,
        }
        for file_name, ov_model in models.items():
            openvino.save_model(ov_model, save_directory / file_name)

        # Copy the non-IR artifacts (assets, config and tokenizer) from the source dir.
        if self.model_save_dir is not None:
            src = Path(self.model_save_dir)
            for extra in ("chatterbox_assets.safetensors", "chatterbox_config.json", "tokenizer.json"):
                src_path = src / extra
                if src_path.is_file():
                    shutil.copy(src_path, save_directory / extra)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        from safetensors.torch import load_file

        device = kwargs.pop("device", "CPU")
        ov_config = kwargs.pop("ov_config", None)
        compile_only = kwargs.pop("compile_only", False)
        enable_compilation = kwargs.pop("compile", True)

        file_names = {
            "t3": "openvino_t3.xml",
            "flow": "openvino_flow.xml",
            "hifigan": "openvino_hifigan.xml",
        }
        tokenizer_file = getattr(config, "tokenizer_file", "tokenizer.json")
        extra_files = ["chatterbox_assets.safetensors", tokenizer_file]
        if getattr(config, "multilingual", False):
            extra_files.append("Cangjie5_TC.json")

        if os.path.isdir(model_id):
            model_save_dir = Path(model_id)
            resolved = {k: os.path.join(model_id, v) for k, v in file_names.items()}
            for extra in extra_files:
                resolved[extra] = os.path.join(model_id, extra)
        else:
            resolved = {}
            for name, file_name in {**file_names, **{e: e for e in extra_files}}.items():
                for suffix in [".bin"] if name in file_names else []:
                    bin_name = file_name.replace(".xml", suffix)
                    hf_hub_download(
                        repo_id=str(model_id),
                        filename=bin_name,
                        token=token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                resolved[name] = hf_hub_download(
                    repo_id=str(model_id),
                    filename=file_name,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            model_save_dir = Path(resolved["t3"]).parent

        t3_model = cls._core_for_models().read_model(resolved["t3"])
        flow_model = cls._core_for_models().read_model(resolved["flow"])
        hifigan_model = cls._core_for_models().read_model(resolved["hifigan"])

        assets = load_file(resolved["chatterbox_assets.safetensors"])

        tokenizer = cls._load_tokenizer(
            resolved.get(tokenizer_file), multilingual=getattr(config, "multilingual", False)
        )

        return cls(
            t3=t3_model,
            flow=flow_model,
            hifigan=hifigan_model,
            config=config,
            device=device,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            compile_only=compile_only,
            compile=enable_compilation,
            assets=assets,
            tokenizer=tokenizer,
        )

    def compile(self):
        core = self._core_for_models()
        if self.request_t3 is None:
            self.request_t3 = core.compile_model(self._t3_model, self._device, self.ov_config).create_infer_request()
        if self.request_flow is None:
            self.request_flow = core.compile_model(self._flow_model, self._device, self.ov_config)
        if self.request_hifigan is None:
            self.request_hifigan = core.compile_model(self._hifigan_model, self._device, self.ov_config)

    def clear_requests(self):
        self.request_t3 = None
        self.request_flow = None
        self.request_hifigan = None

    def reshape(self, *args, **kwargs):
        logger.warning("Static shapes are not supported for Chatterbox model.")
        return self

    def can_generate(self) -> bool:
        return True

    # ------------------------------------------------------------------ tokenization
    def _text_to_tokens(self, text: str, language_id: Optional[str] = None) -> torch.Tensor:
        if self._tokenizer is None:
            raise ValueError(
                "The Chatterbox tokenizer is not available. Make sure the tokenizer file is present "
                "in the model directory."
            )
        if self.multilingual and language_id is None:
            language_id = "en"

        # The original Chatterbox tokenizer classes expose `text_to_tokens`, applying
        # language-specific normalization and the `[lang]` prefix for the multilingual model.
        if hasattr(self._tokenizer, "text_to_tokens"):
            if self.multilingual:
                tokens = self._tokenizer.text_to_tokens(text, language_id=language_id.lower())
            else:
                tokens = self._tokenizer.text_to_tokens(text)
            return tokens.to(dtype=torch.long)

        # Raw `tokenizers.Tokenizer` fallback (English only, no language handling).
        text = text.replace(" ", "[SPACE]")
        ids = self._tokenizer.encode(text).ids
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def preprocess_input(self, text: str, language_id: Optional[str] = None, **kwargs) -> dict:
        """Normalize and tokenize text into ``input_ids`` ready for ``generate``.

        Args:
            text: The input text to synthesize.
            language_id: Two-letter language code (e.g. ``"ru"``, ``"fr"``, ``"zh"``) for the
                multilingual model. Ignored by the English-only model.
        """
        text = _chatterbox_punc_norm(text)
        return {"input_ids": self._text_to_tokens(text, language_id=language_id)}

    # ------------------------------------------------------------------ T3 stage
    def _emb(self, weight_key: str, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self._assets[weight_key])

    def _run_t3(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        from transformers.generation.logits_process import (
            MinPLogitsWarper,
            RepetitionPenaltyLogitsProcessor,
            TopPLogitsWarper,
        )

        cfg = self.config
        sot, eot = cfg.start_text_token, cfg.stop_text_token
        sst, est = cfg.start_speech_token, cfg.stop_speech_token

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # CFG: duplicate the sequence (conditional + unconditional).
        text_tokens = torch.cat([input_ids, input_ids], dim=0)
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        text_e = self._emb("text_emb_weight", text_tokens)
        # The unconditional (CFG) branch zeros the token embedding BEFORE the positional
        # embedding is added, so the position information is preserved (matches the reference).
        text_e[1].zero_()
        text_e = text_e + self._assets["text_pos_emb_weight"][: text_tokens.shape[1]]

        cond = self._assets["cond_prefix_emb"].expand(2, -1, -1)
        # The reference pipeline prefixes the speech part with an initial start-of-speech
        # token (inside `prepare_input_embeds`) and then concatenates an explicit BOS token,
        # so the prefill contains two identical speech-token embeddings at position 0.
        bos = torch.full((2, 1), sst, dtype=torch.long)
        bos_e = self._emb("speech_emb_weight", bos) + self._assets["speech_pos_emb_weight"][0:1]
        embeds = torch.cat([cond, text_e, bos_e, bos_e], dim=1)

        rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        topp = TopPLogitsWarper(top_p=top_p)
        minp = MinPLogitsWarper(min_p=min_p)

        self.request_t3.reset_state()

        # The stateful decoder derives RoPE positions from the attention-mask length, which
        # must cover the full sequence seen so far (past cache + current tokens).
        past_len = 0

        def run(ie):
            nonlocal past_len
            am = np.ones((ie.shape[0], ie.shape[1] + past_len), dtype=np.int64)
            res = self.request_t3.infer(
                {
                    "inputs_embeds": ie.numpy().astype(np.float32),
                    "attention_mask": am,
                    "beam_idx": np.arange(ie.shape[0], dtype=np.int32),
                }
            )
            past_len += ie.shape[1]
            return torch.from_numpy(next(iter(res.values())))

        logits = run(embeds)
        generated = bos[:1].clone()  # track only the conditional batch
        predicted = []
        for i in range(max_new_tokens):
            step = logits[:, -1, :]
            cond_logits, uncond_logits = step[0:1], step[1:2]
            scaled = cond_logits + cfg_weight * (cond_logits - uncond_logits)
            ids = generated
            scaled = rep(ids, scaled)
            if temperature != 1.0:
                scaled = scaled / temperature
            scaled = minp(ids, scaled)
            scaled = topp(ids, scaled)
            probs = torch.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            predicted.append(next_token)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.view(-1).item() == est:
                break
            next_e = self._emb("speech_emb_weight", next_token) + self._assets["speech_pos_emb_weight"][i + 1 : i + 2]
            next_e = torch.cat([next_e, next_e])  # CFG
            logits = run(next_e)

        self.request_t3.reset_state()
        if not predicted:
            return torch.zeros((1, 0), dtype=torch.long)
        return torch.cat(predicted, dim=1)

    # ------------------------------------------------------------------ flow + vocoder
    def _run_flow_and_vocoder(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        st = speech_tokens.view(-1)
        st = st[st < self.SPEECH_VOCAB_SIZE]
        token = st.unsqueeze(0).to(torch.float32)
        token_len = torch.tensor([token.shape[1]], dtype=torch.float32)

        prompt_token = self._assets["gen_prompt_token"]
        prompt_token_len = self._assets["gen_prompt_token_len"]
        prompt_feat = self._assets["gen_prompt_feat"]
        embedding = self._assets["gen_embedding"]

        token_mel_ratio = getattr(self.config, "token_mel_ratio", 2)
        n_mels = getattr(self.config, "n_mels", 80)
        total_tokens = prompt_token.shape[1] + token.shape[1]
        mel_t = token_mel_ratio * total_tokens
        noise = torch.randn(1, n_mels, mel_t)

        flow_out = self.request_flow(
            [
                token.numpy(),
                token_len.numpy(),
                prompt_token.numpy(),
                prompt_token_len.numpy(),
                prompt_feat.numpy(),
                embedding.numpy(),
                noise.numpy(),
            ]
        )
        mel = flow_out[0]
        wav = self.request_hifigan([mel])[0]
        return torch.from_numpy(wav)

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        text: Optional[str] = None,
        language_id: Optional[str] = None,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        **kwargs,
    ) -> torch.FloatTensor:
        """Generate an audio waveform from token ids or raw text.

        Args:
            input_ids: Tokenized input from ``preprocess_input``. If not provided, ``text``
                is tokenized internally.
            text: Raw text to synthesize (used when ``input_ids`` is ``None``).
            language_id: Two-letter language code (e.g. ``"ru"``, ``"fr"``, ``"zh"``) for the
                multilingual model; ignored by the English-only model. Used only when the
                text is tokenized internally (i.e. ``input_ids`` is ``None``).
        """
        if input_ids is None:
            if text is None:
                raise ValueError("Either `input_ids` or `text` must be provided.")
            input_ids = self.preprocess_input(text, language_id=language_id)["input_ids"]
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)

        speech_tokens = self._run_t3(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        if speech_tokens.shape[1] == 0:
            raise RuntimeError("Chatterbox T3 produced no speech tokens for the given input.")
        return self._run_flow_and_vocoder(speech_tokens)

    @property
    def sampling_rate(self) -> int:
        return int(getattr(self.config, "sampling_rate", 24000))
