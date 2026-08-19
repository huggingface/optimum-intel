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

import gc
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino
import torch
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
                if getattr(config, "model_type", None) in ("kokoro", "qwen3_tts"):
                    kwargs["config"] = config
            except Exception as e:
                logger.warning(f"Could not pre-load config for text-to-speech model detection: {e}")

        # Qwen3-TTS is a multi-component autoregressive TTS model with a fully custom
        # generation orchestration, so it is handled by a dedicated runtime class.
        if _is_qwen3_tts_config(kwargs.get("config")):
            # ``export`` is honoured by the dedicated runtime (it converts the checkpoint
            # first); ``compile`` has no meaning there, as each component compiles when it is
            # installed.
            kwargs.pop("compile", None)
            return _OVModelForQwen3TTS.from_pretrained(model_id, **kwargs)

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
        elif getattr(config, "architectures", None) and "SpeechT5ForTextToSpeech" in config.architectures:
            return _OVModelForSpeechT5ForTextToSpeech._from_pretrained(model_id, config, **kwargs)
        else:
            raise ValueError(f"{getattr(config, 'model_type')} are not supported text-to-audio model using OpenVINO")

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


# ---------------------------------------------------------------------------
# Qwen3-TTS
#
# Qwen3-TTS (``Qwen3TTSForConditionalGeneration``) is a multi-component autoregressive
# TTS system that is distributed as a ``trust_remote_code`` model (the modelling code
# lives in the ``qwen_tts`` package). It is composed of:
#
# * a 28-layer *talker* decoder that autoregressively predicts the first codebook of
#   each acoustic frame (interleaved m-RoPE, KV-cache),
# * a 5-layer *code predictor* (sub-talker) that is run as a nested loop inside every
#   talker step to predict the remaining 15 codebook groups of a frame,
# * an ECAPA-TDNN *speaker encoder* (only for the ``base`` / voice-clone variant),
# * a neural audio *codec* (``speech_tokenizer``) used to encode the reference audio
#   into codes (ICL mode) and to decode the generated codes back into a 24 kHz waveform.
#
# Because the generation orchestration is highly model specific, the runtime keeps the
# original PyTorch orchestration from ``qwen_tts`` and offloads all five neural components
# to OpenVINO (hybrid component-wise export): the talker and code-predictor decoder stacks,
# the speaker encoder, and both directions of the codec. Everything in between - sampling,
# m-RoPE index math, ICL prompt assembly, chunked decoding - stays in PyTorch.
# ---------------------------------------------------------------------------

# File names of the serialized OpenVINO IRs, one per offloaded component.
_TALKER_OV_IR_NAME = "openvino_talker_model.xml"
_CODE_PREDICTOR_OV_IR_NAME = "openvino_code_predictor_model.xml"
_SPEAKER_ENCODER_OV_IR_NAME = "openvino_speaker_encoder.xml"
_CODEC_ENCODER_OV_IR_NAME = "openvino_codec_encoder.xml"
_CODEC_DECODER_OV_IR_NAME = "openvino_codec_decoder.xml"
_TEXT_EMBEDDINGS_OV_IR_NAME = "openvino_text_embeddings.xml"
_TALKER_EMBEDDINGS_OV_IR_NAME = "openvino_talker_embeddings.xml"
_CODE_PREDICTOR_EMBEDDINGS_OV_IR_NAME = "openvino_code_predictor_embeddings.xml"

# Every IR a Qwen3-TTS export can contain.
_QWEN3_TTS_OV_IR_NAMES = (
    _TALKER_OV_IR_NAME,
    _CODE_PREDICTOR_OV_IR_NAME,
    _TEXT_EMBEDDINGS_OV_IR_NAME,
    _TALKER_EMBEDDINGS_OV_IR_NAME,
    _CODE_PREDICTOR_EMBEDDINGS_OV_IR_NAME,
    _SPEAKER_ENCODER_OV_IR_NAME,
    _CODEC_ENCODER_OV_IR_NAME,
    _CODEC_DECODER_OV_IR_NAME,
)

# Weight-only compression is applied to the language-model side of the pipeline and kept away
# from the neural codec, the same split diffusion pipelines use to leave the VAE alone. The
# codec is a waveform autoencoder rather than a classifier over a large vocabulary: measured on
# this model, int8 weights cost the vocoder ~29 dB of SNR (47 dB -> 18 dB) and drop the
# encoder's exact code agreement from 96% to 40%, while the decoder stacks and the embedding
# tables tolerate it. The speaker encoder is excluded too - it is 9M parameters, so compressing
# it saves nothing worth the risk to voice similarity.
_QWEN3_TTS_COMPRESSIBLE_OV_IR_NAMES = (
    _TALKER_OV_IR_NAME,
    _CODE_PREDICTOR_OV_IR_NAME,
    _TEXT_EMBEDDINGS_OV_IR_NAME,
    _TALKER_EMBEDDINGS_OV_IR_NAME,
    _CODE_PREDICTOR_EMBEDDINGS_OV_IR_NAME,
)

# Of those, the components 4-bit weights are worth spending on, so `--weight-format int4`
# produces a mixed int4/int8 model. The talker stack is 60% of the pipeline's parameters and
# its job is picking one coarse code per 80 ms frame, which survives aggressive quantization:
# measured here, int4 keeps 100% top-1 code agreement with the uncompressed graph. Everything
# else stays 8-bit even when int4 is requested, because it is either small enough that 4 bits
# buy little or carries detail that quantizes badly:
#   * the code predictor is 11% of the weights but emits 15 of every 16 codes - all the fine
#     acoustic structure - and at int4 it flips 5% of them, to save ~35 MB;
#   * the embedding tables are the model's input representation, which is why NNCF's own int4
#     defaults already fall back to 8-bit for them.
# Within an IR, `--ratio` still splits layers between 4-bit and the 8-bit backup precision as
# usual; this only decides which IRs are offered 4-bit at all.
_QWEN3_TTS_INT4_OV_IR_NAMES = (_TALKER_OV_IR_NAME,)


def _resolve_ir_dir(model_id, cache_dir) -> Path:
    """Resolve a writable directory for the Qwen3-TTS OpenVINO IRs.

    Uses the model directory when ``model_id`` is a local path, otherwise a stable
    location under the Hugging Face cache keyed by the (sanitized) model id.
    """
    path = Path(str(model_id))
    if path.is_dir():
        return path
    sanitized = str(model_id).replace("/", "--")
    base = Path(cache_dir) if cache_dir else Path(HUGGINGFACE_HUB_CACHE)
    return base / "openvino_qwen3_tts" / sanitized


@contextmanager
def _qwen3_tts_weightless_codec(model_id):
    """Let ``qwen_tts`` build the neural codec from its config when its weights are absent.

    Every codec parameter is baked into the exported ``codec_encoder`` / ``codec_decoder`` IRs,
    so the export does not copy ``speech_tokenizer/*.safetensors``. ``Qwen3TTSTokenizer`` would
    still insist on a checkpoint, so within this context it is built structurally instead: the
    modules are materialized on the meta device (no allocation - nothing ever reads their
    weights, both entry points being replaced by ``_install_ov_codec_*``) while the feature
    extractor and config, which the surrounding Python code does read, load normally.

    Outside this context - and for export directories that still carry codec weights - the
    original loader is used unchanged.
    """
    try:
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    except ImportError as exc:
        raise ImportError(
            "Qwen3-TTS requires the `qwen_tts` package. Install it with `pip install qwen-tts`."
        ) from exc

    original_from_pretrained = Qwen3TTSTokenizer.from_pretrained

    def from_config(cls, pretrained_model_name_or_path, **kwargs):
        try:
            from qwen_tts.core import (
                Qwen3TTSTokenizerV1Config,
                Qwen3TTSTokenizerV1Model,
                Qwen3TTSTokenizerV2Config,
                Qwen3TTSTokenizerV2Model,
            )
        except ImportError as exc:
            raise ImportError(
                "Qwen3-TTS requires the `qwen_tts` package. Install it with `pip install qwen-tts`."
            ) from exc
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

        for config_cls, model_cls in (
            (Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model),
            (Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model),
        ):
            AutoConfig.register(config_cls.model_type, config_cls, exist_ok=True)
            AutoModel.register(config_cls, model_cls, exist_ok=True)

        instance = cls()
        instance.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        with torch.device("meta"):
            instance.model = AutoModel.from_config(instance.config)
        instance.model.eval()
        instance.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        # ``device`` is a plain attribute on the wrapper, so it can point at CPU even though the
        # (unused) module parameters live on meta; the codec's own tensors all come from OpenVINO.
        instance.device = torch.device("cpu")
        return instance

    Qwen3TTSTokenizer.from_pretrained = classmethod(from_config)
    try:
        yield
    finally:
        Qwen3TTSTokenizer.from_pretrained = original_from_pretrained


def _qwen3_tts_codec_weights_present(model_id) -> bool:
    """True when the model directory still ships the codec checkpoint."""
    codec_dir = Path(str(model_id)) / "speech_tokenizer"
    if not codec_dir.is_dir():
        # Hub repos are resolved by ``qwen_tts`` itself; assume the original layout.
        return True
    return any(codec_dir.glob("*.safetensors")) or any(codec_dir.glob("*.bin"))


def _qwen3_tts_weights_present(model_id) -> bool:
    """True when the model directory still ships the main Qwen3-TTS checkpoint."""
    model_dir = Path(str(model_id))
    if not model_dir.is_dir():
        return True
    return any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))


def _strip_qwen3_tts_weights(model: "torch.nn.Module") -> None:
    """Turn a meta-device Qwen3-TTS module tree into a structural, weightless one.

    Every parameter is swapped for an empty CPU tensor. That keeps ``model.device`` and
    ``model.dtype`` - which ``qwen_tts`` reads when it builds its prompt tensors - reporting
    CPU/float32 without allocating the checkpoint, and makes any weight that is unexpectedly
    still needed fail loudly on shape rather than silently return garbage.

    Rotary ``inv_freq`` buffers are the exception: they are non-persistent, derived from the
    config at construction time, and read by the PyTorch side of the decoder-stack shims, so
    they are recomputed on CPU.
    """
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is not None:
                module._parameters[name] = nn.Parameter(torch.empty(0), requires_grad=False)

    for module in model.modules():
        if hasattr(module, "rope_init_fn"):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, torch.device("cpu"))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = inv_freq

    remaining = [name for name, buffer in model.named_buffers() if buffer.device.type == "meta"]
    if remaining:
        raise RuntimeError(f"Qwen3-TTS: buffers left on the meta device and cannot be rebuilt: {remaining}")


def _qwen3_tts_weight_compression_config(load_in_8bit, quantization_config):
    """Resolve the weight-compression request, mirroring the other OpenVINO model classes."""
    from .configuration import OVWeightQuantizationConfig

    if quantization_config is not None:
        if isinstance(quantization_config, dict):
            return OVWeightQuantizationConfig.from_dict(quantization_config)
        return quantization_config
    if load_in_8bit:
        return OVWeightQuantizationConfig(bits=8)
    return None


def _export_qwen3_tts(model_id, cache_dir, weight_compression=None):
    """Convert a Qwen3-TTS checkpoint to OpenVINO and return the directory holding the IRs.

    Backs ``from_pretrained(..., export=True)``. The output goes to the directory
    :func:`_resolve_ir_dir` resolves for this model, so a second load finds the IRs already
    there instead of converting again.
    """
    from optimum.exporters.openvino import main_export

    source = Path(str(model_id))
    if source.is_dir() and (source / _TALKER_OV_IR_NAME).is_file():
        return source  # already an exported directory

    # Never convert into the source directory: it may be a read-only Hub snapshot, and mixing
    # IRs into a checkpoint makes the result neither one thing nor the other.
    sanitized = re.sub(r"[^\w.-]+", "--", str(model_id)).strip("-")
    # Compressed and uncompressed conversions of the same checkpoint must not share a cache
    # entry, or the first one exported would be reused for both.
    if weight_compression is not None:
        sanitized += f"--{getattr(weight_compression, 'bits', 8)}bit"
    base = Path(cache_dir) if cache_dir else Path(HUGGINGFACE_HUB_CACHE)
    output_dir = base / "openvino_qwen3_tts" / sanitized
    if (output_dir / _TALKER_OV_IR_NAME).is_file():
        logger.info(f"Qwen3-TTS: reusing the OpenVINO export at {output_dir}.")
        return output_dir

    logger.info(f"Qwen3-TTS: exporting {model_id} to OpenVINO in {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_export(
        model_name_or_path=str(model_id),
        output=output_dir,
        task="text-to-audio",
        cache_dir=cache_dir,
    )
    if weight_compression is not None:
        compress_qwen3_tts_irs(output_dir, weight_compression)
    return output_dir


def _build_weightless_qwen3_tts_pipeline(model_id, generate_config_name: str = "generation_config.json"):
    """Build the ``qwen_tts`` pipeline for an export that carries no PyTorch weights.

    Mirrors ``Qwen3TTSModel.from_pretrained`` / ``Qwen3TTSForConditionalGeneration.from_pretrained``
    but constructs the model from its config on the meta device instead of loading a
    checkpoint, since every parameter has been exported to an OpenVINO IR. The processor,
    configs and generation defaults - all of which the surrounding Python code really does
    read - are loaded normally.
    """
    import json

    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
        from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError(
            "Qwen3-TTS requires the `qwen_tts` package. Install it with `pip install qwen-tts`."
        ) from exc

    model_dir = Path(str(model_id))
    config = Qwen3TTSConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = Qwen3TTSForConditionalGeneration(config)
    _strip_qwen3_tts_weights(model)
    model.eval()

    with _qwen3_tts_weightless_codec(model_dir):
        try:
            from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        except ImportError as exc:
            raise ImportError(
                "Qwen3-TTS requires the `qwen_tts` package. Install it with `pip install qwen-tts`."
            ) from exc

        model.load_speech_tokenizer(Qwen3TTSTokenizer.from_pretrained(model_dir / "speech_tokenizer"))

    with open(model_dir / generate_config_name, encoding="utf-8") as generate_config_file:
        model.load_generate_config(json.load(generate_config_file))

    processor = Qwen3TTSProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    return Qwen3TTSModel(model=model, processor=processor, generate_defaults=model.generate_config)


def _as_float32_numpy(tensor) -> np.ndarray:
    """Return a contiguous float32 numpy view of a torch tensor or array."""
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(tensor, dtype=np.float32)
    return np.ascontiguousarray(tensor.detach().cpu().to(torch.float32).numpy())


def _make_ov_embedding_forward(compiled, embedding_dim, step=None):
    """Build an OpenVINO-backed ``forward`` for one embedding table.

    ``step`` is passed only for the code predictor's stacked per-depth tables, which share a
    graph and select a depth at run time. ``qwen_tts`` looks embeddings up with 0-d, 1-d and
    2-d id tensors while the graph has a fixed rank, so ids are flattened to ``[1, N]`` on the
    way in and the original shape is restored on the way out.
    """
    extra_inputs = [] if step is None else [np.array(step, dtype=np.int64)]

    def ov_forward(input_ids):
        ids = input_ids.reshape(1, -1).to(torch.int64).numpy()
        embeddings = torch.from_numpy(compiled([ids, *extra_inputs])[0]).clone()
        return embeddings.reshape(*input_ids.shape, embedding_dim)

    return ov_forward


def _make_ov_decoder_stack_forward(decoder_model, compiled, rope_fn):
    """Build an OpenVINO-backed ``forward`` for one Qwen3-TTS decoder stack.

    Shared by the talker and the code predictor, whose exported graphs have the same
    stateless, KV-explicit signature and differ only in how the rotary ``cos``/``sin`` are
    produced (``rope_fn``). The returned callable preserves the PyTorch I/O contract of
    ``Qwen3TTS*Model.forward`` (``DynamicCache`` in and out, ``BaseModelOutputWithPast``),
    so the ``qwen_tts`` generation loops around it are untouched.

    The graph returns only the newly computed keys/values; concatenating them onto the past
    happens here, in a per-stack store that is reset whenever a sequence starts over.

    The output head is folded into the same graph, so every call also produces logits. They
    are handed back through ``head_state`` (returned alongside the forward), because the
    PyTorch code applies the head one call later, from a different module.
    """
    from transformers import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    cfg = decoder_model.config
    num_layers = len(decoder_model.layers)
    num_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    neg = torch.finfo(torch.float32).min

    # Per-call KV store (mirrors the HF cache, but version independent).
    state: Dict[str, Any] = {"k": None, "v": None}
    # Logits produced by the folded output head, consumed by the caller right after.
    head_state: Dict[str, Any] = {"logits": None, "hidden_shape": None}

    def _build_mask(bs_, seq_, past_len, attention_mask):
        """Build the additive [B, 1, seq, kv] causal mask the graph consumes.

        Both call sites hand the stack the 2D padding mask that ``generate`` maintains and
        rely on the model to derive the causal mask, which is what happens here. A caller
        that already built the 4D additive mask is passed straight through.
        """
        total = past_len + seq_
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.to(torch.float32)

        rows = torch.arange(seq_).view(seq_, 1)
        cols = torch.arange(total).view(1, total)
        allowed = cols <= (past_len + rows)
        mask = torch.zeros(seq_, total, dtype=torch.float32)
        mask = mask.masked_fill(~allowed, neg)
        mask = mask.view(1, 1, seq_, total).expand(bs_, 1, seq_, total).clone()
        if attention_mask is not None:
            pad = attention_mask[:, :total] == 0
            mask = mask.masked_fill(pad.view(bs_, 1, 1, total), neg)
        return mask

    def ov_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        step=None,
        **kw,
    ):
        if past_key_values is None:
            past_key_values = DynamicCache()
        inputs_embeds = inputs_embeds.to(torch.float32)
        bs_, seq_ = inputs_embeds.shape[0], inputs_embeds.shape[1]
        past_len = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_)

        cos, sin = rope_fn(position_ids, cache_position, bs_, seq_, inputs_embeds)
        mask = _build_mask(bs_, seq_, past_len, attention_mask)

        if past_len == 0 or state["k"] is None:
            past_k = torch.zeros(num_layers, bs_, num_kv, 0, head_dim, dtype=torch.float32)
            past_v = torch.zeros(num_layers, bs_, num_kv, 0, head_dim, dtype=torch.float32)
        else:
            past_k = state["k"]
            past_v = state["v"]

        graph_inputs = [
            inputs_embeds.numpy(),
            mask.numpy(),
            cos.numpy(),
            sin.numpy(),
            past_k.numpy(),
            past_v.numpy(),
        ]
        if step is not None:
            graph_inputs.append(np.array(step, dtype=np.int64))
        outputs = compiled(graph_inputs)
        hidden = torch.from_numpy(outputs[0])
        head_state["logits"] = torch.from_numpy(outputs[1]).clone()
        head_state["hidden_shape"] = tuple(hidden.shape)
        new_k = torch.from_numpy(outputs[2])
        new_v = torch.from_numpy(outputs[3])

        if past_k.shape[3] == 0:
            state["k"], state["v"] = new_k, new_v
        else:
            state["k"] = torch.cat([past_k, new_k], dim=3)
            state["v"] = torch.cat([past_v, new_v], dim=3)

        # Keep the HF cache length in sync so cache_position is computed correctly.
        for idx in range(num_layers):
            past_key_values.update(new_k[idx], new_v[idx], idx)

        hidden_states = (hidden,) if output_hidden_states else None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )

    return ov_forward, head_state


def _is_qwen3_tts_config(config: Optional["PretrainedConfig"]) -> bool:
    """Return True when the given config describes a Qwen3-TTS model."""
    if config is None:
        return False
    if getattr(config, "model_type", None) == "qwen3_tts":
        return True
    architectures = getattr(config, "architectures", None) or []
    return "Qwen3TTSForConditionalGeneration" in architectures


def compress_qwen3_tts_irs(ir_dir, quantization_config, output_dir=None) -> None:
    """Weight-compress the exported Qwen3-TTS IRs in ``ir_dir``.

    Shared by ``optimum-cli export openvino --weight-format ...`` (through
    :meth:`_OVModelForQwen3TTS._apply_quantization`) and by ``main_export`` when an
    ``OVConfig`` carrying a quantization config is passed, so both entry points produce the
    same model.
    """
    from openvino import save_model

    from .configuration import OVWeightQuantizationConfig
    from .quantization import _weight_only_quantization

    ir_dir = Path(ir_dir)
    output_dir = Path(output_dir) if output_dir is not None else ir_dir
    core = openvino.Core()

    requested_bits = (
        quantization_config.get("bits") if isinstance(quantization_config, dict) else quantization_config.bits
    )
    fallback_config = None
    if requested_bits is not None and requested_bits < 8:
        symmetric = (
            quantization_config.get("sym", False)
            if isinstance(quantization_config, dict)
            else getattr(quantization_config, "sym", False)
        )
        fallback_config = OVWeightQuantizationConfig(bits=8, sym=symmetric, group_size=-1, ratio=1.0)

    for ir_name in _QWEN3_TTS_COMPRESSIBLE_OV_IR_NAMES:
        ir_path = ir_dir / ir_name
        if not ir_path.is_file():
            continue
        config = quantization_config
        if fallback_config is not None and ir_name not in _QWEN3_TTS_INT4_OV_IR_NAMES:
            config = fallback_config
        bits = config.get("bits") if isinstance(config, dict) else config.bits
        logger.info(f"Qwen3-TTS: applying {bits}-bit weight compression to {ir_name}.")
        compressed = _weight_only_quantization(core.read_model(ir_path), config)

        # The source weights are still memory-mapped, both by the model this method was
        # called on and by ``read_model`` above, so the compressed graph is written under a
        # temporary name and renamed into place. Writing over a mapped .bin would truncate
        # it underneath its mappings and take the process down with SIGBUS.
        staged_xml = output_dir / f"{ir_path.stem}.compressed.xml"
        save_model(compressed, staged_xml, compress_to_fp16=False)
        del compressed
        gc.collect()

        target_xml = output_dir / ir_name
        os.replace(staged_xml, target_xml)
        os.replace(staged_xml.with_suffix(".bin"), target_xml.with_suffix(".bin"))


class _OVModelForQwen3TTS:
    """OpenVINO-backed runtime for Qwen3-TTS.

    The class loads the reference PyTorch pipeline from the ``qwen_tts`` package and
    exposes a small, OpenVINO-friendly inference surface (``preprocess_input`` and
    ``generate``). The heavy neural sub-networks are progressively replaced by
    OpenVINO models while the model-specific generation orchestration is reused from
    ``qwen_tts``.
    """

    export_feature = "text-to-audio"
    main_input_name = "input_ids"

    def __init__(self, pipeline, config: "PretrainedConfig", model_save_dir=None, **kwargs):
        # ``pipeline`` is a ``qwen_tts.Qwen3TTSModel`` wrapper instance.
        self._pipeline = pipeline
        self.model = pipeline.model
        self.processor = pipeline.processor
        self.config = config
        self.model_save_dir = model_save_dir
        self._device = "CPU"
        # Populated by ``_install_ov_components``; components with no IR stay on PyTorch.
        self._ov_ir_paths: Dict[str, str] = {}
        # Overridden in ``from_pretrained``; exports drop both checkpoints.
        self._codec_weights_present = True
        self._weights_present = True
        self.sampling_rate = int(getattr(self.model, "speaker_encoder_sample_rate", 24000))
        try:
            self.sampling_rate = int(self.model.speech_tokenizer.get_output_sample_rate())
        except Exception:
            pass

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Optional["PretrainedConfig"] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        export: bool = False,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Optional[Any] = None,
        **kwargs,
    ) -> "_OVModelForQwen3TTS":
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise ImportError(
                "Qwen3-TTS requires the `qwen_tts` package to be installed. " "Install it with: pip install qwen-tts"
            ) from exc

        # Only forward arguments understood by the underlying loader.
        load_kwargs: Dict[str, Any] = {}
        dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", None))
        # OpenVINO inference runs in float32 on CPU; default to float32 for clean,
        # numerically-faithful conversion of the offloaded sub-networks.
        load_kwargs["dtype"] = dtype if dtype is not None else torch.float32
        if token is not None:
            load_kwargs["token"] = token
        if revision is not None:
            load_kwargs["revision"] = revision
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        load_kwargs["force_download"] = force_download
        load_kwargs["local_files_only"] = local_files_only

        # ``export=True`` accepts an original checkpoint and converts it first, matching the
        # other OpenVINO model classes. The IRs land in the same location a later load resolves
        # to, so the conversion is done once and reused.
        weight_compression = _qwen3_tts_weight_compression_config(load_in_8bit, quantization_config)
        if export:
            model_id = _export_qwen3_tts(model_id, cache_dir, weight_compression)
        elif weight_compression is not None:
            raise ValueError(
                "Weight compression of Qwen3-TTS is applied while exporting. Pass `export=True` to convert the "
                "checkpoint here, or compress at export time with "
                "`optimum-cli export openvino --weight-format int8/int4`."
            )

        # An export produced by this exporter carries no weights at all - every parameter is
        # in an IR - so the pipeline is built structurally. Original checkpoints (a Hub repo,
        # or a directory exported by an older version) keep taking the normal loader.
        weights_present = _qwen3_tts_weights_present(model_id)
        codec_weights_present = _qwen3_tts_codec_weights_present(model_id)
        if not weights_present:
            pipeline = _build_weightless_qwen3_tts_pipeline(model_id)
            # That path always builds the codec structurally too, whatever is on disk.
            codec_weights_present = False
        elif codec_weights_present:
            pipeline = Qwen3TTSModel.from_pretrained(str(model_id), **load_kwargs)
        else:
            with _qwen3_tts_weightless_codec(model_id):
                pipeline = Qwen3TTSModel.from_pretrained(str(model_id), **load_kwargs)
        pipeline.model.eval()

        if config is None:
            config = pipeline.model.config

        instance = cls(pipeline=pipeline, config=config, model_save_dir=model_id)
        instance._ir_dir = _resolve_ir_dir(model_id, cache_dir)
        instance._codec_weights_present = codec_weights_present
        instance._weights_present = weights_present
        instance._install_ov_components()
        return instance

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        # OpenVINO components run on CPU; ignore device moves to stay API compatible.
        return self

    def can_generate(self) -> bool:
        return True

    def _compile_ov_component(self, ir_name: str, label: str):
        """Read and compile one exported component IR, or raise when it is not on disk."""
        ir_dir = Path(getattr(self, "_ir_dir", None) or _resolve_ir_dir(self.model_save_dir, None))
        ir_xml = ir_dir / ir_name
        if not ir_xml.is_file():
            raise FileNotFoundError(f"{label} OpenVINO IR not found at {ir_xml}")
        core = openvino.Core()
        logger.info(f"Qwen3-TTS: loading {label} OpenVINO IR from {ir_xml}.")
        compiled = core.compile_model(core.read_model(ir_xml), self._device)
        self._ov_ir_paths[label] = str(ir_xml)
        return compiled

    def _install_ov_components(self) -> None:
        """Offload every exported Qwen3-TTS component to OpenVINO.

        The components are installed independently: each one replaces the forward of the
        corresponding PyTorch module while the ``qwen_tts`` orchestration around it is left
        untouched, and each one falls back to PyTorch on its own when its IR is missing (or
        anything else fails). A directory exported by an older version, holding only the
        talker IR, therefore still runs.
        """
        self._install_ov_talker()
        self._install_ov_code_predictor()
        self._install_ov_embeddings()
        self._install_ov_speaker_encoder()
        self._install_ov_codec_encoder()
        self._install_ov_codec_decoder()

        # Falling back to PyTorch is only possible where PyTorch weights exist. When the export
        # left them out, the modules behind them are structural only, so a component whose IR
        # is missing has to be an error rather than a silent switch to empty weights.
        required = set()
        if not self._codec_weights_present:
            required |= {"codec encoder", "codec decoder"}
        if not self._weights_present:
            required |= {
                "talker",
                "code predictor",
                "text embeddings",
                "talker embeddings",
                "code predictor embeddings",
            }
            if self.model.speaker_encoder is not None:
                required.add("speaker encoder")
        missing = required - set(self._ov_ir_paths)
        if missing:
            raise RuntimeError(
                f"{self.model_save_dir} ships no PyTorch weights for these components, so they can only run "
                f"from OpenVINO, but the IR for: {', '.join(sorted(missing))} could not be loaded. "
                "Re-export the model with `optimum-cli export openvino`."
            )

    def _install_ov_talker(self) -> None:
        """Offload the talker decoder stack (28 layers, run every frame) to OpenVINO.

        The rotary sections are merged outside the graph: the talker uses interleaved
        m-RoPE, and rather than duplicating that merge, ``apply_multimodal_rotary_pos_emb``
        is evaluated on a basis probe whose first half is 1 and second half 0 - the rotated
        output's halves are then exactly the merged cos and sin.
        """
        try:
            from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

            talker = self.model.talker
            talker_model = talker.model
            talker_model.eval()
            cfg = talker_model.config
            mrope_section = cfg.rope_scaling["mrope_section"]
            mrope_interleaved = cfg.rope_scaling.get("interleaved", False)
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            half = head_dim // 2

            compiled = self._compile_ov_component(_TALKER_OV_IR_NAME, "talker")

            def merged_mrope(position_ids, cache_position, batch_size, seq_len, inputs_embeds):
                if position_ids is None:
                    position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
                elif position_ids.ndim == 2:
                    position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                    position_ids = position_ids[1:]

                cos, sin = talker_model.rotary_emb(inputs_embeds, position_ids)
                probe = torch.cat(
                    [
                        torch.ones(batch_size, 1, seq_len, half, dtype=torch.float32),
                        torch.zeros(batch_size, 1, seq_len, half, dtype=torch.float32),
                    ],
                    dim=-1,
                )
                merged, _ = apply_multimodal_rotary_pos_emb(
                    probe, probe, cos.to(torch.float32), sin.to(torch.float32), mrope_section, mrope_interleaved
                )
                merged = merged.squeeze(1)
                return (
                    torch.cat([merged[..., :half], merged[..., :half]], dim=-1),
                    torch.cat([merged[..., half:], merged[..., half:]], dim=-1),
                )

            ov_forward, head_state = _make_ov_decoder_stack_forward(talker_model, compiled, merged_mrope)
            talker_model.forward = ov_forward

            def ov_codec_head(hidden_states):
                # ``codec_head`` is folded into the talker graph, which computed these logits
                # on the call that produced ``hidden_states`` a moment ago.
                if head_state["hidden_shape"] != tuple(hidden_states.shape):
                    raise RuntimeError(
                        "Qwen3-TTS: codec_head called on hidden states the talker graph did not just produce "
                        f"(expected {head_state['hidden_shape']}, got {tuple(hidden_states.shape)})."
                    )
                return head_state["logits"]

            talker.codec_head.forward = ov_codec_head
            self._ov_talker = compiled
            logger.info("Qwen3-TTS: talker decoder stack offloaded to OpenVINO (IR-backed).")
        except Exception as exc:  # pragma: no cover - fall back to pure PyTorch
            logger.warning(f"Qwen3-TTS: OpenVINO talker offload disabled ({exc}); using PyTorch.")

    def _install_ov_code_predictor(self) -> None:
        """Offload the code-predictor decoder stack (5 layers) to OpenVINO.

        The code predictor runs ``num_code_groups - 1`` times inside every talker step, so it
        is invoked far more often than the talker itself. Only its decoder stack moves to
        OpenVINO: the per-depth ``lm_head`` / codec embeddings and the sampling stay in
        PyTorch, which keeps the ``qwen_tts`` ``generate()`` semantics (multinomial sampling
        with the ``subtalker_*`` knobs) bit-for-bit intact. This is where the design differs
        from Qwen3-Omni, whose code predictor is a single-step stateful graph with in-graph
        Gumbel-max sampling driven by a Python loop.
        """
        try:
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorOutputWithPast

            code_predictor = self.model.talker.code_predictor
            code_predictor_model = code_predictor.model
            code_predictor_model.eval()

            compiled = self._compile_ov_component(_CODE_PREDICTOR_OV_IR_NAME, "code predictor")

            def plain_rope(position_ids, cache_position, batch_size, seq_len, inputs_embeds):
                # Plain 1D RoPE: cos/sin already come out with duplicated halves.
                if position_ids is None:
                    position_ids = cache_position.view(1, -1).expand(batch_size, -1)
                cos, sin = code_predictor_model.rotary_emb(inputs_embeds, position_ids)
                return cos.to(torch.float32), sin.to(torch.float32)

            ov_forward, head_state = _make_ov_decoder_stack_forward(code_predictor_model, compiled, plain_rope)
            code_predictor_model.forward = ov_forward

            def ov_code_predictor_forward(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                cache_position=None,
                generation_steps=None,
                **kwargs,
            ):
                # Mirrors Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward, with
                # the stack and the per-depth lm_head served by one graph call. The depth index
                # has to be known before that call, which is why the wrapper's forward is
                # replaced rather than only the inner stack's.
                if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
                    generation_steps = inputs_embeds.shape[1] - 2
                else:
                    inputs_embeds = code_predictor.model.get_input_embeddings()[generation_steps - 1](input_ids)
                inputs_embeds = code_predictor.small_to_mtp_projection(inputs_embeds)

                outputs = ov_forward(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    step=generation_steps,
                )
                return Qwen3TTSTalkerCodePredictorOutputWithPast(
                    loss=None,
                    logits=head_state["logits"],
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    generation_steps=generation_steps + 1,
                )

            code_predictor.forward = ov_code_predictor_forward
            self._ov_code_predictor = compiled
            logger.info("Qwen3-TTS: code predictor offloaded to OpenVINO (IR-backed).")
        except Exception as exc:  # pragma: no cover - fall back to pure PyTorch
            logger.warning(f"Qwen3-TTS: OpenVINO code predictor offload disabled ({exc}); using PyTorch.")

    def _install_ov_embeddings(self) -> None:
        """Point the model's embedding tables at their exported IRs.

        Three graphs cover them: the talker's text table, its first-codebook table, and the
        code predictor's per-depth tables stacked into one and selected by a ``step`` index.

        ``text_projection`` was baked into the rows of the text table at export time, so the
        module that would apply it becomes an identity - every call site in ``qwen_tts``
        applies it directly to the text embeddings, which now already carry it.
        """
        try:
            talker = self.model.talker
            talker_config = talker.model.config
            code_predictor = talker.code_predictor
            hidden_size = talker_config.hidden_size

            text_embeddings = self._compile_ov_component(_TEXT_EMBEDDINGS_OV_IR_NAME, "text embeddings")
            talker_embeddings = self._compile_ov_component(_TALKER_EMBEDDINGS_OV_IR_NAME, "talker embeddings")
            code_predictor_embeddings = self._compile_ov_component(
                _CODE_PREDICTOR_EMBEDDINGS_OV_IR_NAME, "code predictor embeddings"
            )

            talker.get_text_embeddings().forward = _make_ov_embedding_forward(text_embeddings, hidden_size)
            talker.get_input_embeddings().forward = _make_ov_embedding_forward(talker_embeddings, hidden_size)
            for step, embedding in enumerate(code_predictor.get_input_embeddings()):
                embedding.forward = _make_ov_embedding_forward(code_predictor_embeddings, hidden_size, step=step)

            # The projection is already applied to the exported text table.
            talker.text_projection.forward = lambda hidden_states: hidden_states

            self._ov_embeddings = (text_embeddings, talker_embeddings, code_predictor_embeddings)
            logger.info("Qwen3-TTS: embedding tables offloaded to OpenVINO (IR-backed).")
        except Exception as exc:  # pragma: no cover - fall back to pure PyTorch
            logger.warning(f"Qwen3-TTS: OpenVINO embeddings offload disabled ({exc}); using PyTorch.")

    def _install_ov_speaker_encoder(self) -> None:
        """Offload the ECAPA-TDNN speaker encoder (mel spectrogram -> x-vector) to OpenVINO."""
        try:
            speaker_encoder = self.model.speaker_encoder
            if speaker_encoder is None:
                raise ValueError("model has no speaker encoder (non-`base` variant)")
            speaker_encoder.eval()

            compiled = self._compile_ov_component(_SPEAKER_ENCODER_OV_IR_NAME, "speaker encoder")

            def ov_forward(hidden_states):
                mel_features = _as_float32_numpy(hidden_states)
                return torch.from_numpy(compiled(mel_features)[0]).clone()

            speaker_encoder.forward = ov_forward
            self._ov_speaker_encoder = compiled
            logger.info("Qwen3-TTS: speaker encoder offloaded to OpenVINO (IR-backed).")
        except Exception as exc:  # pragma: no cover - fall back to pure PyTorch
            logger.warning(f"Qwen3-TTS: OpenVINO speaker encoder offload disabled ({exc}); using PyTorch.")

    def _install_ov_codec_encoder(self) -> None:
        """Offload the codec encoder (reference waveform -> residual codes) to OpenVINO.

        The graph is length-agnostic only for waveforms that are a whole number of codec
        frames, so the waveform is zero-padded up to the next frame boundary here. That
        matches what the causal convolutions pad internally, and produces the same
        ``ceil(samples / 1920)`` frames the PyTorch path returns; the caller then trims the
        code stream back with its own padding mask.
        """
        try:
            from transformers.models.mimi.modeling_mimi import MimiEncoderOutput

            codec_model = self.model.speech_tokenizer.model
            codec_encoder = codec_model.encoder
            codec_encoder.eval()
            downsample_rate = int(codec_model.encode_downsample_rate)

            compiled = self._compile_ov_component(_CODEC_ENCODER_OV_IR_NAME, "codec encoder")

            def ov_encode(input_values, padding_mask=None, return_dict=True, **kw):
                waveform = _as_float32_numpy(input_values)
                remainder = waveform.shape[-1] % downsample_rate
                if remainder:
                    waveform = np.pad(waveform, [(0, 0)] * (waveform.ndim - 1) + [(0, downsample_rate - remainder)])
                audio_codes = torch.from_numpy(compiled(waveform)[0]).clone().to(torch.int64)
                if not return_dict:
                    return (audio_codes, None, None)
                return MimiEncoderOutput(audio_codes)

            codec_encoder.encode = ov_encode
            self._ov_codec_encoder = compiled
            logger.info("Qwen3-TTS: codec encoder offloaded to OpenVINO (IR-backed).")
        except Exception as exc:  # pragma: no cover - fall back to pure PyTorch
            logger.warning(f"Qwen3-TTS: OpenVINO codec encoder offload disabled ({exc}); using PyTorch.")

    def _install_ov_codec_decoder(self) -> None:
        """Offload the codec decoder (codes -> 24 kHz waveform) to OpenVINO.

        The decoder's forward is replaced rather than ``decode``, so the surrounding
        ``chunked_decode`` windowing in ``qwen_tts`` keeps driving it unchanged.
        """
        try:
            codec_decoder = self.model.speech_tokenizer.model.decoder
            codec_decoder.eval()

            compiled = self._compile_ov_component(_CODEC_DECODER_OV_IR_NAME, "codec decoder")

            def ov_forward(codes):
                audio_codes = codes if isinstance(codes, np.ndarray) else codes.detach().cpu().numpy()
                return torch.from_numpy(compiled(audio_codes.astype(np.int64))[0]).clone()

            codec_decoder.forward = ov_forward
            self._ov_codec_decoder = compiled
            logger.info("Qwen3-TTS: codec decoder offloaded to OpenVINO (IR-backed).")
        except Exception as exc:  # pragma: no cover - fall back to pure PyTorch
            logger.warning(f"Qwen3-TTS: OpenVINO codec decoder offload disabled ({exc}); using PyTorch.")

    @property
    def ov_models(self) -> Dict[str, openvino.Model]:
        """The exported graphs, keyed by component name.

        Reads them back from disk rather than holding them alongside the compiled models,
        since an ``ov.Model`` keeps its own copy of the weights and the compiled ones are what
        inference runs on. Used for inspection - e.g. checking the compression state of each
        component - not on any hot path.
        """
        core = openvino.Core()
        models = {}
        for label, ir_path in self._ov_ir_paths.items():
            name = Path(ir_path).stem[len("openvino_") :]
            models[name] = core.read_model(ir_path)
        return models

    def _apply_quantization(
        self,
        quantization_config,
        save_directory=None,
        **kwargs,
    ) -> None:
        """Weight-compress every exported IR in place.

        Called by ``optimum-cli export openvino --weight-format int8`` (and the other
        weight-only formats) after the floating-point export has been written. Only the
        components in :data:`_QWEN3_TTS_COMPRESSIBLE_OV_IR_NAMES` are compressed - see the note
        there on why the codec is left in floating point - and each is compressed on its own,
        so this works whatever subset of components a given Qwen3-TTS variant produced.

        A 4-bit request is applied per component rather than across the board: only the IRs in
        :data:`_QWEN3_TTS_INT4_OV_IR_NAMES` are quantized to 4 bits, the rest fall back to 8,
        so `--weight-format int4` yields a mixed int4/int8 model.
        """
        compress_qwen3_tts_irs(
            self._ir_dir,
            quantization_config,
            output_dir=save_directory if save_directory is not None else self._ir_dir,
        )

    def preprocess_input(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "Auto",
        ref_audio: Optional[Any] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build the inputs for :meth:`generate` from raw text and reference audio.

        For the ``base`` (voice-clone) model this performs the reference-audio
        encoding and speaker-embedding extraction (ICL mode when ``ref_text`` is
        provided), mirroring ``qwen_tts.Qwen3TTSModel.create_voice_clone_prompt``.

        Returns a dictionary that can be unpacked directly into :meth:`generate`.
        """
        if ref_audio is None:
            raise ValueError("`ref_audio` must be provided for Qwen3-TTS voice cloning.")

        voice_clone_prompt = self._pipeline.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        inputs: Dict[str, Any] = {
            "text": text,
            "language": language,
            "voice_clone_prompt": voice_clone_prompt,
        }
        inputs.update(kwargs)
        return inputs

    @torch.no_grad()
    def generate(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "Auto",
        voice_clone_prompt: Optional[Any] = None,
        ref_audio: Optional[Any] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        return_sample_rate: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, "tuple[torch.Tensor, int]"]:
        """Generate a speech waveform.

        The talker/code-predictor generation and the codec decoding are driven by the
        original ``qwen_tts`` orchestration (optionally OpenVINO-accelerated).

        Returns a single waveform tensor (batch size 1) or a list of tensors for
        batched inputs.
        """
        wavs, sr = self._pipeline.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_clone_prompt=voice_clone_prompt,
            **kwargs,
        )
        self.sampling_rate = int(sr)

        waveforms = [torch.from_numpy(np.ascontiguousarray(w)) for w in wavs]
        output: Union[torch.Tensor, List[torch.Tensor]]
        output = waveforms[0] if len(waveforms) == 1 else waveforms

        if return_sample_rate:
            return output, int(sr)
        return output
