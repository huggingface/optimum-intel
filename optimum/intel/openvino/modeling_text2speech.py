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

import logging
import os
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
            kwargs.pop("export", None)
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
# original PyTorch orchestration from ``qwen_tts`` and offloads the compute-heavy talker
# decoder stack to OpenVINO (hybrid component-wise export).
# ---------------------------------------------------------------------------

# File name of the serialized talker decoder stack OpenVINO IR.
_TALKER_OV_IR_NAME = "openvino_talker_model.xml"


def _resolve_talker_ir_dir(model_id, cache_dir) -> Path:
    """Resolve a writable directory for the talker OpenVINO IR.

    Uses the model directory when ``model_id`` is a local path, otherwise a stable
    location under the Hugging Face cache keyed by the (sanitized) model id.
    """
    path = Path(str(model_id))
    if path.is_dir():
        return path
    sanitized = str(model_id).replace("/", "--")
    base = Path(cache_dir) if cache_dir else Path(HUGGINGFACE_HUB_CACHE)
    return base / "openvino_qwen3_tts" / sanitized


def _is_qwen3_tts_config(config: Optional["PretrainedConfig"]) -> bool:
    """Return True when the given config describes a Qwen3-TTS model."""
    if config is None:
        return False
    if getattr(config, "model_type", None) == "qwen3_tts":
        return True
    architectures = getattr(config, "architectures", None) or []
    return "Qwen3TTSForConditionalGeneration" in architectures


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
        **kwargs,
    ) -> "_OVModelForQwen3TTS":
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise ImportError(
                "Qwen3-TTS requires the `qwen_tts` package to be installed. "
                "Install it with: pip install qwen-tts"
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

        pipeline = Qwen3TTSModel.from_pretrained(str(model_id), **load_kwargs)
        pipeline.model.eval()

        if config is None:
            config = pipeline.model.config

        instance = cls(pipeline=pipeline, config=config, model_save_dir=model_id)
        instance._ir_dir = _resolve_talker_ir_dir(model_id, cache_dir)
        instance._install_ov_talker()
        return instance

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        # OpenVINO components run on CPU; ignore device moves to stay API compatible.
        return self

    def can_generate(self) -> bool:
        return True

    def _install_ov_talker(self) -> None:
        """Offload the talker decoder stack (28 layers, run every frame) to OpenVINO.

        The talker stack is loaded from a standalone OpenVINO IR
        (``openvino_talker_model.xml`` / ``.bin``) on disk and used for inference. The
        original ``talker.model.forward`` is replaced by an OpenVINO-backed implementation
        that preserves the exact I/O contract (``DynamicCache`` in/out,
        ``BaseModelOutputWithPast``). All other orchestration stays in PyTorch. When the
        IR is not available (or anything else fails) the model transparently falls back
        to the original PyTorch path.
        """
        try:
            from transformers import DynamicCache
            from transformers.modeling_outputs import BaseModelOutputWithPast

            # Reuse the model's own multimodal-RoPE implementation from ``qwen_tts``
            # instead of duplicating it here.
            from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

            talker_model = self.model.talker.model
            talker_model.eval()
            cfg = talker_model.config
            mrope_section = cfg.rope_scaling["mrope_section"]
            mrope_interleaved = cfg.rope_scaling.get("interleaved", False)
            num_layers = len(talker_model.layers)
            num_kv = cfg.num_key_value_heads
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

            ir_dir = Path(getattr(self, "_ir_dir", None) or _resolve_talker_ir_dir(self.model_save_dir, None))
            ir_xml = ir_dir / _TALKER_OV_IR_NAME
            if not ir_xml.is_file():
                raise FileNotFoundError(f"talker OpenVINO IR not found at {ir_xml}")
            core = openvino.Core()
            ov_model = core.read_model(ir_xml)
            logger.info(f"Qwen3-TTS: loading talker OpenVINO IR from {ir_xml}.")
            compiled = core.compile_model(ov_model, "CPU")
            self._ov_talker_ir_path = str(ir_xml)

            neg = torch.finfo(torch.float32).min

            # Per-call KV store (mirrors the HF cache, but version independent).
            state: Dict[str, Any] = {"k": None, "v": None}

            def _build_mask(bs_, seq_, past_len, attention_mask):
                total = past_len + seq_
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
                **kw,
            ):
                if past_key_values is None:
                    past_key_values = DynamicCache()
                inputs_embeds = inputs_embeds.to(torch.float32)
                bs_, seq_ = inputs_embeds.shape[0], inputs_embeds.shape[1]
                past_len = past_key_values.get_seq_length()

                if cache_position is None:
                    cache_position = torch.arange(past_len, past_len + seq_)
                if position_ids is None:
                    position_ids = cache_position.view(1, 1, -1).expand(3, bs_, -1)
                elif position_ids.ndim == 2:
                    position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                    position_ids = position_ids[1:]

                cos, sin = talker_model.rotary_emb(inputs_embeds, position_ids)
                # Recover the merged rotary cos/sin that the talker IR expects by reusing
                # ``apply_multimodal_rotary_pos_emb`` on a basis probe: with the first half
                # of the probe set to 1 and the second half to 0, the rotated output's
                # first half equals the merged cos and its second half equals the merged
                # sin (cos/sin both have duplicated halves), so no merge math is duplicated.
                half = head_dim // 2
                probe = torch.cat(
                    [
                        torch.ones(bs_, 1, seq_, half, dtype=torch.float32),
                        torch.zeros(bs_, 1, seq_, half, dtype=torch.float32),
                    ],
                    dim=-1,
                )
                merged, _ = apply_multimodal_rotary_pos_emb(
                    probe, probe, cos.to(torch.float32), sin.to(torch.float32), mrope_section, mrope_interleaved
                )
                merged = merged.squeeze(1)
                cos_m = torch.cat([merged[..., :half], merged[..., :half]], dim=-1)
                sin_m = torch.cat([merged[..., half:], merged[..., half:]], dim=-1)
                mask = _build_mask(bs_, seq_, past_len, attention_mask)

                if past_len == 0 or state["k"] is None:
                    past_k = torch.zeros(num_layers, bs_, num_kv, 0, head_dim, dtype=torch.float32)
                    past_v = torch.zeros(num_layers, bs_, num_kv, 0, head_dim, dtype=torch.float32)
                else:
                    past_k = state["k"]
                    past_v = state["v"]

                outputs = compiled(
                    [
                        inputs_embeds.numpy(),
                        mask.numpy(),
                        cos_m.numpy(),
                        sin_m.numpy(),
                        past_k.numpy(),
                        past_v.numpy(),
                    ]
                )
                hidden = torch.from_numpy(outputs[0])
                new_k = torch.from_numpy(outputs[1])
                new_v = torch.from_numpy(outputs[2])

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

            talker_model.forward = ov_forward
            self._ov_talker = compiled
            logger.info("Qwen3-TTS: talker decoder stack offloaded to OpenVINO (IR-backed).")
        except Exception as exc:  # pragma: no cover - fall back to pure PyTorch
            logger.warning(f"Qwen3-TTS: OpenVINO talker offload disabled ({exc}); using PyTorch.")

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
