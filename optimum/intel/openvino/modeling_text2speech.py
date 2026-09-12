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


class _OVQwen3TTSPart(OVModelPart):
    """Base for the Qwen3-TTS parts.

    Each wraps one exported graph and is installed over the forward of the PyTorch module it
    replaces, so ``qwen_tts`` keeps orchestrating generation and calls OpenVINO without knowing it.
    """

    @staticmethod
    def _as_float32(tensor) -> np.ndarray:
        """A contiguous float32 numpy view of a torch tensor or array."""
        if isinstance(tensor, np.ndarray):
            return np.ascontiguousarray(tensor, dtype=np.float32)
        return np.ascontiguousarray(tensor.detach().cpu().to(torch.float32).numpy())


class OVQwen3TTSDecoderStack(_OVQwen3TTSPart):
    """One of Qwen3-TTS's two decoder stacks - the talker, or its code predictor.

    The graph keeps its key/value cache in OpenVINO state, so nothing is passed in or read back
    here, and it builds its own rotary embeddings from ``position_ids``. What this has to get
    right is *when to reset*: the code predictor's cache lives for a single talker frame, covering
    the ``num_code_groups - 1`` inner steps that fill one frame's residual codes, and a new frame
    must start from an empty cache. The reset is therefore tied to the prefill call of each frame
    - the one where the HF cache is still empty - rather than to a step counter.

    The returned ``BaseModelOutputWithPast`` still carries a cache object, because the surrounding
    ``generate`` uses its length to derive ``cache_position``. It is fed one-element-wide dummy
    tensors: the real keys and values live in the graph, and only the sequence length is read.

    ``head_state`` holds the logits the graph produced alongside the hidden states, for the output
    head that ``qwen_tts`` applies next - folded into this graph at export time, so it must be
    handed back rather than recomputed.
    """

    def __init__(
        self,
        model: openvino.Model,
        parent_model: OVBaseModel,
        model_name: str,
        num_layers: int,
        num_key_value_heads: int,
        ov_config: Optional[Dict[str, str]] = None,
        position_fn: Optional[Any] = None,
        with_step: bool = False,
    ) -> None:
        super().__init__(model, parent_model, ov_config=ov_config, model_name=model_name)
        self._num_layers = num_layers
        self._num_key_value_heads = num_key_value_heads
        self._position_fn = position_fn
        self._with_step = with_step
        # A graph produced by the standard stateful transformation also takes ``beam_idx`` and
        # gathers its cache through it, which is detected rather than assumed, so both that form
        # and a plain stateful graph can be driven by this one part.
        self._has_beam_idx = "beam_idx" in self.input_names
        self.head_state: Dict[str, Any] = {"logits": None, "hidden_shape": None}

    def compile(self):
        # The cache lives in OpenVINO state, which is per request rather than per compiled model,
        # and the graph is driven asynchronously - so this part holds a request, as ``OVDecoder``
        # does for the same reason.
        super().compile()
        if isinstance(self.request, openvino.CompiledModel):
            self.request = self.request.create_infer_request()

    def reset_state(self) -> None:
        if self.request is not None:
            self.request.reset_state()

    @staticmethod
    def mrope_positions(position_ids, cache_position, batch_size):
        """The talker's ``position_fn``: normalize positions to the three m-RoPE streams.

        Its graph takes ``[3, batch, sequence]``, one row per stream, and merges the sections
        internally; ``qwen_tts`` hands the positions over in several shapes.
        """
        if position_ids is None:
            return cache_position.view(1, 1, -1).expand(3, batch_size, -1)
        if position_ids.ndim == 2:
            return position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            # `qwen_tts` sometimes carries a leading text row ahead of the three.
            return position_ids[1:]
        return position_ids

    def _positions(self, position_ids, cache_position, batch_size):
        if self._position_fn is not None:
            return self._position_fn(position_ids, cache_position, batch_size)
        if position_ids is None:
            position_ids = cache_position.view(1, -1).expand(batch_size, -1)
        return position_ids

    def forward(
        self,
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
        **kwargs,
    ):
        from transformers import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        self.compile()
        if past_key_values is None:
            past_key_values = DynamicCache()
        inputs_embeds = inputs_embeds.to(torch.float32)
        batch_size, sequence_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        past_length = past_key_values.get_seq_length()

        if past_length == 0:
            # Start of a frame: drop the previous frame's residual codes from the state.
            self.request.reset_state()

        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + sequence_length)
        position_ids = self._positions(position_ids, cache_position, batch_size)

        total = past_length + sequence_length
        rows = torch.arange(sequence_length).view(sequence_length, 1)
        cols = torch.arange(total).view(1, total)
        neg = torch.finfo(torch.float32).min
        mask = torch.zeros(sequence_length, total, dtype=torch.float32).masked_fill(~(cols <= past_length + rows), neg)
        mask = mask.view(1, 1, sequence_length, total).expand(batch_size, 1, sequence_length, total).clone()
        if attention_mask is not None and attention_mask.ndim == 2:
            mask = mask.masked_fill((attention_mask[:, :total] == 0).view(batch_size, 1, 1, total), neg)

        graph_inputs = {
            "inputs_embeds": inputs_embeds.numpy(),
            "attention_mask": mask.numpy(),
            "position_ids": position_ids.to(torch.int64).numpy(),
        }
        if self._with_step:
            graph_inputs["step"] = np.array(step if step is not None else 0, dtype=np.int64)
        if self._has_beam_idx:
            # The graph gathers every cache read through `beam_idx`, so it always has to be fed;
            # the talker samples one continuation per sequence, so the batch keeps its order and
            # identity indices are what a reorder would produce.
            graph_inputs["beam_idx"] = np.arange(batch_size, dtype=np.int32)
        self.request.start_async(graph_inputs, share_inputs=True)
        self.request.wait()

        hidden = torch.from_numpy(self.request.get_tensor("last_hidden_state").data).clone()
        self.head_state["logits"] = torch.from_numpy(self.request.get_tensor("logits").data).clone()
        self.head_state["hidden_shape"] = tuple(hidden.shape)

        # Advance the HF cache length only; the tensors themselves are never read back.
        marker = torch.zeros(batch_size, self._num_key_value_heads, sequence_length, 1)
        for idx in range(self._num_layers):
            past_key_values.update(marker, marker, idx)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=past_key_values,
            hidden_states=(hidden,) if output_hidden_states else None,
            attentions=None,
        )

    def head_logits(self, hidden_states) -> torch.Tensor:
        """The logits the graph computed for ``hidden_states`` on the call that produced them."""
        if self.head_state["hidden_shape"] != tuple(hidden_states.shape):
            raise RuntimeError(
                f"Qwen3-TTS: the {self._model_name} output head was called on hidden states the graph did not "
                f"just produce (expected {self.head_state['hidden_shape']}, got {tuple(hidden_states.shape)})."
            )
        return self.head_state["logits"]


class OVQwen3TTSEmbedding(_OVQwen3TTSPart):
    """One exported embedding table.

    ``qwen_tts`` looks embeddings up with 0-d, 1-d and 2-d id tensors while the graph has a fixed
    rank, so ids are flattened to ``[1, N]`` on the way in and the original shape is restored on
    the way out. The code predictor's per-depth tables share one graph and select a depth with
    ``step``, which :meth:`forward_for_step` binds.
    """

    def __init__(
        self,
        model: openvino.Model,
        parent_model: OVBaseModel,
        model_name: str,
        embedding_dim: int,
        ov_config: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(model, parent_model, ov_config=ov_config, model_name=model_name)
        self._embedding_dim = embedding_dim

    def forward(self, input_ids, step: Optional[int] = None):
        self.compile()
        ids = input_ids.reshape(1, -1).to(torch.int64).numpy()
        extra_inputs = [] if step is None else [np.array(step, dtype=np.int64)]
        embeddings = torch.from_numpy(self.request([ids, *extra_inputs])[0]).clone()
        return embeddings.reshape(*input_ids.shape, self._embedding_dim)

    def forward_for_step(self, step: int):
        """A ``forward`` bound to one depth of the stacked per-depth tables."""
        return lambda input_ids: self.forward(input_ids, step=step)


class OVQwen3TTSSpeakerEncoder(_OVQwen3TTSPart):
    """The ECAPA-TDNN speaker encoder: mel spectrogram -> x-vector, once per reference audio."""

    def forward(self, hidden_states):
        self.compile()
        return torch.from_numpy(self.request(self._as_float32(hidden_states))[0]).clone()


class OVQwen3TTSCodecEncoder(_OVQwen3TTSPart):
    """The codec encoder: reference waveform -> the residual codes that seed voice cloning.

    The waveform is handed over as it comes. The exported convolutions derive their own right
    padding from the traced shape (see ``_traceable_extra_padding_for_conv1d`` in the exporter),
    so the graph returns the same ``ceil(samples / 1920)`` frames as PyTorch for any length, with
    the same values; the caller then trims the code stream back with its own padding mask.
    Padding the waveform up to a frame boundary here instead would change the last frame's codes,
    because the stock convs pad per layer rather than once at the input.
    """

    def encode(self, input_values, padding_mask=None, return_dict=True, **kwargs):
        from transformers.models.mimi.modeling_mimi import MimiEncoderOutput

        self.compile()
        audio_codes = torch.from_numpy(self.request(self._as_float32(input_values))[0]).clone().to(torch.int64)
        if not return_dict:
            return (audio_codes, None, None)
        return MimiEncoderOutput(audio_codes)


class OVQwen3TTSCodecDecoder(_OVQwen3TTSPart):
    """The codec decoder: generated codes -> 24 kHz waveform.

    Replaces the decoder's ``forward`` rather than ``decode``, so the surrounding
    ``chunked_decode`` windowing in ``qwen_tts`` keeps driving it unchanged.
    """

    def forward(self, codes):
        self.compile()
        audio_codes = codes if isinstance(codes, np.ndarray) else codes.detach().cpu().numpy()
        return torch.from_numpy(self.request(audio_codes.astype(np.int64))[0]).clone()


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
        if _OVModelForQwen3TTS.is_qwen3_tts_config(kwargs.get("config")):
            # ``export`` and ``compile`` are both honoured by the dedicated runtime: it converts
            # the checkpoint first, and defers installing its components when ``compile=False``.
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


class _OVModelForQwen3TTS(OVModelForTextToSpeechSeq2Seq):
    """OpenVINO-backed runtime for Qwen3-TTS.

    Eight exported graphs cover the whole network, each held by an :class:`OVModelPart` in
    :attr:`components`. The generation itself - sampling, m-RoPE index math, ICL prompt assembly,
    chunked decoding - stays in ``qwen_tts``, so the parts are installed over the forwards of the
    PyTorch modules they replace rather than being called from a ``generate`` written here. That
    is the one structural difference from :class:`_OVModelForSpeechT5ForTextToSpeech`, whose four
    parts are driven by its own generation loop.

    A component whose IR is absent stays on PyTorch, so a partial export still runs; where the
    export ships no weights at all, the missing IRs are an error instead.
    """

    export_feature = "text-to-audio"
    main_input_name = "input_ids"
    _supports_cache_class = False

    @classproperty
    def _all_ov_model_paths(cls) -> Dict[str, str]:
        return {
            "talker_model": "openvino_talker_model.xml",
            "code_predictor_model": "openvino_code_predictor_model.xml",
            "text_embeddings": "openvino_text_embeddings.xml",
            "talker_embeddings": "openvino_talker_embeddings.xml",
            "code_predictor_embeddings": "openvino_code_predictor_embeddings.xml",
            "speaker_encoder": "openvino_speaker_encoder.xml",
            "codec_encoder": "openvino_codec_encoder.xml",
            "codec_decoder": "openvino_codec_decoder.xml",
        }

    # Both decoder stacks keep their key/value cache in OpenVINO state, which makes the CPU
    # plugin apply its default `u8` cache compression. That trade is made for long-context LLMs;
    # here the cache spans a couple of hundred 12.5 Hz frames at most, so it saves little, while
    # quantizing the keys is enough to move sampled codes away from what PyTorch produces.
    _TALKER_OV_CONFIG = {"KV_CACHE_PRECISION": "f32"}

    # The code predictor is additionally pinned to f32 arithmetic. It runs `num_code_groups - 1`
    # steps inside every talker frame off a cache that is reset each frame, and in f16 - the GPU
    # plugin's default inference precision - its logits go non-finite within the first few frames,
    # which surfaces as `probability tensor contains either inf, nan or element < 0` out of the
    # multinomial sampling in `code_predictor.generate`. The talker stack is unaffected and keeps
    # the device default, so the bulk of the compute (28 layers vs 5) still runs in f16 on GPU.
    _CODE_PREDICTOR_OV_CONFIG = {**_TALKER_OV_CONFIG, "INFERENCE_PRECISION_HINT": "f32"}

    # The ports :class:`OVQwen3TTSDecoderStack` addresses by name, checked when a stack is loaded
    # so that an IR from an older exporter is rejected up front rather than at the first generated
    # frame. ``beam_idx`` is deliberately absent: the part detects it and drives both the plain
    # stateful graph and the beam-reordered one.
    _DECODER_STACK_INPUTS = ("inputs_embeds", "attention_mask", "position_ids")
    _DECODER_STACK_OUTPUTS = ("last_hidden_state", "logits")
    _DECODER_STACK_PORTS = {
        "talker_model": (_DECODER_STACK_INPUTS, _DECODER_STACK_OUTPUTS),
        # The code predictor's graph additionally picks a depth with ``step``.
        "code_predictor_model": (_DECODER_STACK_INPUTS + ("step",), _DECODER_STACK_OUTPUTS),
    }

    # The components every export carries, whatever the variant, and so the ones a directory has
    # to hold before a previous conversion may be reused instead of repeated. The speaker encoder
    # is left out because only the voice-clone (``base``) checkpoints have one.
    _MANDATORY_COMPONENTS = (
        "talker_model",
        "code_predictor_model",
        "text_embeddings",
        "talker_embeddings",
        "code_predictor_embeddings",
        "codec_encoder",
        "codec_decoder",
    )

    # Weight-only compression is applied to the language-model side of the pipeline and kept away
    # from the neural codec, the same split diffusion pipelines use to leave the VAE alone. The
    # codec is a waveform autoencoder rather than a classifier over a large vocabulary: measured
    # on this model, int8 weights cost the vocoder ~29 dB of SNR (47 dB -> 18 dB) and drop the
    # encoder's exact code agreement from 96% to 40%, while the decoder stacks and the embedding
    # tables tolerate it. The speaker encoder is excluded too - it is 9M parameters, so
    # compressing it saves nothing worth the risk to voice similarity.
    _COMPRESSIBLE_COMPONENTS = (
        "talker_model",
        "code_predictor_model",
        "text_embeddings",
        "talker_embeddings",
        "code_predictor_embeddings",
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
    _INT4_COMPONENTS = ("talker_model",)

    @staticmethod
    def is_qwen3_tts_config(config: Optional["PretrainedConfig"]) -> bool:
        """Whether ``config`` describes a Qwen3-TTS model, and so this runtime handles it."""
        if config is None:
            return False
        if getattr(config, "model_type", None) == "qwen3_tts":
            return True
        architectures = getattr(config, "architectures", None) or []
        return "Qwen3TTSForConditionalGeneration" in architectures

    @staticmethod
    def _resolve_ir_dir(model_id, cache_dir) -> Path:
        """Resolve a writable directory for the OpenVINO IRs.

        Uses the model directory when ``model_id`` is a local path, otherwise a stable location
        under the Hugging Face cache keyed by the (sanitized) model id.
        """
        path = Path(str(model_id))
        if path.is_dir():
            return path
        sanitized = str(model_id).replace("/", "--")
        base = Path(cache_dir) if cache_dir else Path(HUGGINGFACE_HUB_CACHE)
        return base / "openvino_qwen3_tts" / sanitized

    @staticmethod
    def _supported_ov_config(device: str, ov_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Drop properties ``device`` does not advertise.

        The compile hints used here are per-device (``KV_CACHE_PRECISION`` is not offered
        everywhere), and passing one a device does not know is an error - which would cost the
        acceleration entirely, for a hint.
        """
        if not ov_config:
            return {}
        try:
            supported = set(openvino.Core().get_property(device, "SUPPORTED_PROPERTIES"))
        except Exception:
            return {}
        dropped = [name for name in ov_config if name not in supported]
        if dropped:
            logger.info(f"Qwen3-TTS: {device} does not support {dropped}; compiling without.")
        return {name: value for name, value in ov_config.items() if name in supported}

    @staticmethod
    def _check_ir_signature(model: openvino.Model, ir_xml, label: str, expected_inputs, expected_outputs) -> None:
        """Reject an IR whose graph is not the one this runtime drives.

        The IR directory is resolved by model id (see :meth:`_resolve_ir_dir`), so it can hold an
        export written by an older version of the exporter - an early talker, say, taking
        ``cos``/``sin``/``past_k``/``past_v`` instead of ``position_ids`` and OpenVINO state. Such
        an IR reads and compiles perfectly well, and would otherwise fail on the first generated
        frame, deep inside ``start_async``, with a bare
        ``Port for tensor name position_ids was not found``.
        """
        available_inputs = {name for port in model.inputs for name in port.get_names()}
        available_outputs = {name for port in model.outputs for name in port.get_names()}
        missing = [name for name in expected_inputs if name not in available_inputs]
        missing += [name for name in expected_outputs if name not in available_outputs]
        if missing:
            # A stale graph can carry a port per layer, so the names it does have are only sampled
            # here - enough to recognize the vintage, not enough to bury the message.
            named = sorted(name for name in available_inputs if not name.isdigit())
            sample = ", ".join(named[:6]) + (f", ... ({len(available_inputs)} inputs in total)" if named else "")
            raise RuntimeError(
                f"the {label} OpenVINO IR at {ir_xml} is not the graph this runtime drives: it has no "
                f"{', '.join(missing)} (its named inputs are {sample}). The IR predates the current "
                "exporter; re-export the model with `optimum-cli export openvino`."
            )

    @staticmethod
    def _has_codec_weights(model_id) -> bool:
        """True when the model directory still ships the codec checkpoint."""
        codec_dir = Path(str(model_id)) / "speech_tokenizer"
        if not codec_dir.is_dir():
            # Hub repos are resolved by ``qwen_tts`` itself; assume the original layout.
            return True
        return any(codec_dir.glob("*.safetensors")) or any(codec_dir.glob("*.bin"))

    @staticmethod
    def _has_checkpoint_weights(model_id) -> bool:
        """True when the model directory still ships the main Qwen3-TTS checkpoint."""
        model_dir = Path(str(model_id))
        if not model_dir.is_dir():
            return True
        return any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))

    @staticmethod
    def _weight_compression_config(load_in_8bit, quantization_config):
        """Resolve the weight-compression request, mirroring the other OpenVINO model classes."""
        from .configuration import OVWeightQuantizationConfig

        if quantization_config is not None:
            if isinstance(quantization_config, dict):
                return OVWeightQuantizationConfig.from_dict(quantization_config)
            return quantization_config
        if load_in_8bit:
            return OVWeightQuantizationConfig(bits=8)
        return None

    @staticmethod
    @contextmanager
    def _weightless_codec(model_id):
        """Let ``qwen_tts`` build the neural codec from its config when its weights are absent.

        Every codec parameter is baked into the exported ``codec_encoder`` / ``codec_decoder``
        IRs, so the export does not copy ``speech_tokenizer/*.safetensors``. ``Qwen3TTSTokenizer``
        would still insist on a checkpoint, so within this context it is built structurally
        instead: the modules are materialized on the meta device (no allocation - nothing ever
        reads their weights, both entry points being replaced by :class:`OVQwen3TTSCodecEncoder`
        and :class:`OVQwen3TTSCodecDecoder`) while the feature extractor and config, which the
        surrounding Python code does read, load normally.

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
            # ``device`` is a plain attribute on the wrapper, so it can point at CPU even though
            # the (unused) module parameters live on meta; the codec's tensors come from OpenVINO.
            instance.device = torch.device("cpu")
            return instance

        Qwen3TTSTokenizer.from_pretrained = classmethod(from_config)
        try:
            yield
        finally:
            Qwen3TTSTokenizer.from_pretrained = original_from_pretrained

    @staticmethod
    def _strip_weights(model: "torch.nn.Module") -> None:
        """Turn a meta-device Qwen3-TTS module tree into a structural, weightless one.

        Every parameter is swapped for an empty CPU tensor. That keeps ``model.device`` and
        ``model.dtype`` - which ``qwen_tts`` reads when it builds its prompt tensors - reporting
        CPU/float32 without allocating the checkpoint, and makes any weight that is unexpectedly
        still needed fail loudly on shape rather than silently return garbage.

        Rotary ``inv_freq`` buffers are the exception: they are non-persistent, derived from the
        config at construction time, and read by the PyTorch side of the decoder stacks, so they
        are recomputed on CPU.
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

    @classmethod
    def _is_complete_export(cls, directory) -> bool:
        """Whether ``directory`` holds an export that can be loaded as one.

        Guards the two places :meth:`_convert_checkpoint` skips a conversion because one appears
        to be there already. A directory holding only some of the graphs must not qualify: it
        would be returned as the export, and - having no checkpoint in it - then be read as a
        weightless one, where the structural build fails on a config that is not there
        (``'Qwen3TTSTalkerConfig' object has no attribute 'text_vocab_size'``). Interrupted
        conversions and directories written by a version that exported only the talker both land
        in that state, and both are indistinguishable from a finished export by one IR alone.

        A full set of files is not enough either: the cache entry is keyed by model id alone, so it
        outlives exporter changes, and a set written before the decoder stacks moved their cache
        into OpenVINO state loads fine and is then rejected component by component. The stacks'
        ports are checked so that such an entry is converted again rather than reused.
        """
        directory = Path(directory)
        if not (directory / "config.json").is_file():
            return False
        if not all((directory / cls._all_ov_model_paths[name]).is_file() for name in cls._MANDATORY_COMPONENTS):
            return False
        core = openvino.Core()
        for name, ports in cls._DECODER_STACK_PORTS.items():
            ir_xml = directory / cls._all_ov_model_paths[name]
            model = core.read_model(ir_xml)
            try:
                cls._check_ir_signature(model, ir_xml, name, *ports)
            except RuntimeError:
                return False
            finally:
                # A reused-or-not verdict must not leave the .bin mapped: a re-conversion writes
                # over it next.
                del model
        return True

    @classmethod
    def _convert_checkpoint(cls, model_id, cache_dir, weight_compression=None):
        """Convert a checkpoint to OpenVINO and return the directory holding the IRs.

        Backs ``from_pretrained(..., export=True)``. The output goes to the directory
        :meth:`_resolve_ir_dir` resolves for this model, so a second load finds the IRs already
        there instead of converting again.
        """
        from optimum.exporters.openvino import main_export

        source = Path(str(model_id))
        if source.is_dir() and cls._is_complete_export(source):
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
        if cls._is_complete_export(output_dir):
            logger.info(f"Qwen3-TTS: reusing the OpenVINO export at {output_dir}.")
            return output_dir
        if output_dir.is_dir() and any(output_dir.iterdir()):
            logger.info(f"Qwen3-TTS: the export at {output_dir} is incomplete; converting again.")

        logger.info(f"Qwen3-TTS: exporting {model_id} to OpenVINO in {output_dir}.")
        output_dir.mkdir(parents=True, exist_ok=True)
        main_export(
            model_name_or_path=str(model_id),
            output=output_dir,
            task="text-to-audio",
            cache_dir=cache_dir,
        )
        if weight_compression is not None:
            cls.compress_irs(output_dir, weight_compression)
        return output_dir

    @classmethod
    def _build_weightless_pipeline(cls, model_id, generate_config_name: str = "generation_config.json"):
        """Build the ``qwen_tts`` pipeline for an export that carries no PyTorch weights.

        Mirrors ``Qwen3TTSModel.from_pretrained`` /
        ``Qwen3TTSForConditionalGeneration.from_pretrained`` but constructs the model from its
        config on the meta device instead of loading a checkpoint, since every parameter has been
        exported to an OpenVINO IR. The processor, configs and generation defaults - all of which
        the surrounding Python code really does read - are loaded normally.
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
        cls._strip_weights(model)
        model.eval()

        with cls._weightless_codec(model_dir):
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

    @classmethod
    def compress_irs(cls, ir_dir, quantization_config, output_dir=None) -> None:
        """Weight-compress the exported IRs in ``ir_dir``.

        Shared by ``optimum-cli export openvino --weight-format ...`` (through
        :meth:`_apply_quantization`) and by :meth:`_convert_checkpoint` when a compression config
        is passed to ``from_pretrained``, so both entry points produce the same model.
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

        for component in cls._COMPRESSIBLE_COMPONENTS:
            ir_name = cls._all_ov_model_paths[component]
            ir_path = ir_dir / ir_name
            if not ir_path.is_file():
                continue
            config = quantization_config
            if fallback_config is not None and component not in cls._INT4_COMPONENTS:
                config = fallback_config
            bits = config.get("bits") if isinstance(config, dict) else config.bits
            logger.info(f"Qwen3-TTS: applying {bits}-bit weight compression to {ir_name}.")
            compressed = _weight_only_quantization(core.read_model(ir_path), config)

            # The source weights may still be mapped - by ``read_model`` above, and by whoever
            # called this - so the compressed graph is written under a temporary name and renamed
            # into place. Writing over a mapped .bin would truncate it underneath its mappings and
            # take the process down with SIGBUS.
            staged_xml = output_dir / f"{ir_path.stem}.compressed.xml"
            save_model(compressed, staged_xml, compress_to_fp16=False)
            del compressed
            gc.collect()

            target_xml = output_dir / ir_name
            os.replace(staged_xml, target_xml)
            os.replace(staged_xml.with_suffix(".bin"), target_xml.with_suffix(".bin"))

    def __init__(
        self,
        pipeline,
        config: "PretrainedConfig",
        ov_models: Optional[Dict[str, openvino.Model]] = None,
        device: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        # ``pipeline`` is a ``qwen_tts.Qwen3TTSModel`` wrapper instance; ``self.model`` is the
        # torch module tree it orchestrates, which the parts are installed into.
        self._pipeline = pipeline
        self.model = pipeline.model
        self.processor = pipeline.processor
        self.config = config
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = True
        self.use_cache = True
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])
        self._compile_only = kwargs.get("compile_only", False)
        self.generation_config = kwargs.get("generation_config", None) or GenerationConfig.from_model_config(config)
        self._openvino_config = OVConfig(quantization_config=quantization_config) if quantization_config else None
        self._set_ov_config_parameters()

        # Set by ``from_pretrained``; an export drops both checkpoints.
        self._ir_dir = Path(kwargs.get("ir_dir", None) or model_save_dir or ".")
        self._weights_present = kwargs.get("weights_present", True)
        self._codec_weights_present = kwargs.get("codec_weights_present", True)

        self.sampling_rate = int(getattr(self.model, "speaker_encoder_sample_rate", 24000))
        try:
            self.sampling_rate = int(self.model.speech_tokenizer.get_output_sample_rate())
        except Exception:
            pass

        # ``(module, attribute)`` for every forward the parts are installed over, so they can be
        # taken back off again - a part reached through a patched forward stays alive, and with it
        # the graph and the .bin it maps.
        self._patched_forwards: List[Tuple[Any, str]] = []
        self._parts_installed = False
        self._build_parts(ov_models or {})
        self._check_required_parts()

        if kwargs.get("compile", True) and not self._compile_only:
            self.compile()

    def _part_ov_config(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {**self.ov_config, **self._supported_ov_config(self._device, extra)}

    def _build_parts(self, ov_models: Dict[str, openvino.Model]) -> None:
        """Wrap each loaded graph in the part that drives it; components with no IR stay ``None``."""
        talker = self.model.talker
        code_predictor = talker.code_predictor
        talker_config = talker.model.config
        hidden_size = talker_config.hidden_size

        def build(name, part_cls, **extra):
            model = ov_models.get(name)
            if model is None:
                return None
            return part_cls(model, self, model_name=name, **extra)

        self.talker_model = build(
            "talker_model",
            OVQwen3TTSDecoderStack,
            num_layers=len(talker.model.layers),
            num_key_value_heads=talker_config.num_key_value_heads,
            ov_config=self._part_ov_config(self._TALKER_OV_CONFIG),
            position_fn=OVQwen3TTSDecoderStack.mrope_positions,
        )
        self.code_predictor_model = build(
            "code_predictor_model",
            OVQwen3TTSDecoderStack,
            num_layers=len(code_predictor.model.layers),
            num_key_value_heads=code_predictor.model.config.num_key_value_heads,
            ov_config=self._part_ov_config(self._CODE_PREDICTOR_OV_CONFIG),
            with_step=True,
        )
        self.text_embeddings = build("text_embeddings", OVQwen3TTSEmbedding, embedding_dim=hidden_size)
        self.talker_embeddings = build("talker_embeddings", OVQwen3TTSEmbedding, embedding_dim=hidden_size)
        self.code_predictor_embeddings = build(
            "code_predictor_embeddings", OVQwen3TTSEmbedding, embedding_dim=hidden_size
        )
        self.speaker_encoder = build("speaker_encoder", OVQwen3TTSSpeakerEncoder)
        self.codec_encoder = build("codec_encoder", OVQwen3TTSCodecEncoder)
        self.codec_decoder = build("codec_decoder", OVQwen3TTSCodecDecoder)

    def _check_required_parts(self) -> None:
        """Falling back to PyTorch is only possible where PyTorch weights exist.

        When the export left them out, the modules behind them are structural only, so a component
        whose IR is missing has to be an error rather than a silent switch to empty weights.
        """
        required = set()
        if not self._codec_weights_present:
            required |= {"codec_encoder", "codec_decoder"}
        if not self._weights_present:
            required |= {
                "talker_model",
                "code_predictor_model",
                "text_embeddings",
                "talker_embeddings",
                "code_predictor_embeddings",
            }
            if self.model.speaker_encoder is not None:
                required.add("speaker_encoder")
        missing = sorted(name for name in required if getattr(self, name, None) is None)
        if missing:
            raise RuntimeError(
                f"{self.model_save_dir} ships no PyTorch weights for these components, so they can only run "
                f"from OpenVINO, but the IR for: {', '.join(missing)} could not be loaded. "
                "Re-export the model with `optimum-cli export openvino`."
            )

    @property
    def _component_names(self) -> List[str]:
        return [name for name in self._all_ov_model_paths if getattr(self, name, None) is not None]

    @property
    def _ov_model_names(self) -> List[str]:
        return self._component_names

    @property
    def ov_models(self) -> Dict[str, openvino.Model]:
        return {name: component.model for name, component in self.components.items()}

    def clear_requests(self):
        for component in self.components.values():
            component.clear_requests()

    def compile(self):
        """Install the parts over the ``qwen_tts`` modules and compile them.

        Called on load unless ``compile=False`` deferred it - which is how a caller that only
        rewrites the IRs, as ``_main_quantize`` does, asks for the graphs to be left alone.
        """
        self._install_parts()
        for component in self.components.values():
            component.compile()

    def _patch_forward(self, module, attribute: str, replacement) -> None:
        """Install ``replacement`` over ``module.attribute``, remembering it for :meth:`_uninstall_parts`."""
        self._patched_forwards.append((module, attribute))
        setattr(module, attribute, replacement)

    def _install_parts(self) -> None:
        """Point the ``qwen_tts`` modules at the parts, leaving its orchestration untouched."""
        if self._parts_installed:
            return
        talker = self.model.talker
        code_predictor = talker.code_predictor

        if self.talker_model is not None:
            self._patch_forward(talker.model, "forward", self.talker_model.forward)
            # ``codec_head`` is folded into the talker graph, which computed these logits on the
            # call that produced the hidden states a moment ago.
            self._patch_forward(talker.codec_head, "forward", self.talker_model.head_logits)
        if self.code_predictor_model is not None:
            self._patch_forward(code_predictor.model, "forward", self.code_predictor_model.forward)
            self._patch_forward(code_predictor, "forward", self._code_predictor_forward)
        if self.text_embeddings is not None:
            self._patch_forward(talker.get_text_embeddings(), "forward", self.text_embeddings.forward)
            # ``text_projection`` was baked into the rows of the exported text table, so the
            # module that would apply it becomes an identity.
            self._patch_forward(talker.text_projection, "forward", lambda hidden_states: hidden_states)
        if self.talker_embeddings is not None:
            self._patch_forward(talker.get_input_embeddings(), "forward", self.talker_embeddings.forward)
        if self.code_predictor_embeddings is not None:
            for step, embedding in enumerate(code_predictor.get_input_embeddings()):
                self._patch_forward(embedding, "forward", self.code_predictor_embeddings.forward_for_step(step))
        if self.speaker_encoder is not None:
            self._patch_forward(self.model.speaker_encoder, "forward", self.speaker_encoder.forward)
        if self.codec_encoder is not None:
            self._patch_forward(self.model.speech_tokenizer.model.encoder, "encode", self.codec_encoder.encode)
        if self.codec_decoder is not None:
            self._patch_forward(self.model.speech_tokenizer.model.decoder, "forward", self.codec_decoder.forward)

        self._parts_installed = True

    def _uninstall_parts(self) -> None:
        """Give the ``qwen_tts`` modules their own forwards back.

        Deleting the instance attribute uncovers the class-level implementation again. Without
        this, the patched forwards keep every part - and every mapped .bin - alive.
        """
        for module, attribute in reversed(self._patched_forwards):
            try:
                delattr(module, attribute)
            except AttributeError:
                pass
        self._patched_forwards.clear()
        self._parts_installed = False

    def _code_predictor_forward(
        self,
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
        """Mirror ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward``.

        The stack and the per-depth ``lm_head`` are served by one graph call, and the depth index
        has to be known before it, which is why this wrapper replaces the outer forward rather
        than only the inner stack's.
        """
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorOutputWithPast

        code_predictor = self.model.talker.code_predictor
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2
        else:
            inputs_embeds = code_predictor.model.get_input_embeddings()[generation_steps - 1](input_ids)
        # `small_to_mtp_projection` is folded into the graph, so the embeddings are handed over in
        # the talker's width.

        outputs = self.code_predictor_model.forward(
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
            logits=self.code_predictor_model.head_state["logits"],
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

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
        weight_compression = cls._weight_compression_config(load_in_8bit, quantization_config)
        if export:
            model_id = cls._convert_checkpoint(model_id, cache_dir, weight_compression)
        elif weight_compression is not None:
            raise ValueError(
                "Weight compression of Qwen3-TTS is applied while exporting. Pass `export=True` to convert the "
                "checkpoint here, or compress at export time with "
                "`optimum-cli export openvino --weight-format int8/int4`."
            )

        # An export produced by this exporter carries no weights at all - every parameter is
        # in an IR - so the pipeline is built structurally. Original checkpoints (a Hub repo,
        # or a directory exported by an older version) keep taking the normal loader.
        weights_present = cls._has_checkpoint_weights(model_id)
        codec_weights_present = cls._has_codec_weights(model_id)
        if not weights_present:
            pipeline = cls._build_weightless_pipeline(model_id)
            # That path always builds the codec structurally too, whatever is on disk.
            codec_weights_present = False
        elif codec_weights_present:
            pipeline = Qwen3TTSModel.from_pretrained(str(model_id), **load_kwargs)
        else:
            with cls._weightless_codec(model_id):
                pipeline = Qwen3TTSModel.from_pretrained(str(model_id), **load_kwargs)
        pipeline.model.eval()

        if config is None:
            config = pipeline.model.config

        ir_dir = cls._resolve_ir_dir(model_id, cache_dir)
        return cls(
            pipeline=pipeline,
            config=config,
            ov_models=cls._load_ov_models(ir_dir),
            device=str(kwargs.pop("device", None) or "CPU").upper(),
            ov_config=kwargs.pop("ov_config", None),
            model_save_dir=model_id,
            ir_dir=ir_dir,
            weights_present=weights_present,
            codec_weights_present=codec_weights_present,
            **kwargs,
        )

    @classmethod
    def _load_ov_models(cls, ir_dir) -> Dict[str, openvino.Model]:
        """Read whichever component IRs are in ``ir_dir``, checking each is one this runtime drives."""
        core = openvino.Core()
        models = {}
        for name, ir_name in cls._all_ov_model_paths.items():
            ir_xml = Path(ir_dir) / ir_name
            try:
                if not ir_xml.is_file():
                    raise FileNotFoundError(f"{name} OpenVINO IR not found at {ir_xml}")
                logger.info(f"Qwen3-TTS: loading {name} OpenVINO IR from {ir_xml}.")
                model = core.read_model(ir_xml)
                cls._check_ir_signature(model, ir_xml, name, *cls._DECODER_STACK_PORTS.get(name, ((), ())))
                models[name] = model
            except Exception as exc:  # pragma: no cover - the component stays on PyTorch
                logger.debug(f"Qwen3-TTS: OpenVINO {name} offload disabled ({exc}); using PyTorch.")
        return models

    def _release_ov_models(self) -> None:
        """Drop every graph, and the file each maps, the way ``_unload_ov_model`` does.

        A part holds its ``ov.Model`` and, once compiled, the request built from it, and both keep
        the component's .bin mapped. :func:`compress_qwen3_tts_irs` replaces those files, which on
        Windows cannot be done at all while they are mapped.
        """
        self._uninstall_parts()
        self.clear_requests()
        for name in self._component_names:
            setattr(self, name, None)
        gc.collect()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        # OpenVINO components run on their configured device; ignore torch device moves.
        return self

    def can_generate(self) -> bool:
        return True

    def reshape(self, *args, **kwargs):
        logger.warning("Static shapes are not supported for Qwen3-TTS.")
        return self

    def _apply_quantization(
        self,
        quantization_config,
        save_directory=None,
        **kwargs,
    ) -> None:
        """Weight-compress every exported IR in place.

        Called by ``optimum-cli export openvino --weight-format int8`` (and the other weight-only
        formats) after the floating-point export has been written. Only the components in
        :data:`_QWEN3_TTS_COMPRESSIBLE_OV_IR_NAMES` are compressed - see the note there on why the
        codec is left in floating point - and each is compressed on its own, so this works whatever
        subset of components a given Qwen3-TTS variant produced.

        A 4-bit request is applied per component rather than across the board: only the IRs in
        :data:`_QWEN3_TTS_INT4_OV_IR_NAMES` are quantized to 4 bits, the rest fall back to 8, so
        `--weight-format int4` yields a mixed int4/int8 model.

        The graphs are released first: they are about to be replaced on disk. The instance is left
        without them, so reload it to run inference on the compressed model.
        """
        self._release_ov_models()
        self.compress_irs(
            self._ir_dir,
            quantization_config,
            output_dir=save_directory if save_directory is not None else self._ir_dir,
        )

    @property
    def _tts_model_type(self) -> str:
        """Where the checkpoint gets its voice from.

        Qwen3-TTS ships three variants - ``base`` clones a voice from reference audio,
        ``custom_voice`` picks a built-in speaker by name, and ``voice_design`` invents one
        from a text description. Each takes different inputs and is driven by a different
        ``qwen_tts`` entry point, which rejects the other variants' checkpoints outright.
        """
        return getattr(self.model, "tts_model_type", "base")

    @property
    def _is_custom_voice(self) -> bool:
        return self._tts_model_type == "custom_voice"

    @property
    def _is_voice_design(self) -> bool:
        return self._tts_model_type == "voice_design"

    def get_supported_speakers(self) -> List[str]:
        """Names accepted as ``speaker`` by a ``CustomVoice`` checkpoint."""
        if not self._is_custom_voice:
            raise ValueError(
                f"Built-in speakers exist only on Qwen3-TTS `CustomVoice` checkpoints; this one "
                f"is a `{self._tts_model_type}` model."
            )
        return list(self._pipeline.get_supported_speakers())

    def preprocess_input(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "Auto",
        ref_audio: Optional[Any] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        speaker: Optional[Union[str, List[str]]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build the inputs for :meth:`generate` from raw text and a voice.

        For the ``base`` (voice-clone) model this performs the reference-audio
        encoding and speaker-embedding extraction (ICL mode when ``ref_text`` is
        provided), mirroring ``qwen_tts.Qwen3TTSModel.create_voice_clone_prompt``.

        For a ``CustomVoice`` model the voice is a built-in ``speaker`` name, and for a
        ``VoiceDesign`` model it is described by ``instruct`` ("a calm elderly man, slight
        rasp"). Neither has reference audio to encode, so this only collects the arguments.

        Returns a dictionary that can be unpacked directly into :meth:`generate`.
        """
        if self._is_custom_voice:
            if speaker is None:
                raise ValueError(
                    "`speaker` must be provided for a Qwen3-TTS CustomVoice model. "
                    f"Supported speakers: {', '.join(self.get_supported_speakers())}."
                )
            inputs: Dict[str, Any] = {
                "text": text,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
            }
            inputs.update(kwargs)
            return inputs

        if self._is_voice_design:
            if not instruct:
                raise ValueError(
                    "`instruct` must be provided for a Qwen3-TTS VoiceDesign model: it is the "
                    "description the voice is built from, not an optional style hint."
                )
            inputs = {"text": text, "language": language, "instruct": instruct}
            inputs.update(kwargs)
            return inputs

        if ref_audio is None:
            raise ValueError("`ref_audio` must be provided for Qwen3-TTS voice cloning.")

        self.compile()  # the speaker encoder and the codec encoder run inside this call
        voice_clone_prompt = self._pipeline.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        inputs = {
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
        speaker: Optional[Union[str, List[str]]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        return_sample_rate: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, "tuple[torch.Tensor, int]"]:
        """Generate a speech waveform.

        The talker/code-predictor generation and the codec decoding are driven by the
        original ``qwen_tts`` orchestration (optionally OpenVINO-accelerated), through
        whichever entry point the checkpoint supports: a built-in speaker for ``CustomVoice``,
        a described voice for ``VoiceDesign``, a cloned reference voice otherwise.

        Returns a single waveform tensor (batch size 1) or a list of tensors for
        batched inputs.
        """
        self.compile()
        if self._is_custom_voice:
            wavs, sr = self._pipeline.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                **kwargs,
            )
        elif self._is_voice_design:
            wavs, sr = self._pipeline.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                **kwargs,
            )
        else:
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
