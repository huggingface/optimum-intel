"""OpenVINO export helpers for streaming Voxtral Realtime components."""

from pathlib import Path
from shutil import copyfile
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
    VoxtralRealtimeConv1dCacheLayer,
    VoxtralRealtimeConv1dPaddingCache,
    VoxtralRealtimeTextModel,
    VoxtralRealtimeTimeEmbedding,
)

from optimum.exporters.openvino.base import OpenVINOConfig
from optimum.utils.normalized_config import NormalizedConfig


def _padding_cache(embedder, conv1_cache, conv2_cache):
    padding_cache = VoxtralRealtimeConv1dPaddingCache()
    for cache_key, cache, conv in (
        ("conv1", conv1_cache, embedder.conv1),
        ("conv2", conv2_cache, embedder.conv2),
    ):
        cache_layer = VoxtralRealtimeConv1dCacheLayer()
        cache_layer.cache = cache.clone()
        cache_layer.is_initialized = True
        cache_layer.left_pad = conv.left_pad
        cache_layer.in_channels = conv.in_channels
        padding_cache.layers[cache_key] = cache_layer
    return padding_cache


def _dynamic_cache(past_key_values):
    cache = DynamicCache()
    cache.layers = []
    for layer_data in past_key_values:
        key, value = layer_data[:2]
        cache_layer = DynamicLayer()
        cache_layer.keys = key
        cache_layer.values = value
        cache_layer.is_initialized = True
        cache.layers.append(cache_layer)
    return cache


class VoxtralRealtimeStreamingEncoder(nn.Module):
    """Expose Voxtral's audio encoder stream state as tensor inputs and outputs."""

    def __init__(self, encoder: nn.Module, projector: nn.Module | None = None, downsample_factor: int = 4):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.downsample_factor = downsample_factor
        self.config = encoder.config

    def forward(self, input_features, conv1_cache, conv2_cache, past_key_values):
        padding_cache = _padding_cache(self.encoder.embedder, conv1_cache, conv2_cache)
        cache = _dynamic_cache(past_key_values)
        result = self.encoder(
            input_features=input_features,
            past_key_values=cache,
            padding_cache=padding_cache,
            use_cache=True,
            use_padding_cache=True,
        )

        audio_features = result.last_hidden_state
        output_name = "last_hidden_state"
        if self.projector is not None:
            audio_features = audio_features.reshape(
                audio_features.shape[0],
                -1,
                self.encoder.config.hidden_size * self.downsample_factor,
            )
            audio_features = self.projector(audio_features)
            output_name = "audio_embeddings"

        outputs = {
            output_name: audio_features,
            "conv1_cache": padding_cache.layers["conv1"].cache,
            "conv2_cache": padding_cache.layers["conv2"].cache,
        }
        for layer_index, layer in enumerate(result.past_key_values.layers):
            outputs[f"present.{layer_index}.key"] = layer.keys
            outputs[f"present.{layer_index}.value"] = layer.values
        return outputs


class VoxtralRealtimeStreamingEncoderOpenVINOConfig(OpenVINOConfig):
    """Export config for one stateful Voxtral audio-encoder chunk."""

    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    MIN_TRANSFORMERS_VERSION = "5.2.0"

    def __init__(self, config, task: str = "feature-extraction", projected: bool = False, **kwargs):
        self.chunk_frames = kwargs.pop("chunk_frames", 16)
        self.past_frames = kwargs.pop("past_frames", 8)
        self.projected = projected
        super().__init__(config=config, task=task, **kwargs)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        inputs = {
            "input_features": {0: "batch_size", 2: "audio_chunk_frames"},
            "conv1_cache": {0: "batch_size"},
            "conv2_cache": {0: "batch_size"},
        }
        for layer_index in range(self._config.num_hidden_layers):
            inputs[f"past_key_values.{layer_index}.key"] = {0: "batch_size", 2: "past_audio_tokens"}
            inputs[f"past_key_values.{layer_index}.value"] = {0: "batch_size", 2: "past_audio_tokens"}
        return inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        output_name = "audio_embeddings" if self.projected else "last_hidden_state"
        outputs = {
            output_name: {0: "batch_size", 1: "audio_chunk_tokens"},
            "conv1_cache": {0: "batch_size"},
            "conv2_cache": {0: "batch_size"},
        }
        for layer_index in range(self._config.num_hidden_layers):
            outputs[f"present.{layer_index}.key"] = {
                0: "batch_size",
                2: "past_audio_tokens + audio_chunk_tokens",
            }
            outputs[f"present.{layer_index}.value"] = {
                0: "batch_size",
                2: "past_audio_tokens + audio_chunk_tokens",
            }
        return outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs: Any) -> dict[str, Any]:
        if framework != "pt":
            raise ValueError(f"Unsupported framework: {framework}")

        batch_size = kwargs.get("batch_size", 1)
        chunk_frames = kwargs.get("sequence_length", self.chunk_frames)
        past_frames = kwargs.get("past_sequence_length", self.past_frames)
        config = self._config
        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        past_key_values = tuple(
            (
                torch.zeros(batch_size, config.num_attention_heads, past_frames, head_dim),
                torch.zeros(batch_size, config.num_attention_heads, past_frames, head_dim),
            )
            for _ in range(config.num_hidden_layers)
        )
        return {
            "input_features": torch.zeros(batch_size, config.num_mel_bins, chunk_frames),
            "conv1_cache": torch.zeros(batch_size, config.num_mel_bins, 2),
            "conv2_cache": torch.zeros(batch_size, config.hidden_size, 1),
            "past_key_values": past_key_values,
        }


def export_voxtral_streaming_encoder(encoder: nn.Module, output: str | Path, device: str = "cpu"):
    """Export the Voxtral streaming audio encoder component to OpenVINO IR."""
    from optimum.exporters.openvino.convert import export

    wrapped_encoder = VoxtralRealtimeStreamingEncoder(encoder)
    config = VoxtralRealtimeStreamingEncoderOpenVINOConfig(encoder.config)
    return export(
        wrapped_encoder,
        config,
        Path(output) / "openvino_model.xml",
        device=device,
        input_shapes={"sequence_length": config.chunk_frames, "past_sequence_length": config.past_frames},
        stateful=False,
        library_name="transformers",
    )


class VoxtralRealtimeStreamingTextDecoder(nn.Module):
    """Expose Voxtral's text decoder with combined embeddings and explicit KV state."""

    def __init__(
        self,
        language_model: VoxtralRealtimeTextModel,
        time_embedding: VoxtralRealtimeTimeEmbedding,
        lm_head: nn.Module,
    ):
        super().__init__()
        self.language_model = language_model
        self.time_embedding = time_embedding
        self.lm_head = lm_head
        self.config = language_model.config

    def forward(self, inputs_embeds, attention_mask, position_ids, num_delay_tokens, past_key_values):
        cache = _dynamic_cache(past_key_values)

        t_cond = self.time_embedding(num_delay_tokens.to(dtype=inputs_embeds.dtype))[None, ...]
        result = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            t_cond=t_cond,
        )

        outputs = {"logits": self.lm_head(result.last_hidden_state)}
        for layer_index, layer in enumerate(result.past_key_values.layers):
            outputs[f"present.{layer_index}.key"] = layer.keys
            outputs[f"present.{layer_index}.value"] = layer.values
        return outputs


class VoxtralRealtimeStreamingTextDecoderOpenVINOConfig(OpenVINOConfig):
    """Export config for one time-conditioned Voxtral text-decoder step."""

    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    MIN_TRANSFORMERS_VERSION = "5.2.0"

    def __init__(self, config, **kwargs):
        self.chunk_tokens = kwargs.pop("chunk_tokens", 8)
        self.past_tokens = kwargs.pop("past_tokens", 8)
        super().__init__(config=config, task="text-generation", **kwargs)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        inputs = {
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "past_sequence_length + sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            "num_delay_tokens": {},
        }
        for layer_index in range(self._config.num_hidden_layers):
            inputs[f"past_key_values.{layer_index}.key"] = {0: "batch_size", 2: "past_sequence_length"}
            inputs[f"past_key_values.{layer_index}.value"] = {0: "batch_size", 2: "past_sequence_length"}
        return inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        outputs = {"logits": {0: "batch_size", 1: "sequence_length"}}
        for layer_index in range(self._config.num_hidden_layers):
            outputs[f"present.{layer_index}.key"] = {
                0: "batch_size",
                2: "past_sequence_length + sequence_length",
            }
            outputs[f"present.{layer_index}.value"] = {
                0: "batch_size",
                2: "past_sequence_length + sequence_length",
            }
        return outputs

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs: Any) -> dict[str, Any]:
        if framework != "pt":
            raise ValueError(f"Unsupported framework: {framework}")

        batch_size = kwargs.get("batch_size", 1)
        sequence_length = kwargs.get("sequence_length", self.chunk_tokens)
        past_sequence_length = kwargs.get("past_sequence_length", self.past_tokens)
        config = self._config
        key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        past_key_values = tuple(
            (
                torch.zeros(batch_size, key_value_heads, past_sequence_length, head_dim),
                torch.zeros(batch_size, key_value_heads, past_sequence_length, head_dim),
            )
            for _ in range(config.num_hidden_layers)
        )
        return {
            "inputs_embeds": torch.zeros(batch_size, sequence_length, config.hidden_size),
            "attention_mask": torch.ones(batch_size, past_sequence_length + sequence_length, dtype=torch.long),
            "position_ids": torch.arange(past_sequence_length, past_sequence_length + sequence_length)
            .unsqueeze(0)
            .expand(batch_size, -1),
            "num_delay_tokens": torch.full((1,), 6.0),
            "past_key_values": past_key_values,
        }


class VoxtralRealtimeAudioConv(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.embedder = encoder.embedder
        self.config = encoder.config

    def forward(self, input_features, conv1_cache, conv2_cache):
        cache = _padding_cache(self.embedder, conv1_cache, conv2_cache)
        inputs_embeds = self.embedder(input_features, cache)
        return {
            "inputs_embeds": inputs_embeds,
            "conv1_cache": cache.layers["conv1"].cache,
            "conv2_cache": cache.layers["conv2"].cache,
        }


class VoxtralRealtimeAudioTransformer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config

    def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values):
        from optimum.exporters.openvino.model_patcher import _paged_attention_masks

        full_mask, sliding_mask = _paged_attention_masks(
            attention_mask, None, self.config.sliding_window, inputs_embeds.dtype, inputs_embeds.shape[1]
        )
        result = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=sliding_mask if sliding_mask is not None else full_mask,
            position_ids=position_ids,
            past_key_values=_dynamic_cache(past_key_values),
            use_cache=True,
        )
        outputs = {"last_hidden_state": result.last_hidden_state}
        for layer_index, layer in enumerate(result.past_key_values.layers):
            outputs[f"present.{layer_index}.key"] = layer.keys
            outputs[f"present.{layer_index}.value"] = layer.values
        return outputs


class VoxtralRealtimeAudioAdapter(nn.Module):
    def __init__(self, encoder, projector, downsample_factor):
        super().__init__()
        self.projector = projector
        self.config = encoder.config
        self.downsample_factor = downsample_factor

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], -1, self.config.hidden_size * self.downsample_factor
        )
        return {"audio_embeddings": self.projector(hidden_states)}


class VoxtralRealtimeAudioConvOpenVINOConfig(VoxtralRealtimeStreamingEncoderOpenVINOConfig):
    @property
    def inputs(self):
        return {key: value for key, value in super().inputs.items() if not key.startswith("past_key_values")}

    @property
    def outputs(self):
        return {
            "inputs_embeds": {0: "batch_size", 1: "audio_chunk_tokens"},
            "conv1_cache": {0: "batch_size"},
            "conv2_cache": {0: "batch_size"},
        }

    def generate_dummy_inputs(self, framework="pt", **kwargs):
        inputs = super().generate_dummy_inputs(framework, **kwargs)
        del inputs["past_key_values"]
        return inputs


class VoxtralRealtimeAudioTransformerOpenVINOConfig(VoxtralRealtimeStreamingTextDecoderOpenVINOConfig):
    @property
    def inputs(self):
        return {key: value for key, value in super().inputs.items() if key != "num_delay_tokens"}

    @property
    def outputs(self):
        outputs = super().outputs
        hidden_states = outputs.pop("logits")
        return {"last_hidden_state": hidden_states, **outputs}

    def generate_dummy_inputs(self, framework="pt", **kwargs):
        inputs = super().generate_dummy_inputs(framework, **kwargs)
        del inputs["num_delay_tokens"]
        return inputs


class VoxtralRealtimeAudioAdapterOpenVINOConfig(OpenVINOConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    MIN_TRANSFORMERS_VERSION = "5.2.0"

    def __init__(self, config, downsample_factor, **kwargs):
        self.downsample_factor = downsample_factor
        super().__init__(config=config, task="feature-extraction", **kwargs)

    @property
    def inputs(self):
        return {"hidden_states": {0: "batch_size", 1: "audio_chunk_tokens"}}

    @property
    def outputs(self):
        return {"audio_embeddings": {0: "batch_size", 1: "audio_output_tokens"}}

    def generate_dummy_inputs(self, framework="pt", **kwargs):
        if framework != "pt":
            raise ValueError(f"Unsupported framework: {framework}")
        return {
            "hidden_states": torch.zeros(
                kwargs.get("batch_size", 1), self.downsample_factor * 2, self._config.hidden_size
            )
        }


def _voxtral_audio_parts(model):
    encoder = model.model.audio_tower
    factor = model.config.downsample_factor
    return {
        "audio_conv": (VoxtralRealtimeAudioConv(encoder), VoxtralRealtimeAudioConvOpenVINOConfig(encoder.config)),
        "audio_transformer": (
            VoxtralRealtimeAudioTransformer(encoder),
            VoxtralRealtimeAudioTransformerOpenVINOConfig(encoder.config),
        ),
        "audio_adapter": (
            VoxtralRealtimeAudioAdapter(encoder, model.model.multi_modal_projector, factor),
            VoxtralRealtimeAudioAdapterOpenVINOConfig(encoder.config, factor),
        ),
    }


def _save_voxtral_realtime_assets(model, output):
    from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import (
        VoxtralRealtimeFeatureExtractor,
    )
    from transformers.utils.hub import cached_file

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(output)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output)
    source = model.config._name_or_path
    if not source:
        return
    VoxtralRealtimeFeatureExtractor.from_pretrained(source).save_pretrained(output)
    for name in ("tekken.json", "processor_config.json"):
        asset = cached_file(source, name, _raise_exceptions_for_missing_entries=name == "tekken.json")
        if asset is not None and Path(asset).resolve() != (output / name).resolve():
            copyfile(asset, output / name)


def export_voxtral_streaming_text_decoder(
    language_model: VoxtralRealtimeTextModel,
    time_embedding: VoxtralRealtimeTimeEmbedding,
    lm_head: nn.Module,
    output: str | Path,
    device: str = "cpu",
):
    """Export the Voxtral time-conditioned text decoder component to OpenVINO IR."""
    from optimum.exporters.openvino.convert import export

    wrapped_decoder = VoxtralRealtimeStreamingTextDecoder(language_model, time_embedding, lm_head)
    config = VoxtralRealtimeStreamingTextDecoderOpenVINOConfig(language_model.config)
    return export(
        wrapped_decoder,
        config,
        Path(output) / "openvino_model.xml",
        device=device,
        input_shapes={"sequence_length": config.chunk_tokens, "past_sequence_length": config.past_tokens},
        stateful=True,
        library_name="transformers",
    )


def export_voxtral_realtime_submodels(model: nn.Module, output: str | Path, device: str = "cpu"):
    """Export Voxtral audio, text embeddings, and text decoder as separate IRs."""
    from optimum.exporters.openvino.convert import export

    output_path = Path(output)
    _save_voxtral_realtime_assets(model, output_path)
    configs, get_submodels = _get_voxtral_realtime_submodels_fn_and_export_configs(model)
    submodels = get_submodels(model)
    audio_config = configs["audio_encoder_model"]
    audio_result = export(
        submodels["audio_encoder_model"],
        audio_config,
        output_path / "audio_encoder" / "openvino_model.xml",
        device=device,
        input_shapes={"sequence_length": audio_config.chunk_frames, "past_sequence_length": audio_config.past_frames},
        stateful=False,
        library_name="transformers",
    )
    audio_parts = {}
    for name in ("audio_conv", "audio_transformer", "audio_adapter"):
        audio_parts[name] = export(
            submodels[name],
            configs[name],
            output_path / name / "openvino_model.xml",
            device=device,
            stateful=name == "audio_transformer",
            library_name="transformers",
        )
    embedding_result = export(
        submodels["text_embeddings_model"],
        configs["text_embeddings_model"],
        output_path / "text_embeddings" / "openvino_model.xml",
        device=device,
        stateful=False,
        library_name="transformers",
    )
    text_result = export_voxtral_streaming_text_decoder(
        model.model.language_model,
        model.model.time_embedding,
        model.lm_head,
        output_path / "text_decoder",
        device=device,
    )
    return {
        "audio_encoder": audio_result,
        **audio_parts,
        "text_embeddings_model": embedding_result,
        "text_decoder": text_result,
    }


def _get_voxtral_realtime_submodels_fn_and_export_configs(
    model: nn.Module,
    library_name: str = "transformers",
    task: str | None = None,
    preprocessors: list[Any] | None = None,
    custom_export_configs: dict[str, OpenVINOConfig] | None = None,
    fn_get_submodels=None,
):
    from optimum.exporters.openvino.model_configs import InputEmbedOpenVINOConfig

    audio_encoder = VoxtralRealtimeStreamingEncoder(
        model.model.audio_tower,
        projector=model.model.multi_modal_projector,
        downsample_factor=model.config.downsample_factor,
    )
    text_decoder = VoxtralRealtimeStreamingTextDecoder(
        model.model.language_model, model.model.time_embedding, model.lm_head
    )
    text_embeddings = model.model.language_model.get_input_embeddings()
    text_embeddings.config = model.config.text_config
    configs = {
        "audio_encoder_model": VoxtralRealtimeStreamingEncoderOpenVINOConfig(
            model.model.audio_tower.config,
            projected=True,
        ),
        "text_embeddings_model": InputEmbedOpenVINOConfig(model.config.text_config, task="feature-extraction"),
        "language_model": VoxtralRealtimeStreamingTextDecoderOpenVINOConfig(model.model.language_model.config),
    }
    audio_parts = _voxtral_audio_parts(model)
    configs.update({name: config for name, (_, config) in audio_parts.items()})

    def get_submodels(_model):
        return {
            "audio_encoder_model": audio_encoder,
            "text_embeddings_model": text_embeddings,
            "language_model": text_decoder,
            **{name: part for name, (part, _) in audio_parts.items()},
        }

    return configs, get_submodels
