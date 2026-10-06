import numpy as np
import openvino as ov
import pytest
import torch

transformers = pytest.importorskip("transformers.models.voxtral_realtime")
from transformers.models.voxtral_realtime.configuration_voxtral_realtime import (
    VoxtralRealtimeConfig,
    VoxtralRealtimeEncoderConfig,
)
from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
    VoxtralRealtimeEncoder,
    VoxtralRealtimeForConditionalGeneration,
)

from optimum.exporters.openvino.voxtral_realtime import (
    VoxtralRealtimeStreamingEncoder,
    VoxtralRealtimeStreamingTextDecoder,
    export_voxtral_streaming_encoder,
    export_voxtral_realtime_submodels,
)


def test_streaming_encoder_export_carries_state(tmp_path):
    torch.manual_seed(7)
    config = VoxtralRealtimeEncoderConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_mel_bins=8,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=32,
    )
    encoder = VoxtralRealtimeEncoder(config).eval()
    wrapper = VoxtralRealtimeStreamingEncoder(encoder).eval()
    export_voxtral_streaming_encoder(encoder, tmp_path)

    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(tmp_path / "openvino_model.xml")), "CPU")
    output_names = {port.get_any_name().removesuffix("_1"): port for port in compiled.outputs}

    feed = {
        "conv1_cache": torch.zeros(1, config.num_mel_bins, 2),
        "conv2_cache": torch.zeros(1, config.hidden_size, 1),
    }
    for layer_index in range(config.num_hidden_layers):
        feed[f"past_key_values.{layer_index}.key"] = torch.zeros(1, config.num_attention_heads, 8, config.head_dim)
        feed[f"past_key_values.{layer_index}.value"] = torch.zeros(1, config.num_attention_heads, 8, config.head_dim)

    for _ in range(2):
        features = torch.randn(1, config.num_mel_bins, 16)
        past_key_values = tuple(
            (feed[f"past_key_values.{layer}.key"], feed[f"past_key_values.{layer}.value"])
            for layer in range(config.num_hidden_layers)
        )
        with torch.no_grad():
            expected = wrapper(features, feed["conv1_cache"], feed["conv2_cache"], past_key_values)

        actual = compiled({key: value.numpy() for key, value in {**feed, "input_features": features}.items()})
        np.testing.assert_allclose(
            actual[output_names["last_hidden_state"]], expected["last_hidden_state"].numpy(), rtol=1e-2, atol=1.5e-2
        )

        for cache_name in ("conv1_cache", "conv2_cache"):
            np.testing.assert_allclose(
                actual[output_names[cache_name]], expected[cache_name].numpy(), rtol=1e-2, atol=3e-3
            )
            feed[cache_name] = torch.from_numpy(actual[output_names[cache_name]])

        for layer_index in range(config.num_hidden_layers):
            for kind in ("key", "value"):
                output_name = f"present.{layer_index}.{kind}"
                np.testing.assert_allclose(
                    actual[output_names[output_name]], expected[output_name].numpy(), rtol=1e-2, atol=3e-3
                )
                feed[f"past_key_values.{layer_index}.{kind}"] = torch.from_numpy(actual[output_names[output_name]])


def test_streaming_submodel_exports_carry_state(tmp_path):
    torch.manual_seed(10)
    config = VoxtralRealtimeConfig(
        audio_config={
            "model_type": "voxtral_realtime_encoder",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_mel_bins": 8,
            "head_dim": 8,
            "max_position_embeddings": 256,
            "sliding_window": 32,
        },
        text_config={
            "model_type": "voxtral_realtime_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "max_position_embeddings": 256,
            "sliding_window": 32,
        },
    )
    model = VoxtralRealtimeForConditionalGeneration(config).eval()
    audio_wrapper = VoxtralRealtimeStreamingEncoder(
        model.audio_tower,
        projector=model.multi_modal_projector,
        downsample_factor=config.downsample_factor,
    ).eval()
    text_wrapper = VoxtralRealtimeStreamingTextDecoder(model.language_model, model.time_embedding).eval()
    export_voxtral_realtime_submodels(model, tmp_path)

    core = ov.Core()
    audio = core.compile_model(core.read_model(str(tmp_path / "audio_encoder/openvino_model.xml")), "CPU")
    text = core.compile_model(core.read_model(str(tmp_path / "text_decoder/openvino_model.xml")), "CPU")
    audio_feed = {
        "conv1_cache": torch.zeros(1, config.audio_config.num_mel_bins, 2),
        "conv2_cache": torch.zeros(1, config.audio_config.hidden_size, 1),
    }
    audio_past = tuple((torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8)) for _ in range(2))
    text_past = tuple((torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8)) for _ in range(2))

    for _ in range(2):
        for layer_index, (key, value) in enumerate(audio_past):
            audio_feed[f"past_key_values.{layer_index}.key"] = key
            audio_feed[f"past_key_values.{layer_index}.value"] = value
        features = torch.randn(1, config.audio_config.num_mel_bins, 16)
        with torch.no_grad():
            expected_audio = audio_wrapper(
                features,
                audio_feed["conv1_cache"],
                audio_feed["conv2_cache"],
                audio_past,
            )
        actual_audio = audio(
            {
                **{key: value.numpy() for key, value in audio_feed.items()},
                "input_features": features.numpy(),
            }
        )
        np.testing.assert_allclose(
            actual_audio["audio_embeddings"], expected_audio["audio_embeddings"].numpy(), rtol=1e-2, atol=1.5e-2
        )
        audio_feed["conv1_cache"] = torch.from_numpy(actual_audio["conv1_cache_1"])
        audio_feed["conv2_cache"] = torch.from_numpy(actual_audio["conv2_cache_1"])
        audio_past = tuple(
            (torch.from_numpy(actual_audio[f"present.{layer}.key"]), torch.from_numpy(actual_audio[f"present.{layer}.value"]))
            for layer in range(2)
        )

        audio_embeddings = actual_audio["audio_embeddings"]
        sequence_length = audio_embeddings.shape[1]
        past_length = text_past[0][0].shape[2]
        input_ids = torch.zeros((1, sequence_length), dtype=torch.long)
        position_ids = torch.arange(past_length, past_length + sequence_length).unsqueeze(0)
        attention_mask = torch.ones((1, past_length + sequence_length), dtype=torch.long)
        num_delay_tokens = torch.tensor([6.0])
        with torch.no_grad():
            expected_text = text_wrapper(
                input_ids,
                torch.from_numpy(audio_embeddings),
                attention_mask,
                position_ids,
                num_delay_tokens,
                text_past,
            )
        text_inputs = {
            "input_ids": input_ids.numpy(),
            "audio_embeddings": audio_embeddings,
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
            "num_delay_tokens": num_delay_tokens.numpy(),
        }
        for layer_index, (key, value) in enumerate(text_past):
            text_inputs[f"past_key_values.{layer_index}.key"] = key.numpy()
            text_inputs[f"past_key_values.{layer_index}.value"] = value.numpy()
        actual_text = text(text_inputs)
        np.testing.assert_allclose(actual_text["logits"], expected_text["logits"].numpy(), rtol=1e-2, atol=1.5e-2)
        text_past = tuple(
            (torch.from_numpy(actual_text[f"present.{layer}.key"]), torch.from_numpy(actual_text[f"present.{layer}.value"]))
            for layer in range(2)
        )