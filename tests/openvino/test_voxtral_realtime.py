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

from optimum.exporters.openvino.convert import export_from_model
from optimum.exporters.openvino.voxtral_realtime import (
    VoxtralRealtimeStreamingEncoder,
    VoxtralRealtimeStreamingTextDecoder,
    export_voxtral_streaming_encoder,
    export_voxtral_streaming_text_decoder,
    export_voxtral_realtime_submodels,
)
from optimum.intel.openvino.utils import (
    OV_AUDIO_ENCODER_MODEL_NAME,
    OV_LANGUAGE_MODEL_NAME,
    OV_TEXT_EMBEDDINGS_MODEL_NAME,
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


@pytest.fixture
def voxtral_model():
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
    return VoxtralRealtimeForConditionalGeneration(config).eval()


def test_streaming_submodel_exports_carry_state(tmp_path, voxtral_model):
    model = voxtral_model
    config = model.config
    audio_wrapper = VoxtralRealtimeStreamingEncoder(
        model.model.audio_tower,
        projector=model.model.multi_modal_projector,
        downsample_factor=config.downsample_factor,
    ).eval()
    text_wrapper = VoxtralRealtimeStreamingTextDecoder(
        model.model.language_model, model.model.time_embedding, model.lm_head
    ).eval()
    export_voxtral_realtime_submodels(model, tmp_path)

    core = ov.Core()
    embeddings = core.compile_model(core.read_model(str(tmp_path / "text_embeddings/openvino_model.xml")), "CPU")
    input_ids = torch.tensor([[0, 1, 7], [2, 5, 3]], dtype=torch.long)
    with torch.no_grad():
        expected_embeddings = model.model.language_model.get_input_embeddings()(input_ids).numpy()
    np.testing.assert_allclose(embeddings({"input": input_ids.numpy()})["inputs_embeds"], expected_embeddings)
    audio = core.compile_model(core.read_model(str(tmp_path / "audio_encoder/openvino_model.xml")), "CPU")
    text = core.compile_model(core.read_model(str(tmp_path / "text_decoder/openvino_model.xml")), "CPU")
    text_request = text.create_infer_request()
    text_input_names = {name for port in text.inputs for name in port.get_names()}
    assert "inputs_embeds" in text_input_names
    assert "input_ids" not in text_input_names
    assert "audio_embeddings" not in text_input_names
    assert len(text_request.query_state()) == 2 * config.text_config.num_hidden_layers
    assert not any("past_key_values" in name for port in text.inputs for name in port.get_names())
    assert not any("present" in name for port in text.outputs for name in port.get_names())
    audio_feed = {
        "conv1_cache": torch.zeros(1, config.audio_config.num_mel_bins, 2),
        "conv2_cache": torch.zeros(1, config.audio_config.hidden_size, 1),
    }
    audio_past = tuple((torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8)) for _ in range(2))
    text_past = tuple((torch.zeros(1, 2, 0, 8), torch.zeros(1, 2, 0, 8)) for _ in range(2))

    first_text_inputs = None
    first_logits = None
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
            (
                torch.from_numpy(actual_audio[f"present.{layer}.key"]),
                torch.from_numpy(actual_audio[f"present.{layer}.value"]),
            )
            for layer in range(2)
        )

        audio_embeddings = actual_audio["audio_embeddings"]
        sequence_length = audio_embeddings.shape[1]
        past_length = text_past[0][0].shape[2]
        input_ids = torch.randint(config.text_config.vocab_size, (1, sequence_length))
        token_embeddings = embeddings({"input": input_ids.numpy()})["inputs_embeds"]
        inputs_embeds = token_embeddings + audio_embeddings
        position_ids = torch.arange(past_length, past_length + sequence_length).unsqueeze(0)
        attention_mask = torch.ones((1, past_length + sequence_length), dtype=torch.long)
        num_delay_tokens = torch.tensor([6.0])
        with torch.no_grad():
            expected_text = text_wrapper(
                torch.from_numpy(inputs_embeds),
                attention_mask,
                position_ids,
                num_delay_tokens,
                text_past,
            )
        text_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
            "num_delay_tokens": num_delay_tokens.numpy(),
            "beam_idx": np.arange(input_ids.shape[0], dtype=np.int32),
        }
        actual_text = text_request.infer(text_inputs)
        np.testing.assert_allclose(actual_text["logits"], expected_text["logits"].numpy(), rtol=1e-2, atol=1.5e-2)
        if first_text_inputs is None:
            first_text_inputs = text_inputs
            first_logits = actual_text["logits"].copy()
        text_past = tuple(
            (expected_text[f"present.{layer}.key"], expected_text[f"present.{layer}.value"]) for layer in range(2)
        )

    text_request.reset_state()
    reset_logits = text_request.infer(first_text_inputs)["logits"]
    np.testing.assert_allclose(reset_logits, first_logits, rtol=1e-5, atol=1e-5)


def test_streaming_text_decoder_crosses_sliding_window(tmp_path, voxtral_model, monkeypatch):
    model = voxtral_model
    config = model.config.text_config
    wrapper = VoxtralRealtimeStreamingTextDecoder(
        model.model.language_model, model.model.time_embedding, model.lm_head
    ).eval()
    export_voxtral_streaming_text_decoder(
        model.model.language_model, model.model.time_embedding, model.lm_head, tmp_path
    )
    request = (
        ov.Core()
        .compile_model(str(tmp_path / "openvino_model.xml"), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        .create_infer_request()
    )
    past = tuple(
        (
            torch.zeros(1, config.num_key_value_heads, 0, config.head_dim),
            torch.zeros(1, config.num_key_value_heads, 0, config.head_dim),
        )
        for _ in range(config.num_hidden_layers)
    )
    delay = torch.tensor([float(model.config.default_num_delay_tokens)])
    past_length = 0
    for sequence_length in (config.sliding_window - 1, 1, 1, 2):
        inputs_embeds = torch.randn(1, sequence_length, config.hidden_size)
        position_ids = torch.arange(past_length, past_length + sequence_length).unsqueeze(0)
        attention_mask = torch.ones(1, past_length + sequence_length, dtype=torch.long)
        with torch.no_grad():
            expected = wrapper(inputs_embeds, attention_mask, position_ids, delay, past)
        actual = request.infer(
            {
                "inputs_embeds": inputs_embeds.numpy(),
                "attention_mask": attention_mask.numpy(),
                "position_ids": position_ids.numpy(),
                "num_delay_tokens": delay.numpy(),
                "beam_idx": np.array([0], dtype=np.int32),
            }
        )
        expected_logits = expected["logits"].numpy()
        np.testing.assert_allclose(actual["logits"], expected_logits, rtol=1e-4, atol=1e-5)
        if past_length == config.sliding_window:
            with monkeypatch.context() as patch, torch.no_grad():
                patch.setattr(config, "sliding_window", None)
                unwindowed_logits = wrapper(inputs_embeds, attention_mask, position_ids, delay, past)["logits"].numpy()
            window_error = np.max(np.abs(actual["logits"] - expected_logits))
            unwindowed_error = np.max(np.abs(actual["logits"] - unwindowed_logits))
            assert unwindowed_error > 2 * window_error
        past = tuple(
            (expected[f"present.{layer}.key"], expected[f"present.{layer}.value"])
            for layer in range(config.num_hidden_layers)
        )
        past_length += sequence_length
    assert past_length > config.sliding_window


@pytest.mark.parametrize("delay", [2, 6])
def test_streaming_text_decoder_matches_transformers(voxtral_model, delay):
    model = voxtral_model
    config = model.config.text_config
    wrapper = VoxtralRealtimeStreamingTextDecoder(
        model.model.language_model, model.model.time_embedding, model.lm_head
    ).eval()
    features = torch.randn(1, model.config.audio_config.num_mel_bins, 16)
    with torch.no_grad():
        audio_embeddings = model.model.get_audio_features(input_features=features).pooler_output
        input_ids = torch.zeros(audio_embeddings.shape[:2], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        past = tuple(
            (
                torch.zeros(1, config.num_key_value_heads, 0, config.head_dim),
                torch.zeros(1, config.num_key_value_heads, 0, config.head_dim),
            )
            for _ in range(config.num_hidden_layers)
        )
        inputs_embeds = model.model.language_model.get_input_embeddings()(input_ids) + audio_embeddings
        actual = wrapper(inputs_embeds, attention_mask, position_ids, torch.tensor([float(delay)]), past)
        expected = model(
            input_ids=input_ids,
            input_features=features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_delay_tokens=delay,
        )
    torch.testing.assert_close(actual["logits"], expected.logits)


@pytest.mark.parametrize("stateful", [False, True])
def test_export_from_model_selects_text_state(tmp_path, voxtral_model, stateful):
    paths = export_from_model(voxtral_model, tmp_path, task="automatic-speech-recognition", stateful=stateful)
    expected_names = {OV_AUDIO_ENCODER_MODEL_NAME, OV_TEXT_EMBEDDINGS_MODEL_NAME, OV_LANGUAGE_MODEL_NAME}
    assert set(paths) == expected_names
    assert {path.name for path in tmp_path.glob("*.xml")} == expected_names
    assert {path.name for path in tmp_path.glob("*.bin")} == {
        (tmp_path / name).with_suffix(".bin").name for name in expected_names
    }
    core = ov.Core()
    audio = core.read_model(str(tmp_path / OV_AUDIO_ENCODER_MODEL_NAME))
    text = core.read_model(str(tmp_path / OV_LANGUAGE_MODEL_NAME))
    embeddings = core.compile_model(core.read_model(str(tmp_path / OV_TEXT_EMBEDDINGS_MODEL_NAME)), "CPU")
    input_ids = torch.tensor([[1, 4], [3, 5]], dtype=torch.long)
    with torch.no_grad():
        expected_embeddings = voxtral_model.model.language_model.get_input_embeddings()(input_ids).numpy()
    np.testing.assert_allclose(embeddings({"input": input_ids.numpy()})["inputs_embeds"], expected_embeddings)
    assert not audio.get_sinks()
    assert not embeddings.create_infer_request().query_state()
    assert bool(text.get_sinks()) is stateful
    text_input_names = {name for port in text.inputs for name in port.get_names()}
    assert "inputs_embeds" in text_input_names
    assert "input_ids" not in text_input_names
    assert "audio_embeddings" not in text_input_names
    inputs_embeds = torch.from_numpy(embeddings({"input": input_ids.numpy()})["inputs_embeds"].copy())
    inputs_embeds += torch.randn_like(inputs_embeds)
    batch_size, sequence_length = input_ids.shape
    config = voxtral_model.config.text_config
    past = tuple(
        (
            torch.zeros(batch_size, config.num_key_value_heads, 0, config.head_dim),
            torch.zeros(batch_size, config.num_key_value_heads, 0, config.head_dim),
        )
        for _ in range(config.num_hidden_layers)
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(sequence_length).expand(batch_size, -1)
    delay = torch.tensor([6.0])
    wrapper = VoxtralRealtimeStreamingTextDecoder(
        voxtral_model.model.language_model, voxtral_model.model.time_embedding, voxtral_model.lm_head
    ).eval()
    with torch.no_grad():
        expected = wrapper(inputs_embeds, attention_mask, position_ids, delay, past)["logits"].numpy()
    text_inputs = {
        "inputs_embeds": inputs_embeds.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
        "num_delay_tokens": delay.numpy(),
    }
    if stateful:
        text_inputs["beam_idx"] = np.arange(batch_size, dtype=np.int32)
    else:
        for layer_index, (key, value) in enumerate(past):
            text_inputs[f"past_key_values.{layer_index}.key"] = key.numpy()
            text_inputs[f"past_key_values.{layer_index}.value"] = value.numpy()
    actual = core.compile_model(text, "CPU").create_infer_request().infer(text_inputs)["logits"]
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1.5e-2)
