import json

import numpy as np
import openvino as ov
import pytest
import torch
from openvino._offline_transformations import paged_attention_transformation
from transformers.cache_utils import DynamicCache

transformers = pytest.importorskip("transformers.models.voxtral_realtime")
from transformers.models.voxtral_realtime.configuration_voxtral_realtime import (
    VoxtralRealtimeConfig,
    VoxtralRealtimeEncoderConfig,
)
from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import VoxtralRealtimeFeatureExtractor
from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
    VoxtralRealtimeEncoder,
    VoxtralRealtimeForConditionalGeneration,
)

from optimum.exporters.openvino.convert import export, export_from_model
from optimum.exporters.openvino.model_patcher import _paged_attention_masks
from optimum.exporters.openvino.voxtral_realtime import (
    VoxtralRealtimeAudioTransformer,
    VoxtralRealtimeAudioTransformerOpenVINOConfig,
    VoxtralRealtimeStreamingEncoder,
    VoxtralRealtimeStreamingTextDecoder,
    export_voxtral_realtime_submodels,
    export_voxtral_streaming_encoder,
    export_voxtral_streaming_text_decoder,
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
    source = tmp_path / "source"
    source.mkdir()
    VoxtralRealtimeFeatureExtractor(feature_size=config.audio_config.num_mel_bins).save_pretrained(source)
    (source / "tekken.json").write_text('{"test": true}')
    model.config._name_or_path = str(source)
    audio_wrapper = VoxtralRealtimeStreamingEncoder(
        model.model.audio_tower,
        projector=model.model.multi_modal_projector,
        downsample_factor=config.downsample_factor,
    ).eval()
    text_wrapper = VoxtralRealtimeStreamingTextDecoder(
        model.model.language_model, model.model.time_embedding, model.lm_head
    ).eval()
    export_voxtral_realtime_submodels(model, tmp_path)
    assert (tmp_path / "preprocessor_config.json").is_file()
    assert (
        json.loads((tmp_path / "config.json").read_text())["audio_config"]["sliding_window"]
        == config.audio_config.sliding_window
    )
    assert (tmp_path / "generation_config.json").is_file()
    assert (tmp_path / "tekken.json").read_text() == (source / "tekken.json").read_text()

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


def test_split_audio_matches_streaming_encoder_across_window(tmp_path, voxtral_model):
    model = voxtral_model
    config = model.config.audio_config
    export_voxtral_realtime_submodels(model, tmp_path)
    core = ov.Core()
    options = {"INFERENCE_PRECISION_HINT": "f32"}
    legacy = core.compile_model(str(tmp_path / "audio_encoder/openvino_model.xml"), "CPU", options)
    conv = core.compile_model(str(tmp_path / "audio_conv/openvino_model.xml"), "CPU", options)
    transformer = core.compile_model(str(tmp_path / "audio_transformer/openvino_model.xml"), "CPU", options)
    adapter = core.compile_model(str(tmp_path / "audio_adapter/openvino_model.xml"), "CPU", options)
    request = transformer.create_infer_request()
    assert len(request.query_state()) == 2 * config.num_hidden_layers
    assert {port.get_any_name() for port in transformer.inputs} == {
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "beam_idx",
    }

    conv_caches = {
        "conv1_cache": np.zeros((2, config.num_mel_bins, 2), dtype=np.float32),
        "conv2_cache": np.zeros((2, config.hidden_size, 1), dtype=np.float32),
    }
    past = {
        f"past_key_values.{layer}.{kind}": np.zeros((2, config.num_attention_heads, 0, config.head_dim), np.float32)
        for layer in range(config.num_hidden_layers)
        for kind in ("key", "value")
    }
    past_length = 0
    for frames in (16, 48, 16, 8):
        features = torch.randn(2, config.num_mel_bins, frames).numpy()
        original = legacy({"input_features": features, **conv_caches, **past})
        conv_out = conv({"input_features": features, **conv_caches})
        token_count = conv_out["inputs_embeds"].shape[1]
        mask = np.ones((2, past_length + token_count), dtype=np.int64)
        positions = np.broadcast_to(np.arange(past_length, past_length + token_count), (2, token_count)).copy()
        transformed = request.infer(
            {
                "inputs_embeds": conv_out["inputs_embeds"],
                "attention_mask": mask,
                "position_ids": positions,
                "beam_idx": np.arange(2, dtype=np.int32),
            }
        )
        actual = adapter({"hidden_states": transformed["last_hidden_state"]})["audio_embeddings"]
        np.testing.assert_allclose(actual, original["audio_embeddings"], rtol=1e-2, atol=1e-4)
        for name in conv_caches:
            np.testing.assert_allclose(conv_out[name + "_1"], original[name + "_1"], rtol=1e-5, atol=1e-6)
            conv_caches[name] = conv_out[name + "_1"].copy()
        past = {
            f"past_key_values.{layer}.{kind}": original[f"present.{layer}.{kind}"].copy()
            for layer in range(config.num_hidden_layers)
            for kind in ("key", "value")
        }
        past_length += token_count
    assert past_length > config.sliding_window


def test_audio_transformer_paged_attention_32_layers(tmp_path):
    config = VoxtralRealtimeEncoderConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_mel_bins=8,
        head_dim=8,
        sliding_window=750,
    )
    encoder = VoxtralRealtimeEncoder(config).eval()
    export(
        VoxtralRealtimeAudioTransformer(encoder),
        VoxtralRealtimeAudioTransformerOpenVINOConfig(config),
        tmp_path / "openvino_model.xml",
        stateful=True,
        library_name="transformers",
    )
    model = ov.Core().read_model(str(tmp_path / "openvino_model.xml"))
    sdpa = [node for node in model.get_ops() if node.get_type_name() == "ScaledDotProductAttention"]
    assert len(sdpa) == config.num_hidden_layers
    assert all(len(node.input_values()) >= 4 for node in sdpa)
    paged_attention_transformation(model)
    paged = [node for node in model.get_ops() if node.get_type_name() == "PagedAttentionExtension"]
    assert len(paged) == config.num_hidden_layers
    assert not any(node.get_type_name() in ("ReadValue", "Assign") for node in model.get_ops())
    for node in paged:
        window = node.input_value(10).get_node()
        while window.get_type_name() in ("Convert", "Squeeze", "Unsqueeze", "Reshape"):
            window = window.input_value(0).get_node()
        assert window.get_type_name() == "Constant"
        assert int(window.get_data().flatten()[0]) == config.sliding_window


def test_audio_transformer_positions_and_padding(tmp_path, voxtral_model):
    encoder = voxtral_model.model.audio_tower
    config = encoder.config
    export(
        VoxtralRealtimeAudioTransformer(encoder),
        VoxtralRealtimeAudioTransformerOpenVINOConfig(config),
        tmp_path / "openvino_model.xml",
        stateful=True,
        library_name="transformers",
    )
    request = (
        ov.Core()
        .compile_model(str(tmp_path / "openvino_model.xml"), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        .create_infer_request()
    )
    cache = DynamicCache()
    for positions, mask in (
        ([[10, 11, 12, 13], [30, 31, 32, 33]], [[0, 1, 1, 1], [1, 1, 1, 1]]),
        ([[14, 17], [34, 39]], [[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
    ):
        inputs = torch.randn(2, len(positions[0]), config.hidden_size)
        position_ids = torch.tensor(positions, dtype=torch.int64)
        attention_mask = torch.tensor(mask, dtype=torch.int64)
        with torch.no_grad():
            expected = encoder(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state.numpy()
        actual = request.infer(
            {
                "inputs_embeds": inputs.numpy(),
                "attention_mask": attention_mask.numpy(),
                "position_ids": position_ids.numpy(),
                "beam_idx": np.arange(2, dtype=np.int32),
            }
        )["last_hidden_state"]
        valid_queries = attention_mask[:, -inputs.shape[1] :].numpy().astype(bool)
        np.testing.assert_allclose(actual[valid_queries], expected[valid_queries], rtol=1e-2, atol=1e-3)


def test_paged_mask_retains_dflash_block_attention():
    full, sliding = _paged_attention_masks(
        torch.tensor([[0, 1, 1, 1, 1]]),
        torch.tensor([[0, 1, 1]]),
        sliding_window=2,
        dtype=torch.float32,
    )
    expected_full = torch.tensor(
        [[False, True, True, False, False], [False, True, True, True, True], [False, True, True, True, True]]
    )
    expected_sliding = torch.tensor(
        [[False, True, True, False, False], [False, False, True, True, True], [False, False, False, True, True]]
    )
    torch.testing.assert_close(full[0, 0] == 0, expected_full)
    torch.testing.assert_close(sliding[0, 0] == 0, expected_sliding)


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
    assert set(paths) == expected_names | {
        "audio_conv/openvino_model.xml",
        "audio_transformer/openvino_model.xml",
        "audio_adapter/openvino_model.xml",
    }
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
