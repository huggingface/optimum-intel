#  Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""
Tests for audit finding fixes:
- Pipeline dispatch routing for qwen3_omni_moe (accelerator_utils.py)
- OVTalkerDecoder null guard on _infer_request
- Batch size validation in _run_talker_generation
- OVModelForOmni __init__ attribute initialization
- Per-component device/ov_config override plumbing
- OVCode2Wav chunk stitching
- Projection / audio LRU cache helpers
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from optimum.intel.openvino import OVModelForOmni
from optimum.intel.pipelines.accelerator_utils import OV_TASKS_MAPPING, get_openvino_model_class


def _bind_run_talker_generation(model):
    from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

    return _OVQwen3OmniMoeForCausalLM._run_talker_generation.__get__(model)


class TestQwen3OmniMoeDispatch(unittest.TestCase):
    """Verify get_openvino_model_class routes qwen3_omni_moe to OVModelForOmni for all supported tasks."""

    def _make_omni_config(self):
        config = PretrainedConfig()
        config.model_type = "qwen3_omni_moe"
        config.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
        return config

    def _make_non_omni_config(self, model_type="llava"):
        config = PretrainedConfig()
        config.model_type = model_type
        config.architectures = ["LlavaForConditionalGeneration"]
        return config

    def test_qwen3_omni_moe_dispatch_asr(self):
        config = self._make_omni_config()
        cls = get_openvino_model_class("automatic-speech-recognition", config=config)
        assert cls is OVModelForOmni, f"Expected OVModelForOmni, got {cls}"

    def test_qwen3_omni_moe_dispatch_image_text(self):
        config = self._make_omni_config()
        cls = get_openvino_model_class("image-text-to-text", config=config)
        assert cls is OVModelForOmni, f"Expected OVModelForOmni, got {cls}"

    def test_qwen3_omni_moe_dispatch_text_audio(self):
        config = self._make_omni_config()
        cls = get_openvino_model_class("text-to-audio", config=config)
        assert cls is OVModelForOmni, f"Expected OVModelForOmni, got {cls}"

    def test_asr_non_omni_does_not_get_omni(self):
        config = self._make_non_omni_config("whisper")
        config.architectures = ["WhisperForConditionalGeneration"]
        cls = get_openvino_model_class("automatic-speech-recognition", config=config)
        assert cls is not OVModelForOmni

    def test_asr_ctc_model_dispatched_correctly(self):
        config = PretrainedConfig()
        config.model_type = "wav2vec2"
        config.architectures = ["Wav2Vec2ForCTC"]
        cls = get_openvino_model_class("automatic-speech-recognition", config=config)
        assert cls is OV_TASKS_MAPPING["automatic-speech-recognition"][0]

    def test_image_text_non_omni_dispatched_to_first(self):
        config = self._make_non_omni_config("llava")
        cls = get_openvino_model_class("image-text-to-text", config=config)
        first_class = OV_TASKS_MAPPING["image-text-to-text"][0]
        assert cls is first_class

    def test_text_audio_non_omni_dispatched_to_first(self):
        config = self._make_non_omni_config("speecht5")
        cls = get_openvino_model_class("text-to-audio", config=config)
        first_class = OV_TASKS_MAPPING["text-to-audio"][0]
        assert cls is first_class

    def test_omni_present_in_all_expected_task_tuples(self):
        expected_tasks = {"automatic-speech-recognition", "image-text-to-text", "text-to-audio"}
        for task in expected_tasks:
            assert OVModelForOmni in OV_TASKS_MAPPING[task], f"OVModelForOmni missing from OV_TASKS_MAPPING['{task}']"

    def test_unsupported_task_raises(self):
        config = self._make_omni_config()
        with pytest.raises(KeyError, match="not supported by OpenVINO"):
            get_openvino_model_class("nonexistent-task", config=config)

    def test_text_generation_unaffected_by_omni_config(self):
        config = self._make_omni_config()
        cls = get_openvino_model_class("text-generation", config=config)
        assert cls is OV_TASKS_MAPPING["text-generation"][0]

    def test_translation_prefix_normalization(self):
        cls = get_openvino_model_class("translation_en_to_de")
        assert cls is OV_TASKS_MAPPING["translation"][0]


class TestTalkerNullGuard(unittest.TestCase):
    """Verify OVTalkerDecoder.forward raises ValueError when _infer_request is None."""

    def _build_talker_stub(self):
        """Build a minimal OVTalkerDecoder-like object that exercises the null guard."""
        from optimum.intel.openvino.modeling_visual_language import OVTalkerDecoder

        mock_model = MagicMock()
        mock_model.inputs = [MagicMock()]
        mock_model.inputs[0].get_any_name.return_value = "inputs_embeds"
        mock_model.outputs = [
            MagicMock(),
            MagicMock(),
        ]
        mock_model.outputs[0].get_any_name.return_value = "logits"
        mock_model.outputs[1].get_any_name.return_value = "hidden_states"
        for inp in mock_model.inputs:
            inp.get_element_type.return_value.get_type_name.return_value = "f32"
        for out in mock_model.outputs:
            out.get_element_type.return_value.get_type_name.return_value = "f32"

        parent = MagicMock()
        parent._device = "CPU"
        parent.ov_config = {}
        parent._compile_only = False
        parent.model_save_dir = "/tmp"
        parent.config = PretrainedConfig()
        parent.device = torch.device("cpu")

        talker = OVTalkerDecoder(mock_model, parent)
        # Override compile to be a no-op so _infer_request stays None
        talker.compile = lambda: None
        return talker

    def test_talker_null_guard_raises(self):
        talker = self._build_talker_stub()
        assert talker._infer_request is None

        inputs_embeds = torch.randn(1, 5, 64)
        with pytest.raises(ValueError, match="Talker model not loaded or compilation failed"):
            talker.forward(inputs_embeds=inputs_embeds)

    def test_talker_null_guard_not_triggered_when_request_exists(self):
        talker = self._build_talker_stub()

        mock_request = MagicMock()
        mock_logits = MagicMock()
        mock_logits.data = np.zeros((1, 5, 10), dtype=np.float32)
        mock_hidden = MagicMock()
        mock_hidden.data = np.zeros((1, 5, 64), dtype=np.float32)

        def get_tensor(name):
            if name == "logits":
                return mock_logits
            return mock_hidden

        mock_request.get_tensor = get_tensor
        talker._infer_request = mock_request

        inputs_embeds = torch.randn(1, 5, 64)
        logits, hidden = talker.forward(inputs_embeds=inputs_embeds)
        assert logits.shape == (1, 5, 10)
        assert hidden.shape == (1, 5, 64)


class TestBatchSizeValidation(unittest.TestCase):
    """Verify _run_talker_generation raises ValueError when batch_size > 1."""

    def test_batch_size_2_raises(self):
        model = MagicMock()
        model._run_talker_generation = _bind_run_talker_generation(model)

        talker_input_embeds = torch.randn(2, 10, 64)
        trailing = torch.randn(2, 5, 64)
        pad = torch.randn(1, 1, 64)

        with pytest.raises(ValueError, match="only supports batch_size=1, got 2"):
            model._run_talker_generation(talker_input_embeds, trailing, pad, {})

    def test_batch_size_3_raises(self):
        model = MagicMock()
        model._run_talker_generation = _bind_run_talker_generation(model)

        talker_input_embeds = torch.randn(3, 10, 64)
        trailing = torch.randn(3, 5, 64)
        pad = torch.randn(1, 1, 64)

        with pytest.raises(ValueError, match="only supports batch_size=1, got 3"):
            model._run_talker_generation(talker_input_embeds, trailing, pad, {})

    def test_batch_size_0_raises(self):
        model = MagicMock()
        model._run_talker_generation = _bind_run_talker_generation(model)

        talker_input_embeds = torch.randn(0, 10, 64)
        trailing = torch.randn(0, 5, 64)
        pad = torch.randn(1, 1, 64)

        with pytest.raises(ValueError, match="only supports batch_size=1, got 0"):
            model._run_talker_generation(talker_input_embeds, trailing, pad, {})

    def test_batch_size_1_passes_guard(self):
        model = MagicMock()
        model._run_talker_generation = _bind_run_talker_generation(model)

        talker_config = MagicMock()
        talker_config.codec_eos_token_id = 0
        talker_config.code_predictor_config.num_code_groups = 1
        model.config = MagicMock()
        model.config.talker_config = talker_config

        logits = torch.zeros(1, 10, 256)
        logits[0, -1, 0] = 100.0
        model.talker = MagicMock(return_value=(logits, torch.randn(1, 10, 64)))

        talker_input_embeds = torch.randn(1, 10, 64)
        trailing = torch.randn(1, 5, 64)
        pad = torch.randn(1, 1, 64)

        result = model._run_talker_generation(
            talker_input_embeds, trailing, pad, {"max_new_tokens": 1, "temperature": 0}
        )
        assert result is None or isinstance(result, torch.Tensor)


class TestOVModelForOmniInit(unittest.TestCase):
    """Verify OVModelForOmni.__init__ correctly mirrors _inner attributes and sets GenerationMixin state."""

    def _build_mock_inner(self):
        inner = MagicMock()
        inner.config = PretrainedConfig()
        inner.config.model_type = "qwen3_omni_moe"
        inner.model_save_dir = "/tmp/model"
        inner._device = "CPU"
        inner.ov_config = {"KEY": "VAL"}
        inner._compile_only = False
        inner.use_cache = True
        inner.preprocessors = [MagicMock()]
        inner.generation_config = MagicMock()
        inner.is_dynamic = True
        inner._openvino_config = MagicMock()
        inner.language_model = MagicMock()
        inner.vision_embeddings = MagicMock()
        inner.has_talker = True
        inner.audio_encoder = MagicMock()
        inner.talker = MagicMock()
        inner.talker_text_embeddings = MagicMock()
        inner.talker_projections = MagicMock()
        inner.code_predictor = MagicMock()
        inner.code2wav = MagicMock()
        inner.components = {"language_model": inner.language_model}
        inner._component_names = ["language_model"]
        inner._ov_model_names = ["lm_model"]
        inner.ov_models = {"lm_model": MagicMock()}
        return inner

    def test_config_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.config is inner.config

    def test_model_save_dir_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.model_save_dir is inner.model_save_dir

    def test_device_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model._device is inner._device

    def test_ov_config_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.ov_config is inner.ov_config

    def test_use_cache_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.use_cache is inner.use_cache

    def test_generation_mixin_attributes_set(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model._supports_cache_class is False
        assert model.main_input_name == "input_ids"

    def test_generation_config_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.generation_config is inner.generation_config

    def test_openvino_config_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model._openvino_config is inner._openvino_config

    def test_openvino_config_none_when_inner_lacks_it(self):
        inner = self._build_mock_inner()
        del inner._openvino_config
        model = OVModelForOmni(_inner=inner)
        assert model._openvino_config is None

    def test_inner_stored_via_object_setattr(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model._inner is inner

    def test_property_delegation_language_model(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.language_model is inner.language_model

    def test_property_delegation_vision_embeddings(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.vision_embeddings is inner.vision_embeddings

    def test_property_delegation_has_talker(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.has_talker is inner.has_talker

    def test_property_delegation_audio_encoder(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.audio_encoder is inner.audio_encoder

    def test_property_delegation_talker(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.talker is inner.talker

    def test_property_delegation_code_predictor(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.code_predictor is inner.code_predictor

    def test_property_delegation_code2wav(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.code2wav is inner.code2wav

    def test_property_delegation_components(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.components is inner.components

    def test_generate_delegates_to_inner(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        model.generate("test_arg", key="val")
        inner.generate.assert_called_once_with("test_arg", key="val")

    def test_compile_delegates_to_inner(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        model.compile()
        inner.compile.assert_called_once()

    def test_preprocessors_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.preprocessors is inner.preprocessors

    def test_is_dynamic_mirrored(self):
        inner = self._build_mock_inner()
        model = OVModelForOmni(_inner=inner)
        assert model.is_dynamic is inner.is_dynamic


class TestTalkerConfigGuard(unittest.TestCase):
    """Verify _run_talker_generation raises ValueError when talker_config is missing."""

    def test_missing_talker_config_raises(self):
        model = MagicMock()
        model._run_talker_generation = _bind_run_talker_generation(model)

        config = MagicMock()
        config.talker_config = None
        model.config = config

        talker_input_embeds = torch.randn(1, 10, 64)
        trailing = torch.randn(1, 5, 64)
        pad = torch.randn(1, 1, 64)

        with pytest.raises(ValueError, match="talker_config is required"):
            model._run_talker_generation(talker_input_embeds, trailing, pad, {})


class TestPerComponentDeviceOverrides(unittest.TestCase):
    """Verify _split_component_overrides correctly separates per-component keys from ov_config."""

    def test_device_override_uppercased(self):
        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        dev, _, _ = _OVQwen3OmniMoeForCausalLM._split_component_overrides({"talker.device": "gpu"})
        assert dev == {"talker": "GPU"}

    def test_whitelisted_components_extracted(self):
        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        dev, cfg, base = _OVQwen3OmniMoeForCausalLM._split_component_overrides(
            {
                "audio_encoder.device": "GPU",
                "talker.device": "CPU",
                "talker.ov_config": {"PERFORMANCE_HINT": "LATENCY"},
                "code_predictor.device": "NPU",
                "CACHE_DIR": "/tmp/x",
            }
        )
        assert dev == {"audio_encoder": "GPU", "talker": "CPU", "code_predictor": "NPU"}
        assert cfg == {"talker": {"PERFORMANCE_HINT": "LATENCY"}}
        assert base == {"CACHE_DIR": "/tmp/x"}

    def test_unknown_component_stays_in_base(self):
        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        dev, cfg, base = _OVQwen3OmniMoeForCausalLM._split_component_overrides(
            {"unknown_component.device": "GPU", "PERFORMANCE_HINT": "LATENCY"}
        )
        assert dev == {}
        assert cfg == {}
        assert base == {"unknown_component.device": "GPU", "PERFORMANCE_HINT": "LATENCY"}

    def test_non_dict_ov_config_value_stays_in_base(self):
        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        dev, cfg, base = _OVQwen3OmniMoeForCausalLM._split_component_overrides({"talker.ov_config": "not_a_dict"})
        assert cfg == {}
        assert base == {"talker.ov_config": "not_a_dict"}

    def test_empty_ov_config(self):
        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        dev, cfg, base = _OVQwen3OmniMoeForCausalLM._split_component_overrides({})
        assert dev == {} and cfg == {} and base == {}


class TestCode2WavChunking(unittest.TestCase):
    """Verify OVCode2Wav chunk stitching and guards."""

    def _make_mock_code2wav(self, samples_per_step=40):
        from optimum.intel.openvino.modeling_visual_language import OVCode2Wav

        class Mock(OVCode2Wav):
            def __init__(self, spp):
                self.chunk_size = 25
                self.chunk_overlap = 5
                self._async_infer_requests = []
                self.request = None
                self._spp = spp

            def _run_single(self, chunk_np):
                samples = chunk_np.shape[-1] * self._spp
                return np.ones((1, samples), dtype=np.float32) * float(chunk_np[0, 0, 0])

            def _ensure_infer_pool(self, size):
                return

        return Mock(samples_per_step)

    def test_chunked_produces_2d_waveform(self):
        m = self._make_mock_code2wav()
        codes = torch.ones(1, 16, 60)
        wave = m._run_chunked(codes, 25, 5)
        assert wave.ndim == 2
        assert wave.shape[0] == 1

    def test_chunk_overlap_ge_chunk_size_raises(self):
        m = self._make_mock_code2wav()
        codes = torch.ones(1, 16, 60)
        with pytest.raises(ValueError, match="chunk_overlap"):
            m._run_chunked(codes, 25, 25)
        with pytest.raises(ValueError, match="chunk_overlap"):
            m._run_chunked(codes, 25, 30)

    def test_batch_size_gt_1_raises(self):
        m = self._make_mock_code2wav()
        codes = torch.ones(2, 16, 60)
        with pytest.raises(ValueError, match="batch_size=1"):
            m._run_chunked(codes, 25, 5)

    def test_wrong_ndim_raises(self):
        m = self._make_mock_code2wav()
        codes = torch.ones(16, 60)
        with pytest.raises(ValueError, match=r"\(B, G, T\)"):
            m._run_chunked(codes, 25, 5)

    def test_estimate_samples_per_step(self):
        from optimum.intel.openvino.modeling_visual_language import OVCode2Wav

        assert OVCode2Wav._estimate_samples_per_step(25, 5000) == 200
        assert OVCode2Wav._estimate_samples_per_step(0, 100) == 0
        assert OVCode2Wav._estimate_samples_per_step(10, 0) == 0


class TestCaches(unittest.TestCase):
    """Verify projection + audio LRU helpers on _OVQwen3OmniMoeForCausalLM."""

    def test_lru_get_miss_returns_none(self):
        from collections import OrderedDict

        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        cache = OrderedDict()
        assert _OVQwen3OmniMoeForCausalLM._lru_get(cache, b"x") is None

    def test_lru_put_evicts_lru(self):
        from collections import OrderedDict

        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        cache = OrderedDict()
        _OVQwen3OmniMoeForCausalLM._lru_put(cache, b"a", 1, capacity=2)
        _OVQwen3OmniMoeForCausalLM._lru_put(cache, b"b", 2, capacity=2)
        _OVQwen3OmniMoeForCausalLM._lru_put(cache, b"c", 3, capacity=2)
        assert b"a" not in cache
        assert list(cache.keys()) == [b"b", b"c"]

    def test_lru_get_promotes_to_end(self):
        from collections import OrderedDict

        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        cache = OrderedDict()
        _OVQwen3OmniMoeForCausalLM._lru_put(cache, b"a", 1, capacity=3)
        _OVQwen3OmniMoeForCausalLM._lru_put(cache, b"b", 2, capacity=3)
        assert _OVQwen3OmniMoeForCausalLM._lru_get(cache, b"a") == 1
        # "a" should now be most-recently-used, so "b" evicts first.
        _OVQwen3OmniMoeForCausalLM._lru_put(cache, b"c", 3, capacity=2)
        assert b"b" not in cache
        assert b"a" in cache and b"c" in cache

    def test_tensor_bytes_stable_across_calls(self):
        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        t = torch.arange(6).reshape(2, 3)
        assert _OVQwen3OmniMoeForCausalLM._tensor_bytes(t) == _OVQwen3OmniMoeForCausalLM._tensor_bytes(t)

    def test_tensor_bytes_distinct_for_different_values(self):
        from optimum.intel.openvino.modeling_visual_language import _OVQwen3OmniMoeForCausalLM

        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 4])
        assert _OVQwen3OmniMoeForCausalLM._tensor_bytes(a) != _OVQwen3OmniMoeForCausalLM._tensor_bytes(b)


if __name__ == "__main__":
    unittest.main()
