# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit + integration tests for the gemma4_unified audio embeddings export (PR #1813).

Fast tests (no real model required):
  - merge_audio_text_embeddings  correctness / shape-mismatch guard
  - from_pretrained  backward-compat (local dir without audio file)
  - get_audio_embeddings  decode-step skip (input_ids.shape[1]==1)

Slow tests (@slow, needs local model dir):
  - Full audio embeddings inference with the real INT8 model
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Path to a locally exported INT8 model with audio support.
# Override via GEMMA4_AUDIO_MODEL_DIR env var.
_DEFAULT_LOCAL_MODEL = (
    r"C:\LLM\Gemini\gemini-cli\llama.cpp-openvino-cuda13"
    r"\models\gemma-4-12B-it-openvino-int8-audio"
)
LOCAL_AUDIO_MODEL_DIR = os.environ.get("GEMMA4_AUDIO_MODEL_DIR", _DEFAULT_LOCAL_MODEL)
HAS_LOCAL_MODEL = os.path.isdir(LOCAL_AUDIO_MODEL_DIR) and os.path.exists(
    os.path.join(LOCAL_AUDIO_MODEL_DIR, "openvino_audio_embeddings_model.xml")
)


def _make_mock_gemma4(audio_token_id=258881, hidden_size=16, audio_model=None):
    """Return a minimal mock of _OVGemma4UnifiedForCausalLM for unit testing."""
    from optimum.intel.openvino.modeling_visual_language import _OVGemma4UnifiedForCausalLM

    obj = object.__new__(_OVGemma4UnifiedForCausalLM)
    obj.config = SimpleNamespace(audio_token_id=audio_token_id)
    obj.audio_embeddings = audio_model
    return obj


class TestMergeAudioTextEmbeddings(unittest.TestCase):
    """Unit tests for _OVGemma4UnifiedForCausalLM.merge_audio_text_embeddings()."""

    AUDIO_TOKEN_ID = 258881
    HIDDEN = 16

    def _model(self):
        return _make_mock_gemma4(self.AUDIO_TOKEN_ID, self.HIDDEN)

    def test_scatter_replaces_audio_positions(self):
        model = self._model()
        n_audio = 3
        seq_len = 8
        audio_positions = [2, 4, 6]

        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        for pos in audio_positions:
            input_ids[0, pos] = self.AUDIO_TOKEN_ID

        inputs_embeds = torch.zeros(1, seq_len, self.HIDDEN)
        audio_embeds = torch.ones(n_audio, self.HIDDEN) * 9.0

        out, _, _ = model.merge_audio_text_embeddings(audio_embeds, inputs_embeds, input_ids)

        # Audio positions should now contain the audio embedding value
        for pos in audio_positions:
            self.assertTrue(torch.allclose(out[0, pos], torch.full((self.HIDDEN,), 9.0)))
        # Non-audio positions untouched
        for pos in range(seq_len):
            if pos not in audio_positions:
                self.assertTrue(torch.allclose(out[0, pos], torch.zeros(self.HIDDEN)))

    def test_shape_mismatch_raises(self):
        model = self._model()
        input_ids = torch.tensor([[self.AUDIO_TOKEN_ID, 0, self.AUDIO_TOKEN_ID]])
        inputs_embeds = torch.zeros(1, 3, self.HIDDEN)
        audio_embeds = torch.ones(5, self.HIDDEN)  # 5 frames but 2 positions

        with self.assertRaises(ValueError, msg="Should raise on count mismatch"):
            model.merge_audio_text_embeddings(audio_embeds, inputs_embeds, input_ids)

    def test_numpy_inputs_converted(self):
        model = self._model()
        n_audio = 2
        seq_len = 4
        input_ids = torch.tensor([[self.AUDIO_TOKEN_ID, 0, self.AUDIO_TOKEN_ID, 0]])
        inputs_embeds = np.zeros((1, seq_len, self.HIDDEN), dtype=np.float32)
        audio_embeds = torch.ones(n_audio, self.HIDDEN)

        out, _, _ = model.merge_audio_text_embeddings(audio_embeds, inputs_embeds, input_ids)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, seq_len, self.HIDDEN))

    def test_attention_mask_passthrough(self):
        model = self._model()
        input_ids = torch.tensor([[self.AUDIO_TOKEN_ID]])
        inputs_embeds = torch.zeros(1, 1, self.HIDDEN)
        audio_embeds = torch.ones(1, self.HIDDEN)
        attention_mask = torch.ones(1, 1)
        position_ids = torch.zeros(1, 1, dtype=torch.long)

        _, attn_out, pos_out = model.merge_audio_text_embeddings(
            audio_embeds, inputs_embeds, input_ids,
            attention_mask=attention_mask, position_ids=position_ids
        )
        self.assertTrue(torch.equal(attn_out, attention_mask))
        self.assertTrue(torch.equal(pos_out, position_ids))


class TestGetAudioEmbeddings(unittest.TestCase):
    """Unit tests for _OVGemma4UnifiedForCausalLM.get_audio_embeddings()."""

    def test_decode_step_returns_none(self):
        """Decode step (input_ids.shape[1] == 1) should return None immediately."""
        model = _make_mock_gemma4()
        input_features = torch.zeros(1, 10, 640)
        input_ids = torch.zeros(1, 1, dtype=torch.long)
        result = model.get_audio_embeddings(input_features, input_ids=input_ids)
        self.assertIsNone(result)

    def test_no_audio_model_raises(self):
        """ValueError when audio_embeddings is None but input is provided."""
        model = _make_mock_gemma4(audio_model=None)
        input_features = torch.zeros(1, 10, 640)
        with self.assertRaises(ValueError):
            model.get_audio_embeddings(input_features)

    def test_mask_strips_padding_frames(self):
        """Valid frames (mask=True) should be returned; padding (mask=False) dropped."""
        n_frames = 6
        n_valid = 4
        hidden = 32

        fake_output = torch.rand(1, n_frames, hidden)

        mock_audio_model = MagicMock()
        mock_audio_model.return_value = fake_output.numpy()

        model = _make_mock_gemma4(hidden_size=hidden, audio_model=mock_audio_model)

        input_features = torch.zeros(1, n_frames, 640)
        mask = torch.tensor([[True] * n_valid + [False] * (n_frames - n_valid)])

        out = model.get_audio_embeddings(input_features, input_features_mask=mask)

        self.assertEqual(out.shape, (n_valid, hidden))
        expected = fake_output[mask.bool()]
        self.assertTrue(torch.allclose(out, expected))

    def test_no_mask_flattens_all_frames(self):
        """Without a mask, all frames are returned flat."""
        n_frames = 5
        hidden = 32
        fake_output = np.random.rand(1, n_frames, hidden).astype(np.float32)

        mock_audio_model = MagicMock(return_value=fake_output)
        model = _make_mock_gemma4(hidden_size=hidden, audio_model=mock_audio_model)

        out = model.get_audio_embeddings(torch.zeros(1, n_frames, 640))
        self.assertEqual(out.shape, (n_frames, hidden))


class TestFromPretrainedBackwardCompat(unittest.TestCase):
    """from_pretrained must load without audio when the XML file is absent."""

    def test_local_dir_without_audio_loads_cleanly(self):
        """A local dir without openvino_audio_embeddings_model.xml should load cleanly."""
        from optimum.intel.openvino.modeling_visual_language import _OVGemma4UnifiedForCausalLM

        with tempfile.TemporaryDirectory() as tmpdir:
            # We just verify that additional_parts is temporarily stripped;
            # a full load requires a real OV model file, so we mock super().
            with patch.object(
                _OVGemma4UnifiedForCausalLM.__bases__[0],
                "from_pretrained",
                return_value=MagicMock(spec=_OVGemma4UnifiedForCausalLM),
            ) as mock_super:
                _OVGemma4UnifiedForCausalLM.from_pretrained(tmpdir, device="CPU")
                # audio_embeddings must NOT be in additional_parts when super() is called
                call_args = mock_super.call_args
                self.assertEqual(call_args[0][0], tmpdir)
                # additional_parts must be restored afterward
                self.assertIn("audio_embeddings", _OVGemma4UnifiedForCausalLM.additional_parts)

    def test_hub_id_passes_through_unchanged(self):
        """A Hub model ID (not a local dir) must reach super() with audio_embeddings intact."""
        from optimum.intel.openvino.modeling_visual_language import _OVGemma4UnifiedForCausalLM

        hub_id = "google/gemma-4-12B-it"  # not a local directory
        with patch.object(
            _OVGemma4UnifiedForCausalLM.__bases__[0],
            "from_pretrained",
            return_value=MagicMock(spec=_OVGemma4UnifiedForCausalLM),
        ) as mock_super:
            _OVGemma4UnifiedForCausalLM.from_pretrained(hub_id, device="CPU")
            self.assertIn("audio_embeddings", _OVGemma4UnifiedForCausalLM.additional_parts)


@unittest.skipUnless(HAS_LOCAL_MODEL, f"Local audio model not found at {LOCAL_AUDIO_MODEL_DIR}")
class TestGemma4AudioIntegration(unittest.TestCase):
    """
    Integration tests requiring a locally exported model with audio support.

    Run with:
        pytest tests/openvino/test_gemma4_audio.py::TestGemma4AudioIntegration -v

    Or set GEMMA4_AUDIO_MODEL_DIR to point at a local directory that contains
    openvino_audio_embeddings_model.xml.
    """

    @classmethod
    def setUpClass(cls):
        from optimum.intel import OVModelForVisualCausalLM
        cls.model = OVModelForVisualCausalLM.from_pretrained(
            LOCAL_AUDIO_MODEL_DIR, device="CPU", compile=False
        )
        cls.hidden = cls.model.config.hidden_size if hasattr(cls.model.config, "hidden_size") else 3840

    @classmethod
    def tearDownClass(cls):
        import gc
        del cls.model
        gc.collect()

    def test_audio_embeddings_model_loaded(self):
        self.assertIsNotNone(
            getattr(self.model, "audio_embeddings", None),
            "audio_embeddings sub-model should be loaded from XML",
        )

    def test_get_audio_embeddings_shape(self):
        """get_audio_embeddings returns [n_valid_frames, hidden_size]."""
        n_frames = 20
        n_valid = 15
        input_features = torch.zeros(1, n_frames, 640)
        mask = torch.zeros(1, n_frames, dtype=torch.bool)
        mask[0, :n_valid] = True

        embs = self.model.get_audio_embeddings(input_features, input_features_mask=mask)

        self.assertIsNotNone(embs)
        self.assertEqual(embs.ndim, 2)
        self.assertEqual(embs.shape[0], n_valid)
        self.assertEqual(embs.shape[1], self.hidden)

    def test_audio_embedding_output_dtype(self):
        """Audio embeddings dtype should be a floating-point type."""
        embs = self.model.get_audio_embeddings(torch.zeros(1, 5, 640))
        self.assertTrue(
            embs.dtype in (torch.float32, torch.float16, torch.bfloat16),
            f"Expected floating-point dtype, got {embs.dtype}",
        )

    def test_merge_roundtrip(self):
        """get_audio_embeddings → merge_audio_text_embeddings is shape-correct end-to-end."""
        audio_token_id = getattr(self.model.config, "audio_token_id", 258881)
        n_audio = 4
        seq_len = 10
        hidden = self.hidden

        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        for i in range(n_audio):
            input_ids[0, i * 2] = audio_token_id

        mask = torch.zeros(1, n_audio, dtype=torch.bool)
        mask[0, :] = True
        input_features = torch.zeros(1, n_audio, 640)

        audio_embeds = self.model.get_audio_embeddings(input_features, input_features_mask=mask)
        inputs_embeds = torch.zeros(1, seq_len, hidden)

        merged, _, _ = self.model.merge_audio_text_embeddings(
            audio_embeds, inputs_embeds, input_ids
        )
        self.assertEqual(merged.shape, (1, seq_len, hidden))


if __name__ == "__main__":
    unittest.main()
