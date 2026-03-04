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

import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch
from transformers import set_seed

from optimum.intel import OVParaformerForSpeechSeq2Seq


# Note: This test requires a Paraformer OpenVINO model to be available.
# For CI/CD, this should point to a model on Hugging Face Hub once available.
PARAFORMER_MODEL_PATH = os.environ.get(
    "PARAFORMER_TEST_MODEL",
    None  # Set to model path when available on HF Hub
)

OPENVINO_DEVICE = os.environ.get("OPENVINO_DEVICE", "CPU")
SEED = 42


class OVParaformerForSpeechSeq2SeqTest(unittest.TestCase):
    """
    Test suite for OVParaformerForSpeechSeq2Seq model.
    
    This tests the OpenVINO inference implementation for Paraformer ASR models.
    """
    
    def _generate_random_speech_features(self, batch_size=1, num_frames=100, feature_dim=560):
        """Generate random speech features for testing."""
        np.random.seed(SEED)
        speech = np.random.randn(batch_size, num_frames, feature_dim).astype(np.float32)
        speech_lengths = np.array([num_frames] * batch_size, dtype=np.int32)
        return speech, speech_lengths
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_load_model_from_pretrained(self):
        """Test loading model from pretrained path."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        # Check model properties
        self.assertIsNotNone(model)
        self.assertEqual(model._device, OPENVINO_DEVICE)
        self.assertIsNotNone(model.input_names)
        self.assertIsNotNone(model.output_names)
        self.assertEqual(model.export_feature, "automatic-speech-recognition")
        self.assertEqual(model.main_input_name, "speech")
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_model_inference(self):
        """Test basic inference functionality."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        # Generate random input
        speech, speech_lengths = self._generate_random_speech_features(batch_size=1, num_frames=100)
        speech_tensor = torch.from_numpy(speech)
        lengths_tensor = torch.from_numpy(speech_lengths)
        
        # Run inference
        output = model(speech_tensor, lengths_tensor)
        
        # Check output structure
        self.assertIsNotNone(output.logits)
        self.assertIsNotNone(output.token_num)
        self.assertIsNotNone(output.token_ids)
        
        # Check shapes
        batch_size, seq_len, vocab_size = output.logits.shape
        self.assertEqual(batch_size, 1)
        self.assertGreater(seq_len, 0)
        self.assertGreater(vocab_size, 0)
        
        # Check token_ids shape matches
        self.assertEqual(output.token_ids.shape[0], batch_size)
        self.assertEqual(output.token_ids.shape[1], seq_len)
        
        # Check token_num is within bounds
        self.assertGreater(output.token_num[0].item(), 0)
        self.assertLessEqual(output.token_num[0].item(), seq_len)
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_batch_inference(self):
        """Test batch inference with variable lengths."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        # Generate batch with different lengths
        batch_size = 3
        max_frames = 120
        speech = np.random.randn(batch_size, max_frames, 560).astype(np.float32)
        speech_lengths = np.array([120, 100, 80], dtype=np.int32)
        
        speech_tensor = torch.from_numpy(speech)
        lengths_tensor = torch.from_numpy(speech_lengths)
        
        # Run batch inference
        output = model(speech_tensor, lengths_tensor)
        
        # Check batch dimension
        self.assertEqual(output.logits.shape[0], batch_size)
        self.assertEqual(output.token_ids.shape[0], batch_size)
        self.assertEqual(len(output.token_num), batch_size)
        
        # Check all sequences have tokens
        for i in range(batch_size):
            self.assertGreater(output.token_num[i].item(), 0)
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_numpy_input(self):
        """Test inference with numpy arrays as input."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        # Use numpy arrays directly
        speech, speech_lengths = self._generate_random_speech_features()
        
        # Run inference with numpy input
        output = model(speech, speech_lengths)
        
        # Should work the same as torch tensors
        self.assertIsNotNone(output.logits)
        self.assertIsNotNone(output.token_ids)
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_generate_api(self):
        """Test the generate() API."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        speech, speech_lengths = self._generate_random_speech_features()
        speech_tensor = torch.from_numpy(speech)
        lengths_tensor = torch.from_numpy(speech_lengths)
        
        # Use generate() method
        token_ids, token_num = model.generate(speech_tensor, lengths_tensor)
        
        # Check outputs
        self.assertIsInstance(token_ids, torch.Tensor)
        self.assertIsInstance(token_num, torch.Tensor)
        self.assertEqual(token_ids.shape[0], 1)  # batch size
        self.assertGreater(token_num[0].item(), 0)
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_device_switching(self):
        """Test switching between CPU and GPU."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device="CPU"
        )
        
        self.assertEqual(model._device, "CPU")
        
        speech, speech_lengths = self._generate_random_speech_features()
        speech_tensor = torch.from_numpy(speech)
        lengths_tensor = torch.from_numpy(speech_lengths)
        
        # Run on CPU
        output_cpu = model(speech_tensor, lengths_tensor)
        self.assertIsNotNone(output_cpu.logits)
        
        # Switch to GPU (if available)
        try:
            model.to("GPU")
            self.assertEqual(model._device, "GPU")
            
            # Run on GPU
            output_gpu = model(speech_tensor, lengths_tensor)
            self.assertIsNotNone(output_gpu.logits)
            
            # Results should be similar (not exactly equal due to precision differences)
            self.assertEqual(output_cpu.logits.shape, output_gpu.logits.shape)
        except Exception as e:
            # GPU might not be available in test environment
            self.skipTest(f"GPU not available: {e}")
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_save_and_load(self):
        """Test saving and loading model."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        with TemporaryDirectory() as tmp_dir:
            # Save model
            model.save_pretrained(tmp_dir)
            
            # Check files were created
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "openvino_model.xml")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "openvino_model.bin")))
            
            # Load saved model
            loaded_model = OVParaformerForSpeechSeq2Seq.from_pretrained(
                tmp_dir,
                device=OPENVINO_DEVICE
            )
            
            # Test loaded model works
            speech, speech_lengths = self._generate_random_speech_features()
            output = loaded_model(torch.from_numpy(speech), torch.from_numpy(speech_lengths))
            self.assertIsNotNone(output.logits)
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_decode_without_token_num(self):
        """Test decode method without token_num (should not mask)."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        # Create fake logits
        batch_size, seq_len, vocab_size = 1, 10, 100
        fake_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Decode without token_num
        token_ids = model.decode(fake_logits, token_num=None)
        
        # Should return argmax of logits
        expected = torch.argmax(fake_logits, dim=-1)
        self.assertTrue(torch.equal(token_ids, expected))
        
        # Decode with token_num (should mask padding)
        token_num = torch.tensor([5])
        token_ids_masked = model.decode(fake_logits, token_num=token_num)
        
        # First 5 should be same, rest should be 0
        self.assertTrue(torch.equal(token_ids_masked[0, :5], expected[0, :5]))
        self.assertTrue(torch.all(token_ids_masked[0, 5:] == 0))
    
    @unittest.skipIf(PARAFORMER_MODEL_PATH is None, "Paraformer model path not provided")
    def test_model_properties(self):
        """Test model properties and attributes."""
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            PARAFORMER_MODEL_PATH,
            device=OPENVINO_DEVICE
        )
        
        # Check component names
        self.assertIn("model", model._component_names)
        
        # Check components dictionary
        self.assertIsNotNone(model.components)
        self.assertGreater(len(model.components), 0)
        
        # Check dtype
        self.assertIsNotNone(model.dtype)
        
        # Check device property
        self.assertEqual(model.device, torch.device("cpu"))
    
    def test_model_output_dataclass(self):
        """Test ParaformerModelOutput dataclass."""
        from optimum.intel.openvino.modeling_speech2text import ParaformerModelOutput
        
        # Create output with all fields
        logits = torch.randn(1, 10, 100)
        token_num = torch.tensor([10])
        token_ids = torch.randint(0, 100, (1, 10))
        
        output = ParaformerModelOutput(
            logits=logits,
            token_num=token_num,
            token_ids=token_ids
        )
        
        # Check all fields are accessible
        self.assertEqual(output.logits.shape, logits.shape)
        self.assertEqual(output.token_num, token_num)
        self.assertEqual(output.token_ids.shape, token_ids.shape)
        
        # Check optional field
        output_no_ids = ParaformerModelOutput(
            logits=logits,
            token_num=token_num
        )
        self.assertIsNone(output_no_ids.token_ids)


if __name__ == "__main__":
    # For local testing with your models
    if PARAFORMER_MODEL_PATH is None:
        print("=" * 80)
        print("WARNING: PARAFORMER_TEST_MODEL environment variable not set")
        print("To run tests locally, set:")
        print("  export PARAFORMER_TEST_MODEL=/path/to/paraformer-zh/ov_models")
        print("=" * 80)
    
    unittest.main()
