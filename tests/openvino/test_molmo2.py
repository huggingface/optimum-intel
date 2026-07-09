# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import gc
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from optimum.intel.openvino import OVModelForVisualCausalLM


MODEL_ID = "allenai/MolmoWeb-4B"


class TestMolmo2Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = OVModelForVisualCausalLM.from_pretrained(
            MODEL_ID,
            export=True,
            trust_remote_code=True,
            device="CPU",
        )
        cls.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.processor
        gc.collect()

    def test_export_produces_correct_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            files = os.listdir(tmpdir)
            self.assertIn("openvino_language_model.xml", files)
            self.assertIn("openvino_language_model.bin", files)
            self.assertIn("openvino_text_embeddings_model.xml", files)
            self.assertIn("openvino_text_embeddings_model.bin", files)
            self.assertIn("openvino_vision_embeddings_model.xml", files)
            self.assertIn("openvino_vision_embeddings_model.bin", files)
            self.assertIn("config.json", files)

    def test_model_class_mapping(self):
        from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING, _OVMolmo2ForCausalLM

        self.assertIn("molmo2", MODEL_TYPE_TO_CLS_MAPPING)
        self.assertIsInstance(self.model, _OVMolmo2ForCausalLM)

    def test_inference_with_image(self):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = self.model.preprocess_inputs(
            text="Describe this image", image=img, processor=self.processor
        )

        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_token_pooling", inputs)
        self.assertIn("attention_mask", inputs)

        output = self.model.generate(**inputs, max_new_tokens=5)
        self.assertEqual(output.ndim, 2)
        self.assertGreater(output.shape[1], inputs["input_ids"].shape[1])

    def test_inference_text_only(self):
        inputs = self.model.preprocess_inputs(
            text="Hello, world!", image=None, processor=self.processor
        )
        output = self.model.generate(**inputs, max_new_tokens=5)
        self.assertEqual(output.ndim, 2)
        self.assertGreater(output.shape[1], inputs["input_ids"].shape[1])

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save_pretrained(tmpdir)
            reloaded = OVModelForVisualCausalLM.from_pretrained(
                tmpdir, trust_remote_code=True, device="CPU"
            )

            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            inputs = self.model.preprocess_inputs(
                text="What is this?", image=img, processor=self.processor
            )

            torch.manual_seed(42)
            output_original = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
            torch.manual_seed(42)
            output_reloaded = reloaded.generate(**inputs, max_new_tokens=5, do_sample=False)

            self.assertTrue(torch.equal(output_original, output_reloaded))
            del reloaded
            gc.collect()


if __name__ == "__main__":
    unittest.main()
