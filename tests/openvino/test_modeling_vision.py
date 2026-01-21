#  Copyright 2021 The HuggingFace Team. All rights reserved.
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
import os
import unittest

import requests
import torch
from parameterized import parameterized
from PIL import Image
from transformers import PretrainedConfig, set_seed
from transformers.onnx.utils import get_preprocessor
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, SEED

from optimum.intel import OVSamModel
from optimum.intel.openvino.modeling_sam import OVSamPromptEncoder, OVSamVisionEncoder


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OVSamIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["sam"]
    TASK = "feature-extraction"
    IMAGE_URL = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVSamModel.from_pretrained(model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
        processor = get_preprocessor(model_id)

        self.assertIsInstance(ov_model.vision_encoder, OVSamVisionEncoder)
        self.assertIsInstance(ov_model.prompt_encoder_mask_decoder, OVSamPromptEncoder)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        input_points = [[[450, 600]]]
        IMAGE = Image.open(
            requests.get(
                self.IMAGE_URL,
                stream=True,
            ).raw
        ).convert("RGB")
        inputs = processor(IMAGE, input_points=input_points, return_tensors="pt")

        transformers_model = OVSamModel.from_pretrained(model_id, device=OPENVINO_DEVICE)

        # test end-to-end inference
        ov_outputs = ov_model(**inputs)

        self.assertTrue("pred_masks" in ov_outputs)
        self.assertIsInstance(ov_outputs.pred_masks, torch.Tensor)
        self.assertTrue("iou_scores" in ov_outputs)
        self.assertIsInstance(ov_outputs.iou_scores, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.pred_masks, transformers_outputs.pred_masks, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.iou_scores, transformers_outputs.iou_scores, atol=1e-4))

        # test separated image features extraction
        pixel_values = inputs.pop("pixel_values")
        features = transformers_model.get_image_features(pixel_values)
        ov_features = ov_model.get_image_features(pixel_values)
        self.assertTrue(torch.allclose(ov_features, features, atol=1e-4))
        ov_outputs = ov_model(**inputs, image_embeddings=ov_features)
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, image_embeddings=features)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.pred_masks, transformers_outputs.pred_masks, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.iou_scores, transformers_outputs.iou_scores, atol=1e-4))

        del transformers_model
        del ov_model

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_reshape(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVSamModel.from_pretrained(model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
        processor = get_preprocessor(model_id)
        self.assertTrue(ov_model.is_dynamic)
        input_points = [[[450, 600]]]
        IMAGE = Image.open(
            requests.get(
                self.IMAGE_URL,
                stream=True,
            ).raw
        ).convert("RGB")
        inputs = processor(IMAGE, input_points=input_points, return_tensors="pt")
        ov_dyn_outputs = ov_model(**inputs)
        ov_model.reshape(*inputs["input_points"].shape[:-1])
        self.assertFalse(ov_model.is_dynamic)
        self.assertIsNone(ov_model.vision_encoder.request)
        self.assertIsNone(ov_model.prompt_encoder_mask_decoder.request)
        ov_stat_outputs = ov_model(**inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_dyn_outputs.pred_masks, ov_stat_outputs.pred_masks, atol=1e-4))
        self.assertTrue(torch.allclose(ov_dyn_outputs.iou_scores, ov_stat_outputs.iou_scores, atol=1e-4))

        del ov_model
        gc.collect()
