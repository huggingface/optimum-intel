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
import unittest
from typing import Dict

import numpy as np

from parameterized import parameterized
from sentence_transformers import SentenceTransformer
from transformers import (
    PretrainedConfig,
    set_seed,
)
from utils_tests import MODEL_NAMES

from optimum.intel import OVModelForSentenceTransformer


SEED = 42

F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}

MODEL_NAMES = {
    "bert": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


class OVModelForSTFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "mpnet",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSentenceTransformer.from_pretrained(model_id, from_transformers=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        st_model = SentenceTransformer(model_id)
        sentences = ["This is an example sentence", "Each sentence is converted"]
        st_embeddings = st_model.encode(sentences)
        ov_embeddings = ov_model.encode(sentences)
        # Compare tensor outputs
        self.assertTrue(np.allclose(ov_embeddings, st_embeddings, atol=1e-4))
        del st_embeddings
        del ov_model
        gc.collect()
