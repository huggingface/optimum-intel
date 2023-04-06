#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

import unittest

import torch
from parameterized import parameterized

# TODO : add more tasks
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from optimum.intel import inference_mode as ipex_inference_mode


MODEL_NAMES = {
    "bert": "hf-internal-testing/tiny-random-bert",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "roberta": "hf-internal-testing/tiny-random-roberta",
}

_TASK_TO_AUTOMODELS = {
    "text-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
}


class IPEXIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_classification_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        for task, auto_model_class in _TASK_TO_AUTOMODELS.items():
            model = auto_model_class.from_pretrained(model_id)
            pipe = pipeline(task, model=model, tokenizer=tokenizer)

            with torch.inference_mode():
                outputs = pipe(inputs)
            with ipex_inference_mode(pipe) as ipex_pipe:
                outputs_ipex = ipex_pipe(inputs)

            self.assertEqual(outputs[0]["score"], outputs_ipex[0]["score"])
