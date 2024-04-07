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

import unittest

import torch
from parameterized import parameterized
from transformers.pipelines import pipeline as transformers_pipeline

from optimum.intel.ipex.modeling_base import IPEXModelForCausalLM
from optimum.intel.pipelines import pipeline as ipex_pipeline


MODEL_NAMES = {
    "bert": "hf-internal-testing/tiny-random-bert",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "bloom": "hf-internal-testing/tiny-random-bloom",
    "gptj": "hf-internal-testing/tiny-random-gptj",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
}


class PipelinesIntegrationTest(unittest.TestCase):
    TEXT_GENERATION_SUPPORTED_ARCHITECTURES = ("bloom", "gptj", "gpt2", "gpt_neo")

    @parameterized.expand(TEXT_GENERATION_SUPPORTED_ARCHITECTURES)
    def test_text_generation_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        inputs = "Describe a real-world application of AI."
        transformers_text_generator = transformers_pipeline("text-generation", model_id)
        ipex_text_generator = ipex_pipeline("text-generation", model_id, accelerator="ipex")
        with torch.inference_mode():
            transformers_output = transformers_text_generator(inputs)
        with torch.inference_mode():
            ipex_output = ipex_text_generator(inputs)
        self.assertTrue(isinstance(ipex_text_generator.model, IPEXModelForCausalLM))
        self.assertTrue(isinstance(ipex_text_generator.model.model, torch.jit.RecursiveScriptModule))
        self.assertEqual(transformers_output[0]["generated_text"], ipex_output[0]["generated_text"])
