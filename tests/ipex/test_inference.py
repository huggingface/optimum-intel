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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
from utils_tests import MODEL_NAMES

from optimum.intel import inference_mode as ipex_inference_mode
from optimum.intel.ipex.modeling_base import IPEXModel


_CLASSIFICATION_TASK_TO_AUTOMODELS = {
    "text-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
}


class IPEXClassificationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        for task, auto_model_class in _CLASSIFICATION_TASK_TO_AUTOMODELS.items():
            model = auto_model_class.from_pretrained(model_id, torch_dtype=torch.float32)
            pipe = pipeline(task, model=model, tokenizer=tokenizer)

            with torch.inference_mode():
                outputs = pipe(inputs)
            with ipex_inference_mode(pipe, dtype=model.config.torch_dtype, verbose=False, jit=True) as ipex_pipe:
                outputs_ipex = ipex_pipe(inputs)
            self.assertTrue(isinstance(ipex_pipe.model._optimized.model, torch.jit.RecursiveScriptModule))
            self.assertEqual(outputs[0]["score"], outputs_ipex[0]["score"])


class IPEXQuestionAnsweringTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)

        with torch.inference_mode():
            outputs = pipe(question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris.")
        with ipex_inference_mode(pipe, dtype=model.config.torch_dtype, verbose=False, jit=True) as ipex_pipe:
            outputs_ipex = ipex_pipe(
                question="Where was HuggingFace founded ?", context="HuggingFace was founded in Paris."
            )
        self.assertTrue(isinstance(ipex_pipe.model._optimized.model, torch.jit.RecursiveScriptModule))
        self.assertEqual(outputs["start"], outputs_ipex["start"])
        self.assertEqual(outputs["end"], outputs_ipex["end"])


class IPEXTextGenerationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bloom",
        "gptj",
        "gpt2",
        "gpt_neo",
        "gpt_bigcode",
        "llama",
        "llama2",
        "opt",
        "mpt",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, return_dict=False)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a simple input"
        text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        with torch.inference_mode():
            output = text_generator(inputs)
        with ipex_inference_mode(
            text_generator, dtype=model.config.torch_dtype, verbose=False, jit=True
        ) as ipex_text_generator:
            output_ipex = ipex_text_generator(inputs)
        self.assertTrue(isinstance(ipex_text_generator.model._optimized, IPEXModel))
        self.assertTrue(isinstance(ipex_text_generator.model._optimized.model, torch.jit.RecursiveScriptModule))
        self.assertEqual(output[0]["generated_text"], output_ipex[0]["generated_text"])
