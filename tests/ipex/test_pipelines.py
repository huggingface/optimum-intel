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
from tempfile import TemporaryDirectory

import numpy as np
import torch
from parameterized import parameterized
from transformers import AutoTokenizer
from transformers.pipelines import pipeline as transformers_pipeline
from utils_tests import IS_XPU_AVAILABLE, MODEL_NAMES

from optimum.intel.ipex.modeling_base import (
    IPEXModelForAudioClassification,
    IPEXModelForCausalLM,
    IPEXModelForImageClassification,
    IPEXModelForMaskedLM,
    IPEXModelForQuestionAnswering,
    IPEXModelForSeq2SeqLM,
    IPEXModelForSequenceClassification,
    IPEXModelForTokenClassification,
)
from optimum.intel.pipelines import pipeline as ipex_pipeline


torch.use_deterministic_algorithms(True)


class PipelinesIntegrationTest(unittest.TestCase):
    COMMON_SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        "distilbert",
        "electra",
        "flaubert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
    )
    TEXT_GENERATION_SUPPORTED_ARCHITECTURES = (
        "bart",
        "gpt_bigcode",
        "blenderbot",
        "blenderbot-small",
        "bloom",
        "codegen",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "llama",
        "llama2",
        "mistral",
        "mpt",
        "opt",
    )
    QUESTION_ANSWERING_SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )
    AUDIO_CLASSIFICATION_SUPPORTED_ARCHITECTURES = (
        "unispeech",
        "wav2vec2",
    )
    IMAGE_CLASSIFICATION_SUPPORTED_ARCHITECTURES = (
        "beit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "resnet",
        "vit",
    )
    TEXT2TEXT_GENERATION_SUPPORTED_ARCHITECTURES = ("t5",)

    @parameterized.expand(COMMON_SUPPORTED_ARCHITECTURES)
    def test_token_classification_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        transformers_generator = transformers_pipeline("token-classification", model_id)
        ipex_generator = ipex_pipeline("token-classification", model_id, accelerator="ipex")
        inputs = "Hello I'm Omar and I live in Zürich."
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs)
        self.assertEqual(len(transformers_output), len(ipex_output))
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForTokenClassification))
        for i in range(len(transformers_output)):
            self.assertAlmostEqual(transformers_output[i]["score"], ipex_output[i]["score"], delta=1e-4)

    @parameterized.expand(COMMON_SUPPORTED_ARCHITECTURES)
    def test_sequence_classification_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        transformers_generator = transformers_pipeline("text-classification", model_id)
        ipex_generator = ipex_pipeline("text-classification", model_id, accelerator="ipex")
        inputs = "This restaurant is awesome"
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForSequenceClassification))
        self.assertEqual(transformers_output[0]["label"], ipex_output[0]["label"])
        self.assertAlmostEqual(transformers_output[0]["score"], ipex_output[0]["score"], delta=1e-4)

    @parameterized.expand(COMMON_SUPPORTED_ARCHITECTURES)
    def test_fill_mask_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        inputs = "The Milky Way is a <mask> galaxy."
        transformers_generator = transformers_pipeline("fill-mask", model_id)
        ipex_generator = ipex_pipeline("fill-mask", model_id, accelerator="ipex")
        mask_token = transformers_generator.tokenizer.mask_token
        inputs = inputs.replace("<mask>", mask_token)
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs)
        self.assertEqual(len(transformers_output), len(ipex_output))
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForMaskedLM))
        for i in range(len(transformers_output)):
            self.assertEqual(transformers_output[i]["token"], ipex_output[i]["token"])
            self.assertAlmostEqual(transformers_output[i]["score"], ipex_output[i]["score"], delta=1e-4)

    @parameterized.expand(TEXT_GENERATION_SUPPORTED_ARCHITECTURES)
    def test_text_generation_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        transformers_generator = transformers_pipeline("text-generation", model_id, torch_dtype=dtype)
        ipex_generator = ipex_pipeline("text-generation", model_id, accelerator="ipex", torch_dtype=dtype)
        inputs = "Describe a real-world application of AI."
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs, do_sample=False, max_new_tokens=10)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs, do_sample=False, max_new_tokens=10)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForCausalLM))
        self.assertEqual(transformers_output[0]["generated_text"], ipex_output[0]["generated_text"])

    @parameterized.expand(QUESTION_ANSWERING_SUPPORTED_ARCHITECTURES)
    def test_question_answering_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        transformers_generator = transformers_pipeline("question-answering", model_id)
        ipex_generator = ipex_pipeline("question-answering", model_id, accelerator="ipex")
        question = "How many programming languages does BLOOM support?"
        context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
        with torch.inference_mode():
            transformers_output = transformers_generator(question=question, context=context)
        with torch.inference_mode():
            ipex_output = ipex_generator(question=question, context=context)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForQuestionAnswering))
        self.assertAlmostEqual(transformers_output["score"], ipex_output["score"], delta=1e-4)
        self.assertEqual(transformers_output["start"], ipex_output["start"])
        self.assertEqual(transformers_output["end"], ipex_output["end"])

    @parameterized.expand(AUDIO_CLASSIFICATION_SUPPORTED_ARCHITECTURES)
    def test_audio_classification_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        transformers_generator = transformers_pipeline("audio-classification", model_id)
        ipex_generator = ipex_pipeline("audio-classification", model_id, accelerator="ipex")
        inputs = [np.random.random(16000)]
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForAudioClassification))
        self.assertAlmostEqual(transformers_output[0][0]["score"], ipex_output[0][0]["score"], delta=1e-2)
        self.assertAlmostEqual(transformers_output[0][1]["score"], ipex_output[0][1]["score"], delta=1e-2)

    @parameterized.expand(IMAGE_CLASSIFICATION_SUPPORTED_ARCHITECTURES)
    def test_image_classification_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        transformers_generator = transformers_pipeline("image-classification", model_id)
        ipex_generator = ipex_pipeline("image-classification", model_id, accelerator="ipex")
        inputs = "http://images.cocodataset.org/val2017/000000039769.jpg"
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs)
        self.assertEqual(len(transformers_output), len(ipex_output))
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForImageClassification))
        for i in range(len(transformers_output)):
            self.assertEqual(transformers_output[i]["label"], ipex_output[i]["label"])
            self.assertAlmostEqual(transformers_output[i]["score"], ipex_output[i]["score"], delta=1e-4)

    @parameterized.expand(COMMON_SUPPORTED_ARCHITECTURES)
    def test_pipeline_load_from_ipex_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = IPEXModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ipex_generator = ipex_pipeline("text-classification", model, tokenizer=tokenizer, accelerator="ipex")
        inputs = "This restaurant is awesome"
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForSequenceClassification))
        self.assertGreaterEqual(ipex_output[0]["score"], 0.0)

    @parameterized.expand(COMMON_SUPPORTED_ARCHITECTURES)
    def test_pipeline_load_from_jit_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = IPEXModelForSequenceClassification.from_pretrained(model_id)
        save_dir = TemporaryDirectory().name
        model.save_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ipex_generator = ipex_pipeline("text-classification", save_dir, tokenizer=tokenizer, accelerator="ipex")
        inputs = "This restaurant is awesome"
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForSequenceClassification))
        self.assertGreaterEqual(ipex_output[0]["score"], 0.0)

    @parameterized.expand(TEXT2TEXT_GENERATION_SUPPORTED_ARCHITECTURES)
    def test_text2text_generation_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        transformers_generator = transformers_pipeline("text2text-generation", model_id, torch_dtype=dtype)
        ipex_generator = ipex_pipeline("text2text-generation", model_id, accelerator="ipex", torch_dtype=dtype)
        inputs = "Describe a real-world application of AI."
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs, do_sample=False, max_new_tokens=10)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs, do_sample=False, max_new_tokens=10)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForSeq2SeqLM))
        self.assertEqual(transformers_output[0]["generated_text"], ipex_output[0]["generated_text"])

    @parameterized.expand(TEXT2TEXT_GENERATION_SUPPORTED_ARCHITECTURES)
    def test_summarization_generation_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        transformers_generator = transformers_pipeline("summarization", model_id, torch_dtype=dtype)
        ipex_generator = ipex_pipeline("summarization", model_id, accelerator="ipex", torch_dtype=dtype)
        inputs = "Describe a real-world application of AI."
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs, do_sample=False, max_new_tokens=10)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs, do_sample=False, max_new_tokens=10)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForSeq2SeqLM))
        self.assertEqual(transformers_output[0]["summary_text"], ipex_output[0]["summary_text"])

    @parameterized.expand(TEXT2TEXT_GENERATION_SUPPORTED_ARCHITECTURES)
    def test_translation_generation_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        transformers_generator = transformers_pipeline("translation", model_id, torch_dtype=dtype)
        ipex_generator = ipex_pipeline("translation", model_id, accelerator="ipex", torch_dtype=dtype)
        inputs = "Describe a real-world application of AI."
        with torch.inference_mode():
            transformers_output = transformers_generator(inputs, do_sample=False, max_new_tokens=10)
        with torch.inference_mode():
            ipex_output = ipex_generator(inputs, do_sample=False, max_new_tokens=10)
        self.assertTrue(isinstance(ipex_generator.model, IPEXModelForSeq2SeqLM))
        self.assertEqual(transformers_output[0]["translation_text"], ipex_output[0]["translation_text"])
