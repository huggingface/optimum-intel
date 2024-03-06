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


import time
import unittest

import numpy as np
import requests
import torch
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PretrainedConfig,
    pipeline,
    set_seed,
)

from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS
from optimum.intel import (
    IPEXModel,
    IPEXModelForAudioClassification,
    IPEXModelForCausalLM,
    IPEXModelForImageClassification,
    IPEXModelForMaskedLM,
    IPEXModelForQuestionAnswering,
    IPEXModelForSequenceClassification,
    IPEXModelForTokenClassification,
)


SEED = 42

MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "electra": "hf-internal-testing/tiny-random-electra",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "llama2": "Jiqing/tiny_random_llama2",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mistral": "echarlaix/tiny-random-mistral",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "stas/mt5-tiny-random",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "t5": "hf-internal-testing/tiny-random-t5",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "vit": "hf-internal-testing/tiny-random-vit",
    "wav2vec2": "anton-l/wav2vec2-random-tiny-classifier",
    "xlm": "hf-internal-testing/tiny-random-xlm",
}


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class IPEXModelTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
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

    IPEX_MODEL_CLASS = IPEXModel

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ipex_model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, export=True)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = self.IPEX_MODEL_CLASS.auto_model_class.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        outputs = ipex_model(**tokens)
        # Compare tensor outputs
        for output_name in {"logits", "last_hidden_state"}:
            if output_name in transformers_outputs:
                self.assertTrue(torch.allclose(outputs[output_name], transformers_outputs[output_name], atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline(self.IPEX_MODEL_CLASS.export_feature, model=model, tokenizer=tokenizer)
        text = "This restaurant is awesome"
        if self.IPEX_MODEL_CLASS.export_feature == "fill-mask":
            text += tokenizer.mask_token

        _ = pipe(text)
        self.assertEqual(pipe.device, model.device)


class IPEXModelForSequenceClassificationTest(IPEXModelTest):
    IPEX_MODEL_CLASS = IPEXModelForTokenClassification


class IPEXModelForTokenClassificationTest(IPEXModelTest):
    IPEX_MODEL_CLASS = IPEXModelForSequenceClassification


class IPEXModelForMaskedLMTest(IPEXModelTest):
    IPEX_MODEL_CLASS = IPEXModelForMaskedLM


class IPEXModelForQuestionAnsweringTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ipex_model = IPEXModelForQuestionAnswering.from_pretrained(model_id, export=True)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        outputs = ipex_model(**tokens)
        self.assertIn("start_logits", outputs)
        self.assertIn("end_logits", outputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.start_logits, transformers_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.end_logits, transformers_outputs.end_logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = IPEXModelForQuestionAnswering.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "What's my name?"
        context = "My Name is Sasha and I live in Lyon."
        outputs = pipe(question, context)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)


class IPEXModelForCausalLMTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
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
        # "phi",
        "mpt",
        "opt",
    )
    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.0

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, export=True)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        self.assertTrue(ipex_model.use_cache)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample",
            return_tensors="pt",
            return_token_type_ids=False if model_arch in ("llama", "llama2") else None,
        )
        position_ids = None
        if model_arch.replace("_", "-") in MODEL_TYPES_REQUIRING_POSITION_IDS:
            input_shape = tokens["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        outputs = ipex_model(**tokens, position_ids=position_ids)

        self.assertIsInstance(outputs.logits, torch.Tensor)
        self.assertIsInstance(outputs.past_key_values, (tuple, list))

        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = IPEXModelForCausalLM.from_pretrained(model_id, export=True, use_cache=False)
        model.config.encoder_no_repeat_ngram_size = 0
        model.to("cpu")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = pipe("This is a sample", max_length=10)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all("This is a sample" in item["generated_text"] for item in outputs))

    def test_compare_with_and_without_past_key_values(self):
        model_id = "echarlaix/tiny-random-gpt2-torchscript"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")

        model_with_pkv = IPEXModelForCausalLM.from_pretrained(model_id, use_cache=True, subfolder="model_with_pkv")
        # Warmup
        model_with_pkv.generate(**tokens)
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )
        model_without_pkv = IPEXModelForCausalLM.from_pretrained(
            model_id, use_cache=False, subfolder="model_without_pkv"
        )
        # Warmup
        model_without_pkv.generate(**tokens)
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertTrue(
            without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
            f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
            f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
        )


class IPEXModelForAudioClassificationTest(unittest.TestCase):
    IPEX_MODEL_CLASS = IPEXModelForAudioClassification
    SUPPORTED_ARCHITECTURES = (
        "unispeech",
        "wav2vec2",
    )

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        ipex_model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, export=True)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = self.IPEX_MODEL_CLASS.auto_model_class.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(self._generate_random_audio_data(), return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        outputs = ipex_model(**inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, export=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe([np.random.random(16000)])
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs[0]))


class IPEXModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "beit",
        # "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "resnet",
        "vit",
    )
    IPEX_MODEL_CLASS = IPEXModelForImageClassification

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ipex_model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, export=True)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = self.IPEX_MODEL_CLASS.auto_model_class.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        outputs = ipex_model(**inputs)
        self.assertIn("logits", outputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, export=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
