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

# ruff: noqa

import tempfile
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
    GenerationConfig,
    PretrainedConfig,
    pipeline,
    set_seed,
)

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
from optimum.intel.utils.import_utils import is_ipex_version
from optimum.utils.testing_utils import grid_parameters
from utils_tests import MODEL_NAMES


SEED = 42


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class IPEXModelTest(unittest.TestCase):
    IPEX_MODEL_CLASS = IPEXModel
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

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname)
            loaded_model_outputs = loaded_model(**tokens)
        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model, export=True)
        init_model_outputs = init_model(**tokens)
        self.assertIsInstance(init_model.model, torch.jit.RecursiveScriptModule)

        # Compare tensor outputs
        for output_name in {"logits", "last_hidden_state"}:
            if output_name in transformers_outputs:
                self.assertTrue(torch.allclose(outputs[output_name], transformers_outputs[output_name], atol=1e-4))
                self.assertTrue(torch.equal(outputs[output_name], loaded_model_outputs[output_name]))
                self.assertTrue(torch.equal(outputs[output_name], init_model_outputs[output_name]))

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
    IPEX_MODEL_CLASS = IPEXModelForSequenceClassification


class IPEXModelForTokenClassificationTest(IPEXModelTest):
    IPEX_MODEL_CLASS = IPEXModelForTokenClassification


class IPEXModelForMaskedLMTest(IPEXModelTest):
    IPEX_MODEL_CLASS = IPEXModelForMaskedLM


class IPEXModelForQuestionAnsweringTest(unittest.TestCase):
    IPEX_MODEL_CLASS = IPEXModelForQuestionAnswering
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

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname)
            loaded_model_outputs = loaded_model(**tokens)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model, export=True)
        init_model_outputs = init_model(**tokens)
        self.assertIsInstance(init_model.model, torch.jit.RecursiveScriptModule)

        self.assertIn("start_logits", outputs)
        self.assertIn("end_logits", outputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.start_logits, transformers_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.end_logits, transformers_outputs.end_logits, atol=1e-4))
        self.assertTrue(torch.equal(outputs.start_logits, loaded_model_outputs.start_logits))
        self.assertTrue(torch.equal(outputs.end_logits, loaded_model_outputs.end_logits))
        self.assertTrue(torch.equal(outputs.start_logits, init_model_outputs.start_logits))
        self.assertTrue(torch.equal(outputs.end_logits, init_model_outputs.end_logits))

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

    @unittest.skipIf(is_ipex_version("<", "2.3.0"), reason="Only ipex version > 2.3.0 supports ipex model patching")
    def test_patched_model(self):
        ipex_model = IPEXModelForQuestionAnswering.from_pretrained(
            "Jiqing/patched_tiny_random_bert_for_question_answering"
        )
        transformers_model = AutoModelForQuestionAnswering.from_pretrained("hf-internal-testing/tiny-random-bert")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        outputs = ipex_model(**tokens)
        self.assertTrue(torch.allclose(outputs.start_logits, transformers_outputs.start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.end_logits, transformers_outputs.end_logits, atol=1e-4))


class IPEXModelForCausalLMTest(unittest.TestCase):
    IPEX_MODEL_CLASS = IPEXModelForCausalLM
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
        "mistral",
        "llama",
        "llama2",
        # "phi",
        "mpt",
        "opt",
    )
    IPEX_PATCHED_SUPPORTED_ARCHITECTURES = ("llama2",)
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
        inputs = ipex_model.prepare_inputs_for_generation(**tokens)
        outputs = ipex_model(**inputs)

        self.assertIsInstance(outputs.logits, torch.Tensor)
        self.assertIsInstance(outputs.past_key_values, (tuple, list))

        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model, export=True)
        init_model_outputs = init_model(**inputs)
        self.assertIsInstance(init_model.model, torch.jit.RecursiveScriptModule)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))
        self.assertTrue(torch.equal(outputs.logits, loaded_model_outputs.logits))
        self.assertTrue(torch.equal(outputs.logits, init_model_outputs.logits))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = IPEXModelForCausalLM.from_pretrained(model_id, export=True)
        model.config.encoder_no_repeat_ngram_size = 0
        model.to("cpu")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = pipe("This is a sample", max_length=10)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all("This is a sample" in item["generated_text"] for item in outputs))

    # High optimized model llama is not supported assisted decoding for now.
    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_assisted_decoding(self, model_arch):
        if model_arch == "llama2":
            return
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, export=True)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        ipex_output = ipex_model.generate(**tokens, do_sample=False, max_new_tokens=4)
        ipex_output_assisted = ipex_model.generate(
            **tokens, do_sample=False, assistant_model=transformers_model, max_new_tokens=4
        )
        transformers_output = transformers_model.generate(**tokens, do_sample=False, max_new_tokens=4)
        transformers_output_assisted = transformers_model.generate(
            **tokens, do_sample=False, assistant_model=ipex_model, max_new_tokens=4
        )
        self.assertTrue(torch.equal(ipex_output, ipex_output_assisted))
        self.assertTrue(torch.equal(transformers_output, transformers_output_assisted))

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": IPEX_PATCHED_SUPPORTED_ARCHITECTURES,
                "use_cache": [True, False],
            }
        )
    )
    @unittest.skipIf(is_ipex_version("<", "2.3.0"), reason="Only ipex version > 2.3.0 supports ipex model patching")
    def test_ipex_patching_beam_search(self, test_name, model_arch, use_cache):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        model = IPEXModelForCausalLM.from_pretrained(model_id, export=True, use_cache=use_cache)
        trasnformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        self.assertEqual(model.use_cache, use_cache)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        # Test with batch_size is 1 and 2.
        texts = ["This is a sample", ["This is the first input", "This is the second input"]]
        generation_configs = (
            GenerationConfig(max_new_tokens=4, num_beams=2, do_sample=False),
            GenerationConfig(max_new_tokens=4, num_beams=4, do_sample=False),
            GenerationConfig(max_new_tokens=4, num_beams=8, do_sample=False),
            GenerationConfig(max_new_tokens=4, num_beams=32, do_sample=False),
            GenerationConfig(
                max_new_tokens=4, do_sample=False, top_p=0.9, top_k=0, pad_token_id=tokenizer.eos_token_id
            ),
        )
        for text in texts:
            tokens = tokenizer(text, padding=True, return_tensors="pt")
            for generation_config in generation_configs:
                outputs = model.generate(**tokens, generation_config=generation_config)
                transformers_outputs = trasnformers_model.generate(**tokens, generation_config=generation_config)
                self.assertIsInstance(outputs, torch.Tensor)
                self.assertTrue(torch.equal(outputs, transformers_outputs))

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
        # self.assertTrue(
        #     without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
        #     f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
        #     f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
        # )


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

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model, export=True)
        init_model_outputs = init_model(**inputs)
        self.assertIsInstance(init_model.model, torch.jit.RecursiveScriptModule)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-3))
        self.assertTrue(torch.equal(outputs.logits, loaded_model_outputs.logits))
        self.assertTrue(torch.equal(outputs.logits, init_model_outputs.logits))

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
    IPEX_MODEL_CLASS = IPEXModelForImageClassification
    SUPPORTED_ARCHITECTURES = (
        "beit",
        # "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "resnet",
        "vit",
    )

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

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model, export=True)
        init_model_outputs = init_model(**inputs)
        self.assertIsInstance(init_model.model, torch.jit.RecursiveScriptModule)

        self.assertIn("logits", outputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))
        self.assertTrue(torch.equal(outputs.logits, loaded_model_outputs.logits))
        self.assertTrue(torch.allclose(init_model_outputs.logits, transformers_outputs.logits, atol=1e-4))

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

    @unittest.skipIf(is_ipex_version("<", "2.3.0"), reason="Only ipex version > 2.3.0 supports ipex model patching")
    def test_patched_model(self):
        ipex_model = IPEXModelForImageClassification.from_pretrained(
            "Jiqing/patched_tiny_random_vit_for_image_classification"
        )
        transformers_model = self.IPEX_MODEL_CLASS.from_pretrained("hf-internal-testing/tiny-random-vit")
        preprocessor = AutoFeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-vit")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        outputs = ipex_model(**inputs)
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))
