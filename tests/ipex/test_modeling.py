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
from optimum.utils.testing_utils import grid_parameters
from utils_tests import MODEL_NAMES, IS_XPU_AVAILABLE


SEED = 42
torch.use_deterministic_algorithms(True)
DEVICE = "xpu:0" if IS_XPU_AVAILABLE else "cpu"


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
    IPEX_PATCHED_SUPPORTED_ARCHITECTURES = ("bert",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ipex_model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, device_map=DEVICE)
        if model_arch in self.IPEX_PATCHED_SUPPORTED_ARCHITECTURES:
            self.assertTrue(ipex_model.add_patch)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = self.IPEX_MODEL_CLASS.auto_model_class.from_pretrained(model_id, device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        outputs = ipex_model(**tokens)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**tokens)
        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**tokens)

        # Compare tensor outputs
        for output_name in {"logits", "last_hidden_state"}:
            if output_name in transformers_outputs:
                self.assertTrue(torch.allclose(outputs[output_name], transformers_outputs[output_name], atol=1e-3))
                self.assertTrue(torch.allclose(outputs[output_name], loaded_model_outputs[output_name]))
                self.assertTrue(torch.allclose(outputs[output_name], init_model_outputs[output_name]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, device_map=DEVICE)
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
        ipex_model = IPEXModelForQuestionAnswering.from_pretrained(model_id, device_map=DEVICE)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id, device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        outputs = ipex_model(**tokens)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**tokens)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**tokens)

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
        model = IPEXModelForQuestionAnswering.from_pretrained(model_id, device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "What's my name?"
        context = "My Name is Sasha and I live in Lyon."
        outputs = pipe(question, context)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

    def test_patched_model(self):
        ipex_model = IPEXModelForQuestionAnswering.from_pretrained(
            "Intel/tiny-random-bert_ipex_model", device_map=DEVICE
        )
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(
            "hf-internal-testing/tiny-random-bert", device_map=DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt").to(DEVICE)
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
        "falcon",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "mistral",
        "llama",
        "llama2",
        # "phi",
        "distilgpt2",
        "mpt",
        "opt",
    )
    IPEX_PATCHED_SUPPORTED_ARCHITECTURES = ("llama2", "falcon", "gpt2")
    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.0

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        # Test model forward do not need cache.
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample",
            return_tensors="pt",
            return_token_type_ids=False if model_arch in ("llama", "llama2") else None,
        ).to(DEVICE)
        inputs = ipex_model.prepare_inputs_for_generation(**tokens)
        outputs = ipex_model(**inputs)

        self.assertIsInstance(outputs.logits, torch.Tensor)

        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, torch_dtype=dtype, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**inputs)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))
        # To avoid float pointing error
        self.assertTrue(torch.allclose(outputs.logits, loaded_model_outputs.logits, atol=1e-7))
        self.assertTrue(torch.allclose(outputs.logits, init_model_outputs.logits, atol=1e-7))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        model.config.encoder_no_repeat_ngram_size = 0
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = pipe("This is a sample", max_new_tokens=10)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all("This is a sample" in item["generated_text"] for item in outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.skip(reason="Paged attention do not support assisted decoding for now")
    def test_assisted_decoding(self, model_arch):
        # assist decoding does not support static cache now
        if model_arch in self.IPEX_PATCHED_SUPPORTED_ARCHITECTURES:
            return
        model_id = MODEL_NAMES[model_arch]
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        tokens = tokenizer("This is a sample input", return_tensors="pt").to(DEVICE)
        ipex_output = ipex_model.generate(**tokens, do_sample=False, max_new_tokens=4)
        ipex_output_assisted = ipex_model.generate(
            **tokens, do_sample=False, assistant_model=transformers_model, max_new_tokens=4
        )
        ipex_output_assisted_2 = ipex_model.generate(
            **tokens, do_sample=False, assistant_model=ipex_model, max_new_tokens=4
        )
        transformers_output = transformers_model.generate(**tokens, do_sample=False, max_new_tokens=4)
        transformers_output_assisted = transformers_model.generate(
            **tokens, do_sample=False, assistant_model=ipex_model, max_new_tokens=4
        )
        self.assertTrue(torch.equal(ipex_output, ipex_output_assisted))
        self.assertTrue(torch.equal(ipex_output, ipex_output_assisted_2))
        self.assertTrue(torch.equal(transformers_output, transformers_output_assisted))

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True, False],
            }
        )
    )
    def test_ipex_beam_search(self, test_name, model_arch, use_cache):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        model = IPEXModelForCausalLM.from_pretrained(
            model_id, use_cache=use_cache, torch_dtype=dtype, device_map=DEVICE
        )
        if use_cache and model_arch in self.IPEX_PATCHED_SUPPORTED_ARCHITECTURES:
            self.assertTrue(model.add_patch)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
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
            tokens = tokenizer(text, padding=True, return_tensors="pt").to(DEVICE)
            for generation_config in generation_configs:
                outputs = model.generate(**tokens, generation_config=generation_config)
                transformers_outputs = transformers_model.generate(**tokens, generation_config=generation_config)
                self.assertIsInstance(outputs, torch.Tensor)
                self.assertTrue(torch.equal(outputs, transformers_outputs))

    def test_compare_with_and_without_past_key_values(self):
        model_id = "Intel/tiny_random_llama2_ipex_model"
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        model_with_pkv = IPEXModelForCausalLM.from_pretrained(
            model_id, use_cache=True, torch_dtype=dtype, device_map=DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt").to(DEVICE)
        # Warmup
        model_with_pkv.generate(**tokens)
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )
        model_without_pkv = IPEXModelForCausalLM.from_pretrained(
            model_id, use_cache=False, torch_dtype=dtype, device_map=DEVICE
        )
        # Warmup
        model_without_pkv.generate(**tokens)
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + tokens.input_ids.shape[1])
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + tokens.input_ids.shape[1])

    @parameterized.expand(IPEX_PATCHED_SUPPORTED_ARCHITECTURES)
    def test_patched_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        patched_model_id = MODEL_NAMES["patched_" + model_arch]
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, export=True, device_map=DEVICE)
        exported_model = IPEXModelForCausalLM.from_pretrained(patched_model_id, device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample", return_tensors="pt").to(DEVICE)
        ipex_outputs = ipex_model.generate(
            **tokens, max_new_tokens=1, return_dict_in_generate=True, output_logits=True
        )
        exported_outputs = exported_model.generate(
            **tokens, max_new_tokens=1, return_dict_in_generate=True, output_logits=True
        )
        self.assertTrue(torch.allclose(ipex_outputs.logits[0], exported_outputs.logits[0], atol=1e-7))


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
        ipex_model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, device_map=DEVICE)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = self.IPEX_MODEL_CLASS.auto_model_class.from_pretrained(model_id, device_map=DEVICE)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(self._generate_random_audio_data(), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        outputs = ipex_model(**inputs)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**inputs)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-3))
        self.assertTrue(torch.equal(outputs.logits, loaded_model_outputs.logits))
        self.assertTrue(torch.equal(outputs.logits, init_model_outputs.logits))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, device_map=DEVICE)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe([np.random.random(16000)])
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs[0]))


class IPEXModelForImageClassificationIntegrationTest(unittest.TestCase):
    IPEX_MODEL_CLASS = IPEXModelForImageClassification
    SUPPORTED_ARCHITECTURES = (
        "beit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "resnet",
        "vit",
    )
    IPEX_PATCHED_SUPPORTED_ARCHITECTURES = ("vit",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ipex_model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, device_map=DEVICE)
        if model_arch in self.IPEX_PATCHED_SUPPORTED_ARCHITECTURES:
            self.assertTrue(ipex_model.add_patch)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = self.IPEX_MODEL_CLASS.auto_model_class.from_pretrained(model_id, device_map=DEVICE)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        outputs = ipex_model(**inputs)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**inputs)

        self.assertIn("logits", outputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.logits, loaded_model_outputs.logits, atol=1e-4))
        self.assertTrue(torch.allclose(init_model_outputs.logits, transformers_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = self.IPEX_MODEL_CLASS.from_pretrained(model_id, device_map=DEVICE)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    def test_patched_model(self):
        ipex_model = IPEXModelForImageClassification.from_pretrained(
            "Intel/tiny-random-vit_ipex_model", device_map=DEVICE
        )
        transformers_model = self.IPEX_MODEL_CLASS.from_pretrained(
            "hf-internal-testing/tiny-random-vit", device_map=DEVICE
        )
        preprocessor = AutoFeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-vit")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        outputs = ipex_model(**inputs)
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))
