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

# ruff: noqa

import tempfile
import time
import unittest

import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, pipeline, set_seed

from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS
from optimum.intel.generation.modeling import TSModelForCausalLM


MODEL_NAMES = {
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "gptj": "hf-internal-testing/tiny-random-gptj",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "mistral": "echarlaix/tiny-random-mistral",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "llama2": "Jiqing/tiny_random_llama2",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
}

SEED = 42


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class ModelingIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bloom",
        "gpt2",
        "gptj",
        "gpt_neo",
        "mistral",
        "llama",
        "llama2",
        "gpt_bigcode",
    )

    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        model = TSModelForCausalLM.from_pretrained(model_id, export=True)
        self.assertIsInstance(model.config, PretrainedConfig)
        trfs_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample", return_tensors="pt")

        position_ids = None
        if model_arch.replace("_", "-") in MODEL_TYPES_REQUIRING_POSITION_IDS:
            input_shape = tokens["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        outputs = model(**tokens, position_ids=position_ids)
        self.assertIsInstance(outputs.logits, torch.Tensor)
        with torch.no_grad():
            trfs_outputs = trfs_model(**tokens)
        # Compare outputs with original transformers model
        self.assertTrue(torch.allclose(outputs.logits, trfs_outputs.logits, atol=1e-4))
        # Compare outputs with loaded model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            loaded_model = TSModelForCausalLM.from_pretrained(tmpdirname)
            loaded_model_outputs = loaded_model(**tokens, position_ids=position_ids)

        self.assertTrue(torch.equal(outputs.logits, loaded_model_outputs.logits))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers_generate(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        model = TSModelForCausalLM.from_pretrained(model_id, export=True)
        self.assertIsInstance(model.config, PretrainedConfig)
        trfs_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample", return_tensors="pt")
        outputs = model.generate(**tokens, do_sample=False, num_beams=1, temperature=0.9, min_length=20, max_length=20)
        self.assertIsInstance(outputs, torch.Tensor)
        with torch.no_grad():
            trfs_outputs = trfs_model.generate(
                **tokens, do_sample=False, num_beams=1, temperature=0.9, min_length=20, max_length=20
            )
        # Compare outputs with original transformers model
        self.assertTrue(torch.equal(outputs, trfs_outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = TSModelForCausalLM.from_pretrained(model_id, export=True)
        model.to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
        outputs = pipe("This is a sample", max_length=10)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all("This is a sample" in item["generated_text"] for item in outputs))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_multiple_inputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        model = TSModelForCausalLM.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        texts = ["this is a simple input", "this is a second simple input", "this is a third simple input"]
        tokens = tokenizer(texts, padding=True, return_tensors="pt")
        outputs = model.generate(**tokens, max_new_tokens=20, num_beams=2)
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape[0], 3)

    def test_compare_with_and_without_past_key_values(self):
        model_id = MODEL_NAMES["gpt2"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        model_with_pkv = TSModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)
        # Warmup
        _ = model_with_pkv.generate(**tokens)
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = TSModelForCausalLM.from_pretrained(model_id, export=True, use_cache=False)
        # Warmup
        _ = model_without_pkv.generate(**tokens)
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )
        self.assertTrue(model_with_pkv.use_cache)
        self.assertFalse(model_without_pkv.use_cache)

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH)
        # self.assertTrue(
        #     without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
        #     f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
        #     f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
        # )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_input_names(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = TSModelForCausalLM.from_pretrained(model_id, export=True)
        self.assertTrue(isinstance(model.input_names, set))
        self.assertTrue("input_ids" in model.input_names)
        self.assertTrue("attention_mask" in model.input_names)
        if model.use_cache:
            self.assertTrue("past_key_values" in model.input_names)
        else:
            self.assertTrue("past_key_values" not in model.input_names)
