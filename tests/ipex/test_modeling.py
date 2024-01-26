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

import torch
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    pipeline,
    set_seed,
)

from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS
from optimum.intel import IPEXModelForCausalLM, IPEXModelForSequenceClassification


SEED = 42

MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "electra": "hf-internal-testing/tiny-random-electra",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mistral": "echarlaix/tiny-random-mistral",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "stas/mt5-tiny-random",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "t5": "hf-internal-testing/tiny-random-t5",
    "xlm": "hf-internal-testing/tiny-random-xlm",
}


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class IPEXModelForSequenceClassificationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        "convbert",
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
        ipex_model = IPEXModelForSequenceClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        outputs = ipex_model(**tokens)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = IPEXModelForSequenceClassification.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        text = "This restaurant is awesome"
        outputs = pipe(text)

        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)


class IPEXModelForCausalLMTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        # "gpt_bigcode",
        "blenderbot",
        "blenderbot-small",
        "bloom",
        "codegen",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "llama",
        # "mistral",
        # "mpt",
        "opt",
    )
    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, export=True)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        self.assertTrue(ipex_model.use_cache)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample", return_tensors="pt", return_token_type_ids=False if model_arch == "llama" else None
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
