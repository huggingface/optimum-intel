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
import os
import tempfile
import time
import unittest
from typing import Dict

import numpy as np
import requests
import timm
import torch
from datasets import load_dataset
from evaluate import evaluator
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GenerationConfig,
    Pix2StructForConditionalGeneration,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor
from utils_tests import MODEL_NAMES

from optimum.intel import (
    OVModelForAudioClassification,
    OVModelForAudioFrameClassification,
    OVModelForAudioXVector,
    OVModelForCausalLM,
    OVModelForCTC,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForPix2Struct,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
    OVStableDiffusionPipeline,
)
from optimum.intel.openvino import OV_DECODER_NAME, OV_DECODER_WITH_PAST_NAME, OV_ENCODER_NAME, OV_XML_FILE_NAME
from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder
from optimum.intel.openvino.modeling_timm import TimmImageProcessor
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)
from optimum.utils.testing_utils import require_diffusers


TENSOR_ALIAS_TO_TYPE = {
    "pt": torch.Tensor,
    "np": np.ndarray,
}

SEED = 42


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class OVModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OV_MODEL_ID = "echarlaix/distilbert-base-uncased-finetuned-sst-2-english-openvino"
        self.OV_DECODER_MODEL_ID = "helenai/gpt2-ov"
        self.OV_SEQ2SEQ_MODEL_ID = "echarlaix/t5-small-openvino"
        self.OV_DIFFUSION_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-openvino"

    def test_load_from_hub_and_save_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.OV_MODEL_ID)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForSequenceClassification.from_pretrained(self.OV_MODEL_ID)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        loaded_model_outputs = loaded_model(**tokens)

        # Test that model caching is automatically enabled
        openvino_cache_dir = loaded_model.model_save_dir / "model_cache"
        self.assertTrue(openvino_cache_dir.is_dir())
        self.assertGreaterEqual(len(list(openvino_cache_dir.glob("*.blob"))), 1)

        # Test specifying ov_config with throughput hint and manual cache dir
        manual_openvino_cache_dir = loaded_model.model_save_dir / "manual_model_cache"
        ov_config = {"CACHE_DIR": str(manual_openvino_cache_dir), "PERFORMANCE_HINT": "THROUGHPUT"}
        loaded_model = OVModelForSequenceClassification.from_pretrained(self.OV_MODEL_ID, ov_config=ov_config)
        self.assertTrue(manual_openvino_cache_dir.is_dir())
        self.assertGreaterEqual(len(list(manual_openvino_cache_dir.glob("*.blob"))), 1)
        self.assertEqual(loaded_model.request.get_property("PERFORMANCE_HINT").name, "THROUGHPUT")

        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForSequenceClassification.from_pretrained(tmpdirname)

        outputs = model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))

        del loaded_model
        del model
        gc.collect()

    @parameterized.expand((True, False))
    def test_load_from_hub_and_save_decoder_model(self, use_cache):
        model_id = "vuiseng9/ov-gpt2-fp32-kv-cache" if use_cache else "vuiseng9/ov-gpt2-fp32-no-cache"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_cache)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        loaded_model_outputs = loaded_model(**tokens)

        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForCausalLM.from_pretrained(tmpdirname, use_cache=use_cache)
            self.assertEqual(model.use_cache, use_cache)

        outputs = model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
        del loaded_model
        del model
        gc.collect()

    def test_load_from_hub_and_save_seq2seq_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.OV_SEQ2SEQ_MODEL_ID)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForSeq2SeqLM.from_pretrained(self.OV_SEQ2SEQ_MODEL_ID, compile=False)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        loaded_model.to("cpu")
        loaded_model_outputs = loaded_model.generate(**tokens)

        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_ENCODER_NAME in folder_contents)
            self.assertTrue(OV_DECODER_NAME in folder_contents)
            self.assertTrue(OV_DECODER_WITH_PAST_NAME in folder_contents)
            model = OVModelForSeq2SeqLM.from_pretrained(tmpdirname, device="cpu")

        outputs = model.generate(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs, outputs))
        del loaded_model
        del model
        gc.collect()

    @require_diffusers
    def test_load_from_hub_and_save_stable_diffusion_model(self):
        loaded_pipeline = OVStableDiffusionPipeline.from_pretrained(self.OV_DIFFUSION_MODEL_ID, compile=False)
        self.assertIsInstance(loaded_pipeline.config, Dict)
        batch_size, height, width = 2, 16, 16
        np.random.seed(0)
        inputs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "height": height,
            "width": width,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        pipeline_outputs = loaded_pipeline(**inputs).images
        self.assertEqual(pipeline_outputs.shape, (batch_size, height, width, 3))
        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_pipeline.save_pretrained(tmpdirname)
            pipeline = OVStableDiffusionPipeline.from_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(loaded_pipeline.config_name, folder_contents)
            for subfoler in {
                DIFFUSION_MODEL_UNET_SUBFOLDER,
                DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            }:
                folder_contents = os.listdir(os.path.join(tmpdirname, subfoler))
                self.assertIn(OV_XML_FILE_NAME, folder_contents)
                self.assertIn(OV_XML_FILE_NAME.replace(".xml", ".bin"), folder_contents)
        np.random.seed(0)
        outputs = pipeline(**inputs).images
        self.assertTrue(np.array_equal(pipeline_outputs, outputs))
        del pipeline
        gc.collect()


class OVModelForSequenceClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        # "camembert",
        "convbert",
        # "data2vec_text",
        # "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        # "mobilebert",
        # "nystromformer",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        # "xlm_roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSequenceClassification.from_pretrained(model_id, export=True, compile=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        text = "This restaurant is awesome"
        outputs = pipe(text)
        self.assertTrue(model.is_dynamic)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)
        if model_arch == "bert":
            # Test FP16 conversion
            model.half()
            model.to("cpu")
            model.compile()
            outputs = pipe(text)
            self.assertGreaterEqual(outputs[0]["score"], 0.0)
            self.assertIsInstance(outputs[0]["label"], str)
            # Test static shapes
            model.reshape(1, 25)
            model.compile()
            outputs = pipe(text)
            self.assertTrue(not model.is_dynamic)
            self.assertGreaterEqual(outputs[0]["score"], 0.0)
            self.assertIsInstance(outputs[0]["label"], str)
            # Test that model caching was not automatically enabled for exported model
            openvino_cache_dir = model.model_save_dir / "model_cache"
            self.assertFalse(openvino_cache_dir.is_dir())

        del model
        del pipe
        gc.collect()


class OVModelForQuestionAnsweringIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("start_logits", ov_outputs)
            self.assertIn("end_logits", ov_outputs)
            self.assertIsInstance(ov_outputs.start_logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(ov_outputs.end_logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_outputs.start_logits), transformers_outputs.start_logits, atol=1e-4)
            )
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_outputs.end_logits), transformers_outputs.end_logits, atol=1e-4)
            )
        del ov_model
        del transformers_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForQuestionAnswering.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "What's my name?"
        context = "My Name is Arthur and I live in Lyon."
        outputs = pipe(question, context)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)
        del model
        gc.collect()

    def test_metric(self):
        model_id = "distilbert-base-cased-distilled-squad"
        set_seed(SEED)
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, export=True)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        data = load_dataset("squad", split="validation").select(range(50))
        task_evaluator = evaluator("question-answering")
        transformers_pipe = pipeline("question-answering", model=transformers_model, tokenizer=tokenizer)
        ov_pipe = pipeline("question-answering", model=ov_model, tokenizer=tokenizer)
        transformers_metric = task_evaluator.compute(model_or_pipeline=transformers_pipe, data=data, metric="squad")
        ov_metric = task_evaluator.compute(model_or_pipeline=ov_pipe, data=data, metric="squad")
        self.assertEqual(ov_metric["exact_match"], transformers_metric["exact_match"])
        self.assertEqual(ov_metric["f1"], transformers_metric["f1"])
        del transformers_pipe
        del transformers_model
        del ov_pipe
        del ov_model
        gc.collect()


class OVModelForTokenClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForTokenClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForTokenClassification.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
        outputs = pipe("My Name is Arthur and I live in Lyon.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        del model
        del pipe
        gc.collect()


class OVModelForFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForFeatureExtraction.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("last_hidden_state", ov_outputs)
            self.assertIsInstance(ov_outputs.last_hidden_state, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(
                torch.allclose(
                    torch.Tensor(ov_outputs.last_hidden_state), transformers_outputs.last_hidden_state, atol=1e-4
                )
            )
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForFeatureExtraction.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        outputs = pipe("My Name is Arthur and I live in Lyon.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))
        del pipe
        del model
        gc.collect()


class OVModelForCausalLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        "gpt_bigcode",
        "blenderbot",
        "blenderbot-small",
        "bloom",
        "codegen",
        # "data2vec-text", # TODO : enable when enabled in exporters
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "llama",
        "marian",
        # "mistral",
        "mpt",
        "opt",
        "pegasus",
    )
    GENERATION_LENGTH = 100

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForCausalLM.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample", return_tensors="pt", return_token_type_ids=False if model_arch == "llama" else None
        )
        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=False, compile=False)
        model.config.encoder_no_repeat_ngram_size = 0
        model.to("cpu")
        model.half()
        model.compile()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = pipe("This is a sample", max_length=10)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all("This is a sample" in item["generated_text"] for item in outputs))
        del pipe
        del model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_multiple_inputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, compile=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        texts = ["this is a simple input", "this is a second simple input", "this is a third simple input"]
        tokens = tokenizer(texts, padding=True, return_tensors="pt")
        generation_config = GenerationConfig(encoder_no_repeat_ngram_size=0, max_new_tokens=20, num_beams=2)
        outputs = model.generate(**tokens, generation_config=generation_config)
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.shape[0], 3)
        del model
        gc.collect()

    def test_model_and_decoder_same_device(self):
        model_id = MODEL_NAMES["gpt2"]
        model = OVModelForCausalLM.from_pretrained(model_id, export=True)
        model.to("TEST")
        self.assertEqual(model._device, "TEST")
        # Verify that request is being reset
        self.assertEqual(model.request, None)
        del model
        gc.collect()

    def test_compare_with_and_without_past_key_values(self):
        model_id = MODEL_NAMES["gpt2"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")

        model_with_pkv = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)
        outputs_model_with_pkv = model_with_pkv.generate(
            **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
        )
        model_without_pkv = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=False)
        outputs_model_without_pkv = model_without_pkv.generate(
            **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
        )
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH)

        del model_with_pkv
        del model_without_pkv
        gc.collect()

    def test_auto_device_loading(self):
        model_id = MODEL_NAMES["gpt2"]
        for device in ("AUTO", "AUTO:CPU"):
            model = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True, device=device)
            model.half()
            self.assertEqual(model._device, device)
            del model
            gc.collect()

    def test_default_filling_attention_mask(self):
        model_id = MODEL_NAMES["gpt2"]
        model_with_cache = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        texts = ["this is a simple input"]
        tokens = tokenizer(texts, return_tensors="pt")
        self.assertTrue("attention_mask" in model_with_cache.input_names)
        outs = model_with_cache(**tokens)
        attention_mask = tokens.pop("attention_mask")
        outs_without_attn_mask = model_with_cache(**tokens)
        self.assertTrue(torch.allclose(outs.logits, outs_without_attn_mask.logits))
        input_ids = torch.argmax(outs.logits, dim=2)
        past_key_values = outs.past_key_values
        attention_mask = torch.ones((input_ids.shape[0], tokens.input_ids.shape[1] + 1), dtype=torch.long)
        outs_step2 = model_with_cache(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values
        )
        outs_without_attn_mask_step2 = model_with_cache(input_ids=input_ids, past_key_values=past_key_values)
        self.assertTrue(torch.allclose(outs_step2.logits, outs_without_attn_mask_step2.logits))
        del model_with_cache
        gc.collect()


class OVModelForMaskedLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        # "camembert",
        # "convbert",
        # "data2vec_text",
        "deberta",
        # "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        # "mobilebert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm_roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForMaskedLM.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = f"This is a sample {tokenizer.mask_token}"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForMaskedLM.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        outputs = pipe(f"This is a {tokenizer.mask_token}.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        del pipe
        del model
        gc.collect()


class OVModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "beit",
        "convnext",
        # "data2vec_vision",
        # "deit",
        "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        # "poolformer",
        "resnet",
        # "segformer",
        # "swin",
        "vit",
    )

    TIMM_MODELS = ("timm/pit_s_distilled_224.in1k", "timm/vit_tiny_patch16_224.augreg_in21k")

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForImageClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForImageClassification.from_pretrained(model_id, export=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
        del model
        del pipe
        gc.collect()

    @parameterized.expand(TIMM_MODELS)
    def test_compare_to_timm(self, model_id):
        ov_model = OVModelForImageClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        timm_model = timm.create_model(model_id, pretrained=True)
        preprocessor = TimmImageProcessor.from_pretrained(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            timm_model.eval()
            timm_outputs = timm_model(inputs["pixel_values"].float())
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), timm_outputs, atol=1e-3))
        gc.collect()

    @parameterized.expand(TIMM_MODELS)
    def test_timm_save_and_infer(self, model_id):
        ov_model = OVModelForImageClassification.from_pretrained(model_id, export=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_save_path = os.path.join(tmpdirname, "timm_ov_model")
            ov_model.save_pretrained(model_save_path)
            model = OVModelForImageClassification.from_pretrained(model_save_path)
            model(pixel_values=torch.zeros((5, 3, model.config.image_size, model.config.image_size)))
        gc.collect()


class OVModelForSeq2SeqLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        # "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        # "longt5",
        "m2m_100",
        "marian",
        "mbart",
        "mt5",
        "pegasus",
        "t5",
    )

    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.2

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True)

        self.assertIsInstance(ov_model.encoder, OVEncoder)
        self.assertIsInstance(ov_model.decoder, OVDecoder)
        self.assertIsInstance(ov_model.decoder_with_past, OVDecoder)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        ov_outputs = ov_model(**tokens, **decoder_inputs)

        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens, **decoder_inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, compile=False)
        model.half()
        model.to("cpu")
        model.compile()

        # Text2Text generation
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        # Summarization
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
        text = "This is a test"
        outputs = pipe(text)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["translation_text"], str)
        del pipe
        del model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "This is a sample input"
        tokens = tokenizer(text, return_tensors="pt")

        # General case
        outputs = model.generate(**tokens)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(outputs[0], str)

        # With input ids
        outputs = model.generate(input_ids=tokens["input_ids"])
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(outputs[0], str)
        del model

        gc.collect()

    def test_compare_with_and_without_past_key_values(self):
        model_id = MODEL_NAMES["t5"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "This is a sample input"
        tokens = tokenizer(text, return_tensors="pt")

        model_with_pkv = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, use_cache=True)
        _ = model_with_pkv.generate(**tokens)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, use_cache=False)
        _ = model_without_pkv.generate(**tokens)  # warmup
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
        del model_with_pkv
        del model_without_pkv
        gc.collect()


class OVModelForAudioClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        # "audio_spectrogram_transformer",
        # "data2vec_audio",
        # "hubert",
        # "sew",
        # "sew_d",
        # "wav2vec2-conformer",
        "unispeech",
        # "unispeech_sat",
        # "wavlm",
        "wav2vec2",
        # "wav2vec2-conformer",
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
        set_seed(SEED)
        ov_model = OVModelForAudioClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForAudioClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-3))

        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForAudioClassification.from_pretrained(model_id, export=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe([np.random.random(16000)])
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs[0]))
        del pipe
        del model
        gc.collect()


class OVModelForCTCIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "data2vec_audio",
        "hubert",
        "sew",
        "sew_d",
        "unispeech",
        "unispeech_sat",
        "wavlm",
        "wav2vec2-hf",
        "wav2vec2-conformer",
    ]

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = OVModelForCTC.from_pretrained(MODEL_NAMES["t5"], export=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForCTC.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForCTC.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**input_values)

            self.assertTrue("logits" in ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))

        del transformers_model
        del ov_model
        gc.collect()


class OVModelForAudioXVectorIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "data2vec_audio",
        "unispeech_sat",
        "wavlm",
        "wav2vec2-hf",
        "wav2vec2-conformer",
    ]

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = OVModelForAudioXVector.from_pretrained(MODEL_NAMES["t5"], export=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioXVector.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioXVector.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)
        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**input_values)

            self.assertTrue("logits" in ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_outputs.embeddings), transformers_outputs.embeddings, atol=1e-4)
            )

        del transformers_model
        del ov_model
        gc.collect()


class OVModelForAudioFrameClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "data2vec_audio",
        "unispeech_sat",
        "wavlm",
        "wav2vec2-hf",
        "wav2vec2-conformer",
    ]

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = OVModelForAudioFrameClassification.from_pretrained(MODEL_NAMES["t5"], export=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioFrameClassification.from_pretrained(model_id, export=True)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)
        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**input_values)

            self.assertTrue("logits" in ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))

        del transformers_model
        del ov_model
        gc.collect()


class OVModelForPix2StructIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["pix2struct"]
    TASK = "image-to-text"  # is it fine as well with visual-question-answering?

    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    IMAGE = Image.open(
        requests.get(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
            stream=True,
        ).raw
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForPix2Struct.from_pretrained(model_id, export=True)

        self.assertIsInstance(ov_model.encoder, OVEncoder)
        self.assertIsInstance(ov_model.decoder, OVDecoder)
        self.assertIsInstance(ov_model.decoder_with_past, OVDecoder)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        question = "Who am I?"
        transformers_model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)

        inputs = preprocessor(images=self.IMAGE, text=question, padding=True, return_tensors="pt")
        ov_outputs = ov_model(**inputs)

        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForPix2Struct.from_pretrained(model_id, export=True)
        preprocessor = get_preprocessor(model_id)
        question = "Who am I?"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")

        # General case
        outputs = model.generate(**inputs)
        outputs = preprocessor.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(outputs[0], str)
        del model

        gc.collect()

    def test_compare_with_and_without_past_key_values(self):
        model_id = MODEL_NAMES["pix2struct"]
        preprocessor = get_preprocessor(model_id)
        question = "Who am I?"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")

        model_with_pkv = OVModelForPix2Struct.from_pretrained(model_id, export=True, use_cache=True)
        _ = model_with_pkv.generate(**inputs)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **inputs, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = OVModelForPix2Struct.from_pretrained(model_id, export=True, use_cache=False)
        _ = model_without_pkv.generate(**inputs)  # warmup
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **inputs, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertTrue(
            without_pkv_timer.elapsed / with_pkv_timer.elapsed > self.SPEEDUP_CACHE,
            f"With pkv latency: {with_pkv_timer.elapsed:.3f} ms, without pkv latency: {without_pkv_timer.elapsed:.3f} ms,"
            f" speedup: {without_pkv_timer.elapsed / with_pkv_timer.elapsed:.3f}",
        )
        del model_with_pkv
        del model_without_pkv
        gc.collect()
