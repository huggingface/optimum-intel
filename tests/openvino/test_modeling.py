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
import pytest
import requests
import timm
import torch
from datasets import load_dataset
from evaluate import evaluator
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
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
    AutoModelForSpeechSeq2Seq,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoTokenizer,
    GenerationConfig,
    Pix2StructForConditionalGeneration,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import slow
from utils_tests import MODEL_NAMES

from optimum.intel import (
    OVModelForAudioClassification,
    OVModelForAudioFrameClassification,
    OVModelForAudioXVector,
    OVModelForCausalLM,
    OVModelForCTC,
    OVModelForCustomTasks,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForPix2Struct,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVModelForVision2Seq,
    OVStableDiffusionPipeline,
)
from optimum.intel.openvino import OV_DECODER_NAME, OV_DECODER_WITH_PAST_NAME, OV_ENCODER_NAME, OV_XML_FILE_NAME
from optimum.intel.openvino.modeling_base import OVBaseModel
from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder
from optimum.intel.openvino.modeling_timm import TimmImageProcessor
from optimum.intel.openvino.utils import _print_compiled_model_properties
from optimum.intel.pipelines import pipeline as optimum_pipeline
from optimum.intel.utils.import_utils import is_openvino_version, is_transformers_version
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

F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}


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
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        self.assertEqual(loaded_model.request.get_property("PERFORMANCE_HINT"), "LATENCY")
        loaded_model_outputs = loaded_model(**tokens)

        # Test specifying ov_config with throughput hint and manual cache dir
        manual_openvino_cache_dir = loaded_model.model_save_dir / "manual_model_cache"
        ov_config = {"CACHE_DIR": str(manual_openvino_cache_dir), "PERFORMANCE_HINT": "THROUGHPUT"}
        loaded_model = OVModelForSequenceClassification.from_pretrained(self.OV_MODEL_ID, ov_config=ov_config)
        self.assertTrue(manual_openvino_cache_dir.is_dir())
        self.assertGreaterEqual(len(list(manual_openvino_cache_dir.glob("*.blob"))), 1)
        if is_openvino_version("<", "2023.3"):
            self.assertEqual(loaded_model.request.get_property("PERFORMANCE_HINT").name, "THROUGHPUT")
        else:
            self.assertEqual(loaded_model.request.get_property("PERFORMANCE_HINT"), "THROUGHPUT")

        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForSequenceClassification.from_pretrained(tmpdirname, ov_config={"NUM_STREAMS": 2})
            # Test that PERFORMANCE_HINT is set to LATENCY by default even with ov_config provided
            self.assertEqual(model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
            self.assertEqual(model.request.get_property("PERFORMANCE_HINT"), "LATENCY")

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
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        self.assertEqual(loaded_model.request.get_compiled_model().get_property("PERFORMANCE_HINT"), "LATENCY")
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
        loaded_model.compile()
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        self.assertEqual(loaded_model.decoder.request.get_compiled_model().get_property("PERFORMANCE_HINT"), "LATENCY")

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
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_pipeline.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        loaded_pipeline.compile()
        self.assertEqual(loaded_pipeline.unet.request.get_property("PERFORMANCE_HINT"), "LATENCY")
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

    def test_load_model_from_hub_private_with_token(self):
        token = os.environ.get("HF_HUB_READ_TOKEN", None)
        if token is None:
            self.skipTest("Test requires a token `HF_HUB_READ_TOKEN` in the environment variable")

        model = OVModelForCausalLM.from_pretrained(
            "optimum-internal-testing/tiny-random-phi-private", use_auth_token=token, revision="openvino"
        )
        self.assertIsInstance(model.config, PretrainedConfig)


class PipelineTest(unittest.TestCase):
    def test_load_model_from_hub(self):
        model_id = "echarlaix/tiny-random-PhiForCausalLM"

        # verify could load both pytorch and openvino model (export argument should automatically infered)
        ov_exported_pipe = optimum_pipeline("text-generation", model_id, revision="pt", accelerator="openvino")
        ov_pipe = optimum_pipeline("text-generation", model_id, revision="ov", accelerator="openvino")
        self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)
        self.assertIsInstance(ov_pipe.model, OVBaseModel)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ov_exported_pipe.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            ov_exported_pipe = optimum_pipeline("text-generation", tmpdirname, accelerator="openvino")
            self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)

        del ov_exported_pipe
        del ov_pipe
        gc.collect()

    def test_seq2seq_load_from_hub(self):
        model_id = "echarlaix/tiny-random-t5"
        # verify could load both pytorch and openvino model (export argument should automatically infered)
        ov_exported_pipe = optimum_pipeline("text2text-generation", model_id, accelerator="openvino")
        ov_pipe = optimum_pipeline("text2text-generation", model_id, revision="ov", accelerator="openvino")
        self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)
        self.assertIsInstance(ov_pipe.model, OVBaseModel)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ov_exported_pipe.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_DECODER_WITH_PAST_NAME in folder_contents)
            self.assertTrue(OV_DECODER_WITH_PAST_NAME.replace(".xml", ".bin") in folder_contents)
            ov_exported_pipe = optimum_pipeline("text2text-generation", tmpdirname, accelerator="openvino")
            self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)

        del ov_exported_pipe
        del ov_pipe
        gc.collect()


class OVModelForSequenceClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        "convbert",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSequenceClassification.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
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
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSequenceClassification.from_pretrained(model_id, export=True, compile=False)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        inputs = "This restaurant is awesome"
        outputs = pipe(inputs)
        self.assertTrue(model.is_dynamic)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        ov_pipe = optimum_pipeline("text-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["score"], ov_outputs[-1]["score"])
        del ov_pipe

        if model_arch == "bert":
            # Test FP16 conversion
            model.half()
            model.to("cpu")
            model.compile()
            outputs = pipe(inputs)
            self.assertGreaterEqual(outputs[0]["score"], 0.0)
            self.assertIsInstance(outputs[0]["label"], str)
            # Test static shapes
            model.reshape(1, 25)
            model.compile()
            outputs = pipe(inputs)
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
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
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
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForQuestionAnswering.from_pretrained(model_id, export=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "What's my name?"
        context = "My Name is Arthur and I live in Lyon."
        outputs = pipe(question, context)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)
        ov_pipe = optimum_pipeline("question-answering", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(question, context)
        self.assertEqual(outputs["score"], ov_outputs["score"])
        del model
        del ov_pipe
        gc.collect()

    @pytest.mark.run_slow
    @slow
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
        ov_model = OVModelForTokenClassification.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
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
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForTokenClassification.from_pretrained(model_id, export=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
        inputs = "My Name is Arthur and I live in"
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        ov_pipe = optimum_pipeline("token-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["score"], ov_outputs[-1]["score"])
        del ov_pipe
        del model
        del pipe
        gc.collect()

    def test_default_token_type_ids(self):
        model_id = MODEL_NAMES["bert"]
        model = OVModelForTokenClassification.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("this is a simple input", return_tensors="np")
        self.assertTrue("token_type_ids" in model.input_names)
        token_type_ids = tokens.pop("token_type_ids")
        outs = model(token_type_ids=token_type_ids, **tokens)
        outs_without_token_type_ids = model(**tokens)
        self.assertTrue(np.allclose(outs.logits, outs_without_token_type_ids.logits))

        tokens["attention_mask"] = None
        with self.assertRaises(Exception) as context:
            _ = model(**tokens)

        self.assertIn("Got unexpected inputs: ", str(context.exception))
        del model
        gc.collect()


class OVModelForFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
        "sentence-transformers-bert",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForFeatureExtraction.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
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
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForFeatureExtraction.from_pretrained(model_id, export=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        inputs = "My Name is Arthur and I live in"
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))
        ov_pipe = optimum_pipeline("feature-extraction", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1][-1][-1], ov_outputs[-1][-1][-1])
        del ov_pipe
        del pipe
        del model
        gc.collect()


class OVModelForCausalLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        "baichuan2",
        "baichuan2-13b",
        "gpt_bigcode",
        "blenderbot",
        "blenderbot-small",
        "bloom",
        "chatglm",
        "codegen",
        "codegen2",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "llama",
        # "llama_gptq",
        "marian",
        "minicpm",
        "mistral",
        "mixtral",
        "mpt",
        "opt",
        "pegasus",
        "qwen",
        "phi",
        "internlm2",
        "orion",
        "falcon",
        "falcon-40b",
        "persimmon",
        "biogpt",
        "gpt_neox_japanese",
        "xglm",
        "aquila",
        "aquila2",
        "xverse",
        "internlm",
        "jais",
        "glm4",
    )

    if is_transformers_version(">=", "4.40.0"):
        SUPPORTED_ARCHITECTURES += (
            "gemma",
            "olmo",
            "stablelm",
            "starcoder2",
            "dbrx",
            "phi3",
            "cohere",
            "qwen2",
            "qwen2-moe",
            "arctic",
        )

    GENERATION_LENGTH = 100
    REMOTE_CODE_MODELS = (
        "chatglm",
        "minicpm",
        "baichuan2",
        "baichuan2-13b",
        "jais",
        "qwen",
        "internlm2",
        "orion",
        "aquila",
        "aquila2",
        "xverse",
        "internlm",
        "codegen2",
        "arctic",
        "glm4",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        not_stateful = []
        if is_openvino_version("<", "2024.0"):
            not_stateful.append("mixtral")

        if is_openvino_version("<", "2024.1"):
            not_stateful.extend(["llama", "gemma", "gpt_bigcode"])

        if "gptq" in model_arch:
            self.skipTest("GPTQ model loading unsupported with AutoModelForCausalLM")

        set_seed(SEED)

        model_kwargs = {}
        if model_arch in self.REMOTE_CODE_MODELS:
            model_kwargs = {"trust_remote_code": True}

        ov_model = OVModelForCausalLM.from_pretrained(model_id, export=True, ov_config=F32_CONFIG, **model_kwargs)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        self.assertTrue(ov_model.use_cache)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS)
        tokens = tokenizer("This is a sample output", return_tensors="pt")
        tokens.pop("token_type_ids", None)

        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        self.assertTrue("past_key_values" in ov_outputs)
        self.assertIsInstance(ov_outputs.past_key_values, tuple)
        is_stateful = ov_model.config.model_type not in not_stateful
        self.assertEqual(ov_model.stateful, is_stateful)
        if is_stateful:
            self.assertTrue(len(ov_outputs.past_key_values) == 1 and len(ov_outputs.past_key_values[0]) == 0)

        set_seed(SEED)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if model_arch in ["qwen", "arctic", "glm4"]:
            transformers_model.to(torch.float32)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, equal_nan=True, atol=1e-4))

        # Qwen tokenizer does not support padding
        if model_arch == "qwen":
            return

        if model_arch not in ["chatglm", "glm4", "persimmon"]:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if model_arch == "persimmon":
            tokenizer.pad_token_id = tokenizer.bos_token_id
        # Compare batched generation
        tokenizer.padding_side = "left"
        tokens = tokenizer(["Today is a nice day and I am longer", "This is me"], return_tensors="pt", padding=True)
        tokens.pop("token_type_ids", None)
        ov_model.generation_config.eos_token_id = None
        transformers_model.generation_config.eos_token_id = None
        ov_model.config.eos_token_id = None
        transformers_model.config.eos_token_id = None
        gen_config = GenerationConfig(
            max_new_tokens=30,
            min_new_tokens=30,
            num_beams=3,
            do_sample=False,
            eos_token_id=None,
        )

        ov_outputs = ov_model.generate(**tokens, generation_config=gen_config)
        transformers_outputs = transformers_model.generate(**tokens, generation_config=gen_config)
        self.assertTrue(torch.allclose(ov_outputs, transformers_outputs))

        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_kwargs = {}
        model_id = MODEL_NAMES[model_arch]
        if model_arch in self.REMOTE_CODE_MODELS:
            model_kwargs = {
                "config": AutoConfig.from_pretrained(model_id, trust_remote_code=True),
                "trust_remote_code": True,
            }
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS)

        if model_arch == "qwen":
            tokenizer._convert_tokens_to_ids = lambda x: 0

        model = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=False, compile=False, **model_kwargs
        )
        model.eval()
        model.config.encoder_no_repeat_ngram_size = 0
        model.to("cpu")
        model.half()
        model.compile()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        inputs = "My name is Arthur and I live in"
        set_seed(SEED)
        outputs = pipe(inputs, max_new_tokens=5)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(inputs in item["generated_text"] for item in outputs))
        ov_pipe = optimum_pipeline(
            "text-generation",
            model_id,
            accelerator="openvino",
            trust_remote_code=model_arch in self.REMOTE_CODE_MODELS,
            tokenizer=tokenizer if model_arch == "qwen" else None,
        )
        set_seed(SEED)
        ov_outputs = ov_pipe(inputs, max_new_tokens=5)
        self.assertEqual(outputs[-1]["generated_text"], ov_outputs[-1]["generated_text"])
        del ov_pipe
        del pipe
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
        model_with_pkv = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True, stateful=False)
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
        model_stateful = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True, stateful=True)
        outputs_model_stateful = model_stateful.generate(
            **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
        )
        self.assertTrue(torch.equal(outputs_model_without_pkv, outputs_model_stateful))

        del model_with_pkv
        del model_without_pkv
        gc.collect()

    def test_print_model_properties(self):
        # test setting OPENVINO_LOG_LEVEL to 3, which calls _print_compiled_model_properties
        openvino_log_level = os.environ.get("OPENVINO_LOG_LEVEL", None)
        os.environ["OPENVINO_LOG_LEVEL"] = "3"
        model = OVModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], export=True)
        if openvino_log_level is not None:
            os.environ["OPENVINO_LOG_LEVEL"] = openvino_log_level
        # test calling function directly
        _print_compiled_model_properties(model.request)

    def test_auto_device_loading(self):
        OV_MODEL_ID = "echarlaix/distilbert-base-uncased-finetuned-sst-2-english-openvino"
        for device in ("AUTO", "AUTO:CPU"):
            model = OVModelForSequenceClassification.from_pretrained(OV_MODEL_ID, device=device)
            model.half()
            self.assertEqual(model._device, device)
            if device == "AUTO:CPU":
                model = OVModelForSequenceClassification.from_pretrained(OV_MODEL_ID, device=device)
                message = "Model should not be loaded from cache without explicitly setting CACHE_DIR"
                self.assertFalse(model.request.get_property("LOADED_FROM_CACHE"), message)
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
        input_ids = torch.argmax(outs.logits[:, -1:, :], dim=2)
        past_key_values = outs.past_key_values
        attention_mask = torch.ones((input_ids.shape[0], tokens.input_ids.shape[1] + 1), dtype=torch.long)
        outs_step2 = model_with_cache(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values
        )
        outs_without_attn_mask_step2 = model_with_cache(input_ids=input_ids, past_key_values=past_key_values)
        self.assertTrue(torch.allclose(outs_step2.logits, outs_without_attn_mask_step2.logits))
        del model_with_cache
        gc.collect()

    def test_default_filling_attention_mask_and_position_ids(self):
        model_id = MODEL_NAMES["llama"]
        model_with_cache = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        texts = ["this is a simple input"]
        tokens = tokenizer(texts, return_tensors="pt")
        self.assertTrue("position_ids" in model_with_cache.input_names)
        outs = model_with_cache(**tokens)
        attention_mask = tokens.pop("attention_mask")
        outs_without_attn_mask = model_with_cache(**tokens)
        self.assertTrue(torch.allclose(outs.logits, outs_without_attn_mask.logits))
        input_ids = torch.argmax(outs.logits[:, -1:, :], dim=2)
        past_key_values = outs.past_key_values
        attention_mask = torch.ones((input_ids.shape[0], tokens.input_ids.shape[1] + 1), dtype=torch.long)
        outs_step2 = model_with_cache(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values
        )
        outs_without_attn_mask_step2 = model_with_cache(input_ids=input_ids, past_key_values=past_key_values)
        self.assertTrue(torch.allclose(outs_step2.logits, outs_without_attn_mask_step2.logits))
        del model_with_cache
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_beam_search(self, model_arch):
        model_kwargs = {}
        model_id = MODEL_NAMES[model_arch]
        if model_arch in self.REMOTE_CODE_MODELS:
            model_kwargs = {
                "config": AutoConfig.from_pretrained(model_id, trust_remote_code=True),
                "trust_remote_code": True,
            }
        # Qwen tokenizer does not support padding, chatgm testing model produces nan that incompatible with beam search
        if model_arch in ["qwen", "chatglm"]:
            return

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS)
        if model_arch == "persimmon":
            tokenizer.pad_token_id = tokenizer.bos_token_id
            tokenizer.eos_token_id = tokenizer.bos_token_id

        beam_search_gen_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=10,
            num_beams=4,
            do_sample=False,
            eos_token_id=None,
        )

        beam_sample_gen_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=10,
            num_beams=4,
            do_sample=True,
            eos_token_id=None,
        )

        if model_arch == "minicpm":
            beam_sample_gen_config.top_k = 1

        group_beam_search_gen_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=10,
            num_beams=4,
            do_sample=False,
            eos_token_id=None,
            num_beam_groups=2,
            diversity_penalty=0.0000001,
        )
        force_word = "cat"
        force_words_ids = [tokenizer([force_word], add_special_tokens=False).input_ids]
        constrained_beam_search_gen_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=10,
            num_beams=4,
            do_sample=False,
            eos_token_id=None,
            force_words_ids=force_words_ids,
        )

        gen_configs = [
            beam_search_gen_config,
            beam_sample_gen_config,
            group_beam_search_gen_config,
            constrained_beam_search_gen_config,
        ]
        ov_model_stateful = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True, stateful=True, **model_kwargs
        )
        ov_model_stateless = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True, stateful=False, **model_kwargs
        )
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if model_arch == "arctic":
            transformers_model.to(torch.float32)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokens = tokenizer(["Today is a nice day and I am longer", "This is me"], return_tensors="pt", padding=True)
        tokens.pop("token_type_ids", None)
        ov_model_stateful.generation_config.eos_token_id = None
        ov_model_stateless.generation_config.eos_token_id = None
        transformers_model.generation_config.eos_token_id = None
        ov_model_stateful.config.eos_token_id = None
        ov_model_stateless.config.eos_token_id = None
        transformers_model.config.eos_token_id = None

        for gen_config in gen_configs:
            if gen_config.do_sample and model_arch in ["baichuan2-13b", "olmo"]:
                continue
            set_seed(SEED)
            transformers_outputs = transformers_model.generate(**tokens, generation_config=gen_config)
            set_seed(SEED)
            ov_stateful_outputs = ov_model_stateful.generate(**tokens, generation_config=gen_config)
            self.assertTrue(
                torch.equal(ov_stateful_outputs, transformers_outputs),
                f"generation config : {gen_config}, transformers output {transformers_outputs}, ov_model_stateful output {ov_stateful_outputs}",
            )
            set_seed(SEED)
            ov_stateless_outputs = ov_model_stateless.generate(**tokens, generation_config=gen_config)
            self.assertTrue(
                torch.equal(ov_stateless_outputs, transformers_outputs),
                f"generation config : {gen_config}, transformers output {transformers_outputs}, ov_model_stateless output {ov_stateless_outputs}",
            )


class OVModelForMaskedLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "mobilebert",
        "mpnet",
        "nystromformer",
        "perceiver_text",
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
        ov_model = OVModelForMaskedLM.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        set_seed(SEED)
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
        set_seed(SEED)
        model = OVModelForMaskedLM.from_pretrained(model_id, export=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        inputs = f"This is a {tokenizer.mask_token}."
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        set_seed(SEED)
        ov_pipe = optimum_pipeline("fill-mask", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["score"], ov_outputs[-1]["score"])
        del ov_pipe
        del pipe
        del model
        gc.collect()


class OVModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "beit",
        "convnext",
        # "convnextv2",
        "data2vec_vision",
        "deit",
        "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "poolformer",
        "perceiver_vision",
        "resnet",
        "segformer",
        "swin",
        "vit",
    )
    TIMM_MODELS = ("timm/pit_s_distilled_224.in1k", "timm/vit_tiny_patch16_224.augreg_in21k")

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForImageClassification.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        set_seed(SEED)
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
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForImageClassification.from_pretrained(model_id, export=True)
        model.eval()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        inputs = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
        set_seed(SEED)
        ov_pipe = optimum_pipeline("image-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["score"], ov_outputs[-1]["score"])
        del ov_pipe
        del model
        del pipe
        gc.collect()

    @parameterized.expand(TIMM_MODELS)
    def test_compare_to_timm(self, model_id):
        ov_model = OVModelForImageClassification.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        self.assertEqual(ov_model.request.get_property("INFERENCE_PRECISION_HINT").to_string(), "f32")
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
    SPEEDUP_CACHE = 1.1

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)

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
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a test"
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, compile=False)
        model.eval()
        model.half()
        model.to("cpu")
        model.compile()

        # Summarization
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["translation_text"], str)

        # Text2Text generation
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        ov_pipe = optimum_pipeline("text2text-generation", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["generated_text"], ov_outputs[-1]["generated_text"])
        del ov_pipe
        del pipe
        del model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
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
        "audio_spectrogram_transformer",
        "data2vec_audio",
        "hubert",
        "sew",
        "sew_d",
        "unispeech",
        "unispeech_sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
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
        ov_model = OVModelForAudioClassification.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        set_seed(SEED)
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
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForAudioClassification.from_pretrained(model_id, export=True)
        model.eval()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
        inputs = [np.random.random(16000)]
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs[0]))
        set_seed(SEED)
        ov_pipe = optimum_pipeline("audio-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1][-1]["score"], ov_outputs[-1][-1]["score"])
        del ov_pipe
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

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForCTC.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
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

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioXVector.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
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

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioFrameClassification.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
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
        ov_model = OVModelForPix2Struct.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)

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


class OVModelForSpeechSeq2SeqIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("whisper",)

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
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        decoder_start_token_id = transformers_model.config.decoder_start_token_id
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

        with torch.no_grad():
            transformers_outputs = transformers_model(**features, **decoder_inputs)

        for input_type in ["pt", "np"]:
            features = processor.feature_extractor(data, return_tensors=input_type)

            if input_type == "np":
                decoder_inputs = {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id}

            ov_outputs = ov_model(**features, **decoder_inputs)
            self.assertIn("logits", ov_outputs)
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-3))

        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
        model.eval()
        processor = get_preprocessor(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
        inputs = self._generate_random_audio_data()
        outputs = pipe(inputs)
        self.assertIsInstance(outputs["text"], str)

        ov_pipe = optimum_pipeline("automatic-speech-recognition", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs["text"], ov_outputs["text"])
        del ov_pipe
        del pipe
        del model
        gc.collect()


class OVModelForVision2SeqIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["vision-encoder-decoder", "trocr", "donut"]

    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    def _get_sample_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def _get_preprocessors(self, model_id):
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        return image_processor, tokenizer

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = OVModelForVision2Seq.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_generate_utils(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForVision2Seq.from_pretrained(model_id, export=True)
        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = feature_extractor(data, return_tensors="pt")

        outputs = model.generate(inputs=features["pixel_values"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        ov_model = OVModelForVision2Seq.from_pretrained(model_id, export=True)

        self.assertIsInstance(ov_model.encoder, OVEncoder)

        self.assertIsInstance(ov_model.decoder, OVDecoder)
        self.assertIsInstance(ov_model.decoder_with_past, OVDecoder)

        self.assertIsInstance(ov_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForVision2Seq.from_pretrained(model_id)
        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()

        start_token = "<s>"
        decoder_start_token_id = tokenizer.encode(start_token)[0]

        extra_inputs = [{}, {}]

        for extra_inps in extra_inputs:
            features = feature_extractor(data, return_tensors="pt")
            decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

            with torch.no_grad():
                transformers_outputs = transformers_model(**features, **decoder_inputs, **extra_inps, use_cache=True)
            input_type = "pt"
            features = feature_extractor(data, return_tensors=input_type)
            ov_outputs = ov_model(**features, **decoder_inputs, **extra_inps)

            self.assertTrue("logits" in ov_outputs)

            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-3))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch: str):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        ov_model = OVModelForVision2Seq.from_pretrained(model_id, export=True, compile=False)
        feature_extractor, tokenizer = self._get_preprocessors(model_id)
        ov_model.reshape(1, -1)
        ov_model.compile()

        # Speech recogition generation
        pipe = pipeline(
            "image-to-text",
            model=ov_model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )
        inputs = self._get_sample_image()
        outputs = pipe(inputs, max_new_tokens=3)
        self.assertEqual(pipe.device, ov_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)
        ov_pipe = optimum_pipeline("image-to-text", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs, max_new_tokens=3)
        self.assertEqual(outputs[-1]["generated_text"], ov_outputs[-1]["generated_text"])

        gc.collect()


class OVModelForCustomTasksIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_ATTENTION = ["vit-with-attentions"]
    SUPPORTED_ARCHITECTURES_WITH_HIDDEN_STATES = ["vit-with-hidden-states"]

    def _get_sample_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_ATTENTION)
    def test_compare_output_attentions(self, model_arch):
        model_id = MODEL_NAMES[model_arch]

        image = self._get_sample_image()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(images=image, return_tensors="pt")

        transformers_model = AutoModelForImageClassification.from_pretrained(model_id, attn_implementation="eager")
        transformers_model.eval()
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, output_attentions=True)

        ov_model = OVModelForCustomTasks.from_pretrained(model_id, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(len(ov_outputs.attentions) == len(transformers_outputs.attentions))
            for i in range(len(ov_outputs.attentions)):
                self.assertTrue(
                    torch.allclose(
                        torch.Tensor(ov_outputs.attentions[i]),
                        transformers_outputs.attentions[i],
                        atol=1e-4,  # attentions are accurate
                        rtol=1e-4,  # attentions are accurate
                    ),
                    f"Attention mismatch at layer {i}",
                )

        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_HIDDEN_STATES)
    def test_compare_output_hidden_states(self, model_arch):
        model_id = MODEL_NAMES[model_arch]

        image = self._get_sample_image()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(images=image, return_tensors="pt")

        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        transformers_model.eval()
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, output_hidden_states=True)

        ov_model = OVModelForCustomTasks.from_pretrained(model_id, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(len(ov_outputs.hidden_states) == len(transformers_outputs.hidden_states))
            for i in range(len(ov_outputs.hidden_states)):
                self.assertTrue(
                    torch.allclose(
                        torch.Tensor(ov_outputs.hidden_states[i]),
                        transformers_outputs.hidden_states[i],
                        atol=1e-3,  # hidden states are less accurate
                        rtol=1e-2,  # hidden states are less accurate
                    ),
                    f"Hidden states mismatch at layer {i}",
                )
        del transformers_model
        del ov_model
        gc.collect()
