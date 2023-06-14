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
import torch
from datasets import load_dataset
from evaluate import evaluator
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    pipeline,
    set_seed,
)

from optimum.intel.openvino import (
    OV_DECODER_NAME,
    OV_DECODER_WITH_PAST_NAME,
    OV_ENCODER_NAME,
    OV_XML_FILE_NAME,
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
    OVStableDiffusionPipeline,
)
from optimum.intel.openvino.modeling_diffusion import (
    OVModelTextEncoder,
    OVModelUnet,
    OVModelVaeDecoder,
    OVModelVaeEncoder,
)
from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)
from optimum.utils.testing_utils import require_diffusers


MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "audio_spectrogram_transformer": "Ericwang/tiny-random-ast",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-blenderbot",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "codegen": "hf-internal-testing/tiny-random-CodeGenModel",
    "data2vec_text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec_vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec_audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "deberta": "hf-internal-testing/tiny-random-deberta",
    "deberta_v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "hf-internal-testing/tiny-random-deit",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "electra": "hf-internal-testing/tiny-random-electra",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    # "gpt_bigcode": "bigcode/tiny_starcoder_py",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "ibert": "hf-internal-testing/tiny-random-ibert",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "longt5": "hf-internal-testing/tiny-random-LongT5Model",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "marian": "sshleifer/tiny-marian-en-de",  # hf-internal-testing ones are broken
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mt5": "hf-internal-testing/tiny-random-mt5",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "pegasus": "hf-internal-testing/tiny-random-pegasus",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew_d": "hf-internal-testing/tiny-random-SEWDModel",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "t5": "hf-internal-testing/tiny-random-t5",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech_sat": "hf-internal-testing/tiny-random-UnispeechSatModel",
    "vit": "hf-internal-testing/tiny-random-vit",
    "wavlm": "hf-internal-testing/tiny-random-WavlmModel",
    "wav2vec2": "anton-l/wav2vec2-random-tiny-classifier",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "xlm": "hf-internal-testing/tiny-random-xlm",
    "xlm_roberta": "hf-internal-testing/tiny-xlm-roberta",
}


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

        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForSequenceClassification.from_pretrained(tmpdirname)

        outputs = model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))

    def test_load_from_hub_and_save_decoder_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.OV_DECODER_MODEL_ID)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForCausalLM.from_pretrained(self.OV_DECODER_MODEL_ID, use_cache=True)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        loaded_model_outputs = loaded_model(**tokens)

        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForCausalLM.from_pretrained(tmpdirname, use_cache=True)

        outputs = model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))

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

    @require_diffusers
    def test_load_from_hub_and_save_stable_diffusion_model(self):
        loaded_pipeline = OVStableDiffusionPipeline.from_pretrained(self.OV_DIFFUSION_MODEL_ID, compile=False)
        self.assertIsInstance(loaded_pipeline.config, Dict)
        prompt = "sailing ship in storm by Leonardo da Vinci"
        height = 16
        width = 16
        vae_scale_factor = 4  # needed for dummy stable diffusion model
        np.random.seed(0)
        pipeline_outputs = loaded_pipeline(prompt, num_inference_steps=1, height=height, width=width, output_type="np")
        self.assertEqual(pipeline_outputs.images.shape, (1, height // vae_scale_factor, width // vae_scale_factor, 3))
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
        outputs = pipeline(prompt, num_inference_steps=1, height=height, width=width, output_type="np").images
        self.assertTrue(np.array_equal(pipeline_outputs.images, outputs))


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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        # TODO : Replace from_transformers with export for optimum-intel v1.8
        model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True, compile=False)
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
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "What's my name?"
        context = "My Name is Arthur and I live in Lyon."
        outputs = pipe(question, context)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)
        gc.collect()

    def test_metric(self):
        model_id = "distilbert-base-cased-distilled-squad"
        set_seed(SEED)
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
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
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForTokenClassification.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
        outputs = pipe("My Name is Arthur and I live in Lyon.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
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
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        outputs = pipe("My Name is Arthur and I live in Lyon.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))
        gc.collect()


class OVModelForCausalLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        "blenderbot",
        "blenderbot_small",
        "bloom",
        # "codegen",
        # "data2vec-text", # TODO : enable when enabled in exporters
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        # "llama",
        "marian",
        "opt",
        "pegasus",
    )
    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.2

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
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OVModelForCausalLM.from_pretrained(model_id, from_transformers=True, use_cache=False, compile=False)
        model.config.encoder_no_repeat_ngram_size = 0
        model.to("cpu")
        model.half()
        model.compile()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = pipe("This is a sample", max_length=10)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all("This is a sample" in item["generated_text"] for item in outputs))
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

    def test_model_and_decoder_same_device(self):
        model_id = MODEL_NAMES["gpt2"]
        model = OVModelForCausalLM.from_pretrained(model_id, export=True)
        model.to("TEST")
        self.assertEqual(model._device, "TEST")
        # Verify that request is being reset
        self.assertEqual(model.request, None)

    def test_compare_with_and_without_past_key_values(self):
        model_id = MODEL_NAMES["gpt2"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")

        model_with_pkv = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)
        # Warmup
        _ = model_with_pkv.generate(**tokens)
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=False)
        # Warmup
        _ = model_without_pkv.generate(**tokens)
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


class OVModelForMaskedLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        # "albert",
        "bert",
        # "camembert",
        # "convbert",
        # "data2vec_text",
        # "deberta",
        # "deberta_v2",
        "distilbert",
        # "electra",
        # "flaubert",
        # "ibert",
        # "mobilebert",
        "roberta",
        # "roformer",
        # "squeezebert",
        # "xlm",
        # "xlm_roberta",
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
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForMaskedLM.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        outputs = pipe(f"This is a {tokenizer.mask_token}.")
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
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
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForImageClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
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
        # "mt5",
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

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, compile=False)
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

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True)
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

        gc.collect()

    def test_compare_with_and_without_past_key_values(self):
        model_id = MODEL_NAMES["t5"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = "This is a sample input"
        tokens = tokenizer(text, return_tensors="pt")

        model_with_pkv = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=True)
        _ = model_with_pkv.generate(**tokens)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )

        model_without_pkv = OVModelForSeq2SeqLM.from_pretrained(model_id, from_transformers=True, use_cache=False)
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForAudioClassification.from_pretrained(model_id, from_transformers=True)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe([np.random.random(16000)])
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs[0]))


class OVStableDiffusionPipelineIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("stable-diffusion",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        ov_pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True, compile=False)
        self.assertIsInstance(ov_pipeline.text_encoder, OVModelTextEncoder)
        self.assertIsInstance(ov_pipeline.vae_encoder, OVModelVaeEncoder)
        self.assertIsInstance(ov_pipeline.vae_decoder, OVModelVaeDecoder)
        self.assertIsInstance(ov_pipeline.unet, OVModelUnet)
        self.assertIsInstance(ov_pipeline.config, Dict)

        from diffusers import StableDiffusionPipeline

        diffusers_pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        diffusers_pipeline.safety_checker = None
        num_images_per_prompt, height, width, scale_factor = 1, 512, 512, 8
        latents_shape = (
            num_images_per_prompt,
            diffusers_pipeline.unet.in_channels,
            height // scale_factor,
            width // scale_factor,
        )
        latents = np.random.randn(*latents_shape).astype(np.float32)
        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "num_inference_steps": 1,
            "output_type": "np",
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
        }
        ov_pipeline.to("cpu")
        ov_pipeline.compile()
        ov_outputs = ov_pipeline(latents=latents, **kwargs).images
        self.assertIsInstance(ov_outputs, np.ndarray)
        with torch.no_grad():
            diffusers_outputs = diffusers_pipeline(latents=torch.from_numpy(latents), **kwargs).images
        # Compare model outputs
        self.assertTrue(np.allclose(ov_outputs, diffusers_outputs, atol=1e-4))
        # Compare model devices
        self.assertEqual(diffusers_pipeline.device.type, ov_pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        from diffusers import DPMSolverMultistepScheduler

        model_id = MODEL_NAMES[model_arch]
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True, scheduler=scheduler)
        prompt = "sailing ship in storm by Leonardo da Vinci"

        for batch_size in [1, 3]:
            for num_images in [1, 2]:
                outputs = pipeline(
                    [prompt] * batch_size, num_inference_steps=2, num_images_per_prompt=num_images, output_type="np"
                )
                self.assertEqual(outputs.images.shape, (batch_size * num_images, 128, 128, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        batch_size = 3
        num_images_per_prompt = 4
        height = 128
        width = 64
        vae_scale_factor = 4
        prompt = "sailing ship in storm by Leonardo da Vinci"
        pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True, compile=False)
        pipeline.half()
        pipeline.reshape(
            batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt
        )
        self.assertFalse(pipeline.is_dynamic)
        pipeline.compile()
        # Verify output shapes requirements not matching the static model don't impact the final outputs
        outputs = pipeline(
            [prompt] * batch_size,
            num_inference_steps=2,
            num_images_per_prompt=num_images_per_prompt,
            height=width,
            width=width,
            output_type="np",
        ).images
        self.assertEqual(
            outputs.shape,
            (batch_size * num_images_per_prompt, height // vae_scale_factor, width // vae_scale_factor, 3),
        )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)

        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "output_type": "np",
            "num_inference_steps": 2,
        }
        np.random.seed(0)
        outputs_1 = pipeline(**kwargs)
        np.random.seed(0)
        outputs_2 = pipeline(**kwargs)
        outputs_3 = pipeline(**kwargs)

        # Compare model outputs
        self.assertTrue(np.array_equal(outputs_1.images[0], outputs_2.images[0]))
        self.assertFalse(np.array_equal(outputs_1.images[0], outputs_3.images[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_height_width_properties(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        batch_size = 1
        num_images_per_prompt = 4
        height = 128
        width = 64
        pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True, compile=False, dynamic=True)
        self.assertTrue(pipeline.is_dynamic)
        self.assertEqual(pipeline.height, -1)
        self.assertEqual(pipeline.width, -1)
        pipeline.reshape(
            batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt
        )
        self.assertFalse(pipeline.is_dynamic)
        self.assertEqual(pipeline.height, height)
        self.assertEqual(pipeline.width, width)
