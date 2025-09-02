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
import time
import unittest

import numpy as np
import pytest
import requests
import torch
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import slow
from utils_tests import (
    MODEL_NAMES,
    TEST_IMAGE_URL,
)

from optimum.exporters.openvino.stateful import model_has_state
from optimum.intel import (
    OVModelForSeq2SeqLM,
    OVModelForSpeechSeq2Seq,
    OVModelForVision2Seq,
)
from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder
from optimum.intel.pipelines import pipeline as optimum_pipeline
from optimum.intel.utils.import_utils import is_openvino_version, is_transformers_version


SEED = 42
F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}
TENSOR_ALIAS_TO_TYPE = {"pt": torch.Tensor, "np": np.ndarray}


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class OVModelForSeq2SeqLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        # "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        # "longt5",
        "m2m_100",
        "mbart",
        "mt5",
        "pegasus",
        "t5",
    )

    if not (is_openvino_version(">=", "2025.3.0") and is_openvino_version("<", "2025.4.0")):
        # There are known issues with marian model on OpenVINO 2025.3.x
        SUPPORTED_ARCHITECTURES += ("marian",)

    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    SUPPORT_STATEFUL = ("t5", "mt5")
    if is_transformers_version(">=", "4.52.0"):
        SUPPORT_STATEFUL += ("bart", "blenderbot", "blenderbot-small", "m2m_100", "marian", "mbart")
    if is_transformers_version(">=", "4.53.0"):
        SUPPORT_STATEFUL += ("pegasus",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        ov_stateless_model = OVModelForSeq2SeqLM.from_pretrained(
            model_id, export=True, use_cache=False, stateful=False, ov_config=F32_CONFIG
        )
        expected_stateful = is_transformers_version(">", "4.43") and model_arch in self.SUPPORT_STATEFUL
        self.assertEqual(ov_model.decoder.stateful, expected_stateful)
        self.assertEqual(model_has_state(ov_model.decoder.model), expected_stateful)
        check_with_past_available = self.assertIsNone if expected_stateful else self.assertIsNotNone
        check_with_past_available(ov_model.decoder_with_past)
        self.assertIsInstance(ov_model.encoder, OVEncoder)
        self.assertIsInstance(ov_model.decoder, OVDecoder)
        if not ov_model.decoder.stateful:
            self.assertIsInstance(ov_model.decoder_with_past, OVDecoder)
            self.assertIsInstance(ov_model.config, PretrainedConfig)

        transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}
        ov_outputs = ov_model(**tokens, **decoder_inputs)
        ov_stateless_outputs = ov_stateless_model(**tokens, **decoder_inputs)

        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens, **decoder_inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=5e-3))
        self.assertTrue(torch.allclose(ov_stateless_outputs.logits, transformers_outputs.logits, atol=5e-3))
        gen_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=10,
            num_beams=2,
            do_sample=False,
            eos_token_id=None,
        )

        set_seed(SEED)
        generated_tokens = transformers_model.generate(**tokens, generation_config=gen_config)
        set_seed(SEED)
        ov_generated_tokens = ov_model.generate(**tokens, generation_config=gen_config)
        set_seed(SEED)
        ov_stateless_generated_tokens = ov_stateless_model.generate(**tokens, generation_config=gen_config)

        self.assertTrue(torch.equal(generated_tokens, ov_generated_tokens))
        self.assertTrue(torch.equal(generated_tokens, ov_stateless_generated_tokens))

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
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, compile=False)
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
        outputs = pipe(inputs, max_new_tokens=20)
        self.assertEqual(pipe.device, model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        ov_pipe = optimum_pipeline("text2text-generation", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs, max_new_tokens=20)
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
        model_id = MODEL_NAMES["bart"]
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
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        ov_model_stateless = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, stateful=False
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        # whisper cache class support implemented in 4.43
        expected_stateful = is_transformers_version(">", "4.43")
        self.assertEqual(ov_model.decoder.stateful, expected_stateful)
        self.assertEqual(model_has_state(ov_model.decoder.model), expected_stateful)
        check_with_past_available = self.assertIsNone if expected_stateful else self.assertIsNotNone
        check_with_past_available(ov_model.decoder_with_past)

        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        pt_features = processor.feature_extractor(data, return_tensors="pt")
        decoder_start_token_id = transformers_model.config.decoder_start_token_id
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

        with torch.no_grad():
            transformers_outputs = transformers_model(**pt_features, **decoder_inputs)

        for input_type in ["pt", "np"]:
            features = processor.feature_extractor(data, return_tensors=input_type)

            if input_type == "np":
                decoder_inputs = {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id}

            ov_outputs = ov_model(**features, **decoder_inputs)
            ov_stateless_outputs = ov_model_stateless(**features, **decoder_inputs)
            self.assertIn("logits", ov_outputs)
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-3))
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_stateless_outputs.logits), transformers_outputs.logits, atol=1e-3)
            )

        generate_kwrgs = {}
        if is_transformers_version(">=", "4.50"):
            generate_kwrgs = {"use_model_defaults": False}

        gen_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=10,
            num_beams=2,
            do_sample=False,
            eos_token_id=None,
        )

        set_seed(SEED)
        generated_tokens = transformers_model.generate(**pt_features, generation_config=gen_config, **generate_kwrgs)
        set_seed(SEED)
        ov_generated_tokens = ov_model.generate(**pt_features, generation_config=gen_config, **generate_kwrgs)
        set_seed(SEED)
        ov_stateless_generated_tokens = ov_model_stateless.generate(
            **pt_features, generation_config=gen_config, **generate_kwrgs
        )

        self.assertTrue(torch.equal(generated_tokens, ov_generated_tokens))
        self.assertTrue(torch.equal(generated_tokens, ov_stateless_generated_tokens))

        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id)
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
        url = TEST_IMAGE_URL
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
        ov_model = OVModelForVision2Seq.from_pretrained(model_id, compile=False)
        feature_extractor, tokenizer = self._get_preprocessors(model_id)
        ov_model.reshape(1, -1)
        ov_model.compile()

        # Image caption generation
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
