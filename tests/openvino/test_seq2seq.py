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

import copy
import gc
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import openvino
import pytest
import requests
import torch
from huggingface_hub import hf_hub_download
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTextToSpectrogram,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import slow
from transformers.utils import http_user_agent
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, SEED, TEST_IMAGE_URL, Timer

from optimum.exporters.openvino.model_patcher import patch_update_causal_mask
from optimum.exporters.openvino.stateful import model_has_state
from optimum.exporters.openvino.utils import ONNX_SUPPORTED_ARCHITECTURES
from optimum.exporters.tasks import TasksManager
from optimum.intel import (
    OVModelForSeq2SeqLM,
    OVModelForSpeechSeq2Seq,
    OVModelForTextToSpeechSeq2Seq,
    OVModelForVision2Seq,
    OVModelForVisualCausalLM,
)
from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder
from optimum.intel.openvino.modeling_text2speech import (
    OVTextToSpeechDecoder,
    OVTextToSpeechEncoder,
    OVTextToSpeechPostNet,
    OVTextToSpeechVocoder,
)
from optimum.intel.openvino.modeling_visual_language import MODEL_PARTS_CLS_MAPPING, MODEL_TYPE_TO_CLS_MAPPING
from optimum.intel.pipelines import pipeline as optimum_pipeline
from optimum.intel.utils.import_utils import is_openvino_version, is_transformers_version


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OVSeq2SeqTestMixin(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = None

    # TODO : Add phi4_multimodal (transformers modeling) not tested and needs to be fixed (converted model outputs not matching original model)
    UNSUPPORTED_ARCHITECTURES = {"phi4_multimodal"}
    if is_openvino_version(">=", "2025.3.0") and is_openvino_version("<", "2025.5.0"):
        UNSUPPORTED_ARCHITECTURES.add("marian")

    def _check_openvino_model_attributes(self, openvino_model, use_cache: bool = True, stateful: bool = True):
        self.assertIsInstance(openvino_model, self.OVMODEL_CLASS)
        self.assertIsInstance(openvino_model.config, PretrainedConfig)
        self.assertIsInstance(openvino_model.generation_config, GenerationConfig)

        self.assertIsInstance(openvino_model.encoder, OVEncoder)
        self.assertIsInstance(openvino_model.decoder, OVDecoder)
        self.assertIsInstance(openvino_model.encoder.model, openvino.Model)
        self.assertIsInstance(openvino_model.decoder.model, openvino.Model)

        if not stateful and use_cache:
            self.assertIsInstance(openvino_model.decoder_with_past, OVDecoder)
            self.assertIsInstance(openvino_model.decoder_with_past.model, openvino.Model)
        else:
            self.assertIsNone(openvino_model.decoder_with_past)

        self.assertEqual(openvino_model.use_cache, use_cache)
        self.assertEqual(openvino_model.decoder.stateful, stateful)
        self.assertEqual(model_has_state(openvino_model.decoder.model), stateful)

    def _test_find_untested_architectures(self):
        if len(self.SUPPORTED_ARCHITECTURES) != len(set(self.SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"For the task `{self.TASK}`, some architectures are duplicated in the list of tested architectures: "
                f"{self.SUPPORTED_ARCHITECTURES}.\n"
            )

        tested_architectures = set(self.SUPPORTED_ARCHITECTURES)
        transformers_architectures = set(CONFIG_MAPPING_NAMES.keys())
        ov_architectures = {
            model_type
            for model_type in TasksManager._SUPPORTED_MODEL_TYPE
            if self.TASK in TasksManager._SUPPORTED_MODEL_TYPE[model_type].get("openvino", {})
        }
        supported_architectures = ov_architectures & transformers_architectures
        supported_architectures -= ONNX_SUPPORTED_ARCHITECTURES

        untested_architectures = supported_architectures - tested_architectures

        if len(untested_architectures - self.UNSUPPORTED_ARCHITECTURES) > 0:
            raise ValueError(
                f"For the task `{self.TASK}`, the OpenVINO exporter supports {untested_architectures} which are not tested"
            )


class OVModelForSeq2SeqLMIntegrationTest(OVSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = (
        "bart",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        "longt5",
        "m2m_100",
        "mbart",
        "mt5",
        "pegasus",
        "t5",
    )
    OVMODEL_CLASS = OVModelForSeq2SeqLM
    AUTOMODEL_CLASS = AutoModelForSeq2SeqLM
    TASK = "text2text-generation"
    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    if not (is_openvino_version(">=", "2025.3.0") and is_openvino_version("<", "2025.5.0")):
        # There are known issues with marian model on OpenVINO 2025.3.x and 2025.4.x
        SUPPORTED_ARCHITECTURES += ("marian",)

    SUPPORT_STATEFUL = ("t5", "mt5", "longt5")
    if is_transformers_version(">=", "4.52.0"):
        SUPPORT_STATEFUL += ("bart", "blenderbot", "blenderbot-small", "m2m_100", "marian", "mbart")
    if is_transformers_version(">=", "4.53.0"):
        SUPPORT_STATEFUL += ("pegasus", "bigbird_pegasus")

    def test_find_untested_architectures(self):
        self._test_find_untested_architectures()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        ov_stateless_model = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, use_cache=False, stateful=False, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        expected_stateful = is_transformers_version(">", "4.46") and model_arch in self.SUPPORT_STATEFUL
        self._check_openvino_model_attributes(ov_model, use_cache=True, stateful=expected_stateful)
        self._check_openvino_model_attributes(ov_stateless_model, use_cache=False, stateful=False)

        transformers_model = self.AUTOMODEL_CLASS.from_pretrained(model_id)
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
        model = self.OVMODEL_CLASS.from_pretrained(model_id, compile=False, device=OPENVINO_DEVICE)
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
        model = self.OVMODEL_CLASS.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
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
        model_with_pkv = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, use_cache=True, device=OPENVINO_DEVICE
        )
        _ = model_with_pkv.generate(**tokens)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )
        model_without_pkv = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, use_cache=False, device=OPENVINO_DEVICE
        )
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


class OVModelForSpeechSeq2SeqIntegrationTest(OVSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = ("whisper",)
    OVMODEL_CLASS = OVModelForSpeechSeq2Seq
    AUTOMODEL_CLASS = AutoModelForSpeechSeq2Seq
    TASK = "automatic-speech-recognition"

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
        transformers_model = self.AUTOMODEL_CLASS.from_pretrained(model_id)
        ov_model = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        ov_model_stateless = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, stateful=False, device=OPENVINO_DEVICE
        )
        self._check_openvino_model_attributes(ov_model, use_cache=True, stateful=True)
        self._check_openvino_model_attributes(ov_model_stateless, use_cache=True, stateful=False)

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
        model = self.OVMODEL_CLASS.from_pretrained(model_id, device=OPENVINO_DEVICE)
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


class OVModelForVision2SeqIntegrationTest(OVSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = ["vision-encoder-decoder", "trocr", "donut"]
    # GOT-OCR2 models shouldn't be exported using the task image-to-text (currently equivalent to exporting the model using image-text-to-text) and will be deprecated v1.29
    # TODO: move pix2struct tests from OVModelForPix2StructIntegrationTest
    UNSUPPORTED_ARCHITECTURES = {"got_ocr2", "pix2struct"}
    TASK = "image-to-text"
    OVMODEL_CLASS = OVModelForVision2Seq
    AUTOMODEL_CLASS = AutoModelForVision2Seq
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

    def test_find_untested_architectures(self):
        self._test_find_untested_architectures()

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True, device=OPENVINO_DEVICE)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_generate_utils(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        model = self.OVMODEL_CLASS.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
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
        ov_model = self.OVMODEL_CLASS.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
        self._check_openvino_model_attributes(ov_model, use_cache=True, stateful=False)

        set_seed(SEED)
        transformers_model = self.AUTOMODEL_CLASS.from_pretrained(model_id)
        feature_extractor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()

        start_token = "<s>"
        decoder_start_token_id = tokenizer.encode(start_token)[0]
        features = feature_extractor(data, return_tensors="pt")
        decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

        with torch.no_grad():
            transformers_outputs = transformers_model(**features, **decoder_inputs, use_cache=True)

        ov_outputs = ov_model(**features, **decoder_inputs)
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
        ov_model = self.OVMODEL_CLASS.from_pretrained(model_id, compile=False, device=OPENVINO_DEVICE)
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


class OVModelForVisualCausalLMIntegrationTest(OVSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "internvl_chat",
        "llava",
        "llava_next",
        "llava_next_mistral",
        "llava_next_video",
        "llava-qwen2",
        "minicpmv",
        "phi3_v",
        "qwen2_vl",
    ]
    SUPPORT_VIDEO = ["llava_next_video", "qwen2_vl"]
    SUPPORT_AUDIO = []
    OVMODEL_CLASS = OVModelForVisualCausalLM
    TASK = "image-text-to-text"

    if is_transformers_version(">=", "4.46.0"):
        SUPPORTED_ARCHITECTURES += ["maira2", "idefics3"]

    if is_transformers_version(">=", "4.49.0"):
        SUPPORTED_ARCHITECTURES += ["qwen2_5_vl", "got_ocr2", "phi4mm"]
        SUPPORT_VIDEO.append("qwen2_5_vl")
        SUPPORT_AUDIO.append("phi4mm")
    if is_transformers_version(">", "4.49"):
        SUPPORTED_ARCHITECTURES += ["gemma3", "smolvlm"]
    if is_transformers_version(">=", "4.51"):
        # SUPPORTED_ARCHITECTURES += ["llama4", "phi4_multimodal"]
        SUPPORTED_ARCHITECTURES += ["llama4"]

    if is_transformers_version("<", "4.52"):
        SUPPORTED_ARCHITECTURES += ["minicpmo"]
    if is_transformers_version(">=", "4.57.0"):
        SUPPORTED_ARCHITECTURES += ["qwen3_vl"]
        SUPPORT_VIDEO += ["qwen3_vl"]

    if is_transformers_version(">=", "4.54.0"):
        # remote code models differs after transformers v4.54
        SUPPORTED_ARCHITECTURES = set(SUPPORTED_ARCHITECTURES) - {"llava-qwen2", "phi3_v", "phi4mm"}

    REMOTE_CODE_MODELS = ["internvl_chat", "minicpmv", "minicpmo", "llava-qwen2", "phi3_v", "maira2", "phi4mm"]
    IMAGE = Image.open(
        requests.get(
            TEST_IMAGE_URL,
            stream=True,
        ).raw
    )

    def get_transformer_model_class(self, model_arch):
        if is_transformers_version(">=", "4.46") and model_arch in [
            "llava",
            "llava_next",
            "llava_next_mistral",
            "qwen2_vl",
            "qwen2_5_vl",
            "got_ocr2",
            "gemma3",
            "idefics3",
            "smolvlm",
            "llama4",
            "qwen3_vl",
        ]:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        if model_arch == "llava_next_video":
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq
        if model_arch == "llava":
            from transformers import LlavaForConditionalGeneration

            return LlavaForConditionalGeneration
        if model_arch.startswith("llava_next"):
            from transformers import LlavaNextForConditionalGeneration

            return LlavaNextForConditionalGeneration
        if model_arch == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        return AutoModelForCausalLM

    def _check_device_and_request(self, ov_model, expected_device, has_request):
        request_check_fn = self.assertFalse if has_request else self.assertTrue
        self.assertEqual(ov_model._device, expected_device)
        for component_name, component in ov_model.components.items():
            if component_name == "language_model":
                request_check_fn(component.text_emb_request is None)
            self.assertEqual(component._device, expected_device)
            request_check_fn(component.request is None)

    def _check_openvino_model_attributes(self, openvino_model, use_cache: bool = True, stateful: bool = True):
        self.assertIsInstance(openvino_model, self.OVMODEL_CLASS)
        self.assertIsInstance(openvino_model.config, PretrainedConfig)
        self.assertIsInstance(openvino_model.generation_config, GenerationConfig)
        self.assertIsInstance(openvino_model, MODEL_TYPE_TO_CLS_MAPPING[openvino_model.config.model_type])

        for component_name, component in openvino_model.components.items():
            self.assertIsInstance(component, MODEL_PARTS_CLS_MAPPING[component_name])
            self.assertIsInstance(component.model, openvino.Model)

        self.assertEqual(openvino_model.use_cache, use_cache)
        self.assertEqual(openvino_model.language_model.stateful, stateful)
        self.assertEqual(model_has_state(openvino_model.language_model.model), stateful)

    def test_find_untested_architectures(self):
        self._test_find_untested_architectures()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        prompt = "What is shown in this image?"
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        loading_kwargs = {}

        trust_remote_code = model_arch in self.REMOTE_CODE_MODELS
        if "llama4" in model_arch:
            loading_kwargs = {"_attn_implementation": "sdpa"}
        transformers_model = self.get_transformer_model_class(model_arch).from_pretrained(
            model_id, trust_remote_code=trust_remote_code, **loading_kwargs
        )
        transformers_model.eval()
        if "internvl_chat" in model_arch:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trast_remote_code=trust_remote_code)
            img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            transformers_model.img_context_token_id = img_context_token_id
        if "llava-qwen2" in model_arch:
            transformers_model.get_vision_tower().load_model()
        preprocessors = self.get_preprocessors(model_arch)
        set_seed(SEED)
        ov_model = self.OVMODEL_CLASS.from_pretrained(
            model_id,
            export=True,
            trust_remote_code=trust_remote_code,
            compile=False,
            device=OPENVINO_DEVICE,
        )
        self._check_openvino_model_attributes(ov_model, use_cache=True, stateful=True)

        inputs = ov_model.preprocess_inputs(**preprocessors, text=prompt, image=self.IMAGE.resize((600, 600)))
        if model_arch == "gemma3":
            # validate that preprocessed input ids contain exactly one bos token
            bos_token = preprocessors["processor"].tokenizer.vocab["<bos>"]
            input_ids = inputs["input_ids"]
            bos_token_counts = (input_ids == bos_token).sum(dim=1)
            self.assertTrue(
                torch.all(bos_token_counts == 1),
                f"Each batch of input_ids must contain exactly one BOS token, "
                f"but found counts: {bos_token_counts.tolist()}",
            )

            if is_transformers_version(">=", "4.57.0"):
                inputs.pop("token_type_ids")

        transformers_inputs = copy.deepcopy(inputs)
        # llama4 preprocessing force bf16 dtype for pixel_values, that does not work on CPU with fp32 model
        # if past key values are not initialized, llama4 creates HybridCache with bf16 precision
        if model_arch == "llama4":
            transformers_inputs["pixel_values"] = transformers_inputs["pixel_values"].to(torch.float32)
            transformers_model.generation_config.cache_implementation = None
            from transformers.cache_utils import DynamicCache

            transformers_inputs["past_key_values"] = DynamicCache()

        test_device = "AUTO"
        ov_model.to(test_device)
        self._check_device_and_request(ov_model, test_device, False)
        test_device = "CPU"
        ov_model.to(test_device)
        ov_model.compile()
        self._check_device_and_request(ov_model, test_device, True)
        ov_model.clear_requests()
        self._check_device_and_request(ov_model, test_device, False)

        # pytorch minicpmv and internvl_chat are not designed to be used via forward
        if model_arch not in ["minicpmv", "minicpmo", "internvl_chat"]:
            set_seed(SEED)
            ov_outputs = ov_model(**inputs)
            set_seed(SEED)
            with torch.no_grad():
                transformers_outputs = transformers_model(**transformers_inputs)
            self.assertTrue(
                torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=4e-3),
                f"Max abs diff {(torch.abs(ov_outputs.logits - transformers_outputs.logits).max())}",
            )

        ov_model.generation_config.eos_token_id = None
        transformers_model.generation_config.eos_token_id = None
        transformers_model.generation_config.do_sample = False
        ov_model.config.eos_token_id = None
        transformers_model.config.eos_token_id = None
        ov_model.generation_config.do_sample = False
        # minicpmo diverges after 20 tokens
        tokens_to_generate = 20 if model_arch == "minicpmo" else 30
        gen_config = GenerationConfig(
            max_new_tokens=tokens_to_generate,
            min_new_tokens=tokens_to_generate,
            do_sample=False,
            eos_token_id=None,
        )
        set_seed(SEED)
        ov_outputs = ov_model.generate(**inputs, generation_config=gen_config)
        set_seed(SEED)

        additional_inputs = {}
        # gemma3 does not support dynamic cache, it is unfair to compare dynamic cache result vs hybrid cache,
        # align cache representation in torch model
        if model_arch == "gemma3":
            patch_update_causal_mask(
                transformers_model if is_transformers_version("<", "4.52.0") else transformers_model.language_model,
                "4.43.0",
            )
            transformers_model._supports_cache_class = True
            transformers_model.generation_config.cache_implementation = None
            from transformers.cache_utils import DynamicCache

            additional_inputs = {"past_key_values": DynamicCache()}

        if model_arch == "llama4":
            transformers_inputs["past_key_values"] = DynamicCache()

        with torch.no_grad():
            if model_arch in ["minicpmo"]:
                # `generate` method for minicpmo requires tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS
                )
                additional_inputs["tokenizer"] = tokenizer
            transformers_outputs = transformers_model.generate(
                **transformers_inputs, generation_config=gen_config, **additional_inputs
            )
            if model_arch in ["minicpmo"]:
                # retrieve decoded tokens for comparation
                transformers_outputs = transformers_outputs[1].sequences

        # original minicpmv, internvl always skip input tokens in generation results, while transformers based approach provide them
        if model_arch in ["minicpmv", "minicpmo", "internvl_chat"]:
            ov_outputs = ov_outputs[:, inputs["input_ids"].shape[1] :]
        self.assertTrue(
            torch.equal(ov_outputs, transformers_outputs),
            f"generation config : {gen_config}, transformers output {transformers_outputs}, ov_model output {ov_outputs}",
        )

        # video loader helper only available for transformers >= 4.49
        if model_arch in self.SUPPORT_VIDEO and is_transformers_version(">=", "4.49"):
            if is_transformers_version("<=", "4.52"):
                from transformers.image_utils import load_video
            else:
                from transformers.video_utils import load_video

            video_path = hf_hub_download(
                repo_id="raushan-testing-hf/videos-test",
                filename="sample_demo_1.mp4",
                repo_type="dataset",
                user_agent=http_user_agent(),
            )
            input_video, _ = load_video(video_path, num_frames=2, backend="opencv")
            question = "Why is this video funny?"
            inputs = ov_model.preprocess_inputs(**preprocessors, text=question, video=input_video)
            transformers_inputs = copy.deepcopy(inputs)
            ov_outputs = ov_model.generate(**inputs, generation_config=gen_config)
            # original minicpmv, internvl always skip input tokens in generation results, while transformers based approach provide them
            if model_arch in ["minicpmv", "minicpmo", "internvl_chat"]:
                ov_outputs = ov_outputs[:, inputs["input_ids"].shape[1] :]
            with torch.no_grad():
                transformers_outputs = transformers_model.generate(
                    **transformers_inputs, generation_config=gen_config, **additional_inputs
                )
            self.assertTrue(
                torch.equal(ov_outputs, transformers_outputs),
                f"generation config : {gen_config}, transformers output {transformers_outputs}, ov_model output {ov_outputs}",
            )

        if model_arch in self.SUPPORT_AUDIO:
            input_audio = self._generate_random_audio_data()
            question = "Translate this audio to French"
            inputs = ov_model.preprocess_inputs(**preprocessors, text=question, audio=[input_audio])
            transformers_inputs = copy.deepcopy(inputs)
            ov_outputs = ov_model.generate(**inputs, generation_config=gen_config)
            # original minicpmv, internvl always skip input tokens in generation results, while transformers based approach provide them
            if model_arch in ["minicpmv", "minicpmo", "internvl_chat"]:
                ov_outputs = ov_outputs[:, inputs["input_ids"].shape[1] :]
            with torch.no_grad():
                transformers_outputs = transformers_model.generate(
                    **transformers_inputs, generation_config=gen_config, **additional_inputs
                )
            self.assertTrue(
                torch.equal(ov_outputs, transformers_outputs),
                f"generation config : {gen_config}, transformers output {transformers_outputs}, ov_model output {ov_outputs}",
            )
        del transformers_model
        del ov_model

        gc.collect()

    @parameterized.expand(["llava", "llava_next", "llava_next_video", "llava_next_mistral"])
    def test_llava_with_new_preprocessing(self, model_arch):
        prompt = "<image>\n What is shown in this image?"
        model_id = MODEL_NAMES[model_arch]
        trust_remote_code = model_arch in self.REMOTE_CODE_MODELS
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor = AutoProcessor.from_pretrained(
            model_id,
            patch_size=config.vision_config.patch_size,
            vision_feature_select_strategy=config.vision_feature_select_strategy,
            trust_remote_code=model_arch in self.REMOTE_CODE_MODELS,
            num_additional_image_tokens=1,
        )
        transformers_model = self.get_transformer_model_class(model_arch).from_pretrained(model_id)
        ov_model = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, trust_remote_code=trust_remote_code, device=OPENVINO_DEVICE
        )
        self.assertTrue(ov_model._support_new_processing)
        self.assertTrue(processor.patch_size is not None)
        self.assertTrue(processor.vision_feature_select_strategy is not None)
        inputs = processor(images=self.IMAGE, text=prompt, return_tensors="pt")
        self.assertGreaterEqual(
            (inputs.input_ids == ov_model.config.image_token_index).sum().max().item(),
            ov_model.config.image_seq_length,
        )
        set_seed(SEED)
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        set_seed(SEED)
        ov_outputs = ov_model(**inputs)
        self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=1e-4))
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
        set_seed(SEED)
        ov_outputs = ov_model.generate(**inputs, generation_config=gen_config)
        set_seed(SEED)
        with torch.no_grad():
            transformers_outputs = transformers_model.generate(**inputs, generation_config=gen_config)
        self.assertTrue(
            torch.equal(ov_outputs, transformers_outputs),
            f"generation config : {gen_config}, transformers output {transformers_outputs}, ov_model output {ov_outputs}",
        )

        del ov_model
        del transformers_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        trust_remote_code = model_arch in self.REMOTE_CODE_MODELS
        model = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, trust_remote_code=trust_remote_code, device=OPENVINO_DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        question = "Describe image"
        preprocessors = self.get_preprocessors(model_arch)
        inputs = model.preprocess_inputs(**preprocessors, text=question, image=self.IMAGE.resize((600, 600)))
        # General case
        outputs = model.generate(**inputs, max_new_tokens=10)
        outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIsInstance(outputs[0], str)

        # GOT-OCR2 does not support text-only input
        if model_arch != "got_ocr2":
            # No input image case
            question = "Hi, how are you?"
            inputs = model.preprocess_inputs(**preprocessors, text=question, image=None)
            outputs = model.generate(**inputs, max_new_tokens=10)
            # filter out original prompt becuase it may contains out of tokenizer tokens e.g. in nanollva text separator = -200
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.assertIsInstance(outputs[0], str)

            if model_arch in self.SUPPORT_VIDEO and is_transformers_version(">=", "4.49"):
                # video loader helper only available for transformers >= 4.49
                if is_transformers_version("<=", "4.52"):
                    from transformers.image_utils import load_video
                else:
                    from transformers.video_utils import load_video

                video_path = hf_hub_download(
                    repo_id="raushan-testing-hf/videos-test",
                    filename="sample_demo_1.mp4",
                    repo_type="dataset",
                    user_agent=http_user_agent(),
                )
                input_video, _ = load_video(video_path, num_frames=2, backend="opencv")
                question = "Why is this video funny?"
                inputs = model.preprocess_inputs(**preprocessors, text=question, video=input_video)
                outputs = model.generate(**inputs, max_new_tokens=10)
                # filter out original prompt becuase it may contains out of tokenizer tokens e.g. in nanollva text separator = -200
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                self.assertIsInstance(outputs[0], str)

        if model_arch in self.SUPPORT_AUDIO:
            input_audio = self._generate_random_audio_data()
            question = "Translate this audio to French"
            inputs = model.preprocess_inputs(**preprocessors, text=question, audio=[input_audio])
            outputs = model.generate(**inputs, max_new_tokens=10)
            # filter out original prompt becuase it may contains out of tokenizer tokens e.g. in nanollva text separator = -200
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.assertIsInstance(outputs[0], str)
        del model

        gc.collect()

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return (audio_data, 16000)

    def get_preprocessors(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS)

        if model_arch == "llava-qwen2":
            processor = AutoProcessor.from_pretrained(
                config.mm_vision_tower, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS
            )
            preprocessors = {"processor": processor, "tokenizer": tokenizer, "config": config}
        elif model_arch == "internvl_chat":
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS
            )
            preprocessors = {"processor": None, "tokenizer": tokenizer, "config": config}
        else:
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS
            )
            preprocessors = {"processor": processor, "tokenizer": None, "config": config}

        return preprocessors

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_model_can_be_loaded_after_saving(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        with TemporaryDirectory() as save_dir:
            ov_model = self.OVMODEL_CLASS.from_pretrained(
                model_id,
                compile=False,
                trust_remote_code=model_arch in self.REMOTE_CODE_MODELS,
                device=OPENVINO_DEVICE,
            )
            ov_model.save_pretrained(save_dir)
            ov_restored_model = self.OVMODEL_CLASS.from_pretrained(
                save_dir,
                compile=False,
                trust_remote_code=model_arch in self.REMOTE_CODE_MODELS,
                device=OPENVINO_DEVICE,
            )
            self.assertIsInstance(ov_restored_model, type(ov_model))


class OVModelForTextToSpeechSeq2SeqIntegrationTest(OVSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = ("speecht5",)
    OVMODEL_CLASS = OVModelForTextToSpeechSeq2Seq
    AUTOMODEL_CLASS = AutoModelForTextToSpectrogram

    def _generate_text(self):
        return "This text is converted to speech using OpenVINO backend"

    def _generate_speaker_embedding(self):
        np.random.seed(42)
        speaker_embedding = np.random.randn(1, 512).astype(np.float32)
        return torch.tensor(speaker_embedding)

    def _get_processor(self, model_id, model_arch):
        if model_arch == "speecht5":
            return AutoProcessor.from_pretrained(model_id)
        else:
            raise Exception("{} unknown processor for text-to-speech".format(model_arch))

    def _get_vocoder(self, vocoder_id, model_arch):
        if model_arch == "speecht5":
            from transformers import SpeechT5HifiGan

            vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
            return vocoder
        else:
            raise Exception("{} unknown model for text-to-speech".format(model_arch))

    def _check_openvino_model_attributes(self, openvino_model, use_cache: bool = True):
        self.assertIsInstance(openvino_model, self.OVMODEL_CLASS)
        self.assertIsInstance(openvino_model.config, PretrainedConfig)
        self.assertIsInstance(openvino_model.generation_config, GenerationConfig)

        self.assertIsInstance(openvino_model.encoder, OVTextToSpeechEncoder)
        self.assertIsInstance(openvino_model.decoder, OVTextToSpeechDecoder)
        self.assertIsInstance(openvino_model.postnet, OVTextToSpeechPostNet)
        self.assertIsInstance(openvino_model.vocoder, OVTextToSpeechVocoder)
        self.assertIsInstance(openvino_model.encoder.model, openvino.Model)
        self.assertIsInstance(openvino_model.decoder.model, openvino.Model)
        self.assertIsInstance(openvino_model.postnet.model, openvino.Model)
        self.assertIsInstance(openvino_model.vocoder.model, openvino.Model)

        self.assertEqual(openvino_model.use_cache, use_cache)
        self.assertEqual(model_has_state(openvino_model.decoder.model), use_cache)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        set_seed(SEED)
        text_data = self._generate_text()
        speaker_embeddings = self._generate_speaker_embedding()
        model_id = MODEL_NAMES[model_arch]

        # since Auto class for text-to-audio is not implemented in optimum
        # generate model classes for reference generation
        vocoder_id = "fxmarty/speecht5-hifigan-tiny"
        processor = self._get_processor(model_id, model_arch)
        vocoder = self._get_vocoder(vocoder_id, model_arch)
        model = self.AUTOMODEL_CLASS.from_pretrained(model_id)
        inputs = processor(text=text_data, return_tensors="pt")
        ref_speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        ref_speech = ref_speech.unsqueeze(0) if ref_speech.dim() == 1 else ref_speech
        ov_model = self.OVMODEL_CLASS.from_pretrained(model_id, vocoder=vocoder_id, device=OPENVINO_DEVICE)
        ov_speech = ov_model.generate(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings)
        self._check_openvino_model_attributes(ov_model, use_cache=True)
        self.assertTrue(torch.allclose(ov_speech, ref_speech, atol=1e-3))

        del vocoder
        del model
        del processor
        gc.collect()


class OVModelForPix2StructIntegrationTest(OVSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = ["pix2struct"]
    TASK = "image-to-text"  # is it fine as well with visual-question-answering?
    OVMODEL_CLASS = OVModelForVision2Seq
    AUTOMODEL_CLASS = AutoModelForVision2Seq
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
        ov_model = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self._check_openvino_model_attributes(ov_model, use_cache=True, stateful=False)

        question = "Who am I?"
        transformers_model = self.AUTOMODEL_CLASS.from_pretrained(model_id)
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
        model = self.OVMODEL_CLASS.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
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
        model_with_pkv = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, use_cache=True, device=OPENVINO_DEVICE
        )
        _ = model_with_pkv.generate(**inputs)  # warmup
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **inputs, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )
        model_without_pkv = self.OVMODEL_CLASS.from_pretrained(
            model_id, export=True, use_cache=False, device=OPENVINO_DEVICE
        )
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
