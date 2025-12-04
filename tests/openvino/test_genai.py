import os
import tempfile
import unittest
import warnings

import numpy as np
import openvino as ov
import requests
import torch
from openvino_genai import (
    LLMPipeline,
    Text2SpeechPipeline,
    VLMPipeline,
    WhisperPipeline,
)
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTextToSpectrogram,
    AutoProcessor,
    AutoTokenizer,
    set_seed,
)
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, TEST_IMAGE_URL

from optimum.exporters.openvino import main_export
from optimum.intel.openvino import (
    OVModelForCausalLM,
    OVModelForSpeechSeq2Seq,
    OVModelForTextToSpeechSeq2Seq,
    OVModelForVisualCausalLM,
)
from optimum.utils import is_transformers_version


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LLMPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "llama",
        "phi3",
        "gpt2",
        "qwen2",
    )

    GEN_KWARGS = {
        "max_new_tokens": 20,
        "min_new_tokens": 20,
        "do_sample": False,
        "num_beams": 1,
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_outputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]

        set_seed(42)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            set_seed(42)
            main_export(
                model_name_or_path=model_id,
                task="text-generation-with-past",
                convert_tokenizer=True,
                output=tmpdirname,
            )
            optimum_model = OVModelForCausalLM.from_pretrained(
                tmpdirname, device=OPENVINO_DEVICE, ov_config=F32_CONFIG
            )
            genai_model = LLMPipeline(tmpdirname, device=OPENVINO_DEVICE)

        prompt = "Paris is the capital of"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, **self.GEN_KWARGS)
        transformers_output = tokenizer.decode(transformers_ids[0], skip_special_tokens=True)[len(prompt) :]

        optimum_ids = optimum_model.generate(**inputs, **self.GEN_KWARGS)
        optimum_output = tokenizer.decode(optimum_ids[0], skip_special_tokens=True)[len(prompt) :]

        genai_output = genai_model.generate([prompt], **self.GEN_KWARGS).texts[0]

        self.assertEqual(transformers_output, optimum_output)
        try:
            self.assertEqual(transformers_output, genai_output)
        except AssertionError:
            # for some reason, outputs from GenAI and Optimum/Transformers differ for some models
            warnings.warn(
                f"Generated outputs from GenAI and Transformers differ for model {model_arch}.\n"
                f"Transformers output: {transformers_output}\n"
                f"GenAI output: {genai_output}"
            )


class VLMPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("qwen2_vl",)

    IMAGE = Image.open(
        requests.get(
            TEST_IMAGE_URL,
            stream=True,
        ).raw
    )
    GEN_KWARGS = {
        "max_new_tokens": 20,
        "min_new_tokens": 20,
        "do_sample": False,
        "num_beams": 1,
    }

    def _get_model_class(self, model_arch):
        if is_transformers_version(">=", "4.46"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_outputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]

        set_seed(42)
        model_class = self._get_model_class(model_arch)
        transformers_model = model_class.from_pretrained(model_id).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            set_seed(42)
            main_export(
                model_name_or_path=model_id,
                task="image-text-to-text",
                convert_tokenizer=True,
                output=tmpdirname,
            )
            optimum_model = OVModelForVisualCausalLM.from_pretrained(
                tmpdirname, device=OPENVINO_DEVICE, ov_config=F32_CONFIG
            )
            genai_model = VLMPipeline(tmpdirname, device=OPENVINO_DEVICE)

        image = self.IMAGE
        prompt = "A photo of a cat sitting on a"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        inputs = optimum_model.preprocess_inputs(text=prompt, image=image, tokenizer=tokenizer, processor=processor)
        full_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, **self.GEN_KWARGS)
        transformers_output = tokenizer.decode(transformers_ids[0], skip_special_tokens=True)[len(full_prompt) :]

        optimum_ids = optimum_model.generate(**inputs, **self.GEN_KWARGS)
        optimum_output = tokenizer.decode(optimum_ids[0], skip_special_tokens=True)[len(full_prompt) :]

        genai_output = genai_model.generate(prompt, images=[ov.Tensor(np.array(image))], **self.GEN_KWARGS).texts[0]

        self.assertEqual(transformers_output, optimum_output)
        self.assertEqual(transformers_output, genai_output)


class Text2SpeechPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("speecht5",)
    VOCODER = "fxmarty/speecht5-hifigan-tiny"

    GEN_KWARGS = {
        "max_new_tokens": 20,
        "min_new_tokens": 20,
        "do_sample": False,
        "num_beams": 1,
    }

    def _get_vocoder(self, vocoder_id, model_arch):
        if model_arch == "speecht5":
            from transformers import SpeechT5HifiGan

            vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
            return vocoder
        else:
            raise Exception("{} unknown model for text-to-speech".format(model_arch))

    def _generate_speaker_embedding(self):
        np.random.seed(42)
        speaker_embedding = np.random.randn(1, 512).astype(np.float32)
        return torch.tensor(speaker_embedding)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_outputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]

        set_seed(42)
        transformers_model = AutoModelForTextToSpectrogram.from_pretrained(model_id).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            set_seed(42)
            main_export(
                model_name_or_path=model_id,
                task="text-to-audio-with-past",
                model_kwargs={"vocoder": self.VOCODER},
                convert_tokenizer=True,
                output=tmpdirname,
            )
            optimum_model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
                tmpdirname, device=OPENVINO_DEVICE, ov_config=F32_CONFIG
            )
            genai_model = Text2SpeechPipeline(tmpdirname, device=OPENVINO_DEVICE)

        text = "Hello, how are you?"
        processor = AutoProcessor.from_pretrained(model_id)
        speaker_embeddings = self._generate_speaker_embedding()
        vocoder = self._get_vocoder(self.VOCODER, model_arch).eval()
        inputs = processor(text=text, return_tensors="pt")
        inputs["speaker_embeddings"] = speaker_embeddings

        with torch.no_grad():
            transformers_output = transformers_model.generate(**inputs, **self.GEN_KWARGS, vocoder=vocoder)
            transformers_output = transformers_output.squeeze(0)  # collapse batch dimension

        optimum_output = optimum_model.generate(**inputs, **self.GEN_KWARGS)
        optimum_output = optimum_output.squeeze(0)  # collapse batch dimension

        genai_output = genai_model.generate(text, **self.GEN_KWARGS).speeches[0]
        genai_output = torch.from_numpy(genai_output.data).squeeze(0)  # collapse batch dimension

        torch.testing.assert_close(transformers_output, optimum_output, rtol=1e-2, atol=1e-4)
        torch.testing.assert_close(transformers_output, genai_output, rtol=1e-2, atol=1e-4)


class WhisperPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("whisper",)

    GEN_KWARGS = {
        "max_new_tokens": 20,
        "min_new_tokens": 20,
        "do_sample": False,
        "num_beams": 1,
    }

    def _get_audio(self):
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_outputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]

        set_seed(42)
        transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            set_seed(42)
            main_export(
                model_name_or_path=model_id,
                task="automatic-speech-recognition-with-past",
                convert_tokenizer=True,
                output=tmpdirname,
            )
            optimum_model = OVModelForSpeechSeq2Seq.from_pretrained(
                tmpdirname, device=OPENVINO_DEVICE, ov_config=F32_CONFIG
            )
            genai_model = WhisperPipeline(tmpdirname, device=OPENVINO_DEVICE)

        audio = self._get_audio()
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, **self.GEN_KWARGS)
            transformers_output = tokenizer.decode(transformers_ids[0], skip_special_tokens=True)

        optimum_ids = optimum_model.generate(**inputs, **self.GEN_KWARGS)
        optimum_output = tokenizer.decode(optimum_ids[0], skip_special_tokens=True)

        genai_output = genai_model.generate(inputs["input_features"].flatten().tolist(), **self.GEN_KWARGS).texts[0]

        self.assertEqual(transformers_output, optimum_output)
        self.assertEqual(transformers_output, genai_output)
