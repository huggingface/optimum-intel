import os
import tempfile
import unittest

import numpy as np
import openvino as ov
import requests
import torch
from openvino_genai import (
    LLMPipeline,
    Text2SpeechPipeline,
    VLMPipeline,
    WhisperPipeline,
    draft_model,
)
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTextToSpectrogram,
    AutoProcessor,
    AutoTokenizer,
    set_seed,
)
from utils_tests import EAGLE3_MODELS, F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, REMOTE_CODE_MODELS, TEST_IMAGE_URL

from optimum.exporters.openvino import main_export
from optimum.intel.openvino import (
    OVModelForCausalLM,
    OVModelForSpeechSeq2Seq,
    OVModelForTextToSpeechSeq2Seq,
    OVModelForVisualCausalLM,
)
from optimum.intel.utils.import_utils import is_openvino_version
from optimum.utils import is_transformers_version


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LLMPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "gpt_bigcode",
        "bloom",
        "codegen",
        "codegen2",
        "gpt2",
        "gptj",
        "gpt_neox",
        "llama",
        "mistral",
        "mixtral",
        "phi",
        "internlm2",
        "orion",
        "falcon",
        "persimmon",
        "xglm",
        "aquila",
        "aquila2",
        "internlm",
        "jais",
        "decilm",
        "gemma",
        "olmo",
        "stablelm",
        "starcoder2",
        "dbrx",
        "cohere",
        "qwen2",
        "qwen2_moe",
        "phi3",
        "gemma2",
        "exaone",
        "granite",
        "granitemoe",
    )

    if is_transformers_version(">=", "4.46.0"):
        SUPPORTED_ARCHITECTURES += ("glm", "mistral-nemo", "phimoe", "opt")
        if is_transformers_version("<", "4.54.0"):
            SUPPORTED_ARCHITECTURES += ("deepseek",)
        if is_transformers_version("<", "4.56.0"):
            SUPPORTED_ARCHITECTURES += ("qwen",)
    if is_transformers_version(">=", "4.49"):
        SUPPORTED_ARCHITECTURES += ("gemma3_text",)
    if is_transformers_version(">=", "4.51.0"):
        SUPPORTED_ARCHITECTURES += ("qwen3", "qwen3_moe")
    if is_transformers_version(">=", "4.51.3"):
        SUPPORTED_ARCHITECTURES += ("glm4",)
    if is_transformers_version(">=", "4.53.0"):
        SUPPORTED_ARCHITECTURES += ("arcee",)
    if is_transformers_version(">=", "4.54.0"):
        SUPPORTED_ARCHITECTURES += ("exaone4",)
    if is_transformers_version(">=", "4.55.0"):
        SUPPORTED_ARCHITECTURES += ("gpt_oss",)
    if is_transformers_version("<", "4.54.0"):
        SUPPORTED_ARCHITECTURES += ("minicpm", "minicpm3", "arctic")
    if is_transformers_version("<", "4.56.0"):
        SUPPORTED_ARCHITECTURES += ("chatglm", "chatglm4")

    REMOTE_CODE_MODELS = (
        "chatglm",
        "minicpm",
        "jais",
        "qwen",
        "internlm2",
        "orion",
        "aquila",
        "aquila2",
        "internlm",
        "codegen2",
        "arctic",
        "chatglm4",
        "exaone",
        "exaone4",
        "decilm",
        "minicpm3",
        "deepseek",
    )
    NO_CACHE_MODELS = (  # mostly remote that are broken with past key values
        "aquila",
        "aquila2",
        "decilm",
        "internlm",
        "internlm2",
        "orion",
        "jais",
        "qwen",
    )
    NO_ECHO_MODELS = (  # weird
        "gpt_oss",
        "orion",
        "xglm",
    )

    GEN_KWARGS = {
        "max_new_tokens": 10,
        "min_new_tokens": 10,
        "do_sample": False,
        "num_beams": 1,
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_outputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        echo = model_arch not in self.NO_ECHO_MODELS
        use_cache = model_arch not in self.NO_CACHE_MODELS
        trust_remote_code = model_arch in self.REMOTE_CODE_MODELS

        set_seed(42)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            set_seed(42)
            main_export(
                model_name_or_path=model_id,
                task="text-generation-with-past",
                trust_remote_code=trust_remote_code,
                convert_tokenizer=True,
                output=tmpdirname,
            )
            optimum_model = OVModelForCausalLM.from_pretrained(
                tmpdirname, trust_remote_code=trust_remote_code, device=OPENVINO_DEVICE, ov_config=F32_CONFIG
            )
            genai_model = LLMPipeline(tmpdirname, device=OPENVINO_DEVICE, **F32_CONFIG)

        prompt = "Paris is the capital of"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, use_cache=use_cache, **self.GEN_KWARGS)
        transformers_output = tokenizer.decode(transformers_ids[0], skip_special_tokens=True)

        optimum_ids = optimum_model.generate(**inputs, use_cache=use_cache, **self.GEN_KWARGS)
        optimum_output = tokenizer.decode(optimum_ids[0], skip_special_tokens=True)

        genai_output = genai_model.generate(
            prompt, echo=echo, apply_chat_template=False, ignore_eos=True, **self.GEN_KWARGS
        )

        if not echo:
            # if echo is not supported, trim the prompt from the outputs and trim spaces
            # NOTE: this is an approximation, as detokenize(prompt_ids + generated_ids) - prompt != detokenize(generated_ids)
            transformers_output = transformers_output[len(prompt) :].strip()
            optimum_output = optimum_output[len(prompt) :].strip()

        # assert they are not empty
        self.assertTrue(transformers_output)
        self.assertTrue(optimum_output)
        self.assertTrue(genai_output)

        # compare outputs
        self.assertEqual(transformers_output, optimum_output)
        self.assertEqual(transformers_output, genai_output)


class VLMPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "llava",
        "llava_next",
        "llava_next_video",
        # "minicpmv", # output is truncated for some reason
        "qwen2_vl",
    )
    if is_transformers_version(">=", "4.46.0"):
        SUPPORTED_ARCHITECTURES += ("llava_next_mistral",)
        if is_transformers_version("<", "4.52.0"):
            SUPPORTED_ARCHITECTURES += ("minicpmo",)
        if is_transformers_version("<", "4.54.0"):
            SUPPORTED_ARCHITECTURES += ("llava-qwen2", "phi3_v")
    if is_transformers_version(">=", "4.49.0"):
        SUPPORTED_ARCHITECTURES += ("qwen2_5_vl",)
        if is_transformers_version("<", "4.54.0"):
            SUPPORTED_ARCHITECTURES += ("phi4mm",)
    if is_transformers_version(">=", "4.49"):
        SUPPORTED_ARCHITECTURES += ("gemma3",)

    REMOTE_CODE_MODELS = (
        "minicpmv",
        "minicpmo",
        "llava-qwen2",
        "phi3_v",
        "phi4mm",
    )

    GEN_KWARGS = {
        "max_new_tokens": 10,
        "min_new_tokens": 10,
        "do_sample": False,
        "num_beams": 1,
    }

    IMAGE = Image.open(requests.get(TEST_IMAGE_URL, stream=True).raw).convert("RGB")

    def _get_model_class(self, model_arch):
        if is_transformers_version(">=", "4.46") and model_arch in {
            "llava",
            "llava_next",
            "llava_next_mistral",
            "qwen2_vl",
            "qwen2_5_vl",
            "gemma3",
        }:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        elif model_arch == "llava_next_video":
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq
        elif model_arch == "llava":
            from transformers import LlavaForConditionalGeneration

            return LlavaForConditionalGeneration
        elif model_arch in {"llava_next", "llava_next_mistral"}:
            from transformers import LlavaNextForConditionalGeneration

            return LlavaNextForConditionalGeneration
        elif model_arch == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        else:
            return AutoModelForCausalLM

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_outputs(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        trust_remote_code = model_arch in self.REMOTE_CODE_MODELS

        set_seed(42)
        transformers_class = self._get_model_class(model_arch)
        transformers_model = transformers_class.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            set_seed(42)
            main_export(
                model_name_or_path=model_id,
                trust_remote_code=trust_remote_code,
                task="image-text-to-text",
                convert_tokenizer=True,
                output=tmpdirname,
            )
            optimum_model = OVModelForVisualCausalLM.from_pretrained(
                tmpdirname, device=OPENVINO_DEVICE, ov_config=F32_CONFIG, trust_remote_code=trust_remote_code
            )
            genai_model = VLMPipeline(tmpdirname, device=OPENVINO_DEVICE, **F32_CONFIG)

        image = self.IMAGE
        prompt = "A photo of a cat sitting on a"
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        inputs = optimum_model.preprocess_inputs(
            text=prompt, image=image, tokenizer=tokenizer, processor=processor, config=config
        )

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, **self.GEN_KWARGS)
        transformers_output = tokenizer.decode(transformers_ids[0], skip_special_tokens=True)

        optimum_ids = optimum_model.generate(**inputs, **self.GEN_KWARGS)
        optimum_output = tokenizer.decode(optimum_ids[0], skip_special_tokens=True)

        genai_output = genai_model.generate(
            prompt, images=[ov.Tensor(np.array(image))], ignore_eos=True, **self.GEN_KWARGS
        ).texts[0]

        full_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        transformers_output = transformers_output[len(full_prompt) :].strip()
        optimum_output = optimum_output[len(full_prompt) :].strip()

        # assert they are not empty
        self.assertTrue(transformers_output)
        self.assertTrue(optimum_output)
        self.assertTrue(genai_output)

        # compare outputs
        self.assertEqual(transformers_output, optimum_output)
        self.assertEqual(transformers_output, genai_output)


class Speeh2TextPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("whisper",)

    GEN_KWARGS = {
        "max_new_tokens": 10,
        "min_new_tokens": 10,
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
            genai_model = WhisperPipeline(tmpdirname, device=OPENVINO_DEVICE, **F32_CONFIG)

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


class Text2SpeechPipelineTestCase(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("speecht5",)
    VOCODER = "fxmarty/speecht5-hifigan-tiny"

    GEN_KWARGS = {
        "max_new_tokens": 10,
        "min_new_tokens": 10,
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
            genai_model = Text2SpeechPipeline(tmpdirname, device=OPENVINO_DEVICE, **F32_CONFIG)

        text = "Hello, how are you?"
        processor = AutoProcessor.from_pretrained(model_id)
        speaker_embeddings = self._generate_speaker_embedding()
        vocoder = self._get_vocoder(self.VOCODER, model_arch).eval()
        inputs = processor(text=text, return_tensors="pt")
        inputs["speaker_embeddings"] = speaker_embeddings

        with torch.no_grad():
            transformers_output = transformers_model.generate(**inputs, **self.GEN_KWARGS, vocoder=vocoder)
            transformers_output = transformers_output.squeeze(0)  # collapse batch dimension (if any)

        optimum_output = optimum_model.generate(**inputs, **self.GEN_KWARGS)
        optimum_output = optimum_output.squeeze(0)  # collapse batch dimension (if any)

        genai_output = genai_model.generate(text, **self.GEN_KWARGS).speeches[0]
        genai_output = torch.from_numpy(genai_output.data).squeeze(0)  # collapse batch dimension (if any)

        torch.testing.assert_close(transformers_output, optimum_output, rtol=1e-2, atol=1e-4)
        torch.testing.assert_close(transformers_output, genai_output, rtol=1e-2, atol=1e-4)


class LLMPipelineWithEagle3TestCase(unittest.TestCase):
    GEN_KWARGS = {
        "max_new_tokens": 10,
        "min_new_tokens": 10,
        "do_sample": False,
        "num_beams": 1,
    }

    @parameterized.expand(EAGLE3_MODELS.items())
    def test_compare_outputs(self, model_arch, model_pair):
        if is_transformers_version("<", "4.54"):
            self.skipTest("Eagle3 requires transformers >= 4.54")
        if is_openvino_version("<", "2026.0"):
            self.skipTest("Eagle3 requires openvino-genai >= 2026.0")

        draft_model_id, target_model_id = model_pair
        trust_remote_code = model_arch in REMOTE_CODE_MODELS

        # export main and draft eagle3 models and initialize OV LLM pipelines w/o Eagle3
        with tempfile.TemporaryDirectory() as draft_model_path, tempfile.TemporaryDirectory() as main_model_path:
            main_export(
                model_name_or_path=draft_model_id,
                task="text-generation-with-past",
                trust_remote_code=trust_remote_code,
                convert_tokenizer=False,
                output=draft_model_path,
            )
            main_export(
                model_name_or_path=target_model_id,
                task="text-generation-with-past",
                convert_tokenizer=True,
                output=main_model_path,
            )

            ov_draft_model = draft_model(draft_model_path, "CPU")
            ov_eagle3_pipe = LLMPipeline(main_model_path, OPENVINO_DEVICE, draft_model=ov_draft_model, **F32_CONFIG)
            ov_pipe = LLMPipeline(main_model_path, OPENVINO_DEVICE, **F32_CONFIG)

        prompt = "Paris is the capital of"
        genai_eagle3_output = ov_eagle3_pipe.generate(
            prompt, echo=True, apply_chat_template=False, ignore_eos=True, **self.GEN_KWARGS
        )
        genai_output = ov_pipe.generate(
            prompt, echo=True, apply_chat_template=False, ignore_eos=True, **self.GEN_KWARGS
        )

        # assert they are not empty
        self.assertTrue(genai_eagle3_output)
        self.assertTrue(genai_output)

        # compare outputs
        self.assertEqual(genai_eagle3_output, genai_output)
