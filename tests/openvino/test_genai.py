"""
Test OpenVINO GenAI inference on models exported with optimum-intel

- OpenVINO device can be set by environment variable OPENVINO_TEST_DEVICE
  - For NPU, Text2Speech test is not supported; for LLM and VLM only a limited list of models is currently supported. This will be expanded. Only the latest supported transformers/OpenVINO/optimum-intel versions are tested.
  - For GPU, there are known failed tests on some GPUs. This is under investigation.
"""

import gc
import os
import shutil
import tempfile
import traceback as traceback_mod
import unittest
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
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
from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING
from optimum.intel.utils.import_utils import is_openvino_version
from optimum.utils import is_transformers_version

# NPU does not support f32 inference
TEST_CONFIG = {"CACHE_DIR": ""} if OPENVINO_DEVICE == "NPU" else {**F32_CONFIG, "CACHE_DIR": ""}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_temp_dirs = []  # Collect temp dirs for batch cleanup after all tests finish


class _ClearFramesPlugin:
    """Pytest plugin that clears traceback frames and deletes temp dirs after all tests finish.

    On Windows, when a test fails, pytest holds the exception traceback which keeps
    references to all local variables in the test frame — including OpenVINO model
    objects that hold file handles on temp directory contents.

    Clearing frames between tests can cause access violations in subsequent tests. Instead, all
    tracebacks are collected and cleared once at session end, then temp dirs are deleted.
    """

    def __init__(self):
        self._pending_tracebacks = []

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        yield
        if call.excinfo is not None and call.excinfo.value is not None:
            tb = call.excinfo.value.__traceback__
            if tb is not None:
                self._pending_tracebacks.append(tb)

    def pytest_sessionfinish(self, session, exitstatus):
        for tb in self._pending_tracebacks:
            traceback_mod.clear_frames(tb)
        self._pending_tracebacks.clear()
        gc.collect()
        for tmp_path in _temp_dirs:
            shutil.rmtree(tmp_path, ignore_errors=True)
        _temp_dirs.clear()


_clear_frames_plugin = _ClearFramesPlugin()


@pytest.fixture(autouse=True)
def temp_dir_fixture(request):
    """
    Provides a temporary directory as self.temp_dir, cleaned up after all tests finish.

    Immediate cleanup is attempted but may fail on Windows if OpenVINO model objects
    still hold file handles. Failed directories are cleaned up at session end after
    traceback frames are cleared and objects are garbage collected.
    """
    if not request.config.pluginmanager.has_plugin("_clear_frames"):
        request.config.pluginmanager.register(_clear_frames_plugin, "_clear_frames")
    tmp_path = tempfile.mkdtemp()
    request.instance.temp_dir = tmp_path
    yield
    try:
        shutil.rmtree(tmp_path)
    except (PermissionError, OSError):
        _temp_dirs.append(tmp_path)


class LLMPipelineTestCase(unittest.TestCase):
    ALL_SUPPORTED_ARCHITECTURES = (
        "gpt_bigcode",
        "bloom",
        "codegen",
        "gpt2",
        "gptj",
        "gpt_neox",
        "llama",
        "mistral",
        "mixtral",
        "phi",
        "falcon",
        "persimmon",
        "xglm",
        "gemma",
        "olmo",
        "stablelm",
        "starcoder2",
        "cohere",
        "qwen2",
        "qwen2_moe",
        "phi3",
        "gemma2",
        "granite",
        "granitemoe",
    )

    # to be expanded, other architectures work on NPU too
    # qwen2, phi and phi3 tests are flaky on NPU, not including for now
    NPU_SUPPORTED_ARCHITECTURES = ("gpt2", "glm", "opt", "qwen3_moe", "gpt_oss")

    # min versions
    if is_transformers_version(">=", "4.46.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("glm", "mistral-nemo", "opt")
        if is_transformers_version("<", "5"):
            ALL_SUPPORTED_ARCHITECTURES += ("phimoe",)
    if is_transformers_version(">=", "4.48.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("cohere2",)
    if is_transformers_version(">=", "4.50"):
        ALL_SUPPORTED_ARCHITECTURES += ("gemma3_text",)
    if is_transformers_version(">=", "4.51.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("qwen3", "qwen3_moe")
    if is_transformers_version(">=", "4.51.3"):
        ALL_SUPPORTED_ARCHITECTURES += ("glm4",)
    if is_transformers_version(">=", "4.53.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("arcee",)
    if is_transformers_version(">=", "4.54.0") and is_transformers_version("<", "5"):
        ALL_SUPPORTED_ARCHITECTURES += ("exaone4",)
    if is_transformers_version(">=", "4.55.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("gpt_oss",)

    # max versions
    if is_transformers_version("<", "4.54.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("minicpm", "minicpm3", "arctic", "deepseek")
    if is_transformers_version("<", "4.56.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("chatglm", "chatglm4", "qwen")

    if is_transformers_version("<", "5"):
        ALL_SUPPORTED_ARCHITECTURES += (
            # remote modeling incompatible with v5
            "codegen2",
            "exaone",
            "decilm",
            "internlm2",
            "orion",
            "aquila2",
            "jais",
            # remote modeling code failing with v5
            "aquila",
            "internlm",
            # TODO: add fix for v5 and update MAX_TRANSFORMERS_VERSION accordingly
            "dbrx",
            # "phimoe",
        )

    # for now we do not test NPU with old transformers versions
    SUPPORTED_ARCHITECTURES = NPU_SUPPORTED_ARCHITECTURES if OPENVINO_DEVICE == "NPU" else ALL_SUPPORTED_ARCHITECTURES

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

    GEN_KWARGS = {
        "max_new_tokens": 10,
        "min_new_tokens": 10,
        "do_sample": False,
        "num_beams": 1,
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_outputs(self, model_arch):
        if model_arch in ("xglm",) and is_openvino_version(">=", "2026.1.0"):
            self.skipTest("CVS-185350: OpenVINO 2026.1.0 inference results mismatch")
        if (
            model_arch in ("mixtral", "qwen2_moe", "qwen3_moe", "gpt_oss")
            and is_openvino_version(">=", "2026.1.0")
            and is_transformers_version(">=", "5.0.0")
        ):
            self.skipTest("CVS-185350: OpenVINO 2026.1.0 inference results mismatch")

        model_id = MODEL_NAMES[model_arch]
        use_cache = model_arch not in self.NO_CACHE_MODELS
        trust_remote_code = model_arch in self.REMOTE_CODE_MODELS

        set_seed(42)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

        set_seed(42)
        main_export(
            model_name_or_path=model_id,
            task="text-generation-with-past",
            trust_remote_code=trust_remote_code,
            convert_tokenizer=True,
            output=self.temp_dir,
        )
        genai_model = LLMPipeline(self.temp_dir, device=OPENVINO_DEVICE, **TEST_CONFIG)

        prompt = "Paris is the capital of"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, use_cache=use_cache, **self.GEN_KWARGS)
            transformers_ids = transformers_ids.squeeze()[input_len:]

        if OPENVINO_DEVICE != "NPU":
            optimum_model = OVModelForCausalLM.from_pretrained(
                self.temp_dir, trust_remote_code=trust_remote_code, device=OPENVINO_DEVICE, ov_config=TEST_CONFIG
            )
            optimum_ids = optimum_model.generate(**inputs, use_cache=use_cache, **self.GEN_KWARGS)
            optimum_ids = optimum_ids.squeeze()[input_len:]
            self.assertEqual(
                transformers_ids.squeeze().tolist(),
                optimum_ids.squeeze().tolist(),
                "Transformers ids and Optimum ids are not the same",
            )

        genai_ids = genai_model(
            ov.Tensor(inputs["input_ids"].numpy()), apply_chat_template=False, **self.GEN_KWARGS
        ).tokens[0]
        self.assertEqual(
            transformers_ids.tolist(), genai_ids, "Transformers ids and OpenVINO GenAI ids are not the same"
        )


class VLMPipelineTestCase(unittest.TestCase):
    ALL_SUPPORTED_ARCHITECTURES = (
        "llava_next",
        # "minicpmv", # output is truncated for some reason
        "qwen2_vl",
    )
    if is_transformers_version(">=", "4.46.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("llava_next_mistral",)
        if is_transformers_version("<", "4.52.0"):
            ALL_SUPPORTED_ARCHITECTURES += ("minicpmo",)
        if is_transformers_version("<", "4.54.0"):
            ALL_SUPPORTED_ARCHITECTURES += ("llava-qwen2", "phi3_v")
    if is_transformers_version(">=", "4.49.0"):
        ALL_SUPPORTED_ARCHITECTURES += ("qwen2_5_vl",)
        if is_transformers_version("<", "4.54.0"):
            ALL_SUPPORTED_ARCHITECTURES += ("phi4mm",)
    if is_transformers_version(">=", "4.50"):
        ALL_SUPPORTED_ARCHITECTURES += ("gemma3",)
    if is_transformers_version("<", "5"):
        ALL_SUPPORTED_ARCHITECTURES += ("llava", "llava_next_video")

    # for now we do not test NPU with old transformers versions
    NPU_SUPPORTED_ARCHITECTURES = ("qwen2_vl", "qwen2_5_vl")

    SUPPORTED_ARCHITECTURES = NPU_SUPPORTED_ARCHITECTURES if OPENVINO_DEVICE == "NPU" else ALL_SUPPORTED_ARCHITECTURES

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
            from transformers import LlavaNextVideoForConditionalGeneration

            return LlavaNextVideoForConditionalGeneration
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

        set_seed(42)
        main_export(
            model_name_or_path=model_id,
            trust_remote_code=trust_remote_code,
            task="image-text-to-text",
            convert_tokenizer=True,
            output=self.temp_dir,
        )
        genai_model = VLMPipeline(self.temp_dir, device=OPENVINO_DEVICE, **TEST_CONFIG)

        image = self.IMAGE
        prompt = "A photo of a cat sitting on a"
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        # On NPU, the optimum models cannot be loaded, so we use the preprocess_inputs method from the model class directly
        model_cls = MODEL_TYPE_TO_CLS_MAPPING[config.model_type]
        inputs = model_cls.preprocess_inputs(
            text=prompt, image=image, tokenizer=tokenizer, processor=processor, config=config
        )
        full_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, **self.GEN_KWARGS)
        transformers_output = tokenizer.decode(transformers_ids[0], skip_special_tokens=True)
        transformers_output = transformers_output[len(full_prompt) :].strip()

        if OPENVINO_DEVICE != "NPU":
            optimum_model = OVModelForVisualCausalLM.from_pretrained(
                self.temp_dir, device=OPENVINO_DEVICE, ov_config=TEST_CONFIG, trust_remote_code=trust_remote_code
            )
            optimum_ids = optimum_model.generate(**inputs, **self.GEN_KWARGS)
            optimum_output = tokenizer.decode(optimum_ids[0], skip_special_tokens=True)
            optimum_output = optimum_output[len(full_prompt) :].strip()
            self.assertTrue(optimum_output)
            self.assertEqual(transformers_output, optimum_output, "Transformers and Optimum outputs are not the same")

        # apply_chat_template is set to True because it is also set in preprocess_inputs()
        genai_output = genai_model.generate(
            prompt, images=[ov.Tensor(np.array(image))], ignore_eos=True, apply_chat_template=True, **self.GEN_KWARGS
        ).texts[0]

        # assert they are not empty
        self.assertTrue(transformers_output)
        self.assertTrue(genai_output)

        # compare outputs
        self.assertEqual(transformers_output, genai_output, "Transformers and OpenVINO GenAI outputs are not the same")


@pytest.mark.skipif(
    OPENVINO_DEVICE == "NPU" and is_transformers_version(">=", "5.0"),
    reason="Speech2Text test on NPU is only supported with transformers < 5.0",
)
class Speech2TextPipelineTestCase(unittest.TestCase):
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

        set_seed(42)
        main_export(
            model_name_or_path=model_id,
            task="automatic-speech-recognition-with-past",
            convert_tokenizer=True,
            output=self.temp_dir,
        )

        genai_model = WhisperPipeline(self.temp_dir, device=OPENVINO_DEVICE, **TEST_CONFIG)

        audio = self._get_audio()
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            transformers_ids = transformers_model.generate(**inputs, **self.GEN_KWARGS)
            transformers_output = tokenizer.decode(transformers_ids[0], skip_special_tokens=True)

        if OPENVINO_DEVICE != "NPU":
            optimum_model = OVModelForSpeechSeq2Seq.from_pretrained(
                self.temp_dir, device=OPENVINO_DEVICE, ov_config=TEST_CONFIG
            )
            optimum_ids = optimum_model.generate(**inputs, **self.GEN_KWARGS)
            optimum_output = tokenizer.decode(optimum_ids[0], skip_special_tokens=True)
            self.assertEqual(transformers_output, optimum_output)

        genai_output = genai_model.generate(inputs["input_features"].flatten().tolist(), **self.GEN_KWARGS).texts[0]

        self.assertEqual(transformers_output, genai_output)


@pytest.mark.skipif(OPENVINO_DEVICE == "NPU", reason="Text2Speech test is not yet supported on NPU")
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
        if model_arch in ("speecht5",) and is_openvino_version(">=", "2026.1.0"):
            self.skipTest("CVS-185350: OpenVINO 2026.1.0 inference results mismatch")
        model_id = MODEL_NAMES[model_arch]

        set_seed(42)
        transformers_model = AutoModelForTextToSpectrogram.from_pretrained(model_id).eval()

        set_seed(42)
        main_export(
            model_name_or_path=model_id,
            task="text-to-audio-with-past",
            model_kwargs={"vocoder": self.VOCODER},
            convert_tokenizer=True,
            output=self.temp_dir,
        )
        optimum_model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
            self.temp_dir, device=OPENVINO_DEVICE, ov_config=TEST_CONFIG
        )
        genai_model = Text2SpeechPipeline(self.temp_dir, device=OPENVINO_DEVICE, **TEST_CONFIG)

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

        torch.testing.assert_close(transformers_output, optimum_output, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(transformers_output, genai_output, rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(OPENVINO_DEVICE == "NPU", reason="Eagle3 test is not yet supported on NPU")
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
        draft_model_path = Path(self.temp_dir) / "draft_model"
        main_model_path = Path(self.temp_dir) / "main_model"
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

        prompt = "Paris is the capital of"

        # Phase 1: generate with Eagle3 speculative decoding
        ov_draft_model = draft_model(draft_model_path, "CPU")
        ov_eagle3_pipe = LLMPipeline(main_model_path, OPENVINO_DEVICE, draft_model=ov_draft_model, **TEST_CONFIG)
        genai_eagle3_output = str(
            ov_eagle3_pipe.generate(prompt, echo=True, apply_chat_template=False, ignore_eos=True, **self.GEN_KWARGS)
        )
        del ov_eagle3_pipe
        del ov_draft_model
        gc.collect()

        # Phase 2: generate without Eagle3
        ov_pipe = LLMPipeline(main_model_path, OPENVINO_DEVICE, **TEST_CONFIG)
        genai_output = str(
            ov_pipe.generate(prompt, echo=True, apply_chat_template=False, ignore_eos=True, **self.GEN_KWARGS)
        )
        del ov_pipe
        gc.collect()

        # assert they are not empty
        self.assertTrue(genai_eagle3_output)
        self.assertTrue(genai_output)

        # compare outputs
        self.assertEqual(genai_eagle3_output, genai_output)
