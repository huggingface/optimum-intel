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
import platform
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import open_clip
import openvino as ov
import pytest
import requests
import timm
import torch
from datasets import load_dataset
from evaluate import evaluator
from huggingface_hub import hf_hub_download, snapshot_download
from parameterized import parameterized
from PIL import Image
from sentence_transformers import SentenceTransformer
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
    AutoModelForTextToSpectrogram,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForZeroShotImageClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    Pix2StructForConditionalGeneration,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import slow
from transformers.utils import http_user_agent
from utils_tests import (
    MODEL_NAMES,
    TEST_IMAGE_URL,
    get_num_sdpa,
    mock_torch_cuda_is_available,
    patch_awq_for_inference,
)

from optimum.exporters.openvino.model_patcher import patch_update_causal_mask
from optimum.exporters.openvino.stateful import model_has_state
from optimum.intel import (
    OVDiffusionPipeline,
    OVFluxPipeline,
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
    OVModelForTextToSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVModelForVision2Seq,
    OVModelForVisualCausalLM,
    OVModelForZeroShotImageClassification,
    OVModelOpenCLIPForZeroShotImageClassification,
    OVSamModel,
    OVSentenceTransformer,
    OVStableDiffusionPipeline,
)
from optimum.intel.openvino import OV_DECODER_NAME, OV_DECODER_WITH_PAST_NAME, OV_ENCODER_NAME, OV_XML_FILE_NAME
from optimum.intel.openvino.modeling_base import OVBaseModel
from optimum.intel.openvino.modeling_seq2seq import OVDecoder, OVEncoder
from optimum.intel.openvino.modeling_timm import TimmImageProcessor
from optimum.intel.openvino.modeling_visual_language import (
    MODEL_PARTS_CLS_MAPPING,
    MODEL_TYPE_TO_CLS_MAPPING,
)
from optimum.intel.openvino.utils import (
    OV_LANGUAGE_MODEL_NAME,
    OV_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME,
    OV_TEXT_EMBEDDINGS_MODEL_NAME,
    OV_VISION_EMBEDDINGS_MODEL_NAME,
    OV_VISION_ENCODER_MODEL_NAME,
    TemporaryDirectory,
    _print_compiled_model_properties,
)
from optimum.intel.pipelines import pipeline as optimum_pipeline
from optimum.intel.utils.import_utils import (
    _langchain_hf_available,
    is_openvino_version,
    is_transformers_version,
)
from optimum.intel.utils.modeling_utils import _find_files_matching_pattern
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
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
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "llama",
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
        "chatglm4",
        "decilm",
        "arctic",
    )

    if is_transformers_version(">=", "4.40.0"):
        SUPPORTED_ARCHITECTURES += (
            "gemma",
            "olmo",
            "stablelm",
            "starcoder2",
            "dbrx",
            "cohere",
            "qwen2",
            "qwen2-moe",
        )

    if is_transformers_version(">=", "4.41.0"):
        SUPPORTED_ARCHITECTURES += ("phi3",)

    if is_transformers_version(">=", "4.43.0"):
        SUPPORTED_ARCHITECTURES += ("gemma2", "exaone")

    if is_transformers_version(">=", "4.44.0"):
        SUPPORTED_ARCHITECTURES += ("granite", "granite-moe")

    if is_transformers_version(">=", "4.46.0"):
        SUPPORTED_ARCHITECTURES += ("glm", "mistral-nemo", "minicpm3", "phi3-moe")
        # openvino 2025.0 required for disabling check_trace
        if is_openvino_version(">=", "2025.0"):
            SUPPORTED_ARCHITECTURES += ("deepseek",)

        # gptq and awq install disabled for windows test environment
        # if platform.system() != "Windows":
        # SUPPORTED_ARCHITECTURES += ("opt_gptq",)

        # autoawq install disabled for windows test environment
        # if is_openvino_version(">=", "2024.6.0") and platform.system() != "Windows":
        # SUPPORTED_ARCHITECTURES += ("mixtral_awq",)

    if is_transformers_version(">", "4.49"):
        SUPPORTED_ARCHITECTURES += ("gemma3-text",)

    if is_transformers_version(">=", "4.51.0"):
        SUPPORTED_ARCHITECTURES += ("qwen3", "qwen3-moe")

    if is_transformers_version(">=", "4.51.3"):
        SUPPORTED_ARCHITECTURES += ("glm4",)

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
        "chatglm4",
        "exaone",
        "decilm",
        "minicpm3",
        "deepseek",
    )

    EXPECTED_NUM_SDPA = {
        "bart": 2,
        "baichuan2": 2,
        "baichuan2-13b": 2,
        "gpt_bigcode": 5,
        "blenderbot": 2,
        "blenderbot-small": 2,
        "bloom": 5,
        "chatglm": 2,
        "codegen": 5,
        "codegen2": 2,
        "gpt2": 5,
        "gptj": 5,
        "gpt_neo": 4,
        "gpt_neox": 5,
        "llama": 2,
        "marian": 2,
        "minicpm": 4,
        "mistral": 2 if is_transformers_version(">=", "4.40.0") else 0,
        "mixtral": 2 if is_transformers_version(">=", "4.40.0") else 0,
        "mpt": 5,
        "opt": 5,
        "pegasus": 2,
        "qwen": 2,
        "phi": 2 if is_transformers_version(">=", "4.40.0") else 0,
        "internlm2": 4,
        "falcon": 2,
        "falcon-40b": 2,
        "persimmon": 2,
        "biogpt": 5 if is_transformers_version(">=", "4.45.0") else 0,
        "aquila": 2,
        "aquila2": 2,
        "xverse": 2,
        "internlm": 2,
        "jais": 2,
        "chatglm4": 6,
        "decilm": 4,
        "gemma": 1,
        "olmo": 2,
        "stablelm": 2,
        "starcoder2": 2,
        "dbrx": 2,
        "cohere": 2,
        "qwen2": 2,
        "qwen2-moe": 4,
        "arctic": 4,
        "phi3": 2,
        "gemma2": 4,
        "exaone": 8,
        "granite": 6,
        "granite-moe": 6,
        "glm": 28,
        "mistral-nemo": 8,
        "minicpm3": 6,
        "phi3-moe": 2,
        "deepseek": 2,
        "opt_gptq": 12,
        "mixtral_awq": 2,
        "gemma3-text": 2,
        "glm4": 2,
        "qwen3": 2,
        "qwen3-moe": 2,
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]

        not_stateful = []
        if is_openvino_version("<", "2024.0"):
            not_stateful.append("mixtral")

        if is_openvino_version("<", "2024.1"):
            not_stateful.extend(["llama", "gemma", "gpt_bigcode"])

        set_seed(SEED)

        model_kwargs = {}
        if model_arch in self.REMOTE_CODE_MODELS:
            model_kwargs = {"trust_remote_code": True}

        # starting from transformers 4.45.0 gemma2 uses eager attention by default, while ov - sdpa
        if model_arch == "gemma2" and is_transformers_version(">=", "4.45.0"):
            model_kwargs["attn_implementation"] = "sdpa"

        ov_model = OVModelForCausalLM.from_pretrained(model_id, export=True, ov_config=F32_CONFIG, **model_kwargs)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        self.assertTrue(ov_model.use_cache)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS)
        tokens = tokenizer("This is a sample output", return_tensors="pt")

        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        self.assertTrue("past_key_values" in ov_outputs)
        self.assertIsInstance(ov_outputs.past_key_values, tuple)
        is_stateful = ov_model.config.model_type not in not_stateful
        self.assertEqual(ov_model.stateful, is_stateful)
        if is_stateful:
            self.assertTrue(len(ov_outputs.past_key_values) == 1 and len(ov_outputs.past_key_values[0]) == 0)

        expected_num_sdpa = self.EXPECTED_NUM_SDPA.get(model_arch, 0)
        num_sdpa = get_num_sdpa(ov_model.model)
        self.assertEqual(
            expected_num_sdpa,
            num_sdpa,
            f"Expected number of SDPA {expected_num_sdpa}, while model contains {num_sdpa}",
        )

        if "awq" in model_arch or "gptq" in model_arch:
            # infer in FP32
            model_kwargs["torch_dtype"] = torch.float32

        set_seed(SEED)
        with mock_torch_cuda_is_available("awq" in model_arch or "gptq" in model_arch):
            transformers_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if model_arch in ["qwen", "arctic", "chatglm4"]:
            transformers_model.to(torch.float32)

        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Compare tensor outputs
        atol = 3e-3 if model_arch in ["minicpm", "qwen2-moe"] else 1e-4
        # quantized models have different logits value range
        if "awq" not in model_arch and "gptq" not in model_arch:
            print(torch.abs(ov_outputs.logits - transformers_outputs.logits).max())
            print(atol)
            self.assertTrue(torch.allclose(ov_outputs.logits, transformers_outputs.logits, equal_nan=True, atol=atol))

        # Qwen tokenizer does not support padding
        if model_arch in ["qwen"]:
            return

        if model_arch not in ["chatglm", "chatglm4", "persimmon"]:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if model_arch == "persimmon":
            tokenizer.pad_token_id = tokenizer.bos_token_id
        # Compare batched generation
        tokenizer.padding_side = "left"
        tokens = tokenizer(["Today is a nice day and I am longer", "This is me"], return_tensors="pt", padding=True)
        ov_model.generation_config.eos_token_id = None
        transformers_model.generation_config.eos_token_id = None
        ov_model.config.eos_token_id = None
        transformers_model.config.eos_token_id = None
        gen_config = GenerationConfig(
            max_new_tokens=30,
            min_new_tokens=30,
            num_beams=2 if model_arch != "chatglm4" else 1,
            do_sample=False,
            eos_token_id=None,
        )
        if is_transformers_version(">=", "4.51"):
            tokens["use_model_defaults"] = False

        ov_outputs = ov_model.generate(**tokens, generation_config=gen_config)

        # TODO: add back once https://huggingface.co/katuni4ka/tiny-random-minicpm3/discussions/1 merged (for all models) as current mdoeling incompatible with transformers >= v4.49
        if model_arch in {"deepseek"} and is_transformers_version(">=", "4.49"):
            self.skipTest("Incompatible modeling code")

        additional_inputs = {}
        # gemma2 does not support dynamic cache, it is unfair to compare dynamic cache result vs hybrid cache,
        # align cache representation in torch model
        if model_arch in ["gemma2", "gemma3-text"]:
            patch_update_causal_mask(transformers_model, "4.43.0")
            transformers_model._supports_cache_class = True
            transformers_model.generation_config.cache_implementation = None

            from transformers.cache_utils import DynamicCache
            additional_inputs = {"past_key_values": DynamicCache()}

        with patch_awq_for_inference("awq" in model_arch):
            transformers_outputs = transformers_model.generate(
                **tokens, generation_config=gen_config, **additional_inputs
            )
        print(f"ov_outputs: {ov_outputs}")
        print(f"transformers_outputs: {transformers_outputs}")
        self.assertTrue(
            torch.allclose(ov_outputs, transformers_outputs),
            "OV output {ov_outputs}\nTransformers output  {transformers_output}",
        )

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

        additional_args = {}
        if is_transformers_version(">=", "4.51"):
            additional_args["use_model_defaults"] = False

        model = OVModelForCausalLM.from_pretrained(model_id, use_cache=False, compile=False, **model_kwargs)
        model.eval()
        model.config.encoder_no_repeat_ngram_size = 0
        model.to("cpu")
        model.half()
        model.compile()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        inputs = "My name is Arthur and I live in"
        set_seed(SEED)
        outputs = pipe(inputs, max_new_tokens=5, **additional_args, do_sample=False)
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
        ov_outputs = ov_pipe(inputs, max_new_tokens=5, **additional_args, do_sample=False)
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
        del model_with_pkv

        model_without_pkv = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=False)
        outputs_model_without_pkv = model_without_pkv.generate(
            **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
        )
        del model_without_pkv

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH)

        model_stateful = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True, stateful=True)
        outputs_model_stateful = model_stateful.generate(
            **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
        )
        self.assertTrue(torch.equal(outputs_model_without_pkv, outputs_model_stateful))

        logits = model_stateful(**tokens).logits
        copy_logits = copy.deepcopy(logits)
        tokens = tokenizer("Input sample", return_tensors="pt")
        model_stateful(**tokens).logits
        self.assertTrue(torch.equal(copy_logits, logits))
        del model_stateful
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
        model_with_cache = OVModelForCausalLM.from_pretrained(model_id, stateful=False, use_cache=True)
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
        model_with_cache = OVModelForCausalLM.from_pretrained(model_id, stateful=False, use_cache=True)
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

        # starting from transformers 4.45.0 gemma2 uses eager attention by default, while ov - sdpa
        if model_arch == "gemma2" and is_transformers_version(">=", "4.45.0"):
            model_kwargs["attn_implementation"] = "sdpa"

        # Qwen tokenizer does not support padding, chatglm, glm4 testing models produce nan that incompatible with beam search
        if model_arch in ["qwen", "chatglm", "chatglm4"]:
            return

        # TODO: add back once https://huggingface.co/katuni4ka/tiny-random-minicpm3/discussions/1 merged (for all models) as current mdoeling incompatible with transformers >= v4.49
        if model_arch in {"deepseek"} and is_transformers_version(">=", "4.49"):
            self.skipTest("Incompatible modeling code")

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

        if model_arch in ["minicpm", "internlm2"]:
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
        set_seed(SEED)
        ov_model_stateful = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True, stateful=True, **model_kwargs
        )
        set_seed(SEED)
        ov_model_stateless = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True, stateful=False, **model_kwargs
        )
        if "awq" in model_arch or "gptq" in model_arch:
            # infer in FP32
            model_kwargs["torch_dtype"] = torch.float32

        set_seed(SEED)
        with mock_torch_cuda_is_available("awq" in model_arch or "gptq" in model_arch):
            transformers_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if model_arch == "arctic":
            transformers_model.to(torch.float32)
        additional_inputs = {}
        # gemma2 does not support dynamic cache, it is unfair to compare dynamic cache result vs hybrid cache, align cache representation in torch model
        if model_arch in ["gemma2", "gemma3-text"]:
            patch_update_causal_mask(transformers_model, "4.43.0")
            transformers_model._supports_cache_class = True
            transformers_model.generation_config.cache_implementation = None
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenization_args = {}
        if is_transformers_version(">=", "4.45") and model_arch == "gpt_neo":
            tokenization_args["padding_side"] = "left"
        tokens = tokenizer(
            ["Today is a nice day and I am longer", "This is me"],
            return_tensors="pt",
            padding=True,
            **tokenization_args,
        )
        ov_model_stateful.generation_config.eos_token_id = None
        ov_model_stateful.generation_config.forced_eos_token_id = None
        ov_model_stateful.generation_config.encoder_no_repeat_ngram_size = None
        ov_model_stateful.generation_config.do_sample = False
        ov_model_stateless.generation_config.eos_token_id = None
        ov_model_stateless.generation_config.forced_eos_token_id = None
        ov_model_stateless.generation_config.encoder_no_repeat_ngram_size = None
        ov_model_stateless.generation_config.do_sample = False
        transformers_model.generation_config.eos_token_id = None
        transformers_model.generation_config.forced_eos_token_id = None
        transformers_model.generation_config.encoder_no_repeat_ngram_size = None
        transformers_model.generation_config.do_sample = False
        ov_model_stateful.config.eos_token_id = None
        ov_model_stateless.config.eos_token_id = None
        transformers_model.config.eos_token_id = None

        if is_transformers_version(">=", "4.51"):
            additional_inputs["use_model_defaults"] = False

        for gen_config in gen_configs:
            if gen_config.do_sample and model_arch in ["baichuan2-13b", "olmo"]:
                continue
            if gen_config.num_beams > 1 and is_transformers_version(">=", "4.51.0") and model_arch in ["mixtral_awq"]:
                continue
            set_seed(SEED)

            if model_arch in ["gemma2", "gemma3-text"]:
                from transformers.cache_utils import DynamicCache

                additional_inputs["past_key_values"] = DynamicCache()
            with patch_awq_for_inference("awq" in model_arch):
                transformers_outputs = transformers_model.generate(
                    **tokens, generation_config=gen_config, **additional_inputs
                )
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

    def test_load_with_different_dtype(self):
        set_seed(SEED)
        model_id = MODEL_NAMES["llama"]
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_id,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        texts = ["this is a simple input"]
        test_input = tokenizer(texts, return_tensors="pt")

        ref_logits = pt_model(**test_input).logits
        torch_dtypes = [None, "auto", "float32", torch.float16]
        if is_openvino_version(">", "2024.2.0"):
            torch_dtypes.append("bfloat16")

        for dtype in torch_dtypes:
            ov_model = OVModelForCausalLM.from_pretrained(model_id=model_id, export=True, torch_dtype=dtype)
            ov_logits = ov_model(**test_input).logits
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_logits), ref_logits, atol=5e-3),
                f"values are not close for {dtype if dtype is not None else 'None'}, max diff = {torch.abs(ov_logits - ref_logits).max()}",
            )





