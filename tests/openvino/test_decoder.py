import copy
import gc
import os
import platform
import unittest

import pytest
import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig, pipeline, set_seed
from transformers.testing_utils import slow
from utils_tests import (
    F32_CONFIG,
    MODEL_NAMES,
    OPENVINO_DEVICE,
    SEED,
    get_num_sdpa,
    mock_torch_cuda_is_available,
    patch_awq_for_inference,
)

from optimum.exporters.openvino.model_patcher import patch_update_causal_mask
from optimum.intel import OVModelForCausalLM, OVModelForSequenceClassification
from optimum.intel.openvino.utils import _print_compiled_model_properties
from optimum.intel.pipelines import pipeline as optimum_pipeline
from optimum.intel.utils.import_utils import is_openvino_version, is_transformers_version


if is_transformers_version(">=", "4.55"):
    from transformers import Mxfp4Config


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
        "gemma",
        "olmo",
        "stablelm",
        "starcoder2",
        "dbrx",
        "cohere",
        "qwen2",
        "qwen2_moe",
        "arctic",
        "phi3",
        "gemma2",
        "exaone",
        "granite",
        "granite-moe",
    )

    SUPPORTED_SSM_ARCHITECTURES = ("mamba", "falcon-mamba")

    SUPPORTED_ARCHITECTURES += SUPPORTED_SSM_ARCHITECTURES

    if is_transformers_version(">=", "4.46.0"):
        SUPPORTED_ARCHITECTURES += ("glm", "mistral-nemo", "minicpm3", "phi3-moe")
        # openvino 2025.0 required for disabling check_trace
        if is_openvino_version(">=", "2025.0"):
            SUPPORTED_ARCHITECTURES += ("deepseek",)

        # gptq and awq install disabled for windows test environment
        if platform.system() != "Windows":
            SUPPORTED_ARCHITECTURES += ("opt_gptq",)

        # autoawq install disabled for windows test environment
        if is_openvino_version(">=", "2024.6.0") and platform.system() != "Windows":
            SUPPORTED_ARCHITECTURES += ("mixtral_awq",)

    if is_transformers_version(">", "4.49"):
        SUPPORTED_ARCHITECTURES += ("gemma3_text",)

    if is_transformers_version(">=", "4.51.0"):
        SUPPORTED_ARCHITECTURES += ("qwen3", "qwen3_moe")

    if is_transformers_version(">=", "4.51.3"):
        SUPPORTED_ARCHITECTURES += ("glm4",)

    if is_transformers_version(">=", "4.53.0"):
        SUPPORTED_ARCHITECTURES += ("arcee",)

    if is_transformers_version(">=", "4.54.0"):
        # remote code models differs after transformers v4.54
        SUPPORTED_ARCHITECTURES += ("exaone4",)
        SUPPORTED_ARCHITECTURES = tuple(set(SUPPORTED_ARCHITECTURES) - {"minicpm", "minicpm3", "arctic", "deepseek"})

    if is_transformers_version(">=", "4.55.0"):
        SUPPORTED_ARCHITECTURES += ("gpt_oss", "gpt_oss_mxfp4")

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
        "exaone4",
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
        "mistral": 2,
        "mixtral": 2,
        "mpt": 5,
        "opt": 5 if is_transformers_version(">=", "4.46.0") else 0,
        "pegasus": 2,
        "qwen": 2,
        "phi": 2,
        "internlm2": 4,
        "falcon": 2,
        "falcon-40b": 2,
        "persimmon": 2,
        "biogpt": 5,
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
        "qwen2_moe": 4,
        "arctic": 4,
        "phi3": 2,
        "gemma2": 4,
        "exaone": 8,
        "exaone4": 1,
        "granite": 6,
        "granite-moe": 6,
        "glm": 28,
        "mistral-nemo": 8,
        "minicpm3": 6,
        "phi3-moe": 2,
        "deepseek": 2,
        "opt_gptq": 12,
        "mixtral_awq": 2,
        "gemma3_text": 2,
        "glm4": 2,
        "qwen3": 2,
        "qwen3_moe": 2,
        "mamba": 0,
        "falcon-mamba": 0,
        "arcee": 2,
    }

    # TODO: remove gptq/awq from here
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
        if model_arch == "gemma2":
            model_kwargs["attn_implementation"] = "sdpa"

        ov_model = OVModelForCausalLM.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE, **model_kwargs
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        self.assertTrue(ov_model.use_cache)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS)
        tokens = tokenizer("This is a sample output", return_tensors="pt")

        ov_outputs = ov_model(**tokens)
        self.assertTrue("logits" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits, torch.Tensor)
        if model_arch in self.SUPPORTED_SSM_ARCHITECTURES:
            from optimum.intel.openvino.modeling_decoder import OVMambaCache

            self.assertTrue("cache_params" in ov_outputs)
            self.assertIsInstance(ov_outputs.cache_params, OVMambaCache)
            is_stateful = ov_model.config.model_type not in not_stateful
            self.assertEqual(ov_model.stateful, is_stateful)
            if is_stateful:
                self.assertIsInstance(ov_outputs.cache_params.conv_states, list)
                self.assertIsInstance(ov_outputs.cache_params.ssm_states, list)
                self.assertTrue(
                    len(ov_outputs.cache_params.conv_states) > 0 and len(ov_outputs.cache_params.ssm_states) > 0
                )
        else:
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
            model_kwargs["torch_dtype"] = torch.float32

        # the mxfp4 model will be dequantized to bf16 by the Mxfp4HfQuantizer, we later cast it to fp32
        if "mxfp4" in model_arch:
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

        set_seed(SEED)
        with mock_torch_cuda_is_available("awq" in model_arch or "gptq" in model_arch):
            transformers_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if model_arch in ["qwen", "arctic", "chatglm4", "gpt_oss_mxfp4"]:
            transformers_model.to(torch.float32)

        with torch.no_grad():
            with patch_awq_for_inference("awq" in model_arch):
                transformers_outputs = transformers_model(**tokens)

        # Compare tensor outputs
        atol = 3e-3 if model_arch in ["minicpm", "qwen2-moe"] else 1e-4
        # quantized models have different logits value range
        if "awq" not in model_arch and "gptq" not in model_arch:
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
            num_beams=1 if model_arch == "chatglm4" else 2,
            do_sample=False,
        )

        ov_outputs = ov_model.generate(**tokens, generation_config=gen_config)

        # TODO: add back once https://huggingface.co/katuni4ka/tiny-random-minicpm3/discussions/1 merged (for all models) as current mdoeling incompatible with transformers >= v4.49
        if model_arch in {"deepseek"} and is_transformers_version(">=", "4.49"):
            self.skipTest("Incompatible modeling code")

        additional_inputs = {}
        # gemma2 does not support dynamic cache, it is unfair to compare dynamic cache result vs hybrid cache,
        # align cache representation in torch model
        if model_arch in {"gemma2", "gemma3_text"}:
            patch_update_causal_mask(transformers_model, "4.43.0")
            transformers_model._supports_cache_class = True
            transformers_model.generation_config.cache_implementation = None
            from transformers.cache_utils import DynamicCache

            additional_inputs = {"past_key_values": DynamicCache()}

        elif model_arch in {
            "aquila",
            "aquila2",
            "baichuan2",
            "baichuan2-13b",
            "decilm",
            "internlm",
            "internlm2",
            "jais",
            "orion",
            "xverse",
        }:
            additional_inputs = {"use_cache": False}
        set_seed(SEED)
        with patch_awq_for_inference("awq" in model_arch):
            transformers_outputs = transformers_model.generate(
                **tokens, generation_config=gen_config, **additional_inputs
            )

        self.assertTrue(
            torch.allclose(ov_outputs, transformers_outputs),
            f"OV output {ov_outputs}\nTransformers output  {transformers_outputs}",
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
            model_kwargs = {"trust_remote_code": True}
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=model_arch in self.REMOTE_CODE_MODELS)

        if model_arch == "qwen":
            tokenizer._convert_tokens_to_ids = lambda x: 0

        additional_args = {}
        if is_transformers_version(">=", "4.51"):
            additional_args["use_model_defaults"] = False

        set_seed(SEED)
        model = OVModelForCausalLM.from_pretrained(
            model_id, use_cache=True, compile=False, device=OPENVINO_DEVICE, **model_kwargs
        )
        model.eval()
        model.config.encoder_no_repeat_ngram_size = 0
        model.to("cpu")
        model.half()
        model.compile()
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        inputs = "My name is Arthur and I live in"
        set_seed(SEED)
        outputs = pipe(inputs, min_new_tokens=5, max_new_tokens=5, **additional_args, do_sample=False)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(inputs in item["generated_text"] for item in outputs))
        ov_pipe = optimum_pipeline(
            "text-generation",
            model_id,
            accelerator="openvino",
            trust_remote_code=model_arch in self.REMOTE_CODE_MODELS,
            tokenizer=(
                # in older transformers versions, qwen tokenizer didn't have a _convert_tokens_to_ids
                # method, which made it fail during inference using pipelines
                tokenizer
                if is_transformers_version("<=", "4.46") and model_arch == "qwen"
                # in older transformers versions, remote code tokenizers (and granite/granite-moe)
                # were not loaded in pipelines because they were not registered in TOKENIZER_MAPPING
                else model_id
                if is_transformers_version("<=", "4.46")
                and model_arch in self.REMOTE_CODE_MODELS + ("granite", "granite-moe")
                else None
            ),
        )
        set_seed(SEED)
        ov_outputs = ov_pipe(inputs, min_new_tokens=5, max_new_tokens=5, **additional_args, do_sample=False)
        self.assertEqual(outputs[-1]["generated_text"], ov_outputs[-1]["generated_text"])
        del ov_pipe
        del pipe
        del model
        gc.collect()

    def test_model_and_decoder_same_device(self):
        model_id = MODEL_NAMES["gpt2"]
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
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

        model_with_pkv = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True, stateful=False, device=OPENVINO_DEVICE
        )
        outputs_model_with_pkv = model_with_pkv.generate(
            **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
        )
        del model_with_pkv

        model_without_pkv = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=False, device=OPENVINO_DEVICE
        )
        outputs_model_without_pkv = model_without_pkv.generate(
            **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
        )
        del model_without_pkv

        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH)

        model_stateful = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True, stateful=True, device=OPENVINO_DEVICE
        )
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
        model = OVModelForSequenceClassification.from_pretrained(
            MODEL_NAMES["bert"], export=True, device=OPENVINO_DEVICE
        )
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
        model_with_cache = OVModelForCausalLM.from_pretrained(
            model_id, stateful=False, use_cache=True, device=OPENVINO_DEVICE
        )
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
        model_with_cache = OVModelForCausalLM.from_pretrained(
            model_id, stateful=False, use_cache=True, device=OPENVINO_DEVICE
        )
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
            model_kwargs = {"trust_remote_code": True}

        # starting from transformers 4.45.0 gemma2 uses eager attention by default, while ov - sdpa
        if model_arch == "gemma2":
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
            model_id, export=True, use_cache=True, stateful=True, device=OPENVINO_DEVICE, **model_kwargs
        )
        set_seed(SEED)
        ov_model_stateless = OVModelForCausalLM.from_pretrained(
            model_id, export=True, use_cache=True, stateful=False, device=OPENVINO_DEVICE, **model_kwargs
        )
        if "awq" in model_arch or "gptq" in model_arch:
            # infer in FP32
            model_kwargs["torch_dtype"] = torch.float32

        # the mxfp4 model will be dequantized to bf16 by the Mxfp4HfQuantizer, we later cast it to fp32
        if "mxfp4" in model_arch:
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

        set_seed(SEED)
        with mock_torch_cuda_is_available("awq" in model_arch or "gptq" in model_arch):
            transformers_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if model_arch == "arctic" or "mxfp4" in model_arch:
            transformers_model.to(torch.float32)
        additional_inputs = {}
        # gemma2 does not support dynamic cache, it is unfair to compare dynamic cache result vs hybrid cache, align cache representation in torch model
        if model_arch in ["gemma2", "gemma3_text"]:
            patch_update_causal_mask(transformers_model, "4.43.0")
            transformers_model._supports_cache_class = True
            transformers_model.generation_config.cache_implementation = None
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenization_args = {}
        if model_arch == "gpt_neo":
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

            if model_arch in {"gemma2", "gemma3_text"}:
                from transformers.cache_utils import DynamicCache

                additional_inputs["past_key_values"] = DynamicCache()

            elif model_arch in {
                "aquila",
                "aquila2",
                "baichuan2",
                "baichuan2-13b",
                "decilm",
                "internlm",
                "internlm2",
                "jais",
                "orion",
                "xverse",
            }:
                additional_inputs["use_cache"] = False

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
            ov_model = OVModelForCausalLM.from_pretrained(
                model_id=model_id, export=True, torch_dtype=dtype, device=OPENVINO_DEVICE
            )
            ov_logits = ov_model(**test_input).logits
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_logits), ref_logits, atol=5e-3),
                f"values are not close for {dtype if dtype is not None else 'None'}, max diff = {torch.abs(ov_logits - ref_logits).max()}",
            )
