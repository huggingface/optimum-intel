import tempfile
import unittest

import torch
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    set_seed,
)
from transformers.utils import is_auto_awq_available, is_bitsandbytes_available
from utils_tests import IS_XPU_AVAILABLE, MODEL_NAMES

from optimum.intel import IPEXModelForCausalLM
from optimum.intel.utils.import_utils import is_torch_version
from optimum.utils.testing_utils import grid_parameters
from .test_modeling import Timer


SEED = 42
torch.use_deterministic_algorithms(True)
DEVICE = "xpu:0" if IS_XPU_AVAILABLE else "cpu"


class IPEXModelForCausalLMTest(unittest.TestCase):
    IPEX_MODEL_CLASS = IPEXModelForCausalLM
    SUPPORTED_ARCHITECTURES = (
        "bart",
        "gpt_bigcode",
        "blenderbot",
        "bloom",
        "codegen",
        "falcon",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "mistral",
        "llama2",
        "mpt",
        "opt",
        "phi",
        "qwen2",
    )
    IPEX_PATCHED_SUPPORTED_ARCHITECTURES = ("llama2", "falcon", "gpt2", "qwen2")
    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.0

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample",
            return_tensors="pt",
            return_token_type_ids=False if model_arch in ("llama2",) else None,
        ).to(DEVICE)
        inputs = ipex_model.prepare_inputs_for_generation(**tokens)
        outputs = ipex_model(**inputs)

        self.assertIsInstance(outputs.logits, torch.Tensor)

        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, torch_dtype=dtype, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**inputs)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=1e-3))
        # To avoid float pointing error
        self.assertTrue(torch.allclose(outputs.logits, loaded_model_outputs.logits, atol=1e-7))
        self.assertTrue(torch.allclose(outputs.logits, init_model_outputs.logits, atol=1e-7))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.skip(reason="Paged attention do not support assisted decoding for now")
    def test_assisted_decoding(self, model_arch):
        # assist decoding does not support static cache now
        if model_arch in self.IPEX_PATCHED_SUPPORTED_ARCHITECTURES:
            return
        model_id = MODEL_NAMES[model_arch]
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        tokens = tokenizer("This is a sample input", return_tensors="pt").to(DEVICE)
        ipex_output = ipex_model.generate(**tokens, do_sample=False, max_new_tokens=4)
        ipex_output_assisted = ipex_model.generate(
            **tokens, do_sample=False, assistant_model=transformers_model, max_new_tokens=4
        )
        ipex_output_assisted_2 = ipex_model.generate(
            **tokens, do_sample=False, assistant_model=ipex_model, max_new_tokens=4
        )
        transformers_output = transformers_model.generate(**tokens, do_sample=False, max_new_tokens=4)
        transformers_output_assisted = transformers_model.generate(
            **tokens, do_sample=False, assistant_model=ipex_model, max_new_tokens=4
        )
        self.assertTrue(torch.equal(ipex_output, ipex_output_assisted))
        self.assertTrue(torch.equal(ipex_output, ipex_output_assisted_2))
        self.assertTrue(torch.equal(transformers_output, transformers_output_assisted))

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True, False],
            }
        )
    )
    def test_ipex_beam_search(self, test_name, model_arch, use_cache):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        model = IPEXModelForCausalLM.from_pretrained(
            model_id, use_cache=use_cache, torch_dtype=dtype, device_map=DEVICE
        )
        # It will be removed when torch 2.6 released
        if (
            model_arch == "opt"
            and not use_cache
            and getattr(model.config, "compile", False)
            and is_torch_version("<", "2.6.0")
        ):
            return
        if use_cache and model_arch in self.IPEX_PATCHED_SUPPORTED_ARCHITECTURES:
            self.assertTrue(model.add_patch)
        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        self.assertEqual(model.use_cache, use_cache)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        # Test with batch_size is 1 and 2.
        texts = ["This is a sample", ["This is the first input", "This is the second input"]]
        generation_configs = (
            GenerationConfig(max_new_tokens=4, num_beams=2, do_sample=False),
            GenerationConfig(max_new_tokens=4, num_beams=4, do_sample=False),
            GenerationConfig(max_new_tokens=4, num_beams=8, do_sample=False),
            GenerationConfig(max_new_tokens=4, num_beams=32, do_sample=False),
            GenerationConfig(
                max_new_tokens=4, do_sample=False, top_p=0.9, top_k=0, pad_token_id=tokenizer.eos_token_id
            ),
        )
        for text in texts:
            tokens = tokenizer(text, padding=True, return_tensors="pt").to(DEVICE)
            for generation_config in generation_configs:
                outputs = model.generate(**tokens, generation_config=generation_config)
                transformers_outputs = transformers_model.generate(**tokens, generation_config=generation_config)
                self.assertIsInstance(outputs, torch.Tensor)
                self.assertTrue(torch.equal(outputs, transformers_outputs))

    def test_compare_with_and_without_past_key_values(self):
        model_id = "Intel/tiny_random_llama2_ipex_model"
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        model_with_pkv = IPEXModelForCausalLM.from_pretrained(
            model_id, use_cache=True, torch_dtype=dtype, device_map=DEVICE
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt").to(DEVICE)
        # Warmup
        model_with_pkv.generate(**tokens)
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )
        model_without_pkv = IPEXModelForCausalLM.from_pretrained(
            model_id, use_cache=False, torch_dtype=dtype, device_map=DEVICE
        )
        # Warmup
        model_without_pkv.generate(**tokens)
        with Timer() as without_pkv_timer:
            outputs_model_without_pkv = model_without_pkv.generate(
                **tokens, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
            )
        self.assertTrue(torch.equal(outputs_model_with_pkv, outputs_model_without_pkv))
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + tokens.input_ids.shape[1])
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + tokens.input_ids.shape[1])

    @parameterized.expand(IPEX_PATCHED_SUPPORTED_ARCHITECTURES)
    def test_patched_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        patched_model_id = MODEL_NAMES["patched_" + model_arch]
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, export=True, torch_dtype=dtype, device_map=DEVICE)
        exported_model = IPEXModelForCausalLM.from_pretrained(patched_model_id, torch_dtype=dtype, device_map=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample", return_tensors="pt").to(DEVICE)
        ipex_outputs = ipex_model.generate(
            **tokens, max_new_tokens=1, return_dict_in_generate=True, output_logits=True
        )
        exported_outputs = exported_model.generate(
            **tokens, max_new_tokens=1, return_dict_in_generate=True, output_logits=True
        )
        self.assertTrue(torch.allclose(ipex_outputs.logits[0], exported_outputs.logits[0], atol=1e-4))

    @unittest.skipIf(not is_bitsandbytes_available(), reason="Test requires bitsandbytes")
    def test_bnb(self):
        model_id = "PrunaAI/JackFram-llama-68m-bnb-4bit-smashed"
        set_seed(SEED)
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        # Test model forward do not need cache.
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample",
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(DEVICE)
        inputs = ipex_model.prepare_inputs_for_generation(**tokens)
        outputs = ipex_model(**inputs)

        self.assertIsInstance(outputs.logits, torch.Tensor)

        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, torch_dtype=dtype, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**inputs)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=5e-2))
        # To avoid float pointing error
        self.assertTrue(torch.allclose(outputs.logits, loaded_model_outputs.logits, atol=1e-7))
        self.assertTrue(torch.allclose(outputs.logits, init_model_outputs.logits, atol=1e-7))

    @unittest.skipIf(not is_auto_awq_available(), reason="Test requires autoawq")
    def test_awq(self):
        model_id = "PrunaAI/JackFram-llama-68m-AWQ-4bit-smashed"
        set_seed(SEED)
        dtype = torch.float16 if IS_XPU_AVAILABLE else torch.float32
        # Test model forward do not need cache.
        ipex_model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        self.assertIsInstance(ipex_model.config, PretrainedConfig)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer(
            "This is a sample",
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(DEVICE)
        inputs = ipex_model.prepare_inputs_for_generation(**tokens)
        outputs = ipex_model(**inputs)

        self.assertIsInstance(outputs.logits, torch.Tensor)

        transformers_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=DEVICE)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Test re-load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            ipex_model.save_pretrained(tmpdirname)
            loaded_model = self.IPEX_MODEL_CLASS.from_pretrained(tmpdirname, torch_dtype=dtype, device_map=DEVICE)
            loaded_model_outputs = loaded_model(**inputs)

        # Test init method
        init_model = self.IPEX_MODEL_CLASS(transformers_model)
        init_model_outputs = init_model(**inputs)

        # Compare tensor outputs
        self.assertTrue(torch.allclose(outputs.logits, transformers_outputs.logits, atol=5e-2))
        # To avoid float pointing error
        self.assertTrue(torch.allclose(outputs.logits, loaded_model_outputs.logits, atol=1e-7))
        self.assertTrue(torch.allclose(outputs.logits, init_model_outputs.logits, atol=1e-7))
