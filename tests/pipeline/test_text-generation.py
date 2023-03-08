import argparse
import time
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from optimum.intel import inference_mode as ipex_inference_mode
from parameterized import parameterized


MODEL_NAMES = {
    "bloom": "bigscience/bloom-7b1",
    "gptj": "EleutherAI/gpt-j-6B",
}


class IPEXIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bloom",
        "gptj",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_inference(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        kwargs = dict(torch_dtype=torch.bfloat16, use_cache=True, low_cpu_mem_usage=True, return_dict=False)
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["use_cache"] = True
        kwargs["low_cpu_mem_usage"] = True
        kwargs["return_dict"] = False
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, return_dict=False)
        model.to("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "DeepSpeed is a machine learning framework for deep neural networks and deep reinforcement learning. It is written in C++ and is available for Linux, Mac OS X,"
        generate_kwargs = {"max_new_tokens": 32, "do_sample": False, "num_beams": 4, "num_beam_groups": 1}
        text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        with torch.inference_mode():
            output = text_generator(inputs, **generate_kwargs)
        with ipex_inference_mode(text_generator, dtype=torch.bfloat16, verbose=False, jit=True) as ipex_text_generator:
            output_ipex = ipex_text_generator(inputs, **generate_kwargs)
        self.assertTrue(output[0]["generated_text"] == output_ipex[0]["generated_text"])
