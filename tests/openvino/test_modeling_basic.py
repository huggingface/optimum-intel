"""
The goal of the test in this file is to test that basic functionality of optimum[openvino] works:
- Load the model with automatic conversion from transformers (export)
- Do inference with appropriate pipeline
- Save the model to disk

This test is meant to run quickly with tiny test models. More extensive tests are in
test_modeling.py.
"""

# ruff: noqa

import gc
import unittest

from parameterized import parameterized
from transformers import AutoTokenizer, pipeline
from utils_tests import OPENVINO_DEVICE
from optimum.intel import (
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
from optimum.intel.utils.import_utils import is_transformers_version

# Make sure that common architectures are used in combination with common tasks
MODEL_NAMES = {
    "hf-internal-testing/tiny-random-bert": "OVModelForMaskedLM",
    "hf-internal-testing/tiny-random-distilbert": "OVModelForSequenceClassification",
    "hf-internal-testing/tiny-random-mbart": "OVModelForSeq2SeqLM",
    "hf-internal-testing/tiny-random-roberta": "OVModelForQuestionAnswering",
    "hf-internal-testing/tiny-random-gpt2": "OVModelForCausalLM",
    "hf-internal-testing/tiny-random-t5": "OVModelForSeq2SeqLM",
    "hf-internal-testing/tiny-random-bart": "OVModelForSeq2SeqLM",
}

TASKS = {
    "OVModelForMaskedLM": "fill-mask",
    "OVModelForSequenceClassification": "text-classification",
    "OVModelForQuestionAnswering": "question-answering",
    "OVModelForCausalLM": "text-generation",
    "OVModelForSeq2SeqLM": "text2text-generation",
}


class OVModelBasicIntegrationTest(unittest.TestCase):
    @parameterized.expand(MODEL_NAMES.keys())
    def test_pipeline(self, model_id):
        """
        Test that loading, inference and saving works for all models in MODEL_NAMES
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_class_str = MODEL_NAMES[model_id]
        if model_class_str == "OVModelForSeq2SeqLM" and is_transformers_version(">=", "5"):
            self.skipTest("text2text-generation pipeline was deprecated in transformers v5")

        model_class = eval(model_class_str)
        model = model_class.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.save_pretrained(f"{model_id}_ov")
        model = model_class.from_pretrained(f"{model_id}_ov", device=OPENVINO_DEVICE)

        input_text = ["hello world"]
        if model_class_str == "OVModelForQuestionAnswering":
            input_text *= 2
        elif model_class_str == "OVModelForMaskedLM":
            input_text[0] = f"{input_text[0]} {tokenizer.mask_token}"

        if model_class_str in TASKS:
            task = TASKS[model_class_str]
            pipe = pipeline(task, model=model, tokenizer=tokenizer)
            pipe(*input_text)
        gc.collect()

    def test_openvino_methods(self):
        """
        Sanity check for .reshape() .to() and .half()
        """
        model_id = "hf-internal-testing/tiny-random-distilbert"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OVModelForSequenceClassification.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.reshape(1, 16)
        model.half()
        model.to("cpu")
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            max_length=16,
            padding="max_length",
            truncation=True,
        )
        pipe("hello world")
        gc.collect()
