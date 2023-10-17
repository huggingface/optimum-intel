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


import os
import tempfile
import time
import unittest

import torch
from parameterized import parameterized
from transformers import AutoTokenizer, pipeline, set_seed

from optimum.exporters import TasksManager
from optimum.intel import (  # noqa
    INCConfig,
    INCModel,
    INCModelForCausalLM,
    INCModelForMaskedLM,
    INCModelForMultipleChoice,
    INCModelForQuestionAnswering,
    INCModelForSeq2SeqLM,
    INCModelForSequenceClassification,
    INCModelForTokenClassification,
    INCQuantizer,
    INCSeq2SeqTrainer,
    INCStableDiffusionPipeline,
    INCTrainer,
)
from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS, WEIGHTS_NAME


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(1009)


QUANTIZED_MODEL_NAMES_TO_TASK = (
    ("echarlaix/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic", "text-classification"),
    ("Intel/distilbert-base-uncased-distilled-squad-int8-static", "question-answering"),
    ("Intel/t5-small-xsum-int8-dynamic", "text2text-generation"),
)


MODEL_NAMES_TO_TASK = (
    ("hf-internal-testing/tiny-random-gpt2", "text-generation"),
    ("hf-internal-testing/tiny-random-BertForMaskedLM", "fill-mask"),
    ("hf-internal-testing/tiny-random-DistilBertForSequenceClassification", "text-classification"),
    ("hf-internal-testing/tiny-random-DebertaV2Model", "feature-extraction"),
    ("hf-internal-testing/tiny-random-MobileBertForQuestionAnswering", "question-answering"),
    ("hf-internal-testing/tiny-random-BartForConditionalGeneration", "text2text-generation"),
    ("hf-internal-testing/tiny-random-RobertaForTokenClassification", "token-classification"),
    ("hf-internal-testing/tiny-random-BertForMultipleChoice", "multiple-choice"),
)

DIFFUSERS_MODEL_NAMES_TO_TASK = (("echarlaix/stable-diffusion-v1-5-inc-int8-dynamic", "stable-diffusion"),)


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3


class INCModelingTest(unittest.TestCase):
    GENERATION_LENGTH = 100
    SPEEDUP_CACHE = 1.1

    @parameterized.expand(MODEL_NAMES_TO_TASK + QUANTIZED_MODEL_NAMES_TO_TASK)
    def test_compare_to_transformers(self, model_id, task):
        model_class = eval(_HEAD_TO_AUTOMODELS[task])
        inc_model = model_class.from_pretrained(model_id)
        model_type = inc_model.config.model_type.replace("_", "-")
        config_class = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model=inc_model,
            task=task,
            model_name=model_id,
            model_type=model_type,
        )
        config = config_class(inc_model.config)
        model_inputs = config.generate_dummy_inputs(framework="pt")
        outputs = inc_model(**model_inputs)
        with tempfile.TemporaryDirectory() as tmpdirname:
            inc_model.save_pretrained(tmpdirname)
            loaded_model = model_class.from_pretrained(tmpdirname, file_name=WEIGHTS_NAME)
            outputs_loaded = loaded_model(**model_inputs)

        if task == "feature-extraction":
            output_name = "last_hidden_state"
        elif task == "question-answering":
            output_name = "end_logits"
        else:
            output_name = "logits"

        # Compare to saved and loaded model
        self.assertTrue(torch.equal(outputs_loaded[output_name], outputs[output_name]))

        if inc_model._q_config is None:
            transformers_model = model_class.auto_model_class.from_pretrained(model_id)
            transformers_outputs = transformers_model(**model_inputs)
            # Compare to original transformers model
            self.assertTrue(torch.equal(transformers_outputs[output_name], outputs[output_name]))

    @parameterized.expand(MODEL_NAMES_TO_TASK + QUANTIZED_MODEL_NAMES_TO_TASK)
    def test_pipeline(self, model_id, task):
        if task == "multiple-choice":
            self.skipTest("No pipeline for multiple choice")

        model = eval(_HEAD_TO_AUTOMODELS[task]).from_pretrained(model_id)
        model.to("cpu")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline(task, model=model, tokenizer=tokenizer)
        self.assertEqual(pipe.device, model.device)

        inputs = ["This is a simple input" + (f"{tokenizer.mask_token}" if task == "fill-mask" else "")]
        if task == "question-answering":
            inputs *= 2

        pipe(*inputs)

    def test_compare_with_and_without_past_key_values(self):
        model_id = "echarlaix/tiny-random-gpt2-torchscript"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")

        model_with_pkv = INCModelForCausalLM.from_pretrained(model_id, use_cache=True, subfolder="model_with_pkv")
        # Warmup
        model_with_pkv.generate(**tokens)
        with Timer() as with_pkv_timer:
            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_length=self.GENERATION_LENGTH, max_length=self.GENERATION_LENGTH, num_beams=1
            )
        model_without_pkv = INCModelForCausalLM.from_pretrained(
            model_id, use_cache=False, subfolder="model_without_pkv"
        )
        # Warmup
        model_without_pkv.generate(**tokens)
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
