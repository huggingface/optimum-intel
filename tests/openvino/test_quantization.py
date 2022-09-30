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

import tempfile
import unittest
from functools import partial

from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from evaluate import evaluator
from optimum.intel.openvino.modeling import OVModelForQuestionAnswering
from optimum.intel.openvino.nncf_config import DEFAULT_QUANTIZATION_CONFIG
from optimum.intel.openvino.quantization import OVQuantizer
from parameterized import parameterized


class OVQuantizerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (("distilbert-base-cased-distilled-squad", 80),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_static_quantization(self, model_name, expected_fake_quantize):
        def preprocess_function(examples, tokenizer):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            quantizer = OVQuantizer.from_pretrained(transformers_model)
            calibration_dataset = quantizer.get_calibration_dataset(
                "glue",
                dataset_config_name="sst2",
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=40,
                dataset_split="train",
            )
            quantizer.quantize(
                save_directory=tmp_dir,
                quantization_config=DEFAULT_QUANTIZATION_CONFIG,
                calibration_dataset=calibration_dataset,
            )
            ov_model = OVModelForQuestionAnswering.from_pretrained(tmp_dir)

            num_fake_quantize = 0
            for elem in ov_model.model.get_ops():
                if "FakeQuantize" in elem.name:
                    num_fake_quantize += 1
            self.assertEqual(expected_fake_quantize, num_fake_quantize)

            data = load_dataset("squad", split="validation").select(range(50))
            eval = evaluator("question-answering")
            transformers_pipe = pipeline("question-answering", model=transformers_model, tokenizer=tokenizer)
            ov_pipe = pipeline("question-answering", model=ov_model, tokenizer=tokenizer)
            transformers_metric = eval.compute(model_or_pipeline=transformers_pipe, data=data, metric="squad")
            ov_metric = eval.compute(model_or_pipeline=ov_pipe, data=data, metric="squad")

            self.assertGreaterEqual(ov_metric["f1"], transformers_metric["f1"] * 0.80)
