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

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from optimum.intel.openvino.configuration import OVConfig
from optimum.intel.openvino.modeling import OVModelForSequenceClassification
from optimum.intel.openvino.quantization import OVQuantizer
from parameterized import parameterized


class OVQuantizerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        ("distilbert-base-uncased-finetuned-sst-2-english", 44),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_static_quantization(self, model_name, expected_fake_quantize):
        def preprocess_function(examples, tokenizer):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        quantization_config = OVConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            transformers_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            quantizer = OVQuantizer.from_pretrained(transformers_model)
            calibration_dataset = quantizer.get_calibration_dataset(
                "glue",
                dataset_config_name="sst2",
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=10,
                dataset_split="train",
            )
            quantizer.quantize(
                save_directory=tmp_dir,
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
            )
            ov_model = OVModelForSequenceClassification.from_pretrained(tmp_dir)

            num_fake_quantize = 0
            for elem in ov_model.model.get_ops():
                if "FakeQuantize" in elem.name:
                    num_fake_quantize += 1
            self.assertEqual(expected_fake_quantize, num_fake_quantize)

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = ov_model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(quantization_config.to_dict(), loaded_config.to_dict())
