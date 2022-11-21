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
import unittest

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from neural_compressor import PostTrainingConfig
from optimum.intel.neural_compressor import INCQuantizer
from optimum.intel.neural_compressor.configuration import (
    IncDistillationConfig,
    IncPruningConfig,
    IncQuantizationConfig,
)
from optimum.intel.neural_compressor.quantization import (
    IncQuantizationMode,
    IncQuantizedModelForSequenceClassification,
    INCQuantizer,
)
from functools import partial


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(1009)


class INCQuantizationTest(unittest.TestCase):
    def test_dynamic_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        approach = "post_training_dynamic_quant"
        quantization_config = PostTrainingConfig(approach="post_training_dynamic_quant")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer = INCQuantizer.from_pretrained(model)
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=tmp_dir,
                save_onnx_model=True,
            )
            # TODO : Add quantization + loading verification
            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)


    def test_static_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        approach = "post_training_dynamic_quant"
        quantization_config = PostTrainingConfig(approach="post_training_static_quant")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        quantizer = INCQuantizer.from_pretrained(model)
        calibration_dataset = quantizer.get_calibration_dataset(
            "glue",
            dataset_config_name="sst2",
            preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
            num_samples=300,
            dataset_split="train",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer = INCQuantizer.from_pretrained(model)
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
                save_onnx_model=True,
            )
            # TODO : Add quantization + loading verification
            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
