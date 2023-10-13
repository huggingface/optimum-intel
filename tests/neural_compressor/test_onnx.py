#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

# ruff: noqa


import os
import tempfile

from neural_compressor.config import PostTrainingQuantConfig
from parameterized import parameterized
from transformers import AutoTokenizer, set_seed
from utils_tests import SEED, INCTestMixin, _generate_dataset

from optimum.intel import (
    INCConfig,
    INCModelForCausalLM,
    INCModelForSeq2SeqLM,
    INCModelForQuestionAnswering,
    INCModelForSequenceClassification,
    INCModelForMaskedLM,
    INCModelForTokenClassification,
    INCQuantizer,
)
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification
from optimum.pipelines import ORT_SUPPORTED_TASKS

os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(SEED)


class OptimizationTest(INCTestMixin):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        ("text-classification", "hf-internal-testing/tiny-random-bert", 64),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_static_quantization(self, task, model_name, expected_quantized_matmuls):
        num_samples = 10
        model = ORT_SUPPORTED_TASKS[task]["class"][0].auto_model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantizer = INCQuantizer.from_pretrained(model, task=task)
        calibration_dataset = _generate_dataset(quantizer, tokenizer, num_samples=num_samples)
        save_onnx_model = True
        op_type_dict = (
            {"Embedding": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}}
            if save_onnx_model
            else None
        )
        quantization_config = PostTrainingQuantConfig(approach="static", op_type_dict=op_type_dict)
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
                save_onnx_model=save_onnx_model,
            )
            self.check_model_outputs(
                q_model=quantizer._quantized_model,
                task=task,
                tokenizer=tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
                num_samples=num_samples,
                load_onnx_model=save_onnx_model,
            )
