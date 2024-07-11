#  Copyright 2024 The HuggingFace Team. All rights reserved.
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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    set_seed,
)
from utils_tests import MODEL_NAMES, SEED, INCTestMixin, _generate_dataset


from optimum.intel import (
    INCConfig,
    INCModelForCausalLM,
    INCModelForSeq2SeqLM,
    INCModelForQuestionAnswering,
    INCModelForSequenceClassification,
    INCModelForMaskedLM,
    INCModelForTokenClassification,
    INCQuantizer,
    INCSeq2SeqTrainer,
)
from optimum.exporters import TasksManager


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(SEED)


class IPEXQuantizationTest(INCTestMixin):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (("text-classification", "bert", 21),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_ipex_static_quantization_with_smoothquant(self, task, model_arch, expected_quantized_matmuls):
        recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}}
        num_samples = 10
        model_name = MODEL_NAMES[model_arch]
        quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex", recipes=recipes)
        model = TasksManager.get_model_class_for_task(task).from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantizer = INCQuantizer.from_pretrained(model, task=task)
        calibration_dataset = _generate_dataset(quantizer, tokenizer, num_samples=num_samples)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
            )
            self.check_model_outputs(
                q_model=quantizer._quantized_model,
                task=task,
                tokenizer=tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
                num_samples=num_samples,
                load_inc_model=False,
                load_ipex_model=True,
            )
