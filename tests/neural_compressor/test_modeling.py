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
import unittest

from parameterized import parameterized
from transformers import set_seed

from optimum.exporters import TasksManager
from optimum.intel import (  # noqa
    INCConfig,
    INCModelForCausalLM,
    INCModelForMaskedLM,
    INCModelForQuestionAnswering,
    INCModelForSeq2SeqLM,
    INCModelForSequenceClassification,
    INCModelForTokenClassification,
    INCQuantizer,
    INCSeq2SeqTrainer,
    INCStableDiffusionPipeline,
    INCTrainer,
)
from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(1009)


MODEL_NAMES_TO_TASK = (
    ("echarlaix/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic", "text-classification"),
    ("echarlaix/distilbert-sst2-inc-dynamic-quantization-magnitude-pruning-0.1", "text-classification"),
    ("hf-internal-testing/tiny-random-bert", "fill-mask"),
    ("Intel/distilbert-base-uncased-distilled-squad-int8-static", "question-answering"),
    ("hf-internal-testing/tiny-random-gpt2", "text-generation"),
    ("Intel/t5-small-xsum-int8-dynamic", "text2text-generation"),
    # ("echarlaix/stable-diffusion-v1-5-inc-int8-dynamic", "stable-diffusion")
)


class INCModelingTest(unittest.TestCase):
    @parameterized.expand(MODEL_NAMES_TO_TASK)
    def test_modeling(self, model_id, task):
        inc_model = eval(_HEAD_TO_AUTOMODELS[task]).from_pretrained(model_id)  # TRANSFORMERS_AUTO_CLASS
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
        inc_model(**model_inputs)
