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

import tempfile
import unittest
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer_utils import TrainOutput
from transformers.utils import WEIGHTS_NAME

import evaluate
from optimum.intel.openvino import OVTrainingArguments
from optimum.intel.openvino.configuration import DEFAULT_QUANTIZATION_CONFIG, OVConfig
from optimum.intel.openvino.modeling import OVModelForSequenceClassification
from optimum.intel.openvino.trainer import OVTrainer
from parameterized import parameterized_class


CUSTOMIZED_QUANTIZATION_CONFIG = {
    "algorithm": "quantization",
    "initializer": {
        "range": {
            "num_init_samples": 16,
            "type": "percentile",
            "params": {"min_percentile": 0.01, "max_percentile": 99.99},
        },
        "batchnorm_adaptation": {"num_bn_adaptation_samples": 4},
    },
    "scope_overrides": {"activations": {"{re}.*matmul_0": {"mode": "asymmetric"}}},
    "ignored_scopes": [],
}

MOVEMENT_SPARSITY_CONFIG_FOR_BERT = {
    "algorithm": "movement_sparsity",
    "params": {
        "warmup_start_epoch": 1,
        "warmup_end_epoch": 2,
        "importance_regularization_factor": 1.0,
        "enable_structured_masking": True,
    },
    "sparse_structure_by_scopes": [
        {"mode": "block", "sparse_factors": [32, 32], "target_scopes": "{re}.*BertAttention.*"},
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*BertIntermediate.*"},
        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*BertOutput.*"},
    ],
    "ignored_scopes": ["{re}.*NNCFEmbedding", "{re}.*LayerNorm.*", "{re}.*pooler.*", "{re}.*classifier.*"],
}


@dataclass
class OVTrainerTestDescriptor:
    model_id: str
    teacher_model_id: Optional[str] = None
    nncf_compression_config: Union[List[Dict], Dict, None] = None
    expected_fake_quantize: int = 0
    expected_int8: int = 0
    expected_binary_masks: int = 0
    compression_metrics: List[str] = field(default_factory=list)


OVTRAINER_TEST_DESCRIPTORS = [
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[],
        compression_metrics=["distillation_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
        expected_fake_quantize=19,
        expected_int8=14,
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
        expected_fake_quantize=19,
        expected_int8=14,
        compression_metrics=["distillation_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
        expected_fake_quantize=31,
        expected_int8=17,
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
        expected_fake_quantize=31,
        expected_int8=17,
        compression_metrics=["distillation_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=19,
        expected_int8=14,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=31,
        expected_int8=17,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=19,
        expected_int8=14,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
    OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=31,
        expected_int8=17,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
]


@parameterized_class("descriptor", zip(OVTRAINER_TEST_DESCRIPTORS))
class OVTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor: OVTrainerTestDescriptor
        desc = self.descriptor
        self.ov_config = OVConfig()
        self.ov_config.compression = desc.nncf_compression_config
        self.tokenizer = AutoTokenizer.from_pretrained(desc.model_id)
        self.task = "sequence-classification"
        self.model = AutoModelForSequenceClassification.from_pretrained(desc.model_id)
        self.teacher_model = None
        if desc.teacher_model_id:
            self.teacher_model = AutoModelForSequenceClassification.from_pretrained(desc.teacher_model_id)

        def tokenizer_fn(examples):
            return self.tokenizer(examples["sentence"], padding="max_length", max_length=128)

        self.dataset = load_dataset("glue", "sst2")
        self.train_dataset = self.dataset["train"].select(range(8)).map(tokenizer_fn, batched=True)
        self.eval_dataset = self.dataset["validation"].select(range(4)).map(tokenizer_fn, batched=True)
        self.metric = evaluate.load("glue", "sst2")

    def test_training(self):
        desc: OVTrainerTestDescriptor = self.descriptor
        num_train_epochs = 3
        train_batch_size = 4
        total_steps = ceil(len(self.train_dataset) / train_batch_size) * num_train_epochs
        with tempfile.TemporaryDirectory() as output_dir:
            args = OVTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                do_train=True,
                do_eval=True,
                logging_steps=1,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=1,
                no_cuda=True,
            )
            trainer = OVTrainer(
                model=self.model,
                teacher_model=self.teacher_model,
                args=args,
                ov_config=self.ov_config,
                task=self.task,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metric,
            )

            # check evaluation can work even before training.
            metrics = trainer.evaluate()
            self.assertIn("eval_loss", metrics)
            self.assertIn("eval_accuracy", metrics)

            # check trainining & saving
            train_outputs = trainer.train()
            self.assertIsInstance(train_outputs, TrainOutput)
            self.assertEqual(train_outputs.global_step, total_steps)
            trainer.save_model()

            for metric in desc.compression_metrics:
                self.assertIn(metric, trainer.compression_metrics)

            # check saved OVModel can output
            ovmodel = OVModelForSequenceClassification.from_pretrained(output_dir)
            inputs = next(iter(trainer.get_eval_dataloader()))
            outputs = ovmodel(**inputs)
            self.assertIn("logits", outputs)

            # check quantization ops
            num_fake_quantize, num_int8 = self.count_quantization_op_number(ovmodel)
            self.assertEqual(desc.expected_fake_quantize, num_fake_quantize)
            self.assertEqual(desc.expected_int8, num_int8)

            # check binary mask in sparsity/pruning algorithms
            state_dict = torch.load(Path(output_dir, WEIGHTS_NAME), map_location="cpu")
            num_binary_masks = sum(key.endswith("_binary_mask") for key in state_dict)
            self.assertEqual(desc.expected_binary_masks, num_binary_masks)

    def compute_metric(self, p):
        return self.metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def count_quantization_op_number(self, ovmodel):
        num_fake_quantize = 0
        num_int8 = 0
        for elem in ovmodel.model.get_ops():
            if "FakeQuantize" in elem.name:
                num_fake_quantize += 1
            if "8" in elem.get_element_type().get_type_name():
                num_int8 += 1
        return num_fake_quantize, num_int8
