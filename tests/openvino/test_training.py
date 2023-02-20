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
from copy import deepcopy
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
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsityController
from optimum.intel.openvino import OVTrainingArguments
from optimum.intel.openvino.configuration import DEFAULT_QUANTIZATION_CONFIG, OVConfig
from optimum.intel.openvino.modeling import OVModelForSequenceClassification
from optimum.intel.openvino.trainer import OVTrainer
from optimum.intel.openvino.utils import OV_XML_FILE_NAME
from parameterized import parameterized


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

STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT = {
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

UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT = deepcopy(STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT)
UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT["params"]["enable_structured_masking"] = False


def initialize_movement_sparsifier_parameters_by_sparsity(
    movement_controller: MovementSparsityController,
    sparsity: float = 0.95,
    seed: int = 42,
    negative_value: float = -10.0,
    positive_value: float = 10.0,
):
    for minfo in movement_controller.sparsified_module_info:
        operand = minfo.operand
        device = operand.weight_importance.device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        with torch.no_grad():
            weight_rand_idx = torch.randperm(operand.weight_importance.numel(), generator=generator, device=device)
            num_negatives = int(operand.weight_importance.numel() * sparsity)
            num_positives = operand.weight_importance.numel() - num_negatives
            data = [negative_value] * num_negatives + [positive_value] * num_positives
            weight_init_tensor = torch.FloatTensor(data, device=device)[weight_rand_idx].reshape_as(
                operand.weight_importance
            )
            operand.weight_importance.copy_(weight_init_tensor)
            if operand.prune_bias:
                bias_init_tensor = torch.ones_like(operand.bias_importance) * negative_value
                operand.bias_importance.copy_(bias_init_tensor)


@dataclass
class OVTrainerTestDescriptor:
    model_id: str
    teacher_model_id: Optional[str] = None
    nncf_compression_config: Union[List[Dict], Dict, None] = None
    expected_fake_quantize: int = 0
    expected_int8: int = 0
    expected_binary_masks: int = 0
    compression_metrics: List[str] = field(default_factory=list)


OVTRAINER_TRAINING_TEST_DESCRIPTORS = {
    "distillation": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[],
        compression_metrics=["distillation_loss"],
    ),
    "default_quantization": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
        expected_fake_quantize=19,
        expected_int8=14,
    ),
    "distillation,default_quantization": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
        expected_fake_quantize=19,
        expected_int8=14,
        compression_metrics=["distillation_loss"],
    ),
    "customized_quantization": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
        expected_fake_quantize=31,
        expected_int8=17,
    ),
    "distillation,customized_quantization": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
        expected_fake_quantize=31,
        expected_int8=17,
        compression_metrics=["distillation_loss"],
    ),
    "structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    "distillation,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
    "default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=19,
        expected_int8=14,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    "customized_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=31,
        expected_int8=17,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    "distillation,default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=19,
        expected_int8=14,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
    "distillation,customized_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=31,
        expected_int8=17,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
    "unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    "distillation,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
    "default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=19,
        expected_int8=14,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    "customized_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=31,
        expected_int8=17,
        expected_binary_masks=24,
        compression_metrics=["compression_loss"],
    ),
    "distillation,default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=19,
        expected_int8=14,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
    "distillation,customized_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-bert",
        teacher_model_id="hf-internal-testing/tiny-bert",
        nncf_compression_config=[CUSTOMIZED_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=31,
        expected_int8=17,
        expected_binary_masks=24,
        compression_metrics=["compression_loss", "distillation_loss"],
    ),
}


class OVTrainerTrainingTest(unittest.TestCase):
    @parameterized.expand(OVTRAINER_TRAINING_TEST_DESCRIPTORS.items())
    def test_training(self, _, desc: OVTrainerTestDescriptor):
        self.prepare(desc)
        num_train_epochs = 3
        train_batch_size = 4
        total_steps = ceil(len(self.train_dataset) / train_batch_size) * num_train_epochs
        with tempfile.TemporaryDirectory() as output_dir:
            args = OVTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                learning_rate=1e-7,
                do_train=True,
                do_eval=True,
                logging_steps=1,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=1,
                no_cuda=True,
                full_determinism=True,
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

            movement_controller = trainer._get_compression_controller_by_cls(
                MovementSparsityController
            )  # pylint: disable=protected-access
            if movement_controller is not None:
                # make sure the binary masks will have many zeros
                initialize_movement_sparsifier_parameters_by_sparsity(movement_controller, sparsity=0.95)

            # check evaluation can work even before training.
            metrics = trainer.evaluate()
            self.assertIn("eval_loss", metrics)
            self.assertIn("eval_accuracy", metrics)

            # check trainining & saving
            train_outputs = trainer.train()
            self.assertIsInstance(train_outputs, TrainOutput)
            self.assertEqual(train_outputs.global_step, total_steps)
            for metric in desc.compression_metrics:
                self.assertIn(metric, trainer.compression_metrics)

            # check model can be saved
            trainer.save_model()
            self.assertTrue(Path(output_dir, WEIGHTS_NAME).is_file())
            self.assertTrue(Path(output_dir, OV_XML_FILE_NAME).is_file())

            # check saved ovmodel can output
            ovmodel = OVModelForSequenceClassification.from_pretrained(output_dir)
            self.check_irmodel_is_dynamic(ovmodel.model)
            self.check_ovmodel_output_equals_torch_output(ovmodel, trainer.model)

            # check ovmodel quantization ops
            num_fake_quantize, num_int8 = self.count_quantization_op_number(ovmodel)
            self.assertEqual(desc.expected_fake_quantize, num_fake_quantize)
            self.assertEqual(desc.expected_int8, num_int8)

            # check binary mask in sparsity/pruning algorithms
            state_dict = torch.load(Path(output_dir, WEIGHTS_NAME), map_location="cpu")
            num_binary_masks = sum(key.endswith("_binary_mask") for key in state_dict)
            self.assertEqual(desc.expected_binary_masks, num_binary_masks)

    def prepare(self, desc: OVTrainerTestDescriptor):
        torch.manual_seed(42)
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
        self.train_dataset = self.dataset["train"].sort("sentence").select(range(8)).map(tokenizer_fn, batched=True)
        self.eval_dataset = (
            self.dataset["validation"].sort("sentence").select(range(4)).map(tokenizer_fn, batched=True)
        )
        self.metric = evaluate.load("glue", "sst2")
        self.compute_metric = lambda p: self.metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    def count_quantization_op_number(self, ovmodel):
        num_fake_quantize = 0
        num_int8 = 0
        for elem in ovmodel.model.get_ops():
            if "FakeQuantize" in elem.name:
                num_fake_quantize += 1
            if "8" in elem.get_element_type().get_type_name():
                num_int8 += 1
        return num_fake_quantize, num_int8

    def check_ovmodel_output_equals_torch_output(self, ovmodel, torch_model):
        torch_model = torch_model.eval()
        for max_seq_length in [16, 128]:
            for batch_size in [1, 3]:
                examples = self.dataset["train"].sort("sentence")[:batch_size]
                inputs = self.tokenizer(
                    examples["sentence"],
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                ovmodel_outputs = ovmodel(**inputs)
                self.assertIn("logits", ovmodel_outputs)
                ovmodel_logits = ovmodel_outputs.logits
                torch_logits = torch_model(**inputs).logits
                self.assertTrue(
                    torch.allclose(
                        torch.softmax(ovmodel_logits, dim=-1),
                        torch.softmax(torch_logits, dim=-1),
                        rtol=0.2,
                    )
                )

    def check_irmodel_is_dynamic(self, irmodel):
        self.assertTrue(irmodel.is_dynamic())
