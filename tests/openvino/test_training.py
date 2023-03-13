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

import re
import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_utils import TrainOutput
from transformers.utils import WEIGHTS_NAME

import cpuinfo
import evaluate
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsityController
from openvino.runtime import PartialShape
from optimum.intel.openvino import OVTrainingArguments
from optimum.intel.openvino.configuration import DEFAULT_QUANTIZATION_CONFIG, OVConfig
from optimum.intel.openvino.modeling import (
    OVModel,
    OVModelForAudioClassification,
    OVModelForImageClassification,
    OVModelForSequenceClassification,
)
from optimum.intel.openvino.trainer import OVTrainer
from optimum.intel.openvino.utils import OV_XML_FILE_NAME
from parameterized import parameterized


CUSTOMIZED_QUANTIZATION_CONFIG = {
    "algorithm": "quantization",
    "overflow_fix": "disable",
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


def is_avx_vnni_supported() -> bool:
    return any(re.search("avx.*vnni", flag.lower()) is not None for flag in cpuinfo.get_cpu_info()["flags"])


@dataclass
class OVTrainerTestDescriptor:
    model_id: str
    teacher_model_id: Optional[str] = None
    nncf_compression_config: Union[List[Dict], Dict, None] = None
    expected_fake_quantize: int = 0
    expected_int8: int = 0
    expected_binary_masks: int = 0
    compression_metrics: List[str] = field(default_factory=list)


OVTRAINER_TEXT_CLASSIFICATION_TEST_DESCRIPTORS = {
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


class OVTrainerTextClassificationTrainingTest(unittest.TestCase):
    @parameterized.expand(OVTRAINER_TEXT_CLASSIFICATION_TEST_DESCRIPTORS.items())
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

            # check saved ovmodel IR and output
            ovmodel = OVModelForSequenceClassification.from_pretrained(output_dir)
            self.check_irmodel_is_dynamic(ovmodel.model)
            self.check_irmodel_reshaping(ovmodel.model)
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
        nncf_compression_config = desc.nncf_compression_config
        if not is_avx_vnni_supported():
            # should enable "overflow_fix" in quantization otherwise accuracy degradation may be seen
            nncf_compression_config = self.get_nncf_config_with_overflow_fix_override(
                nncf_compression_config, "enable"
            )

        self.ov_config.compression = nncf_compression_config
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

    def get_nncf_config_with_overflow_fix_override(
        self, nncf_compression_config: Union[List[Dict], Dict, None], value: str = "enable"
    ):
        overrided_config = deepcopy(nncf_compression_config)
        quantization_config = None
        if isinstance(overrided_config, list):
            for config in overrided_config:
                if config["algorithm"] == "quantization":
                    quantization_config = config
                    break
        elif isinstance(overrided_config, dict):
            if overrided_config["algorithm"] == "quantization":
                quantization_config = overrided_config
        if quantization_config is not None:
            quantization_config["overflow_fix"] = value
        return overrided_config

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
                        rtol=0.0001,
                    )
                )

    def check_irmodel_is_dynamic(self, irmodel):
        self.assertTrue(irmodel.is_dynamic())

    def check_irmodel_reshaping(self, irmodel):
        def _reshape_ir_by_input_shape(ov_model, batch_size, seqlen):
            new_input_cfg = dict()
            for input_ in ov_model.inputs:
                new_input_cfg[input_.any_name] = PartialShape([batch_size, seqlen])
            ov_model.reshape(new_input_cfg)
            return ov_model

        def _assertInputsEqual(ov_model, shape):
            for input_ in ov_model.inputs:
                self.assertSequenceEqual(list(input_.get_shape()), shape)

        bs, sl = 4, 256
        irmodel = _reshape_ir_by_input_shape(irmodel, batch_size=bs, seqlen=sl)
        _assertInputsEqual(irmodel, shape=[bs, sl])

        irmodel = _reshape_ir_by_input_shape(irmodel, batch_size=-1, seqlen=-1)
        self.assertTrue(irmodel.is_dynamic())

        bs, sl = 1, 89
        irmodel = _reshape_ir_by_input_shape(irmodel, batch_size=bs, seqlen=sl)
        _assertInputsEqual(irmodel, shape=[bs, sl])

        irmodel = _reshape_ir_by_input_shape(irmodel, batch_size=-1, seqlen=-1)
        self.assertTrue(irmodel.is_dynamic())


STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN = {
    "algorithm": "movement_sparsity",
    "params": {
        "warmup_start_epoch": 1,
        "warmup_end_epoch": 2,
        "importance_regularization_factor": 1.0,
        "enable_structured_masking": True,
    },
    "sparse_structure_by_scopes": [
        {"mode": "block", "sparse_factors": [8, 8], "target_scopes": "{re}.*SwinAttention.*"},
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*SwinIntermediate.*"},
        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*SwinOutput.*"},
    ],
    "ignored_scopes": ["{re}.*PatchEmbed.*", "{re}.*PatchMerging.*", "{re}.*classifier.*", "{re}.*LayerNorm.*"],
}
UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN = deepcopy(STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN)
UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN["params"]["enable_structured_masking"] = False

OVTRAINER_IMAGE_CLASSIFICATION_TEST_DESCRIPTORS = {
    "default_quantization": OVTrainerTestDescriptor(
        model_id="yujiepan/tiny-random-SwinModel",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG],
        expected_fake_quantize=27,
        expected_int8=27,
        compression_metrics=["compression_loss"],
    ),
    "structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="yujiepan/tiny-random-SwinModel",
        nncf_compression_config=[STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN],
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="yujiepan/tiny-random-SwinModel",
        nncf_compression_config=[UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN],
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="yujiepan/tiny-random-SwinModel",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN],
        expected_fake_quantize=27,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="yujiepan/tiny-random-SwinModel",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN],
        expected_fake_quantize=27,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "distillation,default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="yujiepan/tiny-random-SwinModel",
        teacher_model_id="yujiepan/tiny-random-SwinModel",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN],
        expected_fake_quantize=27,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "distillation,default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="yujiepan/tiny-random-SwinModel",
        teacher_model_id="yujiepan/tiny-random-SwinModel",
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN],
        expected_fake_quantize=27,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
}


class OVTrainerImageClassificationTrainingTest(unittest.TestCase):
    @parameterized.expand(OVTRAINER_IMAGE_CLASSIFICATION_TEST_DESCRIPTORS.items())
    def test_training(self, _, desc: OVTrainerTestDescriptor):
        self.prepare(desc)
        num_train_epochs = 3
        train_batch_size = 4
        total_steps = ceil(len(self.train_dataset) / train_batch_size) * num_train_epochs
        with tempfile.TemporaryDirectory() as output_dir:
            self.args = OVTrainingArguments(
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
                remove_unused_columns=False,
            )
            self.trainer = OVTrainer(
                model=self.model,
                teacher_model=self.teacher_model,
                args=self.args,
                ov_config=self.ov_config,
                task=self.task,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.image_processor,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metric,
            )

            trainer = self.trainer
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

            # check trainining
            train_outputs = trainer.train()
            self.assertIsInstance(train_outputs, TrainOutput)
            self.assertEqual(train_outputs.global_step, total_steps)
            self.assertEqual(sorted(desc.compression_metrics), sorted(trainer.compression_metrics.keys()))

            # check model can be saved
            trainer.save_model()
            self.assertTrue(Path(output_dir, WEIGHTS_NAME).is_file())
            self.assertTrue(Path(output_dir, OV_XML_FILE_NAME).is_file())

            # check saved ovmodel IR and output
            ovmodel = OVModelForImageClassification.from_pretrained(output_dir)
            self.check_if_ovmodel_is_dynamic(ovmodel, True)
            self.check_ovmodel_reshaping(ovmodel)
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
        self.task = "image-classification"
        self.image_processor = AutoImageProcessor.from_pretrained(desc.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(desc.model_id, num_labels=3)
        self.teacher_model = None
        if desc.teacher_model_id:
            self.teacher_model = AutoModelForImageClassification.from_pretrained(desc.teacher_model_id, num_labels=3)

        def data_transform(examples, size=None):
            batch = self.image_processor(examples["image"], size=size, return_tensors="pt")
            batch["labels"] = examples["labels"]
            return batch

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        self.dataset = load_dataset("beans", task="image-classification")
        self.dataset.set_transform(data_transform)
        self.train_dataset = self.dataset["train"].select(range(8))
        self.eval_dataset = self.dataset["validation"].select(range(4))
        self.data_transform = data_transform
        self.collate_fn = collate_fn
        self.metric = evaluate.load("accuracy")
        self.compute_metric = lambda p: self.metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    def count_quantization_op_number(self, ovmodel):
        num_fake_quantize = 0
        num_int8 = 0
        for elem in ovmodel.model.get_ops():
            if "FakeQuantize" in elem.name:
                num_fake_quantize += 1
            for i in range(elem.get_output_size()):
                if "8" in elem.get_output_element_type(i).get_type_name():
                    num_int8 += 1
        return num_fake_quantize, num_int8

    def check_ovmodel_output_equals_torch_output(self, ovmodel, torch_model):
        torch_model = torch_model.eval()
        for batch_size in [1, 4]:
            for size in [128, 224, 256]:
                self.trainer.args.per_device_eval_batch_size = batch_size
                dataset = self.eval_dataset.set_transform(partial(self.data_transform, size=size))
                for inputs in self.trainer.get_eval_dataloader(dataset):
                    ovmodel_outputs = ovmodel(**inputs)
                    self.assertIn("logits", ovmodel_outputs)
                    ovmodel_logits = ovmodel_outputs.logits
                    torch_logits = torch_model(**inputs).logits
                    self.assertTrue(
                        torch.allclose(
                            torch.softmax(ovmodel_logits, dim=-1),
                            torch.softmax(torch_logits, dim=-1),
                            rtol=0.0001,
                        )
                    )

    def check_if_ovmodel_is_dynamic(self, ovmodel: OVModel, expected_result: bool = True):
        if expected_result is True:
            self.assertTrue(ovmodel.model.is_dynamic())
        else:
            self.assertFalse(ovmodel.model.is_dynamic())

    def check_ovmodel_reshaping(self, ovmodel: OVModel):
        for batch_size in [1, 4]:
            for size in [128, 224, 256]:
                shape = [batch_size, 3, size, size]
                ovmodel.reshape(*shape)
                self.check_if_ovmodel_is_dynamic(ovmodel, False)
                for input_ in ovmodel.model.inputs:
                    self.assertSequenceEqual(list(input_.get_shape()), shape)
                ovmodel.reshape(-1, -1, -1, -1)
                self.check_if_ovmodel_is_dynamic(ovmodel, True)


QUANTIZATION_CONFIG_FOR_WAV2VEC2 = {
    "algorithm": "quantization",
    "quantize_inputs": False,
    "preset": "mixed",
    "overflow_fix": "enable",
    "initializer": {
        "range": {"num_init_samples": 10, "type": "mean_min_max"},
        "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
    },
    "scope_overrides": {"activations": {"{re}.*matmul_0": {"mode": "symmetric"}}},
    "ignored_scopes": ["{re}.*feature_extractor.*", "{re}.*__add___[0-1]", "{re}.*layer_norm_0"],
}

STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2 = {
    "algorithm": "movement_sparsity",
    "params": {
        "warmup_start_epoch": 1,
        "warmup_end_epoch": 2,
        "importance_regularization_factor": 0.1,
        "enable_structured_masking": True,
    },
    "sparse_structure_by_scopes": [
        {"mode": "block", "sparse_factors": [8, 8], "target_scopes": "{re}.*Wav2Vec2Attention.*"},
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*intermediate_dense.*"},
        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*output_dense.*"},
    ],
    "ignored_scopes": [
        "{re}projector",
        "{re}classifier",
        "{re}feature_extractor",
        "{re}feature_projection",
        "{re}pos_conv_embed",
    ],
}

UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2 = deepcopy(STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2)
UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2["params"]["enable_structured_masking"] = False


OVTRAINER_AUDIO_CLASSIFICATION_TEST_DESCRIPTORS = {
    "quantization": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=45,
        expected_int8=28,
        compression_metrics=["compression_loss"],
    ),
    "structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        nncf_compression_config=[STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        nncf_compression_config=[UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=45,
        expected_int8=28,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=45,
        expected_int8=28,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "distillation,quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        teacher_model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=45,
        expected_int8=28,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "distillation,quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        teacher_model_id="hf-internal-testing/tiny-random-Wav2Vec2Model",
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=45,
        expected_int8=28,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
}


class OVTrainerAudioClassificationTrainingTest(unittest.TestCase):
    @parameterized.expand(OVTRAINER_AUDIO_CLASSIFICATION_TEST_DESCRIPTORS.items())
    def test_training(self, _, desc: OVTrainerTestDescriptor):
        self.prepare(desc)
        num_train_epochs = 3
        train_batch_size = 4
        total_steps = ceil(len(self.train_dataset) / train_batch_size) * num_train_epochs
        with tempfile.TemporaryDirectory() as output_dir:
            self.args = OVTrainingArguments(
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
                remove_unused_columns=False,
            )
            self.trainer = OVTrainer(
                model=self.model,
                teacher_model=self.teacher_model,
                args=self.args,
                ov_config=self.ov_config,
                task=self.task,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.feature_extractor,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metric,
            )

            trainer = self.trainer
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

            # check trainining
            train_outputs = trainer.train()
            self.assertIsInstance(train_outputs, TrainOutput)
            self.assertEqual(train_outputs.global_step, total_steps)
            self.assertEqual(sorted(desc.compression_metrics), sorted(trainer.compression_metrics.keys()))

            # check model can be saved
            trainer.save_model()
            self.assertTrue(Path(output_dir, WEIGHTS_NAME).is_file())
            self.assertTrue(Path(output_dir, OV_XML_FILE_NAME).is_file())

            # check saved ovmodel IR and output
            ovmodel = OVModelForAudioClassification.from_pretrained(output_dir)
            self.check_if_ovmodel_is_dynamic(ovmodel, True)
            self.check_ovmodel_reshaping(ovmodel)
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
        self.task = "audio-classification"
        self.dataset = load_dataset("superb", "ks")
        self.num_labels = len(self.dataset["train"].features["label"].names)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(desc.model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(desc.model_id, num_labels=self.num_labels)
        self.teacher_model = None
        if desc.teacher_model_id:
            self.teacher_model = AutoModelForAudioClassification.from_pretrained(
                desc.teacher_model_id, num_labels=self.num_labels
            )

        def random_subsample(wav: np.ndarray, max_length: float = 1, sample_rate: int = 16000):
            """Randomly sample chunks of `max_length` seconds from the input audio"""
            sample_length = int(round(sample_rate * max_length))
            if len(wav) <= sample_length:
                return wav
            random_offset = random.randint(0, len(wav) - sample_length - 1)
            return wav[random_offset : random_offset + sample_length]

        def data_transform(examples, max_length=1):
            sampling_rate = self.feature_extractor.sampling_rate
            audio = random_subsample(examples["audio"][0]["array"], max_length=max_length, sample_rate=sampling_rate)
            batch = self.feature_extractor(audio, return_tensors="pt", sampling_rate=sampling_rate)
            batch["labels"] = examples["label"]
            return batch

        def collate_fn(examples):
            pixel_values = torch.stack([example["input_values"] for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"input_values": pixel_values, "labels": labels}

        self.dataset.set_transform(data_transform)
        self.train_dataset = self.dataset["train"].select(range(8))
        self.eval_dataset = self.dataset["validation"].select(range(4))
        self.data_transform = data_transform
        self.collate_fn = collate_fn
        self.metric = evaluate.load("accuracy")
        self.compute_metric = lambda p: self.metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    def count_quantization_op_number(self, ovmodel):
        num_fake_quantize = 0
        num_int8 = 0
        for elem in ovmodel.model.get_ops():
            if "FakeQuantize" in elem.name:
                num_fake_quantize += 1
            for i in range(elem.get_output_size()):
                if "8" in elem.get_output_element_type(i).get_type_name():
                    num_int8 += 1
        return num_fake_quantize, num_int8

    def check_ovmodel_output_equals_torch_output(self, ovmodel, torch_model):
        torch_model = torch_model.eval()
        for batch_size in [1, 4]:
            for max_length in [1, 0.2]:
                self.trainer.args.per_device_eval_batch_size = batch_size
                dataset = self.eval_dataset.set_transform(partial(self.data_transform, max_length=max_length))
                for inputs in self.trainer.get_eval_dataloader(dataset):
                    ovmodel_outputs = ovmodel(**inputs)
                    self.assertIn("logits", ovmodel_outputs)
                    ovmodel_logits = ovmodel_outputs.logits
                    torch_logits = torch_model(**inputs).logits
                    self.assertTrue(
                        torch.allclose(
                            torch.softmax(ovmodel_logits, dim=-1),
                            torch.softmax(torch_logits, dim=-1),
                            rtol=0.0001,
                        )
                    )

    def check_if_ovmodel_is_dynamic(self, ovmodel: OVModel, expected_result: bool = True):
        if expected_result is True:
            self.assertTrue(ovmodel.model.is_dynamic())
        else:
            self.assertFalse(ovmodel.model.is_dynamic())

    def check_ovmodel_reshaping(self, ovmodel: OVModel):
        for batch_size in [1, 4]:
            for seq_len in [1234, 16000]:
                shape = [batch_size, seq_len]
                ovmodel.reshape(*shape)
                self.check_if_ovmodel_is_dynamic(ovmodel, False)
                for input_ in ovmodel.model.inputs:
                    self.assertSequenceEqual(list(input_.get_shape()), shape)
                ovmodel.reshape(-1, -1)
                self.check_if_ovmodel_is_dynamic(ovmodel, True)
