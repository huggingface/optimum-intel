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

import random
import re
import shutil
import tempfile
import unittest
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Union

import cpuinfo
import evaluate
import numpy as np
import pytest
import torch
from datasets import load_dataset
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsityController
from parameterized import parameterized
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)
from transformers.testing_utils import slow
from transformers.trainer_utils import EvalPrediction, TrainOutput
from transformers.utils import WEIGHTS_NAME
from utils_tests import MODEL_NAMES

from optimum.intel.openvino import OVTrainingArguments
from optimum.intel.openvino.configuration import OVConfig
from optimum.intel.openvino.modeling import (
    OVModel,
    OVModelForAudioClassification,
    OVModelForImageClassification,
    OVModelForSequenceClassification,
)
from optimum.intel.openvino.trainer import DEFAULT_QUANTIZATION_CONFIG, OVTrainer
from optimum.intel.openvino.utils import OV_XML_FILE_NAME
from optimum.intel.utils.import_utils import is_transformers_version


F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}


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


class OVTrainerBaseTrainingTest(unittest.TestCase, ABC):
    ovmodel_cls = OVModel
    task = "unknown"

    def setUp(self):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        self.output_dir = tempfile.mkdtemp()

    def run_ovtrainer_training_checks(self, desc: OVTrainerTestDescriptor):
        self.prepare_model_and_dataset(desc)
        self.args = self.get_training_args()
        self.ov_config = self.get_ov_config(desc.nncf_compression_config)
        self.trainer = self.get_ov_trainer()

        trainer = self.trainer
        self.override_movement_sparsifier_initialization(trainer)

        # check evaluation can work even before training
        metrics = trainer.evaluate()
        self.check_eval_metrics(metrics)

        # check trainining & saving
        train_output = trainer.train()
        self.check_train_output(train_output)
        self.check_compression_metrics(desc.compression_metrics)

        # check model can be saved
        trainer.save_model()
        self.check_model_saving()

        # check saved ovmodel IR and output
        ovmodel = self.get_ov_model()
        # dynamic batch size for tiny-swin does not work in OpenVINO 2023.0
        is_swin = "swin" in desc.model_id.lower()
        self.check_if_ovmodel_is_dynamic(ovmodel, expected_result=not is_swin)
        self.check_ovmodel_output_equals_torch_output(ovmodel, trainer.model)
        self.check_ovmodel_reshaping(ovmodel)

        # check ovmodel quantization ops
        self.check_quantization_op_number(ovmodel, desc.expected_fake_quantize, desc.expected_int8)

        # check binary mask in sparsity/pruning algorithms
        self.check_binary_mask_number(desc.expected_binary_masks)

    @abstractmethod
    def prepare_model_and_dataset(self, desc: OVTrainerTestDescriptor):
        pass

    @abstractmethod
    def check_ovmodel_output_equals_torch_output(self, ovmodel, torch_model):
        pass

    @abstractmethod
    def check_ovmodel_reshaping(self, ovmodel: OVModel):
        pass

    def compute_metric(self, predictions: EvalPrediction):
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=np.argmax(predictions.predictions, axis=1), references=predictions.label_ids)

    def check_eval_metrics(self, metrics: Dict[str, float]):
        for eval_metric in ["loss", "accuracy"]:
            self.assertIn(f"eval_{eval_metric}", metrics)

    def check_train_output(self, train_output: TrainOutput):
        self.assertIsInstance(train_output, TrainOutput)
        total_steps = (
            ceil(len(self.train_dataset) / self.args.per_device_train_batch_size) * self.args.num_train_epochs
        )
        self.assertEqual(train_output.global_step, total_steps)

    def check_model_saving(self):
        for file_name in [WEIGHTS_NAME, OV_XML_FILE_NAME, OV_XML_FILE_NAME.replace(".xml", ".bin")]:
            self.assertTrue(Path(self.output_dir, file_name).is_file())

    def check_compression_metrics(self, expected_compression_metrics: List[str]):
        self.assertEqual(sorted(expected_compression_metrics), sorted(self.trainer.compression_metrics.keys()))

    def check_quantization_op_number(self, ovmodel: OVModel, expected_fake_quantize: int, expected_int8: int):
        num_fake_quantize = 0
        num_int8 = 0
        for elem in ovmodel.model.get_ops():
            if "FakeQuantize" in elem.name:
                num_fake_quantize += 1
            for i in range(elem.get_output_size()):
                if "8" in elem.get_output_element_type(i).get_type_name():
                    num_int8 += 1
        self.assertEqual(expected_fake_quantize, num_fake_quantize)
        self.assertEqual(expected_int8, num_int8)

    def check_binary_mask_number(self, expected_binary_masks: int):
        state_dict = torch.load(Path(self.output_dir, WEIGHTS_NAME), map_location="cpu")
        num_binary_masks = sum(key.endswith("_binary_mask") for key in state_dict)
        self.assertEqual(expected_binary_masks, num_binary_masks)

    def check_if_ovmodel_is_dynamic(self, ovmodel: OVModel, expected_result: bool = True):
        if expected_result is True:
            self.assertTrue(ovmodel.model.is_dynamic())
        else:
            self.assertFalse(ovmodel.model.is_dynamic())

    def override_movement_sparsifier_initialization(self, trainer: OVTrainer, sparsity=0.95):
        movement_controller = trainer._get_compression_controller_by_cls(
            MovementSparsityController
        )  # pylint: disable=protected-access
        if movement_controller is not None:
            # make sure the binary masks will have many zeros
            initialize_movement_sparsifier_parameters_by_sparsity(movement_controller, sparsity=sparsity)

    def get_training_args(self, train_batch_size=4, eval_batch_size=1, num_train_epochs=3) -> OVTrainingArguments:
        args = OVTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=1e-7,
            do_train=True,
            do_eval=True,
            logging_steps=1,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            no_cuda=True,
            full_determinism=True,
            remove_unused_columns=False,
        )
        return args

    def get_ov_trainer(self) -> OVTrainer:
        return OVTrainer(
            model=self.model,
            teacher_model=self.teacher_model,
            args=self.args,
            ov_config=self.ov_config,
            task=self.task,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metric,
            data_collator=self.data_collator,
        )

    def get_ov_config(self, nncf_compression_config: Union[List[Dict], Dict, None]) -> OVConfig:
        ov_config = OVConfig()
        if not is_avx_vnni_supported():
            # should enable "overflow_fix" in quantization otherwise accuracy degradation may be seen
            nncf_compression_config = self.get_nncf_config_with_overflow_fix_override(
                nncf_compression_config, "enable"
            )
        ov_config.compression = nncf_compression_config
        return ov_config

    def get_ov_model(self, model_id=None) -> OVModel:
        model_id = model_id or self.output_dir
        return self.ovmodel_cls.from_pretrained(model_id, ov_config=F32_CONFIG)

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

    def tearDown(self):
        shutil.rmtree(self.output_dir)


CUSTOMIZED_QUANTIZATION_CONFIG = deepcopy(DEFAULT_QUANTIZATION_CONFIG)
CUSTOMIZED_QUANTIZATION_CONFIG.update(
    {
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
    }
)

STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT = {
    "algorithm": "movement_sparsity",
    "params": {
        "warmup_start_epoch": 1,
        "warmup_end_epoch": 2,
        "importance_regularization_factor": 1.0,
        "enable_structured_masking": True,
    },
    "sparse_structure_by_scopes": [
        {"mode": "block", "sparse_factors": [8, 8], "target_scopes": "{re}.*BertAttention.*"},
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*BertIntermediate.*"},
        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*BertOutput.*"},
    ],
    "ignored_scopes": ["{re}.*NNCFEmbedding", "{re}.*LayerNorm.*", "{re}.*pooler.*", "{re}.*classifier.*"],
}

UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT = deepcopy(STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT)
UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT["params"]["enable_structured_masking"] = False

# TODO: Uncomment failes tests after NNCF 2.8.1 patch release
OVTRAINER_TEXT_CLASSIFICATION_TEST_DESCRIPTORS = {
    "distillation": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[],
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "default_quantization": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
        expected_fake_quantize=22,
        expected_int8=32,
        compression_metrics=["compression_loss"],
    ),
    "distillation,default_quantization": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
        expected_fake_quantize=22,
        expected_int8=32,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "customized_quantization": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
        expected_fake_quantize=22,
        expected_int8=32,
        compression_metrics=["compression_loss"],
    ),
    "distillation,customized_quantization": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=CUSTOMIZED_QUANTIZATION_CONFIG,
        expected_fake_quantize=22,
        expected_int8=32,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=60,
        compression_metrics=["compression_loss"],
    ),
    "distillation,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=60,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss"],
    ),
    "customized_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[
            CUSTOMIZED_QUANTIZATION_CONFIG,
            STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        ],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss"],
    ),
    "distillation,default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "distillation,customized_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[
            CUSTOMIZED_QUANTIZATION_CONFIG,
            STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        ],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=60,
        compression_metrics=["compression_loss"],
    ),
    "distillation,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        expected_binary_masks=60,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss"],
    ),
    "customized_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[
            CUSTOMIZED_QUANTIZATION_CONFIG,
            UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        ],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss"],
    ),
    "distillation,default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[DEFAULT_QUANTIZATION_CONFIG, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "distillation,customized_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["bert"],
        teacher_model_id=MODEL_NAMES["bert"],
        nncf_compression_config=[
            CUSTOMIZED_QUANTIZATION_CONFIG,
            UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_BERT,
        ],
        expected_fake_quantize=22,
        expected_int8=32,
        expected_binary_masks=60,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
}


class OVTrainerTextClassificationTrainingTest(OVTrainerBaseTrainingTest):
    ovmodel_cls = OVModelForSequenceClassification
    task = "sequence-classification"

    @parameterized.expand(OVTRAINER_TEXT_CLASSIFICATION_TEST_DESCRIPTORS.items())
    @unittest.skipIf(is_transformers_version("<", "4.41.0"), reason="Mismatch in expected fake quantized op")
    def test_training(self, _, desc: OVTrainerTestDescriptor):
        self.run_ovtrainer_training_checks(desc)

    def prepare_model_and_dataset(self, desc: OVTrainerTestDescriptor):
        self.dataset = load_dataset("glue", "sst2")
        self.num_labels = len(self.dataset["train"].features["label"].names)

        self.tokenizer = AutoTokenizer.from_pretrained(desc.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(desc.model_id, num_labels=self.num_labels)
        self.teacher_model = None
        if desc.teacher_model_id:
            self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
                desc.teacher_model_id, num_labels=self.num_labels
            )

        def data_transform(examples, max_length: int = 128):
            result = self.tokenizer(
                examples["sentence"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            result["labels"] = examples["label"]
            return result

        self.data_transform = data_transform
        self.train_dataset = self.dataset["train"].select(range(8))
        self.eval_dataset = self.dataset["validation"].select(range(4))
        self.train_dataset.set_transform(data_transform)
        self.eval_dataset.set_transform(data_transform)
        self.data_collator = None

    def check_ovmodel_output_equals_torch_output(self, ovmodel, torch_model):
        torch_model = torch_model.eval()
        for batch_size in [1, 4]:
            self.trainer.args = self.get_training_args(eval_batch_size=batch_size)
            self.trainer.create_accelerator_and_postprocess()
            for seq_length in [16, 89, 128]:
                dataset = deepcopy(self.eval_dataset)
                dataset.set_transform(partial(self.data_transform, max_length=seq_length))
                for inputs in self.trainer.get_eval_dataloader(dataset):
                    self.assertSequenceEqual(inputs["input_ids"].shape, [batch_size, seq_length])
                    ovmodel_outputs = ovmodel(**inputs)
                    self.assertIn("logits", ovmodel_outputs)
                    ovmodel_logits = ovmodel_outputs.logits
                    with torch.no_grad():
                        torch_logits = torch_model(**inputs).logits
                    torch.testing.assert_close(
                        ovmodel_logits,
                        torch_logits,
                        atol=1e-3,
                        rtol=1e-4,
                    )

    def check_ovmodel_reshaping(self, ovmodel: OVModel):
        self.check_if_ovmodel_is_dynamic(ovmodel, True)
        for batch_size in [1, 4]:
            for seq_length in [16, 89, 128]:
                static_shape = [batch_size, seq_length]
                ovmodel.reshape(*static_shape)
                self.check_if_ovmodel_is_dynamic(ovmodel, False)
                for input_ in ovmodel.model.inputs:
                    self.assertSequenceEqual(list(input_.get_shape()), static_shape)
                ovmodel.reshape(-1, -1)
                self.check_if_ovmodel_is_dynamic(ovmodel, True)


STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN = {
    "algorithm": "movement_sparsity",
    "params": {
        "warmup_start_epoch": 1,
        "warmup_end_epoch": 2,
        "importance_regularization_factor": 1.0,
        "enable_structured_masking": True,
    },
    "sparse_structure_by_scopes": [
        {"mode": "block", "sparse_factors": [4, 4], "target_scopes": "{re}.*SwinAttention.*"},
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*SwinIntermediate.*"},
        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*SwinOutput.*"},
    ],
    "ignored_scopes": ["{re}.*PatchEmbed.*", "{re}.*PatchMerging.*", "{re}.*classifier.*", "{re}.*LayerNorm.*"],
}
UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN = deepcopy(STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN)
UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN["params"]["enable_structured_masking"] = False
OVTRAINER_IMAGE_CLASSIFICATION_TEST_DESCRIPTORS = {
    "default_quantization": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["swin"],
        nncf_compression_config=DEFAULT_QUANTIZATION_CONFIG,
        expected_fake_quantize=35,
        expected_int8=27,
        compression_metrics=["compression_loss"],
    ),
    "structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["swin"],
        nncf_compression_config=STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["swin"],
        nncf_compression_config=UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["swin"],
        nncf_compression_config=[STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN, DEFAULT_QUANTIZATION_CONFIG],
        expected_fake_quantize=35,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["swin"],
        nncf_compression_config=[UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN, DEFAULT_QUANTIZATION_CONFIG],
        expected_fake_quantize=35,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "distillation,default_quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["swin"],
        teacher_model_id=MODEL_NAMES["swin"],
        nncf_compression_config=[STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN, DEFAULT_QUANTIZATION_CONFIG],
        expected_fake_quantize=35,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "distillation,default_quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["swin"],
        teacher_model_id=MODEL_NAMES["swin"],
        nncf_compression_config=[UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_SWIN, DEFAULT_QUANTIZATION_CONFIG],
        expected_fake_quantize=35,
        expected_int8=27,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
}
# TODO : can be moved to MODEL_NAMES["swin-window"] after transformers v4.42.3


class OVTrainerImageClassificationTrainingTest(OVTrainerBaseTrainingTest):
    ovmodel_cls = OVModelForImageClassification
    task = "image-classification"

    @parameterized.expand(OVTRAINER_IMAGE_CLASSIFICATION_TEST_DESCRIPTORS.items())
    @pytest.mark.run_slow
    @slow
    @unittest.skipIf(is_transformers_version("<", "4.41.0"), reason="Mismatch in expected fake quantized op")
    def test_training(self, _, desc: OVTrainerTestDescriptor):
        self.run_ovtrainer_training_checks(desc)

    def prepare_model_and_dataset(self, desc: OVTrainerTestDescriptor):
        self.dataset = load_dataset("hf-internal-testing/cats_vs_dogs_sample", trust_remote_code=True)
        self.num_labels = len(self.dataset["train"].features["labels"].names)

        self.feature_extractor = AutoImageProcessor.from_pretrained(desc.model_id)
        self.tokenizer = self.feature_extractor
        self.model = AutoModelForImageClassification.from_pretrained(desc.model_id, num_labels=self.num_labels)
        self.teacher_model = None
        if desc.teacher_model_id:
            self.teacher_model = AutoModelForImageClassification.from_pretrained(
                desc.teacher_model_id, num_labels=self.num_labels
            )

        def data_transform(examples, size=None):
            result = self.feature_extractor(examples["image"], size=size, return_tensors="pt")
            result["labels"] = examples["labels"]
            return result

        self.data_transform = data_transform
        self.dataset.set_transform(data_transform)
        raw_dataset = self.dataset["train"].shuffle(seed=42)
        self.train_dataset = raw_dataset.select(range(8))
        self.eval_dataset = raw_dataset.select(range(8, 12))
        self.data_collator = default_data_collator
        self.is_swin = "swin" in desc.model_id.lower()

    def get_ov_model(self, model_id=None) -> OVModel:
        # image models, e.g. swin, may require a determined image size
        model_id = model_id or self.output_dir
        size = (self.feature_extractor.size["height"], self.feature_extractor.size["width"])
        ovmodel = self.ovmodel_cls.from_pretrained(model_id, compile=False, ov_config=F32_CONFIG)
        # dynamic batch size for tiny-swin does not work in OpenVINO 2023.0
        batch_size = 1 if self.is_swin else -1
        ovmodel.reshape(batch_size, 3, *size)
        ovmodel.compile()
        return ovmodel

    def check_ovmodel_output_equals_torch_output(self, ovmodel, torch_model):
        torch_model = torch_model.eval()
        batch_sizes = [1] if self.is_swin else [1, 4]
        for batch_size in batch_sizes:
            self.trainer.args = self.get_training_args(eval_batch_size=batch_size)
            self.trainer.create_accelerator_and_postprocess()
            for inputs in self.trainer.get_eval_dataloader():
                self.assertEqual(inputs["pixel_values"].shape[0], batch_size)
                ovmodel_outputs = ovmodel(**inputs)
                self.assertIn("logits", ovmodel_outputs)
                ovmodel_logits = ovmodel_outputs.logits
                with torch.no_grad():
                    torch_logits = torch_model(**inputs).logits
                torch.testing.assert_close(
                    ovmodel_logits,
                    torch_logits,
                    atol=1e-3,
                    rtol=1e-4,
                )

    def check_ovmodel_reshaping(self, ovmodel: OVModel):
        # dynamic batch size for tiny-swin does not work in OpenVINO 2023.0
        self.check_if_ovmodel_is_dynamic(ovmodel, not self.is_swin)
        size = (self.feature_extractor.size["height"], self.feature_extractor.size["width"])
        dynamic_shape = [-1, 3, *size]
        for batch_size in [1, 4]:
            static_shape = [batch_size] + dynamic_shape[1:]
            ovmodel.reshape(*static_shape)
            self.check_if_ovmodel_is_dynamic(ovmodel, False)
            for input_ in ovmodel.model.inputs:
                self.assertSequenceEqual(list(input_.get_shape()), static_shape)
            if not self.is_swin:
                ovmodel.reshape(*dynamic_shape)
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
    "ignored_scopes": ["{re}.*__add___[0-1]", "{re}.*layer_norm_0"],
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
        model_id=MODEL_NAMES["wav2vec2-hf"],
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=40,
        expected_int8=30,
        compression_metrics=["compression_loss"],
    ),
    "structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["wav2vec2-hf"],
        nncf_compression_config=[STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["wav2vec2-hf"],
        nncf_compression_config=[UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["wav2vec2-hf"],
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=40,
        expected_int8=30,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["wav2vec2-hf"],
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=40,
        expected_int8=30,
        expected_binary_masks=48,
        compression_metrics=["compression_loss"],
    ),
    "distillation,quantization,structured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["wav2vec2-hf"],
        teacher_model_id=MODEL_NAMES["wav2vec2-hf"],
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, STRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=40,
        expected_int8=30,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
    "distillation,quantization,unstructured_movement_sparsity": OVTrainerTestDescriptor(
        model_id=MODEL_NAMES["wav2vec2-hf"],
        teacher_model_id=MODEL_NAMES["wav2vec2-hf"],
        nncf_compression_config=[QUANTIZATION_CONFIG_FOR_WAV2VEC2, UNSTRUCTURED_MOVEMENT_SPARSITY_CONFIG_FOR_WAV2VEC2],
        expected_fake_quantize=40,
        expected_int8=30,
        expected_binary_masks=48,
        compression_metrics=["compression_loss", "distillation_loss", "task_loss"],
    ),
}


class OVTrainerAudioClassificationTrainingTest(OVTrainerBaseTrainingTest):
    ovmodel_cls = OVModelForAudioClassification
    task = "audio-classification"

    @parameterized.expand(OVTRAINER_AUDIO_CLASSIFICATION_TEST_DESCRIPTORS.items())
    @pytest.mark.run_slow
    @slow
    def test_training(self, _, desc: OVTrainerTestDescriptor):
        self.run_ovtrainer_training_checks(desc)

    def prepare_model_and_dataset(self, desc: OVTrainerTestDescriptor):
        self.dataset = load_dataset("anton-l/superb_dummy", "ks")
        self.num_labels = len(self.dataset["test"].features["label"].names)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(desc.model_id)
        self.tokenizer = self.feature_extractor
        self.model = AutoModelForAudioClassification.from_pretrained(
            desc.model_id, num_labels=self.num_labels, attn_implementation="eager"
        )
        self.teacher_model = None
        if desc.teacher_model_id:
            self.teacher_model = AutoModelForAudioClassification.from_pretrained(
                desc.teacher_model_id, num_labels=self.num_labels
            )

        def data_transform(examples, max_length: int = 16000):
            sampling_rate = self.feature_extractor.sampling_rate
            batch = self.feature_extractor(
                examples["speech"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            batch["labels"] = examples["label"]
            return batch

        self.data_transform = data_transform
        self.dataset.set_transform(data_transform)
        self.train_dataset = self.dataset["test"].select(range(8))
        self.eval_dataset = self.dataset["test"].select(range(8, 12))
        self.data_collator = None

    def check_ovmodel_reshaping(self, ovmodel: OVModel):
        self.check_if_ovmodel_is_dynamic(ovmodel, True)
        for batch_size in [1, 4]:
            for seq_len in [12345, 16000]:
                static_shape = [batch_size, seq_len]
                ovmodel.reshape(*static_shape)
                self.check_if_ovmodel_is_dynamic(ovmodel, False)
                for input_ in ovmodel.model.inputs:
                    self.assertSequenceEqual(list(input_.get_shape()), static_shape)
                ovmodel.reshape(-1, -1)
                self.check_if_ovmodel_is_dynamic(ovmodel, True)

    def check_ovmodel_output_equals_torch_output(self, ovmodel, torch_model):
        torch_model = torch_model.eval()
        for batch_size in [1, 4]:
            self.trainer.args = self.get_training_args(eval_batch_size=batch_size)
            self.trainer.create_accelerator_and_postprocess()
            for seq_length in [12345, 16000]:
                dataset = deepcopy(self.eval_dataset)
                dataset.set_transform(partial(self.data_transform, max_length=seq_length))
                for inputs in self.trainer.get_eval_dataloader(dataset):
                    self.assertSequenceEqual(inputs["input_values"].shape, [batch_size, seq_length])
                    ovmodel_outputs = ovmodel(**inputs)
                    self.assertIn("logits", ovmodel_outputs)
                    ovmodel_logits = ovmodel_outputs.logits
                    with torch.no_grad():
                        torch_logits = torch_model(**inputs).logits
                    torch.testing.assert_close(
                        ovmodel_logits,
                        torch_logits,
                        atol=1e-3,
                        rtol=1e-4,
                    )
