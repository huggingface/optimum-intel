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
import copy
import torch
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
from transformers.utils import WEIGHTS_NAME

import evaluate
from optimum.intel.openvino.configuration import OVConfig, DEFAULT_QUANTIZATION_CONFIG
from optimum.intel.openvino import OVTrainingArguments
from optimum.intel.openvino.modeling import OVModelForQuestionAnswering, OVModelForSequenceClassification
from optimum.intel.openvino.quantization import OVQuantizer
from optimum.intel.openvino.trainer import OVTrainer
from parameterized import parameterized

from nncf.torch.dynamic_graph.graph_tracer import create_mock_tensor


def generate_mock_tokens(input_infos):
    mock_tokens = dict()
    for info in input_infos:
        single_batch_info = copy.copy(info)
        input_shape = tuple([1] + list(info.shape)[1:])
        single_batch_info.shape = input_shape
        mock_tokens[info.keyword] = create_mock_tensor(single_batch_info, "cpu")
    return mock_tokens


class OVQuantizerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        ("distilbert-base-uncased-finetuned-sst-2-english", 50, 38),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_static_quantization(self, model_name, expected_fake_quantize, expected_int8):
        def preprocess_function(examples, tokenizer):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

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
            quantizer.quantize(save_directory=tmp_dir, calibration_dataset=calibration_dataset)

            model = OVModelForSequenceClassification.from_pretrained(tmp_dir)

            num_int8 = 0
            num_fake_quantize = 0
            for elem in model.model.get_ops():
                if "FakeQuantize" in elem.name:
                    num_fake_quantize += 1
                if "8" in elem.get_element_type().get_type_name():
                    num_int8 += 1
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_int8)

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that that the configuration is correctly saved and loaded
            expected_config = OVConfig()
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(expected_config.to_dict()["compression"], loaded_config.to_dict()["compression"])


class OVQuantizerQATest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (("hf-internal-testing/tiny-random-BertForQuestionAnswering",),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_static_quantization(self, model_name):
        def preprocess_function(examples, tokenizer):
            return tokenizer(
                examples["question"], examples["context"], padding="max_length", max_length=64, truncation=True
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            quantizer = OVQuantizer.from_pretrained(transformers_model)
            calibration_dataset = quantizer.get_calibration_dataset(
                "squadshifts",
                dataset_config_name="new_wiki",
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=10,
                dataset_split="test",
            )
            quantizer.quantize(save_directory=tmp_dir, calibration_dataset=calibration_dataset)

            # Test that inference on quantized model works
            model = OVModelForQuestionAnswering.from_pretrained(tmp_dir)
            tokens = tokenizer.encode_plus(
                "This is a sample question", "This is a sample context", add_special_tokens=True, return_tensors="pt"
            )
            outputs = model(**tokens, return_dict=True)

            # Test loading model a second time to catch issues with caching
            try:
                model = OVModelForQuestionAnswering.from_pretrained(tmp_dir)
            except RuntimeError:
                self.fail("Loading BERT QA model a second time failed")


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


class OVTrainerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (("distilbert-base-uncased", 50, 38),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_aware_training_quantization(self, model_name, expected_fake_quantize, expected_int8):
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ov_config = OVConfig()
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(16))
        eval_dataset = dataset["validation"].select(range(16))
        metric = evaluate.load("glue", "sst2")
        compute_metrics = lambda p: metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = OVTrainer(
                model=model,
                ov_config=ov_config,
                task="sequence-classification",
                args=TrainingArguments(tmp_dir, num_train_epochs=1.0, do_train=True, do_eval=True),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            train_result = trainer.train()
            metrics = trainer.evaluate()
            trainer.save_model()

            model = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            num_int8 = 0
            num_fake_quantize = 0
            for elem in model.model.get_ops():
                if "FakeQuantize" in elem.name:
                    num_fake_quantize += 1
                if "8" in elem.get_element_type().get_type_name():
                    num_int8 += 1
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_int8)

    def build_glue_sst_trainer(
        self, output_dir, tokenizer, model, teacher_model=None, ov_config=OVConfig(), **training_args
    ):
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(16))
        eval_dataset = dataset["validation"].select(range(16))
        metric = evaluate.load("glue", "sst2")

        def compute_metrics(p):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        trainer = OVTrainer(
            model=model,
            teacher_model=teacher_model,
            ov_config=ov_config,
            feature="sequence-classification",
            args=OVTrainingArguments(
                output_dir, num_train_epochs=3.0, do_train=True, do_eval=True, logging_steps=1, **training_args
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
        return trainer

    def count_quantization_op_number(self, ovmodel):
        num_fake_quantize = 0
        num_int8 = 0
        for elem in ovmodel.model.get_ops():
            if "FakeQuantize" in elem.name:
                num_fake_quantize += 1
            if "8" in elem.get_element_type().get_type_name():
                num_int8 += 1
        return num_fake_quantize, num_int8

    def test_training_quantization_distillation(self):
        model_name = "hf-internal-testing/tiny-bert"
        teacher_model_name = "hf-internal-testing/tiny-bert"
        expected_fake_quantize = 19
        expected_int8 = 14

        ov_config = OVConfig()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ov_config.log_dir = tmp_dir
            trainer = self.build_glue_sst_trainer(
                tmp_dir,
                tokenizer=tokenizer,
                model=model,
                teacher_model=teacher_model,
                ov_config=ov_config,
            )
            trainer.train()
            trainer.save_model()
            self.assertIn("distillation_loss", trainer.compression_metrics)

            ovmodel = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            num_fake_quantize, num_int8 = self.count_quantization_op_number(ovmodel)
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_int8)

            tokens = generate_mock_tokens(trainer.model.input_infos)
            outputs = ovmodel(**tokens)
            self.assertTrue("logits" in outputs)

    def test_training_movement_sparsity(self):
        model_name = "hf-internal-testing/tiny-bert"
        expected_binary_masks = 24
        ov_config = OVConfig(compression=MOVEMENT_SPARSITY_CONFIG_FOR_BERT)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ov_config.log_dir = tmp_dir
            trainer = self.build_glue_sst_trainer(
                tmp_dir,
                tokenizer=tokenizer,
                model=model,
                ov_config=ov_config,
            )
            trainer.train()
            trainer.save_model()

            state_dict = torch.load(Path(tmp_dir, WEIGHTS_NAME), map_location="cpu")
            num_binary_masks = sum(key.endswith("_binary_mask") for key in state_dict)
            self.assertEqual(expected_binary_masks, num_binary_masks)

            ovmodel = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            tokens = generate_mock_tokens(trainer.model.input_infos)
            outputs = ovmodel(**tokens)
            self.assertTrue("logits" in outputs)

    def test_training_movement_sparsity_quantization(self):
        model_name = "hf-internal-testing/tiny-bert"
        expected_binary_masks = 24
        expected_fake_quantize = 19
        expected_int8 = 14

        ov_config = OVConfig(compression=[MOVEMENT_SPARSITY_CONFIG_FOR_BERT, DEFAULT_QUANTIZATION_CONFIG])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ov_config.log_dir = tmp_dir
            trainer = self.build_glue_sst_trainer(
                tmp_dir,
                tokenizer=tokenizer,
                model=model,
                ov_config=ov_config,
            )
            trainer.train()
            trainer.save_model()
            self.assertIn("compression_loss", trainer.compression_metrics)

            state_dict = torch.load(Path(tmp_dir, WEIGHTS_NAME), map_location="cpu")
            num_binary_masks = sum(key.endswith("_binary_mask") for key in state_dict)
            self.assertEqual(expected_binary_masks, num_binary_masks)

            ovmodel = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            num_fake_quantize, num_int8 = self.count_quantization_op_number(ovmodel)
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_int8)

            tokens = generate_mock_tokens(trainer.model.input_infos)
            outputs = ovmodel(**tokens)
            self.assertTrue("logits" in outputs)

    def test_training_movement_sparsity_quantization_distillation(self):
        model_name = "hf-internal-testing/tiny-bert"
        teacher_model_name = "hf-internal-testing/tiny-bert"
        expected_binary_masks = 24
        expected_fake_quantize = 19
        expected_int8 = 14

        ov_config = OVConfig(compression=[MOVEMENT_SPARSITY_CONFIG_FOR_BERT, DEFAULT_QUANTIZATION_CONFIG])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ov_config.log_dir = tmp_dir
            trainer = self.build_glue_sst_trainer(
                tmp_dir,
                tokenizer=tokenizer,
                model=model,
                teacher_model=teacher_model,
                ov_config=ov_config,
            )
            trainer.train()
            trainer.save_model()
            self.assertIn("distillation_loss", trainer.compression_metrics)
            self.assertIn("compression_loss", trainer.compression_metrics)

            state_dict = torch.load(Path(tmp_dir, WEIGHTS_NAME), map_location="cpu")
            num_binary_masks = sum(key.endswith("_binary_mask") for key in state_dict)
            self.assertEqual(expected_binary_masks, num_binary_masks)

            ovmodel = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            num_fake_quantize, num_int8 = self.count_quantization_op_number(ovmodel)
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_int8)

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = ovmodel(**tokens)
            self.assertTrue("logits" in outputs)
