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
from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    EvalPrediction,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

import evaluate
from neural_compressor.config import (
    DistillationConfig,
    PostTrainingQuantConfig,
    PruningConfig,
    QuantizationAwareTrainingConfig,
)
from optimum.intel.neural_compressor import INCQuantizedModelForSequenceClassification, INCQuantizer, INCTrainer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(1009)


class QuantizationTest(unittest.TestCase):
    def test_dynamic_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        quantization_config = PostTrainingQuantConfig(approach="dynamic")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer("This is a sample input", return_tensors="pt")

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer = INCQuantizer.from_pretrained(model)
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=tmp_dir,
                save_onnx_model=True,
            )
            transformers_model = INCQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_outputs = onnx_model(**tokens)
            self.assertTrue("logits" in onnx_outputs)
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            # TODO: Enable
            # self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

    def test_static_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        quantization_config = PostTrainingQuantConfig(approach="static")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer("This is a sample input", return_tensors="pt")

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
            transformers_model = INCQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_outputs = onnx_model(**tokens)
            self.assertTrue("logits" in onnx_outputs)
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            # TODO: Enable
            # self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))

    def test_aware_training_quantization(self):
        model_name = "distilbert-base-uncased"
        quantization_config = QuantizationAwareTrainingConfig()
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        metric = evaluate.load("accuracy")
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = INCTrainer(
                model=model,
                quantization_config=quantization_config,
                task="sequence-classification",
                args=TrainingArguments(tmp_dir, num_train_epochs=1.0, do_train=True, do_eval=True),
                train_dataset=dataset["train"].select(range(64)),
                eval_dataset=dataset["validation"].select(range(64)),
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            train_result = trainer.train()
            metrics = trainer.evaluate()
            trainer.save_model(save_onnx_model=True)

            transformers_model = INCQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_outputs = onnx_model(**tokens)
            self.assertTrue("logits" in onnx_outputs)
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            # TODO: Enable
            # self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))


class PruningTest(unittest.TestCase):
    def test_magnitude_pruning(self):
        model_name = "distilbert-base-uncased"
        target_sparsity = 0.9
        pruning_config = PruningConfig(start_epoch=0, end_epoch=1, target_sparsity=target_sparsity)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        metric = evaluate.load("accuracy")
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = INCTrainer(
                model=model,
                pruning_config=pruning_config,
                task="sequence-classification",
                args=TrainingArguments(tmp_dir, num_train_epochs=2.0, do_train=True, do_eval=True),
                train_dataset=dataset["train"].select(range(64)),
                eval_dataset=dataset["validation"].select(range(64)),
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            train_result = trainer.train()
            metrics = trainer.evaluate()
            trainer.save_model(save_onnx_model=True)
            transformers_model = AutoModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_outputs = onnx_model(**tokens)
            self.assertTrue("logits" in onnx_outputs)
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))
            sparsity = trainer.get_model_sparsity()
            self.assertGreaterEqual(sparsity, target_sparsity * 100)


class DistillationTest(unittest.TestCase):
    def test_knowledge_distillation(self):
        model_name = "distilbert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        metric = evaluate.load("accuracy")
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        distillation_config = DistillationConfig(teacher_model=model)

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = INCTrainer(
                model=model,
                distillation_config=distillation_config,
                task="sequence-classification",
                args=TrainingArguments(tmp_dir, num_train_epochs=2.0, do_train=True, do_eval=True),
                train_dataset=dataset["train"].select(range(64)),
                eval_dataset=dataset["validation"].select(range(64)),
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            train_result = trainer.train()
            metrics = trainer.evaluate()
            trainer.save_model(save_onnx_model=True)
            transformers_model = AutoModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_outputs = onnx_model(**tokens)
            self.assertTrue("logits" in onnx_outputs)
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            self.assertTrue(torch.allclose(onnx_outputs.logits, transformers_outputs.logits, atol=1e-4))
