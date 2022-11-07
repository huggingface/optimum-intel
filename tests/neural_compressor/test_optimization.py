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
)

from optimum.intel.neural_compressor import IncDistiller, IncOptimizer, IncPruner, IncQuantizer, IncTrainer
from optimum.intel.neural_compressor.configuration import (
    IncDistillationConfig,
    IncPruningConfig,
    IncQuantizationConfig,
)
from optimum.intel.neural_compressor.quantization import (
    IncQuantizationMode,
    IncQuantizedModelForSequenceClassification,
)


os.environ["CUDA_VISIBLE_DEVICES"] = ""


class IncQuantizationTest(unittest.TestCase):
    @staticmethod
    def helper(model_name, output_dir, do_train=False, max_train_samples=128, max_eval_samples=128):
        task = "sst2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = load_metric("glue", task)

        if do_train:
            dataset = load_dataset("glue", task)
        else:
            dataset = load_dataset("glue", task, split="validation")

        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )

        if do_train:
            train_dataset = dataset["train"].select(range(max_train_samples))
            eval_dataset = dataset["validation"].select(range(max_eval_samples))
        else:
            train_dataset = None
            eval_dataset = dataset.select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        training_args = TrainingArguments(output_dir, num_train_epochs=1.0 if do_train else 0.0)

        trainer = IncTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics["eval_accuracy"]

        return model, trainer, eval_func

    def test_dynamic_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, trainer, eval_func = self.helper(model_name, tmp_dir)
            model_result = eval_func(model)
            q8_config = IncQuantizationConfig.from_pretrained(config_path)
            q8_config.set_config("quantization.approach", IncQuantizationMode.DYNAMIC.value)
            quantizer = IncQuantizer(q8_config, eval_func=eval_func)
            optimizer = IncOptimizer(model, quantizer=quantizer)
            q_model = optimizer.fit()
            q_model_result = eval_func(q_model)

            # Verification accuracy loss is under 3%
            self.assertGreaterEqual(q_model_result, model_result * 0.97)

            optimizer.save_pretrained(tmp_dir)
            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            loaded_model.eval()
            loaded_model_result = eval_func(loaded_model)

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_result, loaded_model_result)

    def test_static_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, trainer, eval_func = self.helper(model_name, tmp_dir)
            model.config.save_pretrained(tmp_dir)
            model_result = eval_func(model)
            q8_config = IncQuantizationConfig.from_pretrained(config_path)
            q8_config.set_config("quantization.approach", IncQuantizationMode.STATIC.value)
            q8_config.set_config("tuning.accuracy_criterion.relative", 0.04)
            q8_config.set_config("model.framework", "pytorch_fx")
            quantizer = IncQuantizer(q8_config, eval_func=eval_func, calib_dataloader=trainer.get_eval_dataloader())
            optimizer = IncOptimizer(model, quantizer=quantizer)
            q_model = optimizer.fit()
            q_model_result = eval_func(q_model)

            # Verification accuracy loss is under 4%
            self.assertGreaterEqual(q_model_result, model_result * 0.96)

            optimizer.save_pretrained(tmp_dir)

            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            loaded_model.eval()
            loaded_model_result = eval_func(loaded_model)

            # Verification quantized model was correctly loaded
            self.assertEqual(q_model_result, loaded_model_result)

    def test_quantization_aware_training(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sst2"
        max_eval_samples = 64
        max_train_samples = 64

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = load_metric("glue", task)
        dataset = load_dataset("glue", task)
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(max_train_samples))
        eval_dataset = dataset["validation"].select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        def train_func(model):
            trainer.model_wrapped = model
            trainer.model = model
            _ = trainer.train()
            return trainer.model

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics["eval_accuracy"]

        config_path = os.path.dirname(os.path.abspath(__file__))

        q8_config = IncQuantizationConfig.from_pretrained(config_path)
        q8_config.set_config("quantization.approach", IncQuantizationMode.AWARE_TRAINING.value)
        q8_config.set_config("tuning.accuracy_criterion.relative", 0.2)
        q8_config.set_config("model.framework", "pytorch_fx")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(tmp_dir, num_train_epochs=1.0)

            trainer = IncTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )

            quantizer = IncQuantizer(
                q8_config, eval_func=eval_func, calib_dataloader=trainer.get_eval_dataloader(), train_func=train_func
            )
            optimizer = IncOptimizer(model, quantizer=quantizer)

            model_result = eval_func(model)
            optimized_model = optimizer.fit()

            optimized_model_result = eval_func(optimized_model)

            optimizer.save_pretrained(tmp_dir)

            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            loaded_model.eval()
            loaded_model_result = eval_func(loaded_model)

            # Verification accuracy loss is under 25%
            self.assertGreaterEqual(optimized_model_result, model_result * 0.75)

            # Verification quantized model was correctly loaded
            self.assertEqual(optimized_model_result, loaded_model_result)


class IncPrunerTest(unittest.TestCase):
    def test_pruning(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sst2"
        max_eval_samples = 64
        max_train_samples = 64
        target_sparsity = 0.02

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        metric = load_metric("glue", task)
        dataset = load_dataset("glue", task)
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(max_train_samples))
        eval_dataset = dataset["validation"].select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        def train_func(model):
            trainer.model_wrapped = model
            trainer.model = model
            _ = trainer.train(agent)
            return trainer.model

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics["eval_accuracy"]

        config_path = os.path.dirname(os.path.abspath(__file__))

        pruning_config = IncPruningConfig.from_pretrained(config_path, config_file_name="prune.yml")
        pruning_config.set_config("pruning.approach.weight_compression.start_epoch", 0)
        pruning_config.set_config("pruning.approach.weight_compression.end_epoch", 1)
        pruning_config.set_config("pruning.approach.weight_compression.initial_sparsity", 0.0)
        pruning_config.set_config("pruning.approach.weight_compression.target_sparsity", target_sparsity)

        pruner = IncPruner(pruning_config, eval_func=eval_func, train_func=train_func)
        optimizer = IncOptimizer(model, quantizer=None, pruner=pruner)
        agent = optimizer.get_agent()

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(tmp_dir, num_train_epochs=2.0)

            trainer = IncTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            model_result = eval_func(model)
            optimized_model = optimizer.fit()
            optimized_model_result = eval_func(optimized_model)
            sparsity = optimizer.get_sparsity()

            # Verification final sparsity is equal to the targeted sparsity
            self.assertGreaterEqual(round(sparsity), 0.5)

            # Verification accuracy loss is under 6%
            self.assertGreaterEqual(optimized_model_result, model_result * 0.94)


class IncDistillationTest(unittest.TestCase):
    def test_knowledge_distillation(self):
        model_name = "distilbert-base-uncased"
        teacher_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sst2"
        max_eval_samples = 128
        max_train_samples = 128

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)
        metric = load_metric("glue", task)
        dataset = load_dataset("glue", task)
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(max_train_samples))
        eval_dataset = dataset["validation"].select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        def train_func(model):
            trainer.model_wrapped = model
            trainer.model = model
            _ = trainer.train(agent)
            return trainer.model

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics["eval_accuracy"]

        config_path = os.path.dirname(os.path.abspath(__file__))
        distillation_config = IncDistillationConfig.from_pretrained(config_path, config_file_name="distillation.yml")
        distiller = IncDistiller(
            teacher_model=teacher_model,
            config=distillation_config,
            eval_func=eval_func,
            train_func=train_func,
        )
        optimizer = IncOptimizer(model, distiller=distiller)
        agent = optimizer.get_agent()

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(tmp_dir, num_train_epochs=1.0)
            trainer = IncTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            model_result = eval_func(model)
            optimized_model = optimizer.fit()
            optimized_model_result = eval_func(optimized_model)

            # Verification of the model's accuracy
            self.assertGreaterEqual(optimized_model_result, model_result)


class IncOptimizerTest(unittest.TestCase):
    def test_one_shot(self):
        model_name = "distilbert-base-uncased"
        teacher_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        task = "sst2"
        max_eval_samples = 64
        max_train_samples = 64

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)
        metric = load_metric("glue", task)
        dataset = load_dataset("glue", task)
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(max_train_samples))
        eval_dataset = dataset["validation"].select(range(max_eval_samples))

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        def train_func(model):
            trainer.model_wrapped = model
            trainer.model = model
            _ = trainer.train(agent)
            return trainer.model

        def eval_func(model):
            trainer.model = model
            metrics = trainer.evaluate()
            return metrics["eval_accuracy"]

        config_path = os.path.dirname(os.path.abspath(__file__))
        q8_config = IncQuantizationConfig.from_pretrained(config_path)
        q8_config.set_config("quantization.approach", IncQuantizationMode.AWARE_TRAINING.value)
        q8_config.set_config("tuning.accuracy_criterion.relative", 0.2)
        q8_config.set_config("model.framework", "pytorch_fx")
        distillation_config = IncDistillationConfig.from_pretrained(config_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(tmp_dir, num_train_epochs=1.0)
            trainer = IncTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )

            quantizer = IncQuantizer(
                q8_config, eval_func=eval_func, calib_dataloader=trainer.get_eval_dataloader(), train_func=train_func
            )
            distiller = IncDistiller(
                teacher_model=teacher_model,
                config=distillation_config,
                eval_func=eval_func,
                train_func=train_func,
            )
            optimizer = IncOptimizer(
                model,
                quantizer=quantizer,
                distiller=distiller,
                one_shot_optimization=True,
                eval_func=eval_func,
                train_func=train_func,
            )
            agent = optimizer.get_agent()

            model_result = eval_func(model)
            optimized_model = optimizer.fit()
            optimized_model_result = eval_func(optimized_model)

            optimizer.save_pretrained(tmp_dir)

            loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(tmp_dir)
            loaded_model.eval()
            loaded_model_result = eval_func(loaded_model)

            # Verification accuracy loss is under 25%
            self.assertGreaterEqual(optimized_model_result, model_result * 0.75)

            # Verification quantized model was correctly loaded
            self.assertEqual(optimized_model_result, loaded_model_result)
