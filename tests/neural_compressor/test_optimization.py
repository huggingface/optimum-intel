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

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from neural_compressor.config import (
    AccuracyCriterion,
    DistillationConfig,
    PostTrainingQuantConfig,
    QuantizationAwareTrainingConfig,
    TuningCriterion,
    WeightPruningConfig,
)
from onnx import load as onnx_load
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    default_data_collator,
    pipeline,
    set_seed,
)

from optimum.intel import (
    INCConfig,
    INCModelForQuestionAnswering,
    INCModelForSequenceClassification,
    INCQuantizer,
    INCStableDiffusionPipeline,
    INCTrainer,
)
from optimum.intel.utils.constant import DIFFUSION_WEIGHTS_NAME
from optimum.onnxruntime import ORTModelForSequenceClassification


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(1009)


def num_quantized_matmul_onnx_model(onnx_model):
    num_quantized_matmul = 0
    for initializer in onnx_model.graph.initializer:
        if "MatMul" in initializer.name and "quantized" in initializer.name:
            num_quantized_matmul += 1
    return num_quantized_matmul


class QuantizationTest(unittest.TestCase):
    def test_dynamic_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        expected_quantized_matmuls = 36
        quantization_config = PostTrainingQuantConfig(approach="dynamic")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        quantizer = INCQuantizer.from_pretrained(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=tmp_dir,
                save_onnx_model=True,
            )
            loaded_model = INCModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = onnx_load(os.path.join(tmp_dir, "model.onnx"))
            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertTrue(inc_config.save_onnx_model)
            self.assertFalse(inc_config.quantization["is_static"])

        num_quantized_matmul = num_quantized_matmul_onnx_model(onnx_model)
        self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)

        ort_outputs = ort_model(**tokens)
        self.assertTrue("logits" in ort_outputs)
        with torch.no_grad():
            model_outputs = quantizer._quantized_model(**tokens)
            loaded_model_outputs = loaded_model(**tokens)
        self.assertTrue(torch.equal(model_outputs.logits, loaded_model_outputs.logits))
        # self.assertTrue(torch.allclose(ort_outputs.logits, loaded_model_outputs.logits, atol=1e-4))

    def test_dynamic_accuracy_strategy_quantization(self):
        model_name = "distilbert-base-cased-distilled-squad"
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        eval_dataset = load_dataset("squad", split="validation").select(range(64))
        task_evaluator = evaluate.evaluator("question-answering")
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        tolerance_criterion = 0.1

        def eval_fn(model):
            qa_pipeline.model = model
            metrics = task_evaluator.compute(model_or_pipeline=qa_pipeline, data=eval_dataset, metric="squad")
            return metrics["f1"]

        original_model_metric = eval_fn(model)
        tuning_criterion = TuningCriterion(max_trials=10)
        accuracy_criterion = AccuracyCriterion(tolerable_loss=tolerance_criterion)
        quantization_config = PostTrainingQuantConfig(
            approach="dynamic", accuracy_criterion=accuracy_criterion, tuning_criterion=tuning_criterion
        )
        tokenizer("This is a sample input", return_tensors="pt")

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer = INCQuantizer.from_pretrained(model, eval_fn=eval_fn)
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=tmp_dir,
                save_onnx_model=True,
            )
            loaded_model = INCModelForQuestionAnswering.from_pretrained(tmp_dir)
            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertTrue(inc_config.save_onnx_model)
            self.assertFalse(inc_config.quantization["is_static"])

        quantized_model_metric = eval_fn(loaded_model)
        # Verification accuracy loss is under 5%
        self.assertGreaterEqual(quantized_model_metric, original_model_metric * (1 - tolerance_criterion))

    def test_static_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        expected_quantized_matmuls = 36
        num_samples = 10
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
            num_samples=num_samples,
            dataset_split="train",
        )
        quantizer = INCQuantizer.from_pretrained(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
                save_onnx_model=True,
            )
            loaded_model = INCModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = onnx_load(os.path.join(tmp_dir, "model.onnx"))
            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertTrue(inc_config.save_onnx_model)
            self.assertTrue(inc_config.quantization["is_static"])
            self.assertEqual(inc_config.quantization["dataset_num_samples"], num_samples)

        num_quantized_matmul = num_quantized_matmul_onnx_model(onnx_model)
        self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)

        ort_outputs = ort_model(**tokens)
        self.assertTrue("logits" in ort_outputs)
        with torch.no_grad():
            model_outputs = quantizer._quantized_model(**tokens)
            loaded_model_outputs = loaded_model(**tokens)
        self.assertTrue(torch.equal(model_outputs.logits, loaded_model_outputs.logits))
        # self.assertTrue(torch.allclose(ort_outputs.logits, loaded_model_outputs.logits, atol=1e-4))

    def test_ipex_static_quantization(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex")
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
            num_samples=10,
            dataset_split="train",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer = INCQuantizer.from_pretrained(model)
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
                save_onnx_model=False,
            )
            transformers_model = INCModelForSequenceClassification.from_pretrained(tmp_dir)
            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertFalse(inc_config.save_onnx_model)
            self.assertTrue(inc_config.quantization["is_static"])

            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
                model_outputs = quantizer._quantized_model(**tokens)

            self.assertTrue(torch.equal(model_outputs["logits"], transformers_outputs["logits"]))

    def test_aware_training_quantization(self):
        model_name = "distilbert-base-uncased"
        expected_quantized_matmuls = 36
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
                task="text-classification",
                args=TrainingArguments(tmp_dir, num_train_epochs=1.0, do_train=True, do_eval=False),
                train_dataset=dataset["train"].select(range(8)),
                eval_dataset=dataset["validation"].select(range(8)),
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            trainer.train()
            trainer.evaluate()
            trainer.save_model(save_onnx_model=True)
            loaded_model = INCModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            onnx_model = onnx_load(os.path.join(tmp_dir, "model.onnx"))
            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertTrue(inc_config.save_onnx_model)
            self.assertTrue(inc_config.quantization["is_static"])

        num_quantized_matmul = num_quantized_matmul_onnx_model(onnx_model)
        self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)

        ort_outputs = ort_model(**tokens)
        self.assertTrue("logits" in ort_outputs)
        trainer.model.eval()
        loaded_model.eval()
        with torch.no_grad():
            model_outputs = trainer.model(**tokens)
            loaded_model_outputs = loaded_model(**tokens)
        # self.assertTrue(torch.allclose(ort_outputs.logits, loaded_model_outputs.logits, atol=1e-4))
        self.assertTrue(torch.equal(model_outputs.logits, loaded_model_outputs.logits))

    def test_aware_training_quantization_pruning(self):
        model_name = "distilbert-base-uncased"
        target_sparsity = 0.9
        pruning_config = WeightPruningConfig(
            pruning_type="magnitude",
            start_step=0,
            end_step=15,
            target_sparsity=target_sparsity,
            pruning_scope="local",
        )
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
                pruning_config=pruning_config,
                task="sequence-classification",
                args=TrainingArguments(tmp_dir, num_train_epochs=1.0, do_train=True, do_eval=False),
                train_dataset=dataset["train"].select(range(8)),
                eval_dataset=dataset["validation"].select(range(8)),
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            trainer.train()
            trainer.evaluate()
            trainer.save_model(save_onnx_model=True)

            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertTrue(inc_config.save_onnx_model)
            self.assertTrue(inc_config.quantization["is_static"])

            transformers_model = INCModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_outputs = ort_model(**tokens)
            self.assertTrue("logits" in ort_outputs)
            with torch.no_grad():
                transformers_model(**tokens)
            # self.assertTrue(torch.allclose(ort_outputs.logits, transformers_outputs.logits, atol=1e-4))

    def test_dynamic_diffusion_model(self):
        model_id = "hf-internal-testing/diffusers-stable-diffusion-tiny-all"
        pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        pipeline.safety_checker = None
        num_images_per_prompt, height, width, scale_factor = 1, 512, 512, 8
        latents_shape = (
            num_images_per_prompt,
            pipeline.unet.in_channels,
            height // scale_factor,
            width // scale_factor,
        )
        latents = np.random.randn(*latents_shape).astype(np.float32)
        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "num_inference_steps": 1,
            "output_type": "np",
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
        }

        pipeline.to("cpu")
        quantization_config = PostTrainingQuantConfig(approach="dynamic")
        quantizer = INCQuantizer.from_pretrained(pipeline.unet)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=os.path.join(tmp_dir, "unet"),
                file_name=DIFFUSION_WEIGHTS_NAME,
            )
            loaded_pipeline = INCStableDiffusionPipeline.from_pretrained(tmp_dir)
            loaded_pipeline.to("cpu")
            pipeline.unet = quantizer._quantized_model
        with torch.no_grad():
            outputs = pipeline(latents=torch.from_numpy(latents), **kwargs).images
            loaded_pipe_outputs = loaded_pipeline(latents=torch.from_numpy(latents), **kwargs).images
        # Compare model outputs
        self.assertTrue(np.allclose(loaded_pipe_outputs, outputs, atol=1e-4))


class PruningTest(unittest.TestCase):
    def test_magnitude_pruning(self):
        model_name = "distilbert-base-uncased"
        target_sparsity = 0.9
        # end_step should be training_args.num_train_epochs * (len(train_dataset) // training_args.per_device_train_batch_size)
        pruning_config = WeightPruningConfig(
            pruning_type="magnitude",
            start_step=0,
            end_step=1,
            target_sparsity=target_sparsity,
            pruning_scope="local",
        )
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
                task="text-classification",
                args=TrainingArguments(tmp_dir, num_train_epochs=2.0, do_train=True, do_eval=False),
                train_dataset=dataset["train"].select(range(64)),
                eval_dataset=dataset["validation"].select(range(4)),
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            trainer.train()
            trainer.evaluate()
            trainer.save_model(save_onnx_model=True)

            inc_config = INCConfig.from_pretrained(tmp_dir)
            transformers_model = INCModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_outputs = ort_model(**tokens)
            self.assertTrue("logits" in ort_outputs)
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            self.assertTrue(torch.allclose(ort_outputs.logits, transformers_outputs.logits, atol=1e-4))
            sparsity = trainer.get_model_sparsity()
            self.assertGreaterEqual(sparsity, target_sparsity * 100 / 2)
            self.assertTrue(inc_config.save_onnx_model)
            self.assertEqual(inc_config.pruning["sparsity"], round(sparsity, 2))
            self.assertEqual(inc_config.pruning["approach"], "magnitude")
            self.assertEqual(inc_config.pruning["pattern"], "4x1")


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
                args=TrainingArguments(tmp_dir, num_train_epochs=2.0, do_train=True, do_eval=False),
                train_dataset=dataset["train"].select(range(8)),
                eval_dataset=dataset["validation"].select(range(8)),
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )
            trainer._set_task()
            self.assertEqual(trainer.task, "text-classification")
            trainer.train()
            trainer.evaluate()
            trainer.save_model(save_onnx_model=True)

            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertTrue(inc_config.save_onnx_model)
            self.assertEqual(inc_config.distillation["teacher_model_name_or_path"], model_name)
            self.assertEqual(inc_config.distillation["temperature"], 1.0)

            transformers_model = INCModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(tmp_dir)
            ort_outputs = ort_model(**tokens)
            self.assertTrue("logits" in ort_outputs)
            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens)
            self.assertTrue(torch.allclose(ort_outputs.logits, transformers_outputs.logits, atol=1e-4))
