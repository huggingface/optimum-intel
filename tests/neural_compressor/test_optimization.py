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

# ruff: noqa

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
from parameterized import parameterized
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
    INCModelForCausalLM,
    INCModelForQuestionAnswering,
    INCModelForSequenceClassification,
    INCQuantizer,
    INCStableDiffusionPipeline,
    INCTrainer,
)
from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS
from optimum.intel.utils.constant import DIFFUSION_WEIGHTS_NAME, ONNX_WEIGHTS_NAME
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification
from optimum.pipelines import ORT_SUPPORTED_TASKS


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(1009)

_TASK_TO_DATASET = {
    "text-classification": ("glue", "sst2", "sentence"),
    "text-generation": ("wikitext", "wikitext-2-raw-v1", "text"),
    "text2text-generation": ("cnn_dailymail", "3.0.0", ("article", "highlights")),
}


def num_quantized_matmul_onnx_model(onnx_model):
    num_quantized_matmul = 0
    for initializer in onnx_model.graph.initializer:
        if "MatMul" in initializer.name and "quantized" in initializer.name:
            num_quantized_matmul += 1
    return num_quantized_matmul


def _preprocess_function(examples, tokenizer, column_name):
    return tokenizer(examples[column_name], padding="max_length", max_length=128, truncation=True)


def _compute_metrics(outputs, metric):
    return metric.compute(predictions=np.argmax(outputs.predictions, axis=1), references=outputs.label_ids)


def _generate_dataset(quantizer, tokenizer, num_samples=10):
    dataset_name, dataset_config_name, column_name = _TASK_TO_DATASET[quantizer.task]
    dataset = quantizer.get_calibration_dataset(
        dataset_name,
        dataset_config_name=dataset_config_name,
        preprocess_function=partial(_preprocess_function, tokenizer=tokenizer, column_name=column_name),
        num_samples=num_samples,
        dataset_split="train",
    )
    return dataset


class OptimizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        ("text-classification", "hf-internal-testing/tiny-random-bert", 30),
        # ("text-generation", "hf-internal-testing/tiny-random-BloomForCausalLM", 1), ## TODO : enable causal lm task once INC ONNX export fixed
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_dynamic_quantization(self, task, model_name, expected_quantized_matmuls):
        quantization_config = PostTrainingQuantConfig(approach="dynamic")
        model = ORT_SUPPORTED_TASKS[task]["class"][0].auto_model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantizer = INCQuantizer.from_pretrained(model, task=task)
        save_onnx_model = task != "text-generation"
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=tmp_dir,
                save_onnx_model=save_onnx_model,
            )
            self.check_model_outputs(
                q_model=quantizer._quantized_model,
                task=task,
                tokenizer=tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=False,
                load_onnx_model=save_onnx_model,
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_static_quantization(self, task, model_name, expected_quantized_matmuls):
        num_samples = 10
        quantization_config = PostTrainingQuantConfig(approach="static")
        model = ORT_SUPPORTED_TASKS[task]["class"][0].auto_model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantizer = INCQuantizer.from_pretrained(model, task=task)
        calibration_dataset = _generate_dataset(quantizer, tokenizer, num_samples=num_samples)
        save_onnx_model = task != "text-generation"

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
                save_onnx_model=save_onnx_model,
            )
            self.check_model_outputs(
                q_model=quantizer._quantized_model,
                task=task,
                tokenizer=tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
                num_samples=num_samples,
                load_onnx_model=save_onnx_model,
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_ipex_static_quantization_with_smoothquant(self, task, model_name, expected_quantized_matmuls):
        recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}}
        num_samples = 10
        quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex", recipes=recipes)
        model = ORT_SUPPORTED_TASKS[task]["class"][0].auto_model_class.from_pretrained(model_name)
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
                save_onnx_model=False,
            )
            self.check_model_outputs(
                q_model=quantizer._quantized_model,
                task=task,
                tokenizer=tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
                load_onnx_model=False,
                num_samples=num_samples,
            )

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
        quantizer = INCQuantizer.from_pretrained(model, eval_fn=eval_fn)

        with tempfile.TemporaryDirectory() as tmp_dir:
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_aware_training_quantization(self, task, model_name, expected_quantized_matmuls):
        quantization_config = QuantizationAwareTrainingConfig()
        save_onnx_model = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                q_config=quantization_config,
                save_onnx_model=save_onnx_model,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
                load_onnx_model=save_onnx_model,
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_aware_training_quantization_pruning(self, task, model_name, expected_quantized_matmuls):
        quantization_config = QuantizationAwareTrainingConfig()
        target_sparsity = 0.9
        pruning_config = WeightPruningConfig(
            pruning_type="magnitude",
            start_step=0,
            end_step=15,
            target_sparsity=target_sparsity,
            pruning_scope="local",
        )
        save_onnx_model = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                q_config=quantization_config,
                p_config=pruning_config,
                save_onnx_model=save_onnx_model,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
                load_onnx_model=save_onnx_model,
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_magnitude_pruning(self, task, model_name, expected_quantized_matmuls):
        target_sparsity = 0.9
        # end_step should be training_args.num_train_epochs * (len(train_dataset) // training_args.per_device_train_batch_size)
        pruning_config = WeightPruningConfig(
            pruning_type="magnitude",
            start_step=0,
            end_step=1,
            target_sparsity=target_sparsity,
            pruning_scope="local",
        )
        save_onnx_model = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                p_config=pruning_config,
                save_onnx_model=save_onnx_model,
                num_train_samples=64,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=0,
                is_static=True,
                load_onnx_model=save_onnx_model,
            )
            sparsity = trainer.get_model_sparsity()
            inc_config = INCConfig.from_pretrained(tmp_dir)
            # Factor modified from 2 to 4 for tiny random model compatibility
            self.assertGreaterEqual(sparsity, target_sparsity * 100 / 4)
            self.assertEqual(inc_config.pruning["sparsity"], round(sparsity, 2))
            self.assertEqual(inc_config.pruning["approach"], "magnitude")
            self.assertEqual(inc_config.pruning["pattern"], "4x1")

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_distillation(self, task, model_name, expected_quantized_matmuls):
        teacher_model = ORT_SUPPORTED_TASKS[task]["class"][0].auto_model_class.from_pretrained(model_name)
        distillation_config = DistillationConfig(teacher_model=teacher_model)
        save_onnx_model = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                d_config=distillation_config,
                save_onnx_model=save_onnx_model,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=0,
                is_static=True,
                load_onnx_model=save_onnx_model,
            )
            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertEqual(inc_config.distillation["teacher_model_name_or_path"], model_name)
            self.assertEqual(inc_config.distillation["temperature"], 1.0)

    def check_model_outputs(
        self,
        q_model,
        task,
        tokenizer,
        save_directory,
        expected_quantized_matmuls,
        is_static=True,
        load_onnx_model=True,
        num_samples=None,
        file_name=ONNX_WEIGHTS_NAME,
    ):
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        inc_model = eval(_HEAD_TO_AUTOMODELS[task]).from_pretrained(save_directory)
        model_kwargs = (
            {"decoder_file_name": file_name, "use_cache": False}
            if task == "text-generation"
            else {"file_name": file_name}
        )
        inc_config = INCConfig.from_pretrained(save_directory)
        self.assertEqual(inc_config.save_onnx_model, load_onnx_model)

        if num_samples is not None:
            self.assertEqual(inc_config.quantization["dataset_num_samples"], num_samples)

        if load_onnx_model:
            onnx_model = onnx_load(os.path.join(save_directory, file_name))
            num_quantized_matmul = num_quantized_matmul_onnx_model(onnx_model)

            if num_quantized_matmul > 0:
                self.assertEqual(inc_config.quantization["is_static"], is_static)

            self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)
            ort_model = ORT_SUPPORTED_TASKS[task]["class"][0].from_pretrained(save_directory, **model_kwargs)
            ort_outputs = ort_model(**tokens)
            self.assertTrue("logits" in ort_outputs)

        with torch.no_grad():
            model_outputs = q_model(**tokens)
            inc_model_outputs = inc_model(**tokens)
        self.assertTrue(torch.equal(model_outputs["logits"], inc_model_outputs["logits"]))
        # self.assertTrue(torch.allclose(ort_outputs.logits, inc_model_outputs.logits, atol=1e-4))

    @staticmethod
    def get_trainer(
        model_name,
        task,
        save_directory,
        q_config=None,
        p_config=None,
        d_config=None,
        save_onnx_model=True,
        num_train_samples=8,
        num_eval_samples=8,
    ):
        model = ORT_SUPPORTED_TASKS[task]["class"][0].auto_model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        metric = evaluate.load("accuracy")
        dataset_name, dataset_config_name, column_name = _TASK_TO_DATASET[task]
        dataset = load_dataset(dataset_name, dataset_config_name)
        dataset = dataset.map(
            partial(_preprocess_function, tokenizer=tokenizer, column_name=column_name), batched=True
        )

        trainer = INCTrainer(
            model=model,
            quantization_config=q_config,
            pruning_config=p_config,
            distillation_config=d_config,
            task=task,
            args=TrainingArguments(save_directory, num_train_epochs=2.0, do_train=True, do_eval=True),
            train_dataset=dataset["train"].select(range(num_train_samples)),
            eval_dataset=dataset["validation"].select(range(num_eval_samples)),
            compute_metrics=partial(_compute_metrics, metric=metric),
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
        trainer.train()
        trainer.evaluate()
        trainer.save_model(save_onnx_model=save_onnx_model)
        trainer.model.eval()
        return trainer
