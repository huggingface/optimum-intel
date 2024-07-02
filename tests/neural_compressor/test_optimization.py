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

# ruff: noqa


import os
import tempfile
import copy

import unittest
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
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainingArguments,
    pipeline,
    set_seed,
)
from utils_tests import MODEL_NAMES, SEED, INCTestMixin, _generate_dataset
from optimum.intel.utils.import_utils import is_torch_version, is_itrex_available

from optimum.intel import (
    INCConfig,
    INCModelForCausalLM,
    INCModelForSeq2SeqLM,
    INCModelForQuestionAnswering,
    INCModelForSequenceClassification,
    INCModelForMaskedLM,
    INCModelForTokenClassification,
    INCQuantizer,
    INCSeq2SeqTrainer,
    INCStableDiffusionPipeline,
)
from optimum.intel.utils.constant import DIFFUSION_WEIGHTS_NAME
from optimum.exporters import TasksManager


os.environ["CUDA_VISIBLE_DEVICES"] = ""
set_seed(SEED)


class QuantizationTest(INCTestMixin):
    SUPPORTED_ARCHITECTURES_STATIC = (
        ("text-generation", "gpt_neo", 17),
        ("text-classification", "bert", 21),
        ("text-generation", "bloom", 21),
    )

    SUPPORTED_ARCHITECTURES_DYNAMIC = SUPPORTED_ARCHITECTURES_STATIC + (
        ("fill-mask", "bert", 22),
        ("token-classification", "albert", 26),
    )

    TEXT_GENERATION_SUPPORTED_ARCHITECTURES = (
        "bloom",
        "gpt_neo",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_DYNAMIC)
    def test_dynamic_quantization(self, task, model_arch, expected_quantized_matmuls):
        model_name = MODEL_NAMES[model_arch]
        model_class = TasksManager.get_model_class_for_task(task)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization_config = PostTrainingQuantConfig(approach="dynamic")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_class.from_pretrained(model_name)
            quantizer = INCQuantizer.from_pretrained(model, task=task)
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=tmp_dir,
            )
            quantized_model = quantizer._quantized_model

            self.check_model_outputs(
                q_model=quantized_model,
                task=task,
                tokenizer=tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=False,
                load_inc_model=True,
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_STATIC)
    def test_static_quantization(self, task, model_arch, expected_quantized_matmuls):
        num_samples = 10
        model_name = MODEL_NAMES[model_arch]
        model_class = TasksManager.get_model_class_for_task(task)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantized_model = None
        quantization_config = PostTrainingQuantConfig(approach="static")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_class.from_pretrained(model_name)
            quantizer = INCQuantizer.from_pretrained(model, task=task)
            calibration_dataset = _generate_dataset(quantizer, tokenizer, num_samples=num_samples)
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
            )
            quantized_model = quantizer._quantized_model

            self.check_model_outputs(
                q_model=quantized_model,
                task=task,
                tokenizer=tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
                load_inc_model=True,
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
            )
            loaded_model = INCModelForQuestionAnswering.from_pretrained(tmp_dir)
            inc_config = INCConfig.from_pretrained(tmp_dir)
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

    @parameterized.expand(TEXT_GENERATION_SUPPORTED_ARCHITECTURES)
    def test_quantize_text_generate_model(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample", return_tensors="pt")

        def calibration_fn(p_model):
            tmp_model = INCModelForCausalLM(p_model, model.config)
            tmp_model.generate(**tokens, max_new_tokens=32, do_sample=False)

        quantization_config = PostTrainingQuantConfig(approach="static")
        model.config.return_dict = False
        quantizer = INCQuantizer.from_pretrained(model, calibration_fn=calibration_fn)
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                save_directory=tmp_dir,
            )
            model = INCModelForCausalLM.from_pretrained(tmp_dir)

        pre_outputs = quantizer._quantized_model.generate(
            **tokens, do_sample=False, num_beams=1, temperature=0.9, min_length=20, max_length=20
        )
        outputs = model.generate(**tokens, do_sample=False, num_beams=1, temperature=0.9, min_length=20, max_length=20)
        self.assertTrue(torch.equal(pre_outputs, outputs))


class TrainingOptimizationTest(INCTestMixin):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (("text-classification", "bert", 21),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_aware_training_quantization(self, task, model_arch, expected_quantized_matmuls):
        model_name = MODEL_NAMES[model_arch]
        quantization_config = QuantizationAwareTrainingConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                q_config=quantization_config,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_aware_training_quantization_pruning(self, task, model_arch, expected_quantized_matmuls):
        model_name = MODEL_NAMES[model_arch]
        quantization_config = QuantizationAwareTrainingConfig()
        target_sparsity = 0.9
        pruning_config = WeightPruningConfig(
            pruning_type="magnitude",
            start_step=0,
            end_step=15,
            target_sparsity=target_sparsity,
            pruning_scope="local",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                q_config=quantization_config,
                p_config=pruning_config,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=expected_quantized_matmuls,
                is_static=True,
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_magnitude_pruning(self, task, model_arch, expected_quantized_matmuls):
        model_name = MODEL_NAMES[model_arch]
        target_sparsity = 0.9
        # end_step should be training_args.num_train_epochs * (len(train_dataset) // training_args.per_device_train_batch_size)
        pruning_config = WeightPruningConfig(
            pruning_type="magnitude",
            start_step=0,
            end_step=1,
            target_sparsity=target_sparsity,
            pruning_scope="local",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                p_config=pruning_config,
                num_train_samples=64,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=0,
                is_static=True,
            )
            sparsity = trainer.get_model_sparsity()
            inc_config = INCConfig.from_pretrained(tmp_dir)
            # Factor modified from 2 to 4 for tiny random model compatibility
            self.assertGreaterEqual(sparsity, target_sparsity * 100 / 4)
            self.assertEqual(inc_config.pruning["sparsity"], round(sparsity, 2))
            self.assertEqual(inc_config.pruning["approach"], "magnitude")
            self.assertEqual(inc_config.pruning["pattern"], "4x1")

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_distillation(self, task, model_arch, expected_quantized_matmuls):
        model_name = MODEL_NAMES[model_arch]
        teacher_model = TasksManager.get_model_class_for_task(task).from_pretrained(model_name)
        distillation_config = DistillationConfig(teacher_model=teacher_model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self.get_trainer(
                model_name=model_name,
                task=task,
                save_directory=tmp_dir,
                d_config=distillation_config,
            )
            self.check_model_outputs(
                q_model=trainer.model,
                task=task,
                tokenizer=trainer.tokenizer,
                save_directory=tmp_dir,
                expected_quantized_matmuls=0,
                is_static=True,
            )
            inc_config = INCConfig.from_pretrained(tmp_dir)
            self.assertEqual(inc_config.distillation["teacher_model_name_or_path"], model_name)
            self.assertEqual(inc_config.distillation["temperature"], 1.0)

    def test_seq2seq_aware_training_quantization(self):
        quantization_config = QuantizationAwareTrainingConfig()
        batch_size = 2
        train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
        val_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")
        train_dataset = train_dataset.select(range(4))
        val_dataset = val_dataset.select(range(4))

        model = EncoderDecoderModel.from_encoder_decoder_pretrained("prajjwal1/bert-tiny", "prajjwal1/bert-tiny")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.max_length = 128
        columns = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]

        def _map_to_encoder_decoder_inputs(batch):
            inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512)
            outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask

            batch["decoder_input_ids"] = outputs.input_ids
            batch["labels"] = outputs.input_ids.copy()
            batch["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
            ]
            batch["decoder_attention_mask"] = outputs.attention_mask
            return batch

        def _compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            accuracy = sum([int(pred_str[i] == label_str[i]) for i in range(len(pred_str))]) / len(pred_str)
            return {"accuracy": accuracy}

        train_dataset = train_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        train_dataset.set_format(type="torch", columns=columns)

        val_dataset = val_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        val_dataset.set_format(type="torch", columns=columns)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = Seq2SeqTrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                predict_with_generate=True,
                evaluation_strategy="steps",
                do_train=True,
                do_eval=True,
                warmup_steps=0,
                eval_steps=1,
                logging_steps=1,
                num_train_epochs=1.0,
            )

            trainer = INCSeq2SeqTrainer(
                model=model,
                quantization_config=quantization_config,
                args=training_args,
                compute_metrics=_compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
            )

            trainer.train()
            trainer.evaluate()
            trainer.save_model()
            trainer.model.eval()
            loaded_model = INCModelForSeq2SeqLM.from_pretrained(tmp_dir)
            tokens = tokenizer("This is a sample input", return_tensors="pt")
            decoder_inputs = {
                "decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * model.config.decoder_start_token_id
            }

            with torch.no_grad():
                model_outputs = trainer.model(**tokens, **decoder_inputs)
                loaded_model_outputs = loaded_model(**tokens, **decoder_inputs)

            self.assertTrue("logits" in loaded_model_outputs)
            self.assertIsInstance(loaded_model_outputs.logits, torch.Tensor)
            # Compare tensor outputs
            # self.assertTrue(torch.allclose(loaded_model_outputs.logits, model_outputs.logits, atol=1e-4))


class WeightOnlyQuantizationTest(INCTestMixin):
    WEIGHT_ONLY_CONFIG = (
        ("rtn", "int4_clip"),
        ("rtn", "int8"),
        ("gptq", "int4_clip"),
    )

    @parameterized.expand(WEIGHT_ONLY_CONFIG)
    @unittest.skipIf(not is_itrex_available(), reason="ITREX not available")
    def test_weight_only_quantization(self, methodology, weight_dtype):
        model_name = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"

        from intel_extension_for_transformers.transformers.utils.config import GPTQConfig, RtnConfig

        bits = 4 if "4" in weight_dtype else 8
        if methodology == "gptq":
            # max_input_length can be removed after neural-compressor > v2.5.1
            quantization_config = GPTQConfig(
                bits=bits, sym=True, damp_percent=0.01, weight_dtype=weight_dtype, max_input_length=128
            )
        else:
            quantization_config = RtnConfig(bits=bits, weight_dtype=weight_dtype)

        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        quantizer = INCQuantizer.from_pretrained(copy.deepcopy(model), task="text-generation")
        calibration_dataset = _generate_dataset(quantizer, tokenizer, num_samples=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantizer.quantize(
                quantization_config=quantization_config,
                calibration_dataset=calibration_dataset,
                save_directory=tmp_dir,
            )
            loaded_model = INCModelForCausalLM.from_pretrained(tmp_dir)

        tokens = tokenizer("This is a sample output", return_tensors="pt")

        with torch.no_grad():
            loaded_outputs = loaded_model(**tokens)
            # quantizer_outputs = model(**tokens)

        self.assertTrue("logits" in loaded_outputs)
        self.assertIsInstance(loaded_outputs.logits, torch.Tensor)
        self.assertTrue("past_key_values" in loaded_outputs)
        self.assertIsInstance(loaded_outputs.past_key_values, tuple)

        # self.assertTrue(torch.allclose(quantizer_outputs.logits, loaded_outputs.logits, equal_nan=True, atol=1e-4))
