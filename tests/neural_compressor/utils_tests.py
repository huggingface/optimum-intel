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
import unittest
from functools import partial

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from onnx import load as onnx_load
from transformers import AutoTokenizer, TrainingArguments, default_data_collator

from optimum.intel import (
    INCConfig,
    INCModelForCausalLM,
    INCModelForSeq2SeqLM,
    INCModelForQuestionAnswering,
    INCModelForSequenceClassification,
    INCModelForMaskedLM,
    INCModelForTokenClassification,
    INCQuantizer,
    INCTrainer,
    INCSeq2SeqTrainer,
    INCStableDiffusionPipeline,
)
from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS
from optimum.intel.utils.constant import ONNX_WEIGHTS_NAME
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification
from optimum.pipelines import ORT_SUPPORTED_TASKS

SEED = 1009
_TASK_TO_DATASET = {
    "text-classification": ("glue", "sst2", "sentence"),
    "text-generation": ("wikitext", "wikitext-2-raw-v1", "text"),
    "text2text-generation": ("cnn_dailymail", "3.0.0", ("article", "highlights")),
}


def num_quantized_matmul_onnx_model(onnx_model):
    num_quantized_matmul = 0
    for node in onnx_model.graph.node:
        if "QuantizeLinear" in node.name:
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


class INCTestMixin(unittest.TestCase):
    def check_model_outputs(
        self,
        q_model,
        task,
        tokenizer,
        save_directory,
        expected_quantized_matmuls,
        is_static=True,
        load_onnx_model=True,
        load_inc_model=True,
        num_samples=None,
        file_name=None,
    ):
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        file_name = ONNX_WEIGHTS_NAME if task != "text-generation" else "decoder_model.onnx"

        model_kwargs = (
            {"decoder_file_name": file_name, "use_cache": False, "use_io_binding": False}
            if task == "text-generation"
            else {"file_name": file_name}
        )
        inc_config = INCConfig.from_pretrained(save_directory)

        if num_samples is not None:
            self.assertEqual(inc_config.quantization["dataset_num_samples"], num_samples)

        with torch.no_grad():
            model_outputs = q_model(**tokens)
            outputs = model_outputs["logits"] if isinstance(model_outputs, dict) else model_outputs[0]
            if load_inc_model:
                inc_model = eval(_HEAD_TO_AUTOMODELS[task]).from_pretrained(save_directory)
                inc_model_outputs = inc_model(**tokens)
                self.assertTrue(torch.allclose(inc_model_outputs["logits"], outputs, atol=1e-2))
                # self.assertEqual(inc_config.save_onnx_model, load_onnx_model)

        if load_onnx_model:
            onnx_model = onnx_load(os.path.join(save_directory, file_name))
            num_quantized_matmul = num_quantized_matmul_onnx_model(onnx_model)

            if num_quantized_matmul > 0:
                self.assertEqual(inc_config.quantization["is_static"], is_static)

            self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)
            ort_model = ORT_SUPPORTED_TASKS[task]["class"][0].from_pretrained(save_directory, **model_kwargs)
            ort_outputs = ort_model(**tokens)
            self.assertTrue("logits" in ort_outputs)
            if task != "fill-mask":
                self.assertTrue(torch.allclose(ort_outputs.logits, outputs, atol=1e-2))

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
