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


from optimum.intel.utils.import_utils import is_ipex_available

from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS
from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS
from optimum.exporters import TasksManager

if is_ipex_available():
    from optimum.intel import (
        IPEXModelForCausalLM,
        IPEXModelForSequenceClassification,
        IPEXModelForMaskedLM,
        IPEXModelForTokenClassification,
    )


SEED = 1009
_TASK_TO_DATASET = {
    "text-classification": ("glue", "sst2", "sentence"),
    "text-generation": ("wikitext", "wikitext-2-raw-v1", "text"),
    "text2text-generation": ("cnn_dailymail", "3.0.0", ("article", "highlights")),
}


MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "electra": "hf-internal-testing/tiny-random-electra",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "llama2": "Jiqing/tiny_random_llama2",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mistral": "echarlaix/tiny-random-mistral",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "stas/mt5-tiny-random",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "phi": "echarlaix/tiny-random-PhiForCausalLM",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "t5": "hf-internal-testing/tiny-random-t5",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "vit": "hf-internal-testing/tiny-random-vit",
    "wav2vec2": "anton-l/wav2vec2-random-tiny-classifier",
    "xlm": "hf-internal-testing/tiny-random-xlm",
}


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
    model_type = quantizer._original_model.config.model_type.replace("_", "-")
    if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS:
        dataset = dataset.map(
            lambda x: {
                "position_ids": np.arange(len(x["input_ids"])),
            }
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
        load_inc_model=True,
        num_samples=None,
        load_ipex_model=False,
    ):
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        inc_config = INCConfig.from_pretrained(save_directory)

        if num_samples is not None:
            self.assertEqual(inc_config.quantization["dataset_num_samples"], num_samples)

        with torch.no_grad():
            model_outputs = q_model(**tokens)
            outputs = model_outputs["logits"] if isinstance(model_outputs, dict) else model_outputs[0]
            auto_class = _HEAD_TO_AUTOMODELS[task]
            if load_ipex_model:
                auto_class = auto_class.replace("INC", "IPEX")
            if load_inc_model or load_ipex_model:
                inc_model = eval(auto_class).from_pretrained(save_directory)
                inc_model_outputs = inc_model(**tokens)
                self.assertTrue(torch.allclose(inc_model_outputs["logits"], outputs, atol=1e-2))

    @staticmethod
    def get_trainer(
        model_name,
        task,
        save_directory,
        q_config=None,
        p_config=None,
        d_config=None,
        num_train_samples=8,
        num_eval_samples=8,
    ):
        model = TasksManager.get_model_class_for_task(task).from_pretrained(model_name)
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
        trainer.save_model()
        trainer.model.eval()
        return trainer
