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
import dataclasses
import inspect

# ruff: noqa

import itertools
import logging
import unittest
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from functools import partial
from typing import Union, Type

import openvino as ov
import pytest
import numpy as np
import torch
from parameterized import parameterized
import nncf
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoProcessor,
)
from transformers.testing_utils import slow
from transformers.utils.quantization_config import QuantizationMethod

from optimum.intel import (
    OVConfig,
    OVFluxPipeline,
    OVLatentConsistencyModelPipeline,
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
    OVModelForSpeechSeq2Seq,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLPipeline,
    OVStableDiffusion3Pipeline,
    OVQuantizer,
    OVSanaPipeline,
    OVQuantizationConfig,
    OVMixedQuantizationConfig,
    OVWeightQuantizationConfig,
    OVDynamicQuantizationConfig,
    OVModelOpenCLIPForZeroShotImageClassification,
    OVModelForVisualCausalLM,
    OVSentenceTransformer,
    OVModelForZeroShotImageClassification,
)
from optimum.intel.openvino.configuration import (
    OVQuantizationMethod,
    OVQuantizationConfigBase,
    _DEFAULT_4BIT_WQ_CONFIGS,
    _DEFAULT_4BIT_WQ_CONFIG,
)
from optimum.intel.openvino.utils import TemporaryDirectory
from copy import deepcopy

from optimum.intel.openvino.quantization import InferRequestWrapper, OVCalibrationDatasetBuilder
from optimum.intel.utils.import_utils import is_openvino_version, is_transformers_version
from utils_tests import (
    MODEL_NAMES,
    get_num_quantized_nodes,
    _ARCHITECTURES_TO_EXPECTED_INT8,
    check_compression_state_per_model,
)

_TASK_TO_DATASET = {
    "text-generation": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "column_name": "text",
    },
    "feature-extraction": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "column_name": "text",
    },
    "fill-mask": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "column_name": "text",
    },
    "text-classification": {
        "dataset_name": "glue",
        "dataset_config_name": "sst2",
        "column_name": "sentence",
    },
    "zero-shot-image-classification": {
        "dataset_name": "conceptual_captions",
        "column_name": "caption",
        "streaming": True,
    },
}

pattern_prefix = (
    "^__module.model.model"
    if is_transformers_version(">=", "4.49") and is_transformers_version("<", "4.51")
    else "^__module.model"
)


class OVQuantizerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_TORCH_MODEL = (
        (OVModelForSequenceClassification, "bert", 32, 35),
        (OVModelForCausalLM, "gpt2", 41 if is_transformers_version("<", "4.42.0") else 31, 22),
    )
    # TODO (nikita-savelyevv): Extend for OVModelForSpeechSeq2Seq and OVStableDiffusionPipeline
    SUPPORTED_ARCHITECTURES_OV_MODEL = (
        (OVModelForSequenceClassification, "bert", 32, 35),
        (OVModelForCausalLM, "gpt2", 31, 22),
        (OVSentenceTransformer, "sentence-transformers-bert", 12, 15),
        (OVModelForFeatureExtraction, "blenderbot", 33, 35),
        (OVModelForMaskedLM, "roberta", 32, 34),
        (OVModelForZeroShotImageClassification, "clip", 65, 65),
    )
    SUPPORTED_ARCHITECTURES_OV_MODEL_WITH_AUTO_DATASET = [
        (
            OVModelForSpeechSeq2Seq,
            "whisper",
            dict(
                dataset="librispeech",
                num_samples=1,
                processor=MODEL_NAMES["whisper"],
                trust_remote_code=True,
                smooth_quant_alpha=0.95,
            ),
            {"encoder": 10, "decoder": 12, "decoder_with_past": 11}
            if is_transformers_version("<=", "4.36.0")
            else {"encoder": 8, "decoder": 12, "decoder_with_past": 25},
            (
                {"encoder": {"int8": 8}, "decoder": {"int8": 11}, "decoder_with_past": {"int8": 9}}
                if is_transformers_version("<=", "4.36.0")
                else {"encoder": {"int8": 8}, "decoder": {"int8": 12}, "decoder_with_past": {"int8": 18}}
            ),
        ),
        (
            OVModelForCausalLM,
            "llama",
            dict(
                dataset="wikitext2",
                num_samples=1,
                dtype="f8e4m3",
                weight_only=False,
            ),
            {
                "model": 13,
            },
            {
                "model": {"f8e4m3": 16},
            },
        ),
        (
            OVModelForCausalLM,
            "llama",
            dict(
                weight_quantization_config=dict(bits=4, dtype="nf4", group_size=16, weight_only=True, ratio=0.5),
                full_quantization_config=dict(dtype="f8e4m3", weight_only=False),
                dataset="wikitext2",
                num_samples=1,
            ),
            {
                "model": 14,
            },
            {
                "model": {"f8e4m3": 11, "nf4": 5},
            },
        ),
        (
            OVModelForCausalLM,
            "llama",
            OVMixedQuantizationConfig(
                weight_quantization_config=OVWeightQuantizationConfig(
                    bits=4,
                    dtype="nf4",
                    group_size=16,
                    ratio=0.5,
                    ignored_scope={"patterns": [f"{pattern_prefix}.layers.0.self_attn"]},
                ),
                full_quantization_config=OVQuantizationConfig(
                    dtype="f8e4m3", ignored_scope={"patterns": [f"{pattern_prefix}.layers.0.mlp"]}
                ),
                ignored_scope={"patterns": [f"{pattern_prefix}.layers.1.self_attn"]},
                dataset="wikitext2",
                num_samples=1,
            ),
            {
                "model": 7,
            },
            {
                "model": {"f8e4m3": 8, "nf4": 2},
            },
        ),
        (
            OVModelForCausalLM,
            "llama",
            OVMixedQuantizationConfig(
                weight_quantization_config=OVWeightQuantizationConfig(
                    bits=4,
                    dtype="nf4",
                    group_size=16,
                    ratio=0.5,
                    ignored_scope={"patterns": [f"{pattern_prefix}.layers.0.self_attn"]},
                ),
                full_quantization_config=OVQuantizationConfig(
                    dtype="f8e5m2", ignored_scope={"patterns": [f"{pattern_prefix}.layers.0.mlp"]}
                ),
                ignored_scope={"patterns": [f"{pattern_prefix}.layers.1.self_attn"]},
                dataset="wikitext2",
                num_samples=1,
            ),
            {
                "model": 7,
            },
            {
                "model": {"f8e5m2": 8, "nf4": 2},
            },
        ),
        (
            OVModelForCausalLM,
            "llama",
            OVMixedQuantizationConfig(
                weight_quantization_config=OVWeightQuantizationConfig(bits=4, group_size=16, ratio=0.5),
                full_quantization_config=OVQuantizationConfig(dtype="f8e4m3"),
                dataset="wikitext2",
                num_samples=1,
            ),
            {
                "model": 14,
            },
            {
                "model": {"f8e4m3": 11, "int4": 10},
            },
        ),
        (
            OVModelForCausalLM,
            "llama",
            OVMixedQuantizationConfig(
                weight_quantization_config=OVWeightQuantizationConfig(bits=4, group_size=16),
                full_quantization_config=OVQuantizationConfig(dtype="f8e5m2"),
                dataset="wikitext2",
                num_samples=1,
            ),
            {
                "model": 13,
            },
            {
                "model": {"f8e5m2": 2, "int4": 28},
            },
        ),
        (
            OVStableDiffusionPipeline,
            "stable-diffusion",
            dict(
                weight_only=False,
                dataset="conceptual_captions",
                num_samples=1,
                processor=MODEL_NAMES["stable-diffusion"],
                trust_remote_code=True,
            ),
            {
                "unet": 112,
                "vae_decoder": 0,
                "vae_encoder": 0,
                "text_encoder": 0,
            },
            {
                "unet": {"int8": 121},
                "vae_decoder": {"int8": 42},
                "vae_encoder": {"int8": 34},
                "text_encoder": {"int8": 64},
            },
        ),
        (
            OVStableDiffusionXLPipeline,
            "stable-diffusion-xl",
            dict(
                weight_only=False,
                dtype="f8e5m2",
                dataset="laion/220k-GPT4Vision-captions-from-LIVIS",
                num_samples=1,
                processor=MODEL_NAMES["stable-diffusion-xl"],
                trust_remote_code=True,
            ),
            {
                "unet": 174,
                "vae_decoder": 0,
                "vae_encoder": 0,
                "text_encoder": 0,
                "text_encoder_2": 0,
            },
            {
                "unet": {"f8e5m2": 183},
                "vae_decoder": {"int8": 42},
                "vae_encoder": {"int8": 34},
                "text_encoder": {"int8": 64},
                "text_encoder_2": {"int8": 66},
            },
        ),
        (
            OVLatentConsistencyModelPipeline,
            "latent-consistency",
            OVQuantizationConfig(
                dtype="f8e4m3",
                dataset="laion/filtered-wit",
                num_samples=1,
                trust_remote_code=True,
            ),
            {
                "unet": 79,
                "vae_decoder": 0,
                "vae_encoder": 0,
                "text_encoder": 0,
            },
            {
                "unet": {"f8e4m3": 84},
                "vae_decoder": {"int8": 42},
                "vae_encoder": {"int8": 34},
                "text_encoder": {"int8": 40},
            },
        ),
        (
            OVModelForFeatureExtraction,
            "blenderbot",
            OVQuantizationConfig(
                dtype="int8",
                dataset="wikitext2",
                num_samples=1,
            ),
            {
                "model": 33,
            },
            {
                "model": {"int8": 35},
            },
        ),
        (
            OVSentenceTransformer,
            "sentence-transformers-bert",
            OVQuantizationConfig(
                dtype="int8",
                dataset="c4",
                num_samples=1,
            ),
            {
                "model": 12,
            },
            {
                "model": {"int8": 15},
            },
        ),
        (
            OVModelForMaskedLM,
            "roberta",
            OVQuantizationConfig(
                dtype="int8",
                dataset="wikitext2",
                num_samples=1,
            ),
            {
                "model": 32,
            },
            {
                "model": {"int8": 34},
            },
        ),
        (
            OVModelForMaskedLM,
            "xlm_roberta",
            OVQuantizationConfig(
                dtype="int8",
                dataset="c4",
                num_samples=1,
            ),
            {
                "model": 14,
            },
            {
                "model": {"int8": 16},
            },
        ),
        (
            OVModelForZeroShotImageClassification,
            "clip",
            OVQuantizationConfig(
                dtype="int8",
                dataset="conceptual_captions",
                num_samples=1,
            ),
            {
                "model": 65,
            },
            {
                "model": {"int8": 65},
            },
        ),
    ]

    @staticmethod
    def get_calibration_dataset(
        quantizer,
        quantization_config,
        preprocess_function,
        as_dataset_instance,
        dataset_name,
        dataset_config_name=None,
        streaming=False,
    ):
        if as_dataset_instance:
            calibration_dataset = quantizer.get_calibration_dataset(
                dataset_name,
                dataset_config_name=dataset_config_name,
                preprocess_function=preprocess_function,
                num_samples=2,
                dataset_split="train",
                trust_remote_code=True,
                streaming=streaming,
            )
        else:
            dataset_builder = OVCalibrationDatasetBuilder(quantizer.model)
            calibration_dataset = dataset_builder.build_from_dataset_name(
                quantization_config,
                dataset_name,
                dataset_config_name=dataset_config_name,
                preprocess_function=preprocess_function,
                num_samples=2,
                dataset_split="train",
                trust_remote_code=True,
                streaming=streaming,
            )
        return calibration_dataset

    @parameterized.expand(
        [(*it[0], it[1]) for it in itertools.product(SUPPORTED_ARCHITECTURES_TORCH_MODEL, [False, True])]
    )
    def test_automodel_static_quantization(
        self, model_cls, model_name, expected_fake_nodes, expected_int8_nodes, from_dataset_instance
    ):
        model_id = MODEL_NAMES[model_name]
        task = model_cls.export_feature
        dataset_kwargs = {**_TASK_TO_DATASET[task]}
        column_name = dataset_kwargs.pop("column_name")
        file_name = "openvino_quantized_model.xml"

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples[column_name], padding="max_length", max_length=128, truncation=True)

        with TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.auto_model_class.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task)

            ov_config = OVConfig(quantization_config=OVQuantizationConfig())
            calibration_dataset = self.get_calibration_dataset(
                quantizer,
                ov_config.quantization_config,
                partial(preprocess_function, tokenizer=tokenizer),
                from_dataset_instance,
                **dataset_kwargs,
            )
            quantizer.quantize(
                save_directory=tmp_dir,
                calibration_dataset=calibration_dataset,
                file_name=file_name,
                ov_config=ov_config,
            )
            model = model_cls.from_pretrained(tmp_dir, file_name=file_name)
            num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_fake_nodes, num_fake_nodes)
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())

    @parameterized.expand(
        [(*it[0], it[1]) for it in itertools.product(SUPPORTED_ARCHITECTURES_OV_MODEL, [False, True])]
    )
    def test_ovmodel_static_quantization(
        self, model_cls, model_name, expected_fake_nodes, expected_int8_nodes, from_dataset_instance
    ):
        model_id = MODEL_NAMES[model_name]
        task = model_cls.export_feature
        dataset_kwargs = {**_TASK_TO_DATASET[task]}
        column_name = dataset_kwargs.pop("column_name")

        with TemporaryDirectory() as tmp_dir:
            ov_model = model_cls.from_pretrained(model_id, export=True)

            is_text_related_task = model_cls in (
                OVModelForSequenceClassification,
                OVModelForCausalLM,
                OVModelForFeatureExtraction,
                OVSentenceTransformer,
                OVModelForMaskedLM,
            )
            if is_text_related_task:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                def preprocess_function(examples):
                    inputs = tokenizer(
                        examples[column_name],
                        padding="max_length",
                        max_length=128,
                        truncation=True,
                        return_tensors="np",
                    )
                    if model_cls == OVModelForMaskedLM:
                        batch_size = inputs["input_ids"].shape[0]
                        random_indices = np.random.randint(0, inputs["input_ids"].shape[1], size=batch_size)
                        inputs["input_ids"][np.arange(batch_size), random_indices] = tokenizer.mask_token_id
                    return inputs

            elif model_cls == OVModelForZeroShotImageClassification:
                processor = AutoProcessor.from_pretrained(model_id)

                def preprocess_function(examples):
                    # Mock dataset data
                    n_examples = len(examples[column_name])
                    text = ["This is a sample text"] * n_examples
                    image = (np.random.rand(n_examples, 224, 224, 3) * 255).astype(np.uint8)
                    inputs = processor(
                        text=text,
                        images=image,
                        return_tensors="pt",
                        padding=True,
                    )
                    return inputs

            else:
                raise ValueError("Unsupported model class.")

            quantizer = OVQuantizer.from_pretrained(ov_model, task=task)

            ov_config = OVConfig(quantization_config=OVQuantizationConfig())
            calibration_dataset = self.get_calibration_dataset(
                quantizer, ov_config.quantization_config, preprocess_function, from_dataset_instance, **dataset_kwargs
            )
            quantizer.quantize(save_directory=tmp_dir, calibration_dataset=calibration_dataset, ov_config=ov_config)

            model = model_cls.from_pretrained(tmp_dir)

            num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_fake_nodes, num_fake_nodes)
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])

            if is_text_related_task:
                tokens = tokenizer("This is a sample input <mask>", return_tensors="pt")
                model(tokens) if model_cls == OVSentenceTransformer else model(**tokens)
            elif model_cls == OVModelForZeroShotImageClassification:
                inputs = preprocess_function({column_name: ["sample text"]})
                model(**inputs)
            else:
                raise ValueError("Unsupported model class.")

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())
            check_optimization_not_applicable_to_optimized_model(
                model, quantization_config=OVWeightQuantizationConfig(bits=8)
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_OV_MODEL_WITH_AUTO_DATASET)
    def test_ov_model_static_quantization_with_auto_dataset(
        self,
        model_cls,
        model_name,
        quantization_config,
        expected_fake_nodes_per_model,
        expected_num_weight_nodes_per_model,
    ):
        model_id = MODEL_NAMES[model_name]

        with TemporaryDirectory() as tmp_dir:
            ov_model = model_cls.from_pretrained(model_id, quantization_config=quantization_config)
            ov_model.save_pretrained(tmp_dir)

            if model_cls == OVModelForSpeechSeq2Seq:
                if ov_model.decoder_with_past is None:
                    del expected_fake_nodes_per_model["decoder_with_past"]
                    del expected_num_weight_nodes_per_model["decoder_with_past"]

                input_features = torch.randn((1, ov_model.config.num_mel_bins, 3000), dtype=torch.float32)
                ov_model.generate(input_features)
            elif model_cls in (OVModelForCausalLM, OVModelForFeatureExtraction, OVModelForMaskedLM):
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokens = tokenizer("This is a sample <mask>", return_tensors="pt")
                ov_model(**tokens)
            elif model_cls in (
                OVStableDiffusionPipeline,
                OVStableDiffusionXLPipeline,
                OVLatentConsistencyModelPipeline,
            ):
                ov_model(prompt="A text-to-image prompt")
            elif model_cls == OVSentenceTransformer:
                ov_model.encode(["This is a sample input"])
            elif model_cls == OVModelForZeroShotImageClassification:
                processor = AutoProcessor.from_pretrained(model_id)
                image = np.random.rand(224, 224, 3).astype(np.uint8)
                inputs = processor(text=["This is a sample text"], images=image, return_tensors="pt")
                ov_model(**inputs)
            else:
                raise Exception("Unexpected model class.")

            check_compression_state_per_model(
                self,
                ov_model.ov_submodels,
                expected_num_weight_nodes_per_model,
                expected_fake_nodes_per_model,
            )


class OVWeightCompressionTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS = (
        (OVModelForSequenceClassification, "bert", 70, 70),
        (OVModelForCausalLM, "gpt2", 44, 44),
    )

    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_COMPRESSED_MATMULS = ((OVModelForCausalLM, "opt125m", 62, 43),)
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_AUTOCOMPRESSED_MATMULS = ((OVModelForCausalLM, "opt125m", 0, 74),)
    SUPPORTED_ARCHITECTURES_STATEFUL_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS = ((OVModelForCausalLM, "gpt2", 44, 44),)

    LOAD_IN_4_BITS_SCOPE = [
        (
            OVModelForCausalLM,  # model cls
            "gpt2",  # model name
            False,  # trust remote code
            dict(bits=4, sym=False, group_size=-1, ratio=0.8),  # quantization config
            {"model": {"int8": 14, "int4": 30}},  # reference number of low-precision nodes
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(bits=4, dtype="mxfp4", group_size=32),
            {"model": {"int8": 4, "f4e2m1": 20, "f8e8m0": 20}},
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(bits=4, dtype="nf4", group_size=32),
            {"model": {"int8": 4, "nf4": 20}},
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(
                bits=4,
                sym=False,
                group_size=32,
                ignored_scope={"names": ["__module.model.transformer.h.2.mlp.c_fc/aten::addmm/MatMul"]},
            ),
            {"model": {"int8": 4, "int4": 38}},
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(bits=4, sym=False, group_size=-1, ratio=0.8, all_layers=True),
            {"model": {"int8": 18, "int4": 26}},
        ),
        (
            OVModelForCausalLM,
            "opt",
            False,
            dict(
                bits=4,
                sym=True,
                group_size=-1,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset="c4",
            ),
            {"model": {"int8": 18, "int4": 23}}
            if is_transformers_version(">=", "4.49")
            else {"model": {"int8": 14, "int4": 25}},
        ),
        (
            OVModelForCausalLM,
            "opt",
            False,
            dict(
                bits=4,
                sym=True,
                group_size=-1,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset=["one two, " * i for i in range(10)],
            ),
            {"model": {"int8": 18, "int4": 23}}
            if is_transformers_version(">=", "4.49")
            else {"model": {"int8": 16, "int4": 24}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(
                bits=4,
                sym=True,
                group_size=16,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset="c4",
                quant_method=QuantizationMethod.AWQ,
                scale_estimation=True,
            ),
            {"model": {"int8": 8, "int4": 12}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(
                bits=4,
                sym=True,
                group_size=16,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset="c4",
                quant_method="awq",
            ),
            {"model": {"int8": 8, "int4": 12}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(
                bits=4,
                sym=True,
                group_size=16,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset="c4",
                gptq=True,
            ),
            {"model": {"int8": 8, "int4": 12}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(
                bits=4,
                group_size=16,
                num_samples=16,
                dataset="auto",
                lora_correction=True,
            ),
            {"model": {"int8": 60, "int4": 28}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="none", group_size=16),
            {"model": {"int4": 28}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="none", group_size=16, ratio=0.5),
            {"model": {"int4": 6}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="int8_sym", group_size=16, ratio=0.5),
            {"model": {"int8": 13, "int4": 6}},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="int8_asym", group_size=16, ratio=0.5),
            {"model": {"int8": 26, "int4": 6}},
        ),
    ]

    if is_transformers_version(">=", "4.40.0"):
        LOAD_IN_4_BITS_SCOPE.extend(
            [
                (
                    OVModelForVisualCausalLM,
                    "llava_next",
                    False,
                    dict(
                        bits=4,
                        group_size=16,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="hessian_input_activation",
                        num_samples=1,
                        processor=MODEL_NAMES["llava_next"],
                    ),
                    {
                        "lm_model": {"int8": 6, "int4": 24},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 9},
                    },
                ),
                (
                    OVModelForVisualCausalLM,
                    "nanollava",
                    True,
                    dict(
                        bits=4,
                        group_size=8,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="mean_activation_variance",
                        num_samples=1,
                        processor=MODEL_NAMES["nanollava_vision_tower"],
                        tokenizer=MODEL_NAMES["nanollava"],
                        trust_remote_code=True,
                    ),
                    {
                        "lm_model": {"int8": 16, "int4": 14},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 15},
                    },
                ),
            ]
        )

    if is_transformers_version(">=", "4.42.0"):
        LOAD_IN_4_BITS_SCOPE.extend(
            [
                (
                    OVModelForVisualCausalLM,
                    "llava_next_video",
                    False,
                    dict(
                        bits=4,
                        group_size=16,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="hessian_input_activation",
                        num_samples=1,
                        processor=MODEL_NAMES["llava_next_video"],
                    ),
                    {
                        "lm_model": {"int8": 6, "int4": 24},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 7},
                        "vision_resampler_model": {},
                        "multi_modal_projector_model": {"int8": 2},
                    },
                ),
            ]
        )

    if is_transformers_version(">=", "4.45.0"):
        LOAD_IN_4_BITS_SCOPE.extend(
            [
                (
                    OVModelForVisualCausalLM,
                    "minicpmv",
                    True,
                    dict(
                        bits=4,
                        group_size=16,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="mean_activation_magnitude",
                        num_samples=1,
                        processor=MODEL_NAMES["minicpmv"],
                        trust_remote_code=True,
                    ),
                    {
                        "lm_model": {"int8": 8, "int4": 22},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 26},
                        "resampler_model": {"int8": 6},
                    },
                ),
                (
                    OVModelForVisualCausalLM,
                    "internvl2",
                    True,
                    dict(
                        bits=4,
                        group_size=4,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="mean_activation_magnitude",
                        num_samples=1,
                        trust_remote_code=True,
                    ),
                    {
                        "lm_model": {"int8": 8, "int4": 22},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 11},
                    },
                ),
                (
                    OVModelForVisualCausalLM,
                    "phi3_v",
                    True,
                    dict(
                        bits=4,
                        group_size=16,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="mean_activation_magnitude",
                        num_samples=1,
                        trust_remote_code=True,
                    ),
                    {
                        "lm_model": {"int8": 4, "int4": 14},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 7},
                        "vision_projection_model": {"int8": 2},
                    },
                ),
                (
                    OVModelForVisualCausalLM,
                    "qwen2_vl",
                    False,
                    dict(
                        bits=4,
                        group_size=16,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="mean_activation_magnitude",
                        num_samples=1,
                    ),
                    {
                        "lm_model": {"int8": 10, "int4": 20},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 1},
                        "vision_embeddings_merger_model": {"int8": 10},
                    },
                ),
            ]
        )

    if is_transformers_version(">=", "4.49.0"):
        LOAD_IN_4_BITS_SCOPE.extend(
            [
                (
                    OVModelForVisualCausalLM,
                    "phi4mm",
                    True,
                    dict(
                        bits=4,
                        group_size=8,
                        dataset="contextual",
                        ratio=0.8,
                        sensitivity_metric="mean_activation_magnitude",
                        num_samples=1,
                        trust_remote_code=True,
                    ),
                    {
                        "lm_model": {"int8": 8, "int4": 42},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 8},
                        "vision_projection_model": {"int8": 2},
                        "audio_embeddings_model": {},
                        "audio_forward_embeddings_model": {"int8": 6},
                        "audio_encoder_model": {"int8": 25},
                        "audio_vision_projection_model": {"int8": 2},
                        "audio_speech_projection_model": {"int8": 2},
                    },
                ),
            ]
        )

    SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION = [
        (OVModelForCausalLM, "gpt2", False),
        (OVModelForMaskedLM, "bert", False),
        (OVModelForTokenClassification, "roberta", False),
        (OVModelForImageClassification, "vit", False),
        (OVModelForSeq2SeqLM, "t5", False),
        (OVModelForSequenceClassification, "albert", False),
        (OVModelForQuestionAnswering, "distilbert", False),
        (OVModelForAudioClassification, "wav2vec2", False),
        (OVModelForFeatureExtraction, "blenderbot", False),
        (OVStableDiffusionPipeline, "stable-diffusion", False),
        (OVStableDiffusionXLPipeline, "stable-diffusion-xl", False),
        (OVModelOpenCLIPForZeroShotImageClassification, "open-clip", False),
        (OVModelForVisualCausalLM, "llava", False),
    ]

    if is_transformers_version(">=", "4.40.0"):
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForVisualCausalLM, "nanollava", True))

    if is_transformers_version(">=", "4.42.0"):
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForVisualCausalLM, "llava_next_video", False))

    if is_transformers_version(">=", "4.45.0"):
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForVisualCausalLM, "minicpmv", True))
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForVisualCausalLM, "qwen2_vl", False))

    SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION = [
        (OVStableDiffusionPipeline, "stable-diffusion", 72, 195),
        (OVStableDiffusionXLPipeline, "stable-diffusion-xl", 84, 331),
        (OVLatentConsistencyModelPipeline, "latent-consistency", 50, 135),
    ]

    if is_transformers_version(">=", "4.45.0"):
        SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION.extend(
            [
                (OVStableDiffusion3Pipeline, "stable-diffusion-3", 9, 65),
                (OVFluxPipeline, "flux", 7, 56),
                (OVSanaPipeline, "sana", 19, 53),
            ]
        )

    IS_SUPPORT_STATEFUL = is_openvino_version(">=", "2023.3")

    DEFAULT_INT4_CONFIG = {"bits": 4, "sym": True, "group_size": 64, "all_layers": True}

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS)
    def test_automodel_weight_compression(self, model_cls, model_name, expected_pt_int8, expected_ov_int8):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]

        with TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.auto_model_class.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task)
            quantizer.quantize(save_directory=tmp_dir)
            model = model_cls.from_pretrained(tmp_dir)

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_pt_int8, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            original_config_as_dict = OVWeightQuantizationConfig().to_dict()
            for k in original_config_as_dict.keys():
                v = original_config_as_dict[k]
                if isinstance(v, Enum):
                    original_config_as_dict[k] = v.value
            self.assertEqual(original_config_as_dict, loaded_config.quantization_config.to_dict())
            self.assertFalse(model.model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS)
    def test_ovmodel_8bit_weight_compression(self, model_cls, model_name, expected_pt_int8, expected_ov_int8):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]

        with TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.from_pretrained(model_id, export=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task)
            quantizer.quantize(save_directory=tmp_dir)
            model = model_cls.from_pretrained(tmp_dir)

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(OVWeightQuantizationConfig().to_dict(), loaded_config.quantization_config.to_dict())
            self.assertFalse(model.model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_COMPRESSED_MATMULS)
    def test_ovmodel_4bit_weight_compression(self, model_cls, model_name, expected_int8_nodes, expected_int4_nodes):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]
        with TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.from_pretrained(model_id, export=True, stateful=False)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task)
            ov_config = OVConfig(quantization_config=OVWeightQuantizationConfig(bits=4, sym=True, ratio=0.8))
            quantizer.quantize(save_directory=tmp_dir, ov_config=ov_config)
            model = model_cls.from_pretrained(tmp_dir)

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])
            self.assertEqual(expected_int4_nodes, num_weight_nodes["int4"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())
            self.assertFalse(model.model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_STATEFUL_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS)
    def test_ovmodel_8bit_weight_compression_stateful(self, model_cls, model_name, expected_pt_int8, expected_ov_int8):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]
        with TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.from_pretrained(model_id, export=True, stateful=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task)
            quantizer.quantize(save_directory=tmp_dir)
            model = model_cls.from_pretrained(tmp_dir)

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(OVWeightQuantizationConfig().to_dict(), loaded_config.quantization_config.to_dict())
            self.assertFalse(model.model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION)
    def test_ovmodel_load_with_compressed_weights(self, model_cls, model_type, trust_remote_code):
        model = model_cls.from_pretrained(
            MODEL_NAMES[model_type],
            export=True,
            load_in_8bit=True,
            stateful=False,
            trust_remote_code=trust_remote_code,
        )

        if model_type == "open-clip":
            self.assertEqual(model.text_model._openvino_config.quantization_config.bits, 8)
            self.assertEqual(model.text_model._openvino_config.dtype, "int8")
            self.assertEqual(model.visual_model._openvino_config.quantization_config.bits, 8)
            self.assertEqual(model.visual_model._openvino_config.dtype, "int8")
        else:
            self.assertEqual(model._openvino_config.quantization_config.bits, 8)
            self.assertEqual(model._openvino_config.dtype, "int8")

        if model_type == "open-clip":
            pytest.skip(reason="ticket 161043")
        elif model_type == "t5":
            pytest.skip(reason="ticket 160958")
        else:
            check_optimization_not_applicable_to_optimized_model(model, quantization_config={"bits": 8})

        submodels = (
            {"text_model": model.text_model, "visual_model": model.visual_model}
            if model_type == "open-clip"
            else model.ov_submodels
        )
        expected_ov_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
        expected_ov_int8 = {k: {"int8": v} for k, v in expected_ov_int8.items()}
        check_compression_state_per_model(self, submodels, expected_ov_int8)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION)
    def test_ovmodel_hybrid_quantization(self, model_cls, model_type, expected_fake_nodes, expected_int8_nodes):
        model_id = MODEL_NAMES[model_type]
        quantization_config = OVWeightQuantizationConfig(bits=8, dataset="conceptual_captions", num_samples=2)
        with TemporaryDirectory() as tmp_dir:
            model = model_cls.from_pretrained(model_id, export=True, quantization_config=quantization_config)

            num_fake, num_weight_nodes = get_num_quantized_nodes(
                model.unet if model.unet is not None else model.transformer
            )
            self.assertEqual(expected_fake_nodes, num_fake)
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])
            self.assertEqual(0, num_weight_nodes["int4"])

            model.save_pretrained(tmp_dir)
            check_optimization_not_applicable_to_optimized_model(model, quantization_config)

    def test_stable_diffusion_with_weight_compression(self):
        int8_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=MODEL_NAMES["stable-diffusion"], export=True)
        quantization_config = OVWeightQuantizationConfig(bits=8, quant_method=OVQuantizationMethod.DEFAULT)
        quantizer = OVQuantizer(int8_pipe)

        quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config))

        num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(
            int8_pipe.unet if int8_pipe.unet is not None else int8_pipe.transformer
        )
        self.assertEqual(0, num_fake_nodes)
        self.assertEqual(242, num_weight_nodes["int8"])
        self.assertEqual(0, num_weight_nodes["int4"])
        quantization_config = OVWeightQuantizationConfig(
            bits=8, dataset="conceptual_captions", num_samples=2, quant_method=OVQuantizationMethod.HYBRID
        )
        check_optimization_not_applicable_to_optimized_model(int8_pipe, quantization_config)

    @parameterized.expand(
        [
            (*it[0], it[1])
            for it in itertools.product(SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION[-1:], [False, True])
        ]
    )
    def test_ovmodel_hybrid_quantization_with_custom_dataset(
        self,
        model_cls,
        model_type,
        expected_fake_nodes,
        expected_int8_nodes,
        dataset_via_config,
    ):
        model_id = MODEL_NAMES[model_type]
        # TODO: Run only dataset_via_config=True after v1.25
        dataset = [
            "dream rose covered with clean crystal, sharp edges, transparent, beautiful, highly detailed, high render"
        ]
        model = model_cls.from_pretrained(model_id, export=True)
        quantizer = OVQuantizer(model)
        quantization_config = OVWeightQuantizationConfig(bits=8, num_samples=3, quant_method="hybrid")
        self.assertEqual(quantization_config.quant_method, OVQuantizationMethod.HYBRID)

        quantization_config.dataset = dataset if dataset_via_config else None
        dataset = None if dataset_via_config else dataset
        quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config), calibration_dataset=dataset)
        num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(
            model.unet if model.unet is not None else model.transformer
        )
        self.assertEqual(expected_fake_nodes, num_fake_nodes)
        self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])
        self.assertEqual(0, num_weight_nodes["int4"])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_AUTOCOMPRESSED_MATMULS)
    @unittest.mock.patch.dict(
        "optimum.intel.openvino.configuration._DEFAULT_4BIT_WQ_CONFIGS", {"facebook/opt-125m": DEFAULT_INT4_CONFIG}
    )
    def test_ovmodel_4bit_auto_compression(self, model_cls, model_type, expected_ov_int8, expected_ov_int4):
        with TemporaryDirectory() as tmp_dir:
            model_id = MODEL_NAMES[model_type]
            model = model_cls.from_pretrained(model_id, export=True, quantization_config={"bits": 4})
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_ov_int4, num_weight_nodes["int4"])
            self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])
            model.save_pretrained(tmp_dir)

            openvino_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(openvino_config.quantization_config.bits, 4)
            self.assertEqual(openvino_config.dtype, "int4")
            if model_id == "facebook/opt-125m":
                for key, value in self.DEFAULT_INT4_CONFIG.items():
                    self.assertEqual(value, getattr(openvino_config.quantization_config, key))
            check_optimization_not_applicable_to_optimized_model(model, quantization_config={"bits": 8})

    @parameterized.expand(LOAD_IN_4_BITS_SCOPE)
    def test_ovmodel_4bit_auto_compression_with_config(
        self, model_cls, model_name, trust_remote_code, quantization_config, expected_num_weight_nodes_per_model
    ):
        model_id = MODEL_NAMES[model_name]
        with TemporaryDirectory() as tmp_dir:
            quantization_config = OVWeightQuantizationConfig.from_dict(quantization_config)
            model = model_cls.from_pretrained(
                model_id, export=True, quantization_config=quantization_config, trust_remote_code=trust_remote_code
            )
            if quantization_config.quant_method.lower() == "awq":
                # TODO: Check that AWQ was actually applied
                pass

            submodels = model.ov_submodels
            check_compression_state_per_model(self, submodels, expected_num_weight_nodes_per_model)

            model.save_pretrained(tmp_dir)
            # At the moment the first model in the list is the only one we apply data-aware compression to
            wc_rt_info = next(iter(submodels.values())).get_rt_info()["nncf"]["weight_compression"]
            self.assertEqual(quantization_config.quant_method.lower() == "awq", wc_rt_info["awq"].value == "True")
            self.assertEqual(
                quantization_config.scale_estimation or False, wc_rt_info["scale_estimation"].value == "True"
            )
            self.assertEqual(quantization_config.gptq or False, wc_rt_info["gptq"].value == "True")
            self.assertEqual(
                quantization_config.lora_correction or False, wc_rt_info["lora_correction"].value == "True"
            )

            openvino_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(openvino_config.quantization_config.bits, 4)
            self.assertEqual(openvino_config.dtype, quantization_config.dtype)

    @parameterized.expand(((OVModelForCausalLM, "gpt2"),))
    def test_ovmodel_stateful_load_with_compressed_weights(self, model_cls, model_type):
        model = model_cls.from_pretrained(MODEL_NAMES[model_type], export=True, load_in_8bit=True, stateful=True)
        self.assertTrue(model.stateful)
        self.assertTrue(model.use_cache)

        _, num_weight_nodes = get_num_quantized_nodes(model)
        expected_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
        expected_int8 = {k: {"int8": v} for k, v in expected_int8.items()}
        check_compression_state_per_model(self, model.ov_submodels, expected_int8)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION)
    def test_ovmodel_load_with_uncompressed_weights(self, model_cls, model_type, trust_remote_code):
        model = model_cls.from_pretrained(
            MODEL_NAMES[model_type], export=True, load_in_8bit=False, trust_remote_code=trust_remote_code
        )

        submodels = (
            [model.text_model, model.visual_model] if model_type == "open-clip" else model.ov_submodels.values()
        )
        for i, submodel in enumerate(submodels):
            ov_model = submodel if isinstance(submodel, ov.Model) else submodel.model
            _, num_weight_nodes = get_num_quantized_nodes(ov_model)
            self.assertEqual(0, num_weight_nodes["int8"])
            if "text-generation" in model.export_feature or ("image-text-to-text" in model.export_feature and i == 0):
                self.assertTrue(ov_model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))
                kv_cache_precision = ov_model.get_rt_info(["runtime_options", "KV_CACHE_PRECISION"]).value
                self.assertTrue(kv_cache_precision == "f16")

    def test_ovmodel_load_large_model_with_default_compressed_weights(self):
        compressed_model_mock_obj = unittest.mock.Mock()
        compressed_model_mock_obj.has_rt_info.return_value = False

        def main_export_in_stacktrace(*args, **kwargs):
            # Compression was called from `main_export`
            self.assertTrue(inspect.stack()[5].function == "main_export")
            return compressed_model_mock_obj

        with unittest.mock.patch(
            "openvino.op.Constant.shape", new_callable=unittest.mock.PropertyMock
        ) as ov_constant_shape:
            ov_constant_shape.return_value = (2000000000,)
            with unittest.mock.patch(
                "nncf.compress_weights", side_effect=main_export_in_stacktrace
            ) as compress_weights_patch:
                _ = OVModelForCausalLM.from_pretrained(
                    MODEL_NAMES["llama"], export=True, compile=False, use_cache=False
                )
                compression_params = {
                    "mode": nncf.CompressWeightsMode.INT8_ASYM,
                    "ratio": 1.0,
                    "group_size": -1,
                    "all_layers": None,
                    "sensitivity_metric": None,
                    "dataset": None,
                    "ignored_scope": nncf.IgnoredScope(),
                    "awq": None,
                    "subset_size": 128,
                    "scale_estimation": None,
                    "gptq": None,
                    "lora_correction": None,
                    "backup_mode": None,
                }
                compress_weights_patch.assert_called_with(
                    unittest.mock.ANY,
                    **compression_params,
                )

    def test_ovmodel_load_large_model_with_uncompressed_weights(self):
        with unittest.mock.patch(
            "openvino.op.Constant.shape", new_callable=unittest.mock.PropertyMock
        ) as ov_constant_shape:
            ov_constant_shape.return_value = (2000000000,)
            with unittest.mock.patch("nncf.compress_weights") as compress_weights_patch:
                model = OVModelForCausalLM.from_pretrained(
                    MODEL_NAMES["llama"], export=True, load_in_8bit=False, compile=False, use_cache=False
                )
                compress_weights_patch.assert_not_called()
                self.assertTrue(model.model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))

    def test_ovmodel_load_large_model_with_additional_quantization_config(self):
        compressed_model_mock_obj = unittest.mock.Mock()
        compressed_model_mock_obj.has_rt_info.return_value = False

        def main_export_not_in_stacktrace(*args, **kwargs):
            # Compression was not called from `main_export`
            self.assertTrue(all(frame_info.function != "main_export" for frame_info in inspect.stack()))
            return compressed_model_mock_obj

        with unittest.mock.patch(
            "openvino.op.Constant.shape", new_callable=unittest.mock.PropertyMock
        ) as ov_constant_shape:
            ov_constant_shape.return_value = (2000000000,)
            with unittest.mock.patch(
                "nncf.compress_weights", side_effect=main_export_not_in_stacktrace
            ) as compress_weights_patch:
                _ = OVModelForCausalLM.from_pretrained(
                    MODEL_NAMES["llama"],
                    export=True,
                    compile=False,
                    use_cache=False,
                    quantization_config=OVWeightQuantizationConfig(bits=4, sym=True, group_size=-1, ratio=0.8),
                )
                compression_params = {
                    "mode": nncf.CompressWeightsMode.INT4_SYM,
                    "ratio": 0.8,
                    "group_size": -1,
                    "all_layers": None,
                    "sensitivity_metric": None,
                    "dataset": None,
                    "ignored_scope": nncf.IgnoredScope(),
                    "awq": None,
                    "subset_size": 128,
                    "scale_estimation": None,
                    "gptq": None,
                    "lora_correction": None,
                    "backup_mode": None,
                }
                compress_weights_patch.assert_called_with(unittest.mock.ANY, **compression_params)

    @parameterized.expand(LOAD_IN_4_BITS_SCOPE[::5])
    def test_ovmodel_4bit_dynamic_with_config(
        self, model_cls, model_name, trust_remote_code, quantization_config, expected_num_weight_nodes_per_model
    ):
        model_id = MODEL_NAMES[model_name]
        with TemporaryDirectory() as tmp_dir:
            group_size = quantization_config.pop("group_size", 32)
            quantization_config = OVDynamicQuantizationConfig(
                weights_group_size=group_size, activations_group_size=group_size, **quantization_config
            )
            model = model_cls.from_pretrained(
                model_id, export=True, quantization_config=quantization_config, trust_remote_code=trust_remote_code
            )
            self.assertEqual(model.ov_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"], str(group_size))
            self.assertEqual(model.ov_config["KV_CACHE_PRECISION"], "u8")

            check_compression_state_per_model(self, model.ov_submodels, expected_num_weight_nodes_per_model)

            model.save_pretrained(tmp_dir)
            openvino_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(openvino_config.quantization_config.bits, 4)
            self.assertEqual(openvino_config.dtype, quantization_config.dtype)


class OVQuantizerQATest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("hf-internal-testing/tiny-random-BertForQuestionAnswering",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_automodel_static_quantization(self, model_name):
        def preprocess_function(examples, tokenizer):
            return tokenizer(
                examples["question"], examples["context"], padding="max_length", max_length=64, truncation=True
            )

        with TemporaryDirectory() as tmp_dir:
            transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            quantizer = OVQuantizer.from_pretrained(transformers_model)
            calibration_dataset = quantizer.get_calibration_dataset(
                "squadshifts",
                dataset_config_name="new_wiki",
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=10,
                dataset_split="test",
                trust_remote_code=True,
            )
            ov_config = OVConfig(quantization_config=OVQuantizationConfig())
            quantizer.quantize(save_directory=tmp_dir, calibration_dataset=calibration_dataset, ov_config=ov_config)

            # Test that inference on quantized model works
            model = OVModelForQuestionAnswering.from_pretrained(tmp_dir)
            tokens = tokenizer.encode_plus(
                "This is a sample question", "This is a sample context", add_special_tokens=True, return_tensors="pt"
            )
            model(**tokens, return_dict=True)

            # Test loading model a second time to catch issues with caching
            try:
                model = OVModelForQuestionAnswering.from_pretrained(tmp_dir)
            except RuntimeError:
                self.fail("Loading BERT QA model a second time failed")

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_ovmodel_static_quantization(self, model_name):
        def preprocess_function(examples, tokenizer):
            return tokenizer(
                examples["question"], examples["context"], padding="max_length", max_length=64, truncation=True
            )

        with TemporaryDirectory() as tmp_dir:
            transformers_model = OVModelForQuestionAnswering.from_pretrained(model_name, export=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            quantizer = OVQuantizer.from_pretrained(transformers_model)
            calibration_dataset = quantizer.get_calibration_dataset(
                "squadshifts",
                dataset_config_name="new_wiki",
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=10,
                dataset_split="test",
                trust_remote_code=True,
            )
            ov_config = OVConfig(quantization_config=OVQuantizationConfig())
            quantizer.quantize(save_directory=tmp_dir, calibration_dataset=calibration_dataset, ov_config=ov_config)

            # Test that inference on quantized model works
            model = OVModelForQuestionAnswering.from_pretrained(tmp_dir)
            tokens = tokenizer.encode_plus(
                "This is a sample question", "This is a sample context", add_special_tokens=True, return_tensors="pt"
            )
            model(**tokens, return_dict=True)

            # Test loading model a second time to catch issues with caching
            try:
                model = OVModelForQuestionAnswering.from_pretrained(tmp_dir)
            except RuntimeError:
                self.fail("Loading BERT QA model a second time failed")

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())


class OVQuantizationConfigTest(unittest.TestCase):
    QUANTIZATION_CONFIGS = (
        (None,),
        (OVWeightQuantizationConfig(),),
        (OVWeightQuantizationConfig(bits=8, sym=True),),
        (
            OVWeightQuantizationConfig(
                dataset="wikitext2",
                bits=4,
                ignored_scope={"names": ["op_name"]},
                sym=False,
                tokenizer="dbmdz/bert-base-german-cased",
                ratio=1.0,
                group_size=128,
                all_layers=True,
                sensitivity_metric="mean_activation_magnitude",
                num_samples=100,
                quant_method=OVQuantizationMethod.DEFAULT,
            ),
        ),
        (OVWeightQuantizationConfig(bits=4, dataset=["hello world", "i'm alive"]),),
        (
            OVQuantizationConfig(
                ignored_scope={"names": ["op_name"]},
                num_samples=100,
                sym=False,
                model_type="transformer",
                fast_bias_correction=True,
                overflow_fix="disable",
            ),
        ),
        (OVQuantizationConfig(ignored_scope=nncf.IgnoredScope(names=["op_name"])),),
    )

    QUANTIZATION_CONFIG_DICTS = (
        (dict(bits=8, sym=True), OVWeightQuantizationConfig, None),
        (
            dict(
                dataset="wikitext2",
                bits=4,
                ignored_scope={"names": ["op_name"]},
                sym=False,
                tokenizer="dbmdz/bert-base-german-cased",
                ratio=1.0,
                group_size=128,
                all_layers=True,
                sensitivity_metric="mean_activation_magnitude",
                num_samples=100,
                quant_method=OVQuantizationMethod.DEFAULT,
            ),
            OVWeightQuantizationConfig,
            None,
        ),
        (dict(), OVWeightQuantizationConfig, "Can't determine type of OV quantization config"),
        (
            dict(ignored_scope={"names": ["op_name"]}),
            OVWeightQuantizationConfig,
            "Can't determine type of OV quantization config",
        ),
        (dict(num_samples=100), OVWeightQuantizationConfig, "Can't determine type of OV quantization config"),
        (dict(abc="def"), OVWeightQuantizationConfig, "Can't determine type of OV quantization config"),
        (
            dict(bits=8, fast_bias_correction=True, dataset="librispeech"),
            OVQuantizationConfig,
            None,
        ),
        (dict(model_type="transformer"), OVQuantizationConfig, None),
        (
            dict(
                ignored_scope={"names": ["op_name"]},
                num_samples=100,
                sym=False,
                model_type="transformer",
                fast_bias_correction=True,
                overflow_fix="disable",
            ),
            OVQuantizationConfig,
            None,
        ),
        (dict(weight_only=True), OVWeightQuantizationConfig, None),
        (dict(weight_only=False), OVQuantizationConfig, None),
        (dict(abc="def", weight_only=False), OVQuantizationConfig, None),
        (dict(abc="def", weight_only=True), OVWeightQuantizationConfig, None),
        (
            dict(bits=8, fast_bias_correction=True, dataset="librispeech", weight_only=True),
            OVQuantizationConfig,
            None,
        ),
        (
            dict(bits=4, dataset="wikitext2", weight_only=True),
            OVWeightQuantizationConfig,
            None,
        ),
        (dict(bits=8, fast_bias_correction=True, weight_only=False), OVQuantizationConfig, None),
    )

    QUANTIZATION_CONFIGS_WITH_KWARGS = (
        (
            OVWeightQuantizationConfig,
            {
                "advanced_parameters": nncf.AdvancedCompressionParameters(statistics_path="statistics_path"),
                "some_arg": "some_value",
            },
            {
                "advanced_parameters": nncf.AdvancedCompressionParameters(statistics_path="statistics_path"),
                "some_arg": "some_value",
            },
        ),
        (
            OVQuantizationConfig,
            {
                "advanced_parameters": nncf.AdvancedQuantizationParameters(disable_channel_alignment=True),
                "some_arg": "some_value",
            },
            {
                "advanced_parameters": nncf.AdvancedQuantizationParameters(
                    overflow_fix=nncf.OverflowFix.DISABLE,
                    disable_channel_alignment=True,
                ),
                "some_arg": "some_value",
            },
        ),
        (
            OVQuantizationConfig,
            {
                "advanced_parameters": nncf.AdvancedQuantizationParameters(overflow_fix=nncf.OverflowFix.ENABLE),
            },
            {
                "advanced_parameters": nncf.AdvancedQuantizationParameters(
                    overflow_fix=nncf.OverflowFix.DISABLE,
                ),
            },
        ),
        (
            OVQuantizationConfig,
            {
                "smooth_quant_alpha": 0.5,
                "advanced_parameters": nncf.AdvancedQuantizationParameters(
                    smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=0.7, convolution=0.7),
                ),
            },
            {
                "advanced_parameters": nncf.AdvancedQuantizationParameters(
                    overflow_fix=nncf.OverflowFix.DISABLE,
                    smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=0.5, convolution=0.7),
                ),
            },
        ),
    )

    def get_default_configurations() -> dict:
        default_configurations = deepcopy(_DEFAULT_4BIT_WQ_CONFIGS)
        default_configurations.update({"default": _DEFAULT_4BIT_WQ_CONFIG})
        return default_configurations

    DEFAULT_CONFIGURATIONS = get_default_configurations()

    @parameterized.expand(QUANTIZATION_CONFIGS)
    def test_config_serialization(self, quantization_config: OVQuantizationConfigBase):
        ov_config = OVConfig(quantization_config=quantization_config)
        with TemporaryDirectory() as tmp_dir:
            ov_config.save_pretrained(tmp_dir)
            loaded_ov_config = OVConfig.from_pretrained(tmp_dir)

            if quantization_config is None:
                self.assertEqual(loaded_ov_config.quantization_config, None)
                return
            for key, value in loaded_ov_config.quantization_config.to_dict().items():
                initial_value = getattr(ov_config.quantization_config, key)
                self.assertEqual(value, initial_value)

    @parameterized.expand(QUANTIZATION_CONFIG_DICTS)
    def test_config_from_dict(self, quantization_config: dict, config_type: type, warning_log: Union[str, None]):
        from optimum.intel.openvino.configuration import logger as configuration_logger

        if warning_log is not None:
            with self.assertLogs(configuration_logger, logging.WARN) as cm:
                ov_config = OVConfig(quantization_config=quantization_config)
                self.assertTrue(any(warning_log in log for log in cm.output))
        else:
            ov_config = OVConfig(quantization_config=quantization_config)
        self.assertIsInstance(ov_config.quantization_config, config_type)
        for k, v in quantization_config.items():
            if hasattr(ov_config.quantization_config, k):
                self.assertEqual(getattr(ov_config.quantization_config, k), v)

    @parameterized.expand(DEFAULT_CONFIGURATIONS)
    def test_named_default_configurations(self, config_id: str):
        custom_configuration = self.DEFAULT_CONFIGURATIONS[config_id]
        prepared_config = OVModelForCausalLM._prepare_quantization_config(custom_configuration)
        for field_name, reference_value in custom_configuration.items():
            value = prepared_config.__getattribute__(field_name)
            self.assertEqual(value, reference_value)

    def test_for_no_short_id_duplicates(self):
        short_ids = set()
        for model_id in _DEFAULT_4BIT_WQ_CONFIGS.keys():
            short_id = model_id.split("/")[1]
            assert short_id not in short_ids
            short_ids.add(short_id)

    @parameterized.expand(QUANTIZATION_CONFIGS_WITH_KWARGS)
    def test_config_init_kwargs(
        self,
        config_type: Type[Union[OVWeightQuantizationConfig, OVQuantizationConfig]],
        config_kwargs: dict,
        ref_nncf_dict: dict,
    ):
        nncf_dict = config_type(**config_kwargs).to_nncf_dict()
        ref_nncf_dict = config_type().to_nncf_dict() | ref_nncf_dict
        self.assertTrue(self.compare_objects(nncf_dict, ref_nncf_dict))

    @parameterized.expand(
        [
            ("nncf.compress_weights", "_weight_only_quantization", "dataset", OVWeightQuantizationConfig),
            ("nncf.quantize", "_full_quantization", "calibration_dataset", OVQuantizationConfig),
        ]
    )
    def test_quantization_kwargs_override(self, mock_method_name, quantization_function, dataset_key, config_type):
        from optimum.intel.openvino.quantization import _weight_only_quantization, _full_quantization

        with unittest.mock.patch(mock_method_name) as mock_method:
            mock_model = unittest.mock.Mock([])
            mock_model.get_rt_info = unittest.mock.Mock(return_value={})

            mock_quantization_config = unittest.mock.Mock(config_type)
            mock_quantization_config.to_nncf_dict.return_value = {"param1": "value1", "param2": "value2"}

            additional_kwargs = {"param2": "new_value2", "param3": "value3"}

            quantization_function = (
                _weight_only_quantization
                if quantization_function == "_weight_only_quantization"
                else _full_quantization
            )
            quantization_function(mock_model, mock_quantization_config, None, **additional_kwargs)

            expected_kwargs = {"param1": "value1", "param2": "new_value2", "param3": "value3", dataset_key: None}

            mock_method.assert_called_once_with(mock_model, **expected_kwargs)

    @staticmethod
    def compare_objects(o1, o2) -> bool:
        if dataclasses.is_dataclass(o1) and dataclasses.is_dataclass(o2):
            o1 = o1.__dict__
            o2 = o2.__dict__
        if isinstance(o1, dict) and isinstance(o2, dict):
            for k in set(o1.keys()) | set(o2.keys()):
                if not OVQuantizationConfigTest.compare_objects(o1[k], o2[k]):
                    return False
            return True
        if isinstance(o1, Iterable) and isinstance(o2, Iterable) and not (isinstance(o1, str) or isinstance(o2, str)):
            for it1, it2 in zip(o1, o2):
                if not OVQuantizationConfigTest.compare_objects(it1, it2):
                    return False
            return True
        return o1 == o2


class InferRequestWrapperTest(unittest.TestCase):
    MODEL_NAME = ("whisper",)
    APPLY_CACHING = (False, True)

    @staticmethod
    def _generate_random_audio_data(processor):
        t = np.linspace(0, 1.0, int(1000), endpoint=False)
        audio_data = 0.5 * np.sin((2 + np.random.random()) * np.pi * t)
        input_features = processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features
        return input_features

    @parameterized.expand(itertools.product(MODEL_NAME, APPLY_CACHING))
    def test_calibration_data_uniqueness(self, model_name, apply_caching):
        model_id = MODEL_NAMES[model_name]
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, compile=True)
        processor = AutoProcessor.from_pretrained(model_id)

        calibration_data = []
        if not ov_model.decoder.stateful:
            ov_model.decoder_with_past.request = InferRequestWrapper(
                ov_model.decoder_with_past.request, calibration_data, apply_caching=apply_caching
            )
        else:
            ov_model.decoder.request = InferRequestWrapper(
                ov_model.decoder.request, calibration_data, apply_caching=apply_caching
            )
        for _ in range(2):
            input_features = self._generate_random_audio_data(processor)
            ov_model.generate(input_features, max_new_tokens=10, min_new_tokens=10)

        data_hashes_per_key = defaultdict(list)
        data_id_per_key = defaultdict(set)

        for inputs_dict in calibration_data:
            for k, v in inputs_dict.items():
                if k in ["input_ids", "beam_idx"]:
                    continue

                x = (v.numpy() if isinstance(v, torch.Tensor) else v).copy()
                data_hashes_per_key[k].append(hash(x.tobytes()))
                data_id_per_key[k].add(id(v))
        for k, data_hashes in data_hashes_per_key.items():
            # All hashes can not be equal because calibration dataset contains at least 2 different samples
            self.assertTrue(any(data_hashes[0] != it for it in data_hashes))
        if apply_caching:
            # With caching, encoder hidden states tensors should be cached, resulting in only 2 tensors stored
            self.assertEqual(len(data_id_per_key["encoder_hidden_states"]), 2)
        else:
            # Without caching, encoder hidden states tensors will be unique for each collected input
            self.assertGreater(len(data_id_per_key["encoder_hidden_states"]), 2)


def check_optimization_not_applicable_to_optimized_model(model, quantization_config):
    quantizer = OVQuantizer(model)
    with pytest.raises(
        RuntimeError,
        match="Cannot apply optimization to the model because it was already optimized with the following config",
    ):
        quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config))
