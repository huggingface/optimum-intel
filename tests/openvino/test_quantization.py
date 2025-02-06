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
import inspect

# ruff: noqa

import itertools
import logging
import unittest
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Union

import openvino as ov
import pytest
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from parameterized import parameterized
import nncf
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    default_data_collator,
)
from transformers.testing_utils import slow
from transformers.utils.quantization_config import QuantizationMethod

from optimum.intel.openvino.utils import deepcopy_data


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
    OVTrainer,
    OVQuantizationConfig,
    OVWeightQuantizationConfig,
    OVDynamicQuantizationConfig,
    OVModelOpenCLIPForZeroShotImageClassification,
    OVModelForVisualCausalLM,
)
from optimum.intel.openvino.configuration import (
    OVQuantizationMethod,
    OVQuantizationConfigBase,
    _DEFAULT_4BIT_CONFIGS,
    _DEFAULT_4BIT_CONFIG,
)
from optimum.intel.openvino.utils import TemporaryDirectory
from copy import deepcopy

from optimum.intel.openvino.quantization import InferRequestWrapper
from optimum.intel.utils.import_utils import is_openvino_version, is_transformers_version
from utils_tests import (
    MODEL_NAMES,
    get_num_quantized_nodes,
    _ARCHITECTURES_TO_EXPECTED_INT8,
    check_compression_state_per_model,
)

_TASK_TO_DATASET = {
    "text-generation": ("wikitext", "wikitext-2-raw-v1", "text"),
    "text-classification": ("glue", "sst2", "sentence"),
}


class OVQuantizerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_TORCH_MODEL = (
        (OVModelForSequenceClassification, "bert", 32, 35),
        (OVModelForCausalLM, "gpt2", 41 if is_transformers_version("<", "4.42.0") else 31, 22),
    )
    SUPPORTED_ARCHITECTURES_OV_MODEL = (
        (OVModelForSequenceClassification, "bert", 32, 35),
        (OVModelForCausalLM, "gpt2", 31, 22),
    )
    SUPPORTED_ARCHITECTURES_OV_MODEL_WITH_AUTO_DATASET = [
        (
            OVModelForSpeechSeq2Seq,
            "whisper",
            OVQuantizationConfig(
                dataset="librispeech",
                num_samples=1,
                processor=MODEL_NAMES["whisper"],
                trust_remote_code=True,
                weight_only=False,
                smooth_quant_alpha=0.95,
            ),
            (14, 22, 21) if is_transformers_version("<=", "4.42.4") else (14, 22, 25),
            (14, 21, 17) if is_transformers_version("<=", "4.42.4") else (14, 22, 18),
        ),
        (
            OVModelForCausalLM,
            "llama",
            OVQuantizationConfig(
                dataset="wikitext2",
                num_samples=1,
                weight_only=False,
                weight_format="f8e4m3",
                activation_format="f8e4m3",
            ),
            (13,),
            (16,),
        ),
    ]

    @parameterized.expand(SUPPORTED_ARCHITECTURES_TORCH_MODEL)
    def test_automodel_static_quantization(self, model_cls, model_name, expected_fake_nodes, expected_int8_nodes):
        model_id = MODEL_NAMES[model_name]
        task = model_cls.export_feature
        dataset_name, dataset_config_name, column_name = _TASK_TO_DATASET[task]
        file_name = "openvino_quantized_model.xml"

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples[column_name], padding="max_length", max_length=128, truncation=True)

        with TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.auto_model_class.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task)

            calibration_dataset = quantizer.get_calibration_dataset(
                dataset_name,
                dataset_config_name=dataset_config_name,
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=10,
                dataset_split="train",
                trust_remote_code=True,
            )
            ov_config = OVConfig(quantization_config=OVQuantizationConfig())
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_OV_MODEL)
    def test_ovmodel_static_quantization(self, model_cls, model_name, expected_fake_nodes, expected_int8_nodes):
        model_id = MODEL_NAMES[model_name]
        task = model_cls.export_feature
        dataset_name, dataset_config_name, column_name = _TASK_TO_DATASET[task]

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples[column_name], padding="max_length", max_length=128, truncation=True)

        with TemporaryDirectory() as tmp_dir:
            ov_model = model_cls.from_pretrained(model_id, export=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            quantizer = OVQuantizer.from_pretrained(ov_model, task=task)

            calibration_dataset = quantizer.get_calibration_dataset(
                dataset_name,
                dataset_config_name=dataset_config_name,
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=10,
                dataset_split="train",
                trust_remote_code=True,
            )
            ov_config = OVConfig(quantization_config=OVQuantizationConfig())
            quantizer.quantize(save_directory=tmp_dir, calibration_dataset=calibration_dataset, ov_config=ov_config)

            model = model_cls.from_pretrained(tmp_dir)

            num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_fake_nodes, num_fake_nodes)
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())
            check_optimization_not_applicable_to_optimized_model(
                model, quantization_config=OVWeightQuantizationConfig(bits=8)
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_OV_MODEL_WITH_AUTO_DATASET)
    def test_ov_model_static_quantization_with_auto_dataset(
        self, model_cls, model_name, quantization_config, expected_fake_nodes, expected_low_precision_nodes
    ):
        model_id = MODEL_NAMES[model_name]
        quant_mode = quantization_config.activation_format

        with TemporaryDirectory() as tmp_dir:
            ov_model = model_cls.from_pretrained(model_id, quantization_config=quantization_config)
            ov_model.save_pretrained(tmp_dir)

            if model_cls == OVModelForSpeechSeq2Seq:
                models = [ov_model.encoder.model, ov_model.decoder.model]

                if ov_model.decoder_with_past is not None:
                    models.append(ov_model.decoder_with_past.model)
                for model, expected_fake_nodes, expected_lp_nodes in zip(
                    models,
                    expected_fake_nodes,
                    expected_low_precision_nodes,
                ):
                    num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(model)
                    self.assertEqual(expected_fake_nodes, num_fake_nodes)
                    self.assertEqual(expected_lp_nodes, num_weight_nodes[quant_mode])

                input_features = torch.randn((1, 128, 3000), dtype=torch.float32)
                ov_model.generate(input_features)
            elif model_cls == OVModelForCausalLM:
                num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(ov_model.model)
                self.assertEqual(expected_fake_nodes[0], num_fake_nodes)
                self.assertEqual(expected_low_precision_nodes[0], num_weight_nodes[quant_mode])

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokens = tokenizer("This is a sample input", return_tensors="pt")
                outputs = ov_model(**tokens)
                self.assertTrue("logits" in outputs)
            else:
                raise Exception("Unexpected model class.")


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
            [{"int8": 14, "int4": 30}],  # reference number of low-precision nodes
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(bits=4, weight_format="mxfp4", group_size=32),
            [{"int8": 4, "f4e2m1": 20, "f8e8m0": 20}],
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(bits=4, weight_format="nf4", group_size=32),
            [
                {
                    "int8": 4,
                    "nf4": 20,
                }
            ],
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
            [{"int8": 4, "int4": 38}],
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(bits=4, sym=False, group_size=-1, ratio=0.8, all_layers=True),
            [{"int8": 18, "int4": 26}],
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
            [{"int8": 14, "int4": 25}],
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
            [{"int8": 16, "int4": 24}],
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
            [{"int8": 8, "int4": 12}],
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
            [{"int8": 8, "int4": 12}],
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
            [{"int8": 8, "int4": 12}],
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
            [{"int8": 60, "int4": 28}],
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="none", group_size=16),
            [{"int4": 28}],
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="none", group_size=16, ratio=0.5),
            [{"int4": 6}],
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="int8_sym", group_size=16, ratio=0.5),
            [{"int4": 6, "int8": 13}],
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            False,
            dict(bits=4, backup_precision="int8_asym", group_size=16, ratio=0.5),
            [{"int4": 6, "int8": 26}],
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
                    [{"int8": 6, "int4": 24}, {"int8": 1}, {"int8": 9}],
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
                    [{"int8": 16, "int4": 14}, {"int8": 1}, {"int8": 15}],
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
                    [{"int8": 8, "int4": 22}, {"int8": 1}, {"int8": 26}, {"int8": 6}],
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
                    [{"int8": 8, "int4": 22}, {"int8": 1}, {"int8": 11}],
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
                    [{"int8": 4, "int4": 14}, {"int8": 1}, {"int8": 7}, {"int8": 2}],
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
                    [{"int8": 10, "int4": 20}, {"int8": 1}, {"int8": 1}, {"int8": 10}],
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

        if model.export_feature.startswith("text2text-generation"):
            models = [model.encoder, model.decoder]
            if model.decoder_with_past is not None:
                models.append(model.decoder_with_past)
        elif model.export_feature == "text-to-image":
            models = [model.unet, model.vae_encoder, model.vae_decoder]
            models.append(model.text_encoder if model_type in ["stable-diffusion", "sana"] else model.text_encoder_2)
        elif model_type == "open-clip":
            models = [model.text_model, model.visual_model]
        elif model.export_feature == "image-text-to-text":
            models = list(model.submodels.values())
        else:
            models = [model]

        if model_type == "open-clip":
            pytest.skip(reason="ticket 161043")
        elif model_type == "t5":
            pytest.skip(reason="ticket 160958")
        else:
            check_optimization_not_applicable_to_optimized_model(model, quantization_config={"bits": 8})

        expected_ov_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
        expected_ov_int8 = [{"int8": it} for it in expected_ov_int8]
        check_compression_state_per_model(self, models, expected_ov_int8)

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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION[-1:])
    def test_ovmodel_hybrid_quantization_with_custom_dataset(
        self, model_cls, model_type, expected_fake_nodes, expected_int8_nodes
    ):
        model_id = MODEL_NAMES[model_type]
        dataset = [
            "dream rose covered with clean crystal, sharp edges, transparent, beautiful, highly detailed, high render"
        ]
        model = model_cls.from_pretrained(model_id, export=True)
        quantizer = OVQuantizer(model)
        quantization_config = OVWeightQuantizationConfig(bits=8, num_samples=3, quant_method="hybrid")
        self.assertEqual(quantization_config.quant_method, OVQuantizationMethod.HYBRID)

        quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config), calibration_dataset=dataset)
        num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(
            model.unet if model.unet is not None else model.transformer
        )
        self.assertEqual(expected_fake_nodes, num_fake_nodes)
        self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])
        self.assertEqual(0, num_weight_nodes["int4"])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_AUTOCOMPRESSED_MATMULS)
    @unittest.mock.patch.dict(
        "optimum.intel.openvino.configuration._DEFAULT_4BIT_CONFIGS", {"facebook/opt-125m": DEFAULT_INT4_CONFIG}
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

            submodels = []
            if isinstance(model, OVModelForCausalLM):
                submodels = [model.model]
            elif isinstance(model, OVModelForVisualCausalLM):
                submodels = list(model.submodels.values())
            check_compression_state_per_model(self, submodels, expected_num_weight_nodes_per_model)

            model.save_pretrained(tmp_dir)
            # At the moment the first model in the list is the only one we apply data-aware compression to
            wc_rt_info = submodels[0].get_rt_info()["nncf"]["weight_compression"]
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
            self.assertEqual(openvino_config.dtype, quantization_config.weight_format)

    @parameterized.expand(((OVModelForCausalLM, "gpt2"),))
    def test_ovmodel_stateful_load_with_compressed_weights(self, model_cls, model_type):
        model = model_cls.from_pretrained(MODEL_NAMES[model_type], export=True, load_in_8bit=True, stateful=True)
        self.assertTrue(model.stateful)
        self.assertTrue(model.use_cache)

        expected_ov_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type][0]
        _, num_weight_nodes = get_num_quantized_nodes(model)
        check_compression_state_per_model(self, [model.model], [{"int8": expected_ov_int8}])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION)
    def test_ovmodel_load_with_uncompressed_weights(self, model_cls, model_type, trust_remote_code):
        model = model_cls.from_pretrained(
            MODEL_NAMES[model_type], export=True, load_in_8bit=False, trust_remote_code=trust_remote_code
        )
        if model.export_feature.startswith("text2text-generation"):
            models = [model.encoder, model.decoder]
            if model.decoder_with_past is not None:
                models.append(model.decoder_with_past)
        elif model.export_feature == "text-to-image":
            models = [model.unet, model.vae_encoder, model.vae_decoder]
            models.append(model.text_encoder if model_type in ["stable-diffusion", "sana"] else model.text_encoder_2)
        elif model_type == "open-clip":
            models = [model.text_model, model.visual_model]
        elif model.export_feature == "image-text-to-text":
            models = list(model.submodels.values())
        else:
            models = [model]

        for i, submodel in enumerate(models):
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
            "openvino.runtime.op.Constant.shape", new_callable=unittest.mock.PropertyMock
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
            "openvino.runtime.op.Constant.shape", new_callable=unittest.mock.PropertyMock
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
            "openvino.runtime.op.Constant.shape", new_callable=unittest.mock.PropertyMock
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

            submodels = []
            if isinstance(model, OVModelForCausalLM):
                submodels = [model.model]
            elif isinstance(model, OVModelForVisualCausalLM):
                submodels = list(model.submodels.values())
            check_compression_state_per_model(self, submodels, expected_num_weight_nodes_per_model)

            model.save_pretrained(tmp_dir)
            openvino_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(openvino_config.quantization_config.bits, 4)
            self.assertEqual(openvino_config.dtype, quantization_config.weight_format)


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


class OVTrainerTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (("albert", 61, 39),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    @unittest.skipIf(
        is_transformers_version(">=", "4.46"), reason="OVTrainer is not compatible with transformers>=v4.46"
    )
    def test_aware_training_quantization(self, model_name, expected_fake_nodes, expected_int8_nodes):
        model_id = MODEL_NAMES[model_name]
        model = AutoModelForSequenceClassification.from_pretrained(model_id, attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ov_config = OVConfig()
        dataset = load_dataset("glue", "sst2")
        dataset = dataset.map(
            lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
        )
        train_dataset = dataset["train"].select(range(16))
        eval_dataset = dataset["validation"].select(range(16))
        metric = evaluate.load("glue", "sst2")

        def compute_metrics(p):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

        with TemporaryDirectory() as tmp_dir:
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
            self.assertEqual(trainer.task, "text-classification")
            trainer.train()
            trainer.evaluate()
            trainer.save_model()

            model = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_fake_nodes, num_fake_nodes)
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)


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
        (OVDynamicQuantizationConfig(bits=8, sym=True),),
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

    def get_default_configurations() -> dict:
        default_configurations = deepcopy(_DEFAULT_4BIT_CONFIGS)
        default_configurations.update({"default": _DEFAULT_4BIT_CONFIG})
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
        prepared_config = OVModelForCausalLM._prepare_weight_quantization_config(custom_configuration)
        for field_name, reference_value in custom_configuration.items():
            value = prepared_config.__getattribute__(field_name)
            self.assertEqual(value, reference_value)

    def test_for_no_short_id_duplicates(self):
        short_ids = set()
        for model_id in _DEFAULT_4BIT_CONFIGS.keys():
            short_id = model_id.split("/")[1]
            assert short_id not in short_ids
            short_ids.add(short_id)


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

    def test_deepcopy_data(self):
        data = {
            "a": torch.tensor([1, 2, 3]),
            "b": np.array([1, 2, 3]),
            "c": 1,
            "d": "string",
            "e": {"a": torch.tensor([1, 2, 3]), "b": np.array([1, 2, 3])},
            "f": [ov.Tensor(np.ones((1, 2, 3)), (1, 2, 3), ov.Type.i4), ov.Tensor(np.ones((1, 2, 3)))],
        }
        copied_data = deepcopy_data(data)
        assert copied_data["a"] is not data["a"]
        assert copied_data["b"] is not data["b"]
        assert copied_data["e"]["a"] is not data["e"]["a"]
        assert copied_data["e"]["b"] is not data["e"]["b"]
        assert copied_data["f"][0] is not data["f"][0]
        assert copied_data["f"][1] is not data["f"][1]

        assert torch.equal(copied_data["a"], data["a"])
        assert np.array_equal(copied_data["b"], data["b"])
        assert copied_data["c"] == data["c"]
        assert copied_data["d"] == data["d"]
        assert torch.equal(copied_data["e"]["a"], data["e"]["a"])
        assert np.array_equal(copied_data["e"]["b"], data["e"]["b"])
        assert np.array_equal(copied_data["f"][0].data, data["f"][0].data)
        assert np.array_equal(copied_data["f"][1].data, data["f"][1].data)

        assert copied_data is not data


def check_optimization_not_applicable_to_optimized_model(model, quantization_config):
    quantizer = OVQuantizer(model)
    with pytest.raises(
        RuntimeError,
        match="Cannot apply optimization to the model because it was already optimized with the following config",
    ):
        quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config))
