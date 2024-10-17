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
import tempfile
import unittest
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Union

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

from optimum.intel import (
    OVConfig,
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
    OVQuantizer,
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
from copy import deepcopy

from optimum.intel.openvino.quantization import InferRequestWrapper
from optimum.intel.utils.import_utils import is_openvino_version, is_transformers_version
from utils_tests import MODEL_NAMES, get_num_quantized_nodes, _ARCHITECTURES_TO_EXPECTED_INT8

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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_TORCH_MODEL)
    def test_automodel_static_quantization(self, model_cls, model_name, expected_fake_quantize, expected_int8):
        model_id = MODEL_NAMES[model_name]
        task = model_cls.export_feature
        dataset_name, dataset_config_name, column_name = _TASK_TO_DATASET[task]
        file_name = "openvino_quantized_model.xml"

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples[column_name], padding="max_length", max_length=128, truncation=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
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
            num_fake_quantize, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())

    @parameterized.expand(SUPPORTED_ARCHITECTURES_OV_MODEL)
    def test_ovmodel_static_quantization(self, model_cls, model_name, expected_fake_quantize, expected_int8):
        model_id = MODEL_NAMES[model_name]
        task = model_cls.export_feature
        dataset_name, dataset_config_name, column_name = _TASK_TO_DATASET[task]

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples[column_name], padding="max_length", max_length=128, truncation=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
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

            num_fake_quantize, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())


class OVWeightCompressionTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS = (
        (OVModelForSequenceClassification, "bert", 70, 70),
        (OVModelForCausalLM, "gpt2", 44, 44),
    )

    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_COMPRESSED_MATMULS = ((OVModelForCausalLM, "opt125m", 62, 43),)
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_AUTOCOMPRESSED_MATMULS = ((OVModelForCausalLM, "opt125m", 0, 74),)
    SUPPORTED_ARCHITECTURES_STATEFUL_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS = ((OVModelForCausalLM, "gpt2", 44, 44),)

    LOAD_IN_4_BITS_SCOPE = (
        (OVModelForCausalLM, "gpt2", dict(bits=4, sym=False, group_size=-1, ratio=0.8), {"int4": 30, "int8": 14}),
        (
            OVModelForCausalLM,
            "gpt2",
            dict(bits=4, weight_format="mxfp4", group_size=32),
            {"f4e2m1": 20, "f8e8m0": 20, "int8": 4},
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            dict(
                bits=4,
                sym=False,
                group_size=32,
                ignored_scope={"names": ["__module.model.transformer.h.2.mlp.c_fc/aten::addmm/MatMul"]},
            ),
            {"int4": 38, "int8": 4},
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            dict(bits=4, sym=False, group_size=-1, ratio=0.8, all_layers=True),
            {"int4": 26, "int8": 18},
        ),
        (
            OVModelForCausalLM,
            "opt",
            dict(
                bits=4,
                sym=True,
                group_size=-1,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset="c4",
            ),
            {"int4": 25, "int8": 14},
        ),
        (
            OVModelForCausalLM,
            "opt",
            dict(
                bits=4,
                sym=True,
                group_size=-1,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset=["one two, " * i for i in range(10)],
            ),
            {"int4": 25, "int8": 14},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
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
            {"int4": 12, "int8": 8},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            dict(
                bits=4,
                sym=True,
                group_size=16,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset="c4",
                quant_method="awq",
            ),
            {"int4": 12, "int8": 8},
        ),
        (
            OVModelForCausalLM,
            "llama_awq",
            dict(
                bits=4,
                sym=True,
                group_size=16,
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                dataset="c4",
                gptq=True,
            ),
            {"int4": 12, "int8": 8},
        ),
    )

    SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION = (
        (OVModelForCausalLM, "gpt2"),
        (OVModelForMaskedLM, "bert"),
        (OVModelForTokenClassification, "roberta"),
        (OVModelForImageClassification, "vit"),
        (OVModelForSeq2SeqLM, "t5"),
        (OVModelForSequenceClassification, "albert"),
        (OVModelForQuestionAnswering, "distilbert"),
        (OVModelForAudioClassification, "wav2vec2"),
        (OVModelForFeatureExtraction, "blenderbot"),
        (OVStableDiffusionPipeline, "stable-diffusion"),
        (OVStableDiffusionXLPipeline, "stable-diffusion-xl"),
        (OVModelOpenCLIPForZeroShotImageClassification, "open-clip"),
        (OVModelForVisualCausalLM, "llava"),
    )

    SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION = (
        (OVStableDiffusionPipeline, "stable-diffusion", 72, 195),
        (OVStableDiffusionXLPipeline, "stable-diffusion-xl", 84, 331),
        (OVLatentConsistencyModelPipeline, "latent-consistency", 50, 135),
    )

    IS_SUPPORT_STATEFUL = is_openvino_version(">=", "2023.3")

    DEFAULT_INT4_CONFIG = {"bits": 4, "sym": True, "group_size": 64, "all_layers": True}

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS)
    def test_automodel_weight_compression(self, model_cls, model_name, expected_pt_int8, expected_ov_int8):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]

        with tempfile.TemporaryDirectory() as tmp_dir:
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS)
    def test_ovmodel_8bit_weight_compression(self, model_cls, model_name, expected_pt_int8, expected_ov_int8):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]

        with tempfile.TemporaryDirectory() as tmp_dir:
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_COMPRESSED_MATMULS)
    def test_ovmodel_4bit_weight_compression(self, model_cls, model_name, expected_int8, expected_int4):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]
        with tempfile.TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.from_pretrained(model_id, export=True, stateful=False)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task)
            ov_config = OVConfig(quantization_config=OVWeightQuantizationConfig(bits=4, sym=True, ratio=0.8))
            quantizer.quantize(save_directory=tmp_dir, ov_config=ov_config)
            model = model_cls.from_pretrained(tmp_dir)

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_int8, num_weight_nodes["int8"])
            self.assertEqual(expected_int4, num_weight_nodes["int4"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir)
            self.assertEqual(ov_config.quantization_config.to_dict(), loaded_config.quantization_config.to_dict())

    @parameterized.expand(SUPPORTED_ARCHITECTURES_STATEFUL_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS)
    def test_ovmodel_8bit_weight_compression_stateful(self, model_cls, model_name, expected_pt_int8, expected_ov_int8):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]
        with tempfile.TemporaryDirectory() as tmp_dir:
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION)
    def test_ovmodel_load_with_compressed_weights(self, model_cls, model_type):
        model = model_cls.from_pretrained(MODEL_NAMES[model_type], export=True, load_in_8bit=True, stateful=False)

        if model_type == "open-clip":
            self.assertEqual(model.text_model._openvino_config.quantization_config.bits, 8)
            self.assertEqual(model.text_model._openvino_config.dtype, "int8")
            self.assertEqual(model.visual_model._openvino_config.quantization_config.bits, 8)
            self.assertEqual(model.visual_model._openvino_config.dtype, "int8")
        else:
            self.assertEqual(model._openvino_config.quantization_config.bits, 8)
            self.assertEqual(model._openvino_config.dtype, "int8")

        if model.export_feature.startswith("text2text-generation"):
            models = [model.encoder, model.decoder, model.decoder_with_past]
        elif model.export_feature == "text-to-image":
            models = [model.unet, model.vae_encoder, model.vae_decoder]
            models.append(model.text_encoder if model_type == "stable-diffusion" else model.text_encoder_2)
        elif model_type == "open-clip":
            models = [model.text_model, model.visual_model]
        elif model.export_feature == "image-text-to-text":
            models = [model.lm_model, model.vision_embeddings_model, model.text_embeddings_model]
            models += [getattr(model, part) for part in model.additional_parts]
        else:
            models = [model]

        expected_ov_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
        for i, model in enumerate(models):
            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_ov_int8[i], num_weight_nodes["int8"])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION)
    def test_ovmodel_hybrid_quantization(self, model_cls, model_type, expected_num_fake_quantize, expected_ov_int8):
        model_id = MODEL_NAMES[model_type]
        quantization_config = OVWeightQuantizationConfig(bits=8, dataset="conceptual_captions", num_samples=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_cls.from_pretrained(model_id, export=True, quantization_config=quantization_config)

            num_fake_quantize, num_weight_nodes = get_num_quantized_nodes(model.unet)
            self.assertEqual(expected_num_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])
            self.assertEqual(0, num_weight_nodes["int4"])

            model.save_pretrained(tmp_dir)

    def test_stable_diffusion_with_weight_compression(self):
        int8_pipe = OVStableDiffusionPipeline.from_pretrained(model_id=MODEL_NAMES["stable-diffusion"], export=True)
        quantization_config = OVWeightQuantizationConfig(bits=8, quant_method=OVQuantizationMethod.DEFAULT)
        quantizer = OVQuantizer(int8_pipe)

        quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config))

        num_fake_quantize, num_weight_nodes = get_num_quantized_nodes(int8_pipe.unet)
        self.assertEqual(0, num_fake_quantize)
        self.assertEqual(242, num_weight_nodes["int8"])
        self.assertEqual(0, num_weight_nodes["int4"])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION[-1:])
    def test_ovmodel_hybrid_quantization_with_custom_dataset(
        self, model_cls, model_type, expected_num_fake_quantize, expected_ov_int8
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
        num_fake_quantize, num_weight_nodes = get_num_quantized_nodes(model.unet)
        self.assertEqual(expected_num_fake_quantize, num_fake_quantize)
        self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])
        self.assertEqual(0, num_weight_nodes["int4"])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_AUTOCOMPRESSED_MATMULS)
    @unittest.mock.patch.dict(
        "optimum.intel.openvino.configuration._DEFAULT_4BIT_CONFIGS", {"facebook/opt-125m": DEFAULT_INT4_CONFIG}
    )
    def test_ovmodel_4bit_auto_compression(self, model_cls, model_type, expected_ov_int8, expected_ov_int4):
        with tempfile.TemporaryDirectory() as tmp_dir:
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

    @parameterized.expand(LOAD_IN_4_BITS_SCOPE)
    def test_ovmodel_4bit_auto_compression_with_config(
        self, model_cls, model_name, quantization_config, expected_num_weight_nodes
    ):
        model_id = MODEL_NAMES[model_name]
        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = OVWeightQuantizationConfig.from_dict(quantization_config)
            model = model_cls.from_pretrained(model_id, export=True, quantization_config=quantization_config)
            if quantization_config.quant_method.lower() == "awq":
                # TODO: Check that AWQ was actually applied
                pass

            ov_model = model.model
            if model_cls == OVModelForVisualCausalLM:
                ov_model = model.lm_model

            _, num_weight_nodes = get_num_quantized_nodes(ov_model)
            expected_num_weight_nodes.update({k: 0 for k in set(num_weight_nodes) - set(expected_num_weight_nodes)})
            self.assertEqual(expected_num_weight_nodes, num_weight_nodes)
            model.save_pretrained(tmp_dir)

            wc_rt_info = ov_model.get_rt_info()["nncf"]["weight_compression"]
            self.assertEqual(quantization_config.quant_method.lower() == "awq", wc_rt_info["awq"].value == "True")
            self.assertEqual(
                quantization_config.scale_estimation or False, wc_rt_info["scale_estimation"].value == "True"
            )
            self.assertEqual(quantization_config.gptq or False, wc_rt_info["gptq"].value == "True")

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
        self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION)
    def test_ovmodel_load_with_uncompressed_weights(self, model_cls, model_type):
        model = model_cls.from_pretrained(MODEL_NAMES[model_type], export=True, load_in_8bit=False)
        if model.export_feature.startswith("text2text-generation"):
            models = [model.encoder, model.decoder, model.decoder_with_past]
        elif model.export_feature == "text-to-image":
            models = [model.unet, model.vae_encoder, model.vae_decoder]
            models.append(model.text_encoder if model_type == "stable-diffusion" else model.text_encoder_2)
        elif model_type == "open-clip":
            models = [model.text_model, model.visual_model]
        elif model.export_feature == "image-text-to-text":
            models = [model.lm_model, model.vision_embeddings_model, model.text_embeddings_model]
            models += [getattr(model, part) for part in model.additional_parts]
        else:
            models = [model]

        for i, model in enumerate(models):
            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(0, num_weight_nodes["int8"])

    def test_ovmodel_load_large_model_with_default_compressed_weights(self):
        def main_export_in_stacktrace(*args, **kwargs):
            # Compression was called from `main_export`
            self.assertTrue(inspect.stack()[5].function == "main_export")

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
                _ = OVModelForCausalLM.from_pretrained(
                    MODEL_NAMES["llama"], export=True, load_in_8bit=False, compile=False, use_cache=False
                )
                compress_weights_patch.assert_not_called()

    def test_ovmodel_load_large_model_with_additional_quantization_config(self):
        def main_export_not_in_stacktrace(*args, **kwargs):
            # Compression was not called from `main_export`
            self.assertTrue(all(frame_info.function != "main_export" for frame_info in inspect.stack()))

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
                }
                compress_weights_patch.assert_called_with(unittest.mock.ANY, **compression_params)

    @parameterized.expand(LOAD_IN_4_BITS_SCOPE)
    def test_ovmodel_4bit_dynamic_with_config(
        self, model_cls, model_name, quantization_config, expected_num_weight_nodes
    ):
        model_id = MODEL_NAMES[model_name]
        with tempfile.TemporaryDirectory() as tmp_dir:
            group_size = quantization_config.pop("group_size", 32)
            quantization_config = OVDynamicQuantizationConfig(
                weights_group_size=group_size, activations_group_size=group_size, **quantization_config
            )
            model = model_cls.from_pretrained(model_id, export=True, quantization_config=quantization_config)
            self.assertEqual(model.ov_config["DYNAMIC_QUANTIZATION_GROUP_SIZE"], str(group_size))
            self.assertEqual(model.ov_config["KV_CACHE_PRECISION"], "u8")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _, num_weight_nodes = get_num_quantized_nodes(model)
            expected_num_weight_nodes.update({k: 0 for k in set(num_weight_nodes) - set(expected_num_weight_nodes)})
            self.assertEqual(expected_num_weight_nodes, num_weight_nodes)
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

        with tempfile.TemporaryDirectory() as tmp_dir:
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
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (("albert", 64, 39),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_aware_training_quantization(self, model_name, expected_fake_quantize, expected_int8):
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
            self.assertEqual(trainer.task, "text-classification")
            trainer.train()
            trainer.evaluate()
            trainer.save_model()

            model = OVModelForSequenceClassification.from_pretrained(tmp_dir)
            num_fake_quantize, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_fake_quantize, num_fake_quantize)
            self.assertEqual(expected_int8, num_weight_nodes["int8"])

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
            dict(bits=4, fast_bias_correction=True, dataset="wikitext2"),
            OVWeightQuantizationConfig,
            "Can't determine type of OV quantization config",
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
            dict(bits=4, fast_bias_correction=True, dataset="wikitext2", weight_only=True),
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
        with tempfile.TemporaryDirectory() as tmp_dir:
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
    MODEL_ID = ("openai/whisper-tiny.en",)
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

    @parameterized.expand(itertools.product(MODEL_ID, APPLY_CACHING))
    def test_calibration_data_uniqueness(self, model_id, apply_caching):
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, compile=True)
        processor = AutoProcessor.from_pretrained(model_id)

        calibration_data = []
        ov_model.decoder_with_past.request = InferRequestWrapper(
            ov_model.decoder_with_past.request, calibration_data, apply_caching=apply_caching
        )
        for _ in range(2):
            input_features = self._generate_random_audio_data(processor)
            ov_model.generate(input_features, max_new_tokens=10, min_new_tokens=10)

        data_hashes_per_key = defaultdict(list)
        data_id_per_key = defaultdict(set)

        for inputs_dict in calibration_data:
            for k, v in inputs_dict.items():
                if k == "input_ids":
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
