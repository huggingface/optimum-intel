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
from functools import partial
from typing import Union, Type

import pytest
import numpy as np
import torch
from PIL import Image
from parameterized import parameterized
import nncf
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    GenerationConfig,
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
    OVPipelineQuantizationConfig,
    OVQuantizationConfig,
    OVMixedQuantizationConfig,
    OVWeightQuantizationConfig,
    OVModelOpenCLIPForZeroShotImageClassification,
    OVModelForVisualCausalLM,
    OVSentenceTransformer,
    OVModelForZeroShotImageClassification,
    OVSamModel,
)
from optimum.intel.openvino.configuration import (
    OVQuantizationMethod,
    OVQuantizationConfigBase,
    _DEFAULT_4BIT_WQ_CONFIGS,
    _DEFAULT_4BIT_WQ_CONFIG,
    _quantization_config_from_dict,
    _GPTOSSQuantizationConfig,
)
from optimum.intel.openvino.modeling_visual_language import _OVNanoLlavaForCausalLM
from optimum.intel.openvino.utils import TemporaryDirectory
from copy import deepcopy

from optimum.intel.openvino.quantization import InferRequestWrapper, OVCalibrationDatasetBuilder
from optimum.intel.utils.import_utils import is_transformers_version, is_nncf_version
from utils_tests import (
    MODEL_NAMES,
    get_num_quantized_nodes,
    _ARCHITECTURES_TO_EXPECTED_INT8,
    check_compression_state_per_model,
    get_supported_model_for_library,
    TEST_NAME_TO_MODEL_TYPE,
    OPENVINO_DEVICE,
)

_TASK_TO_DATASET = {
    "text-generation": {
        "dataset_name": "Salesforce/wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "column_name": "text",
    },
    "feature-extraction": {
        "dataset_name": "Salesforce/wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "column_name": "text",
    },
    "fill-mask": {
        "dataset_name": "Salesforce/wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "column_name": "text",
    },
    "text-classification": {
        "dataset_name": "nyu-mll/glue",
        "dataset_config_name": "sst2",
        "column_name": "sentence",
    },
    "zero-shot-image-classification": {
        "dataset_name": "google-research-datasets/conceptual_captions",
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
    maxDiff = None

    # TODO (nikita-savelyevv): Extend for OVModelForSpeechSeq2Seq, OVStableDiffusionPipeline and OVModelForSeq2SeqLM
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
            OVModelForCausalLM,
            "llama",
            dict(
                dataset="wikitext2:seq_len=64",
                num_samples=1,
                dtype="f8e4m3",
            ),
            {
                "model": 15,
            },
            {
                "model": {"f8e4m3": 16},
            },
        ),
        (
            OVModelForCausalLM,
            "llama",
            dict(
                weight_quantization_config=dict(
                    bits=4,
                    dtype="cb4",
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
                "model": 8,
            },
            {
                "model": {"int8": 2, "int4": 2, "f8e4m3": 10},
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
                "model": 16,
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
                dataset="gsm8k",
                num_samples=1,
            ),
            {
                "model": 15,
            },
            {
                "model": {"f8e5m2": 2, "int4": 28},
            },
        ),
        (
            OVStableDiffusionXLPipeline,
            "stable-diffusion-xl",
            dict(
                dtype="f8e5m2",
                dataset="laion/220k-GPT4Vision-captions-from-LIVIS",
                num_samples=1,
                processor=MODEL_NAMES["stable-diffusion-xl"],
            ),
            {
                "unet": 198,
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
            ),
            {
                "unet": 87,
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
                dataset="wikitext2:seq_len=64",
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
                dataset="wikitext2:seq_len=64",
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
            "xlm-roberta",
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
                dataset="conceptual_captions:seq_len=64",
                num_samples=1,
            ),
            {
                "model": 65,
            },
            {
                "model": {"int8": 65},
            },
        ),
        (
            OVModelForSeq2SeqLM,
            "t5",
            OVQuantizationConfig(
                dtype="int8",
                dataset="wikitext2:seq_len=64",
                num_samples=1,
            ),
            {"encoder": 30, "decoder": 52, "decoder_with_past": 61}
            if is_transformers_version("<=", "4.45")
            else {
                "encoder": 30,
                "decoder": 52,
            },
            (
                {"encoder": {"int8": 32}, "decoder": {"int8": 52}, "decoder_with_past": {"int8": 42}}
                if is_transformers_version("<=", "4.45")
                else {"encoder": {"int8": 32}, "decoder": {"int8": 52}}
            ),
        ),
        (
            OVSamModel,
            "sam",
            OVQuantizationConfig(bits=8, dataset="coco", num_samples=1),
            {
                "vision_encoder": 75,
                "prompt_encoder_mask_decoder": 60,
            },
            {
                "vision_encoder": {"int8": 75},
                "prompt_encoder_mask_decoder": {"int8": 49},
            },
        ),
        (
            OVModelForVisualCausalLM,
            "qwen2_vl",
            OVQuantizationConfig(
                bits=8,
                dataset="contextual",
                num_samples=1,
            ),
            {
                "lm_model": 13,
                "text_embeddings_model": 0,
                "vision_embeddings_model": 1,
                "vision_embeddings_merger_model": 14,
            },
            {
                "lm_model": {"int8": 15},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 1},
                "vision_embeddings_merger_model": {"int8": 10},
            },
        ),
        (
            OVModelForVisualCausalLM,
            "qwen2_vl",
            OVMixedQuantizationConfig(
                weight_quantization_config=OVWeightQuantizationConfig(bits=4, group_size=16, ratio=0.7),
                full_quantization_config=OVQuantizationConfig(dtype="f8e4m3", smooth_quant_alpha=0.9),
                dataset="contextual",
                num_samples=1,
            ),
            {
                "lm_model": 16,
                "text_embeddings_model": 0,
                "vision_embeddings_model": 1,
                "vision_embeddings_merger_model": 16,
            },
            {
                "lm_model": {"f8e4m3": 8, "int4": 14},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"f8e4m3": 1},
                "vision_embeddings_merger_model": {"f8e4m3": 2, "int4": 16},
            },
        ),
    ]

    if is_transformers_version(">=", "4.57.0"):
        SUPPORTED_ARCHITECTURES_OV_MODEL_WITH_AUTO_DATASET.extend(
            [
                (
                    OVModelForVisualCausalLM,
                    "qwen3_vl",
                    OVQuantizationConfig(
                        bits=8,
                        dataset="contextual",
                        num_samples=1,
                    ),
                    {
                        "lm_model": 14,
                        "text_embeddings_model": 0,
                        "vision_embeddings_model": 1,
                        "vision_embeddings_merger_model": 44,
                        "vision_embeddings_pos_model": 0,
                    },
                    {
                        "lm_model": {"int8": 15},
                        "text_embeddings_model": {"int8": 1},
                        "vision_embeddings_model": {"int8": 1},
                        "vision_embeddings_merger_model": {"int8": 32},
                        "vision_embeddings_pos_model": {"int8": 1},
                    },
                ),
            ]
        )

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
                streaming=streaming,
            )
        return calibration_dataset

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

            quantizer = OVQuantizer.from_pretrained(ov_model, task=task, device=OPENVINO_DEVICE)

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
            loaded_config = OVConfig.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
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

            check_model_inference(ov_model, model_id, trust_remote_code=False)

            if model_cls in [OVModelForSpeechSeq2Seq, OVModelForSeq2SeqLM] and ov_model.decoder_with_past is None:
                expected_fake_nodes_per_model.pop("decoder_with_past", None)
                expected_num_weight_nodes_per_model.pop("decoder_with_past", None)
            check_compression_state_per_model(
                self,
                ov_model.ov_models,
                expected_num_weight_nodes_per_model,
                expected_fake_nodes_per_model,
            )


class OVWeightCompressionTest(unittest.TestCase):
    maxDiff = None

    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS = (
        (OVModelForSequenceClassification, "bert", 70, 70),
        (OVModelForCausalLM, "gpt2", 44, 44),
    )

    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_COMPRESSED_MATMULS = ((OVModelForCausalLM, "opt125m", 62, 43),)
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_4BIT_AUTOCOMPRESSED_MATMULS = ((OVModelForCausalLM, "opt125m", 0, 74),)
    SUPPORTED_ARCHITECTURES_STATEFUL_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS = ((OVModelForCausalLM, "gpt2", 44, 44),)

    TRANSFORMERS_4BIT_CONFIGURATIONS = [
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
            dict(bits=4, dtype="cb4", group_size=32),
            {"model": {"int8": 24, "int4": 20, "f8e4m3": 20}},
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(
                bits=4,
                sym=False,
                group_size=32,
                ignored_scope={
                    "names": [
                        "__module.model.transformer.h.2.mlp.c_fc/aten::addmm/MatMul"
                        if is_transformers_version("<", "4.57")
                        else "__module.transformer.h.2.mlp.c_fc/aten::addmm/MatMul"
                    ]
                },
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
            {"model": {"int8": 18, "int4": 23}},
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
            {"model": {"int8": 18, "int4": 23}},
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
                quant_method=QuantizationMethod.AWQ,
            ),
            {"model": {"int8": 4, "int4": 14}},
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
        (
            OVSamModel,
            "sam",
            False,
            dict(bits=4, dataset="coco", num_samples=1, group_size=2),
            {
                "vision_encoder": {"int8": 56, "int4": 94},
                "prompt_encoder_mask_decoder": {"int8": 6, "int4": 92},
            },
        ),
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
            "llava-qwen2",
            True,
            dict(
                bits=4,
                group_size=8,
                dataset="contextual",
                ratio=0.8,
                sensitivity_metric="mean_activation_variance",
                num_samples=1,
                processor=MODEL_NAMES["nanollava_vision_tower"],
                tokenizer=MODEL_NAMES["llava-qwen2"],
            ),
            {
                "lm_model": {"int8": 16, "int4": 14},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 15},
            },
        ),
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
            "internvl_chat",
            True,
            dict(
                bits=4,
                group_size=4,
                dataset="contextual",
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                num_samples=1,
            ),
            {
                "lm_model": {"int8": 8, "int4": 22},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 11},
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
        (
            OVModelForVisualCausalLM,
            "qwen3_vl",
            False,
            dict(
                bits=4,
                group_size=8,
                dataset="contextual",
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                num_samples=1,
            ),
            {
                "lm_model": {"int8": 12, "int4": 18},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 1},
                "vision_embeddings_merger_model": {"int8": 32},
                "vision_embeddings_pos_model": {"int8": 1},
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
            "qwen2_5_vl",
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
                "vision_embeddings_merger_model": {"int8": 12},
            },
        ),
        (
            OVModelForVisualCausalLM,
            "llama4",
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
                "lm_model": {"int8": 46, "int4": 56},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 16},
            },
        ),
        (
            OVModelForVisualCausalLM,
            "minicpmo",
            True,
            dict(
                bits=4,
                group_size=4,
                dataset="contextual",
                ratio=0.8,
                sensitivity_metric="mean_activation_magnitude",
                num_samples=1,
                processor=MODEL_NAMES["minicpmo"],
            ),
            {
                "lm_model": {"int8": 6, "int4": 10},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 8},
                "resampler_model": {"int8": 6},
            },
        ),
        (
            OVModelForCausalLM,
            "exaone4",
            True,
            dict(bits=4, sym=False, group_size=32, ratio=1.0),
            {"model": {"int8": 2, "int4": 14}},
        ),
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(bits=4, sym=True, group_size_fallback="adjust"),
            {"model": {"int8": 4, "int4": 20}},
        ),
        (
            OVModelForCausalLM,
            "llama",
            False,
            dict(
                bits=4,
                sym=True,
                group_size_fallback="adjust",
            ),
            {"model": {"int8": 28, "int4": 2}},
        ),
        (
            OVModelForCausalLM,
            "llama",
            False,
            dict(
                bits=4,
                sym=True,
                group_size_fallback="ignore",
            ),
            {"model": {"int8": 4}},
        ),
    ]

    # filter models type depending on min max transformers version
    LOAD_IN_4_BITS_SCOPE = [
        config
        for config in TRANSFORMERS_4BIT_CONFIGURATIONS
        if TEST_NAME_TO_MODEL_TYPE.get(config[1], config[1]) in get_supported_model_for_library("transformers")
    ]

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
        (OVModelForVisualCausalLM, "llava_next_video", False),
        (OVModelForVisualCausalLM, "minicpmv", True),
        (OVModelForVisualCausalLM, "qwen2_vl", False),
    ]

    if is_transformers_version("<", "4.54.0"):
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForVisualCausalLM, "llava-qwen2", True))

    if is_transformers_version("<", "4.52.0"):
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForVisualCausalLM, "minicpmo", True))

    if is_transformers_version(">=", "4.54.0"):
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForCausalLM, "exaone4", True))

    if is_transformers_version(">=", "4.57.0"):
        SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION.append((OVModelForVisualCausalLM, "qwen3_vl", False))

    SUPPORTED_ARCHITECTURES_WITH_HYBRID_QUANTIZATION = [
        (OVStableDiffusionPipeline, "stable-diffusion", 72, 195),
        (OVStableDiffusionXLPipeline, "stable-diffusion-xl", 84, 331),
        (OVLatentConsistencyModelPipeline, "latent-consistency", 50, 135),
        (OVStableDiffusion3Pipeline, "stable-diffusion-3", 9, 65),
        (OVFluxPipeline, "flux", 7, 56),
        (OVSanaPipeline, "sana", 19, 53),
    ]

    DEFAULT_COMPRESSION_CONFIGURATIONS = [
        (OVModelForCausalLM, "llama", 8, {"bits": 8, "dq_group_size": 128}, {"model": {"int8": 32}}),
        (
            OVModelForCausalLM,
            "llama",
            4,
            {"bits": 4, "group_size": 8, "ratio": 0.5, "dq_group_size": 64},
            {"model": {"int8": 26, "int4": 6}},
        ),
        (
            OVModelForFeatureExtraction,
            "llama",
            4,
            {"bits": 4, "group_size": 8, "ratio": 0.5},
            {"model": {"int8": 22, "int4": 8}},
        ),
        (
            OVStableDiffusionPipeline,
            "stable-diffusion",
            4,
            {"quantization_configs": {"unet": {"bits": 4, "group_size": -1, "ratio": 0.5, "dq_group_size": 64}}},
            {
                "unet": {"int8": 182, "int4": 60},
                "vae_decoder": {},
                "vae_encoder": {},
                "text_encoder": {},
            },
        ),
        (
            OVModelForVisualCausalLM,
            "llava",
            4,
            {"bits": 4, "group_size": 8, "ratio": 0.5},
            {
                "lm_model": {"int8": 22, "int4": 8},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 9},
            },
        ),
        (
            OVSamModel,
            "sam",
            4,
            {"bits": 4, "group_size": 8, "ratio": 0.5},
            {
                "vision_encoder": {"int8": 112, "int4": 38},
                "prompt_encoder_mask_decoder": {"int8": 94, "int4": 4},
            },
        ),
        (
            OVModelForSpeechSeq2Seq,
            "whisper",
            4,
            {"bits": 4, "group_size": 8, "ratio": 0.5},
            {
                "decoder": {"int8": 40, "int4": 4},
                "encoder": {"int8": 24, "int4": 4},
            },
        ),
    ]

    DEFAULT_IGNORED_SCOPE_CONFIGURATIONS = [
        (
            OVModelForCausalLM,
            "llama",
            {
                "model": {
                    "names": ["__module.model.layers.1.self_attn.v_proj/ov_ext::linear/MatMul"],
                    "patterns": ["__module.model.layers.\\d.self_attn.o_proj/ov_ext::linear/MatMul"],
                }
            },
        ),
        (
            OVModelForFeatureExtraction,
            "llama",
            {
                "model": {
                    "names": ["__module.layers.1.self_attn.v_proj/aten::linear/MatMul"],
                    "patterns": ["__module.layers.\\d.self_attn.o_proj/aten::linear/MatMul"],
                }
            },
        ),
        (
            OVStableDiffusionPipeline,
            "stable-diffusion",
            {
                "unet": {"names": ["__module.time_embedding.linear_1/aten::linear/MatMul"]},
                "text_encoder": {
                    "names": ["__module.text_model.encoder.layers.0.self_attn.q_proj/aten::linear/MatMul"]
                },
            },
        ),
        (
            OVModelForVisualCausalLM,
            "llava",
            {
                "lm_model": {"patterns": [".*layers.0.self_attn.q_proj/aten::linear/MatMul"]},
                "vision_embeddings_model": {"patterns": [".*layers.0.self_attn.q_proj/aten::linear/MatMul"]},
                "text_embeddings_model": {"patterns": ["."]},
            },
        ),
        (
            OVSamModel,
            "sam",
            {
                "prompt_encoder_mask_decoder": {
                    "names": ["__module.model.prompt_encoder.shared_embedding/aten::matmul/MatMul"]
                },
                "vision_encoder": {"names": ["__module.vision_encoder.layers.0.attn.qkv/aten::linear/MatMul"]},
            },
        ),
        (
            OVModelForSpeechSeq2Seq,
            "whisper",
            {
                "encoder": {"patterns": [".*layers.0.self_attn.q_proj/aten::linear/MatMul"]},
                "decoder": {"patterns": [".*layers.0.encoder_attn.k_proj/aten::linear/MatMul"]},
            },
        ),
    ]

    def test_filtered_architectures(cls):
        expected = set()
        if is_transformers_version("<", "4.49"):
            expected.add("qwen2_5_vl")
        if is_transformers_version("<", "4.51"):
            expected.add("llama4")
        if is_transformers_version("<", "4.54"):
            expected.add("exaone4")
        if is_transformers_version("<", "4.57"):
            expected.add("qwen3_vl")
        if is_transformers_version(">=", "4.54"):
            expected.update({"llava-qwen2", "phi3_v", "minicpmo"})

        all_model_type = {config[1] for config in cls.TRANSFORMERS_4BIT_CONFIGURATIONS}
        filtered_model_type = {config[1] for config in cls.LOAD_IN_4_BITS_SCOPE}
        skipped = all_model_type - filtered_model_type
        cls.assertEqual(skipped, expected)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_8BIT_COMPRESSED_MATMULS)
    def test_ovmodel_8bit_weight_compression(self, model_cls, model_name, expected_pt_int8, expected_ov_int8):
        task = model_cls.export_feature
        model_id = MODEL_NAMES[model_name]

        with TemporaryDirectory() as tmp_dir:
            transformers_model = model_cls.from_pretrained(model_id, export=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task, device=OPENVINO_DEVICE)
            quantizer.quantize(save_directory=tmp_dir)
            model = model_cls.from_pretrained(tmp_dir)

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
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

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task, device=OPENVINO_DEVICE)
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
            loaded_config = OVConfig.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
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

            quantizer = OVQuantizer.from_pretrained(transformers_model, task=task, device=OPENVINO_DEVICE)
            quantizer.quantize(save_directory=tmp_dir)
            model = model_cls.from_pretrained(tmp_dir)

            _, num_weight_nodes = get_num_quantized_nodes(model)
            self.assertEqual(expected_ov_int8, num_weight_nodes["int8"])

            tokens = tokenizer("This is a sample input", return_tensors="pt")
            outputs = model(**tokens)
            self.assertTrue("logits" in outputs)

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
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
        ref_config = OVWeightQuantizationConfig(bits=8, sym=isinstance(model, OVModelForVisualCausalLM)).to_dict()

        if model_type == "open-clip":
            self.assertEqual(
                model.text_model._openvino_config.quantization_config.default_config.to_dict(), ref_config
            )
            self.assertEqual(
                model.visual_model._openvino_config.quantization_config.default_config.to_dict(), ref_config
            )
        else:
            actual_config = model._openvino_config.quantization_config.default_config.to_dict()
            actual_config["tokenizer"] = actual_config["processor"] = None
            self.assertEqual(actual_config, ref_config)

        if model_type != "open-clip":  # ticket 161043
            check_optimization_not_applicable_to_optimized_model(model, quantization_config={"bits": 8})

        expected_ov_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
        expected_ov_int8 = {k: {"int8": v} for k, v in expected_ov_int8.items()}
        check_compression_state_per_model(self, model.ov_models, expected_ov_int8)

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
        int8_pipe = OVStableDiffusionPipeline.from_pretrained(
            model_id=MODEL_NAMES["stable-diffusion"], export=True, device=OPENVINO_DEVICE
        )
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

    @parameterized.expand(DEFAULT_COMPRESSION_CONFIGURATIONS)
    def test_ovmodel_default_compression(
        self, model_cls, model_type, bits, default_config, expected_num_weight_nodes_per_model
    ):
        with unittest.mock.patch.dict(
            f"optimum.intel.openvino.configuration._DEFAULT_{bits}BIT_WQ_CONFIGS",
            {MODEL_NAMES[model_type]: default_config},
            clear=False,
        ):
            model = model_cls.from_pretrained(MODEL_NAMES[model_type], export=True, quantization_config={"bits": bits})
            check_compression_state_per_model(self, model.ov_models, expected_num_weight_nodes_per_model)

            # Check that dynamic quantization group size is correctly set in the runtime info
            if isinstance(default_config, dict):
                default_config = _quantization_config_from_dict(default_config)
            ref_dq_data = []
            if isinstance(default_config, OVWeightQuantizationConfig) and default_config.dq_group_size is not None:
                ref_dq_data = [("model", default_config.dq_group_size)]
            elif isinstance(default_config, OVPipelineQuantizationConfig):
                ref_dq_data = []
                for ov_model_name, q_config in default_config.quantization_configs.items():
                    if isinstance(q_config, OVWeightQuantizationConfig) and q_config.dq_group_size is not None:
                        ref_dq_data.append((ov_model_name, q_config.dq_group_size))
            for ov_model_name, ref_dq_group_size in ref_dq_data:
                rt_info = model.ov_models[ov_model_name].get_rt_info()
                runtime_options = rt_info["runtime_options"]
                self.assertIsInstance(
                    runtime_options,
                    dict,
                    "Runtime options are not found in the runtime info",
                )
                dq_group_size = runtime_options.get("DYNAMIC_QUANTIZATION_GROUP_SIZE", None)
                if dq_group_size is None:
                    self.fail("DYNAMIC_QUANTIZATION_GROUP_SIZE is not found in the runtime options")
                self.assertEqual(
                    ref_dq_group_size,
                    int(dq_group_size.value),
                    f"Dynamic quantization group size {dq_group_size.value} does not match expected {ref_dq_group_size}",
                )

    @parameterized.expand(DEFAULT_IGNORED_SCOPE_CONFIGURATIONS)
    def test_ovmodel_default_ignored_scope(self, model_cls, model_type, expected_ignored_scope_per_model):
        with unittest.mock.patch.dict(
            "optimum.intel.openvino.configuration._DEFAULT_IGNORED_SCOPE_CONFIGS",
            {MODEL_NAMES[model_type]: expected_ignored_scope_per_model},
            clear=False,
        ):
            with TemporaryDirectory() as tmp_dir:
                model_id = MODEL_NAMES[model_type]
                model = model_cls.from_pretrained(
                    model_id,
                    export=True,
                    quantization_config={"bits": 8},
                )
                model.save_pretrained(tmp_dir)

                model = model_cls.from_pretrained(tmp_dir)
                for ov_model_name, expected_ignored_scope in expected_ignored_scope_per_model.items():
                    rt_info = model.ov_models[ov_model_name].get_rt_info()
                    nncf_info = rt_info["nncf"]
                    quantization_info = nncf_info["weight_compression"]

                    self.assertIsInstance(
                        quantization_info["ignored_scope"],
                        dict,
                        "Ignored scope is not found in the runtime info",
                    )

                    ignored_scope = {k: eval(v.value) for k, v in quantization_info["ignored_scope"].items()}
                    self.assertEqual(
                        expected_ignored_scope,
                        ignored_scope,
                        f"Ignored scope {ignored_scope} does not match expected {expected_ignored_scope}",
                    )

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
            ref_quantization_config = model._openvino_config.quantization_config
            if quantization_config.quant_method.lower() == "awq":
                # TODO: Check that AWQ was actually applied
                pass

            check_compression_state_per_model(self, model.ov_models, expected_num_weight_nodes_per_model)

            model.save_pretrained(tmp_dir)
            model = model_cls.from_pretrained(tmp_dir, trust_remote_code=trust_remote_code)
            check_model_inference(model, model_id, trust_remote_code)

            # At the moment the first model in the list is the only one we apply data-aware compression to
            wc_rt_info = next(iter(model.ov_models.values())).get_rt_info()["nncf"]["weight_compression"]
            self.assertEqual(quantization_config.quant_method.lower() == "awq", wc_rt_info["awq"].value == "True")
            self.assertEqual(
                quantization_config.scale_estimation or False, wc_rt_info["scale_estimation"].value == "True"
            )
            self.assertEqual(quantization_config.gptq or False, wc_rt_info["gptq"].value == "True")
            self.assertEqual(
                quantization_config.lora_correction or False, wc_rt_info["lora_correction"].value == "True"
            )

            openvino_config = OVConfig.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
            self.assertEqual(openvino_config.quantization_config.to_dict(), ref_quantization_config.to_dict())

    @parameterized.expand(((OVModelForCausalLM, "gpt2"),))
    def test_ovmodel_stateful_load_with_compressed_weights(self, model_cls, model_type):
        model = model_cls.from_pretrained(MODEL_NAMES[model_type], export=True, load_in_8bit=True, stateful=True)
        self.assertTrue(model.stateful)
        self.assertTrue(model.use_cache)

        _, num_weight_nodes = get_num_quantized_nodes(model)
        expected_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
        expected_int8 = {k: {"int8": v} for k, v in expected_int8.items()}
        check_compression_state_per_model(self, model.ov_models, expected_int8)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION)
    def test_ovmodel_load_with_uncompressed_weights(self, model_cls, model_type, trust_remote_code):
        model = model_cls.from_pretrained(
            MODEL_NAMES[model_type], export=True, load_in_8bit=False, trust_remote_code=trust_remote_code
        )

        for i, ov_model in enumerate(model.ov_models.values()):
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
            self.assertTrue(inspect.stack()[6].function == "main_export")
            return compressed_model_mock_obj

        with unittest.mock.patch(
            "openvino.op.Constant.shape", new_callable=unittest.mock.PropertyMock
        ) as ov_constant_shape:
            ov_constant_shape.return_value = (2000000000,)
            with unittest.mock.patch(
                "nncf.compress_weights", side_effect=main_export_in_stacktrace
            ) as compress_weights_patch:
                _ = OVModelForCausalLM.from_pretrained(
                    MODEL_NAMES["llama"], export=True, compile=False, use_cache=False, device=OPENVINO_DEVICE
                )
                compression_params = {
                    "mode": nncf.CompressWeightsMode.INT8_ASYM,
                    "ratio": 1.0,
                    "group_size": None,
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
                    MODEL_NAMES["llama"],
                    export=True,
                    load_in_8bit=False,
                    compile=False,
                    use_cache=False,
                    device=OPENVINO_DEVICE,
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

    @parameterized.expand([(MODEL_NAMES["gpt2"],)])
    def test_dataset_seq_len_option(self, model_id):
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False)
        dataset_builder = OVCalibrationDatasetBuilder(model)
        dataset = dataset_builder.build_from_quantization_config(
            OVWeightQuantizationConfig(
                bits=4,
                dataset="c4:seq_len=64",
                tokenizer=model_id,
                num_samples=1,
            ),
        )
        self.assertTrue(all(len(sample["input_ids"][0]) == 64 for sample in dataset["model"].get_data()))


class OVPipelineQuantizationTest(unittest.TestCase):
    maxDiff = None

    PIPELINE_QUANTIZATION_SCOPE = [
        (
            OVModelForCausalLM,
            "gpt2",
            False,
            dict(quantization_configs={"model": dict(bits=8, weight_only=True)}),
            {"model": 0},
            {"model": {"int8": 44}},
        ),
        (
            OVModelForCausalLM,
            "llama",
            False,
            dict(
                quantization_configs={
                    "model": dict(
                        weight_quantization_config=dict(
                            bits=4,
                            dtype="cb4",
                            group_size=16,
                            dataset="wikitext2",
                            num_samples=1,
                            gptq=True,
                            ratio=0.5,
                        ),
                        full_quantization_config=dict(dtype="f8e4m3"),
                        dataset="wikitext2",
                        num_samples=1,
                    ),
                }
            ),
            {
                "model": 16,
            },
            {
                "model": {"f8e4m3": 16, "int4": 5, "int8": 5},
            },
        ),
        (
            OVStableDiffusionPipeline,
            "stable-diffusion",
            True,
            dict(
                quantization_configs={
                    "unet": dict(
                        dtype="f8e4m3",
                        dataset="laion/filtered-wit",
                        num_samples=1,
                    ),
                    "vae_decoder": OVWeightQuantizationConfig(),
                    "vae_encoder": OVWeightQuantizationConfig(),
                    "text_encoder": OVWeightQuantizationConfig(),
                }
            ),
            {
                "unet": 124,
                "vae_decoder": 0,
                "vae_encoder": 0,
                "text_encoder": 0,
            },
            {
                "unet": {"f8e4m3": 121},
                "vae_decoder": {"int8": 42},
                "vae_encoder": {"int8": 34},
                "text_encoder": {"int8": 64},
            },
        ),
        (
            OVSamModel,
            "sam",
            False,
            dict(
                quantization_configs={
                    "vision_encoder": dict(
                        bits=8,
                        dataset="coco",
                        num_samples=1,
                        weight_only=False,
                    ),
                }
            ),
            {
                "vision_encoder": 75,
                "prompt_encoder_mask_decoder": 0,
            },
            {
                "vision_encoder": {"int8": 75},
                "prompt_encoder_mask_decoder": {"int8": 0},
            },
        ),
        (
            OVStableDiffusion3Pipeline,
            "stable-diffusion-3",
            False,
            dict(
                quantization_configs={
                    "transformer": dict(
                        dataset="conceptual_captions",
                        num_samples=1,
                        quant_method=OVQuantizationMethod.HYBRID,
                    ),
                    "vae_decoder": OVWeightQuantizationConfig(),
                    "vae_encoder": OVWeightQuantizationConfig(),
                    "text_encoder": OVWeightQuantizationConfig(),
                }
            ),
            {
                "transformer": 9,
                "vae_decoder": 0,
                "vae_encoder": 0,
                "text_encoder": 0,
                "text_encoder_2": 0,
                "text_encoder_3": 0,
            },
            {
                "transformer": {"int8": 65},
                "vae_decoder": {"int8": 58},
                "vae_encoder": {"int8": 42},
                "text_encoder": {"int8": 30},
                "text_encoder_2": {"int8": 0},
                "text_encoder_3": {"int8": 0},
            },
        ),
        (
            OVModelForSpeechSeq2Seq,
            "whisper",
            True,
            dict(
                quantization_configs={
                    "encoder": dict(smooth_quant_alpha=0.95),
                    "decoder": dict(smooth_quant_alpha=0.9),
                },
                dataset="librispeech",
                num_samples=1,
                processor=MODEL_NAMES["whisper"],
            ),
            {"encoder": 14, "decoder": 22},
            {"encoder": {"int8": 14}, "decoder": {"int8": 22}},
        ),
        (
            OVModelForVisualCausalLM,
            "internvl_chat",
            True,
            dict(
                quantization_configs={
                    "lm_model": dict(bits=8, weight_only=True),
                    "vision_embeddings_model": dict(bits=8, weight_only=False),
                },
                dataset="contextual",
                num_samples=1,
                default_config=dict(bits=8, sym=True, weight_only=True),
            ),
            {
                "lm_model": 0,
                "text_embeddings_model": 0,
                "vision_embeddings_model": 15,
            },
            {
                "lm_model": {"int8": 30},
                "text_embeddings_model": {"int8": 1},
                "vision_embeddings_model": {"int8": 11},
            },
        ),
    ]

    if is_transformers_version(">=", "4.49.0") and is_transformers_version("<", "4.54.0"):
        PIPELINE_QUANTIZATION_SCOPE.extend(
            [
                (
                    OVModelForVisualCausalLM,
                    "phi4mm",
                    True,
                    dict(
                        quantization_configs={
                            "lm_model": dict(
                                bits=4,
                                group_size=16,
                                dataset="contextual",
                                num_samples=1,
                                ratio=0.8,
                                sensitivity_metric="mean_activation_magnitude",
                                quant_method=OVQuantizationMethod.AWQ,
                                scale_estimation=True,
                                lora_correction=True,
                                ignored_scope={
                                    "patterns": [
                                        "__module\\.model\\.layers\\.\\d+\\.(mlp\\.(gate_up_proj|down_proj)|self_attn\\."
                                        "(qkv_proj|o_proj))\\.lora_B\\.speech/aten::linear/MatMul",
                                    ],
                                },
                            ),
                            "text_embeddings_model": dict(bits=8, sym=True, weight_only=True),
                            "audio_encoder_model": dict(bits=8, sym=True, weight_only=True),
                            "vision_embeddings_model": dict(bits=8, sym=True, weight_only=True),
                        },
                    ),
                    {
                        "lm_model": 0,
                        "text_embeddings_model": 0,
                        "audio_encoder_model": 0,
                        "vision_embeddings_model": 0,
                        "vision_projection_model": 0,
                        "audio_embeddings_model": 0,
                        "audio_forward_embeddings_model": 0,
                        "audio_vision_projection_model": 0,
                        "audio_speech_projection_model": 0,
                    },
                    {
                        "lm_model": {"int8": 60, "int4": 26},
                        "text_embeddings_model": {"int8": 1},
                        "audio_encoder_model": {"int8": 25},
                        "vision_embeddings_model": {"int8": 8},
                        "vision_projection_model": {},
                        "audio_embeddings_model": {},
                        "audio_forward_embeddings_model": {},
                        "audio_vision_projection_model": {},
                        "audio_speech_projection_model": {},
                    },
                ),
            ]
        )

    @parameterized.expand(PIPELINE_QUANTIZATION_SCOPE)
    def test_ovmodel_pipeline_quantization(
        self,
        model_cls,
        model_name,
        trust_remote_code,
        quantization_config,
        expected_fake_nodes_per_model,
        expected_num_weight_nodes_per_model,
    ):
        def eval_expression_if_possible(expression):
            try:
                return eval(expression)
            except NameError:
                return expression

        model_id = MODEL_NAMES[model_name]
        with TemporaryDirectory() as tmp_dir:
            quantization_config = OVPipelineQuantizationConfig.from_dict(quantization_config)
            model = model_cls.from_pretrained(
                model_id, export=True, quantization_config=quantization_config, trust_remote_code=trust_remote_code
            )
            for save_load_model in [False, True]:
                if save_load_model:
                    model.save_pretrained(tmp_dir)
                    model = model_cls.from_pretrained(tmp_dir, trust_remote_code=trust_remote_code)
                check_model_inference(model, model_id, trust_remote_code)
                check_compression_state_per_model(
                    self, model.ov_models, expected_num_weight_nodes_per_model, expected_fake_nodes_per_model
                )
                # Compare the quantization config with the model runtime info
                for ov_model_name, ov_model in model.ov_models.items():
                    rt_info = ov_model.get_rt_info()
                    config = quantization_config.quantization_configs.get(
                        ov_model_name, quantization_config.default_config
                    )
                    if config is None:
                        self.assertTrue("nncf" not in rt_info)
                        continue

                    if isinstance(config, OVWeightQuantizationConfig):
                        sub_configs = [config]
                        rt_info_keys = ["weight_compression"]
                    elif isinstance(config, OVQuantizationConfig):
                        sub_configs = [config]
                        rt_info_keys = ["quantization"]
                    elif isinstance(config, OVMixedQuantizationConfig):
                        sub_configs = [config.weight_quantization_config, config.full_quantization_config]
                        rt_info_keys = ["weight_compression", "quantization"]
                    else:
                        raise ValueError(f"Unsupported config type: {type(config)}")

                    for sub_config, rt_info_key in zip(sub_configs, rt_info_keys):
                        q_rt_info = rt_info["nncf"][rt_info_key]
                        config_dict = sub_config.to_nncf_dict()
                        for param_name in q_rt_info:
                            if sub_config.num_samples is None and param_name == "subset_size":
                                # Skip subset_size check because num_samples was not explicitly provided
                                continue
                            rt_info_value = q_rt_info[param_name]
                            if isinstance(rt_info_value, dict):
                                # For example, ignored scope case
                                rt_info_value_ = {}
                                for k, v in rt_info_value.items():
                                    rt_info_value_[k] = eval_expression_if_possible(v.value)
                                rt_info_value = rt_info_value_
                            else:
                                rt_info_value = eval_expression_if_possible(rt_info_value.value)

                            if param_name not in config_dict:
                                continue
                            config_value = config_dict[param_name]
                            if param_name == "advanced_parameters":
                                from nncf.quantization.advanced_parameters import convert_to_dict_recursively

                                config_value = convert_to_dict_recursively(config_value)
                            if param_name == "ignored_scope":
                                if sub_config.quant_method == OVQuantizationMethod.HYBRID:
                                    # For hybrid quantization ignored scope is set dynamically
                                    config_value = {"types": ["Convolution"]}
                                else:
                                    from nncf.openvino.rt_info import exclude_empty_fields

                                    config_value = exclude_empty_fields(dataclasses.asdict(config_value))
                                    config_value = [] if config_value == {} else config_value
                            if param_name == "backup_mode" and config_value is None:
                                config_value = "int8_asym"
                            if param_name == "sensitivity_metric" and config_value is None:
                                config_value = (
                                    "max_activation_variance" if sub_config.bits == 4 else "weight_quantization_error"
                                )
                            if param_name == "group_size" and config_value is None:
                                config_value = -1 if sub_config.bits == 8 else 128

                            if config_value is None and rt_info_value is False:
                                continue
                            if param_name == "subset_size":
                                self.assertGreaterEqual(
                                    rt_info_value,
                                    config_value,
                                    f"Actual subset size should not be less than the requested one.",
                                )
                            else:
                                self.assertEqual(
                                    config_value, rt_info_value, f"Mismatch in {param_name} for {ov_model_name}"
                                )


class OVQuantizerQATest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("hf-internal-testing/tiny-random-BertForQuestionAnswering",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_ovmodel_static_quantization(self, model_name):
        def preprocess_function(examples, tokenizer):
            return tokenizer(
                examples["question"], examples["context"], padding="max_length", max_length=64, truncation=True
            )

        with TemporaryDirectory() as tmp_dir:
            transformers_model = OVModelForQuestionAnswering.from_pretrained(
                model_name, export=True, device=OPENVINO_DEVICE
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            quantizer = OVQuantizer.from_pretrained(transformers_model, device=OPENVINO_DEVICE)
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
            model = OVModelForQuestionAnswering.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
            tokens = tokenizer.encode_plus(
                "This is a sample question", "This is a sample context", add_special_tokens=True, return_tensors="pt"
            )
            model(**tokens, return_dict=True)

            # Test loading model a second time to catch issues with caching
            try:
                model = OVModelForQuestionAnswering.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
            except RuntimeError:
                self.fail("Loading BERT QA model a second time failed")

            # Verify that the configuration is correctly saved and loaded
            loaded_config = OVConfig.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
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
        (
            OVMixedQuantizationConfig(
                weight_quantization_config=OVWeightQuantizationConfig(
                    bits=4,
                    dtype="cb4",
                    group_size=16,
                    ratio=0.5,
                    ignored_scope={"patterns": [f"{pattern_prefix}.layers.0.self_attn"]},
                ),
                full_quantization_config=OVQuantizationConfig(
                    dtype="f8e4m3", ignored_scope={"patterns": [f"{pattern_prefix}.layers.0.mlp"]}
                ),
                ignored_scope={"patterns": [f"{pattern_prefix}.layers.1.self_attn"]},
                dataset="gsm8k",
                num_samples=1,
            ),
        ),
        (
            OVPipelineQuantizationConfig(
                quantization_configs={
                    "model1": OVQuantizationConfig(bits=8, dataset="wikitext2"),
                    "model2": OVWeightQuantizationConfig(bits=4, group_size=16),
                    "model3": OVMixedQuantizationConfig(
                        weight_quantization_config=OVWeightQuantizationConfig(bits=4, dtype="cb4"),
                        full_quantization_config=OVQuantizationConfig(dtype="f8e4m3", dataset="wikitext2"),
                    ),
                }
            ),
        ),
        (
            OVQuantizationConfig(
                advanced_parameters=nncf.AdvancedCompressionParameters(),
            ),
        ),
        (
            _GPTOSSQuantizationConfig(
                quantization_config1=OVWeightQuantizationConfig(bits=4, group_size=16),
                quantization_config2=OVWeightQuantizationConfig(bits=8),
            ),
        ),
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
        (dict(bits=8), OVWeightQuantizationConfig, "Can't determine type of OV quantization config"),
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
        (dict(bits=8, weight_only=False), OVQuantizationConfig, None),
        (dict(bits=8, weight_only=True), OVWeightQuantizationConfig, None),
        (
            dict(bits=8, fast_bias_correction=True, dataset="librispeech"),
            OVQuantizationConfig,
            None,
        ),
        (
            dict(bits=4, dataset="wikitext2"),
            OVWeightQuantizationConfig,
            None,
        ),
        (
            dict(bits=4, dataset="gsm8k"),
            OVWeightQuantizationConfig,
            None,
        ),
        (dict(bits=8, fast_bias_correction=True), OVQuantizationConfig, None),
        (
            dict(
                weight_quantization_config=dict(bits=4, dtype="cb4", group_size=16, ratio=0.5),
                full_quantization_config=dict(dtype="f8e4m3"),
                dataset="wikitext2",
                num_samples=1,
            ),
            OVMixedQuantizationConfig,
            None,
        ),
        (
            dict(
                quantization_configs=dict(
                    model1=dict(bits=8, dataset="wikitext2", weight_only=False),
                    model2=dict(bits=4, group_size=16),
                    model3=dict(
                        weight_quantization_config=dict(bits=4, dtype="cb4"),
                        full_quantization_config=dict(dtype="f8e4m3", dataset="wikitext2"),
                    ),
                )
            ),
            OVPipelineQuantizationConfig,
            None,
        ),
        (
            dict(
                quantization_config1=dict(bits=4, group_size=16),
                quantization_config2=dict(bits=8, weight_only=True),
            ),
            _GPTOSSQuantizationConfig,
            None,
        ),
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
            OVWeightQuantizationConfig,
            {
                "advanced_parameters": nncf.AdvancedCompressionParameters(statistics_path="statistics_path"),
                "statistics_path": "statistics_path2",
            },
            {
                "advanced_parameters": nncf.AdvancedCompressionParameters(statistics_path="statistics_path2"),
            },
        ),
        (
            OVWeightQuantizationConfig,
            {
                "statistics_path": "statistics_path",
            },
            {
                "advanced_parameters": nncf.AdvancedCompressionParameters(statistics_path="statistics_path"),
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
            loaded_ov_config = OVConfig.from_pretrained(tmp_dir, device=OPENVINO_DEVICE)
            self.compare_objects(ov_config.quantization_config, loaded_ov_config.quantization_config)

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

        self.compare_config_dict_to_config_object(quantization_config, ov_config.quantization_config)

    @parameterized.expand(DEFAULT_CONFIGURATIONS)
    def test_named_default_configurations(self, config_id: str):
        custom_configuration = self.DEFAULT_CONFIGURATIONS[config_id]
        prepared_config = _quantization_config_from_dict(custom_configuration)
        self.compare_config_dict_to_config_object(custom_configuration, prepared_config)

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
            mock_model.set_rt_info = unittest.mock.Mock(return_value={})

            mock_quantization_config = unittest.mock.Mock(config_type)
            mock_quantization_config.to_nncf_dict.return_value = {"param1": "value1", "param2": "value2"}
            mock_quantization_config.dq_group_size = None

            additional_kwargs = {"param2": "new_value2", "param3": "value3"}

            quantization_function = (
                _weight_only_quantization
                if quantization_function == "_weight_only_quantization"
                else _full_quantization
            )
            quantization_function(mock_model, mock_quantization_config, None, **additional_kwargs)

            expected_kwargs = {"param1": "value1", "param2": "new_value2", "param3": "value3", dataset_key: None}

            mock_method.assert_called_once_with(mock_model, **expected_kwargs)

    def compare_config_dict_to_config_object(self, config_dict: dict, config_obj: OVQuantizationConfigBase):
        if "quantization_configs" in config_dict:
            for k, v in config_dict["quantization_configs"].items():
                self.compare_config_dict_to_config_object(v, getattr(config_obj, "quantization_configs")[k])
            return
        for k, v in config_dict.items():
            if hasattr(config_obj, k):
                config_v = getattr(config_obj, k)
                if isinstance(config_v, OVQuantizationConfigBase):
                    self.compare_config_dict_to_config_object(v, config_v)
                else:
                    self.assertEqual(v, config_v)

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
    STATEFUL = (False, True)
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

    @parameterized.expand(itertools.product(MODEL_NAME, STATEFUL, APPLY_CACHING))
    def test_calibration_data_uniqueness(self, model_name, stateful, apply_caching):
        model_id = MODEL_NAMES[model_name]
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, compile=True, stateful=stateful, device=OPENVINO_DEVICE
        )
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
        n_samples = 3
        for _ in range(n_samples):
            input_features = self._generate_random_audio_data(processor)
            ov_model.generate(input_features, max_new_tokens=10, min_new_tokens=10)

        data_hashes_per_key = defaultdict(list)
        data_id_per_key = defaultdict(set)

        # Check that reset state flag is present and correctly set in collected inputs
        if stateful and is_nncf_version(">", "2.19"):
            from nncf.definitions import NNCF_DATASET_RESET_STATE_KEY

            # All inputs should have reset state key
            self.assertTrue(all(NNCF_DATASET_RESET_STATE_KEY in inputs_dict for inputs_dict in calibration_data))
            # The number of times reset state flag is set to True should be equal to (2 * n_samples), because
            # for each sequence generation, the state is reset twice
            self.assertEqual(
                sum(int(inputs_dict[NNCF_DATASET_RESET_STATE_KEY]) for inputs_dict in calibration_data), 2 * n_samples
            )
            # Remove reset state key from inputs to avoid affecting data uniqueness checks
            [input_dict.pop(NNCF_DATASET_RESET_STATE_KEY) for input_dict in calibration_data]

        for inputs_dict in calibration_data:
            for k, v in inputs_dict.items():
                if k in ["input_ids", "beam_idx"]:
                    continue

                x = (v.numpy() if isinstance(v, torch.Tensor) else v).copy()
                data_hashes_per_key[k].append(hash(x.tobytes()))
                data_id_per_key[k].add(id(v))
        for k, data_hashes in data_hashes_per_key.items():
            # All hashes can not be equal because calibration dataset contains at least n_samples different samples
            self.assertTrue(any(data_hashes[0] != it for it in data_hashes))
        if apply_caching:
            # With caching, encoder hidden states tensors should be cached, resulting in only n_samples tensors stored
            self.assertEqual(len(data_id_per_key["encoder_hidden_states"]), n_samples)
        else:
            # Without caching, encoder hidden states tensors will be unique for each collected input
            self.assertGreater(len(data_id_per_key["encoder_hidden_states"]), n_samples)


def check_optimization_not_applicable_to_optimized_model(model, quantization_config):
    quantizer = OVQuantizer(model)
    with pytest.raises(
        RuntimeError,
        match="Cannot apply optimization to the model because it was already optimized with the following config",
    ):
        quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config))


def check_model_inference(ov_model, model_id, trust_remote_code):
    if isinstance(ov_model, (OVModelForSpeechSeq2Seq, OVModelForSeq2SeqLM)):
        gen_config = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=10,
            num_beams=2,
            do_sample=False,
            eos_token_id=None,
        )
        if isinstance(ov_model, OVModelForSpeechSeq2Seq):
            input_features = torch.randn((1, ov_model.config.num_mel_bins, 3000), dtype=torch.float32)
            generate_kwrgs = {}
            if is_transformers_version(">=", "4.50"):
                generate_kwrgs = {"use_model_defaults": False}
            ov_model.generate(input_features, generation_config=gen_config, **generate_kwrgs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            inputs = tokenizer("This is a sample <mask>", return_tensors="pt")
            ov_model.generate(**inputs, generation_config=gen_config)
    elif isinstance(ov_model, (OVModelForCausalLM, OVModelForFeatureExtraction, OVModelForMaskedLM)):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer("This is a sample <mask>", return_tensors="pt")
        ov_model(**tokens)
    elif isinstance(
        ov_model,
        (
            OVStableDiffusionPipeline,
            OVStableDiffusion3Pipeline,
            OVStableDiffusionXLPipeline,
            OVLatentConsistencyModelPipeline,
        ),
    ):
        ov_model(prompt="A text-to-image prompt")
    elif isinstance(ov_model, OVSentenceTransformer):
        ov_model.encode(["This is a sample input"])
    elif isinstance(ov_model, OVModelForZeroShotImageClassification):
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        image = np.random.rand(224, 224, 3).astype(np.uint8)
        inputs = processor(text=["This is a sample text"], images=image, return_tensors="pt")
        ov_model(**inputs)
    elif isinstance(ov_model, OVModelForVisualCausalLM):
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor_id = config.mm_vision_tower if isinstance(ov_model, _OVNanoLlavaForCausalLM) else model_id
        processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        image = Image.fromarray(np.random.rand(224, 224, 3).astype(np.uint8))
        inputs = ov_model.preprocess_inputs(
            image=image, text="This is a sample text", processor=processor, tokenizer=tokenizer, config=config
        )
        ov_model(**inputs)
    elif isinstance(ov_model, OVSamModel):
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        image = np.random.rand(224, 224, 3).astype(np.uint8)
        inputs = processor(image, input_points=[[[0, 0]]], return_tensors="pt")
        ov_model(**inputs)
    else:
        raise Exception("Unexpected model class.")


class TestDatasetParsing(unittest.TestCase):
    """Test suite for dataset option parsing in OVQuantizationConfigBase."""

    def test_dataset_no_options(self):
        """Test that a simple dataset name without options is preserved."""
        config = OVQuantizationConfigBase(dataset="wikitext")
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config._dataset_kwargs, {})

    def test_dataset_with_seq_len_option(self):
        """Test parsing of seq_len option from dataset string."""
        config = OVQuantizationConfigBase(dataset="wikitext2:seq_len=128")
        for _ in range(2):
            self.assertEqual(config.dataset, "wikitext2")
            self.assertEqual(config._dataset_kwargs, {"seq_len": 128})
            config = _quantization_config_from_dict(config.to_dict())

    def test_dataset_with_seq_len_option_mixed_q_config(self):
        """Test parsing of seq_len option from dataset string."""
        config = OVMixedQuantizationConfig(
            OVWeightQuantizationConfig(dataset="wikitext2:seq_len=128"), OVQuantizationConfig()
        )
        for _ in range(2):
            self.assertEqual(config.dataset, "wikitext2")
            self.assertEqual(config._dataset_kwargs, {"seq_len": 128})
            config = _quantization_config_from_dict(config.to_dict())

    def test_dataset_with_seq_len_option_pipeline_q_config(self):
        """Test parsing of seq_len option from dataset string."""
        config = OVPipelineQuantizationConfig({"model": OVWeightQuantizationConfig(dataset="wikitext2:seq_len=128")})
        for _ in range(2):
            self.assertEqual(config.dataset, "wikitext2")
            self.assertEqual(config._dataset_kwargs, {"seq_len": 128})
            config = _quantization_config_from_dict(config.to_dict())

    def test_dataset_gsm8k_with_seq_len(self):
        """Test parsing of seq_len option for gsm8k dataset."""
        config = OVQuantizationConfigBase(dataset="gsm8k:seq_len=512")
        self.assertEqual(config.dataset, "gsm8k")
        self.assertEqual(config._dataset_kwargs, {"seq_len": 512})

    def test_dataset_with_multiple_spaces(self):
        """Test parsing with spaces around the option."""
        config = OVQuantizationConfigBase(dataset="wikitext:seq_len = 64")
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config._dataset_kwargs, {"seq_len": 64})

    def test_dataset_list_no_parsing(self):
        """Test that list datasets skip parsing and remain unchanged."""
        dataset_list = ["sample text 1", "sample text 2", "sample text 3"]
        config = OVQuantizationConfigBase(dataset=dataset_list)
        self.assertEqual(config.dataset, dataset_list)
        self.assertEqual(config._dataset_kwargs, {})

    def test_dataset_unsupported_option(self):
        """Test that unsupported options raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:foo=bar")
        self.assertIn("Unsupported dataset option 'foo'", str(exc_info.value))
        self.assertIn("Only 'seq_len' is supported", str(exc_info.value))

    def test_dataset_malformed_option_no_equals(self):
        """Test that options without '=' raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:seq_len")
        self.assertIn("Malformed dataset option", str(exc_info.value))
        self.assertIn("Expected format: 'key=value'", str(exc_info.value))

    def test_dataset_invalid_seq_len_value(self):
        """Test that non-integer seq_len values raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:seq_len=abc")
        self.assertIn("Invalid value 'abc' for seq_len", str(exc_info.value))
        self.assertIn("Expected an integer", str(exc_info.value))

    def test_dataset_empty_string_option(self):
        """Test that empty seq_len value raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            OVQuantizationConfigBase(dataset="wikitext:seq_len=")
        self.assertIn("Invalid value '' for seq_len", str(exc_info.value))

    def test_dataset_none(self):
        """Test that None dataset is handled correctly."""
        config = OVQuantizationConfigBase(dataset=None)
        self.assertIsNone(config.dataset)
        self.assertEqual(config._dataset_kwargs, {})

    def test_dataset_with_colon_in_name_only(self):
        """Test handling of dataset string with trailing colon but no options."""
        config = OVQuantizationConfigBase(dataset="wikitext:")
        self.assertEqual(config.dataset, "wikitext")
        self.assertEqual(config._dataset_kwargs, {})
