#  Copyright 2022 The HuggingFace Team. All rights reserved.
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


import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Tuple, Type, Union

import numpy as np
import torch
from huggingface_hub import model_info
from openvino.runtime import Core, Model, properties
from openvino.runtime import Type as OVType
from packaging.version import Version
from transformers import AutoTokenizer, CLIPTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.onnx.utils import ParameterFormat, compute_serialized_parameters_size

from optimum.intel.utils.import_utils import is_torch_version


logger = logging.getLogger(__name__)

OV_XML_FILE_NAME = "openvino_model.xml"
OV_ENCODER_NAME = "openvino_encoder_model.xml"
OV_DECODER_NAME = "openvino_decoder_model.xml"
OV_DECODER_WITH_PAST_NAME = "openvino_decoder_with_past_model.xml"

OV_TOKENIZER_NAME = "openvino_tokenizer{}.xml"
OV_DETOKENIZER_NAME = "openvino_detokenizer{}.xml"

ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model.onnx"

MAX_ONNX_OPSET_2022_2_0 = 10
MAX_ONNX_OPSET = 16
MIN_ONNX_QDQ_OPSET = 13

EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024

TEXTUAL_INVERSION_EMBEDDING_KEY = "self.text_model.embeddings.token_embedding.weight"

OV_TO_NP_TYPE = {
    "boolean": np.bool_,
    "i8": np.int8,
    "u8": np.uint8,
    "i16": np.int16,
    "u16": np.uint16,
    "i32": np.int32,
    "u32": np.uint32,
    "i64": np.int64,
    "u64": np.uint64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}

OV_TO_PT_TYPE = {
    "boolean": torch.bool,
    "i8": torch.int8,
    "u8": torch.uint8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "f16": torch.float16,
    "f32": torch.float32,
    "f64": torch.float64,
}

if is_torch_version(">=", "2.4.0"):
    OV_TO_PT_TYPE.update({"u16": torch.uint16, "u32": torch.uint32, "u64": torch.uint64})


STR_TO_OV_TYPE = {
    "boolean": OVType.boolean,
    "f16": OVType.f16,
    "f32": OVType.f32,
    "f64": OVType.f64,
    "i8": OVType.i8,
    "i16": OVType.i16,
    "i32": OVType.i32,
    "i64": OVType.i64,
    "u8": OVType.u8,
    "u16": OVType.u16,
    "u32": OVType.u32,
    "u64": OVType.u64,
    "bf16": OVType.bf16,
}


_HEAD_TO_AUTOMODELS = {
    "feature-extraction": "OVModelForFeatureExtraction",
    "fill-mask": "OVModelForMaskedLM",
    "text-generation": "OVModelForCausalLM",
    "text2text-generation": "OVModelForSeq2SeqLM",
    "text-classification": "OVModelForSequenceClassification",
    "token-classification": "OVModelForTokenClassification",
    "question-answering": "OVModelForQuestionAnswering",
    "image-classification": "OVModelForImageClassification",
    "audio-classification": "OVModelForAudioClassification",
    "stable-diffusion": "OVStableDiffusionPipeline",
    "stable-diffusion-xl": "OVStableDiffusionXLPipeline",
    "pix2struct": "OVModelForPix2Struct",
    "latent-consistency": "OVLatentConsistencyModelPipeline",
    "open_clip_text": "OVModelOpenCLIPText",
    "open_clip_vision": "OVModelOpenCLIPVisual",
    "open_clip": "OVModelOpenCLIPForZeroShotImageClassification",
}


PREDEFINED_SD_DATASETS = {
    "conceptual_captions": {"split": "train", "inputs": {"prompt": "caption"}},
    "laion/220k-GPT4Vision-captions-from-LIVIS": {"split": "train", "inputs": {"prompt": "caption"}},
    "laion/filtered-wit": {"split": "train", "inputs": {"prompt": "caption"}},
}


NEED_CONVERT_TO_FAST_TOKENIZER: Tuple[Type[PreTrainedTokenizer]] = (CLIPTokenizer,)


def maybe_convert_tokenizer_to_fast(
    hf_tokenizer: PreTrainedTokenizer, tokenizer_path: Path
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if isinstance(hf_tokenizer, PreTrainedTokenizerFast):
        return hf_tokenizer

    if isinstance(hf_tokenizer, NEED_CONVERT_TO_FAST_TOKENIZER):
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            return hf_tokenizer

    return hf_tokenizer


def use_external_data_format(num_parameters: int) -> bool:
    """
    Returns whether or not the model requires using external data format for the ONNX export
    Args:
        num_parameters: Number of parameter on the model
    Returns:
        True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
    """

    return compute_serialized_parameters_size(num_parameters, ParameterFormat.Float) >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT


def _is_timm_ov_dir(model_dir):
    config_file = None
    has_xml = False
    has_bin = False
    if os.path.isdir(model_dir):
        for filename in glob(os.path.join(model_dir, "*")):
            if filename.endswith(".xml"):
                has_xml = True
            if filename.endswith(".bin"):
                has_bin = True
            if filename.endswith("config.json"):
                config_file = filename
    if config_file and has_xml and has_bin:
        with open(config_file) as conf:
            hf_hub_id = json.load(conf).get("hf_hub_id", None)
        if hf_hub_id and model_info(hf_hub_id).library_name == "timm":
            return True
    return False


def _print_compiled_model_properties(compiled_model):
    supported_properties = properties.supported_properties()
    skip_keys = {"SUPPORTED_METRICS", "SUPPORTED_CONFIG_KEYS", supported_properties}
    keys = set(compiled_model.get_property(supported_properties)) - skip_keys
    for k in keys:
        try:
            value = compiled_model.get_property(k)
            if k == properties.device.properties():
                for device_key in value.keys():
                    logger.info(f"  {device_key}:")
                    for k2, value2 in value.get(device_key).items():
                        if k2 not in skip_keys:
                            logger.info(f"    {k2}: {value2}")
            else:
                logger.info(f"  {k}: {value}")
        except Exception:
            logger.error(f"[error] Get property of '{k}' failed")
    try:
        logger.info("EXECUTION_DEVICES:")
        for device in compiled_model.get_property("EXECUTION_DEVICES"):
            logger.info(f"  {device}: {Core().get_property(device, 'FULL_DEVICE_NAME')}")
    except Exception:
        logger.error("[error] Get FULL_DEVICE_NAME failed")


def np_to_pt_generators(np_object, device):
    if isinstance(np_object, np.random.RandomState):
        return torch.Generator(device=device).manual_seed(int(np_object.get_state()[1][0]))
    elif isinstance(np_object, np.random.Generator):
        return torch.Generator(device=device).manual_seed(int(np_object.bit_generator.state[1][0]))
    elif isinstance(np_object, list) and isinstance(np_object[0], (np.random.RandomState, np.random.Generator)):
        return [np_to_pt_generators(a, device) for a in np_object]
    elif isinstance(np_object, dict) and isinstance(
        next(iter(np_object.values())), (np.random.RandomState, np.random.Generator)
    ):
        return {k: np_to_pt_generators(v, device) for k, v in np_object.items()}
    else:
        return np_object


def _raise_invalid_batch_size(
    expected_batch_size: int, batch_size: int, num_images_per_prompt: int, guidance_scale: float
):
    current_batch_size = batch_size * num_images_per_prompt * (1 if guidance_scale <= 1 else 2)

    if expected_batch_size != current_batch_size:
        msg = ""
        if guidance_scale is not None and guidance_scale <= 1:
            msg = f"`guidance_scale` was set to {guidance_scale}, static shapes are currently only supported for `guidance_scale` > 1 "

        raise ValueError(
            "The model was statically reshaped and the pipeline inputs do not match the expected shapes. "
            f"The `batch_size`, `num_images_per_prompt` and `guidance_scale` were respectively set to {batch_size}, {num_images_per_prompt} and {guidance_scale}. "
            f"The static model expects an input of size equal to {expected_batch_size} and got the following value instead : {current_batch_size}. "
            f"To fix this, please either provide a different inputs to your model so that `batch_size` * `num_images_per_prompt` * 2 is equal to {expected_batch_size} "
            "or reshape it again accordingly using the `.reshape()` method by setting `batch_size` to -1. " + msg
        )


def get_export_transformers_version(model, config):
    version_str = None

    if isinstance(model, Model):
        if "optimum" in model.rt_info:
            version_str = model.rt_info["optimum"]["transformers_version"].value
    if version_str is None:
        version_str = getattr(config, "transformers_version", "0.0.0")

    version_str = version_str or "0.0.0"

    return Version(version_str)


def model_has_dynamic_inputs(model):
    is_dynamic = False
    for input in model.inputs:
        is_dynamic = input.get_partial_shape().is_dynamic
        if is_dynamic:
            return is_dynamic
    return is_dynamic
