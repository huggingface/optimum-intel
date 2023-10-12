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
import os
from glob import glob

import numpy as np
from huggingface_hub import model_info
from openvino.runtime import Type
from transformers.onnx.utils import ParameterFormat, compute_serialized_parameters_size


OV_XML_FILE_NAME = "openvino_model.xml"
OV_ENCODER_NAME = "openvino_encoder_model.xml"
OV_DECODER_NAME = "openvino_decoder_model.xml"
OV_DECODER_WITH_PAST_NAME = "openvino_decoder_with_past_model.xml"

ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model.onnx"

MAX_ONNX_OPSET_2022_2_0 = 10
MAX_ONNX_OPSET = 16
MIN_ONNX_QDQ_OPSET = 13

EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024

TEXTUAL_INVERSION_NAME = "learned_embeds.bin"
TEXTUAL_INVERSION_NAME_SAFE = "learned_embeds.safetensors"
TEXTUAL_INVERSION_EMBEDDING_KEY = "text_model.embeddings.token_embedding.weight"


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


STR_TO_OV_TYPE = {
    "boolean": Type.boolean,
    "f16": Type.f16,
    "f32": Type.f32,
    "f64": Type.f64,
    "i8": Type.i8,
    "i16": Type.i16,
    "i32": Type.i32,
    "i64": Type.i64,
    "u8": Type.u8,
    "u16": Type.u16,
    "u32": Type.u32,
    "u64": Type.u64,
    "bf16": Type.bf16,
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
}


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
