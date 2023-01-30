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
MAX_ONNX_OPSET = 13
MIN_ONNX_QDQ_OPSET = 13

EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024


def use_external_data_format(num_parameters: int) -> bool:
    """
    Returns whether or not the model requires using external data format for the ONNX export
    Args:
        num_parameters: Number of parameter on the model
    Returns:
        True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
    """

    return compute_serialized_parameters_size(num_parameters, ParameterFormat.Float) >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT
