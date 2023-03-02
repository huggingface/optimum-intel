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

import importlib.util

from ..utils.import_utils import is_diffusers_available, is_nncf_available
from .utils import OV_DECODER_NAME, OV_DECODER_WITH_PAST_NAME, OV_ENCODER_NAME, OV_XML_FILE_NAME


if is_nncf_available():
    from nncf.torch import patch_torch_operators

    patch_torch_operators()

    from .configuration import OVConfig
    from .quantization import OVQuantizer
    from .trainer import OVTrainer
    from .training_args import OVTrainingArguments

from .modeling import (
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
)
from .modeling_seq2seq import OVModelForSeq2SeqLM


if is_diffusers_available():
    from .modeling_diffusion import OVStableDiffusionPipeline
