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

import logging
import warnings

from ..utils.import_utils import is_accelerate_available, is_diffusers_available, is_nncf_available
from .utils import (
    OV_DECODER_NAME,
    OV_DECODER_WITH_PAST_NAME,
    OV_DETOKENIZER_NAME,
    OV_ENCODER_NAME,
    OV_TOKENIZER_NAME,
    OV_XML_FILE_NAME,
)


warnings.simplefilter(action="ignore", category=FutureWarning)


if is_nncf_available():
    logging.disable(logging.INFO)
    import nncf

    logging.disable(logging.NOTSET)

    # Suppress version mismatch logging
    nncf.set_log_level(logging.ERROR)
    from nncf.torch import patch_torch_operators

    nncf.set_log_level(logging.INFO)

    patch_torch_operators()

    from .quantization import OVQuantizer
    from .training_args import OVTrainingArguments

    if is_accelerate_available():
        from .trainer import OVTrainer


from .configuration import OVConfig, OVDynamicQuantizationConfig, OVQuantizationConfig, OVWeightQuantizationConfig
from .modeling import (
    OVModelForAudioClassification,
    OVModelForAudioFrameClassification,
    OVModelForAudioXVector,
    OVModelForCTC,
    OVModelForCustomTasks,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
)
from .modeling_decoder import OVModelForCausalLM
from .modeling_seq2seq import OVModelForPix2Struct, OVModelForSeq2SeqLM, OVModelForSpeechSeq2Seq, OVModelForVision2Seq


if is_diffusers_available():
    from .modeling_diffusion import (
        OVLatentConsistencyModelPipeline,
        OVStableDiffusionImg2ImgPipeline,
        OVStableDiffusionInpaintPipeline,
        OVStableDiffusionPipeline,
        OVStableDiffusionXLImg2ImgPipeline,
        OVStableDiffusionXLPipeline,
    )
