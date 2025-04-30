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

from ..utils.import_utils import (
    is_diffusers_available,
    is_nncf_available,
    is_sentence_transformers_available,
)
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

    from .quantization import OVCalibrationDataset, OVQuantizer


from .configuration import (
    OVConfig,
    OVDynamicQuantizationConfig,
    OVMixedQuantizationConfig,
    OVQuantizationConfig,
    OVWeightQuantizationConfig,
)
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
    OVModelForZeroShotImageClassification,
)
from .modeling_decoder import OVModelForCausalLM
from .modeling_open_clip import (
    OVModelOpenCLIPForZeroShotImageClassification,
    OVModelOpenCLIPText,
    OVModelOpenCLIPVisual,
)
from .modeling_sam import OVSamModel
from .modeling_seq2seq import OVModelForPix2Struct, OVModelForSeq2SeqLM, OVModelForSpeechSeq2Seq, OVModelForVision2Seq
from .modeling_text2speech import OVModelForTextToSpeechSeq2Seq
from .modeling_visual_language import OVModelForVisualCausalLM


if is_diffusers_available():
    from .modeling_diffusion import (
        OVDiffusionPipeline,
        OVFluxFillPipeline,
        OVFluxImg2ImgPipeline,
        OVFluxInpaintPipeline,
        OVFluxPipeline,
        OVLatentConsistencyModelImg2ImgPipeline,
        OVLatentConsistencyModelPipeline,
        OVLTXPipeline,
        OVPipelineForImage2Image,
        OVPipelineForInpainting,
        OVPipelineForText2Image,
        OVPipelineForText2Video,
        OVSanaPipeline,
        OVSanaSprintPipeline,
        OVStableDiffusion3Img2ImgPipeline,
        OVStableDiffusion3InpaintPipeline,
        OVStableDiffusion3Pipeline,
        OVStableDiffusionImg2ImgPipeline,
        OVStableDiffusionInpaintPipeline,
        OVStableDiffusionPipeline,
        OVStableDiffusionXLImg2ImgPipeline,
        OVStableDiffusionXLInpaintPipeline,
        OVStableDiffusionXLPipeline,
    )


if is_sentence_transformers_available():
    from .modeling_sentence_transformers import OVSentenceTransformer
