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
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule

from .utils import (
    is_accelerate_available,
    is_diffusers_available,
    is_ipex_available,
    is_neural_compressor_available,
    is_nncf_available,
    is_openvino_available,
    is_sentence_transformers_available,
)
from .version import __version__


_import_structure = {
    "openvino": [],
    "utils.dummy_openvino_and_nncf_objects": [],
}

try:
    if not is_ipex_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_ipex_objects

    _import_structure["utils.dummy_ipex_objects"] = [
        "IPEXModelForCausalLM",
        "IPEXModelForSequenceClassification",
        "IPEXModelForMaskedLM",
        "IPEXModelForTokenClassification",
        "IPEXModelForQuestionAnswering",
        "IPEXModelForImageClassification",
        "IPEXModelForAudioClassification",
        "IPEXModel",
    ]
else:
    _import_structure["utils.dummy_ipex_objects"] = []
    _import_structure["ipex"] = [
        "IPEXModelForCausalLM",
        "IPEXModelForSeq2SeqLM",
        "IPEXModelForSequenceClassification",
        "IPEXModelForMaskedLM",
        "IPEXModelForTokenClassification",
        "IPEXModelForQuestionAnswering",
        "IPEXModelForImageClassification",
        "IPEXModelForAudioClassification",
        "IPEXModel",
    ]

try:
    if not (is_ipex_available() and is_sentence_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_ipex_objects"].extend(["IPEXSentenceTransformer"])
else:
    _import_structure["ipex"].extend(["IPEXSentenceTransformer"])


try:
    if not (is_openvino_available() and is_nncf_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_openvino_and_nncf_objects"].extend(
        [
            "OVQuantizer",
            "OVCalibrationDataset",
            "OVQuantizationConfig",
            "OVWeightQuantizationConfig",
            "OVDynamicQuantizationConfig",
            "OVMixedQuantizationConfig",
        ]
    )
else:
    _import_structure["openvino"].extend(
        [
            "OVQuantizer",
            "OVCalibrationDataset",
            "OVQuantizationConfig",
            "OVWeightQuantizationConfig",
            "OVDynamicQuantizationConfig",
            "OVMixedQuantizationConfig",
        ]
    )


try:
    if not (is_openvino_available() and is_diffusers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_openvino_and_diffusers_objects"] = [
        "OVStableDiffusionPipeline",
        "OVStableDiffusionImg2ImgPipeline",
        "OVStableDiffusionInpaintPipeline",
        "OVStableDiffusionXLPipeline",
        "OVStableDiffusionXLImg2ImgPipeline",
        "OVStableDiffusionXLInpaintPipeline",
        "OVStableDiffusion3Pipeline",
        "OVStableDiffusion3Image2ImagePipeline",
        "OVStableDiffusion3InpaintPipeline",
        "OVLatentConsistencyModelPipeline",
        "OVLatentConsistencyModelImg2ImgPipeline",
        "OVLTXPipeline",
        "OVLTXImage2VideoPipeline",
        "OVFluxPipeline",
        "OVFluxImg2ImgPipeline",
        "OVFluxInpaintPipeline",
        "OVFluxFillPipeline",
        "OVSanaPipeline",
        "OVPipelineForImage2Image",
        "OVPipelineForText2Image",
        "OVPipelineForInpainting",
        "OVPipelineForText2Video",
        "OVDiffusionPipeline",
    ]
else:
    _import_structure["openvino"].extend(
        [
            "OVStableDiffusionPipeline",
            "OVStableDiffusionImg2ImgPipeline",
            "OVStableDiffusionInpaintPipeline",
            "OVStableDiffusionXLPipeline",
            "OVStableDiffusionXLImg2ImgPipeline",
            "OVStableDiffusionXLInpaintPipeline",
            "OVStableDiffusion3Pipeline",
            "OVStableDiffusion3Image2ImagePipeline",
            "OVStableDiffusion3InpaintPipeline",
            "OVLatentConsistencyModelPipeline",
            "OVLatentConsistencyModelImg2ImgPipeline",
            "OVLTXPipeline",
            "OVFluxPipeline",
            "OVFluxImg2ImgPipeline",
            "OVFluxInpaintPipeline",
            "OVFluxFillPipeline",
            "OVSanaPipeline",
            "OVPipelineForImage2Image",
            "OVPipelineForText2Image",
            "OVPipelineForInpainting",
            "OVPipelineForText2Video",
            "OVDiffusionPipeline",
        ]
    )

try:
    if not is_openvino_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_openvino_objects

    _import_structure["utils.dummy_openvino_objects"] = [
        name for name in dir(dummy_openvino_objects) if not name.startswith("_")
    ]
else:
    _import_structure["openvino"].extend(
        [
            "OVModelForAudioClassification",
            "OVModelForAudioFrameClassification",
            "OVModelForAudioXVector",
            "OVModelForCausalLM",
            "OVModelForCTC",
            "OVModelForCustomTasks",
            "OVModelForFeatureExtraction",
            "OVModelForImageClassification",
            "OVModelForMaskedLM",
            "OVModelForPix2Struct",
            "OVModelForQuestionAnswering",
            "OVModelForSeq2SeqLM",
            "OVModelForSpeechSeq2Seq",
            "OVModelForTextToSpeechSeq2Seq",
            "OVModelForVision2Seq",
            "OVModelForVisualCausalLM",
            "OVModelForSequenceClassification",
            "OVModelForTokenClassification",
            "OVConfig",
            "OVModelOpenCLIPVisual",
            "OVModelOpenCLIPText",
            "OVModelOpenCLIPForZeroShotImageClassification",
            "OVModelForZeroShotImageClassification",
            "OVSamModel",
        ]
    )

try:
    if not is_neural_compressor_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_neural_compressor_objects

    _import_structure["utils.dummy_neural_compressor_objects"] = [
        name for name in dir(dummy_neural_compressor_objects) if not name.startswith("_")
    ]
else:
    _import_structure["neural_compressor"] = [
        "INCConfig",
        "INCModel",
        "INCModelForCausalLM",
        "INCModelForMaskedLM",
        "INCModelForMultipleChoice",
        "INCModelForQuestionAnswering",
        "INCModelForSeq2SeqLM",
        "INCModelForSequenceClassification",
        "INCModelForTokenClassification",
        "INCModelForVision2Seq",
        "INCQuantizer",
        "INCSeq2SeqTrainer",
        "INCTrainer",
    ]

try:
    if not (is_neural_compressor_available() and is_diffusers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_neural_compressor_and_diffusers_objects"] = ["INCStableDiffusionPipeline"]
else:
    _import_structure["neural_compressor"].append("INCStableDiffusionPipeline")


try:
    if not (is_openvino_available() and is_sentence_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_openvino_and_sentence_transformers_objects"] = ["OVSentenceTransformer"]
else:
    _import_structure["openvino"].extend(["OVSentenceTransformer"])


if TYPE_CHECKING:
    try:
        if not is_ipex_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_ipex_objects import *
    else:
        from .ipex import (
            IPEXModel,
            IPEXModelForAudioClassification,
            IPEXModelForCausalLM,
            IPEXModelForImageClassification,
            IPEXModelForMaskedLM,
            IPEXModelForQuestionAnswering,
            IPEXModelForSeq2SeqLM,
            IPEXModelForSequenceClassification,
            IPEXModelForTokenClassification,
        )

    try:
        if not (is_ipex_available() and is_sentence_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_ipex_objects import IPEXSentenceTransformer
    else:
        from .ipex import IPEXSentenceTransformer

    try:
        if not (is_openvino_available() and is_nncf_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_openvino_and_nncf_objects import (
            OVCalibrationDataset,
            OVDynamicQuantizationConfig,
            OVMixedQuantizationConfig,
            OVQuantizationConfig,
            OVQuantizer,
            OVWeightQuantizationConfig,
        )
    else:
        from .openvino import (
            OVCalibrationDataset,
            OVDynamicQuantizationConfig,
            OVMixedQuantizationConfig,
            OVQuantizationConfig,
            OVQuantizer,
            OVWeightQuantizationConfig,
        )

    try:
        if not (is_openvino_available() and is_diffusers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_openvino_and_diffusers_objects import (
            OVDiffusionPipeline,
            OVFluxPipeline,
            OVLatentConsistencyModelPipeline,
            OVPipelineForImage2Image,
            OVPipelineForInpainting,
            OVPipelineForText2Image,
            OVSanaPipeline,
            OVStableDiffusion3Img2ImgPipeline,
            OVStableDiffusion3InpaintPipeline,
            OVStableDiffusion3Pipeline,
            OVStableDiffusionImg2ImgPipeline,
            OVStableDiffusionInpaintPipeline,
            OVStableDiffusionPipeline,
            OVStableDiffusionXLImg2ImgPipeline,
            OVStableDiffusionXLPipeline,
        )
    else:
        from .openvino import (
            OVDiffusionPipeline,
            OVFluxPipeline,
            OVLatentConsistencyModelImg2ImgPipeline,
            OVLatentConsistencyModelPipeline,
            OVPipelineForImage2Image,
            OVPipelineForInpainting,
            OVPipelineForText2Image,
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

    try:
        if not is_openvino_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_openvino_objects import *
    else:
        from .openvino import (
            OVConfig,
            OVModelForAudioClassification,
            OVModelForAudioFrameClassification,
            OVModelForAudioXVector,
            OVModelForCausalLM,
            OVModelForCTC,
            OVModelForCustomTasks,
            OVModelForFeatureExtraction,
            OVModelForImageClassification,
            OVModelForMaskedLM,
            OVModelForQuestionAnswering,
            OVModelForSeq2SeqLM,
            OVModelForSequenceClassification,
            OVModelForSpeechSeq2Seq,
            OVModelForTextToSpeechSeq2Seq,
            OVModelForTokenClassification,
            OVModelForVision2Seq,
            OVModelForVisualCausalLM,
            OVModelForZeroShotImageClassification,
            OVModelOpenCLIPForZeroShotImageClassification,
            OVModelOpenCLIPText,
            OVModelOpenCLIPVisual,
        )

    try:
        if not is_neural_compressor_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_neural_compressor_objects import *
    else:
        from .neural_compressor import (
            INCConfig,
            INCModel,
            INCModelForCausalLM,
            INCModelForMaskedLM,
            INCModelForMultipleChoice,
            INCModelForQuestionAnswering,
            INCModelForSeq2SeqLM,
            INCModelForSequenceClassification,
            INCModelForTokenClassification,
            INCModelForVision2Seq,
            INCQuantizer,
            INCSeq2SeqTrainer,
            INCTrainer,
        )

    try:
        if not (is_neural_compressor_available() and is_diffusers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_neural_compressor_and_diffusers_objects import INCStableDiffusionPipeline
    else:
        from .neural_compressor import INCStableDiffusionPipeline

    try:
        if not (is_openvino_available() and is_sentence_transformers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_openvino_and_sentence_transformers_objects import OVSentenceTransformer
    else:
        from .openvino import OVSentenceTransformer

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
