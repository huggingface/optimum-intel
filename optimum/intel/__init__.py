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
    is_diffusers_available,
    is_ipex_available,
    is_neural_compressor_available,
    is_nncf_available,
    is_openvino_available,
)
from .version import __version__


_import_structure = {
    "openvino": [],
}

try:
    if not is_ipex_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_ipex_objects"] = ["inference_mode"]
else:
    _import_structure["ipex"] = ["inference_mode"]

try:
    if not (is_openvino_available() and is_nncf_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_openvino_and_nncf_objects"] = [
        "OVConfig",
        "OVQuantizer",
        "OVTrainer",
        "OVTrainingArguments",
    ]
else:
    _import_structure["openvino"].extend(["OVConfig", "OVQuantizer", "OVTrainer", "OVTrainingArguments"])


try:
    if not (is_openvino_available() and is_diffusers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    _import_structure["utils.dummy_openvino_and_diffusers_objects"] = ["OVStableDiffusionPipeline"]
else:
    _import_structure["openvino"].append("OVStableDiffusionPipeline")

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
            "OVModelForCausalLM",
            "OVModelForFeatureExtraction",
            "OVModelForImageClassification",
            "OVModelForMaskedLM",
            "OVModelForQuestionAnswering",
            "OVModelForSeq2SeqLM",
            "OVModelForSequenceClassification",
            "OVModelForTokenClassification",
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


if TYPE_CHECKING:
    try:
        if not is_ipex_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_ipex_objects import inference_mode
    else:
        from .ipex import inference_mode

    try:
        if not (is_openvino_available() and is_nncf_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_openvino_and_nncf_objects import OVConfig, OVQuantizer, OVTrainer, OVTrainingArguments
    else:
        from .openvino import OVConfig, OVQuantizer, OVTrainer, OVTrainingArguments

    try:
        if not (is_openvino_available() and is_diffusers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_openvino_and_diffusers_objects import OVStableDiffusionPipeline
    else:
        from .openvino import OVStableDiffusionPipeline

    try:
        if not is_openvino_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_openvino_objects import *
    else:
        from .openvino import (
            OVModelForAudioClassification,
            OVModelForCausalLM,
            OVModelForFeatureExtraction,
            OVModelForImageClassification,
            OVModelForMaskedLM,
            OVModelForQuestionAnswering,
            OVModelForSeq2SeqLM,
            OVModelForSequenceClassification,
            OVModelForTokenClassification,
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
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
