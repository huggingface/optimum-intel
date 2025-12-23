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


import warnings

from ..utils.import_utils import is_diffusers_available
from .configuration import INCConfig
from .modeling_base import (
    INCModel,
    INCModelForCausalLM,
    INCModelForMaskedLM,
    INCModelForMultipleChoice,
    INCModelForQuestionAnswering,
    INCModelForSeq2SeqLM,
    INCModelForSequenceClassification,
    INCModelForTokenClassification,
    INCModelForVision2Seq,
)
from .quantization import INCQuantizationMode, INCQuantizer
from .trainer import INCTrainer
from .trainer_seq2seq import INCSeq2SeqTrainer


if is_diffusers_available():
    from .modeling_diffusion import INCStableDiffusionPipeline


warnings.warn(
    "`optimum.intel.neural_compressor` is deprecated and will be removed in the next major release of `optimum-intel`.",
    FutureWarning,
    stacklevel=2,
)
