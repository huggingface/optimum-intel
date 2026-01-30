#  Copyright 2024 The HuggingFace Team. All rights reserved.
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

import contextlib
from typing import TYPE_CHECKING, Optional

import transformers.pipelines
from transformers import AutoConfig

from optimum.intel.utils import (
    IPEX_IMPORT_ERROR,
    OPENVINO_IMPORT_ERROR,
    is_ipex_available,
    is_openvino_available,
    is_transformers_version,
)
from optimum.utils.logging import get_logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = get_logger(__name__)

if is_ipex_available():
    from ..ipex import (
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

    IPEX_TASKS_MAPPING = {
        "audio-classification": (IPEXModelForAudioClassification,),
        "feature-extraction": (IPEXModel,),
        "fill-mask": (IPEXModelForMaskedLM,),
        "image-classification": (IPEXModelForImageClassification,),
        "question-answering": (IPEXModelForQuestionAnswering,),
        "summarization": (IPEXModelForSeq2SeqLM,),
        "text2text-generation": (IPEXModelForSeq2SeqLM,),
        "text-classification": (IPEXModelForSequenceClassification,),
        "text-generation": (IPEXModelForCausalLM,),
        "token-classification": (IPEXModelForTokenClassification,),
        "translation": (IPEXModelForSeq2SeqLM,),
        "zero-shot-classification": (IPEXModelForSequenceClassification,),
    }

else:
    IPEX_TASKS_MAPPING = {}


if is_openvino_available():
    from ..openvino import (
        OVModelForAudioClassification,
        OVModelForAudioFrameClassification,
        OVModelForAudioXVector,
        OVModelForCausalLM,
        OVModelForCTC,
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
    )
    from ..openvino.modeling_base import OVBaseModel

    OV_TASKS_MAPPING = {
        "audio-classification": (OVModelForAudioClassification,),
        "audio-frame-classification": (OVModelForAudioFrameClassification,),
        "audio-xvector": (OVModelForAudioXVector,),
        "automatic-speech-recognition": (OVModelForCTC, OVModelForSpeechSeq2Seq),
        "feature-extraction": (OVModelForFeatureExtraction,),
        "fill-mask": (OVModelForMaskedLM,),
        "image-classification": (OVModelForImageClassification,),
        "image-text-to-text": (OVModelForVisualCausalLM,),
        "image-to-text": (OVModelForVision2Seq,),
        "question-answering": (OVModelForQuestionAnswering,),
        "summarization": (OVModelForSeq2SeqLM,),
        "text2text-generation": (OVModelForSeq2SeqLM,),
        "text-classification": (OVModelForSequenceClassification,),
        "text-generation": (OVModelForCausalLM,),
        "text-to-audio": (OVModelForTextToSpeechSeq2Seq,),
        "token-classification": (OVModelForTokenClassification,),
        "translation": (OVModelForSeq2SeqLM,),
        "zero-shot-image-classification": (OVModelForZeroShotImageClassification,),
    }
else:
    OV_TASKS_MAPPING = {}


def get_openvino_model_class(
    task: str, config: Optional["PretrainedConfig"] = None, model_id: Optional[str] = None, **model_kwargs
):
    if task.startswith("translation_"):
        task = "translation"

    if task not in OV_TASKS_MAPPING:
        raise KeyError(
            f"Task '{task}' is not supported by OpenVINO. Only {list(OV_TASKS_MAPPING.keys())} are supported."
        )

    if task == "automatic-speech-recognition":
        if config is None:
            hub_kwargs = {
                "trust_remote_code": model_kwargs.pop("trust_remote_code", False),
                "revision": model_kwargs.pop("revision", None),
                "token": model_kwargs.pop("token", None),
            }
            config = AutoConfig.from_pretrained(model_id, **hub_kwargs)
        if any(arch.endswith("ForCTC") for arch in config.architectures):
            ov_model_class = OV_TASKS_MAPPING[task][0]
        else:
            ov_model_class = OV_TASKS_MAPPING[task][1]
    else:
        ov_model_class = OV_TASKS_MAPPING[task][0]

    return ov_model_class


# a modified transformers.pipelines.base.infer_framework_load_model that loads OpenVINO models
def openvino_infer_framework_load_model(
    model, config: Optional["PretrainedConfig"] = None, task: Optional[str] = None, **model_kwargs
):
    if isinstance(model, str):
        model_kwargs.pop("framework", None)
        model_kwargs.pop("_commit_hash", None)  # not supported for OVModel
        model_kwargs.pop("model_classes", None)
        ov_model_class = get_openvino_model_class(task, config, model, **model_kwargs)
        ov_model = ov_model_class.from_pretrained(model, **model_kwargs)
    elif isinstance(model, OVBaseModel):
        ov_model = model
    else:
        raise TypeError(
            f"""Model {model} is not supported. Please provide a valid model either as string or OVBaseModel.
            You can also provide None as the model to use a default one."""
        )

    if is_transformers_version("<", "5"):
        return "pt", ov_model

    return ov_model


def get_ipex_model_class(task: str, **model_kwargs):
    if task.startswith("translation_"):
        task = "translation"

    if task not in IPEX_TASKS_MAPPING:
        raise KeyError(
            f"Task '{task}' is not supported by IPEX. Only {list(IPEX_TASKS_MAPPING.keys())} are supported."
        )

    ipex_model_class = IPEX_TASKS_MAPPING[task][0]

    return ipex_model_class


# a modified transformers.pipelines.base.infer_framework_load_model that loads IPEX models
def ipex_infer_framework_load_model(
    model, config: Optional["PretrainedConfig"] = None, task: Optional[str] = None, **model_kwargs
):
    if isinstance(model, str):
        model_kwargs.pop("framework", None)
        model_kwargs.pop("_commit_hash", None)  # not supported for IPEXModel
        model_kwargs.pop("model_classes", None)
        ipex_model_class = get_ipex_model_class(task, **model_kwargs)
        ipex_model = ipex_model_class.from_pretrained(model, **model_kwargs)
    elif isinstance(model, IPEXModel):
        ipex_model = model
    else:
        raise TypeError(
            f"""Model {model} is not supported. Please provide a valid model either as string or IPEXModel.
            You can also provide None as the model to use a default one."""
        )

    if is_transformers_version("<", "5"):
        return "pt", ipex_model

    return ipex_model


@contextlib.contextmanager
def patch_pipelines_to_load_accelerator_model(accelerator: str):
    target_fn = "infer_framework_load_model" if is_transformers_version("<", "5") else "load_model"

    original_infer_framework_load_model = getattr(transformers.pipelines, target_fn)

    if accelerator == "openvino":
        if not is_openvino_available():
            raise ImportError(OPENVINO_IMPORT_ERROR.format("`accelerator=openvino`"))

        setattr(transformers.pipelines, target_fn, openvino_infer_framework_load_model)

    elif accelerator == "ipex":
        if not is_ipex_available():
            raise ImportError(IPEX_IMPORT_ERROR.format("`accelerator=ipex`"))

        setattr(transformers.pipelines, target_fn, ipex_infer_framework_load_model)
    else:
        raise ValueError(f"Accelerator '{accelerator}' is not supported. Only 'openvino' and 'ipex' are supported.")

    try:
        yield
    finally:
        setattr(transformers.pipelines, target_fn, original_infer_framework_load_model)
