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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from transformers import (
    AudioClassificationPipeline,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutomaticSpeechRecognitionPipeline,
    AutoTokenizer,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    ImageClassificationPipeline,
    ImageToTextPipeline,
    Pipeline,
    PreTrainedTokenizer,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    ZeroShotClassificationPipeline,
)
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.utils import logging

from optimum.intel.utils.import_utils import (
    IPEX_IMPORT_ERROR,
    OPENVINO_IMPORT_ERROR,
    is_ipex_available,
    is_openvino_available,
)


if is_ipex_available():
    from ..ipex.modeling_base import (
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

    IPEX_SUPPORTED_TASKS = {
        "text-generation": {
            "impl": TextGenerationPipeline,
            "class": (IPEXModelForCausalLM,),
            "default": "gpt2",
            "type": "text",
        },
        "summarization": {
            "impl": SummarizationPipeline,
            "class": (IPEXModelForSeq2SeqLM,),
            "default": "t5-base",
            "type": "text",
        },
        "translation": {
            "impl": TranslationPipeline,
            "class": (IPEXModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
        "text2text-generation": {
            "impl": Text2TextGenerationPipeline,
            "class": (IPEXModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
        "fill-mask": {
            "impl": FillMaskPipeline,
            "class": (IPEXModelForMaskedLM,),
            "default": "bert-base-cased",
            "type": "text",
        },
        "question-answering": {
            "impl": QuestionAnsweringPipeline,
            "class": (IPEXModelForQuestionAnswering,),
            "default": "distilbert-base-cased-distilled-squad",
            "type": "text",
        },
        "image-classification": {
            "impl": ImageClassificationPipeline,
            "class": (IPEXModelForImageClassification,),
            "default": "google/vit-base-patch16-224",
            "type": "image",
        },
        "text-classification": {
            "impl": TextClassificationPipeline,
            "class": (IPEXModelForSequenceClassification,),
            "default": "distilbert-base-uncased-finetuned-sst-2-english",
            "type": "text",
        },
        "token-classification": {
            "impl": TokenClassificationPipeline,
            "class": (IPEXModelForTokenClassification,),
            "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "type": "text",
        },
        "audio-classification": {
            "impl": AudioClassificationPipeline,
            "class": (IPEXModelForAudioClassification,),
            "default": "superb/hubert-base-superb-ks",
            "type": "audio",
        },
    }
else:
    IPEX_SUPPORTED_TASKS = {}


if is_openvino_available():
    from ..openvino import (
        OVModelForAudioClassification,
        OVModelForCausalLM,
        OVModelForFeatureExtraction,
        OVModelForImageClassification,
        OVModelForMaskedLM,
        OVModelForQuestionAnswering,
        OVModelForSeq2SeqLM,
        OVModelForSequenceClassification,
        OVModelForSpeechSeq2Seq,
        OVModelForTokenClassification,
        OVModelForVision2Seq,
    )
    from ..openvino.modeling_base import OVBaseModel

    OPENVINO_SUPPORTED_TASKS = {
        "feature-extraction": {
            "impl": FeatureExtractionPipeline,
            "class": (OVModelForFeatureExtraction,),
            "default": "distilbert-base-cased",
            "type": "text",  # feature extraction is only supported for text at the moment
        },
        "fill-mask": {
            "impl": FillMaskPipeline,
            "class": (OVModelForMaskedLM,),
            "default": "bert-base-cased",
            "type": "text",
        },
        "image-classification": {
            "impl": ImageClassificationPipeline,
            "class": (OVModelForImageClassification,),
            "default": "google/vit-base-patch16-224",
            "type": "image",
        },
        "question-answering": {
            "impl": QuestionAnsweringPipeline,
            "class": (OVModelForQuestionAnswering,),
            "default": "distilbert-base-cased-distilled-squad",
            "type": "text",
        },
        "text-classification": {
            "impl": TextClassificationPipeline,
            "class": (OVModelForSequenceClassification,),
            "default": "distilbert-base-uncased-finetuned-sst-2-english",
            "type": "text",
        },
        "text-generation": {
            "impl": TextGenerationPipeline,
            "class": (OVModelForCausalLM,),
            "default": "distilgpt2",
            "type": "text",
        },
        "token-classification": {
            "impl": TokenClassificationPipeline,
            "class": (OVModelForTokenClassification,),
            "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "type": "text",
        },
        "zero-shot-classification": {
            "impl": ZeroShotClassificationPipeline,
            "class": (OVModelForSequenceClassification,),
            "default": "facebook/bart-large-mnli",
            "type": "text",
        },
        "summarization": {
            "impl": SummarizationPipeline,
            "class": (OVModelForSeq2SeqLM,),
            "default": "t5-base",
            "type": "text",
        },
        "translation": {
            "impl": TranslationPipeline,
            "class": (OVModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
        "text2text-generation": {
            "impl": Text2TextGenerationPipeline,
            "class": (OVModelForSeq2SeqLM,),
            "default": "t5-small",
            "type": "text",
        },
        "automatic-speech-recognition": {
            "impl": AutomaticSpeechRecognitionPipeline,
            "class": (OVModelForSpeechSeq2Seq,),
            "default": "openai/whisper-tiny.en",
            "type": "multimodal",
        },
        "image-to-text": {
            "impl": ImageToTextPipeline,
            "class": (OVModelForVision2Seq,),
            "default": "nlpconnect/vit-gpt2-image-captioning",
            "type": "multimodal",
        },
        "audio-classification": {
            "impl": AudioClassificationPipeline,
            "class": (OVModelForAudioClassification,),
            "default": "superb/hubert-base-superb-ks",
            "type": "audio",
        },
    }
else:
    OPENVINO_SUPPORTED_TASKS = {}


def load_openvino_model(
    model,
    targeted_task,
    SUPPORTED_TASKS,
    hub_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    hub_kwargs = hub_kwargs or {}
    model_kwargs = model_kwargs or {}
    ov_model_class = SUPPORTED_TASKS[targeted_task]["class"][0]

    if isinstance(model, str) or model is None:
        model_id = model or SUPPORTED_TASKS[targeted_task]["default"]
        model = ov_model_class.from_pretrained(model_id, **hub_kwargs, **model_kwargs)
    elif isinstance(model, OVBaseModel):
        model_id = model.model_save_dir
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or ORTModel.
            You can also provide non model then a default one will be used"""
        )
    return model, model_id


def load_ipex_model(
    model,
    targeted_task,
    SUPPORTED_TASKS,
    hub_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    device_map: Optional[torch.device] = None,
):
    hub_kwargs = hub_kwargs or {}
    model_kwargs = model_kwargs or {}
    ipex_model_class = SUPPORTED_TASKS[targeted_task]["class"][0]

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = ipex_model_class.from_pretrained(model_id, **hub_kwargs, **model_kwargs, device_map=device_map)
    elif isinstance(model, str):
        model_id = model
        model = ipex_model_class.from_pretrained(model, **hub_kwargs, **model_kwargs, device_map=device_map)
    elif isinstance(model, IPEXModel):
        model_id = getattr(model.config, "name_or_path", None)
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model name or path or a IPEXModel.
            You can also provide non model then a default one will be used"""
        )

    return model, model_id


MAPPING_LOADING_FUNC = {
    "ipex": load_ipex_model,
    "openvino": load_openvino_model,
}


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)


def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel"]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    accelerator: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
    model_kwargs: Dict[str, Any] = None,
    **kwargs,
) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    Pipelines are made of:

        - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
        - A [model](model) to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
            - `"fill-mask"`: will return a [`FillMaskPipeline`].
            - `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
            - `"image-classificatio"`: will return a [`ImageClassificationPipeline`].
            - `"text-classification"`: will return a [`TextClassificationPipeline`].
            - `"token-classification"`: will return a [`TokenClassificationPipeline`].
            - `"audio-classification"`: will return a [`AudioClassificationPipeline`].

        model (`str` or [`PreTrainedModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from [`PreTrainedModel`] (for PyTorch).

            If not provided, the default for the `task` will be loaded.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        accelerator (`str`, *optional*):
            The optimization backends, choose from ["ipex", "inc", "openvino"].
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
        model_kwargs (`Dict[str, Any]`, *optional*):
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.

    Examples:

    ```python
    >>> import torch
    >>> from optimum.intel.pipelines import pipeline

    >>> pipe = pipeline('text-generation', 'gpt2', torch_dtype=torch.bfloat16)
    >>> pipe("Describe a real-world application of AI in sustainable energy.")
    ```"""

    if model_kwargs is None:
        model_kwargs = {}

    commit_hash = kwargs.pop("_commit_hash", None)

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
            " path/identifier to a pretrained model when providing tokenizer."
        )

    if accelerator not in MAPPING_LOADING_FUNC:
        if accelerator is None:
            msg = "Impossible to instantiate a pipeline without specifying an `accelerator`."
        else:
            msg = f"`accelerator` {accelerator} is not supported."

        raise ValueError(msg + f" Supported list of `accelerator` is : {', '.join(MAPPING_LOADING_FUNC)}.")

    if accelerator == "ipex":
        if not is_ipex_available():
            raise RuntimeError(IPEX_IMPORT_ERROR.format("`accelerator=ipex`"))
        supported_tasks = IPEX_SUPPORTED_TASKS

    if accelerator == "openvino":
        if not is_openvino_available():
            raise RuntimeError(OPENVINO_IMPORT_ERROR.format("`accelerator=openvino`"))
        supported_tasks = OPENVINO_SUPPORTED_TASKS

    if task not in supported_tasks:
        raise ValueError(
            f"Task {task} is not supported for the {accelerator} pipelines. Supported tasks are {', '.join(supported_tasks)}"
        )

    no_feature_extractor_tasks = set()
    no_tokenizer_tasks = set()
    for _task, values in supported_tasks.items():
        if values["type"] == "text":
            no_feature_extractor_tasks.add(_task)
        elif values["type"] in {"image", "video"}:
            no_tokenizer_tasks.add(_task)
        elif values["type"] in {"audio"}:
            no_tokenizer_tasks.add(_task)
        elif values["type"] not in ["multimodal", "audio", "video"]:
            raise ValueError(f"SUPPORTED_TASK {_task} contains invalid type {values['type']}")

    load_tokenizer = task not in no_tokenizer_tasks
    load_feature_extractor = task not in no_feature_extractor_tasks

    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": commit_hash,
    }

    if isinstance(model, Path):
        model = str(model)

    tokenizer_kwargs = model_kwargs.copy()
    if torch_dtype is not None:
        if "torch_dtype" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... torch_dtype=..., model_kwargs={"torch_dtype":...})` as those'
                " arguments might conflict, use only one.)"
            )
        model_kwargs["torch_dtype"] = torch_dtype

    # Load the correct model and convert it to the expected format if needed
    model, model_id = MAPPING_LOADING_FUNC[accelerator](
        model,
        task,
        SUPPORTED_TASKS=supported_tasks,
        hub_kwargs=hub_kwargs,
        model_kwargs=model_kwargs,
        **kwargs,
    )

    if load_tokenizer and tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, **hub_kwargs, **tokenizer_kwargs)
    if load_feature_extractor and feature_extractor is None:
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, **hub_kwargs, **tokenizer_kwargs)
        except Exception:
            feature_extractor = AutoImageProcessor.from_pretrained(model_id, **hub_kwargs, **tokenizer_kwargs)

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        torch_dtype=torch_dtype,
    )
