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

from transformers import SequenceFeatureExtractor
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import (
    AudioClassificationPipeline,
    FillMaskPipeline,
    ImageClassificationPipeline,
    QuestionAnsweringPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
)
from transformers.pipelines.base import Pipeline
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import (
    is_ipex_available,
    is_torch_available,
    logging,
)

from optimum.utils.file_utils import find_files_matching_pattern


if is_ipex_available():
    from ..ipex.modeling_base import (
        IPEXModel,
        IPEXModelForAudioClassification,
        IPEXModelForCausalLM,
        IPEXModelForImageClassification,
        IPEXModelForMaskedLM,
        IPEXModelForQuestionAnswering,
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


def load_ipex_model(
    model,
    targeted_task,
    load_tokenizer,
    tokenizer,
    load_feature_extractor,
    feature_extractor,
    SUPPORTED_TASKS,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    if model_kwargs is None:
        model_kwargs = {}

    ipex_model_class = SUPPORTED_TASKS[targeted_task]["class"][0]

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = ipex_model_class.from_pretrained(model_id, export=True)
    elif isinstance(model, str):
        model_id = model
        ipex_file = find_files_matching_pattern(
            model,
            ".+?.pt",
            glob_pattern="**/*.pt",
            subfolder=model_kwargs.pop("subfolder", None),
            use_auth_token=model_kwargs.pop("token", None),
            revision=model_kwargs.pop("revision", "main"),
        )
        export = len(ipex_file) == 0
        model = ipex_model_class.from_pretrained(model, export=export, **model_kwargs)
    elif isinstance(model, IPEXModel):
        if tokenizer is None and load_tokenizer:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                    tokenizer = preprocessor
                    break
            if tokenizer is None:
                raise ValueError(
                    "Could not automatically find a tokenizer for the IPEXModel, you must pass a tokenizer explictly"
                )
        if feature_extractor is None and load_feature_extractor:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, SequenceFeatureExtractor):
                    feature_extractor = preprocessor
                    break
            if feature_extractor is None:
                raise ValueError(
                    "Could not automatically find a feature extractor for the IPEXModel, you must pass a "
                    "feature_extractor explictly"
                )
        model_id = None
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or IPEXModel.
            You can also provide non model then a default one will be used"""
        )

    return model, model_id, tokenizer, feature_extractor


MAPPING_LOADING_FUNC = {
    "ipex": load_ipex_model,
}


if is_torch_available():
    pass


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)


def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel"]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    accelerator: Optional[str] = "ipex",
    use_fast: bool = True,
    torch_dtype=None,
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
        accelerator (`str`, *optional*, defaults to `"ipex"`):
            The optimization backends, choose from ["ipex", "inc", "openvino"].
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
        model_kwargs (`Dict[str, Any]`, *optional*):
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

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
        raise ValueError(f'Accelerator {accelerator} is not supported. Supported accelerator is "ipex".')

    if accelerator == "ipex":
        if task not in list(IPEX_SUPPORTED_TASKS.keys()):
            raise ValueError(
                f"Task {task} is not supported for the ONNX Runtime pipeline. Supported tasks are { list(IPEX_SUPPORTED_TASKS.keys())}"
            )

    supported_tasks = IPEX_SUPPORTED_TASKS if accelerator == "ipex" else None

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

    load_tokenizer = False if task in no_tokenizer_tasks else True
    load_feature_extractor = False if task in no_feature_extractor_tasks else True

    if isinstance(model, Path):
        model = str(model)

    if torch_dtype is not None:
        if "torch_dtype" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... torch_dtype=..., model_kwargs={"torch_dtype":...})` as those'
                " arguments might conflict, use only one.)"
            )
        model_kwargs["torch_dtype"] = torch_dtype

    # Load the correct model if possible
    # Infer the framework from the model if not already defined
    model, model_id, tokenizer, feature_extractor = MAPPING_LOADING_FUNC[accelerator](
        model,
        task,
        load_tokenizer,
        tokenizer,
        load_feature_extractor,
        feature_extractor,
        IPEX_SUPPORTED_TASKS,
        model_kwargs,
    )

    if tokenizer is None and load_tokenizer:
        tokenizer = get_preprocessor(model_id)
    if feature_extractor is None and load_feature_extractor:
        feature_extractor = get_preprocessor(model_id)

    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_fast=use_fast,
        **kwargs,
    )
