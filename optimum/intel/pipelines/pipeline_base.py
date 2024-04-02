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

from transformers import pipeline as transformers_pipeline
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
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
    SUPPORTED_TASKS,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    if model_kwargs is None:
        model_kwargs = {}

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id, export=True)
    elif isinstance(model, str):
        ipex_model_class = SUPPORTED_TASKS[targeted_task]["class"][0]
        model = ipex_model_class.from_pretrained(model, export=True, **model_kwargs)
    elif isinstance(model, IPEXModel):
        pass
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or IPEXModel.
            You can also provide non model then a default one will be used"""
        )

    return model


MAPPING_LOADING_FUNC = {
    "ipex": load_ipex_model,
}


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)


def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    accelerator: Optional[str] = "ipex",
    use_fast: bool = True,
    device: Optional[Union[int, str, "torch.device"]] = None,
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
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].

            If not provided, the default configuration file for the requested model will be used. That means that if
            `model` is given, its default configuration will be used. However, if `model` is not supplied, this
            `task`'s default model's config is used instead.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        accelerator (`str`, *optional*, defaults to `"ipex"`):
            The optimization backends, choose from ["ipex", "inc", "openvino"].
        revision (`str`, *optional*, defaults to `"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        device (`int` or `str` or `torch.device`):
            Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this
            pipeline will be allocated.
        device_map (`str` or `Dict[str, Union[int, str, torch.device]`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set
            `device_map="auto"` to compute the most optimized `device_map` automatically (see
            [here](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling#accelerate.cpu_offload)
            for more information).

            <Tip warning={true}>

            Do not use `device_map` AND `device` at the same time as they will conflict

            </Tip>

        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,
            tokenization or even pipeline files. This option should only be set to `True` for repositories you trust
            and in which you have read the code, as it will execute code present on the Hub on your local machine.
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

    if isinstance(model, Path):
        model = str(model)

    if torch_dtype is not None:
        if "torch_dtype" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... torch_dtype=..., model_kwargs={"torch_dtype":...})` as those'
                " arguments might conflict, use only one.)"
            )
        model_kwargs["torch_dtype"] = torch_dtype

    model_name = model if isinstance(model, str) else None

    # Load the correct model if possible
    # Infer the framework from the model if not already defined
    model = MAPPING_LOADING_FUNC[accelerator](model, task, IPEX_SUPPORTED_TASKS, model_kwargs)

    model_config = model.config
    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None

    if load_tokenizer:
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer is None:
            if isinstance(model_name, str):
                tokenizer = model_name
            elif isinstance(config, str):
                tokenizer = config
            else:
                # Impossible to guess what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # Instantiate tokenizer if needed
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs.copy()
                tokenizer_kwargs.pop("torch_dtype", None)

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **tokenizer_kwargs
            )

    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    if device is not None:
        kwargs["device"] = device

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        use_fast=use_fast,
        **kwargs,
    )
