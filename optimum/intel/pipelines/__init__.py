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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from huggingface_hub import model_info
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from transformers.pipelines.base import (
    Pipeline,
    PipelineRegistry,
    get_default_model_and_revision,
)
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import (
    CONFIG_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    cached_file,
    extract_commit_hash,
    is_ipex_available,
    is_offline_mode,
    is_torch_available,
    logging,
)

from ..generation.modeling import TSModelForCausalLM


if is_ipex_available():
    import intel_extension_for_pytorch


if is_torch_available():
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)


# Register all the supported tasks here
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
}
SUPPORTED_TASKS = {
    "text-generation": {
        "impl": TextGenerationPipeline,
        "pt": (AutoModelForCausalLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("gpt2", "6c0e608")}},
        "type": "text",
    },
}


PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return PIPELINE_REGISTRY.get_supported_tasks()


def get_task(model: str, token: Optional[str] = None, **deprecated_kwargs) -> str:
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    if is_offline_mode():
        raise RuntimeError("You cannot infer task automatically within `pipeline` when using offline mode")
    try:
        info = model_info(model, token=token)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}")
    if not info.pipeline_tag:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    if getattr(info, "library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {info.library_name} not with transformers")
    task = info.pipeline_tag
    return task


def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"text-generation"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) The normalized task name
        (removed alias and options). The actual dictionary required to initialize the pipeline and some extra task
        options for parametrized tasks like "translation_XX_to_YY"


    """
    return PIPELINE_REGISTRY.check_task(task)


def clean_custom_task(task_info):
    import transformers

    if "impl" not in task_info:
        raise RuntimeError("This model introduces a custom pipeline without specifying its implementation.")
    pt_class_names = task_info.get("pt", ())
    if isinstance(pt_class_names, str):
        pt_class_names = [pt_class_names]
    task_info["pt"] = tuple(getattr(transformers, c) for c in pt_class_names)
    return task_info, None


def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map=None,
    torch_dtype=None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
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
    >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

    >>> # Sentiment analysis pipeline
    >>> analyzer = pipeline("sentiment-analysis")

    >>> # Question answering pipeline, specifying the checkpoint identifier
    >>> oracle = pipeline(
    ...     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
    ... )

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```"""
    if model_kwargs is None:
        model_kwargs = {}

    code_revision = kwargs.pop("code_revision", None)
    commit_hash = kwargs.pop("_commit_hash", None)

    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": commit_hash,
    }

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )

    if task != "text-generation":
        raise ValueError("Optimum-intel ipex optimization only supports text-generation task for now.")

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
            " path/identifier to a pretrained model when providing tokenizer."
        )

    if isinstance(model, Path):
        model = str(model)

    if commit_hash is None:
        pretrained_model_name_or_path = None
        if isinstance(config, str):
            pretrained_model_name_or_path = config
        elif config is None and isinstance(model, str):
            pretrained_model_name_or_path = model

        if not isinstance(config, PretrainedConfig) and pretrained_model_name_or_path is not None:
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **hub_kwargs,
            )
            hub_kwargs["_commit_hash"] = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            hub_kwargs["_commit_hash"] = getattr(config, "_commit_hash", None)

    # Config is the primordial information item.
    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(
            config, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        hub_kwargs["_commit_hash"] = config._commit_hash
    elif config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        hub_kwargs["_commit_hash"] = config._commit_hash

    if task is None and model is not None:
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`. "
                f"{model} is not a valid model_id."
            )
        task = get_task(model, token)

    normalized_task, targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model, default_revision = get_default_model_and_revision(targeted_task, "pt", task_options)
        revision = revision if revision is not None else default_revision
        logger.warning(
            f"No model was supplied, defaulted to {model} and revision"
            f" {revision} ({HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{model}).\n"
            "Using a pipeline without specifying a model name and revision in production is not recommended."
        )
        if config is None and isinstance(model, str):
            config = AutoConfig.from_pretrained(model, _from_pipeline=task, **hub_kwargs, **model_kwargs)
            hub_kwargs["_commit_hash"] = config._commit_hash

    if device_map is not None:
        if "device_map" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... device_map=..., model_kwargs={"device_map":...})` as those'
                " arguments might conflict, use only one.)"
            )
        if device is not None:
            logger.warning(
                "Both `device` and `device_map` are specified. `device` will override `device_map`. You"
                " will most likely encounter unexpected behavior. Please remove `device` and keep `device_map`."
            )
        model_kwargs["device_map"] = device_map
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
    if isinstance(model, str):
        model = TSModelForCausalLM.from_pretrained(model, config=config, export=True, **model_kwargs)

    model_config = model.config
    hub_kwargs["_commit_hash"] = model.config._commit_hash
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
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **hub_kwargs, **tokenizer_kwargs
            )

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    if device is not None:
        kwargs["device"] = device

    return pipeline_class(model=model, framework="pt", task=task, **kwargs)
