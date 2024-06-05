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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from optimum.exporters import TasksManager
from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.utils.import_utils import is_openvino_tokenizers_available, is_transformers_version
from optimum.utils.save_utils import maybe_load_preprocessors


if TYPE_CHECKING:
    from optimum.intel.openvino.configuration import OVConfig

_COMPRESSION_OPTIONS = {
    "int8": {"bits": 8},
    "int4_sym_g128": {"bits": 4, "sym": True, "group_size": 128},
    "int4_asym_g128": {"bits": 4, "sym": False, "group_size": 128},
    "int4_sym_g64": {"bits": 4, "sym": True, "group_size": 64},
    "int4_asym_g64": {"bits": 4, "sym": False, "group_size": 64},
}


logger = logging.getLogger(__name__)


def infer_task(task, model_name_or_path):
    task = TasksManager.map_from_synonym(task)
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
    return task


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    device: str = "cpu",
    framework: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    token: Optional[Union[bool, str]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_export_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    compression_option: Optional[str] = None,
    compression_ratio: Optional[float] = None,
    ov_config: "OVConfig" = None,
    stateful: bool = True,
    convert_tokenizer: bool = False,
    library_name: Optional[str] = None,
    **kwargs_shapes,
):
    """
    Full-suite OpenVINO export.

    Args:
        > Required parameters

        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the model repository to export.
        output (`Union[str, Path]`):
            Path indicating the directory where to store the generated OpenVINO model.

        > Optional parameters

        task (`Optional[str]`, defaults to `None`):
            The task to export the model for. If not specified, the task will be auto-inferred based on the model. For decoder models,
            use `xxx-with-past` to export the model using past key values in the decoder.
        device (`str`, defaults to `"cpu"`):
            The device to use to do the export. Defaults to "cpu".
        framework (`Optional[str]`, defaults to `None`):
            The framework to use for the ONNX export (`"pt"` or `"tf"`). If not provided, will attempt to automatically detect
            the framework for the checkpoint.
        cache_dir (`Optional[str]`, defaults to `None`):
            Path indicating where to store cache. The default Hugging Face cache path will be used by default.
        trust_remote_code (`bool`, defaults to `False`):
            Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
            you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
            model repository.
        pad_token_id (`Optional[int]`, defaults to `None`):
            This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        revision (`str`, defaults to `"main"`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        force_download (`bool`, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        local_files_only (`Optional[bool]`, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (Optional[Union[bool, str]], defaults to `None`):
            Deprecated. Please use `token` instead.
        token (Optional[Union[bool, str]], defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_export_configs` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).
        custom_export_configs (`Optional[Dict[str, OnnxConfig]]`, defaults to `None`):
            Experimental usage: override the default export config used for the given model. This argument may be useful for advanced users that desire a finer-grained control on the export. An example is available [here](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model).
        fn_get_submodels (`Optional[Callable]`, defaults to `None`):
            Experimental usage: Override the default submodels that are used at the export. This is
            especially useful when exporting a custom architecture that needs to split the ONNX (e.g. encoder-decoder). If unspecified with custom models, optimum will try to use the default submodels used for the given task, with no guarantee of success.
        compression_option (`Optional[str]`, defaults to `None`):
            The weight compression option, e.g. `f16` stands for float16 weights, `i8` - INT8 weights, `int4_sym_g128` - INT4 symmetric weights w/ group size 128, `int4_asym_g128` - as previous but asymmetric w/ zero-point,
            `int4_sym_g64` - INT4 symmetric weights w/ group size 64, "int4_asym_g64" - as previous but asymmetric w/ zero-point, `f32` - means no compression.
        compression_ratio (`Optional[float]`, defaults to `None`):
            Compression ratio between primary and backup precision (only relevant to INT4).
        stateful (`bool`, defaults to `True`):
            Produce stateful model where all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. Applicable only for decoder models.
        **kwargs_shapes (`Dict`):
            Shapes to use during inference. This argument allows to override the default shapes used during the ONNX export.

    Example usage:
    ```python
    >>> from optimum.exporters.openvino import main_export

    >>> main_export("gpt2", output="gpt2_ov/")
    ```
    """

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
        token = use_auth_token

    if compression_option is not None:
        logger.warning(
            "The `compression_option` argument is deprecated and will be removed in optimum-intel v1.17.0. "
            "Please, pass an `ov_config` argument instead `OVConfig(..., quantization_config=quantization_config)`."
        )

    if compression_ratio is not None:
        logger.warning(
            "The `compression_ratio` argument is deprecated and will be removed in optimum-intel v1.17.0. "
            "Please, pass an `ov_config` argument instead `OVConfig(quantization_config={ratio=compression_ratio})`."
        )

    if ov_config is None and compression_option is not None:
        from ...intel.openvino.configuration import OVConfig

        if compression_option == "fp16":
            ov_config = OVConfig(dtype="fp16")
        elif compression_option != "fp32":
            q_config = _COMPRESSION_OPTIONS[compression_option] if compression_option in _COMPRESSION_OPTIONS else {}
            q_config["ratio"] = compression_ratio or 1.0
            ov_config = OVConfig(quantization_config=q_config)

    original_task = task
    task = infer_task(task, model_name_or_path)
    framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)
    library_name_is_not_provided = library_name is None
    library_name = TasksManager.infer_library_from_model(
        model_name_or_path, subfolder=subfolder, library_name=library_name
    )

    if library_name == "sentence_transformers" and library_name_is_not_provided:
        logger.warning(
            "Library name is not specified. There are multiple possible variants: `sentence_tenasformers`, `transformers`."
            "`transformers` will be selected. If you want to load your model with the `sentence-transformers` library instead, please set --library sentence_transformers"
        )
        library_name = "transformers"

    do_gptq_patching = False
    custom_architecture = False
    loading_kwargs = {}
    if library_name == "transformers":
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        quantization_config = getattr(config, "quantization_config", None)
        do_gptq_patching = quantization_config and quantization_config["quant_method"] == "gptq"
        model_type = config.model_type.replace("_", "-")
        if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
            custom_architecture = True
            if custom_export_configs is None:
                raise ValueError(
                    f"Trying to export a {model_type} model, that is a custom or unsupported architecture, but no custom export configuration was passed as `custom_export_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models. Please open an issue at https://github.com/huggingface/optimum-intel/issues if you would like the model type {model_type} to be supported natively in the OpenVINO export."
                )
        elif task not in TasksManager.get_supported_tasks_for_model_type(
            model_type, exporter="openvino", library_name=library_name
        ):
            if original_task == "auto":
                autodetected_message = " (auto-detected)"
            else:
                autodetected_message = ""
            model_tasks = TasksManager.get_supported_tasks_for_model_type(
                model_type, exporter="openvino", library_name=library_name
            )
            raise ValueError(
                f"Asked to export a {model_type} model for the task {task}{autodetected_message}, but the Optimum OpenVINO exporter only supports the tasks {', '.join(model_tasks.keys())} for {model_type}. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task {task} to be supported in the ONNX export for {model_type}."
            )

        if is_transformers_version(">=", "4.36") and model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED:
            loading_kwargs["attn_implementation"] = "eager"
        # there are some difference between remote and in library representation of past key values for some models,
        # for avoiding confusion we disable remote code for them
        if (
            trust_remote_code
            and model_type in {"falcon", "mpt", "phi"}
            and ("with-past" in task or original_task == "auto")
            and not custom_export_configs
        ):
            logger.warning(
                f"Model type `{model_type}` export for task `{task}` is not supported for loading with `trust_remote_code=True`"
                "using default export configuration, `trust_remote_code` will be disabled. "
                "Please provide custom export config if you want load model with remote code."
            )
            trust_remote_code = False

    # Patch the modules to export of GPTQ models w/o GPU
    if do_gptq_patching:
        import torch

        torch.set_default_dtype(torch.float32)
        orig_cuda_check = torch.cuda.is_available
        torch.cuda.is_available = lambda: True

        from optimum.gptq import GPTQQuantizer

        orig_post_init_model = GPTQQuantizer.post_init_model

        def post_init_model(self, model):
            from auto_gptq import exllama_set_max_input_length

            class StoreAttr(object):
                pass

            model.quantize_config = StoreAttr()
            model.quantize_config.desc_act = self.desc_act
            if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
                model = exllama_set_max_input_length(model, self.max_input_length)
            return model

        GPTQQuantizer.post_init_model = post_init_model

    model = TasksManager.get_model_from_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework=framework,
        device=device,
        library_name=library_name,
        **loading_kwargs,
    )

    needs_pad_token_id = task == "text-classification" and getattr(model.config, "pad_token_id", None) is None

    if needs_pad_token_id:
        if pad_token_id is not None:
            model.config.pad_token_id = pad_token_id
        else:
            tok = AutoTokenizer.from_pretrained(model_name_or_path)
            pad_token_id = getattr(tok, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                )
            model.config.pad_token_id = pad_token_id

    if "stable-diffusion" in task:
        model_type = "stable-diffusion"
    elif hasattr(model.config, "export_model_type"):
        model_type = model.config.export_model_type.replace("_", "-")
    else:
        model_type = model.config.model_type.replace("_", "-")

    if (
        not custom_architecture
        and library_name != "diffusers"
        and task + "-with-past"
        in TasksManager.get_supported_tasks_for_model_type(model_type, exporter="openvino", library_name=library_name)
    ):
        # Make -with-past the default if --task was not explicitely specified
        if original_task == "auto":
            task = task + "-with-past"
        else:
            logger.info(
                f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                f" if needed, please pass `--task {task}-with-past` to export using the past key values."
            )

    if original_task == "auto":
        synonyms_for_task = sorted(TasksManager.synonyms_for_task(task))
        if synonyms_for_task:
            synonyms_for_task = ", ".join(synonyms_for_task)
            possible_synonyms = f" (possible synonyms are: {synonyms_for_task})"
        else:
            possible_synonyms = ""
        logger.info(f"Automatic task detection to {task}{possible_synonyms}.")

    preprocessors = maybe_load_preprocessors(
        model_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
    )

    export_from_model(
        model=model,
        output=output,
        task=task,
        ov_config=ov_config,
        stateful=stateful,
        model_kwargs=model_kwargs,
        custom_export_configs=custom_export_configs,
        fn_get_submodels=fn_get_submodels,
        preprocessors=preprocessors,
        device=device,
        trust_remote_code=trust_remote_code,
        **kwargs_shapes,
    )

    if convert_tokenizer:
        maybe_convert_tokenizers(library_name, output, model, preprocessors)

    # Unpatch modules after GPTQ export
    if do_gptq_patching:
        torch.cuda.is_available = orig_cuda_check
        GPTQQuantizer.post_init_model = orig_post_init_model


def maybe_convert_tokenizers(library_name: str, output: Path, model=None, preprocessors=None):
    """
    Tries to convert tokenizers to OV format and export them to disk.

    Arguments:
        library_name (`str`):
            The library name.
        output (`Path`):
            Path to save converted tokenizers to.
        model (`PreTrainedModel`, *optional*, defaults to None):
            Model instance.
        preprocessors (`Iterable`, *optional*, defaults to None):
            Iterable possibly containing tokenizers to be converted.
    """
    from optimum.exporters.openvino.convert import export_tokenizer

    if is_openvino_tokenizers_available():
        if library_name != "diffusers" and preprocessors:
            tokenizer = next(filter(lambda it: isinstance(it, PreTrainedTokenizerBase), preprocessors), None)
            if tokenizer:
                try:
                    export_tokenizer(tokenizer, output)
                except Exception as exception:
                    logger.warning(
                        "Could not load tokenizer using specified model ID or path. OpenVINO tokenizer/detokenizer "
                        f"models won't be generated. Exception: {exception}"
                    )
        elif model:
            for tokenizer_name in ("tokenizer", "tokenizer_2"):
                tokenizer = getattr(model, tokenizer_name, None)
                if tokenizer:
                    export_tokenizer(tokenizer, output / tokenizer_name)
    else:
        logger.warning("Tokenizer won't be converted.")
