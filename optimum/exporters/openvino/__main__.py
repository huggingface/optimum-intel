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

import gc
import logging
import operator
import warnings
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin
from transformers.utils import is_torch_available

from openvino import Core, save_model
from openvino import Type as ov_Type
from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.tasks import TasksManager
from optimum.intel.utils.import_utils import (
    DIFFUSERS_IMPORT_ERROR,
    is_diffusers_available,
    is_nncf_available,
    is_openvino_tokenizers_available,
    is_transformers_version,
)
from optimum.intel.utils.modeling_utils import (
    _infer_library_from_model_name_or_path,
    _OpenClipForZeroShotImageClassification,
)

from .utils import (
    _MAX_UNCOMPRESSED_SIZE,
    MULTI_MODAL_TEXT_GENERATION_MODELS,
    clear_class_registry,
    deduce_diffusers_dtype,
    load_preprocessors,
    patch_qwenvl_configs,
)


if is_transformers_version(">=", "4.55"):
    from transformers import Mxfp4Config

FORCE_ATTN_MODEL_CLASSES = {"phi3_v": "eager", "gemma2": "sdpa", "llama4": "sdpa"}

if TYPE_CHECKING:
    from optimum.intel.openvino.configuration import OVConfig


if is_torch_available():
    import torch


logger = logging.getLogger(__name__)

# init core before import openvino tokenizers to prevent failed attempt loading extension
core = Core()


def infer_task(
    task,
    model_name_or_path,
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
    library_name: Optional[str] = None,
    trust_remote_code: bool = False,
):
    original_task = task
    task = TasksManager.map_from_synonym(task)
    if task == "auto":
        if library_name == "open_clip":
            task = "zero-shot-image-classification"
        else:
            try:
                task = TasksManager._infer_task_from_model_name_or_path(
                    model_name_or_path=model_name_or_path,
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    library_name=library_name,
                )
            except KeyError as e:
                try:
                    config = AutoConfig.from_pretrained(model_name_or_path)
                    with_past_arch_list = ["MistralForCausalLM", "Zamba2ForCausalLM"]
                    if any(arch in config.architectures for arch in with_past_arch_list):
                        task = "text-generation-with-past"
                except Exception:
                    raise KeyError(
                        f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                    )
            except RequestsConnectionError as e:
                raise RequestsConnectionError(
                    f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                )

    if library_name == "transformers":
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if hasattr(config, "export_model_type"):
            model_type = config.export_model_type
        else:
            model_type = config.model_type
        custom_architecture = model_type not in TasksManager._SUPPORTED_MODEL_TYPE
        if not custom_architecture and task + "-with-past" in TasksManager.get_supported_tasks_for_model_type(
            model_type, exporter="openvino", library_name=library_name
        ):
            # Make -with-past the default if --task was not explicitly specified
            if original_task == "auto":
                task = task + "-with-past"
            else:
                logger.info(
                    f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                    f" if needed, please pass `--task {task}-with-past` to export using the past key values."
                )
    return task


def infer_library_name(
    model_name_or_path: str,
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
) -> str:
    library_name = _infer_library_from_model_name_or_path(
        model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    if library_name == "sentence_transformers":
        logger.warning(
            "Library name is not specified. There are multiple possible variants: `sentence_tenasformers`, `transformers`."
            "`transformers` will be selected. If you want to load your model with the `sentence-transformers` library instead, please set --library sentence_transformers"
        )
        library_name = "transformers"
    return library_name


def _infer_ov_model_class(
    model_name_or_path: str,
    task: str,
    library_name: str,
    cache_dir: str,
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    token: Optional[Union[bool, str]] = None,
):
    from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS

    original_task = task
    task = infer_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        library_name=library_name,
        trust_remote_code=trust_remote_code,
    )
    if library_name is None:
        library_name = infer_library_name(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )

    # Step 1. Obtain the correct OpenVINO model class
    if library_name == "diffusers":
        if not is_diffusers_available():
            raise ValueError(DIFFUSERS_IMPORT_ERROR.format("Export of diffusers models"))

        from diffusers import DiffusionPipeline

        diffusers_config = DiffusionPipeline.load_config(model_name_or_path)
        class_name = diffusers_config.get("_class_name", None)
        ov_class_name = f"OV{class_name}"
        try:
            model_cls = getattr(__import__("optimum.intel", fromlist=[ov_class_name]), ov_class_name)
        except (AttributeError, ImportError) as e:
            raise RuntimeError(f"Wasn't able to locate OpenVINO class for {class_name} diffusion model.") from e
    else:
        try:
            model_cls_name = _HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]
            if library_name == "sentence_transformers":
                model_cls_name = "OVSentenceTransformer"
            model_cls = getattr(__import__("optimum.intel", fromlist=[model_cls_name]), model_cls_name)
        except (AttributeError, ImportError, KeyError) as e:
            raise RuntimeError(f"Wasn't able to locate OpenVINO class for task {original_task} ({task}).") from e
    return model_cls


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    device: str = "cpu",
    framework: str = "pt",
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
    ov_config: "OVConfig" = None,
    stateful: bool = True,
    convert_tokenizer: bool = False,
    library_name: Optional[str] = None,
    model_loading_kwargs: Optional[Dict[str, Any]] = None,
    variant: Optional[str] = None,
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
        framework (`Optional[str]`, defaults to `pt`):
            The framework to use for the ONNX export. Defaults to 'pt' for PyTorch.
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
    from optimum.exporters.openvino.convert import export_from_model

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
        token = use_auth_token

    if framework is None:
        framework = TasksManager.determine_framework(
            model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=cache_dir, token=token
        )

    if library_name is None:
        library_name = infer_library_name(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )

    original_task = task
    task = infer_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        library_name=library_name,
        trust_remote_code=trust_remote_code,
    )

    do_gptq_patching = False
    do_quant_patching = False
    patch_16bit = False
    loading_kwargs = model_loading_kwargs or {}
    if variant is not None:
        loading_kwargs["variant"] = variant
    dtype = loading_kwargs.get("torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype) if dtype != "auto" else dtype

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
        quant_method = quantization_config.get("quant_method", None) if quantization_config else None

        # mxfp4 quantized model will be dequantized to bf16
        if quant_method == "mxfp4" and is_transformers_version(">=", "4.55"):
            dtype = torch.bfloat16
            loading_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

        supported_quant_methods = ["gptq", "awq", "bitnet"]
        do_quant_patching = quant_method in supported_quant_methods
        do_gptq_patching = quant_method == "gptq"
        do_bitnet_patching = quant_method == "bitnet"

        if is_transformers_version(">=", "4.56") and config.model_type in {"qwen2_vl_text", "qwen2_5_vl_text"}:
            patch_qwenvl_configs()

        model_type = config.model_type
        if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
            if custom_export_configs is None:
                raise ValueError(
                    f"Trying to export a {model_type} model, that is a custom or unsupported architecture, but no "
                    "custom export configuration was passed as `custom_export_configs`. Please refer to "
                    "https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models "
                    "for an example on how to export custom models. Please open an issue at "
                    "https://github.com/huggingface/optimum-intel/issues if you would like the model type "
                    f"{model_type} to be supported natively in the OpenVINO export."
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

        # some models force flash_attn attention by default that does not support load model on cpu
        if model_type in FORCE_ATTN_MODEL_CLASSES:
            loading_kwargs["_attn_implementation"] = FORCE_ATTN_MODEL_CLASSES[model_type]
        if model_type == "phi4mm":
            if "activation_checkpointing" in config.audio_processor["config"]:
                config.audio_processor["config"]["activation_checkpointing"] = ""
            config._attn_implementation = "sdpa"
            loading_kwargs["config"] = config
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
        if dtype == "auto":
            dtype = getattr(config, "torch_dtype")

        if (
            dtype is None
            and framework == "pt"
            and (
                task.startswith("text-generation")
                or getattr(config, "model_type", "") in MULTI_MODAL_TEXT_GENERATION_MODELS
            )
            and getattr(config, "torch_dtype", torch.float32) in [torch.float16, torch.bfloat16]
        ):
            if ov_config is not None and ov_config.dtype in {"fp16", "fp32"}:
                dtype = torch.float16 if ov_config.dtype == "fp16" else torch.float32
            elif config.torch_dtype == torch.float16:
                dtype = torch.float16
            elif config.torch_dtype == torch.bfloat16:
                dtype = torch.bfloat16

        if dtype is not None:
            if dtype in [torch.float16, torch.bfloat16]:
                patch_16bit = True
            loading_kwargs["torch_dtype"] = dtype
        # Patch the modules to export of GPTQ models w/o GPU
        if do_quant_patching:
            orig_cuda_check = torch.cuda.is_available
            torch.cuda.is_available = lambda: True

            if do_gptq_patching:
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
            if do_bitnet_patching:
                from transformers.integrations.bitnet import AutoBitLinear

                orig_load_hook = AutoBitLinear.load_hook

                # rewrite load hook to save original weight
                def bitnet_load_hook(self, state_dict, prefix, *args, **kwargs):
                    if (prefix + "weight") in state_dict and state_dict[prefix + "weight"].dtype != self.weight.dtype:
                        self.original_weight = state_dict[prefix + "weight"]
                        w_shape = self.original_weight.shape
                        state_dict[prefix + "weight"] = torch.empty(
                            (w_shape[0] * 4, w_shape[1]), dtype=self.weight.dtype, device="meta"
                        )
                    return state_dict

                AutoBitLinear.load_hook = bitnet_load_hook
    elif library_name == "diffusers":
        _loading_kwargs = {} if variant is None else {"variant": variant}
        if dtype == "auto" or dtype is None:
            dtype = deduce_diffusers_dtype(
                model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                force_download=force_download,
                trust_remote_code=trust_remote_code,
                **_loading_kwargs,
            )
            if (
                dtype in {torch.bfloat16, torch.float16}
                and ov_config is not None
                and ov_config.dtype in {"fp16", "fp32"}
            ):
                dtype = torch.float16 if ov_config.dtype == "fp16" else torch.float32
        if dtype in [torch.float16, torch.bfloat16]:
            loading_kwargs["torch_dtype"] = dtype
            patch_16bit = True
        if loading_kwargs.get("torch_dtype") == "auto":
            loading_kwargs["torch_dtype"] = dtype

    try:
        if library_name == "open_clip":
            model = _OpenClipForZeroShotImageClassification.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        else:
            # remote code models like phi3_v internvl2, minicpmv, internvl2, nanollava, maira2 should be loaded using AutoModelForCausalLM and not AutoModelForImageTextToText
            # TODO: use config.auto_map to load remote code models instead (for other models we can directly use config.architectures)
            task_model_loading = task
            if library_name == "transformers":
                has_remote_code = hasattr(config, "auto_map")
                if has_remote_code and trust_remote_code and task == "image-text-to-text":
                    task_model_loading = "text-generation"

            model = TasksManager.get_model_from_task(
                task_model_loading,
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

        if hasattr(model.config, "export_model_type"):
            model_type = model.config.export_model_type
        else:
            model_type = model.config.model_type

        if original_task == "auto":
            synonyms_for_task = sorted(TasksManager.synonyms_for_task(task))
            if synonyms_for_task:
                synonyms_for_task = ", ".join(synonyms_for_task)
                possible_synonyms = f" (possible synonyms are: {synonyms_for_task})"
            else:
                possible_synonyms = ""
            logger.info(f"Automatic task detection to {task}{possible_synonyms}.")

        preprocessors = load_preprocessors(
            model_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code, model_type=model_type
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
            patch_16bit_model=patch_16bit,
            **kwargs_shapes,
        )

        if convert_tokenizer:
            maybe_convert_tokenizers(library_name, output, model, preprocessors, task=task)

        clear_class_registry()
        del model
        gc.collect()
    finally:
        # Unpatch modules after quantized model export
        if do_quant_patching:
            torch.cuda.is_available = orig_cuda_check
            if do_gptq_patching:
                GPTQQuantizer.post_init_model = orig_post_init_model
            if do_bitnet_patching:
                AutoBitLinear.load_hook = orig_load_hook


def main_quantize(
    model_name_or_path: str,
    task: str,
    library_name: str,
    quantization_config: Union[Dict, "OVQuantizationConfigBase"],  # noqa: F821
    output: Path,
    cache_dir: str,
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    token: Optional[Union[bool, str]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Apply quantization to the OpenVINO model exported to `output` directory.

    Args:
        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the original model repository.
        task (`str`):
            The task to export the model for.
        library_name (`str`):
            The library name.
        quantization_config (`Union[Dict, OVQuantizationConfigBase]`):
            The quantization configuration to use.
        output (`Path`):
            Path indicating the directory where the exported OpenVINO model is stored and where to save the
            quantized model.
        cache_dir (`Optional[str]`, defaults to `None`):
            Path indicating where to store cache. The default Hugging Face cache path will be used by default.
        trust_remote_code (`bool`, defaults to `False`):
            Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
            you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
            model repository.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        revision (`str`, defaults to `"main"`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        token (Optional[Union[bool, str]], defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_export_configs` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).
    """
    from optimum.intel.openvino.configuration import _GPTOSSQuantizationConfig

    if not is_nncf_available():
        raise ImportError("Applying quantization requires nncf, please install it with `pip install nncf`")

    # Step 1. Obtain the correct OpenVINO model class
    model_cls = _infer_ov_model_class(
        model_name_or_path=model_name_or_path,
        task=task,
        library_name=library_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        subfolder=subfolder,
        revision=revision,
        token=token,
    )

    # Step 2. A special case for quantization of GPT-OSS models
    # TODO: remove this workaround when possible
    if isinstance(quantization_config, _GPTOSSQuantizationConfig):
        # A workaround for GPT-OSS model is required to run quantization twice, this way it is possible to
        # selectively quantize some weights to 4 bits and some to 8 bits.
        from optimum.intel.openvino import OVModelForCausalLM
        from optimum.intel.openvino.quantization import _weight_only_quantization

        if model_cls != OVModelForCausalLM:
            raise ValueError(
                "GPT-OSS quantization is only supported for causal language models. "
                f"Expected model class OVModelForCausalLM but got {model_cls}."
            )

        ov_model_path = output / model_cls._all_ov_model_paths["model"]
        ov_model = core.read_model(ov_model_path)
        _weight_only_quantization(ov_model, quantization_config.quantization_config1)
        _weight_only_quantization(ov_model, quantization_config.quantization_config2, verify_not_optimized=False)

        # Save to a temporary path and replace the original model files to avoid reading and writing to the same file
        compressed_ov_model_path = ov_model_path.parent / f"{ov_model_path.stem}_compressed.xml"
        save_model(ov_model, compressed_ov_model_path, compress_to_fp16=False)
        del ov_model
        gc.collect()
        ov_model_path.unlink()
        ov_model_path.with_suffix(".bin").unlink()
        compressed_ov_model_path.rename(ov_model_path)
        compressed_ov_model_path.with_suffix(".bin").rename(ov_model_path.with_suffix(".bin"))
        return

    # Step 3. Load the exported model
    model = model_cls.from_pretrained(
        output,
        compile=False,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        use_cache=True if task.endswith("with-past") else None,
        **(model_kwargs or {}),
    )

    # Step 4. Apply quantization and save the quantized model
    model._apply_quantization(
        quantization_config,
        compile_only=False,
        compile_model=False,
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        save_directory=output,
        immediate_save=True,
    )


def prepare_quantization_config(
    output: Path,
    model_name_or_path: str,
    task: str,
    library_name: str,
    cache_dir: str,
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    token: Optional[Union[bool, str]] = None,
    # Quantization parameters
    weight_format: Optional[str] = None,
    quant_mode: Optional[str] = None,
    ratio: Optional[float] = None,
    sym: Optional[bool] = None,
    group_size: Optional[int] = None,
    all_layers: Optional[bool] = None,
    dataset: Optional[str] = None,
    num_samples: Optional[int] = None,
    awq: Optional[bool] = None,
    sensitivity_metric: Optional[str] = None,
    scale_estimation: Optional[bool] = None,
    gptq: Optional[bool] = None,
    lora_correction: Optional[bool] = None,
    quantization_statistics_path: Optional[str] = None,
    backup_precision: Optional[str] = None,
    group_size_fallback: Optional[str] = None,
    smooth_quant_alpha: Optional[float] = None,
) -> Optional["OVQuantizationConfigBase"]:  # noqa: F821
    """
    Prepare the quantization configuration based on the provided parameters.
    Full description of quantization-related parameters can be found at OVExportCommand class.

    Args:
        output (`Path`):
            Path indicating the directory where the exported OpenVINO model is stored.
        model_name_or_path (`str`):
            Model ID on huggingface.co or path on disk to the original model repository.
        task (`str`):
            The task to export the model for.
        library_name (`str`):
            The library name.
        cache_dir (`str`):
            Path indicating where to store cache. The default Hugging Face cache path will be used by default.
        trust_remote_code (`bool`, defaults to `False`):
            Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
            you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
            model repository.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        revision (`str`, defaults to `"main"`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
        token (Optional[Union[bool, str]], defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        weight_format (`Optional[str]`, defaults to `None`):
            The weight format of the exported model.
        quant_mode (`Optional[str]`, defaults to `None`):
            Quantization precision mode.
        ratio (`Optional[float]`, defaults to `None`):
            A parameter used when applying 4-bit quantization to control the ratio between 4-bit and 8-bit quantization.
        sym (`Optional[bool]`, defaults to `None`):
            Whether to apply symmetric quantization.
        group_size (`Optional[int]`, defaults to `None`):
            The group size to use for quantization.
        all_layers (`Optional[bool]`, defaults to `None`):
            Whether embeddings and last MatMul layers should be compressed to INT4.
        dataset (`Optional[str]`, defaults to `None`):
            The dataset used for data-aware compression or quantization with NNCF.
        num_samples (`Optional[int]`, defaults to `None`):
            The maximum number of samples to take from the dataset for quantization.
        awq (`Optional[bool]`, defaults to `None`):
            Whether to apply AWQ algorithm.
        sensitivity_metric (`Optional[str]`, defaults to `None`):
            The sensitivity metric for assigning quantization precision to layers.
        scale_estimation (`Optional[bool]`, defaults to `None`):
            Indicates whether to apply a scale estimation algorithm.
        gptq (`Optional[bool]`, defaults to `None`):
            Indicates whether to apply GPTQ algorithm.
        lora_correction (`Optional[bool]`, defaults to `None`):
            Indicates whether to apply LoRA Correction algorithm.
        quantization_statistics_path (`Optional[str]`, defaults to `None`):
            Directory path to dump/load data-aware weight-only quantization statistics.
        backup_precision (`Optional[str]`, defaults to `None`):
            Defines a backup precision for mixed-precision weight compression.
        group_size_fallback (`Optional[str]`, defaults to `None`):
            Specifies how to handle operations that do not support the given group size.
        smooth_quant_alpha (`Optional[float]`, defaults to `None`):
            SmoothQuant alpha parameter that improves the distribution of activations before MatMul layers and
            reduces quantization error.
    Returns:
        `Optional[OVQuantizationConfigBase]`: The prepared quantization configuration or `None` if no quantization is to be applied.
    """
    from optimum.intel.openvino.configuration import (
        _DEFAULT_4BIT_WQ_CONFIG,
        _quantization_config_from_dict,
        get_default_quantization_config,
    )

    no_compression_parameter_provided = _no_compression_parameter_provided(
        ratio,
        group_size,
        sym,
        all_layers,
        dataset,
        num_samples,
        awq,
        scale_estimation,
        gptq,
        lora_correction,
        sensitivity_metric,
        backup_precision,
    )

    no_quantization_parameter_provided = _no_quantization_parameter_provided(
        sym, dataset, num_samples, smooth_quant_alpha
    )

    wc_config = None
    if weight_format is None and quant_mode is None:
        if not no_compression_parameter_provided or quantization_statistics_path is not None:
            raise ValueError(
                "Some compression parameters are provided, but the weight format is not specified. "
                "Please provide it with weight_format argument."
            )
        if not no_quantization_parameter_provided:
            raise ValueError(
                "Some quantization parameters are provided, but the quantization mode is not specified. "
                "Please provide it with quant_mode argument."
            )
    else:
        # wc_config may be needed in case of weight-only quantization or mixed precision quantization
        wc_config = _prepare_wc_config(
            weight_format,
            ratio,
            sym,
            group_size,
            all_layers,
            dataset,
            num_samples,
            awq,
            sensitivity_metric,
            scale_estimation,
            gptq,
            lora_correction,
            quantization_statistics_path,
            backup_precision,
            group_size_fallback,
            _DEFAULT_4BIT_WQ_CONFIG,
        )
    if weight_format is not None and quant_mode is not None:
        raise ValueError("Both weight_format and quant_mode arguments are provided. Please provide only one of them.")
    default_quantization_config = None
    if weight_format is not None or quant_mode is not None:
        default_quantization_config = get_default_quantization_config(model_name_or_path, weight_format, quant_mode)

    # Step 1. If weight_format argument is provided, construct a weight-only quantization config
    if weight_format not in [None, "fp16", "fp32"]:
        quantization_config = wc_config
        # For int4/int8 quantization if no parameter is provided, then use the default config if exists
        if weight_format in ["int4", "int8"]:
            if no_compression_parameter_provided:
                if default_quantization_config is not None:
                    quantization_config = default_quantization_config
                    logger.info(
                        f"Applying the default quantization config for {model_name_or_path}: {quantization_config}."
                    )
                elif weight_format == "int4":
                    quantization_config = _DEFAULT_4BIT_WQ_CONFIG
                    logger.info(f"Applying a default quantization config: {quantization_config}.")
                quantization_config["statistics_path"] = quantization_statistics_path
            elif default_quantization_config is not None:
                logger.info(
                    f"For the given model, we recommend the following `quantization_config` : {default_quantization_config}."
                )
        quantization_config = _quantization_config_from_dict(quantization_config)
        return quantization_config

    # Step 2. If quant_mode argument is provided, construct a full quantization config
    if quant_mode is not None:
        if no_quantization_parameter_provided and default_quantization_config is not None:
            quantization_config = default_quantization_config
            logger.info(f"Applying the default quantization config for {model_name_or_path}: {quantization_config}.")
        else:
            if dataset is None:
                raise ValueError(
                    "Dataset is required for full quantization. Please provide it with --dataset argument."
                )
            if quant_mode in [
                "cb4_f8e4m3",
                "int4_f8e4m3",
                "int4_f8e5m2",
            ]:
                if library_name == "diffusers":
                    raise NotImplementedError("Mixed precision quantization isn't supported for diffusers.")

                wc_dtype, q_dtype = quant_mode.split("_")
                wc_config["dtype"] = wc_dtype

                q_config = _prepare_q_config(quant_mode, sym, dataset, num_samples, smooth_quant_alpha)
                q_config["dtype"] = q_dtype

                quantization_config = {
                    "weight_quantization_config": wc_config,
                    "full_quantization_config": q_config,
                    "num_samples": num_samples,
                    "dataset": dataset,
                }
            else:
                if quantization_statistics_path is not None:
                    logger.warning(
                        "The --quantization-statistics-path argument is only applicable for weight-only "
                        "quantization. It will be ignored."
                    )
                quantization_config = _prepare_q_config(quant_mode, sym, dataset, num_samples, smooth_quant_alpha)
        quantization_config = _quantization_config_from_dict(quantization_config)
        return quantization_config

    # Step 3. No quantization parameters provided, apply int8 weight quantization only to models larger than 1B params
    if weight_format not in ["fp16", "fp32"]:
        model_cls = _infer_ov_model_class(
            model_name_or_path=model_name_or_path,
            task=task,
            library_name=library_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
            revision=revision,
            token=token,
        )
        quantization_config = prepare_model_size_based_quantization_config(output, model_cls)
        return quantization_config

    return None


def prepare_model_size_based_quantization_config(
    model_dir: Path,
    model_cls: Type["OVBaseModel"],  # noqa: F821
) -> Optional["OVPipelineQuantizationConfig"]:  # noqa: F821
    """
    Prepare a quantization configuration based on the model size. If a model has more than 1 billion parameters,
    an 8-bit weight-only quantization configuration will be returned for it. Otherwise, no quantization will be applied.

    Args:
        model_dir (`Path`):
            Path indicating the directory where the exported OpenVINO model is stored.
        model_cls (`Type[OVBaseModel]`):
            The OpenVINO model class.
    Returns:
        `Optional[OVPipelineQuantizationConfig]`: The quantization configuration to use or None if no quantization is needed.
    """
    from optimum.intel.openvino.configuration import OVPipelineQuantizationConfig, OVWeightQuantizationConfig

    ov_model_names_to_quantize = []
    for ov_model_name, ov_model_rel_path in model_cls._all_ov_model_paths.items():
        ov_model_path = model_dir / ov_model_rel_path
        if not ov_model_path.exists():
            continue
        ov_model = core.read_model(ov_model_path)
        num_parameters = 0
        for op in ov_model.get_ops():
            if op.get_type_name() == "Constant" and op.get_element_type() in [
                ov_Type.f16,
                ov_Type.f32,
                ov_Type.bf16,
            ]:
                num_parameters += reduce(operator.mul, op.shape, 1)
        if num_parameters >= _MAX_UNCOMPRESSED_SIZE:
            ov_model_names_to_quantize.append(ov_model_name)
    quantization_config = None
    if ov_model_names_to_quantize:
        wq_config = OVWeightQuantizationConfig(bits=8)
        quantization_config = OVPipelineQuantizationConfig(
            {model_name: wq_config for model_name in ov_model_names_to_quantize}
        )
    return quantization_config


def _prepare_wc_config(
    weight_format: Optional[str],
    ratio: Optional[float],
    sym: Optional[bool],
    group_size: Optional[int],
    all_layers: Optional[bool],
    dataset: Optional[str],
    num_samples: Optional[int],
    awq: Optional[bool],
    sensitivity_metric: Optional[str],
    scale_estimation: Optional[bool],
    gptq: Optional[bool],
    lora_correction: Optional[bool],
    quantization_statistics_path: Optional[str],
    backup_precision: Optional[str],
    group_size_fallback: Optional[str],
    default_configs: Dict[str, Any],
):
    is_int8 = weight_format == "int8"
    return {
        "bits": 8 if is_int8 else 4,
        "ratio": 1.0 if is_int8 else (ratio or default_configs["ratio"]),
        "sym": sym or False,
        "group_size": -1 if is_int8 else group_size,
        "all_layers": None if is_int8 else all_layers,
        "dataset": dataset,
        "num_samples": num_samples,
        "quant_method": "awq" if awq else "default",
        "sensitivity_metric": sensitivity_metric,
        "scale_estimation": scale_estimation,
        "gptq": gptq,
        "lora_correction": lora_correction,
        "dtype": weight_format,
        "backup_precision": backup_precision,
        "statistics_path": quantization_statistics_path,
        "group_size_fallback": group_size_fallback,
    }


def _prepare_q_config(
    quant_mode: str,
    sym: Optional[bool],
    dataset: Optional[str],
    num_samples: Optional[int],
    smooth_quant_alpha: Optional[float],
):
    return {
        "dtype": quant_mode,
        "bits": 8,
        "sym": sym or False,
        "dataset": dataset,
        "num_samples": num_samples,
        "smooth_quant_alpha": smooth_quant_alpha,
    }


def _no_compression_parameter_provided(
    ratio: Optional[float],
    group_size: Optional[int],
    sym: Optional[bool],
    all_layers: Optional[bool],
    dataset: Optional[str],
    num_samples: Optional[int],
    awq: Optional[bool],
    scale_estimation: Optional[bool],
    gptq: Optional[bool],
    lora_correction: Optional[bool],
    sensitivity_metric: Optional[str],
    backup_precision: Optional[str],
):
    # Except statistics path
    return all(
        (
            it is None
            for it in (
                ratio,
                group_size,
                sym,
                all_layers,
                dataset,
                num_samples,
                awq,
                scale_estimation,
                gptq,
                lora_correction,
                sensitivity_metric,
                backup_precision,
            )
        )
    )


def _no_quantization_parameter_provided(
    sym: Optional[bool],
    dataset: Optional[str],
    num_samples: Optional[int],
    smooth_quant_alpha: Optional[float],
):
    return all(
        (
            it is None
            for it in (
                sym,
                dataset,
                num_samples,
                smooth_quant_alpha,
            )
        )
    )


def maybe_convert_tokenizers(library_name: str, output: Path, model=None, preprocessors=None, task=None):
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
        task (`str`, *optional*, defaults to None):
            The task to export the model for. Affects tokenizer conversion parameters.
    """
    from optimum.exporters.openvino.convert import export_tokenizer

    if is_openvino_tokenizers_available():
        if library_name != "diffusers" and preprocessors:
            processor_chat_template = None
            tokenizer = next(filter(lambda it: isinstance(it, PreTrainedTokenizerBase), preprocessors), None)
            if len(preprocessors) > 1:
                for processor in preprocessors:
                    if isinstance(processor, ProcessorMixin) and hasattr(processor, "chat_template"):
                        processor_chat_template = processor.chat_template
            if tokenizer:
                try:
                    export_tokenizer(tokenizer, output, task=task, processor_chat_template=processor_chat_template)
                except Exception as exception:
                    logger.warning(
                        "Could not load tokenizer using specified model ID or path. OpenVINO tokenizer/detokenizer "
                        f"models won't be generated. Exception: {exception}"
                    )
        elif model:
            for tokenizer_name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
                tokenizer = getattr(model, tokenizer_name, None)
                if tokenizer:
                    export_tokenizer(tokenizer, output / tokenizer_name, task=task)
    else:
        logger.warning("Tokenizer won't be converted.")
