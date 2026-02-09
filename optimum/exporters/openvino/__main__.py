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
import importlib.util
import logging
import operator
import os
import shutil
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin
from transformers.utils import is_torch_available

from openvino import Core, Type, save_model
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


def update_config_for_eagle3(config):
    moduler_name = "optimum.exporters.openvino.model_patcher"
    spec = importlib.util.find_spec(moduler_name)
    if spec and spec.origin:
        moduler_path = os.path.dirname(spec.origin)
        config.auto_map = {
            "AutoModel": moduler_path + "--model_patcher.LlamaEagle3Model",
            "AutoModelForCausalLM": moduler_path + "--model_patcher.LlamaEagle3ForCausalLM",
        }
    config.tie_word_embeddings = False
    return config


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
            The framework to use to load the model before conversion. Defaults to 'pt' for PyTorch.
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
            especially useful when exporting a custom architecture that needs to be splitted in multiple components (e.g. encoder-decoder). If unspecified with custom models, optimum will try to use the default submodels used for the given task, with no guarantee of success.
        stateful (`bool`, defaults to `True`):
            Produce stateful model where all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. Applicable only for decoder models.
        **kwargs_shapes (`Dict`):
            Shapes to use during inference. This argument allows to override the default shapes used during the OpenVINO export.

    Example usage:
    ```python
    >>> from optimum.exporters.openvino import main_export

    >>> main_export("gpt2", output="gpt2_ov/")
    ```
    """
    from optimum.exporters.openvino.convert import export_from_model
    from optimum.intel.openvino.configuration import _GPTOSSQuantizationConfig

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

        # update config to load eagle3 models
        archs = getattr(config, "architectures", None)
        if isinstance(archs, list) and len(archs) > 0 and archs[0] == "LlamaForCausalLMEagle3":
            loading_kwargs["config"] = update_config_for_eagle3(config)

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
                f"Asked to export a {model_type} model for the task {task}{autodetected_message}, but the Optimum OpenVINO exporter only supports the tasks {', '.join(model_tasks.keys())} for {model_type}. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum-intel/issues if you would like the task {task} to be supported in the OpenVINO export for {model_type}."
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

        submodel_paths = export_from_model(
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

        # TODO: Remove GPT-OSS workaround when possible
        quantization_config = None if ov_config is None else ov_config.quantization_config
        if not quantization_config or isinstance(quantization_config, _GPTOSSQuantizationConfig):
            _apply_model_size_based_quantization(submodel_paths, ov_config, output)
    finally:
        # Unpatch modules after quantized model export
        if do_quant_patching:
            torch.cuda.is_available = orig_cuda_check
            if do_gptq_patching:
                GPTQQuantizer.post_init_model = orig_post_init_model
            if do_bitnet_patching:
                AutoBitLinear.load_hook = orig_load_hook


def _main_quantize(
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
            Model ID on huggingface.co or path on disk to the model repository.
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
    from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS

    # Step 0. Infer task and library name if needed
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

    # Step 2. Load the exported model
    model = model_cls.from_pretrained(
        output,
        compile=False,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        use_cache=task.endswith("with-past"),
        **(model_kwargs or {}),
    )

    # Step 3. Apply quantization and save the quantized model
    model._apply_quantization(
        quantization_config,
        compile_only=False,
        compile_model=False,
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        save_directory=output,
        immediate_save=True,
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


def _apply_model_size_based_quantization(submodel_paths: List[str], ov_config: "OVConfig", output: Union[str, Path]):
    """
    Apply weight-only quantization to int8_asym to submodels larger than 1B parameters.
    """
    # TODO: Refactor the code below in the following way:
    #   1. Create a OVPipelineQuantizationConfig based on each submodel size
    #   2. Run _main_quantize() with the created quantization config
    # TODO: Apply default int8 and ignored scope configs from configuration.py if matches
    for submodel_path in submodel_paths:
        submodel_path = Path(output) / submodel_path

        if (not submodel_path.is_file()) or (submodel_path.stat().st_size == 0):
            raise RuntimeError(
                f"An issue happened during export : {submodel_path.name} was not converted and saved as expected."
            )

        submodel = core.read_model(submodel_path)

        quantization_config = None
        if ov_config is None:
            num_parameters = 0
            for op in submodel.get_ops():
                if op.get_type_name() == "Constant" and op.get_element_type() in [Type.f16, Type.f32, Type.bf16]:
                    num_parameters += reduce(operator.mul, op.shape, 1)
                del op
            if num_parameters >= _MAX_UNCOMPRESSED_SIZE:
                if is_nncf_available():
                    quantization_config = {"bits": 8, "sym": False}
                    logger.info("The model weights will be quantized to int8_asym.")
                else:
                    logger.warning(
                        "The model will be converted with no weights quantization. Quantization of the weights to int8 "
                        "requires nncf. Please install it with `pip install nncf`"
                    )
                    break
        else:
            # TODO: Remove this case when the workaround for GPT-OSS is removed
            quantization_config = ov_config.quantization_config
        if quantization_config is None:
            del submodel
            gc.collect()
            continue

        if not is_nncf_available():
            raise ImportError("Quantization of the weights requires nncf, please install it with `pip install nncf`")

        from optimum.intel.openvino.configuration import _GPTOSSQuantizationConfig
        from optimum.intel.openvino.quantization import _weight_only_quantization

        if isinstance(quantization_config, _GPTOSSQuantizationConfig):
            # A workaround for GPT-OSS model is required to run quantization twice, this way it is possible to
            # selectively quantize some weights to 4 bits and some to 8 bits.
            _weight_only_quantization(submodel, quantization_config.quantization_config1)
            _weight_only_quantization(submodel, quantization_config.quantization_config2, verify_not_optimized=False)
        else:
            _weight_only_quantization(submodel, quantization_config)
        compressed_submodel_path = submodel_path.parent / f"{submodel_path.stem}_compressed.xml"
        save_model(submodel, compressed_submodel_path, compress_to_fp16=False)
        del submodel
        gc.collect()

        submodel_path.unlink()
        submodel_path.with_suffix(".bin").unlink()
        compressed_submodel_path.rename(submodel_path)
        compressed_submodel_path.with_suffix(".bin").rename(submodel_path.with_suffix(".bin"))


def _merge_move(src: Path, dest: Path):
    """
    Move src to dest.

    - If src is a directory:
        - If dest does not exist: rename src -> dest.
        - If dest is a directory: merge src into dest recursively.
        - If dest is a file: replace file with src directory (delete file, then rename).

    - If src is a file:
        - If dest does not exist: rename src -> dest.
        - If dest is a file: overwrite file (delete dest, then rename).
        - If dest is a directory: replace directory with src file (delete directory recursively, then rename).
    """

    def _safe_rename(src: Path, dest: Path):
        # Try to rename first, fall back to shutil.move if it fails (e.g., cross-device move)
        try:
            src.rename(dest)
        except OSError:
            shutil.move(str(src), str(dest))

    dest_exists = dest.exists()
    dest_is_dir = dest_exists and dest.is_dir()
    if src.is_dir():
        if not dest_exists:
            # No conflict: just rename
            _safe_rename(src, dest)
        elif dest_is_dir:
            # Merge src into dest recursively
            for child in src.iterdir():
                _merge_move(child, dest / child.name)
            # Remove src once empty
            src.rmdir()
        else:
            # dest exists and is a file: replace file with directory
            dest.unlink()
            _safe_rename(src, dest)
    else:
        if not dest_exists:
            # No conflict: just rename
            _safe_rename(src, dest)
        elif dest_is_dir:
            # Replace directory (recursively) with file
            shutil.rmtree(dest)
            _safe_rename(src, dest)
        else:
            # dest is a file: overwrite
            dest.unlink()
            _safe_rename(src, dest)
