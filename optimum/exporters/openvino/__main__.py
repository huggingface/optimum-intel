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
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoTokenizer
from transformers.utils import is_torch_available

from optimum.exporters import TasksManager
from optimum.exporters.onnx import __main__ as optimum_main
from optimum.exporters.onnx.base import OnnxConfig, OnnxConfigWithPast
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.utils.save_utils import maybe_save_preprocessors

from ...intel.openvino.utils import OV_XML_FILE_NAME
from .convert import export_models


logger = logging.getLogger(__name__)

if is_torch_available():
    import torch


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    device: str = "cpu",
    fp16: Optional[bool] = False,
    framework: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_onnx_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    **kwargs_shapes,
):
    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    original_task = task
    task = TasksManager.map_from_synonym(task)

    framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)

    # get the shapes to be used to generate dummy inputs
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = (
            kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
        )

    torch_dtype = None if fp16 is False else torch.float16

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

    model = TasksManager.get_model_from_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework=framework,
        torch_dtype=torch_dtype,
        device=device,
    )

    custom_architecture = False
    is_stable_diffusion = "stable-diffusion" in task
    model_type = "stable-diffusion" if is_stable_diffusion else model.config.model_type.replace("_", "-")

    if not is_stable_diffusion:
        if model_type in TasksManager._UNSUPPORTED_CLI_MODEL_TYPE:
            raise ValueError(
                f"{model_type} is not supported yet. Only {TasksManager._SUPPORTED_CLI_MODEL_TYPE} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        if model.config.model_type.replace("-", "_") not in TasksManager.get_supported_model_type_for_task(
            task, exporter="onnx"
        ):
            custom_architecture = True

    if custom_architecture and custom_onnx_configs is None:
        raise ValueError(
            "Trying to export a model with a custom architecture, but no custom onnx configuration was passed as `custom_onnx_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models."
        )

    if custom_architecture and original_task == "auto":
        raise ValueError(
            f'Automatic task detection is not supported with custom architectures. Please specify the `task` argument. Suggestion: task="{task}" (or task="{task}-with-past" if the model is decoder-based and supports KV cache)'
        )

    if (
        not custom_architecture
        and not is_stable_diffusion
        and task + "-with-past" in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")
    ):
        if original_task == "auto":  # Make -with-past the default if --task was not explicitely specified
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
    onnx_config, models_and_onnx_configs = optimum_main._get_submodels_and_onnx_configs(
        model=model,
        task=task,
        monolith=False,
        custom_onnx_configs=custom_onnx_configs if custom_onnx_configs is not None else {},
        custom_architecture=custom_architecture,
        fn_get_submodels=fn_get_submodels,
    )

    if not is_stable_diffusion:
        needs_pad_token_id = (
            isinstance(onnx_config, OnnxConfigWithPast)
            and getattr(model.config, "pad_token_id", None) is None
            and task in ["text-classification"]
        )
        if needs_pad_token_id:
            if pad_token_id is not None:
                model.config.pad_token_id = pad_token_id
            else:
                try:
                    tok = AutoTokenizer.from_pretrained(model_name_or_path)
                    model.config.pad_token_id = tok.pad_token_id
                except Exception:
                    raise ValueError(
                        "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                    )
        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(output)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output)
        maybe_save_preprocessors(model_name_or_path, output)

        if model.config.is_encoder_decoder and task.startswith("text-generation"):
            raise ValueError(
                f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
                f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
                f" referring to `optimum.exporters.tasks.TaskManager`'s `_TASKS_TO_AUTOMODELS`."
            )

        files_subpaths = None
    else:
        # save the subcomponent configuration
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]
            if hasattr(subcomponent, "save_config"):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
                subcomponent.config.save_pretrained(output / model_name)

        files_subpaths = [os.path.join(name_dir, OV_XML_FILE_NAME) for name_dir in models_and_onnx_configs]

        # Saving the additional components needed to perform inference.
        model.scheduler.save_pretrained(output.joinpath("scheduler"))

        feature_extractor = getattr(model, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output.joinpath("feature_extractor"))

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output.joinpath("tokenizer"))

        tokenizer_2 = getattr(model, "tokenizer_2", None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))

        model.save_config(output)

    export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        output_dir=output,
        output_names=files_subpaths,
        input_shapes=input_shapes,
        device=device,
        model_kwargs=model_kwargs,
    )
