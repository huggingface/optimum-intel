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

import copy
import functools
import gc
import inspect
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.generation import GenerationMixin
from transformers.models.speecht5.modeling_speecht5 import SpeechT5HifiGan
from transformers.utils import is_tf_available, is_torch_available

from openvino import Model, save_model
from openvino.exceptions import OVTypeError
from openvino.tools.ovc import convert_model
from optimum.exporters import TasksManager
from optimum.exporters.utils import (
    DECODER_NAME,
    ENCODER_NAME,
    _get_submodels_for_export_encoder_decoder,
    get_diffusion_models_for_export,
)
from optimum.exporters.utils import (
    _get_submodels_and_export_configs as _default_get_submodels_and_export_configs,
)
from optimum.intel.utils.import_utils import (
    _diffusers_version,
    _nncf_version,
    _open_clip_version,
    _optimum_intel_version,
    _optimum_version,
    _timm_version,
    _torch_version,
    _transformers_version,
    compare_versions,
    is_openvino_tokenizers_version,
    is_tokenizers_version,
    is_transformers_version,
)
from optimum.utils import DEFAULT_DUMMY_SHAPES, is_diffusers_available

from ...intel.utils.import_utils import is_nncf_available
from ...intel.utils.modeling_utils import _infer_library_from_model_or_model_class
from .model_patcher import patch_model_with_bettertransformer
from .stateful import (
    ensure_export_task_support_stateful,
    ensure_model_type_support_stateful,
    ensure_stateful_is_available,
    patch_stateful,
)
from .utils import (
    MULTI_MODAL_TEXT_GENERATION_MODELS,
    OV_XML_FILE_NAME,
    _get_input_info,
    _get_open_clip_submodels_fn_and_export_configs,
    allow_skip_tracing_check,
    clear_class_registry,
    remove_none_from_dummy_inputs,
    save_config,
    save_preprocessors,
    set_simplified_chat_template,
)


logger = logging.getLogger(__name__)

if is_torch_available():
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel

if is_diffusers_available():
    from diffusers import DiffusionPipeline, ModelMixin

if is_tf_available():
    from transformers.modeling_tf_utils import TFPreTrainedModel


if TYPE_CHECKING:
    from optimum.exporters.onnx.base import OnnxConfig
    from optimum.intel.openvino.configuration import OVConfig


def _set_runtime_options(
    models_and_export_configs: Dict[
        str,
        Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin", "DiffusionPipeline"], "OnnxConfig"],
    ],
    task: str,
    library_name: str,
    quantized_model: bool,
):
    for model_name in models_and_export_configs.keys():
        _, sub_export_config = models_and_export_configs[model_name]
        if not hasattr(sub_export_config, "runtime_options"):
            sub_export_config.runtime_options = {}
        if (
            "text-generation" in task
            or ("image-text-to-text" in task and model_name == "language_model")
            or getattr(sub_export_config, "stateful", False)
        ):
            sub_export_config.runtime_options["ACTIVATIONS_SCALE_FACTOR"] = "8.0"
        if not quantized_model and (
            "text-generation" in task
            or ("image-text-to-text" in task and model_name == "language_model")
            or getattr(sub_export_config, "stateful", False)
        ):
            sub_export_config.runtime_options["KV_CACHE_PRECISION"] = "f16"


def _save_model(
    model,
    path: str,
    ov_config: Optional["OVConfig"] = None,
    library_name: Optional[str] = None,
    config: "OnnxConfig" = None,
):
    compress_to_fp16 = ov_config is not None and ov_config.dtype == "fp16"
    model = _add_version_info_to_model(model, library_name)

    runtime_options = config.runtime_options if hasattr(config, "runtime_options") else {}
    model = _add_runtime_options_to_rt_info(model, runtime_options)
    save_model(model, path, compress_to_fp16)
    del model
    gc.collect()


def export(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin", "DiffusionPipeline"],
    config: "OnnxConfig",
    output: Path,
    opset: Optional[int] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
    stateful: bool = True,
    patch_16bit_model: bool = False,
    library_name: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a Pytorch or TensorFlow model to an OpenVINO Intermediate Representation.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported model.
        opset (`Optional[int]`, defaults to `None`):
            The version of the ONNX operator set to use.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        ov_config (`OVConfig`, *optional*):
            The configuration containing the parameters related to quantization.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the exporter.
        stateful (`bool`, defaults to `True`):
            Produce stateful model where all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. Applicable only for decoder models.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are installed. "
            "Please install torch or tensorflow first."
        )

    if "diffusers" in str(model.__class__) and not is_diffusers_available():
        raise ImportError("The package `diffusers` is required to export diffusion models to OpenVINO.")

    if stateful:
        # This will be checked anyway after the model conversion, but checking it earlier will save time for a user if not suitable version is used
        stateful = ensure_stateful_is_available()

    if is_torch_available() and isinstance(model, nn.Module):
        return export_pytorch(
            model,
            config,
            opset,
            output,
            device=device,
            input_shapes=input_shapes,
            ov_config=ov_config,
            model_kwargs=model_kwargs,
            stateful=stateful,
            patch_16bit_model=patch_16bit_model,
            library_name=library_name,
        )

    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        output.parent.mkdir(parents=True, exist_ok=True)
        if opset is None:
            opset = config.DEFAULT_ONNX_OPSET
        if device == "cuda":
            raise RuntimeError("`tf2onnx` does not support export on CUDA device.")
        if input_shapes is not None:
            logger.info("`input_shapes` argument is not supported by the Tensorflow ONNX export and will be ignored.")
        return export_tensorflow(model, config, opset, output, ov_config=ov_config, library_name=library_name)

    else:
        raise RuntimeError(
            "You either provided a PyTorch model with only TensorFlow installed, or a TensorFlow model with only PyTorch installed."
        )


def export_tensorflow(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: "OnnxConfig",
    opset: int,
    output: Path,
    ov_config: Optional["OVConfig"] = None,
    library_name: Optional[str] = None,
):
    """
    Export the TensorFlow model to OpenVINO format.

    Args:
        model (Union[): The model to export.
        config (OnnxConfig): The configuration of the model.
        opset (int): The ONNX opset version to use.
        output (Path): The path to save the model.

    Returns:
        input_names: list of input names from ONNX configuration
        output_names: list of output names from ONNX configuration
        bool:  True if the model was exported successfully.
    """
    from optimum.exporters.onnx.convert import export_tensorflow as export_tensorflow_onnx

    onnx_path = Path(output).with_suffix(".onnx")
    input_names, output_names = export_tensorflow_onnx(model, config, opset, onnx_path)
    ov_model = convert_model(str(onnx_path))

    library_name = _infer_library_from_model_or_model_class(model=model, library_name=library_name)

    _save_model(
        ov_model,
        output.parent / output,
        ov_config=ov_config,
        library_name=library_name,
        config=config,
    )
    del ov_model
    gc.collect()
    return input_names, output_names, True


def export_pytorch_via_onnx(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: "OnnxConfig",
    opset: int,
    output: Path,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
    library_name: Optional[str] = None,
):
    """
    Exports a PyTorch model to an OpenVINO Intermediate Representation via ONNX export.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported model.
        device (`str`, defaults to `"cpu"`):
            The device on which the model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the exporter.
        model_kwargs (optional[Dict[str, Any]], defaults to `None`):
            Additional kwargs for model export.
        ov_config (`OVConfig`, *optional*):
            The configuration containing the parameters related to quantization.

    Returns:
        `Tuple[List[str], List[str], bool]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration and boolean flag - was legacy ONNX path were applied to model or not.
    """
    import torch

    from optimum.exporters.onnx.convert import export_pytorch as export_pytorch_to_onnx

    output = Path(output)
    orig_torch_onnx_export = torch.onnx.export
    torch.onnx.export = functools.partial(orig_torch_onnx_export, do_constant_folding=False)
    model.config.torchscript = False
    model.config.return_dict = True
    onnx_output = output.with_suffix(".onnx")
    input_names, output_names = export_pytorch_to_onnx(
        model, config, opset, onnx_output, device, input_shapes, model_kwargs
    )
    torch.onnx.export = orig_torch_onnx_export
    ov_model = convert_model(str(onnx_output))

    library_name = _infer_library_from_model_or_model_class(model=model, library_name=library_name)

    _save_model(
        ov_model,
        output.parent / OV_XML_FILE_NAME if output.suffix != ".xml" else output,
        ov_config=ov_config,
        library_name=library_name,
        config=config,
    )
    del ov_model
    gc.collect()
    return input_names, output_names, True


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: "OnnxConfig",
    opset: int,
    output: Path,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
    stateful: bool = False,
    patch_16bit_model: bool = False,
    library_name: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to an OpenVINO Intermediate Representation.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported model.
        device (`str`, defaults to `"cpu"`):
            The device on which the model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the exporter.
        model_kwargs (optional[Dict[str, Any]], defaults to `None`):
            Additional kwargs for model export
        ov_config (`OVConfig`, *optional*):
            The configuration containing the parameters related to quantization.
        stateful (`bool`, defaults to `False`):
            Produce stateful model where all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. Applicable only for decoder models.

    Returns:
        `Tuple[List[str], List[str], bool]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration and boolean flag - was legacy ONNX path were applied to model or not.
    """
    import torch
    from torch.utils._pytree import tree_map

    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from optimum.exporters.utils import check_dummy_inputs_are_allowed

    logger.info(f"Using framework PyTorch: {torch.__version__}")
    output = Path(output)

    if stateful:
        # Trigger bettertransformer together with stateful model because OpenVINO HW-dependent transformations expect
        # both of them are applied to demonstrate the best performance.
        # TODO: Consider applying bettertransformer regardless of stateful flag -- requires additional validation.
        model = patch_model_with_bettertransformer(model)
        # TODO: Consider unpatching model after export is done in the end of this function.
        #       Now it is left as-is because the model is not expected to be used after call export_pytorch, and
        #       this function is one of the _internal_ steps in a bigger model conversion pipeline.

    with torch.no_grad():
        if hasattr(model, "config"):
            model.config.torchscript = False
            model.config.return_dict = True
        model.eval()

        # Check if we need to override certain configuration item
        if config.values_override is not None:
            logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
            for override_config_key, override_config_value in config.values_override.items():
                logger.info(f"\t- {override_config_key} -> {override_config_value}")
                setattr(model.config, override_config_key, override_config_value)

        if input_shapes is None:
            input_shapes = {}  # will use the defaults from DEFAULT_DUMMY_SHAPES

        # Check that inputs match, and order them properly
        dummy_inputs = config.generate_dummy_inputs(framework="pt", **input_shapes)
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            model.to(device)
            dummy_inputs = tree_map(
                lambda value: value.to(device) if isinstance(value, torch.Tensor) else value, dummy_inputs
            )

        dummy_inputs = config.rename_ambiguous_inputs(dummy_inputs)
        dummy_inputs, dict_inputs = remove_none_from_dummy_inputs(dummy_inputs)
        # TorchScript used behind OpenVINO conversion. Optimum supports only return_dict=True models for patching,
        # while TorchScript do not support dictionary with values of mixed types (e.g. Tensor and None) in model input/output
        # To handle it, additional wrapper on patcher forward applied.
        # model.config.torchscript = True can not be used for patching, because it overrides return_dict to False
        patcher = config.patch_model_for_export(model, model_kwargs=model_kwargs)
        patched_forward = patcher.patched_forward
        dummy_input_keys = list(dummy_inputs.keys())

        @functools.wraps(patched_forward)
        def ts_patched_forward(*args, **kwargs):
            ordered_example_inputs = [
                param
                for param in inspect.signature(
                    patcher.orig_forward if library_name != "sentence_transformers" else patcher.patched_forward
                ).parameters
                if param in dummy_input_keys
            ]
            kwargs.update(zip(ordered_example_inputs, args))
            for i in range(len(dict_inputs)):
                input_name, keys = dict_inputs[i]
                tuple_input = kwargs[input_name]
                input_dict = dict(zip(keys, tuple_input))
                kwargs[input_name] = input_dict
            outputs = patched_forward(**kwargs)
            return tuple([value if not isinstance(value, list) else tuple(value) for value in outputs.values()])

        patcher.patched_forward = ts_patched_forward

        ts_decoder_kwargs = {}
        model_config = getattr(model, "config", {})
        model_type = getattr(model_config, "model_type", "").replace("_", "-")
        if allow_skip_tracing_check(library_name, model_type):
            ts_decoder_kwargs["trace_kwargs"] = {"check_trace": False}

        with patcher:
            if patch_16bit_model:
                from openvino.frontend.pytorch.patch_model import __make_16bit_traceable

                __make_16bit_traceable(model)
            check_dummy_inputs_are_allowed(model, dummy_inputs)
            input_info = _get_input_info(model, config, dummy_inputs)
            ts_decoder = TorchScriptPythonDecoder(model, example_input=dummy_inputs, **ts_decoder_kwargs)
            ov_model = convert_model(
                ts_decoder,
                example_input=dummy_inputs,
                input=[(item.shape, item.type) for item in input_info],
            )

        ov_model.validate_nodes_and_infer_types()  # TODO: remove as unnecessary validation?

        output_names = list(config.outputs.keys())
        for idx, out_tensor in enumerate(ov_model.outputs):
            if idx < len(output_names):
                out_tensor.get_tensor().set_names({output_names[idx]})

        input_names = [item.name for item in input_info]
        for idx, inp_tensor in enumerate(ov_model.inputs):
            input_name = input_names[idx]
            inp_tensor.get_tensor().set_names({input_name})

        if stateful:
            patch_stateful(model.config, ov_model)

        library_name = _infer_library_from_model_or_model_class(model=model, library_name=library_name)

        _save_model(
            ov_model,
            output,
            ov_config=ov_config,
            library_name=library_name,
            config=config,
        )
        clear_class_registry()
        del ov_model
        del model
        gc.collect()
    return input_names, output_names, False


def export_models(
    models_and_export_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin", "DiffusionPipeline"], "OnnxConfig"]
    ],
    output_dir: Path,
    opset: Optional[int] = None,
    output_names: Optional[List[str]] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
    stateful: bool = True,
    patch_16bit_model: bool = False,
    library_name: Optional[str] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Export the models to OpenVINO IR format

    Args:
        models_and_export_configs (Dict[ str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]):
        output_dir (Path): output directory for saving models
        opset (Optional[int], optional, Default to None): ONNX export opset
        output_names (Optional[List[str]], optional, Defaults to None): model output names
        device (str, optional, Defaults to "cpu"):
            The device on which the model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (Optional[Dict], optional, Defaults to None):
            If specified, allows to use specific shapes for the example input provided to the exporter.
        ov_config (`OVConfig`, *optional*):
            The configuration containing the parameters related to quantization.
        model_kwargs (Optional[Dict[str, Any]], optional):
            Additional kwargs for model export.
        stateful (`bool`, defaults to `True`)
            Produce stateful model where all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. Applicable only for decoder models.

    Raises:
        ValueError: if custom names set not equal of number of models

    Returns:
        list of input_names and output_names from ONNX configuration
    """

    outputs = []

    if output_names is not None and len(output_names) != len(models_and_export_configs):
        raise ValueError(
            f"Provided custom names {output_names} for the export of {len(models_and_export_configs)} models. Please provide the same number of names as models to export."
        )

    for i, model_name in enumerate(models_and_export_configs.keys()):
        submodel, sub_export_config = models_and_export_configs[model_name]
        output_name = output_names[i] if output_names is not None else Path(model_name + ".xml")
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(
            export(
                model=submodel,
                config=sub_export_config,
                output=output_path,
                opset=opset,
                device=device,
                input_shapes=input_shapes,
                model_kwargs=model_kwargs,
                ov_config=ov_config,
                stateful=stateful[i] if isinstance(stateful, (list, tuple)) else stateful,
                patch_16bit_model=patch_16bit_model,
                library_name=library_name,
            )
        )

    outputs = list(map(list, zip(*outputs)))
    return outputs


def export_from_model(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin", "DiffusionPipeline"],
    output: Union[str, Path],
    task: Optional[str] = None,
    ov_config: Optional["OVConfig"] = None,
    stateful: bool = True,
    opset: Optional[int] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_export_configs: Optional[Dict[str, "OnnxConfig"]] = None,
    fn_get_submodels: Optional[Callable] = None,
    preprocessors: List = None,
    device: str = "cpu",
    trust_remote_code: bool = False,
    patch_16bit_model: bool = False,
    **kwargs_shapes,
):
    model_kwargs = model_kwargs or {}

    if ov_config is not None and ov_config.quantization_config and not is_nncf_available():
        raise ImportError(
            f"Compression of the weights to {ov_config.quantization_config} requires nncf, please install it with `pip install nncf`"
        )

    library_name = _infer_library_from_model_or_model_class(model)
    if library_name != "open_clip":
        TasksManager.standardize_model_attributes(model)

    if hasattr(model.config, "export_model_type") and model.config.export_model_type is not None:
        model_type = model.config.export_model_type.replace("_", "-")
    else:
        model_type = (getattr(model.config, "model_type", None) or "").replace("_", "-")

    custom_architecture = library_name == "transformers" and model_type not in TasksManager._SUPPORTED_MODEL_TYPE

    if task is not None and task != "auto":
        task = TasksManager.map_from_synonym(task)
    else:
        try:
            task = TasksManager._infer_task_from_model_or_model_class(model=model)
        except (ValueError, KeyError) as e:
            raise RuntimeError(
                f"The model task could not be automatically inferred in `export_from_model`. Please provide the argument `task` with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

        if (
            not custom_architecture
            and library_name != "diffusers"
            and task + "-with-past"
            in TasksManager.get_supported_tasks_for_model_type(model_type, "openvino", library_name=library_name)
        ):
            # -with-past is the default.
            task = task + "-with-past"

        logger.info(f"Automatic task detection to: {task}.")

    is_encoder_decoder = getattr(getattr(model, "config", {}), "is_encoder_decoder", False)
    stateful = stateful and (
        ensure_export_task_support_stateful(task) or ensure_model_type_support_stateful(model_type)
    )

    if stateful and is_encoder_decoder and not getattr(model, "_supports_cache_class", False):
        stateful = False
    # TODO: support onnx_config.py in the model repo
    if custom_architecture and custom_export_configs is None:
        raise ValueError(
            f"Trying to export a {model_type} model, that is a custom or unsupported architecture, but no custom export configuration was passed as `custom_export_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models. Please open an issue at https://github.com/huggingface/optimum-intel/issues if you would like the model type {model_type} to be supported natively in the OpenVINO export."
        )

    if task.startswith("text-generation") and model.config.is_encoder_decoder:
        raise ValueError(
            f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
            f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
            f" referring to `optimum.exporters.tasks.TaskManager`'s `_TRANSFORMERS_TASKS_TO_MODEL_LOADERS`."
        )
    if library_name != "diffusers" and model_type in TasksManager._UNSUPPORTED_CLI_MODEL_TYPE:
        raise ValueError(
            f"{model_type} is not supported yet. Only {list(TasksManager._SUPPORTED_CLI_MODEL_TYPE.keys())} are supported. "
            f"If you want to support {model_type} please propose a PR or open up an issue."
        )

    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    # Get the shapes to be used to generate dummy inputs
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        if input_name in ["height", "width"]:
            # use H and W from generator defaults
            continue
        input_shapes[input_name] = (
            kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
        )

    if library_name == "open_clip":
        custom_architecture = True
        custom_export_configs, fn_get_submodels = _get_open_clip_submodels_fn_and_export_configs(
            model, library_name, task, preprocessors, custom_export_configs, fn_get_submodels
        )

    if library_name == "diffusers":
        export_config, models_and_export_configs = get_diffusion_models_for_export_ext(model, exporter="openvino")
        stateful_submodels = False
    elif stateful and is_encoder_decoder and not custom_architecture:
        export_config, models_and_export_configs = _get_encoder_decoder_stateful_models_for_export(
            model=model, task=task, preprocessors=preprocessors, library_name=library_name, _variant="default"
        )
        stateful_submodels = [False, True]
    else:
        logging.disable(logging.INFO)
        export_config, models_and_export_configs, stateful_submodels = _get_submodels_and_export_configs(
            model=model,
            task=task,
            monolith=False,
            custom_export_configs=custom_export_configs if custom_export_configs is not None else {},
            custom_architecture=custom_architecture,
            fn_get_submodels=fn_get_submodels,
            preprocessors=preprocessors,
            library_name=library_name,
            model_kwargs=model_kwargs,
            _variant="default",
            legacy=False,
            exporter="openvino",
            stateful=stateful,
        )
        logging.disable(logging.NOTSET)

    if library_name == "open_clip":
        if hasattr(model.config, "save_pretrained"):
            model.config.save_pretrained(output)

        for preprocess in preprocessors:
            if hasattr(preprocess, "save_pretrained"):
                preprocess.save_pretrained(output)

        files_subpaths = ["openvino_" + model_name + ".xml" for model_name in models_and_export_configs.keys()]
    elif library_name != "diffusers":
        if is_transformers_version(">=", "4.44.99"):
            # some model configs may have issues with loading without parameters initialization
            try:
                misplaced_generation_parameters = model.config._get_non_default_generation_parameters()
            except (KeyError, TypeError):
                misplaced_generation_parameters = {}
            if isinstance(model, GenerationMixin) and len(misplaced_generation_parameters) > 0:
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(model.generation_config, param_name, param_value)
                    setattr(model.config, param_name, None)

        # workaround for https://github.com/huggingface/transformers/issues/37172
        if is_transformers_version(">=", "4.50.0") and model_type == "whisper":
            if hasattr(model.config, "forced_decoder_ids"):
                model.config.forced_decoder_ids = None
            if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
                model.generation_config.forced_decoder_ids = None
        # Saving the model config and preprocessor as this is needed sometimes.
        save_config(model.config, output)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            try:
                generation_config.save_pretrained(output)
            except Exception as exception:
                logger.warning(
                    f"The generation config will not be saved, saving failed with following error:\n{exception}"
                )

        save_preprocessors(preprocessors, model.config, output, trust_remote_code)

        files_subpaths = ["openvino_" + model_name + ".xml" for model_name in models_and_export_configs.keys()]

    else:
        # save the subcomponent configuration
        for model_name in models_and_export_configs:
            subcomponent = models_and_export_configs[model_name][0]
            if hasattr(subcomponent, "save_config"):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
                subcomponent.config.save_pretrained(output / model_name)

        files_subpaths = [os.path.join(name_dir, OV_XML_FILE_NAME) for name_dir in models_and_export_configs]

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

        tokenizer_3 = getattr(model, "tokenizer_3", None)
        if tokenizer_3 is not None:
            tokenizer_3.save_pretrained(output.joinpath("tokenizer_3"))
        safety_checker = getattr(model, "safety_checker", None)
        if safety_checker is not None:
            safety_checker.save_pretrained(output.joinpath("safety_checker"))

        model.save_config(output)

    _set_runtime_options(
        models_and_export_configs,
        task,
        library_name,
        hasattr(ov_config, "quantization_config") and ov_config.quantization_config,
    )

    export_models(
        models_and_export_configs=models_and_export_configs,
        output_dir=output,
        output_names=files_subpaths,
        input_shapes=input_shapes,
        device=device,
        ov_config=ov_config,
        stateful=stateful_submodels,
        opset=opset,
        model_kwargs=model_kwargs,
        patch_16bit_model=patch_16bit_model,
        library_name=library_name,
    )

    return files_subpaths


def export_tokenizer(
    tokenizer,
    output: Union[str, Path],
    suffix: Optional[str] = "",
    task: Optional[str] = None,
    processor_chat_template: Optional[str] = None,
):
    # avoid circular imports
    from optimum.intel.openvino import OV_DETOKENIZER_NAME, OV_TOKENIZER_NAME
    from optimum.intel.openvino.utils import maybe_convert_tokenizer_to_fast

    try:
        from openvino_tokenizers import convert_tokenizer
    except ModuleNotFoundError:
        return

    if is_tokenizers_version(">", "0.19") and is_openvino_tokenizers_version("<", "2024.5.0.0"):
        logger.warning(
            "Exporting tokenizers to OpenVINO is not supported for tokenizers version > 0.19 and openvino version <= 2024.4. "
            "Please downgrade to tokenizers version <= 0.19 to export tokenizers to OpenVINO."
        )

    if not isinstance(output, Path):
        output = Path(output)

    if output.exists():
        tokenizer = maybe_convert_tokenizer_to_fast(tokenizer, output)

    if (
        task is not None
        and (task.startswith("text-generation") or task == "image-text-to-text")
        and compare_versions("openvino-tokenizers", ">=", "2024.3.0.0")
    ):
        logger.info(f"Set tokenizer padding side to left for `{task}` task.")
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

    try:
        converted = convert_tokenizer(tokenizer, with_detokenizer=True)
        set_simplified_chat_template(converted[0], processor_chat_template)

    except NotImplementedError:
        logger.info("Detokenizer is not supported, convert tokenizer only.")
        converted = convert_tokenizer(tokenizer, with_detokenizer=False)
    except OVTypeError:
        logger.debug(f"OpenVINO Tokenizer export for {type(tokenizer).__name__} is not supported.")
        return
    except Exception as exception:
        logger.debug(
            f"OpenVINO Tokenizer export for {type(tokenizer).__name__} is not supported. Exception: {exception}"
        )
        return

    if not isinstance(converted, tuple):
        converted = (converted,)

    for model, file_name in zip(converted, (OV_TOKENIZER_NAME, OV_DETOKENIZER_NAME)):
        save_model(model, output / file_name.format(suffix))


def _add_runtime_options_to_rt_info(model: Model, options: Dict):
    """
    Add runtime optinos
    """
    try:
        for name, value in options.items():
            model.set_rt_info(value, ["runtime_options", name])
    except Exception:
        pass

    return model


def _add_version_info_to_model(model: Model, library_name: Optional[str] = None):
    """
    Add dependency versions to OpenVINO model
    """
    try:
        model.set_rt_info(_transformers_version, ["optimum", "transformers_version"])
        model.set_rt_info(_torch_version, ["optimum", "pytorch_version"])
        model.set_rt_info(_optimum_intel_version, ["optimum", "optimum_intel_version"])
        model.set_rt_info(_optimum_version, ["optimum", "optimum_version"])

        if any("token_embeddings" in output.get_names() for output in model.outputs):
            import sentence_transformers

            model.set_rt_info(sentence_transformers.__version__, ["optimum", "sentence_transformers_version"])
        if library_name == "diffusers":
            model.set_rt_info(_diffusers_version, ["optimum", "diffusers_version"])
        elif library_name == "timm":
            model.set_rt_info(_timm_version, ["optimum", "timm_version"])
        elif library_name == "open_clip":
            model.set_rt_info(_open_clip_version, ["optimum", "open_clip_version"])
        rt_info = model.get_rt_info()
        if "nncf" in rt_info:
            model.set_rt_info(_nncf_version, ["optimum", "nncf_version"])
        input_model = rt_info["conversion_parameters"].get("input_model", None)
        if input_model is not None and "onnx" in input_model.value:
            import onnx

            model.set_rt_info(onnx.__version__, ["optimum", "onnx_version"])

    except Exception:
        pass

    return model


def _get_multi_modal_submodels_and_export_configs(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    task: str,
    library_name: str,
    int_dtype: str,
    float_dtype: str,
    preprocessors: Optional[List[Any]] = None,
    model_kwargs: Optional[Dict] = None,
    stateful: bool = True,
):
    models_for_export = {}
    stateful_parts = []

    model_type = model.config.model_type.replace("_", "-")

    if model_type == "internvl-chat" and preprocessors is not None:
        model.config.img_context_token_id = preprocessors[0].convert_tokens_to_ids("<IMG_CONTEXT>")

    if model_type == "phi3-v":
        model.config.glb_GN = model.model.vision_embed_tokens.glb_GN.tolist()
        model.config.sub_GN = model.model.vision_embed_tokens.sub_GN.tolist()

    if model_type == "phi4mm":
        model.config.glb_GN = model.model.embed_tokens_extend.image_embed.glb_GN.tolist()
        model.config.sub_GN = model.model.embed_tokens_extend.image_embed.sub_GN.tolist()
        model.config.num_img_tokens = model.model.embed_tokens_extend.image_embed.num_img_tokens
        model.config.hd_transform_order = model.model.embed_tokens_extend.image_embed.hd_transform_order
        if model.config.img_processor is None:
            model.config.img_processor = model.model.embed_tokens_extend.image_embed.img_processor.config.to_dict()
    if model_type == "phi4-multimodal":
        model.config.glb_GN = model.model.embed_tokens_extend.image_embed.global_img_feature_extensor.tolist()
        model.config.sub_GN = model.model.embed_tokens_extend.image_embed.sub_img_feature_extensor.tolist()
        model.config.num_img_tokens = model.model.embed_tokens_extend.image_embed.num_img_tokens

    if hasattr(model, "image_newline"):
        model.config.image_newline = model.image_newline.tolist()
    main_config_cls = TasksManager.get_exporter_config_constructor(
        model=model, task=task, exporter="openvino", library_name=library_name
    )
    main_config = main_config_cls(
        model.config, int_dtype=int_dtype, float_dtype=float_dtype, preprocessors=preprocessors
    )
    for behavior in main_config.SUPPORTED_BEHAVIORS:
        model_id = f"{behavior}_model"
        model_part_config = main_config.with_behavior(behavior)
        model_part = main_config.get_model_for_behavior(model, behavior)
        models_for_export[model_id] = (model_part, model_part_config)
        stateful_parts.append(stateful if getattr(model_part_config, "use_past", False) else False)
    return main_config, models_for_export, stateful_parts


def _get_submodels_and_export_configs(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline"],
    task: str,
    monolith: bool,
    custom_export_configs: Dict,
    custom_architecture: bool,
    _variant: str,
    library_name: str,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    fn_get_submodels: Optional[Callable] = None,
    preprocessors: Optional[List[Any]] = None,
    legacy: bool = False,
    model_kwargs: Optional[Dict] = None,
    exporter: str = "openvino",
    stateful: bool = False,
):
    if (
        not custom_architecture
        and library_name == "transformers"
        and model.config.model_type.replace("_", "-") in MULTI_MODAL_TEXT_GENERATION_MODELS
    ):
        return _get_multi_modal_submodels_and_export_configs(
            model, task, library_name, int_dtype, float_dtype, preprocessors, model_kwargs, stateful
        )
    elif not custom_architecture and library_name == "transformers" and model.config.model_type == "speecht5":
        return _get_speecht5_tss_model_for_export(
            model, task, library_name, int_dtype, float_dtype, preprocessors, model_kwargs
        )

    export_config, models_for_export = _default_get_submodels_and_export_configs(
        model,
        task,
        monolith,
        custom_export_configs,
        custom_architecture,
        _variant,
        library_name,
        int_dtype,
        float_dtype,
        fn_get_submodels,
        preprocessors,
        legacy,
        model_kwargs,
        exporter,
    )
    stateful_per_model = [stateful] * len(models_for_export)
    return export_config, models_for_export, stateful_per_model


def get_diffusion_models_for_export_ext(
    pipeline: "DiffusionPipeline", int_dtype: str = "int64", float_dtype: str = "fp32", exporter: str = "openvino"
):
    is_sdxl = pipeline.__class__.__name__.startswith("StableDiffusionXL")
    is_sd3 = pipeline.__class__.__name__.startswith("StableDiffusion3")
    is_flux = pipeline.__class__.__name__.startswith("Flux")
    is_sana = pipeline.__class__.__name__.startswith("Sana")
    is_ltx_video = pipeline.__class__.__name__.startswith("LTX")
    is_sd = pipeline.__class__.__name__.startswith("StableDiffusion") and not is_sd3
    is_lcm = pipeline.__class__.__name__.startswith("LatentConsistencyModel")

    if is_sd or is_sdxl or is_lcm:
        tokenizer = pipeline.tokenizer_2 if is_sdxl else pipeline.tokenizer
        model_max_length = getattr(tokenizer, "model_max_length", None)
        pipeline.unet.config.model_max_length = model_max_length
        models_for_export = get_diffusion_models_for_export(pipeline, int_dtype, float_dtype, exporter)
        if is_sdxl and pipeline.vae.config.force_upcast:
            models_for_export["vae_encoder"][1].runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "128.0"}
            models_for_export["vae_decoder"][1].runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "128.0"}

        # only SD 2.1 has overflow issue, it uses different prediction_type than other models
        if is_sd and pipeline.scheduler.config.prediction_type == "v_prediction":
            models_for_export["vae_encoder"][1].runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
            models_for_export["vae_decoder"][1].runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}

    elif is_sd3:
        models_for_export = get_sd3_models_for_export(pipeline, exporter, int_dtype, float_dtype)
    elif is_flux:
        models_for_export = get_flux_models_for_export(pipeline, exporter, int_dtype, float_dtype)
    elif is_sana:
        models_for_export = get_sana_models_for_export(pipeline, exporter, int_dtype, float_dtype)
    elif is_ltx_video:
        models_for_export = get_ltx_video_models_for_export(pipeline, exporter, int_dtype, float_dtype)
    else:
        raise ValueError(f"Unsupported pipeline type `{pipeline.__class__.__name__}` provided")
    return None, models_for_export


def get_ltx_video_models_for_export(pipeline, exporter, int_dtype, float_dtype):
    models_for_export = {}
    text_encoder = pipeline.text_encoder
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=text_encoder,
        exporter=exporter,
        library_name="diffusers",
        task="feature-extraction",
        model_type="t5-encoder-model",
    )
    export_config = export_config_constructor(
        text_encoder.config,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
    )
    export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["text_encoder"] = (text_encoder, export_config)
    transformer = pipeline.transformer
    transformer.config.vae_temporal_compression_ratio = pipeline.vae_temporal_compression_ratio
    transformer.config.vae_spatial_compression_ratio = pipeline.vae_spatial_compression_ratio
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=transformer,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="ltx-video-transformer",
    )
    transformer_export_config = export_config_constructor(
        transformer.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["transformer"] = (transformer, transformer_export_config)
    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_parameters": vae_encoder.encode(x=sample)["latent_dist"].parameters}
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="ltx-vae-encoder",
    )
    vae_encoder_export_config = vae_config_constructor(
        vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    vae_encoder_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["vae_encoder"] = (vae_encoder, vae_encoder_export_config)

    # VAE Decoder
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.register_to_config(
        **{
            "latents_mean_data": vae_decoder.latents_mean.tolist(),
            "latents_std_data": vae_decoder.latents_std.tolist(),
        }
    )

    vae_decoder.forward = lambda latent_sample, timestep=None: vae_decoder.decode(z=latent_sample, temb=timestep)

    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="ltx-vae-decoder",
    )
    vae_decoder_export_config = vae_config_constructor(
        vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    vae_decoder_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_export_config)

    return models_for_export


def get_sana_models_for_export(pipeline, exporter, int_dtype, float_dtype):
    models_for_export = {}
    text_encoder = pipeline.text_encoder
    text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=text_encoder,
        exporter=exporter,
        library_name="diffusers",
        task="feature-extraction",
        model_type="gemma2-text-encoder",
    )
    text_encoder_export_config = text_encoder_config_constructor(
        pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    text_encoder_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["text_encoder"] = (text_encoder, text_encoder_export_config)
    transformer = pipeline.transformer
    transformer.config.text_encoder_projection_dim = transformer.config.caption_channels
    transformer.config.requires_aesthetics_score = False
    transformer.config.time_cond_proj_dim = None
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=transformer,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="sana-transformer",
    )
    transformer_export_config = export_config_constructor(
        pipeline.transformer.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["transformer"] = (transformer, transformer_export_config)
    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent": vae_encoder.encode(x=sample)["latent"]}
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="dcae-encoder",
    )
    vae_encoder_export_config = vae_config_constructor(
        vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    vae_encoder_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["vae_encoder"] = (vae_encoder, vae_encoder_export_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-decoder",
    )
    vae_decoder_export_config = vae_config_constructor(
        vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    vae_decoder_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_export_config)

    return models_for_export


def get_sd3_models_for_export(pipeline, exporter, int_dtype, float_dtype):
    models_for_export = {}

    # Text encoder
    text_encoder = getattr(pipeline, "text_encoder", None)
    if text_encoder is not None:
        text_encoder.config.output_hidden_states = True
        text_encoder.text_model.config.output_hidden_states = True
        text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
            model=text_encoder,
            exporter=exporter,
            library_name="diffusers",
            task="feature-extraction",
            model_type="clip-text-with-projection",
        )
        text_encoder_export_config = text_encoder_config_constructor(
            pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder"] = (text_encoder, text_encoder_export_config)

    transformer = pipeline.transformer
    transformer.config.text_encoder_projection_dim = transformer.config.joint_attention_dim
    transformer.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
    transformer.config.time_cond_proj_dim = None
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=transformer,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="sd3-transformer",
    )
    transformer_export_config = export_config_constructor(
        pipeline.transformer.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["transformer"] = (transformer, transformer_export_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_parameters": vae_encoder.encode(x=sample)["latent_dist"].parameters}
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-encoder",
    )
    vae_encoder_export_config = vae_config_constructor(
        vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["vae_encoder"] = (vae_encoder, vae_encoder_export_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-decoder",
    )
    vae_decoder_export_config = vae_config_constructor(
        vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_export_config)

    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        text_encoder_2.config.output_hidden_states = True
        text_encoder_2.text_model.config.output_hidden_states = True
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=text_encoder_2,
            exporter=exporter,
            library_name="diffusers",
            task="feature-extraction",
            model_type="clip-text-with-projection",
        )
        export_config = export_config_constructor(text_encoder_2.config, int_dtype=int_dtype, float_dtype=float_dtype)
        models_for_export["text_encoder_2"] = (text_encoder_2, export_config)

    text_encoder_3 = getattr(pipeline, "text_encoder_3", None)
    if text_encoder_3 is not None:
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=text_encoder_3,
            exporter=exporter,
            library_name="diffusers",
            task="feature-extraction",
            model_type="t5-encoder-model",
        )
        export_config = export_config_constructor(
            text_encoder_3.config,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
        )
        export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
        models_for_export["text_encoder_3"] = (text_encoder_3, export_config)

    return models_for_export


def get_flux_models_for_export(pipeline, exporter, int_dtype, float_dtype):
    models_for_export = {}

    # Text encoder
    text_encoder = getattr(pipeline, "text_encoder", None)
    if text_encoder is not None:
        text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
            model=text_encoder,
            exporter=exporter,
            library_name="diffusers",
            task="feature-extraction",
            model_type="clip-text-model",
        )
        text_encoder_export_config = text_encoder_config_constructor(
            pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder"] = (text_encoder, text_encoder_export_config)

    transformer = pipeline.transformer
    transformer.config.text_encoder_projection_dim = transformer.config.joint_attention_dim
    transformer.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
    transformer.config.time_cond_proj_dim = None
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=transformer,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="flux-transformer",
    )
    transformer_export_config = export_config_constructor(
        pipeline.transformer.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    transformer_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["transformer"] = (transformer, transformer_export_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_parameters": vae_encoder.encode(x=sample)["latent_dist"].parameters}
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-encoder",
    )
    vae_encoder_export_config = vae_config_constructor(
        vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    vae_encoder_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["vae_encoder"] = (vae_encoder, vae_encoder_export_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter=exporter,
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="vae-decoder",
    )
    vae_decoder_export_config = vae_config_constructor(
        vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    vae_decoder_export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
    models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_export_config)

    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=text_encoder_2,
            exporter=exporter,
            library_name="diffusers",
            task="feature-extraction",
            model_type="t5-encoder-model",
        )
        export_config = export_config_constructor(
            text_encoder_2.config,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
        )
        export_config.runtime_options = {"ACTIVATIONS_SCALE_FACTOR": "8.0"}
        models_for_export["text_encoder_2"] = (text_encoder_2, export_config)

    return models_for_export


def _get_encoder_decoder_stateful_models_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    task: str,
    _variant: str,
    library_name: str,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    preprocessors: Optional[List[Any]] = None,
):
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="openvino", task=task, library_name=library_name
    )
    export_config = export_config_constructor(
        model.config,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        preprocessors=preprocessors,
        legacy=False,
    )

    export_config.variant = _variant
    all_variants = "\n".join([f"    - {name}: {description}" for name, description in export_config.VARIANTS.items()])
    logger.info(f"Using the export variant {export_config.variant}. Available variants are:\n{all_variants}")

    models_for_export = _get_submodels_for_export_encoder_decoder(model, use_past=False)

    encoder_export_config = export_config.with_behavior("encoder")
    models_for_export[ENCODER_NAME] = (models_for_export[ENCODER_NAME], encoder_export_config)

    decoder_export_config_with_past = export_config.with_behavior("decoder", use_past=True, use_past_in_inputs=True)

    decoder_export_config_with_past.stateful = True
    models_for_export[DECODER_NAME] = (
        models_for_export[DECODER_NAME],
        decoder_export_config_with_past,
    )
    return None, models_for_export


def _get_speecht5_tss_model_for_export(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    task: str,
    library_name: str,
    int_dtype: str,
    float_dtype: str,
    preprocessors: Optional[List[Any]] = None,
    model_kwargs: Optional[Dict] = None,
):
    if model_kwargs is None or "vocoder" not in model_kwargs:
        raise ValueError(
            'The export of SpeechT5 requires a vocoder. Please pass `--model-kwargs \'{"vocoder": "vocoder_model_name_or_path"}\'` from the command line, or `model_kwargs={"vocoder": "vocoder_model_name_or_path"}` if calling main_export.'
        )
    vocoder_id = model_kwargs["vocoder"]

    # prepare export config
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="openvino", task=task, library_name=library_name
    )
    export_config = export_config_constructor(
        model.config,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        preprocessors=preprocessors,
        legacy=False,
    )
    export_config.variant = "default"

    models_for_export = {}
    encoder_export_config = export_config.with_behavior("encoder")
    decoder_export_config = export_config.with_behavior("decoder")
    postnet_export_config = export_config.with_behavior("postnet")
    vocoder_export_config = export_config.with_behavior("vocoder")

    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id).eval()

    models_for_export[ENCODER_NAME] = (model.speecht5.encoder, encoder_export_config)
    models_for_export[DECODER_NAME] = (model, decoder_export_config)
    models_for_export["postnet"] = (model, postnet_export_config)
    models_for_export["vocoder"] = (vocoder, vocoder_export_config)

    stateful_per_model = [False, True, False, False]

    return export_config, models_for_export, stateful_per_model
