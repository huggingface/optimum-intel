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

import functools
import gc
import inspect
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import onnx
from transformers.utils import is_tf_available, is_torch_available

from openvino.runtime import Model, PartialShape, save_model
from openvino.runtime.exceptions import OVTypeError
from openvino.runtime.utils.types import get_element_type
from openvino.tools.ovc import convert_model
from optimum.exporters import TasksManager
from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.convert import check_dummy_inputs_are_allowed
from optimum.exporters.onnx.convert import export_pytorch as export_pytorch_to_onnx
from optimum.exporters.onnx.convert import export_tensorflow as export_tensorflow_onnx
from optimum.exporters.utils import _get_submodels_and_export_configs
from optimum.intel.utils.import_utils import (
    _nncf_version,
    _optimum_intel_version,
    _optimum_version,
    _timm_version,
    _torch_version,
    _transformers_version,
)
from optimum.utils import DEFAULT_DUMMY_SHAPES, is_diffusers_available
from optimum.utils.save_utils import maybe_save_preprocessors

from ...intel.utils.import_utils import is_nncf_available
from .model_patcher import patch_model_with_bettertransformer
from .stateful import ensure_export_task_support_stateful, ensure_stateful_is_available, patch_stateful
from .utils import (
    _MAX_UNCOMPRESSED_SIZE,
    OV_XML_FILE_NAME,
    clear_class_registry,
    flattenize_inputs,
    get_input_shapes,
    remove_none_from_dummy_inputs,
)


logger = logging.getLogger(__name__)

if is_torch_available():
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel

if is_diffusers_available():
    from diffusers import ModelMixin

if is_tf_available():
    from transformers.modeling_tf_utils import TFPreTrainedModel


if TYPE_CHECKING:
    from optimum.intel.openvino.configuration import OVConfig


def _save_model(model, path: str, ov_config: Optional["OVConfig"] = None):
    compress_to_fp16 = False

    if ov_config is not None:
        if ov_config.quantization_config:
            if not is_nncf_available():
                raise ImportError(
                    "Quantization of the weights to int8 requires nncf, please install it with `pip install nncf`"
                )

            from optimum.intel.openvino.quantization import _weight_only_quantization

            _weight_only_quantization(model, ov_config.quantization_config)

        compress_to_fp16 = ov_config.dtype == "fp16"

    library_name = TasksManager.infer_library_from_model(Path(path).parent)
    model = _add_version_info_to_model(model, library_name)
    save_model(model, path, compress_to_fp16)


def export(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    output: Path,
    opset: Optional[int] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
    stateful: bool = True,
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
        raise ImportError("The pip package `diffusers` is required to export stable diffusion models to ONNX.")

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
        )

    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        output.parent.mkdir(parents=True, exist_ok=True)
        if opset is None:
            opset = config.DEFAULT_ONNX_OPSET
        if device == "cuda":
            raise RuntimeError("`tf2onnx` does not support export on CUDA device.")
        if input_shapes is not None:
            logger.info("`input_shapes` argument is not supported by the Tensorflow ONNX export and will be ignored.")
        return export_tensorflow(model, config, opset, output, ov_config=ov_config)

    else:
        raise RuntimeError(
            "You either provided a PyTorch model with only TensorFlow installed, or a TensorFlow model with only PyTorch installed."
        )


def export_tensorflow(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    ov_config: Optional["OVConfig"] = None,
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
    onnx_path = Path(output).with_suffix(".onnx")
    input_names, output_names = export_tensorflow_onnx(model, config, opset, onnx_path)
    ov_model = convert_model(str(onnx_path))
    _save_model(ov_model, output.parent / output, ov_config=ov_config)
    return input_names, output_names, True


def export_pytorch_via_onnx(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
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
    _save_model(ov_model, output.parent / OV_XML_FILE_NAME if output.suffix != ".xml" else output, ov_config=ov_config)
    return input_names, output_names, True


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
    stateful: bool = False,
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

        try:
            # TorchScript used behind OpenVINO conversion. Optimum supports only return_dict=True models for patching,
            # while TorchScript do not support dictionary with values of mixed types (e.g. Tensor and None) in model input/output
            # To handle it, additional wrapper on patcher forward applied.
            # model.config.torchscript = True can not be used for patching, because it overrides return_dict to False
            patcher = config.patch_model_for_export(model, model_kwargs=model_kwargs)
            patched_forward = patcher.patched_forward

            @functools.wraps(patched_forward)
            def ts_patched_forward(*args, **kwargs):
                for i in range(len(dict_inputs)):
                    input_name, keys = dict_inputs[i]
                    tuple_input = kwargs[input_name]
                    input_dict = dict(zip(keys, tuple_input))
                    kwargs[input_name] = input_dict
                outputs = patched_forward(*args, **kwargs)
                return tuple([value if not isinstance(value, list) else tuple(value) for value in outputs.values()])

            patcher.patched_forward = ts_patched_forward

            with patcher:
                check_dummy_inputs_are_allowed(model, dummy_inputs)
                sig = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.call)
                inputs = config.ordered_inputs(model)
                input_names = list(inputs.keys())
                output_names = list(config.outputs.keys())
                input_info = get_input_shapes(dummy_inputs, inputs)

                ov_model = convert_model(model, example_input=dummy_inputs, input=input_info)

        except Exception as ex:
            logger.warning(f"Export model to OpenVINO directly failed with: \n{ex}.\nModel will be exported to ONNX")

            if stateful:
                # cannot raise because stateful is enabled by default and it would break backward compatibility for models that couldn't convert to OV directly
                # TODO: Implement stateful for ONNX path as well, not doing it right now because of lack of validation
                logger.warn(
                    "[ WARNING ] Making stateful models is not supported when exporting to ONNX as an intermediate step. "
                    "A stateless model will be exported instead. It may result in sub-optimal inference performance."
                    "Provide a model that can be converted to OpenVINO without fallback to ONNX conversion path."
                )
            return export_pytorch_via_onnx(
                model,
                config,
                opset,
                output,
                device,
                input_shapes,
                model_kwargs,
                ov_config=ov_config,
            )

        ordered_dummy_inputs = {param: dummy_inputs[param] for param in sig.parameters if param in dummy_inputs}
        if not ordered_dummy_inputs:
            ordered_dummy_inputs = dummy_inputs
        ordered_input_names = list(inputs)
        flatten_inputs = flattenize_inputs(ordered_dummy_inputs.values())
        ov_model.validate_nodes_and_infer_types()
        for idx, out_tensor in enumerate(ov_model.outputs):
            if idx < len(output_names):
                out_tensor.get_tensor().set_names({output_names[idx]})

        for idx, inp_tensor in enumerate(ov_model.inputs):
            input_name = ordered_input_names[idx]
            inp_tensor.get_tensor().set_names({input_name})
            inp_data = flatten_inputs[idx]
            static_shape = PartialShape(inp_data.shape)
            dims = inputs.get(input_name, [])
            for dim in dims:
                static_shape[dim] = -1
            inp_tensor.get_node().set_partial_shape(static_shape)
            inp_tensor.get_node().set_element_type(get_element_type(inp_data.cpu().numpy().dtype))
        ov_model.validate_nodes_and_infer_types()

        if stateful:
            patch_stateful(model.config, ov_model)

        _save_model(ov_model, output, ov_config=ov_config)
        clear_class_registry()
        del model
        gc.collect()
    return input_names, output_names, False


def export_models(
    models_and_export_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]
    ],
    output_dir: Path,
    opset: Optional[int] = None,
    output_names: Optional[List[str]] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ov_config: Optional["OVConfig"] = None,
    stateful: bool = True,
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
                stateful=stateful,
            )
        )

    outputs = list(map(list, zip(*outputs)))
    return outputs


def export_from_model(
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
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
    **kwargs_shapes,
):
    if ov_config is not None and ov_config.quantization_config and not is_nncf_available():
        raise ImportError(
            f"Compression of the weights to {ov_config.quantization_config} requires nncf, please install it with `pip install nncf`"
        )

    model_kwargs = model_kwargs or {}
    library_name = TasksManager._infer_library_from_model(model)
    TasksManager.standardize_model_attributes(model, library_name)

    if hasattr(model.config, "export_model_type"):
        model_type = model.config.export_model_type.replace("_", "-")
    else:
        model_type = model.config.model_type.replace("_", "-")

    custom_architecture = library_name == "transformers" and model_type not in TasksManager._SUPPORTED_MODEL_TYPE

    if task is not None:
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

    stateful = stateful and ensure_export_task_support_stateful(task)

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
        input_shapes[input_name] = (
            kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
        )

    logging.disable(logging.INFO)
    export_config, models_and_export_configs = _get_submodels_and_export_configs(
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
    )
    logging.disable(logging.NOTSET)

    if ov_config is None:
        if library_name == "diffusers":
            num_parameters = model.unet.num_parameters()
        else:
            num_parameters = sum(param.numel() for param in list(model.parameters()) if param.requires_grad)

        if num_parameters >= _MAX_UNCOMPRESSED_SIZE:
            if is_nncf_available():
                from ...intel.openvino.configuration import OVConfig

                ov_config = OVConfig(quantization_config={"bits": 8})

                logger.info("The model weights will be quantized to int8.")
            else:
                logger.warning(
                    "The model will be converted with no weights quantization. Quantization of the weights to int8 requires nncf."
                    "please install it with `pip install nncf`"
                )

    if library_name != "diffusers":
        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(output)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            try:
                generation_config.save_pretrained(output)
            except Exception as exception:
                logger.warning(
                    f"The generation config will not be saved, saving failed with following error:\n{exception}"
                )

        model_name_or_path = model.config._name_or_path
        maybe_save_preprocessors(model_name_or_path, output, trust_remote_code=trust_remote_code)

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

        model.save_config(output)

    export_models(
        models_and_export_configs=models_and_export_configs,
        output_dir=output,
        output_names=files_subpaths,
        input_shapes=input_shapes,
        device=device,
        ov_config=ov_config,
        stateful=stateful,
        opset=opset,
        model_kwargs=model_kwargs,
    )


def export_tokenizer(
    tokenizer,
    output: Union[str, Path],
    suffix: Optional[str] = "",
):
    # avoid circular imports
    from optimum.intel.openvino import OV_DETOKENIZER_NAME, OV_TOKENIZER_NAME
    from optimum.intel.openvino.utils import maybe_convert_tokenizer_to_fast

    try:
        from openvino_tokenizers import convert_tokenizer
    except ModuleNotFoundError:
        return

    if not isinstance(output, Path):
        output = Path(output)

    if output.exists():
        tokenizer = maybe_convert_tokenizer_to_fast(tokenizer, output)

    try:
        converted = convert_tokenizer(tokenizer, with_detokenizer=True)
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
            model.set_rt_info(_optimum_version, ["optimum", "diffusers_version"])
        elif library_name == "timm":
            model.set_rt_info(_timm_version, ["optimum", "timm_version"])
        rt_info = model.get_rt_info()
        if "nncf" in rt_info:
            model.set_rt_info(_nncf_version, ["optimum", "nncf_version"])
        input_model = rt_info["conversion_parameters"].get("input_model", None)
        if input_model is not None and "onnx" in input_model.value:
            model.set_rt_info(onnx.__version__, ["optimum", "onnx_version"])

    except Exception:
        pass

    return model
