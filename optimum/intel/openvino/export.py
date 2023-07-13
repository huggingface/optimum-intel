import os
import logging
import inspect
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import time

import numpy as np
from transformers.utils import is_tf_available, is_torch_available

from optimum.utils import TORCH_MINIMUM_VERSION, is_diffusers_available, is_torch_onnx_support_available, logging
from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.convert import export_tensorflow, check_dummy_inputs_are_allowed

from openvino.tools import mo
from openvino.runtime import serialize, PartialShape
from openvino.runtime.utils.types import get_element_type
from .utils import OV_XML_FILE_NAME

if is_torch_available():
    import torch
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel
    from transformers.pytorch_utils import is_torch_less_than_1_11

if is_diffusers_available():
    from diffusers import ModelMixin

if is_tf_available():
    from transformers.modeling_tf_utils import TFPreTrainedModel

def is_torch_model(model):
    if not is_torch_available():
        return False
    return isinstance(model, nn.Module)

def export(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    output: Path,
    opset: Optional[int] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a Pytorch or TensorFlow model to an ONNX Intermediate Representation.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        output (`Path`):
            Directory to store the exported ONNX model.
        opset (`Optional[int]`, defaults to `None`):
            The version of the ONNX operator set to use.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.

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

    if is_torch_available() and isinstance(model, nn.Module):
        return export_pytorch(model, config, opset, output, device=device, input_shapes=input_shapes)

    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        output.parent.mkdir(parents=True, exist_ok=True)
        if opset is None:
            opset = config.DEFAULT_ONNX_OPSET
        if device == "cuda":
            raise RuntimeError("`tf2onnx` does not support export on CUDA device.")
        if input_shapes is not None:
            print("`input_shapes` argument is not supported by the Tensorflow ONNX export and will be ignored.")
        return export_tensorflow(model, config, opset, output)

    else:
        raise RuntimeError(
            "You either provided a PyTorch model with only TensorFlow installed, or a TensorFlow model with only PyTorch installed."
        )


def export_pytorch(
    model: Union["PreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to an ONNX Intermediate Representation.

    Args:
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    import torch
    from torch.onnx import export as onnx_export
    from torch.utils._pytree import tree_map

    print(f"Using framework PyTorch: {torch.__version__}")

    with torch.no_grad():
        model.config.return_dict = True
        model.config.torchscript = True
        model.eval()

        # Check if we need to override certain configuration item
        if config.values_override is not None:
            print(f"Overriding {len(config.values_override)} configuration item(s)")
            for override_config_key, override_config_value in config.values_override.items():
                print(f"\t- {override_config_key} -> {override_config_value}")
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
        check_dummy_inputs_are_allowed(model, dummy_inputs)
        inputs = config.ordered_inputs(model)
        input_names = list(inputs.keys())
        output_names = list(config.outputs.keys())

        if hasattr(config, "patch_ops"):
            config.patch_ops()
        
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
        else:
            sig = inspect.signature(model.call)

        input_info = get_input_shapes(dummy_inputs, inputs)
        start0 = time.perf_counter()
        ov_model = mo.convert_model(model, example_input=dummy_inputs, input=input_info)
        end0 = time.perf_counter()
        print(f"Convert model took {end0 - start0}s")
        ordered_dummy_inputs = {param: dummy_inputs[param] for param in sig.parameters if param in dummy_inputs}
        ordered_input_names = list(inputs)
        flatten_inputs = flattenize_inputs(ordered_dummy_inputs.values())
                
        for idx, inp_tensor in enumerate(ov_model.inputs):
            input_name = ordered_input_names[idx]
            inp_tensor.get_tensor().set_names({input_name})
            inp_data = flatten_inputs[idx]
            static_shape = PartialShape(inp_data.shape)
            dims = inputs[input_name]

            for dim in dims:
                static_shape[dim] = -1 
            inp_tensor.get_node().set_partial_shape(static_shape)
            inp_tensor.get_node().set_element_type(get_element_type(inp_data.cpu().numpy().dtype))

        for idx, out_tensor in enumerate(ov_model.outputs):
            if idx < len(output_names):
                out_tensor.get_tensor().set_names({output_names[idx]})
        ov_model.validate_nodes_and_infer_types()
        start1 = time.perf_counter()
        serialize(ov_model, output.parent / OV_XML_FILE_NAME)
        end1 = time.perf_counter()
        print(f"Serailize model took {end1 - start1}s")
        if hasattr(config, "restore_ops"):
            config.restore_ops()

    return input_names, output_names


def export_models(
    models_and_onnx_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]
    ],
    output_dir: Path,
    opset: Optional[int] = None,
    output_names: Optional[List[str]] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Exports a Pytorch or TensorFlow encoder decoder model to an ONNX Intermediate Representation.
    The following method exports the encoder and decoder components of the model as separate
    ONNX files.

    Args:
        models_and_onnx_configs (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]]):
            A dictionnary containing the models to export and their corresponding onnx configs.
        output_dir (`Path`):
            Output directory to store the exported ONNX models.
        opset (`Optional[int]`, defaults to `None`):
            The version of the ONNX operator set to use.
        output_names (`Optional[List[str]]`, defaults to `None`):
            The names to use for the exported ONNX files. The order must be the same as the order of submodels in the ordered dict `models_and_onnx_configs`.
            If None, will use the keys from `models_and_onnx_configs` as names.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
    Returns:
        `Tuple[List[List[str]], List[List[str]]]`: A tuple with an ordered list of the model's inputs, and the named
        inputs from the ONNX configuration.
    """
    outputs = []

    if output_names is not None and len(output_names) != len(models_and_onnx_configs):
        raise ValueError(
            f"Provided custom names {output_names} for the export of {len(models_and_onnx_configs)} models. Please provide the same number of names as models to export."
        )

    for i, model_name in enumerate(models_and_onnx_configs.keys()):
        submodel, sub_onnx_config = models_and_onnx_configs[model_name]
        output_name = output_names[i] if output_names is not None else Path(model_name + ".xml")
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(
            export(
                model=submodel,
                config=sub_onnx_config,
                output=output_path,
                opset=opset,
                device=device,
                input_shapes=input_shapes,
            )
        )

    outputs = list(map(list, zip(*outputs)))
    return outputs


def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def get_input_shapes(dummy_inputs, inputs):
    input_info = []
    for input_name, data in dummy_inputs.items():
        if isinstance(data, (tuple, list)):
            return None
        static_shape = PartialShape(data.shape)
        if input_name in inputs:
            dynamic_dims = inputs[input_name]
            for dim in dynamic_dims:
                static_shape[dim] = -1
        input_info.append((input_name, static_shape))
    return input_info