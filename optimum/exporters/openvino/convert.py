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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.utils import is_tf_available, is_torch_available

from openvino.runtime import PartialShape, save_model
from openvino.runtime.utils.types import get_element_type
from openvino.tools.ovc import convert_model
from optimum.exporters.onnx.base import OnnxConfig
from optimum.exporters.onnx.convert import check_dummy_inputs_are_allowed, export_tensorflow
from optimum.exporters.onnx.convert import export_pytorch as export_pytorch_to_onnx
from optimum.utils import is_diffusers_available

from ...intel.openvino.utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME
from .utils import (
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


def export(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"],
    config: OnnxConfig,
    output: Path,
    opset: Optional[int] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a Pytorch or TensorFlow model to an OpenVINO Intermediate Representation.

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
        return export_pytorch(
            model, config, opset, output, device=device, input_shapes=input_shapes, model_kwargs=model_kwargs
        )

    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        output.parent.mkdir(parents=True, exist_ok=True)
        if opset is None:
            opset = config.DEFAULT_ONNX_OPSET
        if device == "cuda":
            raise RuntimeError("`tf2onnx` does not support export on CUDA device.")
        if input_shapes is not None:
            logger.info("`input_shapes` argument is not supported by the Tensorflow ONNX export and will be ignored.")
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
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Exports a PyTorch model to an OpenVINO Intermediate Representation.

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
    from torch.utils._pytree import tree_map

    logger.info(f"Using framework PyTorch: {torch.__version__}")
    output = Path(output)

    with torch.no_grad():
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
        check_dummy_inputs_are_allowed(model, dummy_inputs)
        inputs = config.ordered_inputs(model)
        input_names = list(inputs.keys())
        output_names = list(config.outputs.keys())
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
        else:
            sig = inspect.signature(model.call)

        dummy_inputs, dict_inputs = remove_none_from_dummy_inputs(dummy_inputs)
        input_info = get_input_shapes(dummy_inputs, inputs)
        try:
            patcher = config.patch_model_for_export(model, model_kwargs=model_kwargs)
            patched_forward = patcher.patched_forward

            @functools.wraps(patched_forward)
            def ts_patched_forward(*args, **kwargs):
                for i in range(len(dict_inputs)):
                    input_name = dict_inputs[i][0]
                    keys = dict_inputs[i][1]
                    tuple_input = kwargs[input_name]
                    input_dict = dict(zip(keys, tuple_input))
                    kwargs[input_name] = input_dict
                outputs = patched_forward(*args, **kwargs)
                return tuple(outputs.values())

            patcher.patched_forward = ts_patched_forward
            with patcher:
                ov_model = convert_model(model, example_input=dummy_inputs, input=input_info)
        except Exception:
            orig_torch_onnx_export = torch.onnx.export

            torch.onnx.export = functools.partial(orig_torch_onnx_export, do_constant_folding=True)
            model.config.torchscript = False
            model.config.return_dict = True
            onnx_output = (
                output.with_suffix(".onnx")
                if not output.name != OV_XML_FILE_NAME
                else output.parent / ONNX_WEIGHTS_NAME
            )
            input_names, output_names = export_pytorch_to_onnx(
                model, config, opset, onnx_output, device, input_shapes, model_kwargs
            )
            torch.onnx.export = orig_torch_onnx_export
            ov_model = convert_model(str(onnx_output))
            save_model(
                ov_model,
                output.parent / OV_XML_FILE_NAME if output.suffix != ".xml" else output,
                compress_to_fp16=False,
            )
            return input_names, output_names, True
        clear_class_registry()
        ordered_dummy_inputs = {param: dummy_inputs[param] for param in sig.parameters if param in dummy_inputs}
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
            dims = inputs[input_name]

            for dim in dims:
                static_shape[dim] = -1
            inp_tensor.get_node().set_partial_shape(static_shape)
            inp_tensor.get_node().set_element_type(get_element_type(inp_data.cpu().numpy().dtype))
        ov_model.validate_nodes_and_infer_types()
        save_model(
            ov_model, output.parent / OV_XML_FILE_NAME if output.suffix != ".xml" else output, compress_to_fp16=False
        )
        del model
        gc.collect()
    return input_names, output_names, False


def export_models(
    models_and_onnx_configs: Dict[
        str, Tuple[Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin"], "OnnxConfig"]
    ],
    output_dir: Path,
    opset: Optional[int] = None,
    output_names: Optional[List[str]] = None,
    device: str = "cpu",
    input_shapes: Optional[Dict] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
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
                model_kwargs=model_kwargs,
            )
        )

    outputs = list(map(list, zip(*outputs)))
    return outputs
