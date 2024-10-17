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

import inspect
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.utils import is_torch_available

from openvino.runtime import Dimension, PartialShape, Symbol
from openvino.runtime.utils.types import get_element_type
from optimum.exporters import TasksManager
from optimum.exporters.onnx.base import OnnxConfig
from optimum.utils import is_diffusers_available


InputInfo = namedtuple("InputInfo", ["name", "shape", "type", "example"])


if is_torch_available():
    import torch
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel

if is_diffusers_available():
    from diffusers import ModelMixin


OV_XML_FILE_NAME = "openvino_model.xml"
_MAX_UNCOMPRESSED_SIZE = 1e9


def is_torch_model(model: Union["PreTrainedModel", "ModelMixin"]):
    """
    Checks whether the model is a torch model.

    Args:
        model (Union[PretrainedModel, ModelMixin]): The model to check.

    Returns:
        bool: True if the model is a torch model.
    """
    if not is_torch_available():
        return False
    return isinstance(model, nn.Module)


def flattenize_inputs(inputs: List[Any]):
    """
    Flatten the inputs into a list.

    Args:
        inputs (List[Any]): The inputs to flatten.

    Returns:
        List[Any]:  The flattened inputs.
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def _get_input_info(
    model: Union["PreTrainedModel", "ModelMixin"], config: OnnxConfig, dummy_inputs: Dict[str, Any]
) -> List[InputInfo]:
    sig = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.call)
    inputs = config.ordered_inputs(model)
    ordered_dummy_inputs = {param: dummy_inputs[param] for param in sig.parameters if param in dummy_inputs}
    if not ordered_dummy_inputs:
        ordered_dummy_inputs = dummy_inputs
    ordered_input_names = list(inputs)
    flatten_inputs = flattenize_inputs(ordered_dummy_inputs.values())
    input_info = []

    name_to_symbol = {}

    for i in range(len(ordered_input_names)):
        name = ordered_input_names[i]
        example = flatten_inputs[i]
        type = get_element_type(example.cpu().numpy().dtype)
        shape = PartialShape(example.shape)
        if name in inputs:
            named_dims = inputs[name]
            for idx, dim_name in named_dims.items():
                if dim_name in name_to_symbol:
                    symbol = name_to_symbol[dim_name]
                else:
                    symbol = Symbol()
                    name_to_symbol[dim_name] = symbol
                dim = Dimension(-1)
                dim.set_symbol(symbol)
                shape[idx] = dim
        info = InputInfo(name=name, shape=shape, type=type, example=example)
        input_info.append(info)
    return input_info


def remove_none_from_dummy_inputs(dummy_inputs: Dict[str, Any]):
    """
    Removes None values from the dictionary.

    Args:
        dummy_inputs (Dict[str, Any]): Dictionary with None values.
    Returns:
        upd_dummy (Dict[str, Any]): updated dictionary with removed None values
        dict_dummy (List[Tuple[str, List[str]]]): list of inputs represented as dictionary provided as pair name and list of nested keys
    """

    def remove_none_from_list_tuple(item: Union[List[Any], Tuple[Any]]):
        """
        Removes None values from a list or tuple.

        Args:
            item (list or tuple): The list or tuple to remove None values from.

        Returns:
            list or tuple: The list or tuple with None values removed.
        """
        new_item = [i for i in item if i is not None]
        return type(item)(new_item)

    upd_dummy = {}
    dict_dummy = []
    for k, v in dummy_inputs.items():
        if v is None:
            continue
        if isinstance(v, dict):
            dict_dummy.append((k, list(v.keys())))
            upd_dummy[k] = remove_none_from_list_tuple(tuple(v.values()))
            continue
        if isinstance(v, (tuple, list)):
            upd_dummy[k] = remove_none_from_list_tuple(v)
            continue
        upd_dummy[k] = v
    return upd_dummy, dict_dummy


def clear_class_registry():
    """
    Removes Torchscript cached modules
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def _get_open_clip_submodels_fn_and_export_configs(
    model,
    library_name: str = "open_clip",
    task: Optional[str] = None,
    preprocessors: List = None,
    custom_export_configs: Dict[str, "OnnxConfig"] = None,
    fn_get_submodels: Callable = None,
):
    custom_export = {}
    if not custom_export_configs or "model_vision" in custom_export_configs:
        visual_model = model.visual
        setattr(visual_model, "config", model.config.vision_config)
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model.visual, exporter="openvino", task="feature-extraction", library_name=library_name
        )
        vision_cfg = export_config_constructor(
            model.config.vision_config,
            int_dtype="int64",
            float_dtype="fp32",
            preprocessors=preprocessors,
        )
        custom_export["model_vision"] = vision_cfg

    if not custom_export_configs or "model_text" in custom_export_configs:
        text_model = model.text
        setattr(text_model, "config", model.config.text_config)
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model.text, exporter="openvino", task="feature-extraction", library_name=library_name
        )
        text_cfg = export_config_constructor(
            model.config.text_config,
            int_dtype="int64",
            float_dtype="fp32",
            preprocessors=preprocessors,
        )
        custom_export["model_text"] = text_cfg

    if fn_get_submodels is None:

        def get_submodels(model):
            return {"model_text": model.text, "model_vision": model.visual}

        fn_get_submodels = get_submodels

    return custom_export, fn_get_submodels


MULTI_MODAL_TEXT_GENERATION_MODELS = ["llava", "llava-next", "internvl-chat"]
