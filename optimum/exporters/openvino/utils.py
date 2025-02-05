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
import logging
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import PretrainedConfig
from transformers.utils import is_torch_available

from openvino.runtime import Dimension, PartialShape, Symbol
from openvino.runtime.utils.types import get_element_type
from optimum.exporters import TasksManager
from optimum.exporters.onnx.base import OnnxConfig
from optimum.intel.utils import is_transformers_version
from optimum.intel.utils.import_utils import is_safetensors_available
from optimum.utils import is_diffusers_available
from optimum.utils.save_utils import maybe_save_preprocessors


logger = logging.getLogger(__name__)


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


MULTI_MODAL_TEXT_GENERATION_MODELS = [
    "llava",
    "llava-next",
    "llava-qwen2",
    "internvl-chat",
    "minicpmv",
    "phi3-v",
    "qwen2-vl",
]


def save_config(config, save_dir):
    try:
        config.save_pretrained(save_dir)
    except Exception as exp:
        logger.warning(
            f"Attempt to save config using standard API has failed with {exp}. There may be an issue with model config, please check its correctness before usage."
        )
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        output_config_file = Path(save_dir / "config.json")
        config.to_json_file(output_config_file, use_diff=True)


def deduce_diffusers_dtype(model_name_or_path, **loading_kwargs):
    dtype = None
    if is_safetensors_available():
        if Path(model_name_or_path).is_dir():
            path = Path(model_name_or_path)
        else:
            from diffusers import DiffusionPipeline

            path = Path(DiffusionPipeline.download(model_name_or_path, **loading_kwargs))
        model_part_name = None
        if (path / "transformer").is_dir():
            model_part_name = "transformer"
        elif (path / "unet").is_dir():
            model_part_name = "unet"
        if model_part_name:
            directory = path / model_part_name

            pattern = "*.safetensors"
            if "variant" in loading_kwargs:
                variant = loading_kwargs["variant"]
                pattern = f"*.{variant}.safetensors"
                safetensors_files = list(directory.glob(pattern))
            else:
                # filter out variant files
                safetensors_files = [filename for filename in directory.glob(pattern) if len(filename.suffixes) == 1]
            safetensors_file = None
            if len(safetensors_files) > 0:
                safetensors_file = safetensors_files.pop(0)
            if safetensors_file:
                from safetensors import safe_open

                with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                    if len(f.keys()) > 0:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            if tensor.dtype.is_floating_point:
                                dtype = tensor.dtype
                                break
    return dtype


def save_preprocessors(
    preprocessors: List, config: PretrainedConfig, output: Union[str, Path], trust_remote_code: bool
):
    model_name_or_path = config._name_or_path
    if hasattr(config, "export_model_type"):
        model_type = config.export_model_type.replace("_", "-")
    else:
        model_type = config.model_type.replace("_", "-")
    if preprocessors is not None:
        # phi3-vision processor does not have chat_template attribute that breaks Processor saving on disk
        if is_transformers_version(">=", "4.45") and model_type == "phi3-v" and len(preprocessors) > 1:
            if not hasattr(preprocessors[1], "chat_template"):
                preprocessors[1].chat_template = getattr(preprocessors[0], "chat_template", None)
        if (
            is_transformers_version(">=", "4.45")
            and model_type in ["llava", "llava-next"]
            and preprocessors is not None
        ):
            if getattr(preprocessors[1], "patch_size", None) is None:
                preprocessors[1].patch_size = config.vision_config.patch_size
                preprocessors[1].vision_feature_select_strategy = config.vision_feature_select_strategy
        for processor in preprocessors:
            try:
                processor.save_pretrained(output)
            except Exception as ex:
                logger.error(f"Saving {type(processor)} failed with {ex}")
    else:
        maybe_save_preprocessors(model_name_or_path, output, trust_remote_code=trust_remote_code)
