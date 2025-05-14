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

from transformers import AutoImageProcessor, PretrainedConfig
from transformers.utils import is_torch_available

from openvino import Dimension, PartialShape, Symbol
from openvino.utils.types import get_element_type
from optimum.exporters import TasksManager
from optimum.exporters.onnx.base import OnnxConfig
from optimum.intel.utils import is_transformers_version
from optimum.intel.utils.import_utils import is_openvino_version, is_safetensors_available
from optimum.utils import is_diffusers_available
from optimum.utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors


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
    "llava-next-video",
    "llava-qwen2",
    "internvl-chat",
    "maira2",
    "minicpmv",
    "phi3-v",
    "qwen2-vl",
    "qwen2-5-vl",
    "got-ocr2",
    "gemma3",
    "idefics3",
    "smolvlm",
    "phi4mm",
    "phi4-multimodal",
    "llama4",
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
            and model_type in ["llava", "llava-next", "llava-next-video"]
            and preprocessors is not None
        ):
            if len(preprocessors) > 1 and getattr(preprocessors[1], "patch_size", None) is None:
                preprocessors[1].patch_size = config.vision_config.patch_size
                preprocessors[1].vision_feature_select_strategy = config.vision_feature_select_strategy
        for processor in preprocessors:
            try:
                processor.save_pretrained(output)
            except Exception as ex:
                logger.error(f"Saving {type(processor)} failed with {ex}")
        # phi4mm does not allow loading chat template in processor, it uses chat_template from tokenizer
        if model_type == "phi4mm" and (Path(output) / "chat_template.json").exists():
            (Path(output) / "chat_template.json").unlink()
    else:
        maybe_save_preprocessors(model_name_or_path, output, trust_remote_code=trust_remote_code)


COMPLEX_CHAT_TEMPLATES = {
    # minicpm3
    "{%- macro json_to_python_type(param_name, json_spec) %}\n{%- set basic_type_map = {\n  'string': 'str',\n  'number': 'float',\n  'integer': 'int',\n  'boolean': 'bool',\n  'null': 'None'\n} %}\n\n{%- if json_spec.enum %}\n  {{- param_name|title }}\n{%- elif basic_type_map[json_spec.type] is defined %}\n  {{- basic_type_map[json_spec.type] }}\n{%- elif json_spec.type == 'array' %}\n  {{- 'List[' +  json_to_python_type(param_name, json_spec['items']) + ']' }}\n{%- elif json_spec.type == 'object' %}\n  {{- 'Dict[str, ' + json_to_python_type(param_name, json_spec.additionalProperties if json_spec.additionalProperties else 'Any') + ']' if not json_spec.properties else param_name|title }}\n{%- elif json_spec.type is iterable %}\n  {{- 'Union[' }}\n  {%- for t in json_spec.type %}\n    {{- json_to_python_type(param_name, {'type': t}) }}\n    {{- ', ' if not loop.last }}\n  {%- endfor %}\n  {{- ']' }}\n{%- else %}\n  {{- 'Any' }}\n{%- endif %}\n{%- endmacro %}\n\n{%- macro object_to_fields(json_spec, field_indent) %}\n  {%- set o_ns = namespace(f = caller()) %}\n  {%- for param_name, param_fields in json_spec.properties|items %}\n    {%- if param_fields.enum %}\n      {{- '\\n\\nclass ' + param_name|title + '(Enum):\\n' }}\n      {%- for enum_option in param_fields.enum %}\n        {{- '    enum_' + loop.index0|string + ' = ' + enum_option|tojson + '\\n' }}\n      {%- endfor %}\n    {%- elif param_fields.type == 'object' and param_fields.properties %}\n      {%- call object_to_fields(param_fields, '    ') %}\n        {{- '\\n\\nclass ' + param_name|title + '(BaseModel):\\n' }}\n      {%- endcall %}\n    {%- elif param_fields.type == 'array' and param_fields['items'] and param_fields['items'].type == 'object' and param_fields['items'].properties %}\n      {%- call object_to_fields(param_fields['items'], '    ') %}\n        {{- '\\n\\nclass ' + param_name|title + '(BaseModel):\\n' }}\n      {%- endcall %}\n    {%- endif %}\n    {%- set param_default = param_fields.default|tojson if param_fields.default is string else param_fields.default|string if param_fields.default is defined else 'None' %}\n    {%- set o_ns.f = o_ns.f + field_indent + param_name + ': ' %}\n    {%- set o_ns.f = o_ns.f + ('Optional[' + json_to_python_type(param_name, param_fields) + ']' if param_name not in json_spec.required else json_to_python_type(param_name, param_fields)) %}\n    {%- if not param_fields.title and not param_fields.description and not param_fields.pattern %}\n      {%- set o_ns.f = o_ns.f + (' = ' + param_default if param_name not in json_spec.required else '') %}\n    {%- else %}\n      {%- set o_ns.f = o_ns.f + (' = Field(...' if param_name in json_spec.required else ' = Field(' + param_default) %}\n      {%- set o_ns.f = o_ns.f + (', description=' + param_fields.description|tojson if param_fields.description else '') %}\n      {%- set o_ns.f = o_ns.f + (', regex=' + param_fields.pattern|tojson if param_fields.pattern else '') %}\n      {%- set o_ns.f = o_ns.f + (', title=' + param_fields.title|tojson if param_fields.title else '') %}\n      {%- set o_ns.f = o_ns.f + ')' %}\n    {%- endif %}\n    {%- set o_ns.f = o_ns.f + '\\n' %}\n  {%- endfor %}\n  {{- o_ns.f }}\n{%- endmacro %}\n\n{%- macro tool_parser(tools) %}\n{%- for tool in tools %}\n  {%- if tool.type is not defined or tool.type == 'function' %}\n    {%- if tool.function is defined %}\n      {%- set tool = tool.function %}\n    {%- endif %}\n    {%- set tool_params = tool.parameters if tool.parameters is defined else none %}\n    {%- call object_to_fields(tool_params, '        ') %}\n      {{- '\\n\\ndef ' + tool.name + '(' }}\n      {%- if tool_params %}\n        {%- for param_name, param_fields in tool_params.properties|items %}\n          {%- set param_default = param_fields.default|tojson if param_fields.default is string else param_fields.default|string if param_fields.default is defined else 'None' %}\n          {{- ', ' if loop.index0 != 0 }}\n          {{- param_name }}\n          {{- '=' + param_default if param_name not in tool_params.required }}\n        {%- endfor %}\n      {%- endif %}\n      {{- '):\\n    \"\"\"' }}\n      {{- tool.description }}\n      {{- '\\n\\n    Args:\\n' if tool_params else '\\n' }}\n    {%- endcall %}\n    {{- '    \"\"\"\\n' }}\n  {%- endif %}\n{%- endfor %}\n{%- endmacro %}\n\n{%- if messages[0]['role'] == 'system' %}\n  {%- set loop_messages = messages[1:] %}\n  {%- set system_message = messages[0]['content'] %}\n{%- else %}\n  {%- set loop_messages = messages %}\n  {%- set system_message = '' %}\n{%- endif %}\n{{- '<|im_start|>system\\n' + system_message if system_message or tools }}\n{%- if tools %}\n  {{- '\\n# Functions\\nHere is a list of functions that you can invoke:\\n```python\\nfrom enum import Enum\\nfrom typing import List, Dict, Optional\\nfrom pydantic import BaseModel, Field\\n\\n' }}\n  {{- tool_parser(tools) }}\n  {{- \"\\n```\\n\\n# Function Call Rule and Output Format\\n- If the user's question can be answered without calling any function, please answer the user's question directly. In this situation, you should return your thought and answer the user's question directly.\\n- If the user cannot be answered without calling any function, and the user does not provide enough information to call functions, please ask the user for more information. In this situation, you should return your thought and ask the user for more information.\\n- If the user's question cannot be answered without calling any function, and the user has provided enough information to call functions to solve it, you should call the functions. In this situation, the assistant should return your thought and call the functions.\\n- Use default parameters unless the user has specified otherwise.\\n- You should answer in the following format:\\n\\n<|thought_start|>\\n{explain why the user's question can be answered without calling a function or why you should ask the user for more information or why you should call one or more functions and your plan to solve the user's question.}\\n<|thought_end|>\\n<|tool_call_start|>\\n```python\\nfunc1(params_name=params_value, params_name2=params_value2...)\\nfunc2(params)\\n```\\n<|tool_call_end|>\\n{answer the user's question directly or ask the user for more information}\" }}\n{%- endif %}\n{{- '<|im_end|>\\n' if system_message or tools }}\n{%- for message in loop_messages %}\n  {%- set content = message.content %}\n  {%- if message.role == 'assistant' and message.tool_calls %}\n    {{- '<|im_start|>' + message.role + '\\n' }}\n    {{- '<|thought_start|>\\n' + message.thought + '\\n<|thought_end|>\\n' if message.thought }}\n    {{- '<|tool_call_start|>\\n```python\\n' }}\n    {%- for tool_call in message.tool_calls %}\n      {%- if tool_call.function is defined %}\n        {%- set tool_call = tool_call.function %}\n      {%- endif %}\n      {{- tool_call.name + '(' }}\n      {%- if tool_call.arguments is defined and tool_call.arguments|length > 0 %}\n        {%- for param_name, param_value in tool_call.arguments|items %}\n          {{- param_name + '=' + param_value|tojson }}\n          {{- ',' if not loop.last }}\n        {%- endfor %}\n      {%- endif %}\n      {{- ')\\n' }}\n    {%- endfor %}\n    {{- '```\\n<|tool_call_end|>\\n' }}\n    {{- content if content and not content.startswith('<|tool_call_start|>') }}\n    {{- '<|im_end|>\\n' }}\n  {%- elif message.role == 'assistant' and message.thought %}\n    {{- '<|im_start|>' + message.role + '\\n' + '<|thought_start|>\\n' + message.thought + '\\n<|thought_end|>\\n' + content + '<|im_end|>\\n' }}\n  {%- else %}\n    {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n  {%- endif %}\n{%- endfor %}\n\n{%- if add_generation_prompt %}\n  {{- '<|im_start|>assistant\\n' }}\n{%- endif %}": "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}{% elif message['role'] == 'user' %}{{  '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\\n ' + message['content'] + '<|im_end|>\\n' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}{% endfor %}",
    # deepseek-r1 old
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
    # deepseek-r1
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' in message %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{'<｜Assistant｜>' + message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' not in message %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
    # deepseek-r1-distilled-llama
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
    # llava-1.5
    "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}": "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'] | upper + ': ' }}{% endif %}{{ message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
    # llava-next old
    "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'][0]['text'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'][0]['text'] + '<\\s> '}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}": "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + '<\\s> ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
    # llava-next
    "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'][0]['text'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'][0]['text'] + '</s> '}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}": "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + '</s> ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
    # qwen2-vl-instruct
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    # qwen2-vl
    "{% if messages is string %}{{ messages }}{% else %}{% for content in messages %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    # falcon3-7b-instruct
    "{%- if tools %}\n{{- '<|system|>\\n' }}\n{%- if messages[0]['role'] == 'system' %}\n{{- messages[0]['content'] }}\n{%- set remaining_messages = messages[1:] %}\n{%- else %}\n{%- set remaining_messages = messages %}\n{%- endif %}\n{{- 'You are a Falcon assistant skilled in function calling. You are helpful, respectful, and concise.\\n\\n# Tools\\n\\nYou have access to the following functions. You MUST use them to answer questions when needed. For each function call, you MUST return a JSON object inside <tool_call></tool_call> tags.\\n\\n<tools>' + tools|tojson(indent=2) + '</tools>\\n\\n# Output Format\\n\\nYour response MUST follow this format when making function calls:\\n<tool_call>\\n[\\n  {\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}},\\n  {\"name\": \"another_function\", \"arguments\": {\"arg\": \"value\"}}\\n]\\n</tool_call>\\nIf no function calls are needed, respond normally without the tool_call tags.\\n' }}\n{%- for message in remaining_messages %}\n{%- if message['role'] == 'user' %}\n{{- '<|user|>\\n' + message['content'] + '\\n' }}\n{%- elif message['role'] == 'assistant' %}\n{%- if message.content %}\n{{- '<|assistant|>\\n' + message['content'] }}\n{%- endif %}\n{%- if message.tool_calls %}\n{{- '\\n<tool_call>\\n' }}\n{{- message.tool_calls|tojson(indent=2) }}\n{{- '\\n</tool_call>' }}\n{%- endif %}\n{{- eos_token + '\\n' }}\n{%- elif message['role'] == 'tool' %}\n{{- '<|assistant|>\\n<tool_response>\\n' + message['content'] + '\\n</tool_response>\\n' }}\n{%- endif %}\n{%- endfor %}\n{{- '<|assistant|>\\n' if add_generation_prompt }}\n{%- else %}\n{%- for message in messages %}\n{%- if message['role'] == 'system' %}\n{{- '<|system|>\\n' + message['content'] + '\\n' }}\n{%- elif message['role'] == 'user' %}\n{{- '<|user|>\\n' + message['content'] + '\\n' }}\n{%- elif message['role'] == 'assistant' %}\n{%- if not loop.last %}\n{{- '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}\n{%- else %}\n{{- '<|assistant|>\\n' + message['content'] + eos_token }}\n{%- endif %}\n{%- endif %}\n{%- if loop.last and add_generation_prompt %}\n{{- '<|assistant|>\\n' }}\n{%- endif %}\n{%- endfor %}\n{%- endif %}": "{%- for message in messages %}\n{%- if message['role'] == 'system' %}\n{{- '<|system|>\\n' + message['content'] + '\\n' }}\n{%- elif message['role'] == 'user' %}\n{{- '<|user|>\\n' + message['content'] + '\\n' }}\n{%- elif message['role'] == 'assistant' %}\n{%- if not loop.last %}\n{{- '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}\n{%- else %}\n{{- '<|assistant|>\\n' + message['content'] + eos_token }}\n{%- endif %}\n{%- endif %}\n{%- if loop.last and add_generation_prompt %}\n{{- '<|assistant|>\\n' }}\n{%- endif %}\n{%- endfor %}",
    # Mistral-7B-Instruct-v0.3
    '{%- if messages[0]["role"] == "system" %}\n    {%- set system_message = messages[0]["content"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}\n            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message["role"] == "user" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- "[AVAILABLE_TOOLS] [" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- \'{"type": "function", "function": {\' }}\n                {%- for key, val in tool.items() if key != "return" %}\n                    {%- if val is string %}\n                        {{- \'"\' + key + \'": "\' + val + \'"\' }}\n                    {%- else %}\n                        {{- \'"\' + key + \'": \' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- ", " }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- "}}" }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- else %}\n                    {{- "]" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- "[/AVAILABLE_TOOLS]" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "[/INST]" }}\n        {%- else %}\n            {{- "[INST] " + message["content"] + "[/INST]" }}\n        {%- endif %}\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\n        {{- "[TOOL_CALLS] [" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n            {%- endif %}\n            {{- \', "id": "\' + tool_call.id + \'"}\' }}\n            {%- if not loop.last %}\n                {{- ", " }}\n            {%- else %}\n                {{- "]" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message["role"] == "assistant" %}\n        {{- " " + message["content"]|trim + eos_token}}\n    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- \'[TOOL_RESULTS] {"content": \' + content|string + ", " }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n        {%- endif %}\n        {{- \'"call_id": "\' + message.tool_call_id + \'"}[/TOOL_RESULTS]\' }}\n    {%- else %}\n        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\n    {%- endif %}\n{%- endfor %}\n': "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{% if (messages[0]['role'] == 'system' and messages|length == 2) %}{{ message['content'] + '[/INST]' }}{% else %}{{ '[INST] ' + message['content'] + '[/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}{% elif (message['role'] == 'system' and messages|length == 2) %}{{ '[INST] ' + message['content'] + ' \n\n' }}{% else %}{{ raise_exception('Only system, user and assistant roles are supported!') }}{% endif %}{% endfor %}",
    # Qwen3
    "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
}


def set_simplified_chat_template(ov_tokenizer_model, processor_chat_template=None):
    tokenizer_chat_template = None
    if ov_tokenizer_model.has_rt_info("chat_template"):
        tokenizer_chat_template = ov_tokenizer_model.get_rt_info("chat_template")
    if processor_chat_template is not None:
        tokenizer_chat_template = processor_chat_template
        ov_tokenizer_model.set_rt_info(processor_chat_template, "chat_template")
    if tokenizer_chat_template is not None and tokenizer_chat_template in COMPLEX_CHAT_TEMPLATES:
        ov_tokenizer_model.set_rt_info(COMPLEX_CHAT_TEMPLATES[tokenizer_chat_template], "simplified_chat_template")
    return ov_tokenizer_model


SKIP_CHECK_TRACE_MODELS = (
    "deepseek",
    "deepseek-v2",
    "deepseek-v3",
    "levit",
    "llama4",
)

if is_transformers_version("<", "4.41"):
    SKIP_CHECK_TRACE_MODELS += ("gemma",)


def allow_skip_tracing_check(library_name, model_type):
    if is_openvino_version("<", "2025.0.0"):
        return False
    if library_name == "diffusers":
        return True
    return model_type in SKIP_CHECK_TRACE_MODELS


# TO DO: load_preprocessors should be removed once this is included in https://github.com/huggingface/optimum/blob/fa87c66967595b8af4de529500868840a3443611/optimum/utils/save_utils.py#L27 (
def load_preprocessors(
    src_name_or_path: Union[str, Path], subfolder: str = "", trust_remote_code: bool = False, model_type: str = None
):
    preprocessors = maybe_load_preprocessors(
        src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
    )
    if model_type == "phi4mm":
        # audio feature extractor config overrides image processor config during saving, need to save it explicitly
        try:
            preprocessors.append(
                AutoImageProcessor.from_pretrained(
                    src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
                )
            )
        except Exception:
            pass
    return preprocessors
