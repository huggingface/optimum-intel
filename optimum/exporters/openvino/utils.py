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

from transformers.utils import is_torch_available

from openvino.runtime import PartialShape


if is_torch_available():
    import torch
    import torch.nn as nn


def is_torch_model(model):
    if not is_torch_available():
        return False
    return isinstance(model, nn.Module)


def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def remove_none_from_dummy_inputs(dummy_inputs):
    def remove_none_from_list_tuple(item):
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


def get_input_shapes(dummy_inputs, inputs):
    input_info = []
    for input_name, data in dummy_inputs.items():
        if isinstance(data, (tuple, list, dict)):
            return None
        static_shape = PartialShape(data.shape)
        if input_name in inputs:
            dynamic_dims = inputs[input_name]
            for dim in dynamic_dims:
                static_shape[dim] = -1
        input_info.append((input_name, static_shape))
    return input_info


def clear_class_registry():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()
