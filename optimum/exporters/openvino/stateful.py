#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

import logging as log
from typing import List

import numpy as np
from transformers import PretrainedConfig

import openvino as ov
from openvino.runtime import opset13
from optimum.exporters import TasksManager
from optimum.intel.utils.import_utils import _openvino_version, is_openvino_version


def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model, not_kv_inputs: List[str], key_value_input_names: List[str], gather_dim: int
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("input_ids").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("input_ids")
    batch = opset13.gather(opset13.shape_of(input_ids, output_type="i64"), opset13.constant([0]), opset13.constant(0))
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}
    # TODO: Can we derive the dimensions from the model topology?

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
            else:
                log.warn(f"Rank of {input.get_any_name()} input of the model is not 2, batch size is not set")

    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def ensure_stateful_is_available(warn=True):
    """
    Check openvino version and raise error if it does not support stateful models
    """
    if is_openvino_version("<", "2023.3"):
        if warn:
            log.warn(
                f"Could not create or use stateful model when using old version of openvino=={_openvino_version}. It may result in sub-optimal inference performance."
                "Install openvino>=2023.3.0."
            )
        return False
    return True


def ensure_export_task_support_stateful(task: str):
    task = TasksManager.map_from_synonym(task)
    return task == "text-generation-with-past"


def patch_stateful(config: PretrainedConfig, ov_model: ov.Model):
    """
    Apply stateful transformation to model to hide key values inputs inside model.
    Select transformation parameters based on model architecture

    Parameters:
        config (`PretrainedConfig`):
            model pretrained config
        ov_model (`ov.Model`):
            openvino model
    """

    key_value_input_names = [
        key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())
    ]
    key_value_output_names = [
        key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())
    ]
    not_kv_inputs = [
        input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())
    ]
    if not key_value_input_names or not key_value_output_names:
        return

    # By default, batch is the 0-th but chatglm uses 1-st dimension as batch
    # TODO: Deduce from a model via ordinal reshape (?) and topology
    batch_dim = 1 if config.model_type == "chatglm" and not hasattr(config, "rope_ratio") else 0

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    num_attention_heads = config.num_attention_heads if config.model_type == "bloom" else 1
    make_stateful(
        ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, batch_dim, num_attention_heads, None
    )
