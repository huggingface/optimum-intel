#  Copyright 2026 The HuggingFace Team. All rights reserved.
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

# Conversion rule for the `RecurrentAttentionCellOp` operation in a Torch graph.
# The `RecurrentAttentionCellOp` appears in the Torch graph as a result of replacing
# the `torch.nn.Module` block `RecurrentAttentionCell` via a registered
# `ModuleExtension` for `RecurrentAttentionCell` in the OpenVINO PyTorch frontend.
import numpy as np

import openvino as ov
import openvino.opset14 as ops


def convert_recurrent_attention_cell(context):
    query = context.get_input(0)
    key = context.get_input(1)
    value = context.get_input(2)
    g = context.get_input(3)
    beta = context.get_input(4)
    last_recurrent_state_old = context.get_input(5)

    value_shape = ops.shape_of(value)
    const_zero = ops.constant(0, dtype=np.float32)
    core_attn_out = ops.broadcast(const_zero, value_shape)
    const_two_out = ops.constant(2, dtype=np.int32)
    const_zero_out = ops.constant(0, dtype=np.int32)
    seq_len = ops.gather(value_shape, const_two_out, const_zero_out)

    timestep_param = ops.parameter([], np.int32, "timestep")
    q_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "q_t")
    k_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "k_t")
    v_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "v_t")
    g_t_param = ops.parameter([-1, -1, 1], np.float32, "g_t")
    beta_t_param = ops.parameter([-1, -1, 1], np.float32, "beta_t")
    last_recurrent_state_t = ops.parameter([-1, -1, -1, -1], np.float32, "last_recurrent_state_t")
    core_attn_out_t = ops.parameter([-1, -1, -1, -1], np.float32, "core_attn_out_t")

    const_two = ops.constant(2, dtype=np.int32)
    q_t = ops.squeeze(q_t_param, const_two)
    k_t = ops.squeeze(k_t_param, const_two)
    v_t = ops.squeeze(v_t_param, const_two)
    const_minus_one = ops.constant(-1, dtype=np.int32)
    g_t = ops.unsqueeze(ops.exp(g_t_param), const_minus_one)
    beta_t = beta_t_param

    last_recurrent_state_in = ops.multiply(last_recurrent_state_t, g_t)
    const_minus_two = ops.constant(-2, dtype=np.int32)
    kv_mem = ops.multiply(last_recurrent_state_in, ops.unsqueeze(k_t, const_minus_one))
    kv_mem = ops.reduce_sum(kv_mem, const_minus_two, False)
    delta = ops.multiply(ops.subtract(v_t, kv_mem), beta_t)
    last_recurrent_state_delta = ops.multiply(
        ops.unsqueeze(k_t, const_minus_one), ops.unsqueeze(delta, const_minus_two)
    )
    last_recurrent_state_in = ops.add(last_recurrent_state_in, last_recurrent_state_delta)
    core_attn_update = ops.multiply(last_recurrent_state_in, ops.unsqueeze(q_t, const_minus_one))
    core_attn_update = ops.reduce_sum(core_attn_update, const_minus_two, True)
    const_zero = ops.constant(0, dtype=np.int32)
    timestep = ops.unsqueeze(timestep_param, const_zero)

    core_attn_out_res = ops.scatter_update(core_attn_out_t, timestep, core_attn_update, const_two)
    last_recurrent_state_res = last_recurrent_state_in

    body_cond = ops.constant([True], dtype=bool)

    body_model = ov.Model(
        [body_cond, last_recurrent_state_res, core_attn_out_res],
        [
            timestep_param,
            q_t_param,
            k_t_param,
            v_t_param,
            g_t_param,
            beta_t_param,
            last_recurrent_state_t,
            core_attn_out_t,
        ],
        "body_model",
    )

    seq_len = ops.convert(seq_len, "i32")
    loop = ops.loop(seq_len, ops.constant(True, dtype="bool"))
    loop.set_function(body_model)

    loop.set_sliced_input(q_t_param, query, 0, 1, 1, -1, 2)
    loop.set_sliced_input(k_t_param, key, 0, 1, 1, -1, 2)
    loop.set_sliced_input(v_t_param, value, 0, 1, 1, -1, 2)
    loop.set_sliced_input(g_t_param, g, 0, 1, 1, -1, 2)
    loop.set_sliced_input(beta_t_param, beta, 0, 1, 1, -1, 2)
    loop.set_merged_input(last_recurrent_state_t, last_recurrent_state_old, last_recurrent_state_res.output(0))
    loop.set_merged_input(core_attn_out_t, core_attn_out.output(0), core_attn_out_res.output(0))
    loop.set_special_body_ports([0, 0])

    core_attn_out_new = loop.get_iter_value(core_attn_out_res.output(0), -1)
    last_recurrent_state_new = loop.get_iter_value(last_recurrent_state_res.output(0), -1)

    flatten_shape = ops.constant([-1], dtype=np.int32)
    core_attn_out_new = ops.reshape(core_attn_out_new, flatten_shape, False)
    last_recurrent_state_new = ops.reshape(last_recurrent_state_new, flatten_shape, False)

    final_output = ops.concat([core_attn_out_new, last_recurrent_state_new], 0)

    return [final_output.output(0)]




# Conversion rule for the `Mamba2RecurrentCellOp` operation in a Torch graph.
#
# This generalizes the recurrent-cell-to-`ov::Loop` approach (originally introduced for the
# GatedDeltaNet block, see `convert_recurrent_attention_cell` above) to the Mamba2 selective
# state-space recurrence used by hybrid Mamba2 models such as NemotronH.
#
# The Mamba2 single-step recurrence over the SSM state is a linear recurrence:
#       state_t = state_{t-1} * dA_t + dBx_t            # [B, H, P, N]
#       y_t     = reduce_sum(state_t * C_t, axis=N)      # [B, H, P]
# where `dA` (discretized A), `dBx` (discretized B * x) and `C` are precomputed and vectorized
# over the sequence in the patched mixer forward. The skip connection `x_t * D` does not depend
# on the recurrent state and is therefore added outside the loop.
#
# The `Mamba2RecurrentCellOp` appears in the Torch graph as a result of replacing the
# `Mamba2RecurrentCell` `torch.nn.Module` via a registered `ModuleExtension` in the OpenVINO
# PyTorch frontend; OpenVINO then applies this conversion rule to the resulting operation.
def convert_recurrent_mamba2_cell(context):
    # Inputs match the forward signature of `Mamba2RecurrentCell`.
    # `dA` is broadcastable ([B, H, T, 1, 1]) while `dBx` carries the full state shape
    # ([B, H, T, P, N]); shapes for the accumulator and the trip count are therefore
    # derived from `dBx`.
    dA = context.get_input(0)  # [B, H, T, 1, 1]
    dBx = context.get_input(1)  # [B, H, T, P, N]
    C = context.get_input(2)  # [B, H, T, N]
    last_state_old = context.get_input(3)  # [B, H, P, N]

    const_zero_axis = ops.constant(0, dtype=np.int32)
    const_two = ops.constant(2, dtype=np.int32)
    const_minus_one = ops.constant(-1, dtype=np.int32)

    # Build the zero-initialized output accumulator with shape [B, H, T, P].
    dBx_shape = ops.shape_of(dBx)
    core_shape = ops.gather(dBx_shape, ops.constant([0, 1, 2, 3], dtype=np.int32), const_zero_axis)
    const_zero_f32 = ops.constant(0, dtype=np.float32)
    core_out = ops.broadcast(const_zero_f32, core_shape)

    # Trip count for the loop equals the sequence length (dim 2).
    seq_len = ops.gather(dBx_shape, const_two, const_zero_axis)
    seq_len = ops.convert(seq_len, "i32")

    # Body parameters (one timestep slice each).
    timestep_param = ops.parameter([], np.int32, "timestep")
    dA_t_param = ops.parameter([-1, -1, 1, -1, -1], np.float32, "dA_t")
    dBx_t_param = ops.parameter([-1, -1, 1, -1, -1], np.float32, "dBx_t")
    C_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "C_t")
    last_state_t = ops.parameter([-1, -1, -1, -1], np.float32, "last_state_t")
    core_out_t = ops.parameter([-1, -1, -1, -1], np.float32, "core_out_t")

    # Drop the singleton sequence dimension introduced by slicing.
    dA_t = ops.squeeze(dA_t_param, const_two)  # [B, H, 1, 1] (broadcastable to [B, H, P, N])
    dBx_t = ops.squeeze(dBx_t_param, const_two)  # [B, H, P, N]
    C_t = ops.squeeze(C_t_param, const_two)  # [B, H, N]

    # state_t = state_{t-1} * dA_t + dBx_t
    last_state_new = ops.add(ops.multiply(last_state_t, dA_t), dBx_t)  # [B, H, P, N]

    # y_t = reduce_sum(state_t * C_t, axis=N) -> [B, H, P]
    const_minus_two = ops.constant(-2, dtype=np.int32)
    y_t = ops.multiply(last_state_new, ops.unsqueeze(C_t, const_minus_two))  # [B, H, P, N]
    y_t = ops.reduce_sum(y_t, const_minus_one, False)  # [B, H, P]
    y_t = ops.unsqueeze(y_t, const_two)  # [B, H, 1, P]

    timestep = ops.unsqueeze(timestep_param, const_zero_axis)
    core_out_res = ops.scatter_update(core_out_t, timestep, y_t, const_two)
    last_state_res = last_state_new

    body_cond = ops.constant([True], dtype=bool)
    body_model = ov.Model(
        [body_cond, last_state_res, core_out_res],
        [
            timestep_param,
            dA_t_param,
            dBx_t_param,
            C_t_param,
            last_state_t,
            core_out_t,
        ],
        "mamba2_body_model",
    )

    loop = ops.loop(seq_len, ops.constant(True, dtype="bool"))
    loop.set_function(body_model)

    loop.set_sliced_input(dA_t_param, dA, 0, 1, 1, -1, 2)
    loop.set_sliced_input(dBx_t_param, dBx, 0, 1, 1, -1, 2)
    loop.set_sliced_input(C_t_param, C, 0, 1, 1, -1, 2)
    loop.set_merged_input(last_state_t, last_state_old, last_state_res.output(0))
    loop.set_merged_input(core_out_t, core_out.output(0), core_out_res.output(0))
    loop.set_special_body_ports([0, 0])

    core_out_new = loop.get_iter_value(core_out_res.output(0), -1)
    last_state_new = loop.get_iter_value(last_state_res.output(0), -1)

    flatten_shape = ops.constant([-1], dtype=np.int32)
    core_out_new = ops.reshape(core_out_new, flatten_shape, False)
    last_state_new = ops.reshape(last_state_new, flatten_shape, False)

    final_output = ops.concat([core_out_new, last_state_new], 0)

    return [final_output.output(0)]