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


# Conversion rule for the `SSMRecurrentCellOp` operation in a Torch graph.
#
# This generalizes the recurrent-cell-to-`ov::Loop` approach (originally introduced for the
# GatedDeltaNet block, see `convert_recurrent_attention_cell` above) to the Mamba2 selective
# state-space recurrence used by hybrid Mamba2 models such as NemotronH.
#
# The Mamba2 single-step recurrence follows the standard Mamba-2 discretization:
#       dA_t    = exp(A * dt_t)                           # [B, H] (broadcast)
#       dBx_t   = dt_t * B_t outer x_t                   # [B, H, P, N]
#       state_t = state_{t-1} * dA_t + dBx_t             # [B, H, P, N]
#       y_t     = reduce_sum(state_t * C_t, axis=N)       # [B, H, P]
#
# The raw parameters A (log-decay), dt (time steps), B, x, C are passed directly into the
# loop; discretization (exp, outer product) happens per timestep inside the body.
# Inputs are in [B, T, H, ...] layout; the loop slices along dim 1 (T).
# The skip connection `x_t * D` does not depend on the recurrent state and is added outside.
#
# The `SSMRecurrentCellOp` appears in the Torch graph as a result of replacing the
# `SSMRecurrentCell` `torch.nn.Module` via a registered `ModuleExtension` in the OpenVINO
# PyTorch frontend; OpenVINO then applies this conversion rule to the resulting operation.
def convert_recurrent_ssm_cell(context):
    # Inputs match the forward signature of `SSMRecurrentCell`:
    #   A          [H]          — negative log-decay rates
    #   dt         [B, T, H]   — time steps
    #   B          [B, T, G, N] — input matrix (G groups, expanded to H inside the loop)
    #   x          [B, T, H, P] — input hidden states
    #   C          [B, T, G, N] — output matrix (G groups, expanded to H inside the loop)
    #   last_state [B, H, P, N] — initial recurrent state
    A = context.get_input(0)  # [H]
    dt = context.get_input(1)  # [B, T, H]
    B = context.get_input(2)  # [B, T, G, N]
    x = context.get_input(3)  # [B, T, H, P]
    C = context.get_input(4)  # [B, T, G, N]
    last_state_old = context.get_input(5)  # [B, H, P, N]

    const_zero_axis = ops.constant(0, dtype=np.int32)
    const_one = ops.constant(1, dtype=np.int32)
    const_minus_one = ops.constant(-1, dtype=np.int32)
    const_minus_two = ops.constant(-2, dtype=np.int32)

    # Compute heads_per_group = H / G from the shapes of dt and B.
    dt_shape = ops.shape_of(dt)
    B_shape = ops.shape_of(B)
    const_two = ops.constant(2, dtype=np.int32)
    num_heads = ops.gather(dt_shape, const_two, const_zero_axis)  # H
    num_groups = ops.gather(B_shape, const_two, const_zero_axis)  # G
    heads_per_group = ops.convert(ops.divide(num_heads, num_groups), "i64")

    # Expand B/C from [B, T, G, N] → [B, T, H, N] by repeating each group.
    # reshape to [B, T, G, 1, N] → tile [1, 1, 1, heads_per_group, 1] → reshape [B, T, H, N]
    B_5d = ops.unsqueeze(B, ops.constant(3, dtype=np.int32))  # [B, T, G, 1, N]
    C_5d = ops.unsqueeze(C, ops.constant(3, dtype=np.int32))  # [B, T, G, 1, N]
    tile_shape = ops.concat(
        [ops.constant([1, 1, 1], dtype=np.int64), ops.unsqueeze(heads_per_group, const_zero_axis),
         ops.constant([1], dtype=np.int64)], 0
    )
    B_tiled = ops.tile(B_5d, tile_shape)  # [B, T, G, H/G, N]
    C_tiled = ops.tile(C_5d, tile_shape)  # [B, T, G, H/G, N]

    # Reshape [B, T, G, H/G, N] → [B, T, H, N]
    x_shape = ops.shape_of(x)
    target_4d = ops.gather(x_shape, ops.constant([0, 1], dtype=np.int32), const_zero_axis)
    N_dim = ops.gather(B_shape, ops.constant(3, dtype=np.int32), const_zero_axis)
    BC_shape = ops.concat([target_4d, ops.unsqueeze(num_heads, const_zero_axis),
                           ops.unsqueeze(N_dim, const_zero_axis)], 0)
    B_expanded = ops.reshape(B_tiled, BC_shape, False)  # [B, T, H, N]
    C_expanded = ops.reshape(C_tiled, BC_shape, False)  # [B, T, H, N]

    # Build the zero-initialized output accumulator with shape [B, T, H, P].
    core_shape = ops.gather(x_shape, ops.constant([0, 1, 2, 3], dtype=np.int32), const_zero_axis)
    const_zero_f32 = ops.constant(0, dtype=np.float32)
    core_out = ops.broadcast(const_zero_f32, core_shape)

    # Trip count for the loop equals the sequence length (dim 1 of x).
    seq_len = ops.gather(x_shape, const_one, const_zero_axis)
    seq_len = ops.convert(seq_len, "i32")

    # Body parameters (one timestep slice each along dim 1).
    timestep_param = ops.parameter([], np.int32, "timestep")
    dt_t_param = ops.parameter([-1, 1, -1], np.float32, "dt_t")  # [B, 1, H]
    B_t_param = ops.parameter([-1, 1, -1, -1], np.float32, "B_t")  # [B, 1, H, N]
    x_t_param = ops.parameter([-1, 1, -1, -1], np.float32, "x_t")  # [B, 1, H, P]
    C_t_param = ops.parameter([-1, 1, -1, -1], np.float32, "C_t")  # [B, 1, H, N]
    last_state_t = ops.parameter([-1, -1, -1, -1], np.float32, "last_state_t")  # [B, H, P, N]
    core_out_t = ops.parameter([-1, -1, -1, -1], np.float32, "core_out_t")  # [B, T, H, P]

    # Drop the singleton sequence dimension introduced by slicing.
    dt_t = ops.squeeze(dt_t_param, const_one)  # [B, H]
    B_t = ops.squeeze(B_t_param, const_one)  # [B, H, N]
    x_t = ops.squeeze(x_t_param, const_one)  # [B, H, P]
    C_t = ops.squeeze(C_t_param, const_one)  # [B, H, N]

    # Discretization inside the loop body:
    # dA_t = exp(A * dt_t)  — A is [H], dt_t is [B, H] → broadcast to [B, H]
    A_unsqueeze = ops.unsqueeze(A, const_zero_axis)  # [1, H]
    dA_t = ops.exp(ops.multiply(A_unsqueeze, dt_t))  # [B, H]
    dA_t_4d = ops.unsqueeze(ops.unsqueeze(dA_t, const_minus_one), const_minus_one)  # [B, H, 1, 1]

    # dBx_t = (dt_t * B_t)[:,None,:] * x_t[:,:,None]  → [B, H, P, N]
    dt_B_t = ops.multiply(ops.unsqueeze(dt_t, const_minus_one), B_t)  # [B, H, N]
    dBx_t = ops.multiply(
        ops.unsqueeze(dt_B_t, const_minus_two),  # [B, H, 1, N]
        ops.unsqueeze(x_t, const_minus_one),  # [B, H, P, 1]
    )  # [B, H, P, N]

    # state_t = state_{t-1} * dA_t + dBx_t
    last_state_new = ops.add(ops.multiply(last_state_t, dA_t_4d), dBx_t)  # [B, H, P, N]

    # y_t = reduce_sum(state_t * C_t, axis=N) → [B, H, P]
    y_t = ops.multiply(last_state_new, ops.unsqueeze(C_t, const_minus_two))  # [B, H, P, N]
    y_t = ops.reduce_sum(y_t, const_minus_one, False)  # [B, H, P]
    y_t = ops.unsqueeze(y_t, const_one)  # [B, 1, H, P]

    timestep = ops.unsqueeze(timestep_param, const_zero_axis)
    core_out_res = ops.scatter_update(core_out_t, timestep, y_t, const_one)
    last_state_res = last_state_new

    body_cond = ops.constant([True], dtype=bool)
    body_model = ov.Model(
        [body_cond, last_state_res, core_out_res],
        [
            timestep_param,
            dt_t_param,
            B_t_param,
            x_t_param,
            C_t_param,
            last_state_t,
            core_out_t,
        ],
        "ssm_body_model",
    )

    loop = ops.loop(seq_len, ops.constant(True, dtype="bool"))
    loop.set_function(body_model)

    loop.set_sliced_input(dt_t_param, dt, 0, 1, 1, -1, 1)
    loop.set_sliced_input(B_t_param, B_expanded.output(0), 0, 1, 1, -1, 1)
    loop.set_sliced_input(x_t_param, x, 0, 1, 1, -1, 1)
    loop.set_sliced_input(C_t_param, C_expanded.output(0), 0, 1, 1, -1, 1)
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
