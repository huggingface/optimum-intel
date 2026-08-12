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


# Conversion rule for the `SelectiveSSMRecurrentCellOp` operation in a Torch graph.
#
# This generalizes the recurrent-cell-to-`ov::Loop` approach (originally introduced for the
# GatedDeltaNet block, see `convert_recurrent_attention_cell` above) to the Mamba2 selective
# state-space recurrence used by hybrid Mamba2 models such as NemotronH.
#
# The Mamba2 single-step recurrence follows the standard Mamba-2 discretization:
#       state_t = state_{t-1} * dA_t + dBx_t             # [B, H, P, N]
#       y_t     = reduce_sum(state_t * C_t, axis=N)       # [B, H, P]
#
# All the heavy discretization tensors are computed vectorized over the whole sequence BEFORE
# the loop, so the loop body stays as simple as possible. This mirrors the torch reference:
#       dA  = exp(A * dt).reshape(B, T, H, 1, 1)                 # [B, T, H, 1, 1]
#       dB  = dt.reshape(B, T, H, 1) * B                         # [B, T, H, N]
#       dBx = dB.reshape(B, T, H, 1, N) * x.reshape(B, T, H, P, 1)  # [B, T, H, P, N]
#       C   = C.reshape(B, T, H, 1, N)                          # [B, T, H, 1, N]
# The loop then simply slices dA/dBx/C along dim 1 (T) and runs the two-line recurrence.
# Inputs are in [B, T, H, ...] layout; the skip connection `x_t * D` does not depend on the
# recurrent state and is added outside.
#
# The `SelectiveSSMRecurrentCellOp` appears in the Torch graph as a result of replacing the
# `SelectiveSSMRecurrentCell` `torch.nn.Module` via a registered `ModuleExtension` in the OpenVINO
# PyTorch frontend; OpenVINO then applies this conversion rule to the resulting operation.
def convert_recurrent_selective_ssm_cell(context):
    # Inputs match the forward signature of `SelectiveSSMRecurrentCell`:
    #   A          [H]          — negative log-decay rates
    #   dt         [B, T, H]   — time steps
    #   B          [B, T, G, N] — input matrix (G groups, expanded to H before the loop)
    #   x          [B, T, H, P] — input hidden states
    #   C          [B, T, G, N] — output matrix (G groups, expanded to H before the loop)
    #   recurrent_state [B, H, P, N] — initial recurrent state
    A = context.get_input(0)  # [H]
    dt = context.get_input(1)  # [B, T, H]
    B = context.get_input(2)  # [B, T, G, N]
    x = context.get_input(3)  # [B, T, H, P]
    C = context.get_input(4)  # [B, T, G, N]
    recurrent_state = context.get_input(5)  # [B, H, P, N]

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

    # Expand grouped B/C from [B, T, G, N] → [B, T, H, N] via repeat_interleave along dim 2:
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
    N = ops.gather(B_shape, ops.constant(3, dtype=np.int32), const_zero_axis)
    BC_shape = ops.concat([ops.constant([0, 0, -1], dtype=np.int64),
                           ops.unsqueeze(N, const_zero_axis)], 0)
    B = ops.reshape(B_tiled, BC_shape, True)  # [B, T, H, N]
    C = ops.reshape(C_tiled, BC_shape, True)  # [B, T, H, N]

    # Vectorized discretization over the whole sequence, before the loop (mirrors torch):
    #   dA  = exp(A * dt).reshape(B, T, H, 1, 1)
    #   dB  = dt.reshape(B, T, H, 1) * B
    #   dBx = dB.reshape(B, T, H, 1, N) * x.reshape(B, T, H, P, 1)  → [B, T, H, P, N]
    #   C   = C.reshape(B, T, H, 1, N)
    dA = ops.exp(ops.multiply(A, dt))  # [B, T, H]
    dA = ops.reshape(dA, ops.constant([0, 0, 0, 1, 1], dtype=np.int64), True)  # [B, T, H, 1, 1]
    dB = ops.multiply(ops.unsqueeze(dt, const_minus_one), B)  # [B, T, H, N]
    dBx = ops.multiply(
        ops.unsqueeze(dB, const_minus_two),  # [B, T, H, 1, N]
        ops.unsqueeze(x, const_minus_one),  # [B, T, H, P, 1]
    )  # [B, T, H, P, N]
    C = ops.unsqueeze(C, const_minus_two)  # [B, T, H, 1, N]

    # Build the zero-initialized output accumulator with shape [B, T, H, P].
    const_zero_f32 = ops.constant(0, dtype=np.float32)
    output = ops.broadcast(const_zero_f32, x_shape)

    # Trip count for the loop equals the sequence length (dim 1 of x).
    seq_len = ops.gather(x_shape, const_one, const_zero_axis)
    seq_len = ops.convert(seq_len, "i32")

    # Body parameters (one timestep slice each along dim 1). All tensors are already discretized.
    timestep_param = ops.parameter([], np.int32, "timestep")
    dA_t_param = ops.parameter([-1, 1, -1, 1, 1], np.float32, "dA_t")  # [B, 1, H, 1, 1]
    dBx_t_param = ops.parameter([-1, 1, -1, -1, -1], np.float32, "dBx_t")  # [B, 1, H, P, N]
    C_t_param = ops.parameter([-1, 1, -1, 1, -1], np.float32, "C_t")  # [B, 1, H, 1, N]
    recurrent_state_t = ops.parameter([-1, -1, -1, -1], np.float32, "recurrent_state_t")  # [B, H, P, N]
    output_t = ops.parameter([-1, -1, -1, -1], np.float32, "output_t")  # [B, T, H, P]

    # Drop the singleton sequence dimension introduced by slicing.
    dA_t = ops.squeeze(dA_t_param, const_one)  # [B, H, 1, 1]
    dBx_t = ops.squeeze(dBx_t_param, const_one)  # [B, H, P, N]
    C_t = ops.squeeze(C_t_param, const_one)  # [B, H, 1, N]

    # output_recurrent_state = output_recurrent_state * dA[:, t] + dBx[:, t]
    recurrent_state_new = ops.add(ops.multiply(recurrent_state_t, dA_t), dBx_t)  # [B, H, P, N]

    # output[:, t] = (output_recurrent_state * C[:, t]).sum(dim=-1)
    y_t = ops.reduce_sum(ops.multiply(recurrent_state_new, C_t), const_minus_one, False)  # [B, H, P]
    y_t = ops.unsqueeze(y_t, const_one)  # [B, 1, H, P]

    timestep = ops.unsqueeze(timestep_param, const_zero_axis)
    output_res = ops.scatter_update(output_t, timestep, y_t, const_one)
    recurrent_state_res = recurrent_state_new

    body_cond = ops.constant([True], dtype=bool)
    body_model = ov.Model(
        [body_cond, recurrent_state_res, output_res],
        [
            timestep_param,
            dA_t_param,
            dBx_t_param,
            C_t_param,
            recurrent_state_t,
            output_t,
        ],
        "selective_ssm_body_model",
    )

    loop = ops.loop(seq_len, ops.constant(True, dtype="bool"))
    loop.set_function(body_model)

    loop.set_sliced_input(dA_t_param, dA.output(0), 0, 1, 1, -1, 1)
    loop.set_sliced_input(dBx_t_param, dBx.output(0), 0, 1, 1, -1, 1)
    loop.set_sliced_input(C_t_param, C.output(0), 0, 1, 1, -1, 1)
    loop.set_merged_input(recurrent_state_t, recurrent_state, recurrent_state_res.output(0))
    loop.set_merged_input(output_t, output.output(0), output_res.output(0))
    loop.set_special_body_ports([0, 0])

    output_new = loop.get_iter_value(output_res.output(0), -1)
    output_recurrent_state_new = loop.get_iter_value(recurrent_state_res.output(0), -1)

    flatten_shape = ops.constant([-1], dtype=np.int32)
    output_new = ops.reshape(output_new, flatten_shape, False)
    output_recurrent_state_new = ops.reshape(output_recurrent_state_new, flatten_shape, False)

    final_output = ops.concat([output_new, output_recurrent_state_new], 0)

    return [final_output.output(0)]
