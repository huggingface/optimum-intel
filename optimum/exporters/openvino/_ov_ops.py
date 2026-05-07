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

    # Concatenate along axis 2 (seq/key-head-dim axis) instead of flattening to 1D.
    # core_attn_out_new: (B, H, T, D2), last_recurrent_state_new: (B, H, D1, D2)
    # Combined output: (B, H, T+D1, D2)
    #
    # This avoids flattening+reshape which would require a reshape with two dynamic
    # dimensions (batch and seq) — invalid in OpenVINO (only one -1 allowed per reshape).
    # The downstream code splits using static negative indexing with the known D1=head_k_dim.
    final_output = ops.concat([core_attn_out_new, last_recurrent_state_new], 2)

    return [final_output.output(0)]
