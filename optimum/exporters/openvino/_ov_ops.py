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
import functools
import itertools
from typing import Sequence, Tuple

import numpy as np
import torch

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
# state-space recurrence used by hybrid Mamba2 models such as GraniteMoeHybrid (Granite-4.0).
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
        [
            ops.constant([1, 1, 1], dtype=np.int64),
            ops.unsqueeze(heads_per_group, const_zero_axis),
            ops.constant([1], dtype=np.int64),
        ],
        0,
    )
    B_tiled = ops.tile(B_5d, tile_shape)  # [B, T, G, H/G, N]
    C_tiled = ops.tile(C_5d, tile_shape)  # [B, T, G, H/G, N]

    # Reshape [B, T, G, H/G, N] → [B, T, H, N]
    x_shape = ops.shape_of(x)
    N = ops.gather(B_shape, ops.constant(3, dtype=np.int32), const_zero_axis)
    BC_shape = ops.concat([ops.constant([0, 0, -1], dtype=np.int64), ops.unsqueeze(N, const_zero_axis)], 0)
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


# ----------------------------------------------------------------------------------------------------------------------
# Qwen Sparse Attention (QSA) and n-gram hashing operations.
#
# The torch.nn.Module classes below are analogues of the OpenVINO operations QSAPrepare-17, SparseAttentionIndexer-17 and
# SparseAttention-17 (docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/sequence in the
# OpenVINO repository), and of the n-gram hash NGramHashFunction, which has no OpenVINO operation. Each module:
#   * keeps the operation attributes as Python attributes,
#   * implements the reference semantics of the operation in `forward` (used while the model is traced),
#   * provides `openvino_extensions()` that returns the `ModuleExtension` replacing the module by a single operation
#     in the traced graph and the `ConversionExtension` that converts this operation with the attributes of the
#     module instance: into the public operation of `openvino.opset17` if the installed OpenVINO has it, otherwise
#     into its decomposition into regular operations.
#
# The OpenVINO PyTorch frontend supports a single output per ModuleExtension, so the operations with several outputs
# (QSAPrepare, SparseAttentionIndexer) return them packed into one tensor; `unpack()` restores the outputs of the spec
# through `PackedOutput` modules, whose conversion returns the outputs of the operation itself: the packing does not
# appear in the converted graph, and the public operations stay directly connected. NGramHashFunction is always
# decomposed and unpacked by slicing.
# ----------------------------------------------------------------------------------------------------------------------


_I64_MAX = np.iinfo(np.int64).max


def _const_i64(value):
    return ops.constant(np.array(value, dtype=np.int64))


def _dim(shape, axis: int):
    # scalar i64 dimension `axis` of the 1D shape tensor `shape`
    return ops.gather(shape, _const_i64(axis), _const_i64(0))


def _vec(*values):
    # 1D i64 tensor from scalar nodes / python ints
    return ops.concat(
        [ops.unsqueeze(v if isinstance(v, ov.Node) else _const_i64(v), _const_i64(0)) for v in values], 0
    )


def _public_op(name: str):
    """`openvino.opset17.<name>` if the installed OpenVINO has this public operation, None otherwise."""
    try:
        import openvino.opset17 as opset17
    except ImportError:
        return None
    return getattr(opset17, name, None)


# outputs of the operations packed by `_pack_outputs`, by the id kept in the rt_info of the packing Concat
_PACKED_OUTPUTS = {}
_PACKED_OUTPUTS_KEY = "optimum_packed_outputs"
_packed_outputs_ids = itertools.count()


def _pack_outputs(outputs, like):
    """
    Packs the outputs of an operation into one 1D tensor of the type of `like`, the single output of a ModuleExtension,
    and remembers them: the conversion of `PackedOutput` returns them directly, so the packing Concat is left without
    consumers and is not a part of the converted model.
    """
    flat = _const_i64([-1])
    pack = ops.concat([ops.reshape(ops.convert_like(output, like), flat, False) for output in outputs], 0)
    key = next(_packed_outputs_ids)
    pack.get_rt_info()[_PACKED_OUTPUTS_KEY] = key
    _PACKED_OUTPUTS[key] = [output.output(0) if isinstance(output, ov.Node) else output for output in outputs]
    return pack.output(0)


class _OVOperationModule(torch.nn.Module):
    """Base class of the modules that are replaced by a single OpenVINO operation at conversion."""

    _OP_TYPE = None
    _instance_counter = itertools.count()

    def __init__(self):
        super().__init__()
        self._op_name = f"{self._OP_TYPE}Op{next(_OVOperationModule._instance_counter)}"

    def _conversion_attributes(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def _convert(context, **attributes):
        raise NotImplementedError

    def openvino_extensions(self) -> Tuple["ModuleExtension", "ConversionExtension"]:  # noqa: F821
        from openvino.frontend.pytorch import ConversionExtension, ModuleExtension

        convert_fn = functools.partial(self._convert, **self._conversion_attributes())
        return ModuleExtension(self, self._op_name), ConversionExtension(self._op_name, convert_fn)


class PackedOutput(_OVOperationModule):
    """
    Output `index` of an operation module whose outputs are packed into one tensor (see `_pack_outputs`). While traced,
    it is the slice of the packed tensor given by `references`, tensors with the shapes of all the packed outputs in
    the order of the packing (only tensors: ints derived from traced shapes cannot be passed to a ModuleExtension);
    its conversion returns the output of the operation itself, so the references end up unused.
    """

    _OP_TYPE = "PackedOutput"

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, packed, *references):
        start = sum(reference.numel() for reference in references[: self.index])
        reference = references[self.index]
        return packed[start : start + reference.numel()].reshape(reference.shape)

    def _conversion_attributes(self):
        return {"index": self.index}

    @staticmethod
    def _convert(context, index):
        pack = context.get_input(0).get_node()
        key = pack.get_rt_info()[_PACKED_OUTPUTS_KEY].astype(int)
        return [_PACKED_OUTPUTS[key][index]]


def _unpack_outputs(selectors, packed, references):
    """The outputs packed in this order into `packed`, shaped like the tensors `references`."""
    return [selector(packed, *references) for selector in selectors]


def _rotate_half_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    x_rope, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rope[..., : rotary_dim // 2], x_rope[..., rotary_dim // 2 :]
    x_rope = x_rope * cos + torch.cat([-x2, x1], dim=-1) * sin
    return torch.cat([x_rope, x_pass], dim=-1)


def _sequence_key_length(key_length, batch_size):
    # key_length [N] or [1] -> i64 [N]: the number of valid (non-padding) key tokens of every sequence
    key_length = ops.reshape(ops.convert(key_length, "i64"), _const_i64([-1]), False)
    return ops.broadcast(key_length, ops.unsqueeze(batch_size, _const_i64(0)))


class QSAPrepare(_OVOperationModule):
    """
    Analogue of the QSAPrepare operation: one block index key per complete block of `key_block_size` raw index keys,
    obtained by mean-pooling (f32), RMS normalization with `k_norm_weight` and rotate-half RoPE at the first token
    of the block. It takes the raw index keys and the RoPE of the new tokens only: the keys of the blocks computed by
    the previous calls come from `past_block_index_keys`, and the raw keys and the RoPE row of the past tokens of the
    incomplete block from `past_incomplete_raw_keys` / `past_incomplete_rope`; only the blocks completed by the new
    tokens are computed.

    Batches with padding: sequence `n` has `key_length[n]` valid key tokens, past and current, the last ones of its
    positions (left padding); `max(0, key_length[n] - L)` of them come from the previous calls. Its blocks start at its
    first valid token; the rows after its last complete block are zero.

    Inputs: raw_index_keys [N, L, Di], past_block_index_keys [N, Bp, Di], key_length [N], rope_cos [N, L, Dr],
            rope_sin [N, L, Dr], k_norm_weight [Di], past_incomplete_raw_keys [N, key_block_size - 1, Di],
            past_incomplete_rope [N, 2 * Dr].
    Outputs: block_index_keys [N, B, Di] (B = max(key_length // key_block_size)), incomplete_raw_keys
             [N, key_block_size - 1, Di], incomplete_rope [N, 2 * Dr], returned packed, see `unpack()`.
    An empty `k_norm_weight` disables the normalization, empty `rope_cos` / `rope_sin` / `past_incomplete_rope`
    disable the rotation.
    """

    _OP_TYPE = "QSAPrepare"

    def __init__(self, key_block_size: int, epsilon: float = 1e-6):
        super().__init__()
        if key_block_size < 2:
            raise ValueError("QSAPrepare needs key_block_size >= 2: with single-token blocks no block key is pooled")
        self.key_block_size = key_block_size
        self.epsilon = epsilon
        # packed order: incomplete_raw_keys, incomplete_rope, block_index_keys
        self.packed_outputs = torch.nn.ModuleList([PackedOutput(index) for index in range(3)])

    def unpack(self, packed: torch.Tensor, raw_index_keys: torch.Tensor, rope_cos: torch.Tensor):
        batch_size, _, head_dim = raw_index_keys.shape
        rope_size = 2 * rope_cos.shape[-1] if rope_cos.numel() > 0 else 0
        num_keys, num_rope = batch_size * (self.key_block_size - 1) * head_dim, batch_size * rope_size
        num_blocks = (packed.shape[0] - num_keys - num_rope) // (batch_size * head_dim)
        references = [
            raw_index_keys.new_zeros(batch_size, self.key_block_size - 1, head_dim),
            raw_index_keys.new_zeros(batch_size, rope_size),
            raw_index_keys.new_zeros(batch_size, num_blocks, head_dim),
        ]
        incomplete_raw_keys, incomplete_rope, block_index_keys = _unpack_outputs(
            self.packed_outputs, packed, references
        )
        return block_index_keys, incomplete_raw_keys, incomplete_rope

    def forward(
        self,
        raw_index_keys,
        past_block_index_keys,
        key_length,
        rope_cos,
        rope_sin,
        k_norm_weight,
        past_incomplete_raw_keys,
        past_incomplete_rope,
    ):
        batch_size, seq_len, head_dim = raw_index_keys.shape
        r = self.key_block_size
        key_length = key_length.reshape(-1).long().expand(batch_size)
        past = (key_length - seq_len).clamp(min=0)  # valid tokens of the previous calls
        current = key_length - past  # valid new tokens, the last rows
        num_cached = torch.div(past, r, rounding_mode="floor")
        pending_past = past - num_cached * r  # past tokens of the incomplete block
        pending = pending_past + current  # tokens not yet in a block key
        num_new = torch.div(pending, r, rounding_mode="floor")
        rest = pending - num_new * r
        num_blocks = int(torch.div(key_length, r, rounding_mode="floor").max())
        num_tokens = r - 1 + seq_len
        num_steps = -(-num_tokens // r)
        offset = (r - 1 + seq_len - current)[:, None]

        # the pending tokens: the incomplete block of the previous calls, then the valid new tokens
        t = torch.arange(num_steps * r)[None]
        src = torch.where(t < pending_past[:, None], t, offset + t - pending_past[:, None]).clamp(0, num_tokens - 1)
        tokens = torch.cat([past_incomplete_raw_keys.to(raw_index_keys.dtype), raw_index_keys], dim=1)
        gathered = torch.gather(tokens, 1, src[..., None].expand(-1, -1, head_dim))  # [N, J * r, Di]
        keys = gathered.reshape(batch_size, num_steps, r, head_dim).float().mean(dim=2)
        if k_norm_weight.numel() > 0:
            keys = keys * torch.rsqrt(keys.pow(2).mean(-1, keepdim=True) + self.epsilon) * k_norm_weight.float()

        use_rope = rope_cos.numel() > 0
        if use_rope:
            rotary_dim = rope_cos.shape[-1]
            cos_table = torch.cat([past_incomplete_rope[:, None, :rotary_dim], rope_cos.expand(batch_size, -1, -1)], 1)
            sin_table = torch.cat([past_incomplete_rope[:, None, rotary_dim:], rope_sin.expand(batch_size, -1, -1)], 1)

            def rope_rows(token):  # pending token index [N, K] -> row of the [N, 1 + L, Dr] tables
                rows = torch.where(
                    token < pending_past[:, None], 0, 1 + offset - (r - 1) + token - pending_past[:, None]
                )
                rows = rows.clamp(0, seq_len)[..., None].expand(-1, -1, rotary_dim)
                return torch.gather(cos_table.float(), 1, rows), torch.gather(sin_table.float(), 1, rows)

            cos, sin = rope_rows(torch.arange(num_steps)[None] * r)
            keys = _rotate_half_rope(keys, cos, sin)

        # the cached blocks of every sequence, then the computed ones; one extra column absorbs invalid blocks
        # (Bp <= B for the states of the previous call; the cut keeps inconsistent dummy inputs traceable)
        filler = max(num_blocks + 1 - past_block_index_keys.shape[1], 1)
        output = torch.cat([past_block_index_keys, past_block_index_keys.new_zeros(batch_size, filler, head_dim)], 1)
        output = output[:, : num_blocks + 1]
        output = output * (torch.arange(num_blocks + 1) < num_cached[:, None])[..., None]
        steps = torch.arange(num_steps)[None]
        targets = torch.where(steps < num_new[:, None], num_cached[:, None] + steps, num_blocks)
        output = output.scatter(1, targets[..., None].expand(-1, -1, head_dim), keys.to(output.dtype))[:, :num_blocks]

        # the incomplete block after this call
        slots = torch.arange(r - 1)[None]
        valid = slots < rest[:, None]
        idx = (num_new[:, None] * r + slots).clamp(0, num_steps * r - 1)
        incomplete_raw_keys = torch.gather(gathered, 1, idx[..., None].expand(-1, -1, head_dim))
        incomplete_raw_keys = incomplete_raw_keys * valid[..., None]
        if use_rope:
            cos, sin = rope_rows(num_new[:, None] * r)
            incomplete_rope = torch.cat([cos, sin], -1)[:, 0] * (rest > 0)[:, None]
        else:
            incomplete_rope = raw_index_keys.new_zeros(batch_size, 0)
        dtype = raw_index_keys.dtype
        return torch.cat(
            [incomplete_raw_keys.flatten().to(dtype), incomplete_rope.flatten().to(dtype), output.flatten().to(dtype)]
        )

    def _conversion_attributes(self):
        return {"key_block_size": self.key_block_size, "epsilon": self.epsilon}

    @staticmethod
    def _convert(context, key_block_size, epsilon):
        raw_index_keys = context.get_input(0)
        qsa_prepare = _public_op("qsa_prepare")
        if qsa_prepare is not None:
            node = qsa_prepare(
                *[context.get_input(i) for i in range(8)], key_block_size=key_block_size, epsilon=epsilon
            )
            return [_pack_outputs([node.output(1), node.output(2), node.output(0)], raw_index_keys)]
        return [_pack_outputs(QSAPrepare._decompose(context, key_block_size, epsilon), raw_index_keys)]

    @staticmethod
    def _decompose(context, key_block_size, epsilon):
        raw_index_keys = context.get_input(0)
        past_block_index_keys = context.get_input(1)
        rope_cos = context.get_input(3)
        rope_sin = context.get_input(4)
        k_norm_weight = context.get_input(5)
        past_incomplete_raw_keys = context.get_input(6)
        past_incomplete_rope = context.get_input(7)

        r = key_block_size
        ratio = _const_i64(r)
        zero, one, last_axis, slice_axis = _const_i64(0), _const_i64(1), _const_i64(-1), _const_i64([-1])
        raw_shape = ops.shape_of(raw_index_keys, "i64")
        batch_size, seq_len, head_dim = _dim(raw_shape, 0), _dim(raw_shape, 1), _dim(raw_shape, 2)
        key_length = _sequence_key_length(context.get_input(2), batch_size)
        num_past_blocks = _dim(ops.shape_of(past_block_index_keys, "i64"), 1)

        # all values are below 2^24 and non-negative, so the integer divisions are exact on every device
        past = ops.maximum(ops.subtract(key_length, seq_len), zero)  # [N] valid tokens of the previous calls
        current = ops.subtract(key_length, past)  # [N] valid new tokens, the last rows
        num_cached = ops.divide(past, ratio)
        pending_past = ops.subtract(past, ops.multiply(num_cached, ratio))  # past tokens of the incomplete block
        pending = ops.add(pending_past, current)
        num_new = ops.divide(pending, ratio)
        rest = ops.subtract(pending, ops.multiply(num_new, ratio))
        num_blocks = ops.reduce_max(ops.divide(key_length, ratio), zero, False)  # B
        num_tokens = ops.add(seq_len, _const_i64(r - 1))  # M
        num_steps = ops.divide(ops.add(num_tokens, _const_i64(r - 1)), ratio)  # J >= 1
        offset = ops.unsqueeze(ops.subtract(num_tokens, current), one)  # [N, 1]: row of the first valid new token
        pending_past_col = ops.unsqueeze(pending_past, one)

        def source_rows(token, base, last):
            # pending token index -> row of a table holding the past entries at `0 ..` and the new ones from `base`
            rows = ops.select(
                ops.less(token, pending_past_col), token, ops.add(ops.subtract(token, pending_past_col), base)
            )
            return ops.minimum(ops.maximum(rows, zero), last)

        # 1. gather the pending tokens: the incomplete block of the previous calls, then the valid new tokens
        tokens = ops.concat([ops.convert_like(past_incomplete_raw_keys, raw_index_keys), raw_index_keys], 1)
        t = ops.unsqueeze(ops.range(zero, ops.multiply(num_steps, ratio), one, "i64"), zero)  # [1, J * r]
        src = source_rows(t, offset, ops.subtract(num_tokens, one))
        gathered = ops.gather(tokens, src, one, 1)  # [N, J * r, Di]

        # 2. mean-pool and RMS-normalize every block that can be completed
        keys = ops.reshape(gathered, _vec(batch_size, num_steps, r, head_dim), False)
        keys = ops.reduce_mean(ops.convert(keys, "f32"), _const_i64(2), False)  # [N, J, Di]
        if k_norm_weight.get_partial_shape().is_dynamic or k_norm_weight.get_partial_shape().to_shape()[0] > 0:
            mean_square = ops.reduce_mean(ops.multiply(keys, keys), last_axis, True)
            inv_rms = ops.divide(
                ops.constant(np.float32(1.0)), ops.sqrt(ops.add(mean_square, ops.constant(np.float32(epsilon))))
            )
            keys = ops.multiply(ops.multiply(keys, inv_rms), ops.convert(k_norm_weight, "f32"))

        # 3. rotate-half RoPE at the first token of every block: a past one from past_incomplete_rope
        use_rope = rope_cos.get_partial_shape().is_dynamic or 0 not in rope_cos.get_partial_shape().to_shape()
        if use_rope:
            rotary_dim = _dim(ops.shape_of(rope_cos, "i64"), 2)
            rotary_dim_1d = ops.unsqueeze(rotary_dim, zero)
            past_rope = ops.unsqueeze(ops.convert(past_incomplete_rope, "f32"), one)  # [N, 1, 2 * Dr]
            past_cos = ops.slice(past_rope, _const_i64([0]), rotary_dim_1d, _const_i64([1]), slice_axis)
            past_sin = ops.slice(past_rope, rotary_dim_1d, _const_i64([_I64_MAX]), _const_i64([1]), slice_axis)

            def table(past_rows, current_rows):  # [N, 1 + L, Dr]
                current_rows = ops.broadcast(ops.convert(current_rows, "f32"), _vec(batch_size, seq_len, rotary_dim))
                return ops.concat([past_rows, current_rows], 1)

            cos_table, sin_table = table(past_cos, rope_cos), table(past_sin, rope_sin)
            rope_offset = ops.subtract(offset, _const_i64(r - 2))  # 1 + row of the first valid new token

            def rope_at(token):
                rows = source_rows(token, rope_offset, seq_len)
                return ops.gather(cos_table, rows, one, 1), ops.gather(sin_table, rows, one, 1)

            first = ops.unsqueeze(ops.multiply(ops.range(zero, num_steps, one, "i64"), ratio), zero)  # [1, J]
            cos, sin = rope_at(first)
            half_1d = ops.unsqueeze(ops.divide(rotary_dim, _const_i64(2)), zero)
            x_rope = ops.slice(keys, _const_i64([0]), rotary_dim_1d, _const_i64([1]), slice_axis)
            x_pass = ops.slice(keys, rotary_dim_1d, _const_i64([_I64_MAX]), _const_i64([1]), slice_axis)
            x1 = ops.slice(x_rope, _const_i64([0]), half_1d, _const_i64([1]), slice_axis)
            x2 = ops.slice(x_rope, half_1d, _const_i64([_I64_MAX]), _const_i64([1]), slice_axis)
            rotated = ops.concat([ops.negative(x2), x1], -1)
            keys = ops.concat([ops.add(ops.multiply(x_rope, cos), ops.multiply(rotated, sin)), x_pass], -1)

        # 4. the cached blocks of every sequence, then the computed ones; the extra column B absorbs the invalid ones
        # (Bp <= B for the states of the previous call; the cut keeps any input consistent)
        zero_value = ops.convert_like(ops.constant(np.float32(0.0)), past_block_index_keys)
        filler = ops.maximum(ops.subtract(ops.add(num_blocks, one), num_past_blocks), one)
        output = ops.concat([past_block_index_keys, ops.broadcast(zero_value, _vec(batch_size, filler, head_dim))], 1)
        output = ops.slice(
            output, _const_i64([0]), ops.unsqueeze(ops.add(num_blocks, one), zero), _const_i64([1]), _const_i64([1])
        )
        columns = ops.unsqueeze(ops.range(zero, ops.add(num_blocks, one), one, "i64"), zero)
        cached = ops.less(columns, ops.unsqueeze(num_cached, one))  # [N, B + 1]
        output = ops.select(ops.unsqueeze(cached, last_axis), output, zero_value)
        steps = ops.unsqueeze(ops.range(zero, num_steps, one, "i64"), zero)  # [1, J]
        targets = ops.select(
            ops.less(steps, ops.unsqueeze(num_new, one)), ops.add(ops.unsqueeze(num_cached, one), steps), num_blocks
        )
        targets = ops.broadcast(ops.unsqueeze(targets, last_axis), _vec(batch_size, num_steps, head_dim))
        output = ops.scatter_elements_update(output, targets, ops.convert_like(keys, output), 1)
        output = ops.slice(output, _const_i64([0]), ops.unsqueeze(num_blocks, zero), _const_i64([1]), _const_i64([1]))

        # 5. the incomplete block after this call
        slots = ops.unsqueeze(ops.range(zero, _const_i64(r - 1), one, "i64"), zero)  # [1, r - 1]
        valid = ops.less(slots, ops.unsqueeze(rest, one))  # [N, r - 1]
        idx = ops.add(ops.unsqueeze(ops.multiply(num_new, ratio), one), slots)
        idx = ops.minimum(idx, ops.subtract(ops.multiply(num_steps, ratio), one))
        incomplete_raw_keys = ops.gather(gathered, idx, one, 1)  # [N, r - 1, Di]
        incomplete_raw_keys = ops.select(
            ops.unsqueeze(valid, last_axis), incomplete_raw_keys, ops.convert_like(zero_value, raw_index_keys)
        )
        if use_rope:
            cos, sin = rope_at(ops.unsqueeze(ops.multiply(num_new, ratio), one))  # [N, 1, Dr]
            incomplete_rope = ops.squeeze(ops.concat([cos, sin], -1), one)  # [N, 2 * Dr]
            has_rest = ops.unsqueeze(ops.greater(rest, zero), one)
            incomplete_rope = ops.select(has_rest, incomplete_rope, ops.constant(np.float32(0.0)))
        else:
            incomplete_rope = ops.broadcast(ops.constant(np.float32(0.0)), _vec(batch_size, 0))
        return [incomplete_raw_keys, incomplete_rope, output]


class SparseAttentionIndexer(_OVOperationModule):
    """
    Analogue of the SparseAttentionIndexer operation: for every query token at the position `p = key_length[n] - L + l` of its
    sequence scores the `(p + 1) // key_block_size` complete key blocks of the sequence with
    `sum_h w[h] * ReLU(q[h] . k)` over the index query heads of every selection head and selects the
    `token_budget // key_block_size` best ones (stable top-k: equal scores -> smaller block index first). A padding
    query (`p < 0`) selects no blocks.

    Inputs: index_query [N, Hi, L, Di], block_index_keys [N, B, Di], key_length [N] (or [1]),
            index_weights [N, L, Hi] (optional).
    Outputs: block_indices [N, Hs, L, KB] of int32 (padded with -1) and block_count [N, Hs, L] of int32, returned
             packed, see `unpack()`.
    """

    _OP_TYPE = "SparseAttentionIndexer"

    def __init__(
        self,
        key_block_size: int,
        token_budget: int,
        num_selection_heads: int = 1,
    ):
        super().__init__()
        self.key_block_size = key_block_size
        self.token_budget = token_budget
        self.num_selection_heads = num_selection_heads
        # packed order: block_indices, block_count
        self.packed_outputs = torch.nn.ModuleList([PackedOutput(index) for index in range(2)])

    @property
    def block_topk(self):
        return self.token_budget // self.key_block_size

    def unpack(self, packed: torch.Tensor, index_query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, query_length, _ = index_query.shape
        shape = (batch_size, self.num_selection_heads, query_length)
        references = [packed.new_zeros(*shape, self.block_topk), packed.new_zeros(*shape)]
        block_indices, block_count = _unpack_outputs(self.packed_outputs, packed, references)
        return block_indices, block_count

    def forward(self, index_query, block_index_keys, key_length, index_weights=None):
        batch_size, num_index_heads, query_length, _ = index_query.shape
        num_blocks = block_index_keys.shape[1]
        key_length = torch.as_tensor(key_length).reshape(-1).long().expand(batch_size)
        num_selection_heads, block_topk = self.num_selection_heads, self.block_topk

        scores = torch.relu(torch.matmul(index_query.float(), block_index_keys.float().unsqueeze(1).transpose(-1, -2)))
        if index_weights is not None:
            scores = scores * index_weights.float().permute(0, 2, 1).unsqueeze(-1)
        group_size = num_index_heads // num_selection_heads
        scores = scores.reshape(batch_size, num_selection_heads, group_size, query_length, num_blocks).sum(dim=2)

        positions = key_length[:, None] - query_length + torch.arange(query_length)  # [N, L]
        num_visible_blocks = torch.clamp(torch.div(positions + 1, self.key_block_size, rounding_mode="floor"), min=0)
        visible = torch.arange(num_blocks) < num_visible_blocks[..., None]  # [N, L, B]
        scores = scores.masked_fill(~visible[:, None], float("-inf"))

        # stable top-k: descending score, equal scores -> smaller block index first
        order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :block_topk]
        block_indices = torch.full((batch_size, num_selection_heads, query_length, block_topk), -1, dtype=torch.long)
        block_indices[..., : order.shape[-1]] = order
        block_count = torch.clamp(num_visible_blocks, max=block_topk)  # [N, L]
        invalid = torch.arange(block_topk) >= block_count[..., None]  # [N, L, KB]
        block_indices = block_indices.masked_fill(invalid[:, None], -1)
        block_count = block_count[:, None, :].expand(-1, num_selection_heads, -1)
        return torch.cat([block_indices.flatten(), block_count.flatten()]).to(torch.int32)

    def _conversion_attributes(self):
        return {
            "key_block_size": self.key_block_size,
            "token_budget": self.token_budget,
            "num_selection_heads": self.num_selection_heads,
        }

    @staticmethod
    def _convert(context, key_block_size, token_budget, num_selection_heads):
        sparse_attention_indexer = _public_op("sparse_attention_indexer")
        if sparse_attention_indexer is not None:
            inputs = [context.get_input(i) for i in range(context.get_input_size())]
            node = sparse_attention_indexer(
                *inputs,
                key_block_size=key_block_size,
                token_budget=token_budget,
                num_selection_heads=num_selection_heads,
            )
            return [_pack_outputs([node.output(0), node.output(1)], node.output(0))]
        block_indices, block_count = SparseAttentionIndexer._decompose(
            context, key_block_size, token_budget, num_selection_heads
        )
        return [_pack_outputs([block_indices, block_count], block_indices)]

    @staticmethod
    def _decompose(context, key_block_size, token_budget, num_selection_heads):
        index_query = context.get_input(0)
        block_index_keys = context.get_input(1)
        index_weights = context.get_input(3) if context.get_input_size() > 3 else None
        block_topk = token_budget // key_block_size
        zero, one = _const_i64(0), _const_i64(1)

        query_shape = ops.shape_of(index_query, "i64")
        batch_size = _dim(query_shape, 0)
        num_index_heads = _dim(query_shape, 1)
        query_length = _dim(query_shape, 2)
        num_blocks = _dim(ops.shape_of(block_index_keys, "i64"), 1)
        key_length = _sequence_key_length(context.get_input(2), batch_size)  # [N]

        # One extra never-visible all-zero block key keeps every intermediate tensor non-empty when there is no
        # complete block yet (S < key_block_size).
        keys = ops.convert(block_index_keys, "f32")
        head_dim = _dim(ops.shape_of(block_index_keys, "i64"), 2)
        keys = ops.concat([keys, ops.broadcast(ops.constant(np.float32(0.0)), _vec(batch_size, 1, head_dim))], 1)

        # scores [N, Hi, L, B + 1] = ReLU(q . k) (* w), summed within every selection head -> [N, Hs, L, B + 1]
        scores = ops.relu(ops.matmul(ops.convert(index_query, "f32"), ops.unsqueeze(keys, one), False, True))
        if index_weights is not None:
            weights = ops.unsqueeze(
                ops.transpose(ops.convert(index_weights, "f32"), _const_i64([0, 2, 1])), _const_i64(-1)
            )
            scores = ops.multiply(scores, weights)
        group_size = ops.divide(num_index_heads, _const_i64(num_selection_heads))
        scores_shape = _vec(batch_size, num_selection_heads, group_size, query_length, ops.add(num_blocks, one))
        scores = ops.reduce_sum(ops.reshape(scores, scores_shape, False), _const_i64(2), False)
        neg_inf = ops.constant(np.float32(-np.inf))

        # blocks visible to the query at the position p = key_length[n] - L + l of its sequence:
        # b < (p + 1) // key_block_size; a padding query (p < 0) sees no block
        ratio = _const_i64(key_block_size)
        offsets = ops.range(zero, query_length, one, "i64")
        positions = ops.add(ops.unsqueeze(ops.subtract(key_length, query_length), one), offsets)  # [N, L]
        # (p + 1 <= 0 only for padding queries: 0 with floor and truncating division alike)
        num_visible_blocks = ops.maximum(ops.divide(ops.add(positions, one), ratio), zero)
        block_ids = ops.range(zero, ops.add(num_blocks, one), one, "i64")  # [B + 1]
        visible = ops.less(block_ids, ops.unsqueeze(num_visible_blocks, _const_i64(-1)))  # [N, L, B + 1]
        scores = ops.select(ops.unsqueeze(visible, one), scores, neg_inf)

        # stable top-k: equal scores -> smaller block index first
        k = ops.minimum(_const_i64(block_topk), ops.add(num_blocks, one))
        block_indices = ops.topk(scores, k, -1, "max", "value", "i32", stable=True).output(1)
        pads_end = _vec(0, 0, 0, ops.subtract(_const_i64(block_topk), k))
        minus_one = ops.constant(np.array(-1, dtype=np.int32))
        block_indices = ops.pad(block_indices, _const_i64([0, 0, 0, 0]), pads_end, "constant", minus_one)

        # block_count = min(token_budget / key_block_size, (p + 1) // key_block_size); the rest of a row is -1
        block_count = ops.minimum(num_visible_blocks, _const_i64(block_topk))  # [N, L]
        slots = ops.range(zero, _const_i64(block_topk), one, "i64")
        valid = ops.less(slots, ops.unsqueeze(block_count, _const_i64(-1)))  # [N, L, KB]
        block_indices = ops.select(ops.unsqueeze(valid, one), block_indices, minus_one)
        block_count = ops.broadcast(
            ops.convert(ops.unsqueeze(block_count, one), "i32"), _vec(batch_size, num_selection_heads, query_length)
        )
        return block_indices, block_count


class SparseAttention(_OVOperationModule):
    """
    Analogue of the SparseAttention operation: attention of every query token over the tokens of the key
    blocks selected for it (`block_indices` / `block_count` of its row `l // query_block_size`) and, in the causal
    mode, over the incomplete block that contains it.

    Batches with padding: sequence `n` has `key_length[n]` valid key tokens, the last ones of the `S` key positions
    (left padding); its blocks and positions are counted from its first valid token, padding tokens are never
    attended and padding queries produce zeros. Without `key_length` there is no padding.

    Inputs: query [N, H, L, E], key [N, Hkv, S, E], value [N, Hkv, S, Ev], block_indices [N, Hs, Lq, KB],
            block_count [N, Hs, Lq], key_length [N] (optional), scale (optional scalar, needs `key_length`).
    Output: [N, H, L, Ev].
    """

    _OP_TYPE = "SparseAttention"

    def __init__(self, key_block_size: int, query_block_size: int = 1, causal: bool = True):
        super().__init__()
        self.key_block_size = key_block_size
        self.query_block_size = query_block_size
        self.causal = causal

    def forward(self, query, key, value, block_indices, block_count, key_length=None, scale=None):
        batch_size, num_heads, query_length, head_dim = query.shape
        num_kv_heads, num_positions = key.shape[1], key.shape[2]
        num_selection_heads, block_topk = block_indices.shape[1], block_indices.shape[-1]
        ratio = self.key_block_size
        key_length = num_positions if key_length is None else key_length
        key_length = torch.as_tensor(key_length).reshape(-1).long().expand(batch_size)
        padding = num_positions - key_length  # [N]
        rows = torch.arange(query_length) // self.query_block_size
        block_indices, block_count = block_indices[:, :, rows].long(), block_count[:, :, rows].long()

        # block mask [N, Hs, L, ceil(S / r) + 2]: invalid slots -> column nB, padding keys -> column nB + 1 (never set)
        num_key_blocks = (num_positions + ratio - 1) // ratio
        valid = torch.arange(block_topk) < block_count.unsqueeze(-1)
        block_indices = torch.where(valid, block_indices, num_key_blocks)
        block_mask = torch.zeros((batch_size, num_selection_heads, query_length, num_key_blocks + 2), dtype=torch.bool)
        block_mask = block_mask.scatter(-1, block_indices, torch.ones_like(block_indices, dtype=torch.bool))
        # the block of every key position within its sequence
        sequence_positions = torch.arange(num_positions) - padding[:, None]  # [N, S]
        key_blocks = torch.where(sequence_positions >= 0, sequence_positions // ratio, num_key_blocks + 1)
        mask = torch.gather(
            block_mask, 3, key_blocks[:, None, None, :].expand(-1, num_selection_heads, query_length, -1)
        )
        if self.causal:
            positions = key_length[:, None] - query_length + torch.arange(query_length)  # [N, L]
            tail_begin = torch.div(positions + 1, ratio, rounding_mode="floor") * ratio
            tail = (
                (sequence_positions[:, None, :] >= tail_begin[..., None])
                & (sequence_positions[:, None, :] <= positions[..., None])
                & (sequence_positions[:, None, :] >= 0)
            )  # [N, L, S]
            mask = mask | tail[:, None]
        mask = mask.repeat_interleave(num_heads // num_selection_heads, dim=1)

        key = key.repeat_interleave(num_heads // num_kv_heads, dim=1).float()
        value = value.repeat_interleave(num_heads // num_kv_heads, dim=1).float()
        scale = 1.0 / head_dim**0.5 if scale is None else float(scale)
        attn = torch.matmul(query.float(), key.transpose(-1, -2)) * scale
        attn = torch.softmax(attn.masked_fill(~mask, float("-inf")), dim=-1).nan_to_num(0.0)
        output = torch.matmul(attn, value)
        output = torch.where(mask.any(dim=-1, keepdim=True), output, torch.zeros_like(output))
        return output.to(query.dtype)

    def _conversion_attributes(self):
        return {
            "key_block_size": self.key_block_size,
            "query_block_size": self.query_block_size,
            "causal": self.causal,
        }

    @staticmethod
    def _convert(context, key_block_size, query_block_size, causal):
        sparse_attention = _public_op("sparse_attention")
        if sparse_attention is not None:
            inputs = [context.get_input(i) for i in range(context.get_input_size())]
            node = sparse_attention(
                *inputs, key_block_size=key_block_size, query_block_size=query_block_size, causal=causal
            )
            return [node.output(0)]
        query = context.get_input(0)
        key = context.get_input(1)
        value = context.get_input(2)
        block_indices = context.get_input(3)
        block_count = context.get_input(4)
        key_length = context.get_input(5) if context.get_input_size() > 5 else None
        scale = context.get_input(6) if context.get_input_size() > 6 else None

        query_shape = ops.shape_of(query, "i64")
        batch_size, num_heads, query_length = _dim(query_shape, 0), _dim(query_shape, 1), _dim(query_shape, 2)
        key_shape = ops.shape_of(key, "i64")
        num_kv_heads, num_positions, key_head_dim = _dim(key_shape, 1), _dim(key_shape, 2), _dim(key_shape, 3)
        value_head_dim = _dim(ops.shape_of(value, "i64"), 3)
        index_shape = ops.shape_of(block_indices, "i64")
        num_selection_heads, block_topk = _dim(index_shape, 1), _dim(index_shape, 3)
        zero, one = _const_i64(0), _const_i64(1)
        key_length = _sequence_key_length(num_positions if key_length is None else key_length, batch_size)
        padding = ops.subtract(num_positions, key_length)  # [N]

        block_indices = ops.convert(block_indices, "i64")
        block_count = ops.convert(block_count, "i64")
        if query_block_size != 1:
            rows = ops.divide(ops.range(zero, query_length, one, "i64"), _const_i64(query_block_size))
            block_indices = ops.gather(block_indices, rows, _const_i64(2))
            block_count = ops.gather(block_count, rows, _const_i64(2))

        # block mask [N, Hs, L, ceil(S / r) + 2]: the first block_count entries of every row; the invalid entries ->
        # column nB, padding keys -> column nB + 1, which is never set
        ratio = _const_i64(key_block_size)
        num_key_blocks = ops.divide(ops.add(num_positions, _const_i64(key_block_size - 1)), ratio)
        slots = ops.range(zero, block_topk, one, "i64")
        valid = ops.less(slots, ops.unsqueeze(block_count, _const_i64(-1)))
        block_indices = ops.select(valid, block_indices, num_key_blocks)
        mask_shape = _vec(batch_size, num_selection_heads, query_length, ops.add(num_key_blocks, _const_i64(2)))
        block_mask = ops.broadcast(ops.constant(np.float32(0.0)), mask_shape)
        block_mask = ops.scatter_elements_update(
            block_mask, block_indices, ops.broadcast(ops.constant(np.float32(1.0)), ops.shape_of(block_indices)), -1
        )
        # token mask [N, Hs, L, S]: the block of every key position within its sequence
        key_positions = ops.range(zero, num_positions, one, "i64")
        sequence_positions = ops.subtract(ops.unsqueeze(key_positions, zero), ops.unsqueeze(padding, one))  # [N, S]
        is_valid_key = ops.greater_equal(sequence_positions, zero)
        key_blocks = ops.select(
            is_valid_key, ops.divide(ops.maximum(sequence_positions, zero), ratio), ops.add(num_key_blocks, one)
        )
        key_blocks = ops.broadcast(
            ops.unsqueeze(key_blocks, _const_i64([1, 2])),
            _vec(batch_size, num_selection_heads, query_length, num_positions),
        )
        mask = ops.greater(ops.gather_elements(block_mask, key_blocks, 3), ops.constant(np.float32(0.0)))
        if causal:
            offsets = ops.range(zero, query_length, one, "i64")
            positions = ops.add(ops.unsqueeze(ops.subtract(key_length, query_length), one), offsets)  # [N, L]
            # p < 0 only for padding queries, whose tail is empty for any rounding of the division
            tail_begin = ops.multiply(ops.divide(ops.add(positions, one), ratio), ratio)
            sequence_positions = ops.unsqueeze(sequence_positions, one)  # [N, 1, S]
            tail = ops.logical_and(
                ops.logical_and(
                    ops.greater_equal(sequence_positions, ops.unsqueeze(tail_begin, _const_i64(-1))),
                    ops.less_equal(sequence_positions, ops.unsqueeze(positions, _const_i64(-1))),
                ),
                ops.greater_equal(sequence_positions, zero),
            )  # [N, L, S]
            mask = ops.logical_or(mask, ops.unsqueeze(tail, one))

        # selection head g -> query heads of group g; kv head -> query heads of its group
        def repeat_heads(x, num_groups, rest):
            x = ops.unsqueeze(x, _const_i64(2))
            x = ops.broadcast(x, _vec(batch_size, num_groups, ops.divide(num_heads, num_groups), *rest))
            return ops.reshape(x, _vec(batch_size, num_heads, *rest), False)

        mask = repeat_heads(mask, num_selection_heads, [query_length, num_positions])
        key = repeat_heads(key, num_kv_heads, [num_positions, key_head_dim])
        value = repeat_heads(value, num_kv_heads, [num_positions, value_head_dim])

        if scale is None:
            head_dim = ops.convert(_dim(query_shape, 3), "f32")
            scale = ops.divide(ops.constant(np.float32(1.0)), ops.sqrt(head_dim))
        scale = ops.convert_like(scale, query)
        output = ops.scaled_dot_product_attention(query, key, value, mask, scale, causal=False)
        # a query head without attended positions (e.g. a padding query) produces zeros
        has_keys = ops.reduce_logical_or(mask, _const_i64(-1), True)
        output = ops.select(has_keys, output, ops.convert_like(ops.constant(np.float32(0.0)), output))
        return [output.output(0)]


class NGramHashFunction(_OVOperationModule):
    """
    Analogue of the NGramHashFunction operation: rows of the packed hashed n-gram table for every token and hashed
    head, `(XOR_k back(k) * multipliers[k]) % head_vocab_sizes[h] + head_offsets[h]` with exact int64 math, where
    `back(k)` is the token k steps back, or `eos_token_id` before the start of the history or across an EOS.

    Inputs: token_ids [N, L], history [N, ngram_size - 1].
    Outputs: row_ids [N, L, H] of int32 and new_history [N, ngram_size - 1] of the type of `history`. They are
             returned packed into one 1D tensor of the type of `history` (lossless: the row ids are below 2^31),
             flattened and concatenated, see `unpack()`.
    """

    _OP_TYPE = "NGramHashFunction"

    def __init__(
        self,
        ngram_size: int,
        heads_per_ngram: int,
        multipliers: Sequence[int],
        head_vocab_sizes: Sequence[int],
        head_offsets: Sequence[int],
        eos_token_id: int,
    ):
        super().__init__()
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.multipliers = [int(m) for m in multipliers]
        self.head_vocab_sizes = [int(s) for s in head_vocab_sizes]
        self.head_offsets = [int(o) for o in head_offsets]
        self.eos_token_id = int(eos_token_id)

    @property
    def context_length(self):
        return self.ngram_size - 1

    @property
    def num_heads(self):
        return self.context_length * self.heads_per_ngram

    def unpack(self, packed: torch.Tensor, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = token_ids.shape
        num_rows = batch_size * seq_len * self.num_heads
        row_ids = packed[:num_rows].reshape(batch_size, seq_len, self.num_heads)
        new_history = packed[num_rows:].reshape(batch_size, self.context_length)
        return row_ids.to(torch.int32), new_history

    def forward(self, token_ids, history):
        context, seq_len = self.context_length, token_ids.shape[1]
        sequence = torch.cat([history.long(), token_ids.long()], dim=1)  # [N, c + L]
        shifted = [sequence[:, context - k : context - k + seq_len] for k in range(context + 1)]
        is_eos = [s == self.eos_token_id for s in shifted]
        # back(k) is the token k steps back, unless an EOS lies strictly between it and the current token
        back = [shifted[0]]
        valid = torch.ones_like(shifted[0], dtype=torch.bool)
        for k in range(1, context + 1):
            back.append(torch.where(valid, shifted[k], torch.full_like(shifted[k], self.eos_token_id)))
            valid = valid & ~is_eos[k]

        rows = []
        mix = back[0] * self.multipliers[0]
        for order in range(2, self.ngram_size + 1):
            mix = torch.bitwise_xor(mix, back[order - 1] * self.multipliers[order - 1])
            for e in range(self.heads_per_ngram):
                head = (order - 2) * self.heads_per_ngram + e
                rows.append(torch.remainder(mix, self.head_vocab_sizes[head]) + self.head_offsets[head])
        row_ids = torch.stack(rows, dim=-1)
        new_history = sequence[:, -context:]
        return torch.cat([row_ids.flatten(), new_history.flatten()]).to(history.dtype)

    def _conversion_attributes(self):
        return {
            "ngram_size": self.ngram_size,
            "heads_per_ngram": self.heads_per_ngram,
            "multipliers": self.multipliers,
            "head_vocab_sizes": self.head_vocab_sizes,
            "head_offsets": self.head_offsets,
            "eos_token_id": self.eos_token_id,
        }

    @staticmethod
    def _convert(context, ngram_size, heads_per_ngram, multipliers, head_vocab_sizes, head_offsets, eos_token_id):
        # The hash needs exact 63-bit integer math, but plugins execute i64 arithmetic in i32 (the CPU plugin lowers
        # i64 to i32) and evaluate i32 Divide, FloorMod and comparisons through f32, which is exact only below 2^24.
        # The decomposition below is therefore built from i32 operations that stay exact on the full i32 range:
        # Add, Subtract, Multiply (without overflow), BitwiseXor, BitwiseAnd and BitwiseRightShift.
        #   * every 64-bit value is kept as little-endian limbs of `limb_bits` bits,
        #   * a product token * multiplier is computed limb by limb with carries (shift / mask),
        #   * XOR is applied limb-wise,
        #   * the remainder modulo p is accumulated with Horner steps `acc = (acc * 2^limb_bits + limb) % p`, where
        #     the reduction of `acc < p * 2^limb_bits` is a sequence of branch-free conditional subtractions of
        #     `p * 2^j`, j = limb_bits - 1 .. 0.
        # The limb width keeps every intermediate value below 2^31: `p * 2^limb_bits <= 2^31`, limb products < 2^22.
        from openvino.opset15 import bitwise_right_shift

        if max(head_vocab_sizes) >= 2**30 or max(head_offsets) + max(head_vocab_sizes) >= 2**31:
            raise ValueError("NGramHashFunction decomposition supports head vocabularies and table rows below 2^31")
        limb_bits = min(11, 31 - max(head_vocab_sizes).bit_length())
        limb_base = 1 << limb_bits
        num_limbs = -(-64 // limb_bits)  # limbs of a 64-bit value
        num_token_limbs = -(-31 // limb_bits)  # limbs of a non-negative i32 token id

        def const_i32(value):
            return ops.constant(np.array(value, dtype=np.int32))

        def low_bits(x):
            return ops.bitwise_and(x, const_i32(limb_base - 1))

        def high_bits(x, shift=limb_bits):
            return bitwise_right_shift(x, const_i32(shift))

        def reduce_mod(x, modulus):
            # x % modulus for 0 <= x < modulus * 2^limb_bits
            for j in reversed(range(limb_bits)):
                step = const_i32(modulus << j)
                diff = ops.subtract(x, step)
                negative = high_bits(diff, 31)  # -1 if diff < 0 else 0
                x = ops.add(diff, ops.bitwise_and(step, negative))
            return x

        history_input = context.get_input(1)
        token_ids = ops.convert(context.get_input(0), "i32")
        history = ops.convert(history_input, "i32")
        num_context = ngram_size - 1
        sequence = ops.concat([history, token_ids], 1)  # [N, c + L]
        seq_len = _dim(ops.shape_of(token_ids, "i64"), 1)
        eos = const_i32(eos_token_id)
        axis = _const_i64([1])

        def shifted(k):
            start = _const_i64([num_context - k])
            stop = ops.unsqueeze(ops.add(seq_len, _const_i64(num_context - k)), _const_i64(0))
            return ops.slice(sequence, start, stop, _const_i64([1]), axis)

        # back(k): the token k steps back, or EOS if an EOS lies strictly between it and the current token
        back = [shifted(0)]
        valid = None
        for k in range(1, num_context + 1):
            token = shifted(k)
            back.append(token if valid is None else ops.select(valid, token, eos))
            not_eos = ops.not_equal(token, eos)
            valid = not_eos if valid is None else ops.logical_and(valid, not_eos)

        def product_limbs(token, multiplier):
            # limbs of token * multiplier (exact, < 2^63 by the operation contract)
            token_limbs = [low_bits(high_bits(token, limb_bits * j)) for j in range(num_token_limbs)]
            multiplier_limbs = [(multiplier >> (limb_bits * i)) & (limb_base - 1) for i in range(num_limbs)]
            limbs, carry = [], None
            for c in range(num_limbs):
                column = carry
                for j in range(min(c + 1, num_token_limbs)):
                    if multiplier_limbs[c - j]:
                        term = ops.multiply(token_limbs[j], const_i32(multiplier_limbs[c - j]))
                        column = term if column is None else ops.add(column, term)
                if column is None:
                    column = ops.multiply(token, const_i32(0))
                limbs.append(low_bits(column))
                carry = high_bits(column)
            return limbs

        rows = []
        mix = product_limbs(back[0], multipliers[0])
        for order in range(2, ngram_size + 1):
            term = product_limbs(back[order - 1], multipliers[order - 1])
            mix = [ops.bitwise_xor(a, b) for a, b in zip(mix, term)]
            for e in range(heads_per_ngram):
                head = (order - 2) * heads_per_ngram + e
                remainder = None
                for limb in reversed(mix):
                    remainder = (
                        limb if remainder is None else ops.add(ops.multiply(remainder, const_i32(limb_base)), limb)
                    )
                    remainder = reduce_mod(remainder, head_vocab_sizes[head])
                row = ops.add(remainder, const_i32(head_offsets[head]))
                rows.append(ops.unsqueeze(row, _const_i64(-1)))
        row_ids = ops.concat(rows, -1)  # [N, L, H]
        new_history = ops.slice(sequence, _const_i64([-num_context]), _const_i64([_I64_MAX]), _const_i64([1]), axis)
        flat = _const_i64([-1])
        packed = ops.concat([ops.reshape(row_ids, flat, False), ops.reshape(new_history, flat, False)], 0)
        # packed into the type of history, which is the type of new_history; unpack() restores int32 row ids
        return [ops.convert_like(packed, history_input).output(0)]


def get_openvino_extensions(modules: Sequence[_OVOperationModule]):
    """Returns `module_extensions` and `conversion_extensions` of a patcher for the given operation modules."""
    module_extensions, conversion_extensions = {}, []
    for module in modules:
        # the operation modules and their PackedOutput modules
        for op_module in module.modules():
            if isinstance(op_module, _OVOperationModule):
                module_extension, conversion_extension = op_module.openvino_extensions()
                module_extensions[op_module] = module_extension
                conversion_extensions.append(conversion_extension)
    return module_extensions, conversion_extensions
