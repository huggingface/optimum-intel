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

import dataclasses
import functools
import inspect
import logging
import sys
import types
from typing import Any, Callable

import torch
import transformers
from torch.onnx import symbolic_helper
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput

from optimum.exporters.base import ExporterConfig
from optimum.intel.utils.import_utils import is_torch_version, is_transformers_version


if is_transformers_version(">=", "4.44") and is_transformers_version("<", "4.50"):
    from optimum.exporters.openvino._traceable_cache import TraceableCache


if is_transformers_version(">=", "4.48"):
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache
if is_transformers_version(">=", "4.53"):
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        _ignore_causal_mask_sdpa,
        and_masks,
        causal_mask_function,
        eager_mask,
        padding_mask_function,
        prepare_padding_mask,
        sdpa_mask,
    )


if is_transformers_version(">=", "4.53.1"):
    from transformers.masking_utils import find_packed_sequence_indices


if is_transformers_version(">=", "4.54"):
    from transformers.utils import TransformersKwargs

    from optimum.exporters.openvino._traceable_decorator import traceable_check_model_inputs
else:
    TransformersKwargs = object


if is_transformers_version(">=", "4.56"):
    import transformers.masking_utils
    from transformers.cache_utils import DynamicLayer


logger = logging.getLogger(__name__)


@symbolic_helper.parse_args("v", "v")
def __ior_(g, self: torch._C.Value, other: torch._C.Value) -> torch._C.Value:
    return g.op("Or", self, other)


torch.onnx.register_custom_op_symbolic("aten::__ior__", __ior_, 14)

if is_torch_version("<", "2.9"):
    # this was fixed in torch in 2.9 https://github.com/pytorch/pytorch/pull/159973
    from torch.onnx import JitScalarType
    from torch.onnx.symbolic_opset14 import _attention_scale, _causal_attention_mask

    @symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v", "b")
    def scaled_dot_product_attention(
        g,
        query: torch._C.Value,
        key: torch._C.Value,
        value: torch._C.Value,
        attn_mask: torch._C.Value | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: torch._C.Value | None = None,
        enable_gqa: bool = False,
    ):
        assert (not is_causal) or (
            is_causal and symbolic_helper._is_none(attn_mask)
        ), "is_causal and attn_mask cannot be set at the same time"
        assert not enable_gqa, "conversion of scaled_dot_product_attention not implemented if enable_gqa is True"

        if symbolic_helper._is_none(scale):
            scale = _attention_scale(g, query)

        if is_causal:
            attn_mask = _causal_attention_mask(g, query, key)

        # Swap the last two axes of key
        # NOTE: onnx-script has different logic here, because the attribute perms in
        # transpose needs list of ints
        key_shape_builtin = symbolic_helper._get_tensor_rank(key)
        key_transposed_axes = list(range(key_shape_builtin))
        key_transposed_axes[-1], key_transposed_axes[-2] = (key_transposed_axes[-2], key_transposed_axes[-1])
        key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

        # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
        # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
        query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
        key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
        mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

        if symbolic_helper._is_none(attn_mask):
            mul_qk_add = mul_qk
            attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
        elif JitScalarType.from_value(attn_mask) == JitScalarType.BOOL:
            # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
            const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
            const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
            attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
            mul_qk_add = g.op("Add", mul_qk, attn_mask)
            attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
            # when using scaled dot product attention with a boolean mask, we replace NaN values in attn_weight with 0.0
            attn_weight = g.op(
                "Where", g.op("IsNaN", attn_weight), g.op("Constant", value_t=torch.tensor([0.0])), attn_weight
            )
        elif JitScalarType.from_value(attn_mask) in (
            JitScalarType.FLOAT,
            JitScalarType.HALF,
            JitScalarType.BFLOAT16,
        ):
            mul_qk_add = g.op("Add", mul_qk, attn_mask)
            attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)
        else:
            raise ValueError(f"Unsupported type for attn_mask: {JitScalarType.from_value(attn_mask)}")

        if dropout_p != 0:
            attn_weight = g.op(
                "Dropout",
                attn_weight,
                g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
            )

        return g.op("MatMul", attn_weight, value)

    torch.onnx.register_custom_op_symbolic("aten::scaled_dot_product_attention", scaled_dot_product_attention, 14)


def patch_everywhere(attribute_name: str, patch: Any, module_name_prefix: str | None = None):
    """Finds all occurrences of `attribute_name` in the loaded modules and patches them with `patch`.

    Args:
        attribute_name (`str`):
            The name of attribute to patch.
        patch (`Any`):
            The patch for the attribute.
        module_name_prefix (`Optional[str]`, defaults to `None`):
            If set, only module names starting with this prefix will be considered for patching.
    """
    # sys.modules may be updated while being iterated over, hence the list copy.
    for name in list(sys.modules):
        module = sys.modules[name]
        if module_name_prefix is not None and not name.startswith(module_name_prefix):
            continue
        if hasattr(module, attribute_name):
            setattr(module, attribute_name, patch)


def override_arguments(args, kwargs, forward_signature, model_kwargs: dict[str, Any]):
    """Override the args and kwargs with the argument values from model_kwargs, following the signature forward_signature corresponding to args and kwargs."""
    args = list(args)

    for argument in model_kwargs:
        if argument in forward_signature.parameters:
            argument_index = list(forward_signature.parameters.keys()).index(argument)
            if argument in kwargs or len(args) <= argument_index:
                kwargs[argument] = model_kwargs[argument]
            else:
                args[argument_index] = model_kwargs[argument]
        else:
            kwargs[argument] = model_kwargs[argument]

    return args, kwargs


def preprocess_encoder_outputs(encoder_outputs):
    if is_transformers_version(">=", "4.54") and isinstance(encoder_outputs, (list, tuple)):
        encoder_outputs = BaseModelOutput(*encoder_outputs)

    return encoder_outputs


def preprocess_past_key_values(past_key_values):
    if (
        is_transformers_version(">=", "4.48")
        and isinstance(past_key_values, (list, tuple))
        and isinstance(past_key_values[0], (list, tuple))
    ):
        if len(past_key_values[0]) == 2:
            if hasattr(DynamicCache, "from_legacy_cache"):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            else:
                past_key_values = DynamicCache(past_key_values)
        elif len(past_key_values[0]) == 4:
            if hasattr(EncoderDecoderCache, "from_legacy_cache"):
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            else:
                past_key_values = EncoderDecoderCache(
                    DynamicCache([layer[:2] for layer in past_key_values]),
                    DynamicCache([layer[2:] for layer in past_key_values]),
                )
        else:
            raise ValueError(
                f"past_key_values should have either 2 or 4 elements, but it has {len(past_key_values[0])} elements."
            )

    return past_key_values


def postprocess_past_key_values(past_key_values, output_names: list[str]):
    if is_transformers_version(">=", "4.48") and isinstance(past_key_values, (EncoderDecoderCache, DynamicCache)):
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        elif isinstance(past_key_values, DynamicCache):
            past_key_values = [(lay.keys, lay.values) for lay in past_key_values.layers]
        elif isinstance(past_key_values, EncoderDecoderCache):
            past_key_values = [
                (self_lay.keys, self_lay.values, cross_lay.keys, cross_lay.values)
                for self_lay, cross_lay in zip(
                    past_key_values.self_attention_cache.layers,
                    past_key_values.cross_attention_cache.layers,
                )
            ]
        else:
            raise NotImplementedError(f"Unable to serialize class {type(past_key_values)}.")

    if (
        isinstance(past_key_values, (list, tuple))
        and isinstance(past_key_values[0], (list, tuple))
        and not any("encoder.key" in output_name for output_name in output_names)
    ):
        past_key_values = tuple(pkv[:2] for pkv in past_key_values)

    return past_key_values


@dataclasses.dataclass
class PatchingSpec:
    """Data class that holds patching specifications.

    Args:
        o: Module / object where the op to patch is located
        name: Name of the op to monkey patch
        custom_op: Custom op that patches the original op
        orig_op: Original op that is being patched
        op_wrapper: Wrapper (optional) that wraps both the original and custom ops.
            It is useful for ops that are class or static methods for instance.
    """

    o: Any
    name: str
    custom_op: Callable
    orig_op: Callable | None = None
    op_wrapper: Callable | None = None


# An ONNX-export-compatible version of `tensor.unfold`. Without this, we get:
# torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of operator Unfold, input size not accessible.
# See https://github.com/pytorch/pytorch/issues/81871 for more information
def onnx_compatible_unfold(input_tensor, dimension, size, step):
    """Custom implementation of torch.unfold without using torch.unfold.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        dimension (int): The dimension to unfold.
        size (int): The size of each slice.
        step (int): The step size between slices.

    Returns:
        torch.Tensor: The unfolded tensor.
    """
    # Check if dimension is within the valid range
    if not (-input_tensor.dim() <= dimension < input_tensor.dim()):
        raise ValueError(
            f"Dimension out of range (expected to be in range of [{-input_tensor.dim()}, {input_tensor.dim() - 1}], but got {dimension})"
        )

    # Normalize negative dimension
    dimension = dimension % input_tensor.dim()

    # Compute the shape of the unfolded output
    input_size = input_tensor.size(dimension)
    num_slices = (input_size - size) // step + 1

    # Permute dimension to the end for easier indexing
    input_tensor = input_tensor.transpose(dimension, -1)

    # Extract slices
    slices = []
    for i in range(num_slices):
        start = i * step
        end = start + size
        slices.append(input_tensor[..., start:end])

    # Stack slices and permute dimensions back
    result = torch.stack(slices, dim=-2).transpose(dimension, -2)
    return result


# An ONNX-export-compatible version of `tensor.repeat_interleave`.
# Without this, we get the following error: https://github.com/pytorch/pytorch/issues/145100
# NOTE: This implementation is only necessary for export with dynamo=False (dynamo=True works correctly).
# and can be removed once Optimum switches to dynamo-based exports
def onnx_compatible_repeat_interleave(input_tensor, repeats, dim=None, output_size=None):  # noqa: D417
    """Custom implementation of torch.repeat_interleave without using torch.repeat_interleave.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        repeats (int or torch.Tensor): The number of repetitions for each element.
        dim (int, optional): The dimension along which to repeat. Defaults to None.

    Returns:
        torch.Tensor: The repeated tensor.
    """
    if isinstance(repeats, int) or (torch.is_tensor(repeats) and repeats.dim() == 0):
        if dim is None:
            return input_tensor.flatten().unsqueeze(1).expand(-1, repeats).flatten()
        repeats = torch.full((input_tensor.shape[dim],), repeats, dtype=torch.long, device=input_tensor.device)

    if dim is None:
        return onnx_compatible_repeat_interleave(input_tensor.flatten(), repeats, 0)

    if dim != 0:
        input_tensor = input_tensor.transpose(0, dim)

    # Create expand mask
    max_repeats = repeats.max()
    expanded = input_tensor.unsqueeze(1).expand(-1, max_repeats, *input_tensor.shape[1:])
    mask = torch.arange(max_repeats, device=input_tensor.device) < repeats.unsqueeze(1)
    result = expanded[mask]

    if dim != 0:
        result = result.transpose(0, dim)

    return result


# Custom implementation of torch.linalg.matrix_norm not using torch.linalg.matrix_norm, torch.norm or torch.linalg.norm.
def onnx_compatible_linalg_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> torch.Tensor:
    if ord != 2:
        raise ValueError(
            f"Only ord=2 is supported by onnx_compatible_linalg_norm, but got ord={ord}. "
            "Please extend this function to support other norms."
        )

    if dim is None:
        dim = (-2, -1)

    norm = torch.sqrt(torch.sum(torch.square(x), dim=dim, keepdim=keepdim))

    if dtype is not None:
        norm = norm.to(dtype)
    if out is not None:
        out.copy_(norm)

    return norm


def onnx_compatible_rms_norm(input, normalized_shape, weight=None, eps=None):
    if eps is None:
        eps = torch.finfo(input.dtype).eps

    axis = -len(normalized_shape)
    mean_square = torch.mean(torch.square(input), dim=axis, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    output = input / rms

    if weight is not None:
        output = output * weight

    return output


# A patched version of https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/masking_utils.py#L602
# That returns a tensor of zeros with the same shape as position_ids indicating no packed sequence indices.
def find_packed_sequence_indices_patched(position_ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(position_ids)


if is_transformers_version(">=", "4.53"):
    _prepare_padding_mask_slice = "_slice" in inspect.signature(prepare_padding_mask).parameters
else:
    _prepare_padding_mask_slice = False


# Custom vectorized implementation of sdpa_mask without using vmap
def _orig_sdpa_mask_without_vmap(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable | None = None,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> torch.Tensor | None:
    if mask_function is None:
        mask_function = causal_mask_function

    q_length = cache_position.shape[0]
    # Potentially pad the 2D mask, and slice it correctly
    if _prepare_padding_mask_slice:
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
    else:
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask, instead relying on the `is_causal` argument
    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    # Create broadcatable indices
    device = cache_position.device
    q_indices = cache_position[None, None, :, None]
    head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None, None, None]
    kv_indices = torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :] + kv_offset
    # Apply mask function element-wise through broadcasting
    causal_mask = mask_function(batch_indices, head_indices, q_indices, kv_indices)
    # Expand the mask to match batch size and query length if they weren't used in the mask function
    causal_mask = causal_mask.expand(batch_size, -1, q_length, kv_length)

    return causal_mask


# Compatibility wrapper for sdpa_mask_without_vmap from optimum.
# The installed optimum version expects (batch_size, cache_position: Tensor, kv_length, ...),
# but transformers >= 5.5 passes (batch_size, q_length: int, kv_length: int, q_offset: int, ...).
def sdpa_mask_without_vmap(batch_size, q_length=None, kv_length=None, q_offset=0, kv_offset=0, **kwargs):
    import inspect

    sig = inspect.signature(_orig_sdpa_mask_without_vmap)
    if is_transformers_version(">=", "5.5") and "cache_position" in sig.parameters and q_length is not None:
        # Old optimum signature: (batch_size, cache_position, kv_length, kv_offset, ...)
        cache_position = torch.arange(q_length, dtype=torch.long) + q_offset
        kwargs.pop("q_offset", None)
        kwargs.pop("allow_is_bidirectional_skip", None)
        kwargs.pop("allow_torch_fix", None)
        kwargs.pop("use_vmap", None)
        kwargs.pop("device", None)
        return _orig_sdpa_mask_without_vmap(batch_size, cache_position, kv_length, kv_offset=kv_offset, **kwargs)
    else:
        return _orig_sdpa_mask_without_vmap(
            batch_size, q_length=q_length, kv_length=kv_length, q_offset=q_offset, kv_offset=kv_offset, **kwargs
        )


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
def eager_mask_without_vmap(*args, **kwargs) -> torch.Tensor:
    kwargs.pop("allow_is_causal_skip", None)
    dtype = kwargs.get("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(*args, allow_is_causal_skip=False, **kwargs)
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), torch.finfo(dtype).min)
    return mask


original_triu = torch.triu
original_tril = torch.tril


# Custom implementation of torch.tril that doesn't fail on int32 tensors.
def onnx_compatible_tril(input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if input_tensor.dtype == torch.int32:
        return original_tril(input_tensor.to(torch.int64), *args, **kwargs).to(torch.int32)
    else:
        return original_tril(input_tensor, *args, **kwargs)


# Custom implementation of torch.triu that doesn't fail on int32 tensors.
def onnx_compatible_triu(input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if input_tensor.dtype == torch.int32:
        return original_triu(input_tensor.to(torch.int64), *args, **kwargs).to(torch.int32)
    else:
        return original_triu(input_tensor, *args, **kwargs)


original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention


# A patched `torch.nn.functional.scaled_dot_product_attention` that doesn't fail during tracing
# from passing `is_causal` as a tensor (which is usually obtained with tensor shapes comparisons).
def traceable_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    if isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if "enable_gqa" in kwargs:
        kwargs.pop("enable_gqa")

    attn_weights = original_scaled_dot_product_attention(
        query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs
    )

    return attn_weights


# No-op bfloat16 casting to avoid issues with legacy ONNX export which cast to complex128
def noop_bfloat16_casting(self):
    return self


original_movedim = torch.Tensor.movedim


def onnx_compatible_movedim(self: torch.Tensor, dim1, dim2) -> torch.Tensor:
    dim = self.dim()
    if dim1 < 0:
        dim1 += dim
    if dim2 < 0:
        dim2 += dim
    return original_movedim(self, dim1, dim2)


def patched_dynamic_layer_update(
    self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: dict[str, Any] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if self.keys is None:
        self.keys = key_states
        self.values = value_states
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.is_initialized = True
    else:
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values


UNSUPPORTED_OPS_PATCHING_SPEC = [
    PatchingSpec(torch, "tril", onnx_compatible_tril, torch.tril),
    PatchingSpec(torch, "triu", onnx_compatible_triu, torch.triu),
    PatchingSpec(torch, "rms_norm", onnx_compatible_rms_norm, torch.rms_norm),
    PatchingSpec(torch.Tensor, "unfold", onnx_compatible_unfold, torch.Tensor.unfold),
    PatchingSpec(torch.linalg, "norm", onnx_compatible_linalg_norm, torch.linalg.norm),
    PatchingSpec(torch.Tensor, "bfloat16", noop_bfloat16_casting, torch.Tensor.bfloat16),
    PatchingSpec(torch.Tensor, "movedim", onnx_compatible_movedim, torch.Tensor.movedim),
    PatchingSpec(torch.Tensor, "repeat_interleave", onnx_compatible_repeat_interleave, torch.Tensor.repeat_interleave),
    # TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
    PatchingSpec(torch.Tensor, "__len__", lambda x: x.shape[0], torch.Tensor.__len__),
    PatchingSpec(
        torch.nn.functional,
        "scaled_dot_product_attention",
        traceable_scaled_dot_product_attention,
        torch.nn.functional.scaled_dot_product_attention,
    ),
]


class ModelPatcher:
    def __init__(
        self,
        config: ExporterConfig,
        model: PreTrainedModel,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self._model = model

        patching_specs = config.PATCHING_SPECS or []
        patching_specs.extend(UNSUPPORTED_OPS_PATCHING_SPEC)

        self._patching_specs = []
        for spec in patching_specs:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

        self.orig_forward_name = "forward" if hasattr(self._model, "forward") else "call"
        self.orig_forward = getattr(self._model, self.orig_forward_name)

        if is_transformers_version(">=", "4.54") and hasattr(self.orig_forward, "__wrapped__"):
            # the original check_model_inputs has some failing cases that we fix in traceable_check_model_inputs
            # we fix those issues in a PR in transformers https://github.com/huggingface/transformers/pull/40811
            # issues are: support for positional args (use_cache for instance) and fix for _CAN_RECORD_REGISTRY
            # explicitly mapping to None for some models
            self.orig_forward = types.MethodType(
                traceable_check_model_inputs(self.orig_forward.__wrapped__), self._model
            )

        self.real_config = config
        use_cache = getattr(self.real_config, "use_past", False)
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        for module in self._model.modules():
            if hasattr(module, "config") and hasattr(module.config, "use_cache"):
                module.config.use_cache = use_cache

        @functools.wraps(self.orig_forward)
        def patched_forward(*args, **kwargs):
            signature = inspect.signature(self.orig_forward)
            args, kwargs = override_arguments(args, kwargs, signature, model_kwargs=self.model_kwargs)

            # Transformers doesn't always respect the config.use_cache attribute
            # there are cases where setting use_cache to true in every config and
            # subconfig of a model still doesn't enable past_key_values in the outputs (gemma3)
            # Explicitly setting the use_cache argument of the forward method seems to be the most reliable way
            if "use_cache" in signature.parameters:
                use_cache_index = list(signature.parameters.keys()).index("use_cache")
                if use_cache_index < len(args):
                    args[use_cache_index] = use_cache
                elif "use_cache" in kwargs:
                    kwargs["use_cache"] = use_cache

            if "past_key_values" in signature.parameters:
                # Most models require past_key_values to be a cache instance instead of a tuple now
                pkv_index = list(signature.parameters.keys()).index("past_key_values")
                if pkv_index < len(args) and args[pkv_index] is not None:
                    args[pkv_index] = preprocess_past_key_values(args[pkv_index])
                elif kwargs.get("past_key_values") is not None:
                    kwargs["past_key_values"] = preprocess_past_key_values(kwargs["past_key_values"])

            if "encoder_outputs" in signature.parameters:
                # Some encoder-decoder models started to not accept encoder_outputs as tuple (e.g. moonshine)
                encoder_outputs_index = list(signature.parameters.keys()).index("encoder_outputs")
                if encoder_outputs_index < len(args) and args[encoder_outputs_index] is not None:
                    args[encoder_outputs_index] = preprocess_encoder_outputs(args[encoder_outputs_index])
                elif kwargs.get("encoder_outputs") is not None:
                    kwargs["encoder_outputs"] = preprocess_encoder_outputs(kwargs["encoder_outputs"])

            outputs = self.orig_forward(*args, **kwargs)

            # This code block handles different cases of the filtered_outputs input to align it with the expected
            # format of outputs. It is common for the output type of a model to vary, such as tensor, list,
            # tuple, etc. For Transformers models, the output is encapsulated in a ModelOutput object that
            # contains the output names of the model. In the case of Timm classification models, the output
            # is of type tensor. By default, it is assumed that the output names mentioned in the ONNX config
            # match the outputs in order.
            filtered_outputs = {}
            output_names = list(config.outputs.keys())
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    onnx_output_name = config.torch_to_onnx_output_map.get(name, name)
                    if (
                        onnx_output_name in output_names
                        or (use_cache and name.startswith("past_key_values"))
                        or any(key.startswith(onnx_output_name) for key in output_names)
                    ):
                        filtered_outputs[name] = value
            elif isinstance(outputs, (list, tuple)):
                filtered_outputs = dict(zip(output_names, outputs))
            else:
                if len(output_names) > 1:
                    num_outputs = len(output_names)
                    output_names_str = ", ".join(output_names)
                    raise ValueError(
                        f"{config.__class__.__name__} expects the model to return {num_outputs} outputs: {output_names_str}, "
                        f"but the it returned a single output of type {type(outputs)}. Please make sure either that the model "
                        "returns all the expected outputs, or that the ONNX config is correctly defined with the expected outputs."
                    )
                output_name = output_names[0]
                filtered_outputs[output_name] = outputs

            if filtered_outputs.get("past_key_values") is not None:
                filtered_outputs["past_key_values"] = postprocess_past_key_values(
                    filtered_outputs["past_key_values"], output_names=output_names
                )

            return filtered_outputs

        self.patched_forward = patched_forward

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    def __enter__(self):
        self.patch_ops()
        setattr(self._model, self.orig_forward_name, self.patched_forward)

        # This is a workaround for the Cache class in transformers, we replace it
        # with traceable cache is because the original one used in transformers
        # inherited from nn.Module (for a couple versions), which can't be traced as input.
        if is_transformers_version(">=", "4.44") and is_transformers_version("<", "4.50"):
            self.original_cache_class = transformers.cache_utils.Cache
            transformers.cache_utils.Cache = TraceableCache

        # This is a workaround for mask generation in transformers >= 4.53.
        # The masking process uses vmap which is not traceable by TorchScript.
        if is_transformers_version(">=", "4.53"):
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask_without_vmap)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask_without_vmap)

        # This is a workaround for the find_packed_sequence_indices function in transformers which
        # should only return a tensor of zeros with the same shape as position_ids indicating no packed sequence indices.
        # The function uses torch.diff which is not traceable by TorchScript.
        if is_transformers_version(">=", "4.53.1"):
            self.original_find_packed_sequence_indices = find_packed_sequence_indices
            transformers.masking_utils.find_packed_sequence_indices = find_packed_sequence_indices_patched

        # Starting from transformers 4.56.0, DynamicCache uses DynamicLayer which has an update method
        # that uses torch.cat to concatenate an empty tensor with the key/value states during the first call.
        # This causes issues during TorchScript tracing.
        if is_transformers_version(">=", "4.56"):
            self.original_dynamic_layer_update = DynamicLayer.update
            DynamicLayer.update = patched_dynamic_layer_update

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)

        if is_transformers_version(">=", "4.44") and is_transformers_version("<", "4.50"):
            transformers.cache_utils.Cache = self.original_cache_class

        if is_transformers_version(">=", "4.53"):
            ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", sdpa_mask)
            ALL_MASK_ATTENTION_FUNCTIONS.register("eager", eager_mask)

        if is_transformers_version(">=", "4.53.1"):
            transformers.masking_utils.find_packed_sequence_indices = self.original_find_packed_sequence_indices

        if is_transformers_version(">=", "4.56"):
            DynamicLayer.update = self.original_dynamic_layer_update

    def __call__(self, *args, **kwargs):
        if getattr(self._model, self.orig_forward_name) is self.orig_forward:
            logger.warning("Running the non-patched model")
        return self._model(*args, **kwargs)
