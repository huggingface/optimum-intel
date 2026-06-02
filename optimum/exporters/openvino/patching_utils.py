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
import types
from typing import Any, Callable

import torch
import transformers
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput

from optimum.exporters.base import ExporterConfig
from optimum.intel.utils.import_utils import is_transformers_version


if is_transformers_version(">=", "4.44") and is_transformers_version("<", "4.50"):
    from optimum.exporters.openvino._traceable_cache import TraceableCache


if is_transformers_version(">=", "4.48"):
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache
if is_transformers_version(">=", "4.53"):
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        eager_mask,
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


def postprocess_past_key_values(past_key_values):
    if isinstance(past_key_values, (EncoderDecoderCache, DynamicCache)):
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


# A patched version of https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/masking_utils.py#L602
# That returns a tensor of zeros with the same shape as position_ids indicating no packed sequence indices.
def find_packed_sequence_indices_patched(position_ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(position_ids)


if is_transformers_version(">=", "4.53"):
    _prepare_padding_mask_slice = "_slice" in inspect.signature(prepare_padding_mask).parameters
else:
    _prepare_padding_mask_slice = False


# Compatibility wrapper for sdpa_mask_without_vmap from optimum.
# The installed optimum version expects (batch_size, cache_position: Tensor, kv_length, ...),
# but transformers >= 5.5 passes (batch_size, q_length: int, kv_length: int, q_offset: int, ...).
def sdpa_mask_without_vmap(**kwargs):
    kwargs.pop("use_vmap", None)
    if is_transformers_version("<", "5"):
        return sdpa_mask_without_vmap_legacy(**kwargs)
    elif (
        is_transformers_version(">=", "5.4")
        and is_transformers_version("<", "5.9")
        and isinstance(kwargs.get("q_length", None), torch.Tensor)
    ):
        q_length = kwargs.pop("q_length", None)
        q_offset = kwargs.pop("q_offset", 0)
        cache_position = torch.arange(q_offset, q_offset + q_length, device=q_length.device)
        return sdpa_mask(q_length=cache_position, use_vmap=False, **kwargs)
    else:
        return sdpa_mask(use_vmap=False, **kwargs)


# Adapted from https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/masking_utils.py#L433
# Specifically for OpenVINO, we use torch.finfo(torch.float16).min instead of torch.finfo(dtype).min
def eager_mask_without_vmap(**kwargs) -> Optional[torch.Tensor]:
    kwargs.pop("allow_is_causal_skip", None)
    kwargs.pop("allow_torch_fix", None)
    dtype = kwargs.pop("dtype", torch.float32)
    mask = sdpa_mask_without_vmap(allow_is_causal_skip=False, allow_torch_fix=False, **kwargs)
    if mask is not None:
        # we use torch.finfo(torch.float16).min instead torch.finfo(dtype).min to avoid an overflow but not
        # sure this is the right way to handle this, we are basically pretending that -65,504 is -inf
        mask = torch.where(
            mask,
            torch.tensor(0.0, device=mask.device, dtype=dtype),
            torch.tensor(torch.finfo(torch.float16).min, device=mask.device, dtype=dtype),
        )
    return mask


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
    # TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
    PatchingSpec(torch.Tensor, "__len__", lambda x: x.shape[0], torch.Tensor.__len__),
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
            # is of type tensor. By default, it is assumed that the output names mentioned in the OpenVINO config
            # match the outputs in order.
            filtered_outputs = {}
            output_names = list(config.outputs.keys())
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    ov_output_name = config.torch_to_ov_output_map.get(name, name)
                    if (
                        ov_output_name in output_names
                        or (use_cache and name.startswith("past_key_values"))
                        or any(key.startswith(ov_output_name) for key in output_names)
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
                        "returns all the expected outputs, or that the OpenVINO config is correctly defined with the expected outputs."
                    )
                output_name = output_names[0]
                filtered_outputs[output_name] = outputs

            if filtered_outputs.get("past_key_values") is not None:
                filtered_outputs["past_key_values"] = postprocess_past_key_values(filtered_outputs["past_key_values"])

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
