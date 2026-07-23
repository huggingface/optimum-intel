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

import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Sequence


logger = logging.getLogger(__name__)


def _is_rank3_hidden_output(output, hidden_size: int) -> bool:
    shape = output.get_partial_shape()
    if shape.rank.is_dynamic or len(shape) != 3:
        return False
    last_dim = shape[2]
    return bool(last_dim.is_static and int(last_dim.get_length()) == hidden_size)


def _discover_decoder_stack(source_model) -> Sequence[str]:
    """Return the one ordered decoder-layer module stack."""
    recorders = getattr(source_model, "can_record_outputs", {})
    recorder = recorders.get("hidden_states") if isinstance(recorders, dict) else None
    # Transformers accepts either a decoder-layer class directly or an OutputRecorder wrapper.
    decoder_layer_class = getattr(recorder, "target_class", recorder)
    if not isinstance(decoder_layer_class, type):
        raise ValueError("Transformers does not declare a decoder-layer class for hidden_states.")
    # Composite wrappers can contain multiple decoder stacks; validate one contiguous stack.
    stacks = defaultdict(dict)
    for module_name, module in source_model.named_modules():
        if not isinstance(module, decoder_layer_class):
            continue
        prefix, separator, raw_index = module_name.rpartition(".")
        if not separator or not raw_index.isdigit():
            continue
        stacks[prefix][int(raw_index)] = module_name

    configured_count = int(source_model.config.num_hidden_layers)
    candidates = {}
    for prefix, indexed_modules in stacks.items():
        indices = sorted(indexed_modules)
        if indices != list(range(configured_count)):
            continue
        candidates[prefix] = [indexed_modules[index] for index in indices]

    if len(candidates) != 1:
        raise ValueError(
            "Expected one contiguous decoder-layer stack, " f"found {len(candidates)}: {sorted(candidates)}."
        )
    return next(iter(candidates.values()))


def _infer_scope_prefix(ops: Iterable[Any], decoder_stack: str, num_layers: int) -> str:
    """Find the activation scope prefix that covers every decoder layer."""
    pattern = re.compile(rf"^(?P<prefix>__module(?:\.[^./]+)*)\.{re.escape(decoder_stack)}\.(?P<index>\d+)(?=[./]|$)")
    prefix_indices = defaultdict(set)
    for op in ops:
        match = pattern.match(op.get_friendly_name())
        if match is not None:
            prefix_indices[match.group("prefix") + "."].add(int(match.group("index")))

    expected_indices = set(range(num_layers))
    scope_prefixes = {
        scope_prefix for scope_prefix, indices in prefix_indices.items() if expected_indices.issubset(indices)
    }
    if len(scope_prefixes) != 1:
        raise ValueError(
            "Expected one OpenVINO activation scope for decoder stack "
            f"{decoder_stack!r}, found {sorted(scope_prefixes)}."
        )
    return next(iter(scope_prefixes))


def _boundary_output(
    ops: Iterable[Any],
    source_scope: str,
    target_scope: str,
    hidden_size: int,
):
    """Return the unique hidden-state value crossing from one decoder block to the next."""
    candidates = {}
    for producer in ops:
        producer_name = producer.get_friendly_name()
        # Consider only values produced by the current decoder block.
        if producer_name != source_scope and not producer_name.startswith((source_scope + ".", source_scope + "/")):
            continue
        for output in producer.outputs():
            # A block boundary is an output consumed by an operation in the next block.
            crosses_boundary = any(
                target_input.get_node().get_friendly_name() == target_scope
                or target_input.get_node().get_friendly_name().startswith((target_scope + ".", target_scope + "/"))
                for target_input in output.get_target_inputs()
            )
            if not crosses_boundary:
                continue
            # Ignore cache and attention values that do not have hidden-state shape.
            if _is_rank3_hidden_output(output, hidden_size):
                candidates[(producer_name, int(output.get_index()))] = output

    # Ambiguous boundaries are rejected by the caller.
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _lm_head_input(ops: Iterable[Any], lm_head_scope: str, hidden_size: int):
    candidates = {}
    for node in ops:
        name = node.get_friendly_name()
        if name != lm_head_scope and not name.startswith((lm_head_scope + ".", lm_head_scope + "/")):
            continue
        for input_port in node.inputs():
            output = input_port.get_source_output()
            if _is_rank3_hidden_output(output, hidden_size):
                candidates[(output.get_node().get_friendly_name(), int(output.get_index()))] = output
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _discover_locators_for_scope(
    ops: Iterable[Any],
    producer_counts,
    decoder_modules: Sequence[str],
    lm_head_module: str,
    hidden_size: int,
    scope_prefix: str,
):
    """Resolve one unambiguous producer/output locator for every decoder hidden state."""
    locators = {}
    for layer_id in range(len(decoder_modules) - 1):
        # Each non-final hidden state crosses from this block into the following block.
        output = _boundary_output(
            ops,
            scope_prefix + decoder_modules[layer_id],
            scope_prefix + decoder_modules[layer_id + 1],
            hidden_size,
        )
        # Friendly names are the persistent locator; duplicate producers are unsafe.
        if output is None or producer_counts[output.get_node().get_friendly_name()] != 1:
            return None
        locators[layer_id] = {
            "producer": output.get_node().get_friendly_name(),
            "output_index": int(output.get_index()),
        }

    # The final hidden state is the post-normalization value consumed by lm_head.
    final_output = _lm_head_input(ops, scope_prefix + lm_head_module, hidden_size)
    if final_output is None or producer_counts[final_output.get_node().get_friendly_name()] != 1:
        return None
    locators[len(decoder_modules) - 1] = {
        "producer": final_output.get_node().get_friendly_name(),
        "output_index": int(final_output.get_index()),
    }

    # No two semantic layers may point to the same producer output.
    identities = {(locator["producer"], locator["output_index"]) for locator in locators.values()}
    if len(identities) != len(locators):
        return None
    return locators


def discover_hidden_state_rt_info(source_model, ov_model) -> Dict[str, Any]:
    """
    Return hidden-state locator metadata without changing ``ov_model``.

    Non-final layers resolve to the hidden-width value crossing into the following
    decoder block. The final layer resolves to the hidden-width value consumed by
    the language-model head.
    """

    decoder_modules = _discover_decoder_stack(source_model)
    hidden_size = int(source_model.config.hidden_size)
    ops = ov_model.get_ordered_ops()
    scope_prefix = _infer_scope_prefix(ops, decoder_modules[0].rpartition(".")[0], len(decoder_modules))
    lm_heads = [name for name, _ in source_model.named_modules() if name.rsplit(".", 1)[-1] == "lm_head"]
    if len(lm_heads) != 1:
        raise ValueError(f"Expected one lm_head module, found {len(lm_heads)}.")

    producer_counts = Counter(op.get_friendly_name() for op in ops)
    locators = _discover_locators_for_scope(
        ops,
        producer_counts,
        decoder_modules,
        lm_heads[0],
        hidden_size,
        scope_prefix,
    )
    if locators is None:
        raise ValueError("Could not resolve unambiguous hidden-state boundaries.")

    return {
        "layers": {str(layer_id): locator for layer_id, locator in locators.items()},
    }


def add_hidden_states_rt_info(source_model, ov_model, config: Any):
    """Best-effort hidden-state locator annotation that leaves the graph untouched."""
    hidden_states_rt_info_key = "hidden_states_decoder_layers"

    if "text-generation" not in getattr(config, "task", ""):
        return

    try:
        annotation = discover_hidden_state_rt_info(source_model, ov_model)
        ov_model.set_rt_info(
            json.dumps(annotation, separators=(",", ":")),
            hidden_states_rt_info_key,
        )
    except Exception as error:
        # Discovery is intentionally read-only. On every failure, leave the
        # converted graph and public I/O exactly as the ordinary export produced.
        logger.warning(
            "Skipping hidden-state RT-info annotation; exporting the ordinary model without it: %s",
            error,
        )
