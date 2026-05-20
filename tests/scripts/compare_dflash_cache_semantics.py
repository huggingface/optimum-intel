#!/usr/bin/env python
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compare original DFlash bundle references with the Optimum patched export model.

The bundle is produced by extract_dflash_debug_bundle.py on a machine that can
load the full target model. It contains original DFlash logits and original
DFlash K/V caches cropped to the committed target-backed prefix. This helper
loads only the patched Optimum DFlash model, replays the same steps, and checks
that both logits and committed-prefix caches match.
"""

import argparse

import torch
from transformers import AutoConfig


def cache_to_layers(cache) -> tuple:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if hasattr(cache, "layers"):
        layers = []
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None:
                key = getattr(layer, "key_cache", None)
            if value is None:
                value = getattr(layer, "value_cache", None)
            layers.append((key, value))
        return tuple(layers)
    return tuple(cache)


def cache_seq_length(cache) -> int:
    layers = cache_to_layers(cache)
    if not layers:
        return 0
    return layers[0][0].shape[-2]


def cache_to_cpu(cache) -> tuple:
    return tuple(
        (key.detach().float().cpu().contiguous(), value.detach().float().cpu().contiguous())
        for key, value in cache_to_layers(cache)
    )


def assert_cache_close(actual_cache, expected_cache, *, step_idx: int, rtol: float, atol: float):
    actual_layers = cache_to_cpu(actual_cache)
    if len(actual_layers) != len(expected_cache):
        raise AssertionError(f"Step {step_idx}: layer count mismatch {len(actual_layers)} != {len(expected_cache)}")
    for layer_idx, ((actual_key, actual_value), (expected_key, expected_value)) in enumerate(
        zip(actual_layers, expected_cache)
    ):
        torch.testing.assert_close(
            actual_key,
            expected_key.float(),
            rtol=rtol,
            atol=atol,
            msg=f"Step {step_idx}, layer {layer_idx}: key cache mismatch",
        )
        torch.testing.assert_close(
            actual_value,
            expected_value.float(),
            rtol=rtol,
            atol=atol,
            msg=f"Step {step_idx}, layer {layer_idx}: value cache mismatch",
        )


def load_patched_dflash(draft_model: str, target_model: str, dtype: torch.dtype, device_map: str):
    from optimum.exporters.openvino.__main__ import update_config_for_dflash
    from optimum.exporters.openvino.model_patcher import Qwen3DFlashForCausalLM

    config = AutoConfig.from_pretrained(draft_model, trust_remote_code=True)
    config = update_config_for_dflash(config, dflash_target_model=target_model)
    return Qwen3DFlashForCausalLM.from_pretrained(
        draft_model,
        config=config,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()


def main():
    parser = argparse.ArgumentParser(description="Compare patched DFlash against original committed-cache bundle data.")
    parser.add_argument("--bundle", required=True, help="Path to a bundle produced by extract_dflash_debug_bundle.py")
    parser.add_argument("--draft-model", default=None, help="Override draft model ID/path from the bundle metadata.")
    parser.add_argument("--target-model", default=None, help="Override target model ID/path from the bundle metadata.")
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu")
    metadata = bundle.get("metadata", {})
    draft_model = args.draft_model or metadata.get("draft_model", "z-lab/Qwen3-Coder-30B-A3B-DFlash")
    target_model = args.target_model or metadata.get("target_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    dtype = getattr(torch, args.dtype or metadata.get("dtype", "bfloat16"))
    steps = bundle["steps"]

    patched_draft = load_patched_dflash(draft_model, target_model, dtype, args.device_map)
    patched_device = next(patched_draft.parameters()).device
    patched_past_key_values = None
    with torch.inference_mode():
        for step_idx, step in enumerate(steps):
            if "original_committed_cache" not in step:
                raise ValueError("Bundle is missing original_committed_cache. Regenerate it with the updated extractor.")
            original_logits = step.get("original_cached_logits", step["expected_logits"])
            patched_outputs = patched_draft(
                input_ids=step["input_ids"].to(patched_device),
                hidden_states=step["hidden_states"].to(patched_device),
                position_ids=step["position_ids"].to(patched_device),
                past_key_values=patched_past_key_values,
                use_cache=True,
            )
            patched_past_key_values = patched_outputs.past_key_values
            torch.testing.assert_close(
                patched_outputs.logits.detach().float().cpu(),
                original_logits.float(),
                rtol=args.rtol,
                atol=args.atol,
                msg=f"Step {step_idx}: logits mismatch",
            )
            assert_cache_close(
                patched_past_key_values,
                step["original_committed_cache"],
                step_idx=step_idx,
                rtol=args.rtol,
                atol=args.atol,
            )
            patched_length = cache_seq_length(patched_past_key_values)
            if patched_length != step["expected_present_length"]:
                raise AssertionError(
                    f"Step {step_idx}: patched cache length {patched_length} != {step['expected_present_length']}"
                )
            print(f"Step {step_idx}: patched logits and committed-prefix cache match original bundle reference")

    print("DFlash committed-prefix cache semantics match the original implementation.")


if __name__ == "__main__":
    main()
