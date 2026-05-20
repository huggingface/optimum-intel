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
Create a portable DFlash correctness fixture.

This script is standalone: run it on any machine that can load the PyTorch
target and DFlash draft models, then copy the resulting `.pt` bundle to the
machine that runs the OpenVINO export tests.

Required packages:
  - torch
  - transformers >= 4.57
  - accelerate, if using --device-map auto
  - safetensors
  - huggingface_hub

Example:
  python tests/scripts/extract_dflash_debug_bundle.py \
    --draft-model z-lab/Qwen3-Coder-30B-A3B-DFlash \
    --target-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --dtype float16 \
    --num-steps 4 \
    --output dflash_debug_bundle_kv.pt
"""

import argparse

import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logits / temperature).sample()


def extract_context_feature(hidden_states: list[torch.Tensor], layer_ids: list[int]) -> torch.Tensor:
    # hidden_states[0] is the embedding output, so model layer ids are offset by one.
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


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


def crop_tensor(tensor: torch.Tensor, length: int) -> torch.Tensor:
    return tensor[..., :length, :].contiguous()


def crop_cache(cache, length: int):
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    return tuple((crop_tensor(key, length), crop_tensor(value, length)) for key, value in cache_to_layers(cache))


def cache_to_cpu(cache) -> tuple:
    return tuple(
        (key.detach().cpu().contiguous(), value.detach().cpu().contiguous()) for key, value in cache_to_layers(cache)
    )


def original_outputs_to_hidden_and_cache(outputs, fallback_cache=None):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state, outputs.past_key_values
    if isinstance(outputs, tuple):
        if len(outputs) > 1:
            return outputs[0], outputs[1]
        if fallback_cache is not None:
            return outputs[0], fallback_cache
    if torch.is_tensor(outputs) and fallback_cache is not None:
        return outputs, fallback_cache
    raise TypeError("Original DFlash model did not return a cache. Make sure use_cache=True is supported.")


def run_dflash_block(
    draft,
    target,
    block_input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    target_device = block_input_ids.device
    noise_embedding = target.model.embed_tokens(block_input_ids.to(target_device)).to(next(draft.parameters()).device)
    draft_hidden = draft(
        target_hidden=hidden_states.to(next(draft.parameters()).device),
        noise_embedding=noise_embedding,
        position_ids=position_ids.to(next(draft.parameters()).device),
        use_cache=False,
        is_causal=False,
    )
    return target.lm_head(draft_hidden[:, -block_size + 1 :, :].to(target_device))


def run_dflash_cached_block(
    draft,
    target,
    block_input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values,
    block_size: int,
) -> tuple[torch.Tensor, object]:
    target_device = block_input_ids.device
    draft_device = next(draft.parameters()).device
    if past_key_values is None:
        past_key_values = DynamicCache(config=draft.config)
    noise_embedding = target.model.embed_tokens(block_input_ids.to(target_device)).to(draft_device)
    outputs = draft(
        target_hidden=hidden_states.to(draft_device),
        noise_embedding=noise_embedding,
        position_ids=position_ids.to(draft_device),
        past_key_values=past_key_values,
        use_cache=True,
        is_causal=False,
    )
    draft_hidden, past_key_values = original_outputs_to_hidden_and_cache(outputs, fallback_cache=past_key_values)
    logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :].to(target_device))
    return logits, past_key_values


def main():
    parser = argparse.ArgumentParser(description="Create a lightweight DFlash correctness fixture.")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-Coder-30B-A3B-DFlash")
    parser.add_argument("--target-model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--prompt", default="Write a quicksort in Python.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=4)
    args = parser.parse_args()
    if args.num_steps < 1:
        raise ValueError("--num-steps must be at least 1")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()
    draft = AutoModel.from_pretrained(
        args.draft_model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    device = next(target.parameters()).device
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    block_size = draft.config.block_size
    mask_token_id = draft.config.dflash_config["mask_token_id"]
    target_layer_ids = draft.config.dflash_config["target_layer_ids"]

    steps = []
    committed_input_ids = input_ids
    committed_hidden_length = 0
    original_past_key_values = None
    with torch.inference_mode():
        for step_idx in range(args.num_steps):
            committed_length = committed_input_ids.shape[1]
            target_position_ids = torch.arange(committed_length, device=device).unsqueeze(0)
            target_output = target(
                committed_input_ids,
                position_ids=target_position_ids,
                use_cache=False,
                logits_to_keep=1,
                output_hidden_states=True,
            )
            seed_token = sample(target_output.logits[:, -1, :], args.temperature)

            block_input_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
            block_input_ids[:, 0] = seed_token

            full_hidden_states = extract_context_feature(target_output.hidden_states, target_layer_ids)
            hidden_states = full_hidden_states[:, committed_hidden_length:, :]
            position_start = committed_hidden_length
            position_ids = torch.arange(position_start, committed_length + block_size, device=device).unsqueeze(0)
            full_position_ids = torch.arange(committed_length + block_size, device=device).unsqueeze(0)
            expected_logits = run_dflash_block(
                draft,
                target,
                block_input_ids,
                full_hidden_states,
                full_position_ids,
                block_size,
            )
            original_cached_logits, original_past_key_values = run_dflash_cached_block(
                draft,
                target,
                block_input_ids,
                hidden_states,
                position_ids,
                original_past_key_values,
                block_size,
            )

            committed_hidden_length += hidden_states.shape[1]
            original_past_key_values = crop_cache(original_past_key_values, committed_hidden_length)
            steps.append(
                {
                    "index": step_idx,
                    "input_ids": block_input_ids.cpu(),
                    "hidden_states": hidden_states.cpu(),
                    "position_ids": position_ids.cpu(),
                    "expected_logits": original_cached_logits.cpu(),
                    "expected_logits_full_prefix": expected_logits.cpu(),
                    "original_cached_logits": original_cached_logits.cpu(),
                    "original_committed_cache": cache_to_cpu(original_past_key_values),
                    "seed_token": seed_token.cpu(),
                    "sampled_tokens": sample(expected_logits, args.temperature).cpu(),
                    "expected_present_length": committed_hidden_length,
                }
            )
            committed_input_ids = torch.cat([committed_input_ids, seed_token[:, None]], dim=1)

    bundle = {
        "steps": steps,
        # Keep first-step keys for quick ad-hoc inspection and older local scripts.
        "input_ids": steps[0]["input_ids"],
        "hidden_states": steps[0]["hidden_states"],
        "position_ids": steps[0]["position_ids"],
        "expected_logits": steps[0]["expected_logits"],
        "sampled_tokens": steps[0]["sampled_tokens"],
        "expected_present_length": steps[0]["expected_present_length"],
        "metadata": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "prompt": args.prompt,
            "num_steps": args.num_steps,
            "block_size": block_size,
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layer_ids,
            "dtype": args.dtype,
            "temperature": args.temperature,
            "cache_policy": "committed_prefix",
        },
    }
    torch.save(bundle, args.output)


if __name__ == "__main__":
    main()
