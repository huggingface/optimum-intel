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

import argparse

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    return torch.distributions.Categorical(logits=logits / temperature).sample()


def extract_context_feature(hidden_states: list[torch.Tensor], layer_ids: list[int]) -> torch.Tensor:
    # hidden_states[0] is the embedding output, so model layer ids are offset by one.
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Create a lightweight DFlash correctness fixture.")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-Coder-30B-A3B-DFlash")
    parser.add_argument("--target-model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--prompt", default="Write a quicksort in Python.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

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

    position_ids = torch.arange(input_ids.shape[1] + block_size, device=device).unsqueeze(0)
    with torch.inference_mode():
        target_output = target(
            input_ids,
            position_ids=position_ids[:, : input_ids.shape[1]],
            use_cache=False,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        first_draft_token = sample(target_output.logits[:, -1, :], args.temperature)

        block_input_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        block_input_ids[:, 0] = first_draft_token

        target_hidden = extract_context_feature(target_output.hidden_states, target_layer_ids).to(next(draft.parameters()).device)
        noise_embedding = target.model.embed_tokens(block_input_ids.to(device)).to(next(draft.parameters()).device)
        draft_position_ids = position_ids.to(next(draft.parameters()).device)

        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids,
            use_cache=False,
            is_causal=False,
        )
        expected_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :].to(device))
        sampled_tokens = sample(expected_logits, args.temperature)

    bundle = {
        "input_ids": block_input_ids.cpu(),
        "target_hidden": target_hidden.cpu(),
        "position_ids": position_ids.cpu(),
        "expected_logits": expected_logits.cpu(),
        "sampled_tokens": sampled_tokens.cpu(),
        "metadata": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "prompt": args.prompt,
            "block_size": block_size,
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layer_ids,
            "dtype": args.dtype,
            "temperature": args.temperature,
        },
    }
    torch.save(bundle, args.output)


if __name__ == "__main__":
    main()
