#!/usr/bin/env python3
"""
Script to create tiny-random-MiniCPM-o-4_5 for CI testing.

This script creates a minimal MiniCPM-o-4_5 model with reduced dimensions
for use in integration tests.

Target HF Hub: optimum-intel-internal-testing/tiny-random-MiniCPM-o-4_5

Usage:
    python create_tiny_minicpmo_4_5.py [--push-to-hub]
"""

import argparse
import json
import tempfile
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def create_tiny_model(push_to_hub: bool = False):
    """Create a tiny MiniCPM-o-4_5 model for testing."""
    print("Loading base config from openbmb/MiniCPM-o-4_5...")

    config = AutoConfig.from_pretrained(
        "openbmb/MiniCPM-o-4_5",
        trust_remote_code=True,
    )

    config.hidden_size = 168
    config.intermediate_size = 16
    config.num_hidden_layers = 1
    config.num_attention_heads = 28
    config.num_key_value_heads = 4
    config.max_position_embeddings = 512
    config.vocab_size = 1000
    config.query_num = 64
    config.init_audio = False
    config.init_tts = False

    if hasattr(config, "vision_config"):
        config.vision_config.hidden_size = 8
        config.vision_config.intermediate_size = 8
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 1
        config.vision_config.image_size = 70

    print("Config:", json.dumps(config.to_dict(), indent=2, default=str)[:500])

    print("Creating tiny model...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tiny model parameters: {total_params:,}")

    if push_to_hub:
        repo_id = "optimum-intel-internal-testing/tiny-random-MiniCPM-o-4_5"
        print(f"Pushing to {repo_id}...")
        model.push_to_hub(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-o-4_5",
            trust_remote_code=True,
        )
        tokenizer.push_to_hub(repo_id)
        print("Done!")
    else:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            model.save_pretrained(tmpdir)
            print(f"Saved to {tmpdir}")
            print("Files:", sorted(p.name for p in Path(tmpdir).iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push to HuggingFace Hub (requires auth)",
    )
    args = parser.parse_args()
    create_tiny_model(push_to_hub=args.push_to_hub)
