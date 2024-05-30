#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2024, HuggingFace. INTEL CORPORATION.  All rights reserved.
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


import json
import logging
import os
import sys
from pathlib import Path

import deepspeed
import deepspeed.comm as dist
import intel_extension_for_pytorch as ipex
import torch
from deepspeed.accelerator import get_accelerator
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils import is_offline_mode

from optimum.intel import IPEXModelForCausalLM


sys.path.append(sys.path[0] + "/../../")
logger = logging.getLogger(__name__)


torch._C._jit_set_texpr_fuser_enabled(False)
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID"], "0")
world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE"], "1")

deepspeed.init_distributed(get_accelerator().communication_backend_name())


def print_rank0(*msg):
    if local_rank != 0:
        return
    print(*msg)


# Model loading and instantiating on GPUs
def get_repo_root(model_name_or_path):
    if os.path.exists(model_name_or_path):
        # local path
        return model_name_or_path
    # checks if online or not
    if is_offline_mode():
        print_rank0("Offline mode: forcing local_files_only=True")
    # download only on first process
    allow_patterns = ["*.bin", "*.model", "*.json", "*.txt", "*.py", "*LICENSE"]
    if local_rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            # ignore_patterns=["*.safetensors"],
        )

    dist.barrier()

    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        allow_patterns=allow_patterns,
        # ignore_patterns=["*.safetensors"],
    )


def get_checkpoint_files(model_name_or_path):
    cached_repo_dir = get_repo_root(model_name_or_path)

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)

model = model.eval()
model = model.to(memory_format=torch.channels_last)

checkpoints_json = "checkpoints.json"


def write_checkpoints_json():
    checkpoint_files = get_checkpoint_files(model_name)
    if local_rank == 0:
        # model.config.model_type.upper()
        data = {"type": "ds_model", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, open(checkpoints_json, "w"))


repo_root = get_repo_root(model_name)

write_checkpoints_json()
dist.barrier()

print(model)
model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    base_dir=repo_root,
    dtype=torch.bfloat16,
    checkpoint=checkpoints_json,
    replace_with_kernel_inject=False,
)
print(model)
print(model.module)
model = model.module
model = IPEXModelForCausalLM._from_model(model.eval())

input_tokens = tokenizer.batch_encode_plus(
    ["This is an example input"], return_token_type_ids=False, return_tensors="pt"
)
input_ids = input_tokens.input_ids

output = model.generate(**input_tokens)

print(tokenizer.batch_decode(output, skip_special_tokens=True))
