#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

import re
from pathlib import Path
from typing import List, Optional, Union

import torch
from huggingface_hub import HfApi, HfFolder


MULTI_QUERY_ATTN_MODELS = {"falcon", "gpt_bigcode"}


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Determines the device on which a PyTorch model is currently residing.

    Args:
        model: The PyTorch model to query.

    Returns:
        torch.device: The device where the model's parameters are located.

    Raises:
        StopIteration: If the model has no parameters.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # The model had no parameters at all, doesn't matter which device to choose
        device = torch.device("cpu")
    return device


def recursive_to_device(value, device):
    """
    Recursivley move the tensor element in `value` to `device`
    """
    if isinstance(value, (tuple, list)):
        return type(value)(recursive_to_device(v, device) for v in value)
    elif isinstance(value, dict):
        return {k: recursive_to_device(v, device) for k, v in value.items()}
    elif isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _setattr_from_module(new_module, module):
    for k, v in module.__dict__.items():
        setattr(new_module, k, v)
    for k, v in module.__class__.__dict__.items():
        if k.startswith("__") or k.startswith("forward"):
            continue
        setattr(new_module.__class__, k, getattr(module.__class__, k))


def _find_files_matching_pattern(
    model_name_or_path: Union[str, Path],
    pattern: str,
    subfolder: str = "",
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
) -> List[Path]:
    """
    Scans either a model repo or a local directory to find filenames matching the pattern.

    Args:
        model_name_or_path (`Union[str, Path]`):
            The name of the model repo on the Hugging Face Hub or the path to a local directory.
        pattern (`str`):
            The pattern to use to look for files.
        subfolder (`str`, defaults to `""`):
            In case the model files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        use_auth_token (`Optional[bool, str]`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`Optional[str]`, defaults to `None`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.

    Returns:
        `List[Path]`
    """
    model_path = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    pattern = re.compile(f"{subfolder}/{pattern}" if subfolder != "" else pattern)
    subfolder = subfolder or "."

    if model_path.is_dir():
        glob_pattern = subfolder + "/*"
        files = model_path.glob(glob_pattern)
        files = [p for p in files if re.search(pattern, str(p))]
    else:
        if isinstance(use_auth_token, bool):
            token = HfFolder().get_token()
        else:
            token = use_auth_token
        repo_files = map(Path, HfApi().list_repo_files(model_name_or_path, revision=revision, token=token))
        files = [Path(p) for p in repo_files if re.match(pattern, str(p)) and str(p.parent) == subfolder]

    return files

def get_number_of_sockets():
    """linux only"""
    try:
        sockets = set()
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('physical id'):
                    sockets.add(line.strip().split()[-1])
        return len(sockets)
    except Exception as e:
        print(f"Error retrieving number of sockets: {e}")

def bind_cores_for_best_perf():
    """
    In a multi-socker system binds CPU cores to single socket and numa node for better OOB performance.
    
    System configuration is equivalent than running the following command when launching the script:
    numactl -C '0-'${PHYSICAL_CORES_PER_SOCKET} --membind 0 python script.py

    Returns:
        None
    """
    
    import importlib.util
    import platform
    system = platform.system()
    if system == "Linux":
        if importlib.util.find_spec("numa") is not None:
            import numa
            import psutil
            import os

            nodes = numa.get_max_node() + 1
            n_sockets = get_number_of_sockets()
            if n_sockets != nodes:
                print(f'Warning: number of sockets {n_sockets} does not match number of NUMA nodes {nodes}.')
                print('Newer CPUs enable sub-numa cluster (SNC) but LLMs may show improved performance with SNC disabled in BIOS.')
            if os.getenv("OMP_NUM_THREADS") is None:
                # set OMP_NUM_THREADS to number of physical cores per socket
                num_cpu_threads_per_process = int(psutil.cpu_count(logical=True) / n_sockets)
                os.environ['OMP_NUM_THREADS'] = str(num_cpu_threads_per_process)
                print(f"OMP_NUM_THREADS/MKL_NUM_THREADS unset, we set it at {num_cpu_threads_per_process} to improve oob performance.")
            else:
                #do not override if OMP_NUM_THREADS already set
                num_cpu_threads_per_process = int(os.getenv("OMP_NUM_THREADS"))
            torch.set_num_threads(num_cpu_threads_per_process)

            # Bind the current process to the specified range of CPU cores
            numa.set_affinity(0, range(num_cpu_threads_per_process))
            # Check if the current memory binding policy includes all NUMA nodes
            if len(numa.get_membind()) == nodes:
                # Bind the process's memory allocation to the first NUMA node
                numa.set_membind([0])
        else:
            print("numa module not found, skipping binding cores")
    else:
        print("OS not supported, skipping binding cores")
