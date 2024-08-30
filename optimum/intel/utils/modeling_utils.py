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

import logging
import math
import os
import platform
import re
from pathlib import Path
from typing import List, Optional, Union

import psutil
import torch
from huggingface_hub import HfApi, HfFolder

from optimum.exporters import TasksManager

from .import_utils import is_numa_available


MULTI_QUERY_ATTN_MODELS = {"gpt_bigcode"}

logger = logging.getLogger(__name__)


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
    model_path = Path(model_name_or_path) if not isinstance(model_name_or_path, Path) else model_name_or_path

    if isinstance(use_auth_token, bool):
        token = HfFolder().get_token()
    else:
        token = use_auth_token

    library_name = TasksManager.infer_library_from_model(
        str(model_name_or_path), subfolder=subfolder, revision=revision, token=token
    )
    if library_name == "diffusers":
        subfolder = os.path.join(subfolder, "unet")
    else:
        subfolder = subfolder or "."

    if model_path.is_dir():
        glob_pattern = subfolder + "/*"
        files = model_path.glob(glob_pattern)
        files = [p for p in files if re.search(pattern, str(p))]
    else:
        repo_files = map(Path, HfApi().list_repo_files(model_name_or_path, revision=revision, token=token))
        files = [Path(p) for p in repo_files if re.match(pattern, str(p)) and str(p.parent) == subfolder]

    return files


def replace_customized_linear_with_linear(model):
    """
    Replace custom linear to torch linear so ipex could recognize and replace them to ipex linear.
    """
    if isinstance(model, torch.jit.ScriptModule):
        return
    if not model.training:
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.Linear) and child.__class__.__name__ in [
                "FalconLinear",
                "Linear",
            ]:
                new_m = torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=False if child.bias is None else True,
                )
                new_m.weight = child.weight
                if child.bias is not None:
                    new_m.bias = child.bias
                setattr(model, child_name, new_m)
            else:
                replace_customized_linear_with_linear(child)


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def bind_cores_for_best_perf():
    """
    Set number of threads per rank, numa cpu affinity and numa memory binding if not already set for better OOB performance.
    Works for wold_size >= 1 and rank >= 1

    Example:
    .. code-block:: python

        from optimum.intel.ipex import IPEXModelForCausalLM
        from optimum.intel.utils.modeling_utils import bind_cores_for_best_perf

        bind_cores_for_best_perf()
        model = IPEXModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.bfloat16, export=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        input_sentence = ["tell me a story about a trip to the moon"]
        model_inputs = tokenizer(input_sentence, return_tensors="pt")
        generation_kwargs = dict(max_new_tokens=500)
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    Returns:
        None

    """
    if platform.system() != "Linux":
        logger.error("bind_cores_for_best_perf: OS not supported, this function can only be run on Linux systems.")
        raise OSError("bind_cores_for_best_perf: OS not supported, this function can only be run on Linux systems.")
    if not is_numa_available():
        logger.error("'numa' module not found")
        raise ImportError("'numa' module not found, install with 'pip install numa'")
    import numa

    local_size = get_int_from_env(
        ["LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
    )
    rank_id = get_int_from_env(
        ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
    )
    nodes = numa.get_max_node() + 1
    rank_per_node = math.ceil(local_size / nodes)
    num_cpus_per_nodes = int(psutil.cpu_count(logical=False) / nodes)
    node_id = int(rank_id / rank_per_node)
    rank_offset_per_node = rank_id % rank_per_node
    if os.getenv("OMP_NUM_THREADS") is None:
        num_cpus_per_rank = max(int(num_cpus_per_nodes / rank_per_node), 1)
        logger.info(f"Setting OMP_NUM_THREADS to {num_cpus_per_rank} for better performance")
    else:
        num_cpus_per_rank = int(os.getenv("OMP_NUM_THREADS"))
        logger.info(f"OMP_NUM_THREADS already set to  {num_cpus_per_rank}")
    if len(numa.get_membind()) == nodes:
        # if numa memory binding is not set, set it to the node where the rank is running
        numa.set_membind([node_id])

    torch.set_num_threads(num_cpus_per_rank)

    if len(numa.get_affinity(0)) == psutil.cpu_count(logical=True):
        # if numa affinity is unset (default value is set to all logical cores) set it to the physical cores assigned to the rank
        cpu_start = num_cpus_per_rank * rank_offset_per_node
        numa.set_affinity(
            0,
            list(numa.node_to_cpus(node_id))[cpu_start : cpu_start + num_cpus_per_rank],
        )
    logger.info(f"affinity={numa.get_affinity(0)}, membind = {numa.get_membind()}")
