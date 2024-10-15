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

import json
import logging
import math
import os
import platform
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.hf_api import file_exists
from transformers import CLIPConfig, PretrainedConfig, PreTrainedModel, TFPreTrainedModel

from optimum.exporters import TasksManager

from .import_utils import is_diffusers_available, is_numa_available, is_open_clip_available, is_psutil_available


if is_diffusers_available():
    from diffusers import DiffusionPipeline, ModelMixin


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

    library_name = infer_library_from_model(
        str(model_name_or_path), subfolder=subfolder, revision=revision, token=token
    )
    if library_name == "diffusers":
        subfolders = [os.path.join(subfolder, "unet"), os.path.join(subfolder, "transformer")]
    else:
        subfolders = [subfolder or "."]

    if model_path.is_dir():
        files = []
        for subfolder in subfolders:
            glob_pattern = subfolder + "/*"
            files_ = model_path.glob(glob_pattern)
            files_ = [p for p in files_ if re.search(pattern, str(p))]
            files.extend(files_)
    else:
        repo_files = map(Path, HfApi().list_repo_files(model_name_or_path, revision=revision, token=token))
        files = [Path(p) for p in repo_files if re.match(pattern, str(p)) and str(p.parent) in subfolders]

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
    if not is_psutil_available():
        logger.error("`psutil` module not found")
        raise ImportError("'psutil' module not found, install with 'pip install psutil'")
    import psutil

    if not is_numa_available():
        logger.error("'numa' module not found")
        raise ImportError("'numa' module not found, install with 'pip install py-libnuma'")
    import numa

    local_size = get_int_from_env(
        ["LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
    )
    rank_id = get_int_from_env(
        ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
    )
    nodes = numa.info.get_max_node() + 1
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
    if len(numa.memory.get_membind_nodes()) == nodes:
        # if numa memory binding is not set, set it to the node where the rank is running
        numa.memory.set_membind_nodes((node_id))

    torch.set_num_threads(num_cpus_per_rank)

    if len(numa.schedule.get_affinitive_cpus(0)) == psutil.cpu_count(logical=True):
        # if numa affinity is unset (default value is set to all logical cores) set it to the physical cores assigned to the rank
        cpu_start = num_cpus_per_rank * rank_offset_per_node
        numa.schedule.run_on_cpus(
            0,
            *(numa.info.node_to_cpus(node_id)[cpu_start : cpu_start + num_cpus_per_rank]),
        )

    logger.info(f"affinity={numa.schedule.get_affinitive_cpus(0)}, membind = {numa.memory.get_membind_nodes()}")


def _infer_library_from_model_name_or_path(
    model_name_or_path: Union[str, Path],
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
):
    all_files, _ = TasksManager.get_model_files(
        model_name_or_path, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
    )
    if "open_clip_config.json" in all_files or "open_clip_pytorch_model.bin" in all_files:
        library_name = "open_clip"
    else:
        library_name = TasksManager._infer_library_from_model_name_or_path(
            model_name_or_path=model_name_or_path, cache_dir=cache_dir
        )

    return library_name


def _infer_library_from_model_or_model_class(
    model: Union["PreTrainedModel", "TFPreTrainedModel", "ModelMixin", "DiffusionPipeline"],
    library_name: Optional[str] = None,
):
    if library_name is not None:
        return library_name
    if model.__module__.startswith("open_clip"):
        library_name = "open_clip"
    elif model.__module__.startswith("optimum"):
        # for wrapped models like timm in optimum.intel.openvino.modeling_timm
        library_name = TasksManager._infer_library_from_model_or_model_class(model=model.model)
    else:
        library_name = TasksManager._infer_library_from_model_or_model_class(model=model)

    return library_name


def infer_library_from_model(
    model: Union[str, "PreTrainedModel", "TFPreTrainedModel", "DiffusionPipeline", Type],
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
):
    if isinstance(model, str):
        library_name = _infer_library_from_model_name_or_path(
            model_name_or_path=model,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    elif type(model) == type:
        library_name = _infer_library_from_model_or_model_class(model_class=model)
    else:
        library_name = _infer_library_from_model_or_model_class(model=model)

    return library_name


def collect_open_clip_model_files(model_name_or_path):
    model_files = {}
    if os.path.isdir(model_name_or_path):
        for filename in glob(os.path.join(model_name_or_path, "**/*"), recursive=True):
            if filename.endswith(".bin"):
                model_files["model"] = filename
            if filename.endswith("open_clip_config.json"):
                model_files["open_clip_config"] = filename
    return model_files


class _OpenClipForZeroShotImageClassification(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

    @staticmethod
    def load_config_from_file(config_path: str, model_name: str):
        cfg = {}
        model_config = None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            model_cfg = cfg.get("model_cfg", cfg)

            model_config = CLIPConfig(
                text_config_dict=model_cfg["text_cfg"],
                vision_config_dict=model_cfg["vision_cfg"],
                **cfg,
            )
        except IOError as e:
            raise IOError(f"Could not load config for model {model_name}. {e}")

        return model_config, cfg

    @staticmethod
    def find_config_by_hub_url(model_path: str):
        import open_clip

        cfg = None
        for model_name, model_tag in open_clip.list_pretrained():
            cfg = open_clip.get_pretrained_cfg(model_name, model_tag)
            if model_path in cfg.get("hf_hub", ""):
                cfg = open_clip.get_model_config(model_name)
                break

        return cfg

    @staticmethod
    def create_open_clip_model(model_path: str, open_clip_config: Dict, dtype: str):
        import open_clip

        model_cfg = open_clip_config["model_cfg"] if open_clip_config.get("model_cfg", None) else open_clip_config
        model_cfg.pop("custom_text", False)

        if "multimodal_cfg" in model_cfg:
            model = open_clip.CoCa(**model_cfg, cast_dtype=open_clip.get_cast_dtype(dtype))
        else:
            model = open_clip.CustomTextCLIP(**model_cfg, cast_dtype=open_clip.get_cast_dtype(dtype))

        open_clip.load_checkpoint(model, model_path)

        if "preprocess_cfg" in open_clip_config:
            preprocess_cfg = open_clip.get_model_preprocess_cfg(model)
            for k, val in open_clip_config["preprocess_cfg"].items():
                preprocess_cfg[k] = val
            open_clip.set_model_preprocess_cfg(model, preprocess_cfg)

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        config: Optional["PretrainedConfig"] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        dtype: str = "f32",
        **kwargs,
    ):
        if not is_open_clip_available():
            raise ImportError(
                "To load a open_clip model, open_clip needs to be installed. Please install it with `pip install open-clip-torch`."
            )

        import open_clip

        if os.path.isdir(model_name_or_path):
            local_open_clip_model = collect_open_clip_model_files(model_name_or_path)
            if "model" in local_open_clip_model and "open_clip_config" in local_open_clip_model:
                model_config, config_as_dict = cls.load_config_from_file(
                    local_open_clip_model["open_clip_config"], model_name_or_path
                )

                model = cls.create_open_clip_model(local_open_clip_model["model"], config_as_dict, dtype)
                setattr(model, "config", model_config)

                if not getattr(model.config, "model_type", None) and not getattr(
                    model.config, "export_model_type", None
                ):
                    setattr(model.config, "model_type", "clip")
                    setattr(model.config, "export_model_type", "clip")

                return model
            else:
                raise IOError(
                    f"Fail to load open_clip model from path {model_name_or_path}. Folder should contains open_clip_config.json and open_clip_pytorch_model.bin"
                )
        else:
            try:
                model, _ = open_clip.create_model_from_pretrained(
                    f"hf-hub:{model_name_or_path}", cache_dir=cache_dir, force_custom_text=True
                )

                if not getattr(model, "config", None):
                    config_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename="open_clip_config.json",
                        subfolder=subfolder,
                        token=token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    model_config, config_as_dict = cls.load_config_from_file(config_path, model_name_or_path)
                    setattr(model, "config", model_config)

            except Exception:
                model_file_name = None
                if file_exists(model_name_or_path, "open_clip_pytorch_model.bin"):
                    model_file_name = "open_clip_pytorch_model.bin"
                elif file_exists(model_name_or_path, "pytorch_model.bin"):
                    model_file_name = "pytorch_model.bin"
                else:
                    raise IOError("no model found")

                model_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=model_file_name,
                    subfolder=subfolder,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )

                if file_exists(model_name_or_path, "open_clip_config.json"):
                    config_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename="open_clip_config.json",
                        subfolder=subfolder,
                        token=token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    model_config, config_as_dict = cls.load_config_from_file(config_path, model_name_or_path)
                else:
                    config_as_dict = cls.find_config_by_hub_url(model_name_or_path)

                    model_cfg = config_as_dict.get("model_cfg", config_as_dict)

                    model_config = CLIPConfig(
                        text_config_dict=model_cfg["text_cfg"],
                        vision_config_dict=model_cfg["vision_cfg"],
                        **config_as_dict,
                    )

                    config_path = os.path.join(Path(model_path).parent, "open_clip_config.json")
                    with open(config_path, "w") as fp:
                        json.dump(config_as_dict, fp, indent=4)

                model = cls.create_open_clip_model(model_path, config_as_dict, dtype)

                setattr(model, "config", model_config)

            if not getattr(model.config, "model_type", None) or not getattr(model.config, "export_model_type", None):
                setattr(model.config, "model_type", "clip")
                setattr(model.config, "export_model_type", "clip")

            return model
