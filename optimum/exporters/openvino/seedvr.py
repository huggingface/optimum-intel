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

import contextlib
import importlib
import json
import logging
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig

from optimum.exporters.openvino.model_configs import (
    SeedVR2NaDiTOpenVINOConfig,
    SeedVR2VAEDecoderOpenVINOConfig,
    SeedVR2VAEEncoderOpenVINOConfig,
)


logger = logging.getLogger(__name__)


SEEDVR2_REPO_IDS = {"ByteDance-Seed/SeedVR2-3B", "ByteDance-Seed/SeedVR2-7B"}


_SEEDVR2_3B_CONFIG = {
    "module": "models.dit_v2.nadit",
    "checkpoint_filename": "seedvr2_ema_3b.pth",
    "config_dir": "configs_3b",
    "model_kwargs": {
        "vid_in_channels": 33,
        "vid_out_channels": 16,
        "vid_dim": 2560,
        "vid_out_norm": "fusedrms",
        "txt_in_dim": 5120,
        "txt_in_norm": "fusedln",
        "txt_dim": 2560,
        "emb_dim": 15360,
        "heads": 20,
        "head_dim": 128,
        "expand_ratio": 4,
        "norm": "fusedrms",
        "norm_eps": 1.0e-5,
        "ada": "single",
        "qk_bias": False,
        "qk_norm": "fusedrms",
        "patch_size": (1, 2, 2),
        "num_layers": 32,
        "mm_layers": 10,
        "mlp_type": "swiglu",
        "msa_type": None,
        "block_type": "mmdit_sr",
        "window": (4, 3, 3),
        "window_method": ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
        "rope_type": "mmrope3d",
        "rope_dim": 128,
    },
}


_SEEDVR2_7B_CONFIG = {
    "module": "models.dit.nadit",
    "checkpoint_filename": "seedvr2_ema_7b.pth",
    "config_dir": "configs_7b",
    "model_kwargs": {
        "vid_in_channels": 33,
        "vid_out_channels": 16,
        "vid_dim": 3072,
        "txt_in_dim": 5120,
        "txt_dim": 3072,
        "emb_dim": 18432,
        "heads": 24,
        "head_dim": 128,
        "expand_ratio": 4,
        "norm": "fusedrms",
        "norm_eps": 1.0e-5,
        "ada": "single",
        "qk_bias": False,
        "qk_rope": True,
        "qk_norm": "fusedrms",
        "patch_size": (1, 2, 2),
        "num_layers": 36,
        "mm_layers": 10,
        "shared_mlp": False,
        "shared_qkv": False,
        "mlp_type": "normal",
        "block_type": "mmdit_sr",
        "window": (4, 3, 3),
        "window_method": ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 18,
    },
}


SEEDVR2_VARIANT_CONFIGS = {"3b": _SEEDVR2_3B_CONFIG, "7b": _SEEDVR2_7B_CONFIG}


_SEEDVR2_VAE_CONFIG = {
    "module": "models.video_vae_v3.modules.attn_video_vae",
    "checkpoint_filename": "ema_vae.pth",
    "model_kwargs": {
        "act_fn": "silu",
        "block_out_channels": (128, 256, 512, 512),
        "down_block_types": ("DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"),
        "in_channels": 3,
        "latent_channels": 16,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "slicing_sample_min_size": 4,
        "temporal_scale_num": 2,
        "inflation_mode": "pad",
        "up_block_types": ("UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D"),
        "spatial_downsample_factor": 8,
        "temporal_downsample_factor": 4,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "freeze_encoder": False,
    },
}


class SeedVR2VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.config = vae._seedvr_ov_config

    def forward(self, sample):
        outputs = self.vae.encode(sample)
        if hasattr(outputs, "latent"):
            latent = outputs.latent
        elif hasattr(outputs, "latent_dist"):
            latent = outputs.latent_dist.mode()
        elif getattr(outputs, "posterior", None) is not None:
            latent = outputs.posterior.sample()
        else:
            latent = outputs
        # The causal VAE collapses a single temporal frame; restore it like the reference vae_encode.
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        return latent


class SeedVR2VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.config = vae._seedvr_ov_config

    def forward(self, latent_sample):
        return self.vae.decode(latent_sample).sample


def _is_seedvr2_checkpoint_name(name: str) -> bool:
    checkpoint_name = Path(name).name
    return checkpoint_name.startswith("seedvr2_ema_") and checkpoint_name.endswith((".pth", ".safetensors"))


def is_seedvr2_model(model_name_or_path: Union[str, Path], all_files: Optional[list[str]] = None) -> bool:
    model_name = str(model_name_or_path).replace("\\", "/").lower()
    if model_name in {repo_id.lower() for repo_id in SEEDVR2_REPO_IDS}:
        return True
    if "seedvr2" in model_name and any(size in model_name for size in ("3b", "7b")):
        return True
    files = all_files
    if files is None and Path(model_name_or_path).is_dir():
        files = [str(path.relative_to(model_name_or_path)).replace("\\", "/") for path in Path(model_name_or_path).rglob("*")]
    return bool(files) and any(_is_seedvr2_checkpoint_name(name) for name in files)


def _detect_variant(model_name_or_path: Union[str, Path], checkpoint_filename: Optional[str] = None) -> str:
    marker = f"{model_name_or_path} {checkpoint_filename or ''}".lower()
    if "7b" in marker:
        return "7b"
    if "3b" in marker:
        return "3b"

    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        checkpoint_names = {path.name.lower() for path in model_path.glob("*.pth")}
        checkpoint_names.update(path.name.lower() for path in model_path.glob("*.safetensors"))
        checkpoint_names.update(path.name.lower() for path in (model_path / "ckpts").glob("*.pth"))
        checkpoint_names.update(path.name.lower() for path in (model_path / "ckpts").glob("*.safetensors"))
        if any("7b" in name for name in checkpoint_names):
            return "7b"
        if any("3b" in name for name in checkpoint_names):
            return "3b"

    raise ValueError(
        "Could not infer the SeedVR2 variant. Use a model id/path or checkpoint filename containing `3B` or `7B`, "
        "or pass `checkpoint_filename` in model_loading_kwargs. Repositories with multiple variants, such as "
        "`numz/SeedVR2_comfyUI`, require an explicit checkpoint filename."
    )


def _resolve_seedvr_source_path(seedvr_source_path: Optional[Union[str, Path]] = None) -> Path:
    candidates = []
    if seedvr_source_path is not None:
        candidates.append(Path(seedvr_source_path))
    for env_name in ("SEEDVR_SOURCE_PATH", "SEEDVR2_SOURCE_PATH"):
        env_path = os.environ.get(env_name)
        if env_path:
            candidates.append(Path(env_path))

    for base in (Path.cwd(), *Path.cwd().parents):
        candidates.extend([base / "SeedVR", base.parent / "SeedVR"])

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "models").is_dir() and (candidate / "common").is_dir():
            return candidate

    raise ImportError(
        "SeedVR2 conversion requires the SeedVR source checkout because the public model repos contain raw `.pth` "
        "weights but not the NaDiT Python architecture. Pass `model_loading_kwargs={'seedvr_source_path': '/path/to/SeedVR'}` "
        "to `main_export`, set `SEEDVR_SOURCE_PATH`, or run from a workspace containing a sibling `SeedVR` checkout."
    )


def _install_apex_normalization_fallback():
    try:
        from apex.normalization import FusedLayerNorm, FusedRMSNorm  # noqa: F401

        return None
    except ModuleNotFoundError:
        pass

    class SeedVRRMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1.0e-5, elementwise_affine=True, **kwargs):
            super().__init__()
            self.normalized_shape = (normalized_shape,)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            else:
                self.register_parameter("weight", None)

        def forward(self, hidden_states):
            variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
            if self.weight is not None:
                hidden_states = hidden_states * self.weight
            return hidden_states

    previous_apex = sys.modules.get("apex")
    previous_normalization = sys.modules.get("apex.normalization")

    apex_module = types.ModuleType("apex")
    normalization_module = types.ModuleType("apex.normalization")
    normalization_module.FusedLayerNorm = torch.nn.LayerNorm
    normalization_module.FusedRMSNorm = SeedVRRMSNorm
    apex_module.normalization = normalization_module
    sys.modules["apex"] = apex_module
    sys.modules["apex.normalization"] = normalization_module
    logger.info("Apex is not available; using torch LayerNorm/RMSNorm fallback for SeedVR2 export.")
    return previous_apex, previous_normalization


def _restore_apex_normalization_fallback(previous_modules):
    if previous_modules is None:
        return
    previous_apex, previous_normalization = previous_modules
    if previous_apex is None:
        sys.modules.pop("apex", None)
    else:
        sys.modules["apex"] = previous_apex
    if previous_normalization is None:
        sys.modules.pop("apex.normalization", None)
    else:
        sys.modules["apex.normalization"] = previous_normalization


def _install_flash_attn_fallback():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        return None
    except ModuleNotFoundError:
        pass

    previous_flash_attn = sys.modules.get("flash_attn")

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, **kwargs):
        import torch.nn.functional as F

        outputs = []
        batch_size = cu_seqlens_q.numel() - 1
        for batch_idx in range(batch_size):
            q_start = int(cu_seqlens_q[batch_idx].item())
            q_end = int(cu_seqlens_q[batch_idx + 1].item())
            k_start = int(cu_seqlens_k[batch_idx].item())
            k_end = int(cu_seqlens_k[batch_idx + 1].item())
            query = q[q_start:q_end].permute(1, 0, 2).unsqueeze(0).float()
            key = k[k_start:k_end].permute(1, 0, 2).unsqueeze(0).float()
            value = v[k_start:k_end].permute(1, 0, 2).unsqueeze(0).float()
            output = F.scaled_dot_product_attention(query, key, value)
            outputs.append(output.squeeze(0).permute(1, 0, 2).to(dtype=q.dtype))
        return torch.cat(outputs, dim=0)

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = flash_attn_module
    logger.info("flash-attn is not available; using a PyTorch attention fallback for SeedVR2 export.")
    return previous_flash_attn


def _restore_flash_attn_fallback(previous_flash_attn):
    if previous_flash_attn is None:
        sys.modules.pop("flash_attn", None)
    else:
        sys.modules["flash_attn"] = previous_flash_attn


@contextlib.contextmanager
def _seedvr_import_context(seedvr_source_path: Path):
    source_path = str(seedvr_source_path)
    old_cwd = Path.cwd()
    inserted = False
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
        inserted = True
    os.chdir(seedvr_source_path)
    apex_fallback_modules = _install_apex_normalization_fallback()
    flash_attn_fallback_module = _install_flash_attn_fallback()
    try:
        yield
    finally:
        _restore_flash_attn_fallback(flash_attn_fallback_module)
        _restore_apex_normalization_fallback(apex_fallback_modules)
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(source_path)
            except ValueError:
                pass


def _resolve_checkpoint_path(
    model_name_or_path: Union[str, Path],
    checkpoint_filename: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> Path:
    model_path = Path(model_name_or_path)
    if model_path.is_file():
        return model_path.resolve()
    if model_path.is_dir():
        candidates = [model_path / checkpoint_filename, model_path / "ckpts" / checkpoint_filename]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Could not find `{checkpoint_filename}` in `{model_path}` or `{model_path / 'ckpts'}`.")

    return Path(
        hf_hub_download(
            repo_id=str(model_name_or_path),
            filename=checkpoint_filename,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )
    )


def _default_vae_checkpoint_filename(model_name_or_path: Union[str, Path], checkpoint_filename: Optional[str]) -> str:
    marker = f"{model_name_or_path} {checkpoint_filename or ''}".lower()
    if "numz/seedvr2_comfyui" in marker or (checkpoint_filename or "").endswith(".safetensors"):
        return "ema_vae_fp16.safetensors"
    return _SEEDVR2_VAE_CONFIG["checkpoint_filename"]


def _load_seedvr_state_dict(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exception:
            raise ImportError(
                "Loading SeedVR2 `.safetensors` checkpoints requires `safetensors`. Install it with "
                "`pip install safetensors` or use the official `.pth` checkpoint."
            ) from exception

        state = load_file(checkpoint_path, device="cpu")
    else:
        try:
            state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu", mmap=True)

    if isinstance(state, dict) and not any(torch.is_tensor(value) for value in state.values()):
        for key in ("state_dict", "model", "module", "dit", "ema"):
            nested_state = state.get(key)
            if isinstance(nested_state, dict) and any(torch.is_tensor(value) for value in nested_state.values()):
                state = nested_state
                break

    if not isinstance(state, dict):
        raise ValueError(f"SeedVR2 checkpoint `{checkpoint_path}` did not contain a PyTorch state dict.")

    prefixes = ("module.", "model.", "dit.")
    stripped_state = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        stripped_state[key] = value
    return stripped_state


def _as_python_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _patch_seedvr_ada_modulation(modulation_module_name: str):
    """Rewrite ``AdaSingle.forward`` to gather modulation components from the flat embedding.

    SeedVR2's ada modulation reshapes ``emb`` to ``[b, dim, layers, 3]`` and unbinds the
    inner size-3 axis. The OpenVINO GPU plugin miscomputes that ``reshape -> select ->
    Tile -> inner-axis Split`` pattern when a large downstream MatMul is present. Gather
    the components before expanding them over tokens. For a single-sample export, retain
    SeedVR2's cache reuse but leave cached values as ``[1, dim]`` so downstream arithmetic
    broadcasts them without emitting GPU-problematic ``Tile`` nodes.
    """
    try:
        modulation_module = importlib.import_module(modulation_module_name)
    except ModuleNotFoundError:
        return
    ada_cls = getattr(modulation_module, "AdaSingle", None)
    if ada_cls is None:
        return
    expand_dims = modulation_module.expand_dims

    def patched_forward(self, hid, emb, layer, mode, cache=None, branch_tag="", hid_len=None):
        idx = self.layers.index(layer)
        num_layers = len(self.layers)
        dim = self.dim
        base = torch.arange(dim, device=emb.device, dtype=torch.long) * (num_layers * 3) + idx * 3

        def component(offset):
            comp = emb.index_select(-1, base + offset)  # [b, dim]
            comp = expand_dims(comp, 1, hid.ndim)
            if hid_len is not None:
                comp = cache(
                    f"emb_repeat_{idx}_{branch_tag}_{offset}",
                    lambda: comp,
                )
            return comp

        shiftA, scaleA, gateA = component(0), component(1), component(2)
        shiftB = getattr(self, f"{layer}_shift", None)
        scaleB = getattr(self, f"{layer}_scale", None)
        gateB = getattr(self, f"{layer}_gate", None)

        if mode == "in":
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            return hid.mul_(gateA + gateB)
        raise NotImplementedError

    ada_cls.forward = patched_forward
    logger.info("Patched SeedVR2 AdaSingle.forward for GPU-safe modulation (%s).", modulation_module_name)


def _patch_seedvr_window_ops(window_module_name: str):
    try:
        window_module = importlib.import_module(window_module_name)
    except ModuleNotFoundError:
        return
    original_window = window_module.make_720Pwindows_bysize
    original_shifted_window = window_module.make_shifted_720Pwindows_bysize

    def make_720Pwindows_bysize(size, num_windows):
        t, h, w = size
        return original_window((_as_python_int(t), _as_python_int(h), _as_python_int(w)), num_windows)

    def make_shifted_720Pwindows_bysize(size, num_windows):
        t, h, w = size
        return original_shifted_window((_as_python_int(t), _as_python_int(h), _as_python_int(w)), num_windows)

    window_module.make_720Pwindows_bysize = make_720Pwindows_bysize
    window_module.make_shifted_720Pwindows_bysize = make_shifted_720Pwindows_bysize


def load_seedvr2_nadit_model(
    model_name_or_path: Union[str, Path],
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
    force_download: bool = False,
    seedvr_source_path: Optional[Union[str, Path]] = None,
    checkpoint_filename: Optional[str] = None,
    export_vae: bool = False,
    vae_checkpoint_filename: Optional[str] = None,
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
    **kwargs,
):
    local_model_path = Path(model_name_or_path)
    if local_model_path.exists():
        model_name_or_path = local_model_path.resolve()
    variant = _detect_variant(model_name_or_path, checkpoint_filename)
    variant_config = SEEDVR2_VARIANT_CONFIGS[variant]
    checkpoint_filename = checkpoint_filename or variant_config["checkpoint_filename"]
    seedvr_source_path = _resolve_seedvr_source_path(seedvr_source_path)
    checkpoint_path = _resolve_checkpoint_path(
        model_name_or_path,
        checkpoint_filename,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
        force_download=force_download,
    )

    model_kwargs = dict(variant_config["model_kwargs"])
    model_kwargs.update(kwargs.pop("model_config_overrides", {}) or {})

    vae = None
    with _seedvr_import_context(seedvr_source_path):
        try:
            module = __import__(variant_config["module"], fromlist=["NaDiT"])
            window_module_name = ".".join(variant_config["module"].split(".")[:-1] + ["window"])
            _patch_seedvr_window_ops(window_module_name)
            modulation_module_name = ".".join(variant_config["module"].split(".")[:-1] + ["modulation"])
            _patch_seedvr_ada_modulation(modulation_module_name)
            model = module.NaDiT(**model_kwargs)

            if export_vae:
                vae_module = __import__(_SEEDVR2_VAE_CONFIG["module"], fromlist=["VideoAutoencoderKLWrapper"])
                vae = vae_module.VideoAutoencoderKLWrapper(**_SEEDVR2_VAE_CONFIG["model_kwargs"])
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError(
                "Failed to import SeedVR2 NaDiT from the SeedVR source checkout. Make sure the SeedVR environment "
                "dependencies are installed; the official 3B/7B configs require packages such as diffusers, einops, "
                "and apex for fused normalization."
            ) from exception

        if hasattr(model, "set_gradient_checkpointing"):
            model.set_gradient_checkpointing(False)

        loading_info = model.load_state_dict(_load_seedvr_state_dict(checkpoint_path), strict=True)
        if vae is not None:
            vae_checkpoint_filename = vae_checkpoint_filename or _default_vae_checkpoint_filename(
                model_name_or_path, checkpoint_filename
            )
            vae_checkpoint_path = _resolve_checkpoint_path(
                model_name_or_path,
                vae_checkpoint_filename,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
                force_download=force_download,
            )
            vae_loading_info = vae.load_state_dict(_load_seedvr_state_dict(vae_checkpoint_path))
            vae.eval()

    if torch_dtype == "auto":
        torch_dtype = None
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)

    config = PretrainedConfig(
        model_type="seedvr2",
        export_model_type="seedvr2",
        architectures=["NaDiT"],
        seedvr_variant=variant,
        seedvr_checkpoint_filename=checkpoint_filename,
        seedvr_config_dir=variant_config["config_dir"],
        **model_kwargs,
    )
    model.config = config
    model._seedvr_model = True
    model._seedvr_repo_id = str(model_name_or_path)
    model._seedvr_source_path = str(seedvr_source_path)
    model._seedvr_checkpoint_path = str(checkpoint_path)
    if vae is not None:
        vae._seedvr_ov_config = PretrainedConfig(
            model_type="seedvr2-vae",
            export_model_type="seedvr2-vae",
            **_SEEDVR2_VAE_CONFIG["model_kwargs"],
        )
        vae._seedvr_model = True
        model.vae = vae
        model.config.seedvr_export_vae = True
        model.config.seedvr_vae_checkpoint_filename = vae_checkpoint_filename
        model.config.seedvr_vae_scaling_factor = 0.9152
        logger.info(f"Loaded SeedVR2 VAE checkpoint from {vae_checkpoint_path}: {vae_loading_info}")
    model.eval()
    logger.info(f"Loaded SeedVR2 {variant.upper()} NaDiT checkpoint from {checkpoint_path}: {loading_info}")
    return model


def get_seedvr2_models_for_export(model, task: str = "semantic-segmentation"):
    export_config = SeedVR2NaDiTOpenVINOConfig(model.config, task=task)
    models_and_export_configs = {"seedvr2_nadit": (model, export_config)}
    if hasattr(model, "vae"):
        vae_config = model.vae._seedvr_ov_config
        vae_encoder_config = SeedVR2VAEEncoderOpenVINOConfig(vae_config, task=task)
        vae_decoder_config = SeedVR2VAEDecoderOpenVINOConfig(vae_config, task=task)
        models_and_export_configs["seedvr2_vae_encoder"] = (SeedVR2VAEEncoderWrapper(model.vae), vae_encoder_config)
        models_and_export_configs["seedvr2_vae_decoder"] = (SeedVR2VAEDecoderWrapper(model.vae), vae_decoder_config)
    return export_config, models_and_export_configs, False


def save_seedvr2_config_and_assets(model, output: Union[str, Path]):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(output)
    copied_prompt_embeddings = []
    metadata = {
        "library_name": "seedvr",
        "model_type": "seedvr2",
        "seedvr_variant": getattr(model.config, "seedvr_variant", None),
        "seedvr_checkpoint_filename": getattr(model.config, "seedvr_checkpoint_filename", None),
        "seedvr_config_dir": getattr(model.config, "seedvr_config_dir", None),
        "seedvr_vae_checkpoint_filename": getattr(model.config, "seedvr_vae_checkpoint_filename", None),
        "seedvr_vae_scaling_factor": getattr(model.config, "seedvr_vae_scaling_factor", None),
        "prompt_embeddings": copied_prompt_embeddings,
        "exported_components": ["seedvr2_nadit"]
        + (["seedvr2_vae_encoder", "seedvr2_vae_decoder"] if hasattr(model, "vae") else []),
    }

    repo_path = Path(getattr(model, "_seedvr_repo_id", ""))
    source_path = Path(getattr(model, "_seedvr_source_path", ""))
    for file_name in ("pos_emb.pt", "neg_emb.pt"):
        candidates = []
        if repo_path.is_dir():
            candidates.append(repo_path / file_name)
        if source_path.is_dir():
            candidates.append(source_path / file_name)
        copied = False
        for candidate in candidates:
            if candidate.is_file():
                shutil.copy2(candidate, output / file_name)
                copied_prompt_embeddings.append(file_name)
                copied = True
                break
        if copied:
            continue
        repo_id = getattr(model, "_seedvr_repo_id", None)
        if repo_id and not Path(str(repo_id)).is_dir():
            try:
                repo_files = list_repo_files(str(repo_id))
                if file_name in repo_files:
                    downloaded = hf_hub_download(str(repo_id), file_name)
                    shutil.copy2(downloaded, output / file_name)
                    copied_prompt_embeddings.append(file_name)
            except (EntryNotFoundError, OSError, ValueError):
                pass

    with open(output / "seedvr2_export_config.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)