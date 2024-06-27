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
from typing import List, Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, HfFolder
from transformers.modeling_utils import PreTrainedModel


MULTI_QUERY_ATTN_MODELS = {"falcon", "gpt_bigcode"}


# Modified from transformers.models.bloom.modeling_bloom._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    device: torch.device,
    past_key_values_length: int,
    dtype: torch.dtype = torch.bool,
) -> torch.BoolTensor:
    """
    Make causal mask used for bi-directional self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.zeros((target_length, target_length + past_key_values_length), dtype=dtype, device=device)
    seq_ids = torch.arange(target_length, device=device)

    mask[:, past_key_values_length:] = (
        (seq_ids[:, None] < seq_ids[None, :]) * torch.finfo(dtype).min
        if torch.is_floating_point(mask)
        else seq_ids[:, None] < seq_ids[None, :]
    )

    return mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)


# Modified from transformers.models..bloom.modeling_bloom._prepare_attn_mask
def _prepare_attn_mask(
    attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
) -> torch.BoolTensor:
    from transformers.models.bloom.modeling_bloom import _expand_mask

    # create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape

    combined_attention_mask = _make_causal_mask(
        input_shape, device=device, past_key_values_length=past_key_values_length
    )
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]_prepare_decoder_attention_mask
    expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask


# Modified from transformers.models.llama.modeling_llama._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    from transformers.models.llama.modeling_llama import _expand_mask

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    combined_attention_mask = _make_causal_mask(
        input_shape,
        device=inputs_embeds.device,
        past_key_values_length=past_key_values_length,
        dtype=inputs_embeds.dtype,
    )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


# Modified from transformers.models.mistral.modeling_mistral._prepare_decoder_sliding_window_attention_mask
def _prepare_decoder_sliding_window_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int,
):
    from transformers.models.mistral.modeling_mistral import _expand_mask, _make_sliding_window_causal_mask

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    combined_attention_mask = _make_sliding_window_causal_mask(
        input_shape,
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
        past_key_values_length=past_key_values_length,
        sliding_window=sliding_window,
    )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def patch_decoder_attention_mask(model: "PreTrainedModel"):
    """
    Apply patch on decoder with past model forward to resolve first inference based on model architecture

    Args:
        model (PretrainedModel): The model to patch.

    Returns:
        model with applied patch
    """
    if model.config.model_type in {"bloom", "mpt"}:
        model.transformer._prepare_attn_mask = _prepare_attn_mask
    elif model.config.model_type == "llama":
        model.model._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    elif model.config.model_type == "mistral":
        model.model._prepare_decoder_attention_mask = _prepare_decoder_sliding_window_attention_mask
    elif model.config.model_type in {"blenderbot-small", "blenderbot", "opt", "pegasus", "bart"}:
        model.model.decoder._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    return model


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
