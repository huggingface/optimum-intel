#  Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""OpenVINO export helpers for the ResembleAI Chatterbox TTS model.

Chatterbox is a multi-stage text-to-speech pipeline. For OpenVINO inference it is
decomposed into three exported submodels plus a set of precomputed assets:

* ``openvino_t3``      -- the token-to-token model: a Llama backbone with a speech-token
  head, exported as a stateful decoder that consumes ``inputs_embeds`` (the speech/text
  embedding tables and the conditioning prefix are precomputed assets applied in Python).
* ``openvino_flow``    -- the whole S3Gen flow (conformer encoder + Euler ODE solver +
  flow-matching estimator), exported as a single graph with the diffusion noise as an
  explicit input so that generation is deterministic and loop-free.
* ``openvino_hifigan`` -- the HiFiGAN vocoder turning mel-spectrograms into a waveform.
  ``torch.istft`` is replaced by a traceable overlap-add inverse STFT.

The assets (``chatterbox_assets.safetensors`` + ``chatterbox_config.json``) carry the
embedding tables, the built-in voice conditioning prefix and the S3Gen reference dict.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from optimum.exporters.openvino.base import OpenVINOConfig
from optimum.exporters.openvino.patching_utils import ModelPatcher
from optimum.utils.input_generators import DummyInputGenerator
from optimum.utils.normalized_config import NormalizedConfig


logger = logging.getLogger(__name__)


# Speech tokens >= SPEECH_VOCAB_SIZE are special tokens that must not reach the vocoder.
SPEECH_VOCAB_SIZE = 6561


def manual_istft(
    real: torch.Tensor, img: torch.Tensor, n_fft: int, hop_len: int, window: torch.Tensor
) -> torch.Tensor:
    """Traceable inverse STFT via ``irfft`` + overlap-add.

    Mirrors ``torch.istft`` (center=True, Hann window) but only uses operations that the
    OpenVINO PyTorch frontend can convert. ``real``/``img`` have shape ``[B, n_fft//2+1, T]``.
    """
    complex_spec = torch.complex(real, img)
    frames = torch.fft.irfft(complex_spec, n=n_fft, dim=1)  # [B, n_fft, T]
    frames = frames * window.view(1, -1, 1)
    eye = torch.eye(n_fft, device=frames.device, dtype=frames.dtype).unsqueeze(1)  # [n_fft, 1, n_fft]
    out = F.conv_transpose1d(frames, eye, stride=hop_len)  # [B, 1, L]
    win_sq = (window**2).view(1, -1, 1).expand(1, -1, frames.shape[-1])
    norm = F.conv_transpose1d(win_sq, eye, stride=hop_len)
    out = out / (norm + 1e-8)
    pad = n_fft // 2
    return out[:, 0, pad:-pad] if pad > 0 else out[:, 0, :]


class ChatterboxFlowWrapper(torch.nn.Module):
    """Wraps the S3Gen flow (``CausalMaskedDiffWithXvec``) into a single traceable graph.

    The diffusion noise is provided as an explicit input and the number of ODE timesteps is
    fixed, so the Euler solver is unrolled at export time.
    """

    def __init__(self, flow: torch.nn.Module, n_timesteps: int = 10):
        super().__init__()
        self.flow = flow
        self.n_timesteps = n_timesteps
        cfm = flow.decoder

        # Replace the noise-sampling forward with one that consumes injected noise.
        def cfm_forward(
            mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, noised_mels=None, meanflow=False
        ):
            z = getattr(cfm, "_injected_noise", None)
            if z is None:
                z = torch.randn_like(mu)
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
            if cfm.t_scheduler == "cosine":
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
            return cfm.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, meanflow=False), None

        cfm.forward = cfm_forward

    def forward(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, embedding, noise):
        self.flow.decoder._injected_noise = noise
        feat, _ = self.flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=None,
            embedding=embedding,
            finalize=True,
            n_timesteps=self.n_timesteps,
        )
        return feat


class ChatterboxHifiganWrapper(torch.nn.Module):
    """Wraps the HiFiGAN vocoder, replacing ``torch.istft`` with a traceable implementation."""

    def __init__(self, hift: torch.nn.Module):
        super().__init__()
        self.hift = hift
        n_fft = hift.istft_params["n_fft"]
        hop = hift.istft_params["hop_len"]
        window = hift.stft_window

        def patched_istft(magnitude, phase):
            magnitude = torch.clip(magnitude, max=1e2)
            real = magnitude * torch.cos(phase)
            img = magnitude * torch.sin(phase)
            return manual_istft(real, img, n_fft, hop, window.to(magnitude.device))

        hift._istft = patched_istft

    def forward(self, speech_feat):
        wav, _ = self.hift.inference(speech_feat=speech_feat)
        return wav


class DummyChatterboxFlowInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        "token",
        "token_len",
        "prompt_token",
        "prompt_token_len",
        "prompt_feat",
        "embedding",
        "noise",
    )

    def __init__(self, task: str, normalized_config: NormalizedConfig, **kwargs):
        self.task = task
        self.n_mels = getattr(normalized_config, "n_mels", 80)
        self.token_mel_ratio = getattr(normalized_config, "token_mel_ratio", 2)
        self.prompt_token_len = getattr(normalized_config, "prompt_token_len", 157)
        self.prompt_feat_len = getattr(normalized_config, "prompt_feat_len", 314)
        self.speaker_embedding_dim = getattr(normalized_config, "speaker_embedding_dim", 192)
        self.seq_len = 64

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        total_tokens = self.prompt_token_len + self.seq_len
        mel_t = self.token_mel_ratio * total_tokens
        if input_name == "token":
            return self.random_float_tensor([1, self.seq_len], min_value=0, max_value=SPEECH_VOCAB_SIZE - 1)
        if input_name == "token_len":
            return torch.tensor([self.seq_len], dtype=torch.float32)
        if input_name == "prompt_token":
            return self.random_float_tensor([1, self.prompt_token_len], min_value=0, max_value=SPEECH_VOCAB_SIZE - 1)
        if input_name == "prompt_token_len":
            return torch.tensor([self.prompt_token_len], dtype=torch.float32)
        if input_name == "prompt_feat":
            return self.random_float_tensor([1, self.prompt_feat_len, self.n_mels])
        if input_name == "embedding":
            return self.random_float_tensor([1, self.speaker_embedding_dim])
        if input_name == "noise":
            return self.random_float_tensor([1, self.n_mels, mel_t])
        raise ValueError(f"Unsupported input {input_name} for DummyChatterboxFlowInputGenerator")


def _make_pad_mask_dynamic(lengths, max_len: int = 0):
    """Drop-in for the S3Gen ``make_pad_mask`` that keeps the mask length dynamic.

    The original implementation calls ``lengths.max().item()`` which bakes the sequence
    length into the traced graph as a constant. Deriving the range length from a tensor
    instead keeps the exported model valid for arbitrary token lengths.
    """
    lengths = lengths.long()
    batch_size = lengths.size(0)
    range_len = max_len if max_len > 0 else lengths.max()
    seq_range = torch.arange(0, range_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, seq_range.shape[0])
    return seq_range_expand >= lengths.unsqueeze(-1)


class ChatterboxFlowModelPatcher(ModelPatcher):
    """Patches non-traceable ops used by the S3Gen flow during export.

    Two substitutions are applied for the duration of the export:

    * ``torch.atleast_2d`` -> a ``reshape`` for 1D tensors (no OpenVINO conversion rule).
    * ``make_pad_mask`` -> a variant whose mask length is derived from a tensor rather than
      a Python ``int``, so the exported flow generalizes to arbitrary token lengths.
    """

    def __enter__(self):
        super().__enter__()
        self._orig_atleast_2d = torch.atleast_2d
        torch.atleast_2d = lambda t: t.reshape(1, -1) if (torch.is_tensor(t) and t.ndim == 1) else t

        import chatterbox.models.s3gen.flow as flow_module
        import chatterbox.models.s3gen.utils.mask as mask_module

        self._mask_modules = [mask_module, flow_module]
        self._orig_make_pad_mask = mask_module.make_pad_mask
        for mod in self._mask_modules:
            if hasattr(mod, "make_pad_mask"):
                mod.make_pad_mask = _make_pad_mask_dynamic

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        torch.atleast_2d = self._orig_atleast_2d
        for mod in self._mask_modules:
            if hasattr(mod, "make_pad_mask"):
                mod.make_pad_mask = self._orig_make_pad_mask


class ChatterboxFlowOpenVINOConfig(OpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyChatterboxFlowInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    _MODEL_PATCHER = ChatterboxFlowModelPatcher

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "token": {1: "token_length"},
            "token_len": {},
            "prompt_token": {1: "prompt_token_length"},
            "prompt_token_len": {},
            "prompt_feat": {1: "prompt_feat_length"},
            "embedding": {},
            "noise": {2: "mel_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {"mel": {0: "batch_size", 2: "mel_out_length"}}


class DummyChatterboxHifiganInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("speech_feat",)

    def __init__(self, task: str, normalized_config: NormalizedConfig, **kwargs):
        self.task = task
        self.n_mels = getattr(normalized_config, "n_mels", 80)
        self.seq_len = 114

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "speech_feat":
            return self.random_float_tensor([1, self.n_mels, self.seq_len])
        raise ValueError(f"Unsupported input {input_name} for DummyChatterboxHifiganInputGenerator")


class ChatterboxHifiganOpenVINOConfig(OpenVINOConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyChatterboxHifiganInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedConfig

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"speech_feat": {0: "batch_size", 2: "mel_length"}}

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {"waveform": {0: "batch_size", 1: "audio_length"}}


def build_chatterbox_t3_for_export(tts):
    """Reconstruct a ``LlamaForCausalLM`` from the Chatterbox T3 backbone and speech head.

    Inference is driven by ``inputs_embeds``, so the (unused) ``embed_tokens`` layer keeps
    its original vocabulary while the LM head is replaced by the speech-token head.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    t3 = tts.t3
    speech_vocab = t3.hp.speech_tokens_dict_size
    llama_cfg = LlamaConfig(**t3.cfg.to_dict())
    lm = LlamaForCausalLM(llama_cfg)
    lm.model.load_state_dict(t3.tfmr.state_dict(), strict=True)
    lm.lm_head = torch.nn.Linear(llama_cfg.hidden_size, speech_vocab, bias=t3.speech_head.bias is not None)
    with torch.no_grad():
        lm.lm_head.weight.copy_(t3.speech_head.weight)
        if t3.speech_head.bias is not None:
            lm.lm_head.bias.copy_(t3.speech_head.bias)
    lm.config.vocab_size = speech_vocab
    lm.eval()
    return lm


def get_t3_export_config(lm):
    """Build the inputs_embeds stateful decoder export config for the T3 Llama backbone."""
    from optimum.exporters.tasks import TasksManager

    from .model_configs import LMInputEmbedsConfigHelper

    task = "text-generation-with-past"
    export_config_class = TasksManager._SUPPORTED_MODEL_TYPE["llama"]["openvino"][task]
    base_config = export_config_class(lm.config, task=task, use_past=True, use_past_in_inputs=True)
    return LMInputEmbedsConfigHelper(base_config)


def get_chatterbox_models_for_export(model, n_cfm_timesteps: int = 10):
    """Return ``{name: (submodel, export_config)}`` and the per-submodel statefulness list.

    ``model`` is the ``ChatterboxTTS`` object loaded by ``_ChatterboxForTextToSpeech``.
    """
    config = model.config

    lm = build_chatterbox_t3_for_export(model)
    t3_config = get_t3_export_config(lm)

    flow_wrapper = ChatterboxFlowWrapper(model.s3gen.flow, n_timesteps=n_cfm_timesteps).eval()
    flow_config = ChatterboxFlowOpenVINOConfig(config, task="text-to-audio")

    hifigan_wrapper = ChatterboxHifiganWrapper(model.s3gen.mel2wav).eval()
    hifigan_config = ChatterboxHifiganOpenVINOConfig(config, task="text-to-audio")

    models_and_export_configs = {
        "t3": (lm, t3_config),
        "flow": (flow_wrapper, flow_config),
        "hifigan": (hifigan_wrapper, hifigan_config),
    }
    # Only the autoregressive T3 backbone benefits from a stateful (kv-cache) export.
    stateful_submodels = [True, False, False]
    return models_and_export_configs, stateful_submodels


def save_chatterbox_assets(model, output: Path, n_cfm_timesteps: int = 10):
    """Save embedding tables, the built-in voice conditioning and config needed at inference."""
    from safetensors.torch import save_file

    t3 = model.t3
    config = model.config

    with torch.inference_mode():
        cond_prefix_emb = t3.prepare_conditioning(model.conds.t3).detach().clone()

    gen = model.conds.gen
    assets = {
        "speech_emb_weight": t3.speech_emb.weight.detach().clone(),
        "speech_pos_emb_weight": t3.speech_pos_emb.emb.weight.detach().clone(),
        "text_emb_weight": t3.text_emb.weight.detach().clone(),
        "text_pos_emb_weight": t3.text_pos_emb.emb.weight.detach().clone(),
        "cond_prefix_emb": cond_prefix_emb,
        "gen_prompt_token": gen["prompt_token"].detach().to(torch.float32).clone(),
        "gen_prompt_token_len": gen["prompt_token_len"].detach().to(torch.float32).clone(),
        "gen_prompt_feat": gen["prompt_feat"].detach().to(torch.float32).clone(),
        "gen_embedding": gen["embedding"].detach().to(torch.float32).clone(),
    }
    assets = {k: v.contiguous() for k, v in assets.items()}
    save_file(assets, str(output / "chatterbox_assets.safetensors"))

    # Persist the inference metadata as a JSON config.
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    config_dict["n_cfm_timesteps"] = n_cfm_timesteps
    with open(output / "chatterbox_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Save the tokenizer (and Chinese Cangjie table for the multilingual model) so the
    # text front-end is available at inference time.
    ckpt_dir = getattr(model, "_chatterbox_ckpt_dir", None)
    if ckpt_dir is not None:
        import shutil

        tokenizer_file = getattr(config, "tokenizer_file", "tokenizer.json")
        tokenizer_files = [tokenizer_file]
        if getattr(config, "multilingual", False):
            tokenizer_files.append("Cangjie5_TC.json")
        for fname in tokenizer_files:
            src = Path(ckpt_dir) / fname
            if src.is_file():
                shutil.copy(src, output / fname)
