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

"""Helpers to build a tiny, randomly-initialized Chatterbox TTS model for tests.

The real ``ResembleAI/chatterbox`` checkpoint is ~2 GB, so the tests construct a small
variant at test time instead of downloading it: the T3 Llama backbone is reduced to a
couple of layers, the S3Gen flow estimator is shrunk, and the conditioning is synthetic.
The S3Gen conformer encoder and HiFiGAN vocoder keep their real (offline-built) shapes
because their internal dimensions are tightly coupled. Only the tokenizer file (small) is
fetched from the hub so the text front-end matches the real model.
"""

import contextlib
import copy

import torch


def _tiny_llama_config_patch():
    """Context manager that temporarily shrinks the T3 Llama backbone configuration."""
    from chatterbox.models.t3 import llama_configs

    @contextlib.contextmanager
    def _patch():
        original = llama_configs.LLAMA_CONFIGS["Llama_520M"]
        tiny = copy.deepcopy(original)
        # Keep hidden_size/head_dim intact (the Perceiver resampler and embedding tables
        # assume hidden_size=1024); only reduce depth and the MLP width.
        tiny.update(intermediate_size=256, num_hidden_layers=2)
        llama_configs.LLAMA_CONFIGS["Llama_520M"] = tiny
        try:
            yield
        finally:
            llama_configs.LLAMA_CONFIGS["Llama_520M"] = original

    return _patch()


@contextlib.contextmanager
def _tiny_s3gen_estimator_patch():
    """Temporarily shrink the S3Gen flow estimator (the heaviest exported S3Gen part)."""
    import chatterbox.models.s3gen.s3gen as s3gen_module

    original = s3gen_module.ConditionalDecoder

    def tiny_decoder(*args, **kwargs):
        kwargs.update(channels=[64], n_blocks=1, num_mid_blocks=2, num_heads=2, attention_head_dim=32)
        return original(*args, **kwargs)

    s3gen_module.ConditionalDecoder = tiny_decoder
    try:
        yield
    finally:
        s3gen_module.ConditionalDecoder = original


def build_tiny_chatterbox(tmp_dir, multilingual: bool = False, seed: int = 0):
    """Build a tiny random-weight Chatterbox model and return it ready for export.

    Args:
        tmp_dir: A directory used to stage the (small) tokenizer file.
        multilingual: Whether to build the multilingual variant (multilingual tokenizer +
            larger text vocabulary).
        seed: RNG seed for reproducible random weights.

    Returns:
        A ``ChatterboxTTS``-like object with a ``config`` and synthetic ``conds`` attached,
        suitable for ``optimum.exporters.openvino.export_from_model``.
    """
    from pathlib import Path

    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.tts import ChatterboxTTS, Conditionals
    from huggingface_hub import hf_hub_download

    from optimum.intel.utils.modeling_utils import _ChatterboxForTextToSpeech

    torch.manual_seed(seed)
    tmp_dir = Path(tmp_dir)

    with _tiny_llama_config_patch(), _tiny_s3gen_estimator_patch():
        hp = T3Config.multilingual() if multilingual else T3Config.english_only()
        t3 = T3(hp).eval()
        s3gen = S3Gen().eval()
        ve = VoiceEncoder().eval()

    # The text front-end must match the real model; fetch only the small tokenizer file.
    repo_id = "ResembleAI/chatterbox"
    tokenizer_file = "grapheme_mtl_merged_expanded_v1.json" if multilingual else "tokenizer.json"
    tokenizer_path = hf_hub_download(repo_id, tokenizer_file)
    if multilingual:
        from chatterbox.models.tokenizers import MTLTokenizer

        # The multilingual tokenizer loads the Cangjie table from the file's directory.
        hf_hub_download(repo_id, "Cangjie5_TC.json")
        tokenizer = MTLTokenizer(tokenizer_path)
    else:
        from chatterbox.models.tokenizers import EnTokenizer

        tokenizer = EnTokenizer(tokenizer_path)

    # Synthetic built-in voice conditioning (T3Cond + S3Gen reference dict).
    speaker_emb = torch.randn(1, hp.speaker_embed_size)
    cond_prompt = torch.randint(0, 100, (1, hp.speech_cond_prompt_len))
    t3_cond = T3Cond(
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=cond_prompt,
        emotion_adv=0.5 * torch.ones(1, 1, 1),
    )
    n_prompt = 8
    gen = {
        "prompt_token": torch.randint(0, 6561, (1, n_prompt)),
        "prompt_token_len": torch.tensor([n_prompt]),
        "prompt_feat": torch.randn(1, 2 * n_prompt, 80),
        "prompt_feat_len": None,
        "embedding": torch.randn(1, s3gen.flow.spk_embed_affine_layer.in_features),
    }

    tts = ChatterboxTTS(t3, s3gen, ve, tokenizer, device="cpu", conds=Conditionals(t3_cond, gen))
    tts.config = _ChatterboxForTextToSpeech._build_config(tts, multilingual=multilingual)
    tts._chatterbox_model = True

    # Stage the tokenizer files where the asset saver expects them.
    ckpt_dir = tmp_dir / "tiny_chatterbox_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(tokenizer_path, ckpt_dir / tokenizer_file)
    if multilingual:
        shutil.copy(hf_hub_download(repo_id, "Cangjie5_TC.json"), ckpt_dir / "Cangjie5_TC.json")
    tts._chatterbox_ckpt_dir = str(ckpt_dir)

    return tts
