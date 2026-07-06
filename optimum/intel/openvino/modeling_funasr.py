#  Copyright 2026 The HuggingFace Team. All rights reserved.
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
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import PretrainedConfig

from ..utils.import_utils import is_funasr_available


class _FunASRAudioEncoder(torch.nn.Module):
    """Wraps the FunASR audio encoder (SenseVoice) and audio adaptor as a single encoder module.

    Produces audio embeddings already projected to the LLM hidden size, which are spliced into
    the decoder input embeddings at the audio placeholder positions.
    """

    def __init__(self, audio_encoder: torch.nn.Module, audio_adaptor: torch.nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_adaptor = audio_adaptor

    def forward(self, input_features: "torch.Tensor"):
        # input_features: (batch, num_frames, feature_size)
        speech_lengths = torch.tensor([input_features.shape[1]] * input_features.shape[0], dtype=torch.int32)
        encoder_out, encoder_out_lens = self.audio_encoder(input_features, speech_lengths)
        adaptor_out, _ = self.audio_adaptor(encoder_out, encoder_out_lens)
        return adaptor_out


class _FunASRForSpeechSeq2Seq(torch.nn.Module):
    """Encoder-decoder wrapper around a FunASR model (e.g. Fun-ASR-Nano) for OpenVINO export.

    Structure: WavFrontend (fbank features, handled by the processor) -> audio encoder (SenseVoice) ->
    audio adaptor -> spliced into the Qwen3 LLM input embeddings at audio placeholder positions -> Qwen3 LLM.

    The wrapper exposes a transformers-style interface (`get_encoder`, `config.is_encoder_decoder`, an
    `audio_token_id` placeholder marker) so it flows through the standard speech-seq2seq export path.
    """

    # Audio placeholder token id used by FunASR (zero-id tokens are inserted as audio slots).
    AUDIO_TOKEN_ID = 0

    def __init__(self, funasr_model: torch.nn.Module, config: "PretrainedConfig"):
        super().__init__()
        self.audio_encoder = funasr_model.audio_encoder
        self.audio_adaptor = funasr_model.audio_adaptor
        self.llm = funasr_model.llm
        self.config = config
        self._funasr_model = True
        self._encoder = _FunASRAudioEncoder(self.audio_encoder, self.audio_adaptor)

    def get_encoder(self):
        return self._encoder

    def forward(self, *args, **kwargs):
        # The actual forward used at export time is provided by FunASRModelPatcher,
        # which redirects this depending on the encoder/decoder behavior.
        raise NotImplementedError("FunASR export forward is provided by FunASRModelPatcher.")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        **kwargs,
    ):
        if not is_funasr_available():
            raise ImportError(
                "To load a FunASR model (e.g. Fun-ASR-Nano), the `funasr` package is required. "
                "Please install it with `pip install funasr`."
            )

        import io
        from contextlib import redirect_stderr, redirect_stdout

        from funasr import AutoModel as FunASRAutoModel

        # funasr is very verbose during loading (per-tensor checkpoint warnings); silence it.
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            auto_model = FunASRAutoModel(
                model=str(model_name_or_path),
                hub="hf",
                trust_remote_code=True,
                device="cpu",
                disable_update=True,
            )
        funasr_model = auto_model.model.eval().float()
        llm_config = funasr_model.llm.config

        # Build a transformers-style config so the model flows through the speech-seq2seq export path.
        # The decoder is a standard Qwen3 LLM; we surface its text-config attributes for KV cache shapes.
        config = PretrainedConfig()
        config.model_type = "fun_asr"
        config.export_model_type = "fun_asr"
        config.is_encoder_decoder = True
        config.audio_token_id = cls.AUDIO_TOKEN_ID
        config.decoder_start_token_id = 0
        # text/decoder config (Qwen3)
        config.vocab_size = llm_config.vocab_size
        config.hidden_size = llm_config.hidden_size
        config.num_hidden_layers = llm_config.num_hidden_layers
        config.num_attention_heads = llm_config.num_attention_heads
        config.num_key_value_heads = getattr(llm_config, "num_key_value_heads", llm_config.num_attention_heads)
        config.head_dim = getattr(llm_config, "head_dim", llm_config.hidden_size // llm_config.num_attention_heads)
        config.eos_token_id = llm_config.eos_token_id
        config.pad_token_id = getattr(llm_config, "pad_token_id", None) or llm_config.eos_token_id
        config.bos_token_id = getattr(llm_config, "bos_token_id", None)
        config.max_position_embeddings = getattr(llm_config, "max_position_embeddings", 32768)
        # encoder config: feature size produced by WavFrontend (lfr_m * n_mels)
        config.num_mel_bins = getattr(funasr_model.audio_encoder, "input_size", 560)

        model = cls(funasr_model, config)
        model.config._name_or_path = str(model_name_or_path)
        return model


def _is_funasr_model(
    model_name_or_path: Union[str, Path],
    all_files: list,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
) -> bool:
    """Detect FunASR models (e.g. Fun-ASR-Nano) by checking for funasr-specific artifacts.

    FunASR models are loaded via the `funasr` library (not transformers): they ship a
    `config.yaml` describing the model and a `configuration.json` declaring `model.type == "funasr"`,
    and there is no root `config.json`.
    """
    if "configuration.json" not in all_files or "config.yaml" not in all_files:
        return False
    try:
        config_path = Path(model_name_or_path)
        if config_path.is_dir():
            config_file = config_path / "configuration.json"
        else:
            config_file = hf_hub_download(
                repo_id=str(model_name_or_path), filename="configuration.json", cache_dir=cache_dir, token=token
            )
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model", {}).get("type", None) == "funasr"
    except Exception:
        return False
