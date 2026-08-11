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
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino import Core
from transformers import PretrainedConfig

from ..utils.import_utils import is_funasr_available
from .modeling_seq2seq import OVModelForSpeechSeq2Seq
from .utils import OV_TOKENIZER_NAME


class _FunASRAudioEncoder(torch.nn.Module):
    """Wraps the FunASR audio encoder (SenseVoice) and audio adaptor as a single encoder module."""

    def __init__(self, audio_encoder: torch.nn.Module, audio_adaptor: torch.nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_adaptor = audio_adaptor

    def forward(self, input_features: "torch.Tensor"):
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
        config.audio_token_id = 0
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


def _is_funasr_source(model_id, **kwargs) -> bool:
    """Check whether model_id points to a FunASR source (original repo or exported OV model)."""
    from optimum.exporters.tasks import TasksManager

    cache_dir = kwargs.get("cache_dir", HUGGINGFACE_HUB_CACHE)
    token = kwargs.get("token")
    subfolder = kwargs.get("subfolder", "")
    revision = kwargs.get("revision")
    try:
        all_files, _ = TasksManager.get_model_files(
            model_id, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
        )
    except Exception:
        all_files = []

    if _is_funasr_model(model_id, all_files, cache_dir=cache_dir, token=token):
        return True

    if "config.json" in all_files:
        try:
            cfg = PretrainedConfig.from_pretrained(
                model_id, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
            )
            return getattr(cfg, "export_model_type", None) == "fun_asr"
        except Exception:
            return False
    return False


class _OVModelForFunAsr(OVModelForSpeechSeq2Seq):
    @classmethod
    def _from_pretrained_funasr(cls, model_id, export: bool = False, **kwargs):
        from ..utils.modeling_utils import _find_files_matching_pattern

        _export = export
        try:
            ov_files = _find_files_matching_pattern(
                model_id,
                pattern=cls._search_pattern,
                subfolder=kwargs.get("subfolder", ""),
                use_auth_token=kwargs.get("token"),
                revision=kwargs.get("revision"),
            )
            _export = len(ov_files) == 0
        except Exception:
            pass

        if _export:
            funasr_wrapped = _FunASRForSpeechSeq2Seq.from_pretrained(
                model_id, cache_dir=kwargs.get("cache_dir", HUGGINGFACE_HUB_CACHE), token=kwargs.get("token")
            )
            config = funasr_wrapped.config
            del funasr_wrapped
            return cls._export(model_id, config=config, **kwargs)

        config = PretrainedConfig.from_pretrained(model_id)
        if getattr(config, "export_model_type", None) == "fun_asr":
            config.model_type = "fun_asr"
        config.is_encoder_decoder = True
        return cls._from_pretrained(model_id, config=config, **kwargs)

    @classmethod
    def _from_pretrained(cls, model_id, config, **kwargs):
        return super(OVModelForSpeechSeq2Seq, cls)._from_pretrained(model_id, config, **kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        super()._save_pretrained(save_directory)
        if self.model_save_dir is not None:
            src_dir = Path(self.model_save_dir)
            save_directory = Path(save_directory)
            tokenizer_assets = [
                OV_TOKENIZER_NAME.format(""),
                OV_TOKENIZER_NAME.format("").replace(".xml", ".bin"),
                "openvino_detokenizer.xml",
                "openvino_detokenizer.bin",
            ]
            for name in tokenizer_assets:
                src = src_dir / name
                if src.is_file() and src.resolve() != (save_directory / name).resolve():
                    shutil.copyfile(src, save_directory / name)

    def preprocess_input(
        self,
        waveforms: Union[np.ndarray, torch.Tensor, List],
        sampling_rate: int,
        language: str = "中文",
        itn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Standalone FunASR preprocessing (no `funasr` dependency)."""
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        target_fs, n_mels, frame_length, frame_shift, lfr_m, lfr_n = 16000, 80, 25, 10, 7, 6
        audio_token_id = getattr(self.config, "audio_token_id", 0)

        def _apply_lfr(inputs: torch.Tensor) -> torch.Tensor:
            T = inputs.shape[0]
            T_lfr = int(np.ceil(T / lfr_n))
            left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
            inputs = torch.vstack((left_padding, inputs))
            T = T + (lfr_m - 1) // 2
            feat_dim = inputs.shape[-1]
            strides = (lfr_n * feat_dim, 1)
            sizes = (T_lfr, lfr_m * feat_dim)
            last_idx = (T - lfr_m) // lfr_n + 1
            num_padding = lfr_m - (T - last_idx * lfr_n)
            if num_padding > 0:
                num_padding = (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
                inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
            return inputs.as_strided(sizes, strides).clone().type(torch.float32)

        def _extract_features(waveform: torch.Tensor) -> torch.Tensor:
            if waveform.ndim > 1:
                waveform = waveform.mean(0)
            if sampling_rate != target_fs:
                waveform = torchaudio.transforms.Resample(sampling_rate, target_fs)(waveform[None, :])[0, :]
            wav = waveform.float() * (1 << 15)
            wav = wav.unsqueeze(0)
            mat = kaldi.fbank(
                wav,
                num_mel_bins=n_mels,
                frame_length=min(frame_length, wav.shape[1] / target_fs * 1000),
                frame_shift=frame_shift,
                dither=0.0,
                energy_floor=0.0,
                window_type="hamming",
                sample_frequency=target_fs,
                snip_edges=True,
            )
            return _apply_lfr(mat)

        def _num_audio_tokens(num_frames: int) -> int:
            olens = 1 + (num_frames - 3 + 2 * 1) // 2
            olens = 1 + (olens - 3 + 2 * 1) // 2
            return (olens - 1) // 2 + 1

        if isinstance(waveforms, (list, tuple)):
            wavs = [torch.as_tensor(np.asarray(w)) for w in waveforms]
        else:
            arr = waveforms if isinstance(waveforms, torch.Tensor) else torch.as_tensor(np.asarray(waveforms))
            wavs = [arr] if arr.ndim == 1 else list(arr)

        feats = [_extract_features(w) for w in wavs]
        num_frames = [f.shape[0] for f in feats]
        max_frames = max(num_frames)
        feature_size = feats[0].shape[-1]
        input_features = torch.zeros(len(feats), max_frames, feature_size, dtype=torch.float32)
        attention_mask = torch.zeros(len(feats), max_frames, dtype=torch.long)
        for i, f in enumerate(feats):
            input_features[i, : f.shape[0]] = f
            attention_mask[i, : f.shape[0]] = 1

        asr_prompt = f"语音转写成{language}：" if itn else f"语音转写成{language}，不进行文本规整："
        before = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{asr_prompt}"
        after = "<|im_end|>\n<|im_start|>assistant\n"
        before_ids = self._funasr_tokenizer_encode(before)
        after_ids = self._funasr_tokenizer_encode(after)

        prompt_ids = []
        for nf in num_frames:
            ids = before_ids + [audio_token_id] * _num_audio_tokens(nf) + after_ids
            prompt_ids.append(torch.tensor(ids, dtype=torch.long))
        max_len = max(t.shape[0] for t in prompt_ids)
        decoder_input_ids = torch.zeros(len(prompt_ids), max_len, dtype=torch.long)
        for i, t in enumerate(prompt_ids):
            decoder_input_ids[i, : t.shape[0]] = t

        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        for i, t in enumerate(prompt_ids):
            if t.shape[0] < max_len:
                decoder_attention_mask[i, t.shape[0] :] = 0

        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

    def _funasr_tokenizer_encode(self, text: str) -> List[int]:
        """Encode text to token ids using the exported OpenVINO tokenizer IR."""
        if getattr(self, "_ov_tokenizer", None) is None:
            import openvino_tokenizers  # noqa: F401

            tokenizer_path = Path(self.model_save_dir) / OV_TOKENIZER_NAME.format("")
            if not tokenizer_path.is_file():
                raise FileNotFoundError(
                    f"OpenVINO tokenizer IR not found at {tokenizer_path}. Re-export the model so the "
                    "tokenizer/detokenizer IR is generated."
                )
            self._ov_tokenizer = Core().compile_model(str(tokenizer_path), "CPU")
        result = self._ov_tokenizer([text])
        return result["input_ids"][0].tolist()

    def _prepare_decoder_input_ids_for_generation(
        self, batch_size, model_input_name, model_kwargs, decoder_start_token_id, device=None
    ):
        """Skip prepending decoder_start_token_id — full prompt already in decoder_input_ids."""
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        if decoder_input_ids is None:
            return super()._prepare_decoder_input_ids_for_generation(
                batch_size, model_input_name, model_kwargs, decoder_start_token_id, device
            )
        return decoder_input_ids, model_kwargs

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if decoder_input_ids is not None and past_key_values is None:
            if encoder_outputs is None and input_features is not None:
                encoder_outputs = self.encoder(input_ids=input_features)

            if encoder_outputs is not None:
                audio_token_id = getattr(self.config, "audio_token_id", 0)
                enc_hidden = (
                    encoder_outputs.last_hidden_state
                    if hasattr(encoder_outputs, "last_hidden_state")
                    else encoder_outputs[0]
                )
                num_encoder_features = enc_hidden.shape[1]
                current_audio_count = (decoder_input_ids == audio_token_id).sum(dim=-1).max().item()
                if current_audio_count > 0 and current_audio_count != num_encoder_features:
                    decoder_input_ids = self._adjust_audio_tokens(
                        decoder_input_ids, audio_token_id, num_encoder_features
                    )
                    if decoder_attention_mask is not None:
                        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        return super().forward(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    @staticmethod
    def _adjust_audio_tokens(decoder_input_ids, audio_token_id, target_count):
        """Adjust the number of audio_pad tokens in decoder_input_ids to match encoder output count."""
        result_ids = []
        for batch_idx in range(decoder_input_ids.shape[0]):
            ids = decoder_input_ids[batch_idx]
            audio_mask = ids == audio_token_id
            current_count = audio_mask.sum().item()
            if current_count == target_count:
                result_ids.append(ids)
            else:
                non_audio_before = []
                non_audio_after = []
                in_audio = False
                past_audio = False
                for tok in ids.tolist():
                    if tok == audio_token_id:
                        in_audio = True
                    else:
                        if in_audio:
                            past_audio = True
                            in_audio = False
                        if past_audio:
                            non_audio_after.append(tok)
                        else:
                            non_audio_before.append(tok)
                new_ids = non_audio_before + [audio_token_id] * target_count + non_audio_after
                result_ids.append(torch.tensor(new_ids, dtype=ids.dtype, device=ids.device))
        max_len = max(t.shape[0] for t in result_ids)
        padded = torch.zeros(len(result_ids), max_len, dtype=decoder_input_ids.dtype, device=decoder_input_ids.device)
        for i, t in enumerate(result_ids):
            padded[i, : t.shape[0]] = t
        return padded
