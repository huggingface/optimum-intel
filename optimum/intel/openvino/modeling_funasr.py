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
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import openvino
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino import Core
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from ..utils.import_utils import is_funasr_available
from .configuration import OVConfig, OVWeightQuantizationConfig
from .modeling import OVModel
from .modeling_seq2seq import FunASRPretrainedConfig, OVModelForSpeechSeq2Seq
from .utils import OV_DETOKENIZER_NAME, OV_TOKENIZER_NAME, OV_XML_FILE_NAME


logger = logging.getLogger(__name__)


def _load_funasr_model(
    model_name_or_path: Union[str, Path],
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
    trust_remote_code: bool = False,
) -> torch.nn.Module:
    """Load a FunASR/SenseVoice model via `funasr.AutoModel`, returning the underlying torch module.

    `funasr` is very verbose during loading (per-tensor checkpoint warnings), so its stdout/stderr are
    silenced. The returned module is put in eval mode and cast to float32.
    """
    if not is_funasr_available():
        raise ImportError(
            "To load a FunASR/SenseVoice model, the `funasr` package is required. "
            "Please install it with `pip install funasr`."
        )

    import io
    from contextlib import redirect_stderr, redirect_stdout

    from funasr import AutoModel as FunASRAutoModel

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        auto_model = FunASRAutoModel(
            model=str(model_name_or_path),
            hub="hf",
            trust_remote_code=trust_remote_code,
            device="cpu",
            disable_update=True,
        )
    return auto_model.model.eval().float()


def _extract_fbank_lfr(
    waveform,
    sampling_rate: int,
    target_fs: int = 16000,
    n_mels: int = 80,
    frame_length: int = 25,
    frame_shift: int = 10,
    lfr_m: int = 7,
    lfr_n: int = 6,
) -> torch.Tensor:
    """Compute Kaldi fbank features for a single waveform and apply LFR stacking.

    Mono-downmixes and resamples the waveform to `target_fs`, scales it to the int16 range, extracts
    `n_mels`-bin fbank features, then stacks consecutive frames with `_apply_lfr`. Shared by both the
    FunASR and SenseVoice preprocessing paths.
    """
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi

    waveform = torch.as_tensor(waveform).float()
    if waveform.ndim > 1:
        waveform = waveform.mean(0)
    if sampling_rate != target_fs:
        waveform = torchaudio.transforms.Resample(sampling_rate, target_fs)(waveform[None, :])[0, :]
    wav = (waveform * (1 << 15)).unsqueeze(0)
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
    return _apply_lfr(mat, lfr_n, lfr_m)


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
        funasr_model = _load_funasr_model(model_name_or_path, cache_dir=cache_dir, token=token, trust_remote_code=True)
        llm_config = funasr_model.llm.config

        # Build a transformers-style config so the model flows through the speech-seq2seq export path.
        # The decoder is a standard Qwen3 LLM; we surface its text-config attributes for KV cache shapes.
        config = FunASRPretrainedConfig()
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


def _read_funasr_config(
    config_name: str,
    model_name_or_path: Union[str, Path],
    all_files: list,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
) -> Union[dict, None]:
    """Detect FunASR models (e.g. Fun-ASR-Nano) by checking for funasr-specific artifacts.

    FunASR models are loaded via the `funasr` library (not transformers): they ship a
    `config.yaml` describing the model and a `configuration.json` declaring `model.type == "funasr"`,
    and there is no root `config.json`.
    """
    if config_name not in all_files:
        return None
    try:
        config_path = Path(model_name_or_path)
        if config_path.is_dir():
            config_file = config_path / config_name
        else:
            config_file = hf_hub_download(
                repo_id=str(model_name_or_path), filename=config_name, cache_dir=cache_dir, token=token
            )
        with open(config_file, "r", encoding="utf-8") as f:
            if config_name.endswith(".json"):
                return json.load(f)
            return f.readlines()
    except Exception:
        return None

    return None


def _is_funasr_model(
    model_name_or_path: Union[str, Path],
    all_files: list,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
) -> bool:
    config = _read_funasr_config("configuration.json", model_name_or_path, all_files, cache_dir, token)
    return config is not None and config.get("model", {}).get("type", None) == "funasr"


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
            cfg = FunASRPretrainedConfig.from_pretrained(
                model_id, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
            )
            return getattr(cfg, "export_model_type", None) == "fun_asr"
        except Exception:
            return False
    return False


def _apply_lfr(inputs: torch.Tensor, lfr_n, lfr_m) -> torch.Tensor:
    """Apply Low Frame Rate (LFR) stacking to a sequence of acoustic features.

    Stacks every lfr_m consecutive frames into a single frame and advances by lfr_n frames
    between stacks (lfr_n acts as the downsampling factor), reducing the frame rate while widening
    the feature dimension. The sequence is left-padded by repeating the first frame (lfr_m - 1) // 2
    times and right-padded by repeating the last frame so the final windows are complete.

    Args:
        inputs: Feature tensor of shape (T, feat_dim).
        lfr_n: Stride (number of frames to advance between consecutive stacked frames).
        lfr_m: Number of consecutive frames stacked together into one output frame.

    Returns:
        A float32 tensor of shape (ceil(T / lfr_n), lfr_m * feat_dim), where T = inputs.shape[0], feat_dim = inputs.shape[-1]
    """
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

        config = FunASRPretrainedConfig.from_pretrained(model_id)
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
        audio_token_id = getattr(self.config, "audio_token_id", 0)

        def _extract_features(waveform: torch.Tensor) -> torch.Tensor:
            return _extract_fbank_lfr(waveform, sampling_rate)

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


# SenseVoice ships two auxiliary assets used at (pre/post)-processing time, independent of funasr.
SENSEVOICE_CMVN_FILE = "am.mvn"
SENSEVOICE_BPE_FILE = "chn_jpn_yue_eng_ko_spectok.bpe.model"


class SenseVoicePretrainedConfig(PretrainedConfig):
    model_type = "sense_voice"


class _SenseVoiceForCTC(torch.nn.Module):
    """CTC encoder-only wrapper around a FunASR SenseVoiceSmall model for OpenVINO export.

    SenseVoiceSmall is not an encoder-decoder model: a SANM encoder produces frame-level features and a
    single CTC head projects them to the token vocabulary. This wrapper exposes a single ``forward`` that
    runs the encoder and the CTC head together, so the whole model is exported as one graph
    (``openvino_model.xml``) returning raw CTC logits and the per-sample encoder output lengths (so a
    padded batch can be decoded sample-by-sample).
    """

    def __init__(self, funasr_model: torch.nn.Module, config: "PretrainedConfig"):
        super().__init__()
        self.embed = funasr_model.embed
        self.encoder = funasr_model.encoder
        self.ctc = funasr_model.ctc
        self.config = config
        self._funasr_model = True

    def forward(
        self,
        input_features: "torch.Tensor",
        speech_lengths: "torch.Tensor",
        language: "torch.Tensor",
        textnorm: "torch.Tensor",
    ):
        # input_features: [B, T, 560]; speech_lengths: [B] valid frame counts (padding-aware);
        # language/textnorm: [B] integer ids into self.embed.
        language_query = self.embed(language).unsqueeze(1)  # [B, 1, 560]
        textnorm_query = self.embed(textnorm).unsqueeze(1)  # [B, 1, 560]

        speech = torch.cat((textnorm_query, input_features), dim=1)
        event_emo_query = self.embed(torch.tensor([[1, 2]], dtype=torch.long, device=input_features.device)).repeat(
            input_features.shape[0], 1, 1
        )  # [B, 2, 560]
        input_query = torch.cat((language_query, event_emo_query), dim=1)  # [B, 3, 560]
        speech = torch.cat((input_query, speech), dim=1)  # order: [language, event, emo, textnorm, speech...]

        # Four prefix queries were prepended, so every sequence grows by 4 valid frames.
        speech_lengths_new = speech_lengths + 4
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths_new)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        ctc_logits = self.ctc.ctc_lo(encoder_out)
        return ctc_logits, encoder_out_lens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        **kwargs,
    ):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        funasr_model = _load_funasr_model(
            model_name_or_path, cache_dir=cache_dir, token=token, trust_remote_code=trust_remote_code
        )

        config = SenseVoicePretrainedConfig()
        config.export_model_type = "sense_voice"
        config.is_encoder_decoder = False
        config.num_mel_bins = funasr_model.embed.embedding_dim  # 560
        config.vocab_size = funasr_model.ctc.ctc_lo.out_features  # 25055
        config.encoder_output_size = funasr_model.ctc.ctc_lo.in_features  # 512 (CTC head input dim)
        config.blank_id = int(getattr(funasr_model, "blank_id", 0))
        config.lid_dict = dict(funasr_model.lid_dict)
        config.textnorm_dict = dict(funasr_model.textnorm_dict)

        model = cls(funasr_model, config)
        model.config._name_or_path = str(model_name_or_path)
        model._sensevoice_source = str(model_name_or_path)
        return model


def _is_sensevoice_model(
    model_name_or_path: Union[str, Path],
    all_files: list,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
) -> bool:
    config = _read_funasr_config("config.yaml", model_name_or_path, all_files, cache_dir, token)
    if config is None:
        return False
    for line in config:
        stripped = line.strip()
        if stripped.startswith("model:"):
            if stripped.split(":", 1)[1].strip() == "SenseVoiceSmall":
                return True
    return False


def _is_sensevoice_source(model_id, **kwargs) -> bool:
    """Check whether model_id points to a SenseVoice source (original repo or exported OV model)."""
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

    if _is_sensevoice_model(model_id, all_files, cache_dir=cache_dir, token=token):
        return True

    if "config.json" in all_files:
        try:
            config_dict, _ = PretrainedConfig.get_config_dict(
                model_id, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
            )
            return (
                config_dict.get("export_model_type") == "sense_voice" or config_dict.get("model_type") == "sense_voice"
            )
        except Exception:
            return False
    return False


def _resolve_sensevoice_asset(source_model_id, asset, cache_dir=HUGGINGFACE_HUB_CACHE, token=None):
    """Return a local path to a SenseVoice source asset (from a local dir or the HF Hub), or None."""
    source_dir = Path(source_model_id)
    if source_dir.is_dir():
        candidate = source_dir / asset
        return candidate if candidate.is_file() else None
    try:
        return Path(hf_hub_download(repo_id=str(source_model_id), filename=asset, cache_dir=cache_dir, token=token))
    except Exception:
        return None


def export_sensevoice_tokenizers(source_model_id, output, cache_dir=HUGGINGFACE_HUB_CACHE, token=None):
    """Convert the SenseVoice SentencePiece model to an OpenVINO detokenizer IR under ``output``.

    SenseVoiceSmall does not accept text input, so only the detokenizer (token ids -> text) is needed; the
    tokenizer IR is intentionally not generated. SenseVoice ships a raw SentencePiece model rather than a
    transformers tokenizer, so it is wrapped in a `T5Tokenizer` (which is SentencePiece-backed) before
    conversion. The resulting OpenVINO detokenizer maps token ids to text identically to
    `SentencePieceProcessor.DecodeIds` for SenseVoice ids (including the `<|lang|>`/`<|emo|>` special tokens),
    so CTC greedy output can be detokenized entirely with the exported IR, without a runtime SentencePiece
    dependency.
    """
    from transformers import T5Tokenizer

    try:
        from openvino_tokenizers import convert_tokenizer
    except ModuleNotFoundError:
        return

    bpe_path = _resolve_sensevoice_asset(source_model_id, SENSEVOICE_BPE_FILE, cache_dir=cache_dir, token=token)
    if bpe_path is None:
        return
    tokenizer = T5Tokenizer(vocab_file=str(bpe_path), extra_ids=0, legacy=True)
    _, detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
    openvino.save_model(detokenizer, Path(output) / OV_DETOKENIZER_NAME.format(""))


def copy_sensevoice_cmvn(source_model_id, output, cache_dir=HUGGINGFACE_HUB_CACHE, token=None):
    """Copy the CMVN stats file (``am.mvn``) next to the exported IRs, so preprocessing stays self-contained."""
    dst = Path(output) / SENSEVOICE_CMVN_FILE
    if dst.is_file():
        return
    src = _resolve_sensevoice_asset(source_model_id, SENSEVOICE_CMVN_FILE, cache_dir=cache_dir, token=token)
    if src is not None:
        shutil.copyfile(src, dst)


class _OVModelForSenseVoice(OVModel):
    """OpenVINO inference for SenseVoiceSmall (CTC), mirroring the FunASR asset layout.

    A single OpenVINO graph (``openvino_model.xml``) runs the SANM encoder and the CTC head together and
    returns raw CTC logits; token ids are turned into text with the exported OpenVINO detokenizer. Inference
    has no dependency on the funasr runtime.
    """

    export_feature = "automatic-speech-recognition"
    main_input_name = "input_features"
    _library_name = "funasr"

    def __init__(
        self,
        model,
        config,
        model_save_dir=None,
        detokenizer_model=None,
        device="CPU",
        ov_config=None,
        compile=True,
        **kwargs,
    ):
        # The detokenizer is a second IR that the base class is unaware of; set it up before delegating
        # so that the base `__init__` (which calls `self.compile()`) also compiles the detokenizer.
        self.detokenizer_model = detokenizer_model
        self.detokenizer_request = None
        self._cmvn = None
        super().__init__(
            model,
            config,
            device=device,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            compile=compile,
            **kwargs,
        )
        if self._compile_only:
            # The detokenizer IR is already compiled; reuse it directly as its inference request.
            self.detokenizer_request = self.detokenizer_model

    def _reshape(self, model, batch_size, sequence_length, height=None, width=None):
        # SenseVoice IRs are exported fully dynamic, so the generic
        # rank-2 reshape does not apply; shapes are left dynamic.
        return model

    def compile(self):
        super().compile()
        if self.detokenizer_model is not None and self.detokenizer_request is None:
            self.detokenizer_request = self._compile_model(
                self.detokenizer_model, self._device, {**self.ov_config}, self.model_save_dir
            )

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        self.request = None
        self.detokenizer_request = None

    @classmethod
    def _from_pretrained_sensevoice(cls, model_id, export: bool = False, **kwargs):
        from ..utils.modeling_utils import _find_files_matching_pattern

        _export = export
        try:
            ov_files = _find_files_matching_pattern(
                model_id,
                pattern=r"openvino_model.*\.xml",
                subfolder=kwargs.get("subfolder", ""),
                use_auth_token=kwargs.get("token"),
                revision=kwargs.get("revision"),
            )
            _export = len(ov_files) == 0
        except Exception:
            pass

        if _export:
            sensevoice_wrapped = _SenseVoiceForCTC.from_pretrained(
                model_id, cache_dir=kwargs.get("cache_dir", HUGGINGFACE_HUB_CACHE), token=kwargs.get("token")
            )
            config = sensevoice_wrapped.config
            del sensevoice_wrapped
            return cls._export(model_id, config=config, **kwargs)

        config = SenseVoicePretrainedConfig.from_pretrained(model_id)
        config.is_encoder_decoder = False
        return cls._from_pretrained(model_id, config=config, **kwargs)

    @classmethod
    def _export(cls, model_id, config, **kwargs):
        from tempfile import TemporaryDirectory

        from optimum.exporters.openvino.__main__ import main_export

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        # Keep one reference on the temporary directory so garbage collection does not remove the
        # directory holding the exported OpenVINO IRs before they are reloaded.
        cls._model_save_dir_tempdirectory_instance = save_dir

        compile_only = kwargs.pop("compile_only", False)
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export. "
                "Please provide an OpenVINO model obtained using optimum-cli or saved on disk using `save_pretrained`."
            )
            compile_only = False

        load_in_8bit = kwargs.pop("load_in_8bit", None)
        quantization_config = kwargs.pop("quantization_config", None)
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        variant = kwargs.pop("variant", None)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=kwargs.pop("task", None) or cls.export_feature,
            subfolder=kwargs.pop("subfolder", ""),
            revision=kwargs.pop("revision", None),
            cache_dir=kwargs.pop("cache_dir", HUGGINGFACE_HUB_CACHE),
            token=kwargs.pop("token", None),
            local_files_only=kwargs.pop("local_files_only", False),
            force_download=kwargs.pop("force_download", False),
            trust_remote_code=kwargs.pop("trust_remote_code", False),
            ov_config=ov_config,
            library_name=cls._library_name,
            variant=variant,
            # SenseVoiceSmall ships a SentencePiece BPE model instead of a transformers tokenizer. It has no
            # text input, so only the detokenizer IR (needed to decode CTC ids) is generated at export time.
            convert_tokenizer=True,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            compile_only=compile_only,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        config,
        token=None,
        revision=None,
        force_download=False,
        cache_dir=HUGGINGFACE_HUB_CACHE,
        subfolder="",
        local_files_only=False,
        compile_only=False,
        **kwargs,
    ):
        import openvino_tokenizers  # noqa: F401  — registers the SentencePiece ops extension

        model_dir = cls._resolve_model_dir(
            model_id,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )
        core = Core()
        load_in_8bit = kwargs.pop("load_in_8bit", None)
        quantization_config = kwargs.pop("quantization_config", None)
        quantization_config = quantization_config or (OVWeightQuantizationConfig(bits=8) if load_in_8bit else None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        compile_model = kwargs.get("compile", True)
        device = kwargs.get("device", "CPU")
        ov_config = kwargs.get("ov_config")

        if compile_only and quantization_config is not None:
            raise ValueError(
                "`compile_only` mode is not supported together with quantization, since quantization requires an "
                "editable model. Please set `compile_only=False` to quantize the model."
            )

        model_path = model_dir / OV_XML_FILE_NAME
        detokenizer_path = model_dir / OV_DETOKENIZER_NAME.format("")
        if compile_only:
            model = cls._compile_model(model_path, device, ov_config, model_save_dir=model_dir)
            detokenizer_model = (
                cls._compile_model(core.read_model(detokenizer_path), device, model_save_dir=model_dir)
                if detokenizer_path.is_file()
                else None
            )
        else:
            model = core.read_model(model_path)
            detokenizer_model = core.read_model(detokenizer_path) if detokenizer_path.is_file() else None

        ov_model = cls(
            model=model,
            config=config,
            model_save_dir=model_dir,
            detokenizer_model=detokenizer_model,
            device=device,
            ov_config=ov_config,
            compile=compile_model and not quantization_config,
            compile_only=compile_only,
            quantization_config=quantization_config,
        )

        # Reuse the shared OVBaseModel weight-only quantization pipeline (NNCF via OVQuantizer).
        if quantization_config is not None:
            quantization_config = cls._resolve_default_quantization_config(str(model_id), quantization_config)
            ov_model._apply_quantization(
                quantization_config,
                compile_only,
                compile_model,
                str(model_id),
                trust_remote_code,
            )

        return ov_model

    @staticmethod
    def _resolve_model_dir(
        model_id,
        token=None,
        revision=None,
        force_download=False,
        cache_dir=HUGGINGFACE_HUB_CACHE,
        subfolder="",
        local_files_only=False,
    ) -> Path:
        model_path = Path(model_id)
        if model_path.is_dir():
            return model_path / subfolder if subfolder else model_path
        allow_patterns = ["*.xml", "*.bin", "*.json", SENSEVOICE_CMVN_FILE]
        downloaded = snapshot_download(
            repo_id=str(model_id),
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
        return Path(downloaded) / subfolder if subfolder else Path(downloaded)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        super()._save_pretrained(save_directory)
        save_directory = Path(save_directory)
        # Copy the SenseVoice-specific assets
        if self.model_save_dir is not None:
            src_dir = Path(self.model_save_dir)
            assets = [
                SENSEVOICE_CMVN_FILE,
                OV_DETOKENIZER_NAME.format(""),
                OV_DETOKENIZER_NAME.format("").replace(".xml", ".bin"),
            ]
            for name in assets:
                src = src_dir / name
                if src.is_file() and src.resolve() != (save_directory / name).resolve():
                    shutil.copyfile(src, save_directory / name)

    def forward(self, input_features=None, speech_lengths=None, language=None, textnorm=None, **kwargs):
        np_inputs = isinstance(input_features, np.ndarray)
        if speech_lengths is None:
            n_frames = input_features.shape[1]
            batch = input_features.shape[0]
            speech_lengths = np.full((batch,), n_frames, dtype=np.int32)
        inputs = {
            "input_features": input_features if np_inputs else np.asarray(input_features, dtype=np.float32),
            "speech_lengths": np.asarray(speech_lengths, dtype=np.int32),
            "language": np.asarray(language, dtype=np.int64),
            "textnorm": np.asarray(textnorm, dtype=np.int64),
        }
        outputs = self.request(inputs)
        logits = outputs[self.request.output(0)]
        encoder_out_lens = outputs[self.request.output(1)]
        if not np_inputs:
            logits = torch.from_numpy(logits)
            encoder_out_lens = torch.from_numpy(encoder_out_lens)
        return CausalLMOutput(logits=logits), encoder_out_lens

    @property
    def cmvn(self):
        if self._cmvn is None:
            self._cmvn = self._load_cmvn(Path(self.model_save_dir) / SENSEVOICE_CMVN_FILE)
        return self._cmvn

    @staticmethod
    def _load_cmvn(cmvn_file: Union[str, Path]) -> "torch.Tensor":
        with open(cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        means, variances = [], []
        for i in range(len(lines)):
            items = lines[i].split()
            if items and items[0] == "<AddShift>":
                nxt = lines[i + 1].split()
                if nxt and nxt[0] == "<LearnRateCoef>":
                    means = nxt[3 : len(nxt) - 1]
            elif items and items[0] == "<Rescale>":
                nxt = lines[i + 1].split()
                if nxt and nxt[0] == "<LearnRateCoef>":
                    variances = nxt[3 : len(nxt) - 1]
        return torch.tensor(
            np.array([np.array(means, dtype=np.float32), np.array(variances, dtype=np.float32)]),
            dtype=torch.float32,
        )

    def preprocess_input(
        self,
        waveforms: Union[np.ndarray, torch.Tensor, List],
        sampling_rate: int = 16000,
        language: str = "auto",
        use_itn: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """SenseVoice preprocessing (fbank -> LFR -> CMVN) for one or more waveforms.

        Multiple variable-length waveforms are zero-padded to a common frame count and stacked into a single
        batch; ``speech_lengths`` records each sample's real (pre-padding) frame count so the exported model
        masks the padding correctly.
        """

        def _apply_cmvn(inputs: torch.Tensor) -> torch.Tensor:
            device = inputs.device
            dim = inputs.shape[-1]
            means = self.cmvn[0:1, :dim].to(device)
            variances = self.cmvn[1:2, :dim].to(device)
            return (inputs + means) * variances

        if isinstance(waveforms, (list, tuple)):
            wav_list = [torch.as_tensor(np.asarray(w)) for w in waveforms]
        else:
            arr = waveforms if isinstance(waveforms, torch.Tensor) else torch.as_tensor(np.asarray(waveforms))
            wav_list = [arr] if arr.ndim == 1 else list(arr)

        feats: List[torch.Tensor] = []
        for arr in wav_list:
            mat = _extract_fbank_lfr(arr, sampling_rate)
            mat = _apply_cmvn(mat)
            feats.append(mat)

        lid_dict = self.config.lid_dict
        textnorm_dict = self.config.textnorm_dict
        language_id = lid_dict.get(language, 0)
        textnorm_id = textnorm_dict["withitn"] if use_itn else textnorm_dict["woitn"]

        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.int32)
        max_frames = int(lengths.max())
        feat_dim = feats[0].shape[1]
        batch = torch.zeros(len(feats), max_frames, feat_dim, dtype=torch.float32)
        for i, f in enumerate(feats):
            batch[i, : f.shape[0]] = f

        n = len(feats)
        return {
            "input_features": batch,
            "speech_lengths": lengths,
            "language": torch.full((n,), language_id, dtype=torch.long),
            "textnorm": torch.full((n,), textnorm_id, dtype=torch.long),
        }

    def _ctc_greedy_ids(self, logits: "torch.Tensor") -> List[int]:
        """CTC greedy collapse of a single sequence of logits [T, vocab] to token ids."""
        blank_id = int(getattr(self.config, "blank_id", 0))
        yseq = logits.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)
        return yseq[yseq != blank_id].tolist()

    def _detokenize(self, token_ids: List[int]) -> str:
        """Turn CTC token ids into text using the exported OpenVINO detokenizer IR."""
        if self.detokenizer_request is None:
            raise FileNotFoundError(
                "OpenVINO detokenizer IR not found. Re-export the model so the tokenizer/detokenizer IR is generated."
            )
        if len(token_ids) == 0:
            return ""
        result = self.detokenizer_request(np.array([token_ids], dtype=np.int64))
        return str(result[self.detokenizer_request.output(0)][0])

    def generate(
        self,
        waveforms: Union[np.ndarray, torch.Tensor, List] = None,
        input_features: Union[np.ndarray, torch.Tensor, List] = None,
        speech_lengths: Union[np.ndarray, torch.Tensor, List] = None,
        sampling_rate: int = 16000,
        language: str = "auto",
        use_itn: bool = False,
        **kwargs,
    ) -> List[str]:
        """Transcribe one or more waveforms.

        Multiple waveforms are zero-padded into a single batch and run through the model in one forward; each
        sample is then CTC-decoded using its own valid encoder-output length so the padding is ignored.
        """
        assert (
            input_features is not None or waveforms is not None
        ), "Either input_features or waveform must be specified."
        if waveforms is not None:
            inputs = self.preprocess_input(waveforms, sampling_rate, language=language, use_itn=use_itn)
            outputs, encoder_out_lens = self.forward(**inputs)
        else:
            lid_dict = self.config.lid_dict
            textnorm_dict = self.config.textnorm_dict
            language_id = lid_dict.get(language, 0)
            textnorm_id = textnorm_dict["withitn"] if use_itn else textnorm_dict["woitn"]

            if speech_lengths is None:
                speech_lengths = torch.tensor([f.shape[0] for f in input_features], dtype=torch.int32)

            language = torch.full((input_features.shape[0],), language_id, dtype=torch.long)
            textnorm = torch.full((input_features.shape[0],), textnorm_id, dtype=torch.long)
            outputs, encoder_out_lens = self.forward(
                input_features=input_features, speech_lengths=speech_lengths, language=language, textnorm=textnorm
            )

        logits = outputs.logits

        results = []
        for i in range(logits.shape[0]):
            valid = int(encoder_out_lens[i])
            sample_logits = logits[i, :valid]
            results.append(self._detokenize(self._ctc_greedy_ids(sample_logits)))
        return results
