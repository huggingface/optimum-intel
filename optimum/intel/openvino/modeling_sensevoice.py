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
from .modeling_funasr import _apply_lfr, _read_funasr_config
from .utils import OV_DETOKENIZER_NAME, OV_XML_FILE_NAME


logger = logging.getLogger(__name__)

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
        if not is_funasr_available():
            raise ImportError(
                "To load a SenseVoiceSmall model, the `funasr` package is required. "
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

        config = SenseVoicePretrainedConfig()
        config.export_model_type = "sense_voice"
        config.is_encoder_decoder = False
        config.num_mel_bins = funasr_model.embed.embedding_dim  # 560
        config.vocab_size = funasr_model.ctc.ctc_lo.out_features  # 25055
        config.encoder_output_size = funasr_model.ctc.ctc_lo.in_features  # 512 (CTC head input dim)
        config.blank_id = int(getattr(funasr_model, "blank_id", 0))
        config.lid_dict = dict(funasr_model.lid_dict)
        config.textnorm_dict = dict(funasr_model.textnorm_dict)
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
            cfg = SenseVoicePretrainedConfig.from_pretrained(
                model_id, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
            )
            return (
                getattr(cfg, "export_model_type", None) == "sense_voice"
                or getattr(cfg, "model_type", None) == "sense_voice"
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
        self.config = config
        self._device = device.upper() if isinstance(device, str) else "CPU"
        self.ov_config = dict(ov_config) if ov_config else {}
        self._model_save_dir = Path(model_save_dir) if model_save_dir is not None else None
        self.model = model
        self.detokenizer_model = detokenizer_model
        self.request = None
        self.detokenizer_request = None
        self._cmvn = None
        self.preprocessors = kwargs.get("preprocessors", [])
        self._compile_only = kwargs.get("compile_only", False)
        # Participate in the shared OVBaseModel quantization bookkeeping (set by `_apply_quantization`).
        self._openvino_config = None
        quantization_config = kwargs.get("quantization_config")
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()
        if compile:
            self.compile()

    @property
    def model_save_dir(self):
        return self._model_save_dir

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def ov_models(self) -> Dict[str, "openvino.Model"]:
        return {"model": self.model}

    def compile(self):
        core = Core()
        if self.request is None:
            self.request = core.compile_model(self.model, self._device, self.ov_config)
        if self.detokenizer_model is not None and self.detokenizer_request is None:
            self.detokenizer_request = core.compile_model(self.detokenizer_model, self._device)

    def clear_requests(self):
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

        from .configuration import OVConfig

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
        model = core.read_model(model_dir / OV_XML_FILE_NAME)
        detokenizer_path = model_dir / OV_DETOKENIZER_NAME.format("")
        detokenizer_model = core.read_model(detokenizer_path) if detokenizer_path.is_file() else None

        load_in_8bit = kwargs.pop("load_in_8bit", None)
        quantization_config = kwargs.pop("quantization_config", None)
        quantization_config = quantization_config or (OVWeightQuantizationConfig(bits=8) if load_in_8bit else None)
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        compile_model = kwargs.get("compile", True)

        ov_model = cls(
            model=model,
            config=config,
            model_save_dir=model_dir,
            detokenizer_model=detokenizer_model,
            device=kwargs.get("device", "CPU"),
            ov_config=kwargs.get("ov_config"),
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
        save_directory = Path(save_directory)
        openvino.save_model(self.model, save_directory / OV_XML_FILE_NAME, compress_to_fp16=False)
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

    # ----------------------------- pre/post-processing -----------------------------

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
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        target_fs, n_mels, frame_length, frame_shift, lfr_m, lfr_n = 16000, 80, 25, 10, 7, 6

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
            arr = arr.float()
            if arr.ndim > 1:
                arr = arr.mean(0)
            if sampling_rate != target_fs:
                arr = torchaudio.transforms.Resample(sampling_rate, target_fs)(arr[None, :])[0, :]

            wav = arr * (1 << 15)
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
            mat = _apply_lfr(mat, lfr_n, lfr_m)
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
