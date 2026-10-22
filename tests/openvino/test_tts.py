import gc
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import openvino
import torch
from parameterized import parameterized
from transformers import (
    AutoModelForTextToSpectrogram,
    AutoProcessor,
    GenerationConfig,
    PretrainedConfig,
    set_seed,
)
from utils_tests import (
    F32_CONFIG,
    MODEL_NAMES,
    OPENVINO_DEVICE,
    REMOTE_CODE_MODELS,
    SEED,
)

from optimum.exporters.openvino.stateful import model_has_state
from optimum.exporters.tasks import TasksManager
from optimum.intel import (
    OVModelForTextToSpeechSeq2Seq,
)
from optimum.intel.openvino.modeling_text2speech import (
    OVTextToSpeechDecoder,
    OVTextToSpeechEncoder,
    OVTextToSpeechPostNet,
    OVTextToSpeechVocoder,
)
from optimum.intel.utils.import_utils import (
    is_qwen_tts_available,
    is_transformers_version,
)


if is_transformers_version("<=", "4.52"):
    pass
else:
    pass

from test_seq2seq import OVSeq2SeqTestMixin


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OVModelForTextToSpeechSeq2SeqIntegrationTest(OVSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = ("speecht5",)
    TTS_SUPPORTED_ARCHITECTURES = SUPPORTED_ARCHITECTURES + ("kokoro", "qwen3_tts")
    OVMODEL_CLASS = OVModelForTextToSpeechSeq2Seq
    AUTOMODEL_CLASS = AutoModelForTextToSpectrogram

    def _generate_text(self):
        return [
            "This text is converted to speech using OpenVINO backend",
            "The sun was shining brightly as people walked through the quiet park.",
        ]

    def _generate_speaker_embedding(self):
        np.random.seed(42)
        speaker_embedding = np.random.randn(2, 512).astype(np.float32)
        return torch.tensor(speaker_embedding)

    def _get_processor(self, model_id, model_arch):
        if model_arch == "speecht5":
            return AutoProcessor.from_pretrained(model_id)
        else:
            raise Exception("{} unknown processor for text-to-speech".format(model_arch))

    def _get_vocoder(self, vocoder_id, model_arch):
        if model_arch == "speecht5":
            from transformers import SpeechT5HifiGan

            vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id)
            return vocoder
        else:
            raise Exception("{} unknown model for text-to-speech".format(model_arch))

    def _check_openvino_model_attributes(self, openvino_model, use_cache: bool = True):
        self.assertIsInstance(openvino_model, self.OVMODEL_CLASS)
        self.assertIsInstance(openvino_model.config, PretrainedConfig)
        self.assertIsInstance(openvino_model.generation_config, GenerationConfig)

        self.assertIsInstance(openvino_model.encoder, OVTextToSpeechEncoder)
        self.assertIsInstance(openvino_model.decoder, OVTextToSpeechDecoder)
        self.assertIsInstance(openvino_model.postnet, OVTextToSpeechPostNet)
        self.assertIsInstance(openvino_model.vocoder, OVTextToSpeechVocoder)
        self.assertIsInstance(openvino_model.encoder.model, openvino.Model)
        self.assertIsInstance(openvino_model.decoder.model, openvino.Model)
        self.assertIsInstance(openvino_model.postnet.model, openvino.Model)
        self.assertIsInstance(openvino_model.vocoder.model, openvino.Model)

        self.assertEqual(openvino_model.use_cache, use_cache)
        self.assertEqual(model_has_state(openvino_model.decoder.model), use_cache)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        set_seed(SEED)
        text_data = self._generate_text()
        speaker_embeddings = self._generate_speaker_embedding()
        model_id = MODEL_NAMES[model_arch]

        # since Auto class for text-to-audio is not implemented in optimum
        # generate model classes for reference generation
        vocoder_id = "fxmarty/speecht5-hifigan-tiny"
        processor = self._get_processor(model_id, model_arch)
        vocoder = self._get_vocoder(vocoder_id, model_arch)
        model = self.AUTOMODEL_CLASS.from_pretrained(model_id)
        inputs = processor(text=text_data, padding=True, return_tensors="pt")
        ref_speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        ref_speech = ref_speech.unsqueeze(0) if ref_speech.dim() == 1 else ref_speech
        ov_model = self.OVMODEL_CLASS.from_pretrained(model_id, vocoder=vocoder_id, device=OPENVINO_DEVICE)
        ov_speech = ov_model.generate(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings)
        self._check_openvino_model_attributes(ov_model, use_cache=True)
        self.assertTrue(torch.allclose(ov_speech, ref_speech, atol=1e-3))

        del vocoder
        del model
        del processor
        gc.collect()

    @unittest.skipUnless(is_qwen_tts_available(), "qwen_tts package is not installed")
    def test_compare_to_qwen3_tts(self):
        """Export Qwen3-TTS and check every component runs from OpenVINO and matches PyTorch.

        The pipeline is exported component-wise and the exported directory carries no PyTorch
        weights at all, so this also covers the weightless runtime path: if any component
        failed to load from its IR, generation would run on empty tensors rather than fall
        back silently.
        """
        from qwen_tts import Qwen3TTSModel

        from optimum.exporters.openvino import export_from_model
        from optimum.intel.openvino.modeling_text2speech import _OVModelForQwen3TTS
        from optimum.intel.utils.modeling_utils import _Qwen3TTSForTextToSpeech

        set_seed(SEED)
        model_id = MODEL_NAMES["qwen3_tts"]

        ref_pipeline = Qwen3TTSModel.from_pretrained(model_id, dtype=torch.float32)
        ref_pipeline.model.eval()

        sampling_rate = ref_pipeline.model.speaker_encoder_sample_rate
        ref_audio = (np.random.default_rng(SEED).standard_normal(3 * sampling_rate) * 0.05).astype(np.float32)
        ref_text = "reference transcript"
        text = "a short sentence"

        ref_model = _Qwen3TTSForTextToSpeech.from_pretrained(model_id)

        with TemporaryDirectory() as tmpdir:
            # The checkpoint is bfloat16 and the loader keeps that precision, so tracing needs the
            # 16-bit patch that ``main_export`` would otherwise derive from the loaded model.
            export_from_model(
                model=ref_model, output=tmpdir, task="text-to-audio", stateful=True, patch_16bit_model=True
            )

            # Every neural component must be exported; the runtime has no weights to fall back on.
            for name in (
                "talker_model",
                "code_predictor_model",
                "text_embeddings",
                "talker_embeddings",
                "code_predictor_embeddings",
                "speaker_encoder",
                "codec_encoder",
                "codec_decoder",
            ):
                self.assertTrue(Path(tmpdir, f"openvino_{name}.xml").is_file(), f"missing IR for {name}")
            self.assertFalse(list(Path(tmpdir).glob("*.safetensors")), "export should not ship PyTorch weights")
            self.assertFalse(
                list(Path(tmpdir, "speech_tokenizer").glob("*.safetensors")),
                "export should not ship codec weights",
            )

            # f32, as the speaker encoder is compared to PyTorch: a bf16/f16 default inference precision (e.g. CPUs
            # with AMX) moves its output by ~2e-3, past the tolerance below.
            ov_model = self.OVMODEL_CLASS.from_pretrained(tmpdir, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
            self.assertIsInstance(ov_model, _OVModelForQwen3TTS)
            self.assertEqual(getattr(ov_model.config, "model_type", None), "qwen3_tts")
            self.assertEqual(sum(p.numel() for p in ov_model.model.parameters()), 0)
            # The stateful decoder stacks overflow f16 on GPU without the activations scale.
            for name, component in ov_model.components.items():
                has_scale = component.model.has_rt_info(["runtime_options", "ACTIVATIONS_SCALE_FACTOR"])
                self.assertEqual(has_scale, name in ("talker_model", "code_predictor_model"), name)

            inputs = ov_model.preprocess_input(
                text=text, language="English", ref_audio=(ref_audio, sampling_rate), ref_text=ref_text
            )
            set_seed(SEED)
            ov_waveform = ov_model.generate(**inputs, max_new_tokens=16)

            # The speaker encoder and the codec are deterministic given the same reference audio,
            # so they are compared directly; the talker's sampled codes are not.
            with torch.no_grad():
                ref_speaker_embedding = ref_pipeline.model.extract_speaker_embedding(ref_audio, sr=sampling_rate)
            ov_speaker_embedding = inputs["voice_clone_prompt"][0].ref_spk_embedding
            self.assertEqual(ov_speaker_embedding.shape, ref_speaker_embedding.shape)
            self.assertTrue(
                torch.allclose(ov_speaker_embedding.float(), ref_speaker_embedding.float(), atol=1e-3),
                "speaker encoder output diverged from PyTorch",
            )

            self.assertGreater(ov_waveform.numel(), 0)
            self.assertTrue(torch.isfinite(ov_waveform).all(), "generated waveform contains NaN/inf")
            self.assertGreater(float(ov_waveform.abs().max()), 1e-4, "generated waveform is silent")

        del ref_model
        del ref_pipeline
        del ov_model
        gc.collect()

    def test_compare_to_kokoro(self):
        from optimum.exporters.openvino import export_from_model
        from optimum.exporters.tasks import TasksManager
        from optimum.intel.openvino.modeling_text2speech import _OVModelForKokoroTextToSpeech

        set_seed(SEED)
        model_id = MODEL_NAMES["kokoro"]

        ref_model = TasksManager.get_model_from_task(
            task="text-to-audio",
            model_name_or_path=model_id,
            framework="pt",
            library_name="kokoro",
        )
        ref_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        ref_s = torch.randn(1, 256, dtype=torch.float32)

        with torch.no_grad():
            ref_waveform, ref_phonemes = ref_model.forward_with_tokens(input_ids, ref_s, 1.0)

        with TemporaryDirectory() as tmpdir:
            export_from_model(model=ref_model, output=tmpdir, task="text-to-audio", stateful=True)

            # Verify that local voice .pt files were converted to .bin (regression for local export bug)
            source_voices = sorted(Path(model_id).glob("voices/*.pt"))
            self.assertGreater(len(source_voices), 0, "Test fixture has no source voice .pt files")
            exported_voices = sorted(Path(tmpdir).glob("voices/*.bin"))
            self.assertEqual(
                len(exported_voices),
                len(source_voices),
                f"Expected {len(source_voices)} voice .bin files, found {len(exported_voices)}",
            )
            for bin_path in exported_voices:
                self.assertGreater(bin_path.stat().st_size, 0, f"{bin_path.name} is empty")

            ov_model = self.OVMODEL_CLASS.from_pretrained(tmpdir, device=OPENVINO_DEVICE)

            self.assertIsInstance(ov_model, _OVModelForKokoroTextToSpeech)
            self.assertIsInstance(ov_model.config, PretrainedConfig)
            self.assertEqual(getattr(ov_model.config, "model_type", None), "kokoro")
            self.assertIsInstance(ov_model.model, openvino.Model)

            ov_outputs = ov_model.forward(input_ids=input_ids, ref_s=ref_s, speed=1.0)
            self.assertTrue(hasattr(ov_outputs, "waveform"))
            self.assertTrue(hasattr(ov_outputs, "phonemes"))
            self.assertEqual(ov_outputs.waveform.shape, ref_waveform.shape)
            self.assertEqual(ov_outputs.phonemes.shape, ref_phonemes.shape)
            self.assertTrue(torch.equal(ov_outputs.phonemes, ref_phonemes))

            ov_generated = ov_model.generate(input_ids=input_ids, ref_s=ref_s, speed=1.0)
            self.assertEqual(ov_generated.shape, ref_waveform.shape)

        del ref_model
        del ov_model
        gc.collect()

    @parameterized.expand(TTS_SUPPORTED_ARCHITECTURES)
    def test_model_can_be_loaded_after_saving(self, model_arch):
        if model_arch == "speecht5":
            self.skipTest("SpeechT5 requires a vocoder, so the test needs updating to test this model.")
        if model_arch == "qwen3_tts" and not is_qwen_tts_available():
            self.skipTest("qwen_tts package is not installed")
        model_id = MODEL_NAMES[model_arch]
        with TemporaryDirectory() as save_dir:
            ov_model = self.OVMODEL_CLASS.from_pretrained(
                model_id,
                compile=False,
                trust_remote_code=model_arch in REMOTE_CODE_MODELS,
                device=OPENVINO_DEVICE,
            )
            ov_model.save_pretrained(save_dir)
            ov_restored_model = self.OVMODEL_CLASS.from_pretrained(
                save_dir,
                compile=False,
                trust_remote_code=model_arch in REMOTE_CODE_MODELS,
                device=OPENVINO_DEVICE,
            )
            self.assertIsInstance(ov_restored_model, type(ov_model))
