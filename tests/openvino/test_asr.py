#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

import gc
import unittest

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import AutoProcessor, set_seed
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, SEED

from optimum.intel import OVModelForSpeechSeq2Seq
from optimum.intel.utils.import_utils import is_transformers_version


class OVASRTest(unittest.TestCase):
    """
    Test ASR model types (Qwen3-ASR, FunASR).
    Compares OpenVINO model output to original PyTorch model output.
    """

    SUPPORTED_ARCHITECTURES = ("qwen3_asr", "fun_asr", "sense_voice")

    def _generate_audio_data(self):
        np.random.seed(SEED)
        sample_rate = 16000
        duration = 120
        t = np.linspace(0, 1.0, sample_rate * duration, endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return audio_data, sample_rate

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.skipif(
        is_transformers_version("<", "4.57") or is_transformers_version(">=", "4.58"),
        reason="Currently, we support Qwen3-ASR and FunASR only for transformers==4.57 since they are trust-remote-code models.",
    )
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)

        ref = self._get_pt_reference(model_arch)

        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, trust_remote_code=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )

        # For models with standalone preprocess_input, verify it reproduces the reference inputs.
        if ref.get("preprocess_check") is not None:
            pc = ref["preprocess_check"]
            ov_inputs = ov_model.preprocess_input(pc["waveform"], pc["sampling_rate"], language="中文")
            self.assertEqual(ov_inputs["input_features"].shape, ref["input_features"].shape)
            self.assertTrue(torch.equal(ov_inputs["decoder_input_ids"], ref["decoder_input_ids"]))

        # Generate with OV model using the exact PT-produced inputs.
        ov_gen_kwargs = {
            "input_features": ref["input_features"],
            **ref["gen_kwargs"],
        }
        if ref.get("decoder_input_ids") is not None:
            ov_gen_kwargs["decoder_input_ids"] = ref["decoder_input_ids"]

        if ref.get("speech_lengths") is not None:
            ov_gen_kwargs["speech_lengths"] = ref["speech_lengths"]

        if ref["attention_mask"] is not None:
            ov_gen_kwargs["attention_mask"] = ref["attention_mask"]

        ov_generated_ids = ov_model.generate(**ov_gen_kwargs)
        if hasattr(ov_generated_ids, "sequences"):
            ov_generated_ids = ov_generated_ids.sequences

        if ref.get("decoder_input_ids") is not None:
            prompt_len = ref["decoder_input_ids"].shape[1]
        else:
            prompt_len = None
        ov_text = ref["decode_fn"](ov_generated_ids, prompt_len)

        self.assertEqual(ref["pt_text"], ov_text)

        del ref["pt_model"]
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.skipif(
        is_transformers_version("<", "4.57") or is_transformers_version(">=", "4.58"),
        reason="Currently, we support Qwen3-ASR and FunASR only for transformers==4.57 since they are trust-remote-code models.",
    )
    def test_compile_false(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)

        ref = self._get_pt_reference(model_arch)

        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            trust_remote_code=True,
            ov_config=F32_CONFIG,
            device=OPENVINO_DEVICE,
            compile=False,
        )

        ov_gen_kwargs = {"input_features": ref["input_features"], **ref["gen_kwargs"]}
        if ref.get("decoder_input_ids") is not None:
            ov_gen_kwargs["decoder_input_ids"] = ref["decoder_input_ids"]
        if ref.get("speech_lengths") is not None:
            ov_gen_kwargs["speech_lengths"] = ref["speech_lengths"]
        if ref["attention_mask"] is not None:
            ov_gen_kwargs["attention_mask"] = ref["attention_mask"]

        ov_generated_ids = ov_model.generate(**ov_gen_kwargs)
        if hasattr(ov_generated_ids, "sequences"):
            ov_generated_ids = ov_generated_ids.sequences

        prompt_len = ref["decoder_input_ids"].shape[1] if ref.get("decoder_input_ids") is not None else None
        ov_text = ref["decode_fn"](ov_generated_ids, prompt_len)
        self.assertEqual(ref["pt_text"], ov_text)

        del ref["pt_model"]
        del ov_model
        gc.collect()

    @pytest.mark.skipif(
        is_transformers_version("<", "4.57") or is_transformers_version(">=", "4.58"),
        reason="Currently, we support Qwen3-ASR and FunASR only for transformers==4.57 since they are trust-remote-code models.",
    )
    def test_funasr_audio_features_truncated_to_prompt_length(self):
        """Regression test for the FunASR OpenVINO decoder runaway / WER inflation.

        The exported audio encoder emits one feature per frame, which can be longer than the
        number of audio placeholder tokens the prompt reserves (``fake_token_len`` in native
        FunASR). Native inference splices only the first ``fake_token_len`` adaptor frames into
        the LLM input embeddings. The OpenVINO path used to instead *expand* the audio
        placeholder run up to the full encoder length, feeding extra, misaligned audio frames
        into the decoder. On some utterances this made greedy decoding fall into a repetition
        loop (e.g. "shame on you ... on you on you ...") and inflated WER. The fix truncates the
        encoder features to the prompt's placeholder count; this test asserts that invariant.
        """
        from optimum.intel.openvino import modeling_seq2seq

        model_id = MODEL_NAMES["fun_asr"]
        set_seed(SEED)
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, trust_remote_code=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        audio_token_id = getattr(ov_model.config, "audio_token_id", 0)

        audio_data, sample_rate = self._generate_audio_data()
        inputs = ov_model.preprocess_input(audio_data, sample_rate, language="中文")
        input_features = inputs["input_features"]

        # Number of audio feature frames produced by the exported encoder.
        encoder_outputs = ov_model.encoder(input_ids=input_features)
        enc_hidden = (
            encoder_outputs.last_hidden_state
            if hasattr(encoder_outputs, "last_hidden_state")
            else encoder_outputs[0]
        )
        num_encoder_features = enc_hidden.shape[1]
        self.assertGreater(num_encoder_features, 1)

        # Build a prompt whose audio placeholder run is shorter than the encoder output,
        # reproducing the native situation where fake_token_len < encoder length.
        target_audio_tokens = max(1, num_encoder_features // 2)
        self.assertLess(target_audio_tokens, num_encoder_features)
        decoder_input_ids = torch.tensor(
            [[1, 2, 3] + [audio_token_id] * target_audio_tokens + [4, 5]], dtype=torch.long
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        # Capture what the FunASR forward passes down to the base seq2seq forward.
        captured = {}
        orig_forward = modeling_seq2seq.OVModelForSpeechSeq2Seq.forward

        def capturing_forward(self, *args, **kwargs):
            did = kwargs.get("decoder_input_ids")
            enc_out = kwargs.get("encoder_outputs")
            enc_h = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]
            captured["audio_count"] = int((did == audio_token_id).sum(dim=-1).max())
            captured["encoder_len"] = int(torch.as_tensor(enc_h).shape[1])
            return orig_forward(self, *args, **kwargs)

        modeling_seq2seq.OVModelForSpeechSeq2Seq.forward = capturing_forward
        try:
            ov_model.forward(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
        finally:
            modeling_seq2seq.OVModelForSpeechSeq2Seq.forward = orig_forward

        # Native behavior: the placeholder run is kept as-is (NOT expanded to the encoder
        # length) and the encoder features are truncated to match the placeholder count.
        self.assertEqual(captured["audio_count"], target_audio_tokens)
        self.assertEqual(captured["encoder_len"], target_audio_tokens)
        self.assertNotEqual(captured["encoder_len"], num_encoder_features)

        del ov_model
        gc.collect()

    def _get_pt_reference(self, model_arch):
        if model_arch == "fun_asr":
            return self._get_pt_reference_funasr()
        if model_arch == "sense_voice":
            return self._get_pt_reference_sense_voice()
        else:
            return self._get_pt_reference_qwen3_asr()

    def _get_pt_reference_funasr(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from funasr import AutoModel as FunASRAutoModel

        model_id = MODEL_NAMES["fun_asr"]
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            funasr_model = FunASRAutoModel(
                model=model_id, hub="hf", trust_remote_code=True, device="cpu", disable_update=True
            )
        core = funasr_model.model
        kwargs = dict(funasr_model.kwargs)
        tokenizer = kwargs["tokenizer"]

        audio_data, sample_rate = self._generate_audio_data()
        audio_tensor = torch.from_numpy(audio_data)

        captured = {}
        orig_prepare = core.inference_prepare

        def _capture(*args, **kw):
            inputs_embeds, contents, batch, source_ids, meta = orig_prepare(*args, **kw)
            captured["speech"] = batch["speech"]
            captured["source_ids"] = source_ids
            return inputs_embeds, contents, batch, source_ids, meta

        gen_kwargs = {"max_new_tokens": 64}

        core.inference_prepare = _capture
        # The funasr library prints verbose progress/debug output
        # to stdout and stderr during inference (progress bars,
        # per-step logs). The redirect suppresses that noise
        # so it doesn't pollute the test output.
        with redirect_stdout(buf), redirect_stderr(buf):
            pt_result = funasr_model.generate(
                input=[audio_tensor],
                cache={},
                batch_size=1,
                language="中文",
                itn=True,
                max_length=gen_kwargs["max_new_tokens"],
            )
        core.inference_prepare = orig_prepare
        pt_text = pt_result[0]["text"].strip()

        return {
            "input_features": captured["speech"].float(),
            "decoder_input_ids": captured["source_ids"],
            "attention_mask": None,
            "pt_text": pt_text,
            "gen_kwargs": gen_kwargs,
            "decode_fn": lambda ids, prompt_len: tokenizer.decode(
                ids[0][prompt_len:].tolist(), skip_special_tokens=True
            ).strip(),
            "preprocess_check": {"waveform": audio_data, "sampling_rate": sample_rate},
            "pt_model": funasr_model,
        }

    def _get_pt_reference_qwen3_asr(self):
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

        model_id = MODEL_NAMES["qwen3_asr"]
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        audio_data, sample_rate = self._generate_audio_data()
        text_prompt = processor.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=text_prompt,
            audio=audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        transformers_model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True)
        transformers_model.eval()

        gen_kwargs = {"max_new_tokens": 10}

        with torch.no_grad():
            pt_generated_ids = transformers_model.generate(
                input_ids=inputs["input_ids"],
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
        if hasattr(pt_generated_ids, "sequences"):
            pt_generated_ids = pt_generated_ids.sequences

        prompt_len = inputs["input_ids"].shape[1]
        pt_text = processor.batch_decode(pt_generated_ids[:, prompt_len:], skip_special_tokens=True)[0]

        return {
            "input_features": inputs["input_features"],
            "decoder_input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("feature_attention_mask"),
            "pt_text": pt_text,
            "gen_kwargs": gen_kwargs,
            "decode_fn": lambda ids, prompt_len: processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)[
                0
            ],
            "preprocess_check": None,
            "pt_model": transformers_model,
        }

    def _get_pt_reference_sense_voice(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from funasr import AutoModel
        from funasr.utils.load_utils import extract_fbank

        model_id = MODEL_NAMES["sense_voice"]
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            pt_model = AutoModel(model=model_id, hub="hf", trust_remote_code=True, device="cpu", disable_update=True)

        audio_data, sample_rate = self._generate_audio_data()
        audio_tensor = torch.from_numpy(audio_data)

        speech, speech_lengths = extract_fbank(audio_tensor, data_type="sound", frontend=pt_model.kwargs["frontend"])

        with redirect_stdout(buf), redirect_stderr(buf):
            args = pt_model.kwargs
            args["data_type"] = "fbank"
            pt_result = pt_model.model.inference(speech, speech_lengths, **args)
        pt_text = pt_result[0][0]["text"]

        return {
            "input_features": speech,
            "speech_lengths": speech_lengths,
            "decode_fn": lambda ids, prompt_len: ids[0],
            "pt_text": pt_text,
            "preprocess_check": None,
            "gen_kwargs": {},
            "pt_model": pt_model,
            "attention_mask": None,
        }
