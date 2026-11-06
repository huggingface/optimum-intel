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
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, set_seed
from transformers.generation import GenerationMixin
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.testing_utils import slow
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, SEED

from optimum.exporters.openvino.stateful import model_has_state
from optimum.intel import OVModelForSpeechSeq2Seq
from optimum.intel.openvino.modeling_seq2seq import OVEncoderOutput, _OVModelForCohereAsr, _register_for_asr_pipeline
from optimum.intel.utils.import_utils import is_transformers_version


class OVASRTest(unittest.TestCase):
    """
    Test ASR model types (Qwen3-ASR, FunASR, Cohere ASR).
    Compares OpenVINO model output to original PyTorch model output.
    Cohere ASR additionally gets behavior-specific tests below since it is a native
    (non trust-remote-code) model with export edge cases the other architectures don't hit.
    """

    # Qwen3-ASR, FunASR and SenseVoice are trust-remote-code models we currently support only for transformers==4.57
    REMOTE_CODE_ARCHITECTURES = ()
    if is_transformers_version(">=", "4.57") and is_transformers_version("<", "4.58"):
        REMOTE_CODE_ARCHITECTURES += ("qwen3_asr", "fun_asr", "sense_voice")
    SUPPORTED_ARCHITECTURES = REMOTE_CODE_ARCHITECTURES
    if "cohere_asr" in CONFIG_MAPPING_NAMES:
        SUPPORTED_ARCHITECTURES += ("cohere_asr",)

    def _generate_audio_data(self):
        np.random.seed(SEED)
        sample_rate = 16000
        duration = 120
        t = np.linspace(0, 1.0, sample_rate * duration, endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return audio_data, sample_rate

    def _compare_generated_text(self, model_id, ref, trust_remote_code):
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, trust_remote_code=trust_remote_code, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ref = self._get_pt_reference(model_arch)
        self._compare_generated_text(model_id, ref, trust_remote_code=model_arch in self.REMOTE_CODE_ARCHITECTURES)

    @parameterized.expand(REMOTE_CODE_ARCHITECTURES, skip_on_empty=True)
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
    def test_fun_asr_audio_features_truncated_to_prompt_length(self):
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
            encoder_outputs.last_hidden_state if hasattr(encoder_outputs, "last_hidden_state") else encoder_outputs[0]
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

    @pytest.mark.skipif(
        is_transformers_version("<", "4.57") or is_transformers_version(">=", "4.58"),
        reason="Currently, we support FunASR only for transformers==4.57 since it is a trust-remote-code model.",
    )
    def test_fun_asr_activations_scale_factor(self):
        # The Fun-ASR-Nano decoder overflows fp16 on GPU without activation scaling, and the GPU plugin
        # ignores the default 8.0 for LLM-like models, so the decoder must carry a non-default value.
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAMES["fun_asr"], export=True, trust_remote_code=True, compile=False
        )
        decoder_rt_info = ov_model.decoder.model.get_rt_info()
        self.assertEqual(decoder_rt_info["runtime_options"]["ACTIVATIONS_SCALE_FACTOR"].value, "4.0")
        self.assertFalse(ov_model.encoder.model.has_rt_info(["runtime_options", "ACTIVATIONS_SCALE_FACTOR"]))

        del ov_model
        gc.collect()

    def _get_pt_reference(self, model_arch):
        if model_arch == "fun_asr":
            return self._get_pt_reference_funasr()
        if model_arch == "cohere_asr":
            return self._get_pt_reference_cohere_asr()
        if model_arch == "sense_voice":
            return self._get_pt_reference_sense_voice()
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

    def _get_pt_reference_cohere_asr(self):
        model_id = MODEL_NAMES["cohere_asr"]
        processor = AutoProcessor.from_pretrained(model_id)

        np.random.seed(SEED)
        audio_data = (np.random.randn(16000 * 5).astype(np.float32)) * 0.01
        inputs = processor(audio_data, language="en", sampling_rate=16000, return_tensors="pt")
        inputs.pop("audio_chunk_index", None)

        transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        transformers_model.eval()

        decoder_start_token_id = getattr(transformers_model.config, "decoder_start_token_id", None) or 0
        decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        gen_kwargs = {"max_new_tokens": 8, "do_sample": False, "num_beams": 1}

        with torch.no_grad():
            pt_generated_ids = transformers_model.generate(
                input_features=inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                decoder_input_ids=decoder_input_ids,
                **gen_kwargs,
            )
        if hasattr(pt_generated_ids, "sequences"):
            pt_generated_ids = pt_generated_ids.sequences

        prompt_len = decoder_input_ids.shape[1]
        pt_text = processor.batch_decode(pt_generated_ids[:, prompt_len:], skip_special_tokens=True)[0]

        return {
            "input_features": inputs["input_features"],
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": inputs.get("attention_mask"),
            "pt_text": pt_text,
            "gen_kwargs": gen_kwargs,
            "decode_fn": lambda ids, prompt_len: processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)[
                0
            ],
            "preprocess_check": None,
            "pt_model": transformers_model,
        }

    @pytest.mark.run_slow
    @slow
    @pytest.mark.skipif("cohere_asr" not in CONFIG_MAPPING_NAMES, reason="cohere_asr requires a newer transformers")
    def test_cohere_asr_generate_non_30s_multiple_audio(self):
        # The encoder used to inherit the Whisper dummy generator, which pins input_features to
        # 3000 frames, so any audio that was not a multiple of 30s failed at inference time
        model_id = MODEL_NAMES["cohere_asr"]
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
        processor = AutoProcessor.from_pretrained(model_id)

        encoder_shapes = {
            encoder_input.get_any_name(): encoder_input.get_partial_shape()
            for encoder_input in model.encoder.model.inputs
        }
        self.assertIn("attention_mask", encoder_shapes, "encoder must expose an `attention_mask` input")
        input_features_shape = encoder_shapes["input_features"]
        self.assertTrue(
            input_features_shape[1].is_dynamic,
            f"encoder `input_features` time dim must be dynamic, got {input_features_shape}",
        )
        encoder_output_names = {encoder_output.get_any_name() for encoder_output in model.encoder.model.outputs}
        self.assertIn(
            "encoder_attention_mask",
            encoder_output_names,
            "encoder must return the subsampled mask that cross attention runs against",
        )

        np.random.seed(SEED)
        for duration_in_seconds in (3, 7, 11):
            audio = (np.random.randn(16000 * duration_in_seconds).astype(np.float32)) * 0.01
            inputs = processor(audio, language="en", sampling_rate=16000, return_tensors="pt")
            inputs.pop("audio_chunk_index", None)
            generated_tokens = model.generate(**inputs, max_new_tokens=8)
            self.assertEqual(generated_tokens.shape[0], 1)
            self.assertGreater(generated_tokens.shape[1], 1)

        del model
        gc.collect()

    @pytest.mark.run_slow
    @slow
    @pytest.mark.skipif("cohere_asr" not in CONFIG_MAPPING_NAMES, reason="cohere_asr requires a newer transformers")
    def test_cohere_asr_padded_batch_matches_single(self):
        # Clips of different lengths are padded to a common size, and the frame level mask has to
        # survive the eightfold subsampling for cross attention to skip the padded tail
        model_id = MODEL_NAMES["cohere_asr"]
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
        processor = AutoProcessor.from_pretrained(model_id)

        np.random.seed(SEED)
        long_audio = (np.random.randn(16000 * 9).astype(np.float32)) * 0.01
        short_audio = long_audio[: 16000 * 4]

        generate_kwargs = {"max_new_tokens": 8, "do_sample": False, "num_beams": 1}
        batched_inputs = processor([long_audio, short_audio], language="en", sampling_rate=16000, return_tensors="pt")
        batched_inputs.pop("audio_chunk_index", None)
        batched_tokens = model.generate(**batched_inputs, **generate_kwargs)

        for row, audio in enumerate([long_audio, short_audio]):
            single_inputs = processor(audio, language="en", sampling_rate=16000, return_tensors="pt")
            single_inputs.pop("audio_chunk_index", None)
            single_tokens = model.generate(**single_inputs, **generate_kwargs)
            self.assertTrue(torch.equal(batched_tokens[row : row + 1], single_tokens))

        del model
        gc.collect()

    @pytest.mark.run_slow
    @slow
    @pytest.mark.skipif("cohere_asr" not in CONFIG_MAPPING_NAMES, reason="cohere_asr requires a newer transformers")
    def test_cohere_asr_beam_search_padded_batch_matches_single(self):
        # Beam search repeats the encoder states for every beam, so the subsampled mask has to be repeated
        # with them or cross attention sees a mask of the wrong batch size
        model_id = MODEL_NAMES["cohere_asr"]
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
        processor = AutoProcessor.from_pretrained(model_id)

        np.random.seed(SEED)
        long_audio = (np.random.randn(16000 * 9).astype(np.float32)) * 0.01
        short_audio = long_audio[: 16000 * 4]

        generate_kwargs = {"max_new_tokens": 8, "do_sample": False, "num_beams": 2}

        # Noise audio can yield identical tokens even with a misaligned mask, so also check the shapes the decoder gets
        mismatched_shapes = []
        decoder_forward = model.decoder.forward

        def checked_decoder_forward(*args, **kwargs):
            states, encoder_mask = kwargs["encoder_hidden_states"], kwargs.get("encoder_attention_mask")
            if encoder_mask is not None and encoder_mask.shape[0] != states.shape[0]:
                mismatched_shapes.append((tuple(states.shape), tuple(encoder_mask.shape)))
            return decoder_forward(*args, **kwargs)

        model.decoder.forward = checked_decoder_forward
        batched_inputs = processor([long_audio, short_audio], language="en", sampling_rate=16000, return_tensors="pt")
        batched_inputs.pop("audio_chunk_index", None)
        batched_tokens = model.generate(**batched_inputs, **generate_kwargs)
        self.assertEqual(batched_tokens.shape[0], 2)
        self.assertEqual(mismatched_shapes, [])

        for row, audio in enumerate([long_audio, short_audio]):
            single_inputs = processor(audio, language="en", sampling_rate=16000, return_tensors="pt")
            single_inputs.pop("audio_chunk_index", None)
            single_tokens = model.generate(**single_inputs, **generate_kwargs)
            common_length = min(batched_tokens.shape[1], single_tokens.shape[1])
            self.assertTrue(
                torch.equal(batched_tokens[row : row + 1, :common_length], single_tokens[:, :common_length])
            )

        del model
        gc.collect()

    def test_encoder_output_attention_mask_is_expanded_for_beam_search(self):
        batch_size, num_beams, num_frames = 2, 2, 5
        encoder_outputs = OVEncoderOutput(
            last_hidden_state=torch.randn(batch_size, num_frames, 4),
            attention_mask=torch.ones(batch_size, num_frames, dtype=torch.long),
        )
        _, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            expand_size=num_beams,
            is_encoder_decoder=True,
            input_ids=torch.zeros(batch_size, 1, dtype=torch.long),
            encoder_outputs=encoder_outputs,
        )
        expanded = model_kwargs["encoder_outputs"]
        self.assertEqual(expanded.last_hidden_state.shape[0], batch_size * num_beams)
        self.assertEqual(expanded.attention_mask.shape[0], batch_size * num_beams)

    def test_asr_pipeline_registration_is_limited_to_cohere_asr_class(self):
        from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES

        _register_for_asr_pipeline(_OVModelForCohereAsr)
        registered_names = set(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values())
        self.assertIn(_OVModelForCohereAsr.__name__, registered_names)
        self.assertNotIn(OVModelForSpeechSeq2Seq.__name__, registered_names)

    @pytest.mark.run_slow
    @slow
    @pytest.mark.skipif("cohere_asr" not in CONFIG_MAPPING_NAMES, reason="cohere_asr requires a newer transformers")
    def test_cohere_asr_with_past_decoder_is_stateful(self):
        # The with-past export used to produce a plain decoder without beam_idx or KV cache state,
        # which stateful consumers such as openvino_genai.WhisperPipeline require
        model_id = MODEL_NAMES["cohere_asr"]
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE, stateful=True)
        self.assertTrue(model_has_state(model.decoder.model))
        decoder_input_names = {decoder_input.get_any_name() for decoder_input in model.decoder.model.inputs}
        self.assertIn("beam_idx", decoder_input_names)
        self.assertFalse(
            any(name.startswith("past_key_values") for name in decoder_input_names),
            "the cache has to live in the graph rather than be handed in as inputs",
        )
        self.assertGreater(len(model.decoder.model.get_sinks()), 0)

        # More than one decode step also has to run, since the cache length used to be baked into
        # the graph as a constant while tracing
        processor = AutoProcessor.from_pretrained(model_id)
        np.random.seed(SEED)
        audio = (np.random.randn(16000 * 5).astype(np.float32)) * 0.01
        inputs = processor(audio, language="en", sampling_rate=16000, return_tensors="pt")
        inputs.pop("audio_chunk_index", None)
        generated_tokens = model.generate(**inputs, max_new_tokens=12)
        self.assertGreater(generated_tokens.shape[1], 1)

        del model
        gc.collect()

    @pytest.mark.run_slow
    @slow
    @pytest.mark.skipif("cohere_asr" not in CONFIG_MAPPING_NAMES, reason="cohere_asr requires a newer transformers")
    def test_cohere_asr_exported_processor_is_self_contained(self):
        # Reloading the processor straight from an export directory has to work without copying
        # tokenizer files over by hand
        model_id = MODEL_NAMES["cohere_asr"]
        with TemporaryDirectory() as tmp_dir:
            model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
            model.save_pretrained(tmp_dir)
            processor = AutoProcessor.from_pretrained(model_id)
            processor.save_pretrained(tmp_dir)

            reloaded_processor = AutoProcessor.from_pretrained(tmp_dir)
            self.assertIsNotNone(reloaded_processor.tokenizer)
            del model
            gc.collect()

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
