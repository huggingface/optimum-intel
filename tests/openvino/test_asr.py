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
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.testing_utils import slow
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, REMOTE_CODE_MODELS, SEED

from optimum.exporters.openvino.stateful import model_has_state
from optimum.intel import OVModelForSpeechSeq2Seq
from optimum.intel.utils.import_utils import is_transformers_version


class OVASRTest(unittest.TestCase):
    """
    Test ASR model types (Qwen3-ASR, FunASR, Cohere ASR).
    Compares OpenVINO model output to original PyTorch model output.
    Cohere ASR additionally gets behavior-specific tests below since it is a native
    (non trust-remote-code) model with export edge cases the other architectures don't hit.
    """

    SUPPORTED_ARCHITECTURES = ()
    # Qwen3-ASR and FunASR are trust-remote-code models we currently support only for transformers==4.57
    if is_transformers_version(">=", "4.57") and is_transformers_version("<", "4.58"):
        SUPPORTED_ARCHITECTURES += ("qwen3_asr", "fun_asr")
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
            "decoder_input_ids": ref["decoder_input_ids"],
            **ref["gen_kwargs"],
        }
        if ref["attention_mask"] is not None:
            ov_gen_kwargs["attention_mask"] = ref["attention_mask"]

        ov_generated_ids = ov_model.generate(**ov_gen_kwargs)
        if hasattr(ov_generated_ids, "sequences"):
            ov_generated_ids = ov_generated_ids.sequences

        prompt_len = ref["decoder_input_ids"].shape[1]
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
        self._compare_generated_text(model_id, ref, trust_remote_code=model_arch in REMOTE_CODE_MODELS)

    def _get_pt_reference(self, model_arch):
        if model_arch == "fun_asr":
            return self._get_pt_reference_funasr()
        elif model_arch == "cohere_asr":
            return self._get_pt_reference_cohere_asr()
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
