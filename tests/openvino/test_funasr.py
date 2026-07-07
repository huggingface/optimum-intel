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

import gc
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest
import torch
from transformers import set_seed
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, SEED

from optimum.intel import OVModelForSpeechSeq2Seq
from optimum.intel.utils.import_utils import is_transformers_version


class OVFunASRTest(unittest.TestCase):
    """
    Test FunASR model type in its own CI group.
    Compares OpenVINO model output to original FunASR model output.
    """

    def _generate_audio_data(self):
        np.random.seed(SEED)
        sample_rate = 16000
        duration = 120
        t = np.linspace(0, 1.0, sample_rate * duration, endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return audio_data, sample_rate

    @pytest.mark.skipif(
        is_transformers_version("!=", "4.57.6"),
        reason="requires transformers==4.57.6.",
    )
    def test_compare_to_funasr(self):
        set_seed(SEED)
        ref = self._get_pt_reference()

        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAMES["fun_asr"], export=True, trust_remote_code=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )

        pc = ref["preprocess_check"]
        ov_inputs = ov_model.preprocess_input(pc["waveform"], pc["sampling_rate"], language="中文")
        self.assertEqual(ov_inputs["input_features"].shape, ref["input_features"].shape)
        self.assertTrue(torch.equal(ov_inputs["decoder_input_ids"], ref["decoder_input_ids"]))

        ov_generated_ids = ov_model.generate(
            input_features=ref["input_features"],
            decoder_input_ids=ref["decoder_input_ids"],
            **ref["gen_kwargs"],
        )
        if hasattr(ov_generated_ids, "sequences"):
            ov_generated_ids = ov_generated_ids.sequences

        prompt_len = ref["decoder_input_ids"].shape[1]
        ov_text = ref["decode_fn"](ov_generated_ids, prompt_len)

        self.assertEqual(ref["pt_text"], ov_text)

        del ref["pt_model"]
        del ov_model
        gc.collect()

    def _get_pt_reference(self):
        from funasr import AutoModel as FunASRAutoModel

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            funasr_model = FunASRAutoModel(
                model=MODEL_NAMES["fun_asr"], hub="hf", trust_remote_code=True, device="cpu", disable_update=True
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
            "pt_text": pt_text,
            "gen_kwargs": gen_kwargs,
            "decode_fn": lambda ids, prompt_len: tokenizer.decode(ids[0][prompt_len:].tolist(), skip_special_tokens=True).strip(),
            "preprocess_check": {"waveform": audio_data, "sampling_rate": sample_rate},
            "pt_model": funasr_model,
        }
