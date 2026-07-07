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
import unittest

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import AutoProcessor, set_seed
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, SEED

from optimum.intel import OVModelForSpeechSeq2Seq
from optimum.intel.utils.import_utils import is_transformers_version


class Qwen3ASRTest(unittest.TestCase):
    """
    Test Qwen3-ASR model type in its own CI group.
    Compares OpenVINO model output to original PyTorch transformers model output.
    """

    SUPPORTED_ARCHITECTURES = ("qwen3_asr",)

    def _generate_audio_data(self):
        np.random.seed(SEED)
        sample_rate = 16000
        duration = 120
        t = np.linspace(0, 1.0, sample_rate * duration, endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return audio_data, sample_rate

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.skipif(
        is_transformers_version("!=", "4.57.6"),
        reason="requires transformers==4.57.6.",
    )
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)

        ref = self._get_pt_reference(model_arch)

        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, trust_remote_code=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )

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

    def _get_pt_reference(self, model_arch):
        """
        Obtain PyTorch reference: input_features, decoder_input_ids, pt_text, gen_kwargs, decode_fn.
        Returns a dict with keys consumed by test_compare_to_transformers.
        """
        model_id = MODEL_NAMES[model_arch]
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

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
            "pt_model": transformers_model,
        }
