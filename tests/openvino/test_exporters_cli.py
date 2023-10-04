# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
import unittest
from tempfile import TemporaryDirectory

from parameterized import parameterized
from utils_tests import MODEL_NAMES, get_num_quantized_nodes, _ARCHITECTURES_TO_EXPECTED_INT8

from optimum.exporters.openvino.__main__ import main_export
from optimum.intel import (  # noqa
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLPipeline,
)
from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS



class OVCLIExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    SUPPORTED_ARCHITECTURES = (
        ("text-generation", "gpt2"),
        ("text-generation-with-past", "gpt2"),
        ("text2text-generation", "t5"),
        ("text2text-generation-with-past", "t5"),
        ("text-classification", "albert"),
        ("question-answering", "distilbert"),
        ("token-classification", "roberta"),
        ("image-classification", "vit"),
        ("audio-classification", "wav2vec2"),
        ("fill-mask", "bert"),
        ("feature-extraction", "blenderbot"),
        ("stable-diffusion", "stable-diffusion"),
        ("stable-diffusion-xl", "stable-diffusion-xl"),
        ("stable-diffusion-xl", "stable-diffusion-xl-refiner"),
    )

    def _openvino_export(self, model_name: str, task: str, fp16: bool = False, int8: bool = False):
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path=model_name, output=tmpdir, task=task, fp16=fp16, int8=int8)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_export(self, task: str, model_type: str):
        self._openvino_export(MODEL_NAMES[model_type], task)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_exporters_cli(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} {tmpdir}",
                shell=True,
                check=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            eval(_HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]).from_pretrained(tmpdir, **model_kwargs)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_exporters_cli_fp16(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} --fp16 {tmpdir}",
                shell=True,
                check=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            eval(_HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]).from_pretrained(tmpdir, **model_kwargs)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_exporters_cli_int8(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} --int8 {tmpdir}",
                shell=True,
                check=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            model = eval(_HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]).from_pretrained(tmpdir, **model_kwargs)

            if task.startswith("text2text-generation"):
                models = [model.encoder, model.decoder]
                if task.endswith("with-past"):
                    models.append(model.decoder_with_past)
            elif task.startswith("stable-diffusion"):
                models = [model.unet, model.vae_encoder, model.vae_decoder]
                models.append(model.text_encoder if task == "stable-diffusion" else model.text_encoder_2)
            else:
                models = [model]

            expected_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
            for i, model in enumerate(models):
                _, num_int8 = get_num_quantized_nodes(model)
                self.assertEqual(expected_int8[i], num_int8)
