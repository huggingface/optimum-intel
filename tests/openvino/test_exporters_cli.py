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
from utils_tests import MODEL_NAMES

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
        ["text-generation", "gpt2"],
        ["text-generation-with-past", "gpt2"],
        ["text2text-generation", "t5"],
        ["text2text-generation-with-past", "t5"],
        ["text-classification", "bert"],
        ["question-answering", "distilbert"],
        ["token-classification", "roberta"],
        ["image-classification", "vit"],
        ["audio-classification", "wav2vec2"],
        ["fill-mask", "bert"],
        ["feature-extraction", "blenderbot"],
        ["stable-diffusion", "stable-diffusion"],
        ["stable-diffusion-xl", "stable-diffusion-xl"],
        ["stable-diffusion-xl", "stable-diffusion-xl-refiner"],
    )

    def _openvino_export(self, model_name: str, task: str):
        with TemporaryDirectory() as tmpdir:
            main_export(model_name_or_path=model_name, output=tmpdir, task=task)

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
