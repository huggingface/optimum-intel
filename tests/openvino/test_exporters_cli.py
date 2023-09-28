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


class OVCLIExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    SUPPORTED_ARCHITECTURES = (
        ["causal-lm", "gpt2"],
        ["causal-lm-with-past", "gpt2"],
        ["seq2seq-lm", "t5"],
        ["seq2seq-lm-with-past", "t5"],
        ["sequence-classification", "bert"],
        ["question-answering", "distilbert"],
        ["masked-lm", "bert"],
        ["default", "blenderbot"],
        ["default-with-past", "blenderbot"],
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
        with TemporaryDirectory() as tmpdirname:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} {tmpdirname}",
                shell=True,
                check=True,
            )
