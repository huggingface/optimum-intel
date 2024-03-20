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
from pathlib import Path
from tempfile import TemporaryDirectory

from parameterized import parameterized
from utils_tests import (
    _ARCHITECTURES_TO_EXPECTED_INT4_INT8,
    _ARCHITECTURES_TO_EXPECTED_INT8,
    MODEL_NAMES,
    get_num_quantized_nodes,
)

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
    OVLatentConsistencyModelPipeline,
)
from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS
from optimum.intel.utils.import_utils import is_openvino_tokenizers_available


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
    EXPECTED_NUMBER_OF_TOKENIZER_MODELS = {
        "gpt2": 2,
        "t5": 0,  # failed internal sentencepiece check - no <s> token in the vocab
        "albert": 0,  # not supported yet
        "distilbert": 1,  # no detokenizer
        "roberta": 2,
        "vit": 0,  # no tokenizer for image model
        "wav2vec2": 0,  # no tokenizer
        "bert": 1,  # no detokenizer
        "blenderbot": 2,
        "stable-diffusion": 0,  # not supported
        "stable-diffusion-xl": 0,  # not supported
    }

    SUPPORTED_SD_HYBRID_ARCHITECTURES = (
        ("stable-diffusion", 72, 195),
        ("stable-diffusion-xl", 84, 331),
        ("latent-consistency", 50, 135),
    )

    SUPPORTED_4BIT_ARCHITECTURES = (("text-generation-with-past", "opt125m"),)

    SUPPORTED_4BIT_OPTIONS = ["int4_sym_g128", "int4_asym_g128", "int4_sym_g64", "int4_asym_g64"]

    TEST_4BIT_CONFIGURATONS = []
    for arch in SUPPORTED_4BIT_ARCHITECTURES:
        for option in SUPPORTED_4BIT_OPTIONS:
            TEST_4BIT_CONFIGURATONS.append([arch[0], arch[1], option])

    def _openvino_export(
        self, model_name: str, task: str, compression_option: str = None, compression_ratio: float = None
    ):
        with TemporaryDirectory() as tmpdir:
            main_export(
                model_name_or_path=model_name,
                output=tmpdir,
                task=task,
                compression_option=compression_option,
                compression_ratio=compression_ratio,
            )

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

    @parameterized.expand(
        arch
        for arch in SUPPORTED_ARCHITECTURES
        if not arch[0].endswith("-with-past") and not arch[1].endswith("-refiner")
    )
    @unittest.skipIf(not is_openvino_tokenizers_available(), reason="OpenVINO Tokenizers not available")
    def test_exporters_cli_tokenizers(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            output = subprocess.check_output(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --convert-tokenizer --task {task} {tmpdir}",
                shell=True,
                stderr=subprocess.STDOUT,
            ).decode()
            save_dir = Path(tmpdir)
            number_of_tokenizers = sum("tokenizer" in file for file in map(str, save_dir.rglob("*.xml")))
            self.assertEqual(
                self.EXPECTED_NUMBER_OF_TOKENIZER_MODELS[model_type],
                number_of_tokenizers,
                f"OVT: {is_openvino_tokenizers_available() }",
            )

            if number_of_tokenizers == 1:
                self.assertTrue("Detokenizer is not supported, convert tokenizer only." in output, output)
            elif number_of_tokenizers == 0 and task not in ("image-classification", "audio-classification"):
                self.assertTrue(("OpenVINO Tokenizer export for" in output and "is not supported." in output), output)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_exporters_cli_fp16(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} --weight-format fp16 {tmpdir}",
                shell=True,
                check=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            eval(_HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]).from_pretrained(tmpdir, **model_kwargs)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_exporters_cli_int8(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task}  --weight-format int8 {tmpdir}",
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
                _, num_int8, _ = get_num_quantized_nodes(model)
                self.assertEqual(expected_int8[i], num_int8)

    @parameterized.expand(SUPPORTED_SD_HYBRID_ARCHITECTURES)
    def test_exporters_cli_hybrid_quantization(self, model_type: str, exp_num_fq: int, exp_num_int8: int):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} "
                f"--task {model_type} --dataset laion/filtered-wit --weight-format int8 {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS[model_type]).from_pretrained(tmpdir)
            num_fq, num_int8, _ = get_num_quantized_nodes(model.unet)
            self.assertEqual(exp_num_int8, num_int8)
            self.assertEqual(exp_num_fq, num_fq)

    @parameterized.expand(TEST_4BIT_CONFIGURATONS)
    def test_exporters_cli_int4(self, task: str, model_type: str, option: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task}  --weight-format {option} {tmpdir}",
                shell=True,
                check=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            model = eval(_HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]).from_pretrained(tmpdir, **model_kwargs)

            expected_int8, expected_int4 = _ARCHITECTURES_TO_EXPECTED_INT4_INT8[model_type]
            _, num_int8, num_int4 = get_num_quantized_nodes(model)
            self.assertEqual(expected_int8, num_int8)
            self.assertEqual(expected_int4, num_int4)

    def test_exporters_cli_help(self):
        subprocess.run(
            "optimum-cli export openvino --help",
            shell=True,
            check=True,
        )
