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
from transformers import AutoModelForCausalLM
from utils_tests import (
    _ARCHITECTURES_TO_EXPECTED_INT8,
    MODEL_NAMES,
    get_num_quantized_nodes,
)

from optimum.exporters.openvino.__main__ import main_export
from optimum.intel import (  # noqa
    OVLatentConsistencyModelPipeline,
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
from optimum.intel.openvino.configuration import _DEFAULT_4BIT_CONFIGS
from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS
from optimum.intel.utils.import_utils import is_openvino_tokenizers_available, is_transformers_version


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
        "t5": 0,  # no .model file in the repository
        "albert": 0,  # not supported yet
        "distilbert": 1,  # no detokenizer
        "roberta": 2,
        "vit": 0,  # no tokenizer for image model
        "wav2vec2": 0,  # no tokenizer
        "bert": 1,  # no detokenizer
        "blenderbot": 2,
        "stable-diffusion": 2,
        "stable-diffusion-xl": 4,
    }

    SUPPORTED_SD_HYBRID_ARCHITECTURES = (
        ("stable-diffusion", 72, 195),
        ("stable-diffusion-xl", 84, 331),
        ("latent-consistency", 50, 135),
    )

    TEST_4BIT_CONFIGURATONS = [
        ("text-generation-with-past", "opt125m", "int4_sym_g128", 4, 144),
        ("text-generation-with-past", "opt125m", "int4_asym_g128", 4, 144),
        ("text-generation-with-past", "opt125m", "int4_sym_g64", 4, 144),
        ("text-generation-with-past", "opt125m", "int4_asym_g64", 4, 144),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 8 --all-layers",
            0,
            32 if is_transformers_version("<", "4.39.0") else 34,
        ),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 16 --awq --dataset wikitext2 --num-samples 100 "
            "--sensitivity-metric max_activation_variance",
            6 if is_transformers_version(">=", "4.39") else 4,
            28,
        ),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 16 --scale-estimation --dataset wikitext2 --num-samples 100 ",
            6 if is_transformers_version(">=", "4.39") else 4,
            28,
        ),
    ]

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
    def test_exporters_cli_tokenizers(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            output = subprocess.check_output(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} {tmpdir}",
                shell=True,
                stderr=subprocess.STDOUT,
            ).decode()
            if not is_openvino_tokenizers_available():
                self.assertTrue(
                    "OpenVINO Tokenizers is not available." in output
                    or "OpenVINO and OpenVINO Tokenizers versions are not binary compatible." in output,
                    msg=output,
                )
                return

            number_of_tokenizers = sum("tokenizer" in file for file in map(str, Path(tmpdir).rglob("*.xml")))
            self.assertEqual(self.EXPECTED_NUMBER_OF_TOKENIZER_MODELS[model_type], number_of_tokenizers, output)

            if number_of_tokenizers == 1:
                self.assertTrue("Detokenizer is not supported, convert tokenizer only." in output, output)

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
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --dataset laion/filtered-wit --weight-format int8 {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS[model_type]).from_pretrained(tmpdir)
            num_fq, num_int8, _ = get_num_quantized_nodes(model.unet)
            self.assertEqual(exp_num_int8, num_int8)
            self.assertEqual(exp_num_fq, num_fq)

    @parameterized.expand(TEST_4BIT_CONFIGURATONS)
    def test_exporters_cli_int4(self, task: str, model_type: str, option: str, expected_int8: int, expected_int4: int):
        with TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} --weight-format {option} {tmpdir}",
                shell=True,
                check=True,
                capture_output=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            model = eval(_HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]).from_pretrained(tmpdir, **model_kwargs)

            _, num_int8, num_int4 = get_num_quantized_nodes(model)
            self.assertEqual(expected_int8, num_int8)
            self.assertEqual(expected_int4, num_int4)
            self.assertTrue("--awq" not in option or b"Applying AWQ" in result.stdout)
            self.assertTrue("--scale-estimation" not in option or b"Applying Scale Estimation" in result.stdout)

    def test_exporters_cli_int4_with_local_model_and_default_config(self):
        with TemporaryDirectory() as tmpdir:
            pt_model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES["bloom"])
            # overload for matching with default configuration
            pt_model.config._name_or_path = "bigscience/bloomz-7b1"
            pt_model.save_pretrained(tmpdir)
            subprocess.run(
                f"optimum-cli export openvino --model {tmpdir} --task text-generation-with-past --weight-format int4 {tmpdir}",
                shell=True,
                check=True,
            )

            model = OVModelForCausalLM.from_pretrained(tmpdir)
            rt_info = model.model.get_rt_info()
            self.assertTrue("nncf" in rt_info)
            self.assertTrue("weight_compression" in rt_info["nncf"])
            default_config = _DEFAULT_4BIT_CONFIGS["bigscience/bloomz-7b1"]
            model_weight_compression_config = rt_info["nncf"]["weight_compression"]
            sym = default_config.pop("sym", False)
            bits = default_config.pop("bits", None)
            self.assertEqual(bits, 4)

            mode = f'int{bits}_{"sym" if sym else "asym"}'
            default_config["mode"] = mode
            for key, value in default_config.items():
                self.assertTrue(key in model_weight_compression_config)
                self.assertEqual(
                    model_weight_compression_config[key].value,
                    str(value),
                    f"Parameter {key} not matched with expected, {model_weight_compression_config[key].value} != {value}",
                )

    def test_exporters_cli_help(self):
        subprocess.run(
            "optimum-cli export openvino --help",
            shell=True,
            check=True,
        )

    def test_exporters_cli_sentence_transformers(self):
        model_id = MODEL_NAMES["bge"]
        with TemporaryDirectory() as tmpdir:
            # default export creates transformers model
            subprocess.run(
                f"optimum-cli export openvino --model {model_id} --task feature-extraction {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS["feature-extraction"]).from_pretrained(tmpdir, compile=False)
            self.assertTrue("last_hidden_state" in model.output_names)
            # export with transformers lib creates transformers model
            subprocess.run(
                f"optimum-cli export openvino --model {model_id} --task feature-extraction --library transformers {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS["feature-extraction"]).from_pretrained(tmpdir, compile=False)
            self.assertTrue("last_hidden_state" in model.output_names)
            # export with sentence_transformers lib creates sentence_transformers model
            subprocess.run(
                f"optimum-cli export openvino --model {model_id} --task feature-extraction --library sentence_transformers {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS["feature-extraction"]).from_pretrained(tmpdir, compile=False)
            self.assertFalse("last_hidden_state" in model.output_names)
