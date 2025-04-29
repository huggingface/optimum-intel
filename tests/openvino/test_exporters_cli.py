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
import json
import subprocess
import unittest
from pathlib import Path
from typing import Dict, List

from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from utils_tests import (
    _ARCHITECTURES_TO_EXPECTED_INT8,
    MODEL_NAMES,
    check_compression_state_per_model,
    get_num_quantized_nodes,
)

from optimum.exporters.openvino.__main__ import main_export
from optimum.exporters.openvino.utils import COMPLEX_CHAT_TEMPLATES
from optimum.intel import (  # noqa
    OVFluxFillPipeline,
    OVFluxPipeline,
    OVLatentConsistencyModelPipeline,
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVModelForVisualCausalLM,
    OVModelOpenCLIPForZeroShotImageClassification,
    OVModelOpenCLIPText,
    OVModelOpenCLIPVisual,
    OVSanaPipeline,
    OVSentenceTransformer,
    OVStableDiffusion3Pipeline,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLPipeline,
)
from optimum.intel.openvino.configuration import _DEFAULT_4BIT_CONFIGS
from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS, TemporaryDirectory
from optimum.intel.utils.import_utils import (
    compare_versions,
    is_openvino_tokenizers_available,
    is_openvino_version,
    is_tokenizers_version,
    is_transformers_version,
)


class OVCLIExportTestCase(unittest.TestCase):
    """
    Integration tests ensuring supported models are correctly exported.
    """

    SUPPORTED_ARCHITECTURES = [
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
        ("text-to-image", "stable-diffusion"),
        ("text-to-image", "stable-diffusion-xl"),
        ("image-to-image", "stable-diffusion-xl-refiner"),
        ("feature-extraction", "sam"),
    ]

    if is_transformers_version(">=", "4.45"):
        SUPPORTED_ARCHITECTURES.extend(
            [
                ("text-to-image", "stable-diffusion-3"),
                ("text-to-image", "flux"),
                ("inpainting", "flux-fill"),
                ("text-to-image", "sana"),
            ]
        )
    EXPECTED_NUMBER_OF_TOKENIZER_MODELS = {
        "gpt2": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "t5": 0 if is_openvino_version("<", "2025.1") else 2,  # 2025.1 brings support for unigram tokenizers
        "albert": 0 if is_openvino_version("<", "2025.1") else 2,  # 2025.1 brings support for unigram tokenizers
        "distilbert": 1 if is_openvino_version("<", "2025.0") else 2,  # no detokenizer before 2025.0
        "roberta": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "vit": 0,  # no tokenizer for image model
        "wav2vec2": 0,  # no tokenizer
        "bert": 1 if is_openvino_version("<", "2025.0") else 2,  # no detokenizer before 2025.0
        "blenderbot": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "stable-diffusion": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "stable-diffusion-xl": 4 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "stable-diffusion-3": 6 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 2,
        "flux": 4 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "flux-fill": 4 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "llava": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "sana": 2 if is_tokenizers_version("<", "0.20.0") or is_openvino_version(">=", "2024.5") else 0,
        "sam": 0,  # no tokenizer
    }

    TOKENIZER_CHAT_TEMPLATE_TESTS_MODELS = {
        "gpt2": {  # transformers, no chat template, no processor
            "num_tokenizers": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
            "task": "text-generation-with-past",
            "expected_chat_template": False,
            "simplified_chat_template": False,
            "processor_chat_template": False,
            "remote_code": False,
        },
        "stable-diffusion": {  # diffusers, no chat template
            "num_tokenizers": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
            "task": "text-to-image",
            "processor_chat_template": False,
            "remote_code": False,
            "expected_chat_template": False,
            "simplified_chat_template": False,
        },
        "llava": {  # transformers, chat template in processor, simplified chat template
            "num_tokenizers": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
            "task": "image-text-to-text",
            "processor_chat_template": True,
            "remote_code": False,
            "expected_chat_template": True,
            "simplified_chat_template": True,
        },
        "llava_next": {  # transformers, chat template in processor overrides tokinizer chat template, simplified chat template
            "num_tokenizers": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
            "task": "image-text-to-text",
            "processor_chat_template": True,
            "simplified_chat_template": True,
            "expected_chat_template": True,
            "remote_code": False,
        },
        "minicpm3": {  # transformers, no processor, simplified chat template
            "num_tokenizers": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
            "task": "text-generation-with-past",
            "expected_chat_template": True,
            "simplified_chat_template": True,
            "processor_chat_template": False,
            "remote_code": True,
        },
        "phi3_v": {  # transformers, no processor chat template, no simplified chat template
            "num_tokenizers": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
            "task": "image-text-to-text",
            "expected_chat_template": True,
            "simplified_chat_template": False,
            "processor_chat_template": False,
            "remote_code": True,
        },
        "glm": {  # transformers, no processor, no simplified chat template
            "num_tokenizers": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
            "task": "text-generation-with-past",
            "expected_chat_template": True,
            "simplified_chat_template": False,
            "processor_chat_template": False,
            "remote_code": True,
        },
    }

    SUPPORTED_SD_HYBRID_ARCHITECTURES = [
        ("stable-diffusion", 72, 195),
        ("stable-diffusion-xl", 84, 331),
        ("latent-consistency", 50, 135),
    ]

    if is_transformers_version(">=", "4.45"):
        SUPPORTED_SD_HYBRID_ARCHITECTURES.append(("stable-diffusion-3", 9, 65))
        SUPPORTED_SD_HYBRID_ARCHITECTURES.append(("flux", 7, 56))
        SUPPORTED_SD_HYBRID_ARCHITECTURES.append(("sana", 19, 53))

    SUPPORTED_QUANTIZATION_ARCHITECTURES = [
        (
            "automatic-speech-recognition",
            "whisper",
            "int8",
            "--dataset librispeech --num-samples 1 --smooth-quant-alpha 0.9 --trust-remote-code",
            [10, 12, 11] if is_transformers_version("<=", "4.36.0") else [8, 12, 25],
            (
                [{"int8": 8}, {"int8": 11}, {"int8": 9}]
                if is_transformers_version("<=", "4.36.0")
                else [{"int8": 8}, {"int8": 12}, {"int8": 18}]
            ),
        ),
        (
            "automatic-speech-recognition-with-past",
            "whisper",
            "f8e4m3",
            "--dataset librispeech --num-samples 1 --smooth-quant-alpha 0.9 --trust-remote-code",
            [10, 12, 11] if is_transformers_version("<=", "4.36.0") else [8, 12, 25],
            (
                [{"f8e4m3": 8}, {"f8e4m3": 11}, {"f8e4m3": 9}]
                if is_transformers_version("<=", "4.36.0")
                else [{"f8e4m3": 8}, {"f8e4m3": 12}, {"f8e4m3": 18}]
            ),
        ),
        (
            "text-generation",
            "llama",
            "f8e4m3",
            "--dataset wikitext2 --smooth-quant-alpha 0.9 --trust-remote-code",
            [
                13,
            ],
            [
                {"f8e4m3": 16},
            ],
        ),
        (
            "text-generation",
            "llama",
            "nf4_f8e4m3",
            "--dataset wikitext2 --num-samples 1 --group-size 16 --trust-remote-code --ratio 0.5",
            [
                14,
            ],
            [
                {"f8e4m3": 11, "nf4": 5},
            ],
        ),
        (
            "text-generation",
            "llama",
            "nf4_f8e5m2",
            "--dataset wikitext2 --num-samples 1 --group-size 16 --trust-remote-code --sym --ratio 0.5",
            [
                14,
            ],
            [
                {"f8e5m2": 11, "nf4": 5},
            ],
        ),
        (
            "text-generation",
            "llama",
            "int4_f8e4m3",
            "--dataset wikitext2 --num-samples 1 --group-size 16 --trust-remote-code --sym --ratio 0.5",
            [
                14,
            ],
            [
                {"f8e4m3": 11, "int4": 5},
            ],
        ),
        (
            "text-generation",
            "llama",
            "int4_f8e5m2",
            "--dataset wikitext2 --num-samples 1 --group-size 16 --trust-remote-code",
            [
                13,
            ],
            [
                {"f8e5m2": 2, "int4": 28},
            ],
        ),
        (
            "stable-diffusion",
            "stable-diffusion",
            "int8",
            "--dataset conceptual_captions --num-samples 1 --trust-remote-code",
            [
                112,
                0,
                0,
                0,
            ],
            [
                {"int8": 121},
                {"int8": 42},
                {"int8": 34},
                {"int8": 64},
            ],
        ),
        (
            "stable-diffusion-xl",
            "stable-diffusion-xl",
            "f8e5m2",
            "--dataset laion/220k-GPT4Vision-captions-from-LIVIS --num-samples 1 --trust-remote-code",
            [
                174,
                0,
                0,
                0,
                0,
            ],
            [
                {"f8e5m2": 183},
                {"int8": 42},
                {"int8": 34},
                {"int8": 64},
                {"int8": 66},
            ],
        ),
        (
            "latent-consistency",
            "latent-consistency",
            "f8e4m3",
            "--dataset laion/filtered-wit --num-samples 1 --trust-remote-code",
            [
                79,
                0,
                0,
                0,
            ],
            [
                {"f8e4m3": 84},
                {"int8": 42},
                {"int8": 34},
                {"int8": 40},
            ],
        ),
        (
            "feature-extraction",
            "blenderbot",
            "int8",
            "--dataset wikitext2 --num-samples 1",
            [
                33,
            ],
            [
                {"int8": 35},
            ],
        ),
        (
            "feature-extraction",
            "sentence-transformers-bert",
            "int8",
            "--library sentence_transformers --dataset c4 --num-samples 1",
            [
                12,
            ],
            [
                {"int8": 15},
            ],
        ),
    ]

    TEST_4BIT_CONFIGURATIONS = [
        ("text-generation-with-past", "opt125m", "int4 --sym --group-size 128", [{"int8": 4, "int4": 72}]),
        ("text-generation-with-past", "opt125m", "int4 --group-size 64", [{"int8": 4, "int4": 144}]),
        ("text-generation-with-past", "opt125m", "mxfp4", [{"int8": 4, "f4e2m1": 72, "f8e8m0": 72}]),
        ("text-generation-with-past", "opt125m", "nf4", [{"int8": 4, "nf4": 72}]),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 8 --all-layers",
            [{"int4": 16}],
        ),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 16 --awq --dataset wikitext2 --num-samples 100 "
            "--sensitivity-metric max_activation_variance",
            [{"int8": 4, "int4": 14}],
        ),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 16 --scale-estimation --dataset wikitext2 --num-samples 100 ",
            [{"int8": 4, "int4": 14}],
        ),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 16 --gptq --dataset wikitext2 --num-samples 100 ",
            [{"int8": 4, "int4": 14}],
        ),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --ratio 1.0 --sym --group-size 16 --lora-correction --dataset auto --num-samples 16",
            [{"int8": 60, "int4": 14}],
        ),
        (
            "text-generation-with-past",
            "llama_awq",
            "int4 --group-size 16 --backup-precision none --ratio 0.5",
            [{"int4": 6}],
        ),
    ]

    if is_transformers_version(">=", "4.40.0"):
        TEST_4BIT_CONFIGURATIONS.extend(
            [
                (
                    "image-text-to-text",
                    "llava_next",
                    "int4 --group-size 16 --ratio 0.8",
                    [{"int8": 14, "int4": 16}, {"int8": 1}, {"int8": 9}],
                ),
                (
                    "image-text-to-text",
                    "llava_next",
                    'int4 --group-size 16 --ratio 0.8 --sensitivity-metric "hessian_input_activation" '
                    "--dataset contextual --num-samples 1",
                    [{"int8": 6, "int4": 24}, {"int8": 1}, {"int8": 9}],
                ),
                (
                    "image-text-to-text",
                    "nanollava",
                    "int4 --group-size 8 --ratio 0.8 --trust-remote-code",
                    [{"int8": 16, "int4": 14}, {"int8": 1}, {"int8": 15}],
                ),
                (
                    "image-text-to-text",
                    "nanollava",
                    'int4 --group-size 8 --ratio 0.8 --sensitivity-metric "mean_activation_variance" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 16, "int4": 14}, {"int8": 1}, {"int8": 15}],
                ),
            ]
        )

    if is_transformers_version(">=", "4.42.0"):
        TEST_4BIT_CONFIGURATIONS.extend(
            [
                (
                    "image-text-to-text",
                    "llava_next_video",
                    'int4 --group-size 16 --ratio 0.8 --sensitivity-metric "hessian_input_activation" '
                    "--dataset contextual --num-samples 1",
                    [{"int8": 6, "int4": 24}, {"int8": 1}, {"int8": 7}, {}, {"int8": 2}],
                ),
            ]
        )

    if is_transformers_version(">=", "4.45.0"):
        TEST_4BIT_CONFIGURATIONS.extend(
            [
                (
                    "image-text-to-text",
                    "minicpmv",
                    "int4 --group-size 4 --ratio 0.8 --trust-remote-code",
                    [{"int8": 10, "int4": 20}, {"int8": 1}, {"int8": 26}, {"int8": 6}],
                ),
                (
                    "image-text-to-text",
                    "minicpmv",
                    'int4 --group-size 4 --ratio 0.8 --sensitivity-metric "mean_activation_magnitude" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 8, "int4": 22}, {"int8": 1}, {"int8": 26}, {"int8": 6}],
                ),
                (
                    "image-text-to-text",
                    "internvl2",
                    "int4 --group-size 4 --ratio 0.8 --trust-remote-code",
                    [{"int8": 8, "int4": 22}, {"int8": 1}, {"int8": 11}],
                ),
                (
                    "image-text-to-text",
                    "internvl2",
                    'int4 --group-size 4 --ratio 0.8 --sensitivity-metric "mean_activation_magnitude" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 8, "int4": 22}, {"int8": 1}, {"int8": 11}],
                ),
                (
                    "image-text-to-text",
                    "phi3_v",
                    "int4 --group-size 4 --ratio 0.8 --trust-remote-code",
                    [{"int8": 8, "int4": 10}, {"int8": 1}, {"int8": 7}, {"int8": 2}],
                ),
                (
                    "image-text-to-text",
                    "phi3_v",
                    'int4 --group-size 4 --ratio 0.8 --sensitivity-metric "mean_activation_magnitude" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 4, "int4": 14}, {"int8": 1}, {"int8": 7}, {"int8": 2}],
                ),
                (
                    "image-text-to-text",
                    "qwen2_vl",
                    'int4 --group-size 16 --ratio 0.8 --sensitivity-metric "mean_activation_magnitude" '
                    "--dataset contextual --num-samples 1",
                    [{"int8": 10, "int4": 20}, {"int8": 1}, {"int8": 1}, {"int8": 10}],
                ),
            ]
        )

    def _openvino_export(self, model_name: str, task: str):
        with TemporaryDirectory() as tmpdir:
            main_export(
                model_name_or_path=model_name,
                output=tmpdir,
                task=task,
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
            eval(
                _HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]
                if task.replace("-with-past", "") in _HEAD_TO_AUTOMODELS
                else _HEAD_TO_AUTOMODELS[model_type.replace("-refiner", "")]
            ).from_pretrained(tmpdir, **model_kwargs)

    @parameterized.expand(
        arch
        for arch in SUPPORTED_ARCHITECTURES
        if not arch[0].endswith("-with-past") and not arch[1].endswith("-refiner")
    )
    def test_exporters_cli_tokenizers(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            output = subprocess.check_output(
                f"TRANSFORMERS_VERBOSITY=debug optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} {tmpdir}",
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

            if task.startswith("text-generation") and compare_versions("openvino-tokenizers", ">=", "2024.3.0.0"):
                self.assertIn("Set tokenizer padding side to left", output)

    # some testing models required transformers at least 4.45 for conversion
    @parameterized.expand(TOKENIZER_CHAT_TEMPLATE_TESTS_MODELS)
    @unittest.skipIf(
        is_transformers_version("<", "4.45.0") or not is_openvino_tokenizers_available(),
        reason="test required openvino tokenizers and transformers >= 4.45",
    )
    def test_exporters_cli_tokenizers_chat_template(self, model_type):
        import openvino as ov

        core = ov.Core()
        with TemporaryDirectory() as tmpdir:
            model_test_config = self.TOKENIZER_CHAT_TEMPLATE_TESTS_MODELS[model_type]
            task = model_test_config["task"]
            model_id = MODEL_NAMES[model_type]
            remote_code = model_test_config.get("remote_code", False)
            cmd = f"TRANSFORMERS_VERBOSITY=debug optimum-cli export openvino --model {model_id} --task {task} {tmpdir}"
            if remote_code:
                cmd += " --trust-remote-code"
            output = subprocess.check_output(
                cmd,
                shell=True,
                stderr=subprocess.STDOUT,
            ).decode()
            number_of_tokenizers = sum("tokenizer" in file for file in map(str, Path(tmpdir).rglob("*.xml")))
            expected_num_tokenizers = model_test_config["num_tokenizers"]
            self.assertEqual(expected_num_tokenizers, number_of_tokenizers, output)
            tokenizer_path = (
                Path(tmpdir) / "openvino_tokenizer.xml"
                if "diffusion" not in model_type
                else Path(tmpdir) / "tokenizer/openvino_tokenizer.xml"
            )
            tokenizer_model = core.read_model(tokenizer_path)
            if not model_test_config.get("expected_chat_template", False):
                self.assertFalse(tokenizer_model.has_rt_info("chat_template"))
            else:
                rt_info_chat_template = tokenizer_model.get_rt_info("chat_template")
                if not model_test_config.get("processor_chat_template"):
                    tokenizer = AutoTokenizer.from_pretrained(tmpdir, trust_remote_code=remote_code)
                else:
                    tokenizer = AutoProcessor.from_pretrained(tmpdir, trust_remote_code=remote_code)
                ref_chat_template = tokenizer.chat_template
                self.assertEqual(rt_info_chat_template.value, ref_chat_template)
                if not model_test_config.get("simplified_chat_template", False):
                    self.assertFalse(tokenizer_model.has_rt_info("simplified_chat_template"))
                else:
                    simplified_rt_chat_template = tokenizer_model.get_rt_info("simplified_chat_template").value
                    self.assertTrue(rt_info_chat_template in COMPLEX_CHAT_TEMPLATES)
                    self.assertEqual(simplified_rt_chat_template, COMPLEX_CHAT_TEMPLATES[rt_info_chat_template.value])
                    # there are some difference in content key for conversation templates, simplified templates align to use common
                    if "llava" not in model_type:
                        origin_history_messages = [
                            {
                                "role": "system",
                                "content": "You are a friendly chatbot who always responds in the style of a pirate",
                            },
                            {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
                            {
                                "role": "assistant",
                                "content": " There is no specific limit for how many helicopters a human can eat in one sitting, but it is not recommended to consume large quantities of helicopters.",
                            },
                            {"role": "user", "content": "Why is it not recommended?"},
                        ]
                    else:
                        origin_history_messages = [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "You are a friendly chatbot who always responds in the style of a pirate",
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "How many helicopters can a human eat in one sitting?"}
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "There is no specific limit for how many helicopters a human can eat in one sitting, but it is not recommended to consume large quantities of helicopters.",
                                    }
                                ],
                            },
                            {"role": "user", "content": [{"type": "text", "text": "Why is it not recommended?"}]},
                        ]
                    history_messages = [
                        {
                            "role": "system",
                            "content": "You are a friendly chatbot who always responds in the style of a pirate",
                        },
                        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
                        {
                            "role": "assistant",
                            "content": "There is no specific limit for how many helicopters a human can eat in one sitting, but it is not recommended to consume large quantities of helicopters.",
                        },
                        {"role": "user", "content": "Why is it not recommended?"},
                    ]
                    reference_input_text_no_gen_prompt = tokenizer.apply_chat_template(
                        origin_history_messages,
                        add_generation_prompt=False,
                        chat_template=ref_chat_template,
                        tokenize=False,
                    )
                    simplified_input_text_no_gen_prompt = tokenizer.apply_chat_template(
                        history_messages,
                        add_generation_prompt=False,
                        chat_template=simplified_rt_chat_template,
                        tokenize=False,
                    )
                    self.assertEqual(
                        reference_input_text_no_gen_prompt,
                        simplified_input_text_no_gen_prompt,
                        f"Expected text:\n{reference_input_text_no_gen_prompt}\nSimplified text:\n{simplified_input_text_no_gen_prompt}",
                    )
                    reference_input_text_gen_prompt = tokenizer.apply_chat_template(
                        origin_history_messages,
                        chat_template=ref_chat_template,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    simplified_input_text_gen_prompt = tokenizer.apply_chat_template(
                        history_messages,
                        add_generation_prompt=True,
                        chat_template=simplified_rt_chat_template,
                        tokenize=False,
                    )
                    self.assertEqual(
                        reference_input_text_gen_prompt,
                        simplified_input_text_gen_prompt,
                        f"Expected text:\n{reference_input_text_gen_prompt}\nSimplified text:\n{simplified_input_text_gen_prompt}",
                    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_exporters_cli_fp16(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} --weight-format fp16 {tmpdir}",
                shell=True,
                check=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            eval(
                _HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]
                if task.replace("-with-past", "") in _HEAD_TO_AUTOMODELS
                else _HEAD_TO_AUTOMODELS[model_type.replace("-refiner", "")]
            ).from_pretrained(tmpdir, **model_kwargs)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_exporters_cli_int8(self, task: str, model_type: str):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task}  --weight-format int8 {tmpdir}",
                shell=True,
                check=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            model = eval(
                _HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]
                if task.replace("-with-past", "") in _HEAD_TO_AUTOMODELS
                else _HEAD_TO_AUTOMODELS[model_type.replace("-refiner", "")]
            ).from_pretrained(tmpdir, **model_kwargs)

            expected_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
            expected_int8 = [{"int8": it} for it in expected_int8]
            if task.startswith("text2text-generation") and (not task.endswith("with-past") or model.decoder.stateful):
                expected_int8 = expected_int8[:2]
            check_compression_state_per_model(self, model.ov_submodels.values(), expected_int8)

    @parameterized.expand(SUPPORTED_SD_HYBRID_ARCHITECTURES)
    def test_exporters_cli_hybrid_quantization(
        self, model_type: str, expected_fake_nodes: int, expected_int8_nodes: int
    ):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --dataset laion/filtered-wit --weight-format int8 {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS[model_type.replace("-refiner", "")]).from_pretrained(tmpdir)
            vision_model = model.unet.model if model.unet is not None else model.transformer.model
            num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(vision_model)
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])
            self.assertEqual(expected_fake_nodes, num_fake_nodes)
            self.assertFalse(vision_model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))

    @parameterized.expand(TEST_4BIT_CONFIGURATIONS)
    def test_exporters_cli_4bit(
        self, task: str, model_type: str, option: str, expected_num_weight_nodes_per_model: List[Dict]
    ):
        with TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --task {task} --weight-format {option} {tmpdir}",
                shell=True,
                check=True,
                capture_output=True,
            )
            model_kwargs = {"use_cache": task.endswith("with-past")} if "generation" in task else {}
            if "--trust-remote-code" in option:
                model_kwargs["trust_remote_code"] = True
            model = eval(
                _HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]
                if task.replace("-with-past", "") in _HEAD_TO_AUTOMODELS
                else _HEAD_TO_AUTOMODELS[model_type.replace("-refiner", "")]
            ).from_pretrained(tmpdir, **model_kwargs)

            check_compression_state_per_model(self, model.ov_submodels.values(), expected_num_weight_nodes_per_model)

            self.assertTrue("--awq" not in option or b"Applying AWQ" in result.stdout)
            self.assertTrue("--scale-estimation" not in option or b"Applying Scale Estimation" in result.stdout)
            self.assertTrue("--gptq" not in option or b"Applying GPTQ" in result.stdout)
            self.assertTrue(
                "--lora-correction" not in option or b"with correction of low-rank adapters" in result.stdout
            )

    @parameterized.expand(SUPPORTED_QUANTIZATION_ARCHITECTURES)
    def test_exporters_cli_full_quantization(
        self,
        task: str,
        model_type: str,
        quant_mode: str,
        option: str,
        expected_fake_nodes_per_model: List[int],
        expected_num_weight_nodes_per_model: List[Dict[str, int]],
    ):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --task {task} --model {MODEL_NAMES[model_type]} "
                f"--quant-mode {quant_mode} {option} {tmpdir}",
                shell=True,
                check=True,
            )
            model_cls = (
                OVSentenceTransformer
                if "--library sentence_transformers" in option
                else eval(_HEAD_TO_AUTOMODELS[task])
            )
            model = model_cls.from_pretrained(tmpdir)

            if "automatic-speech-recognition" in task and model.decoder_with_past is None:
                expected_num_weight_nodes_per_model = expected_num_weight_nodes_per_model[:-1]
                expected_fake_nodes_per_model = expected_fake_nodes_per_model[:-1]

            submodels = model.ov_submodels.values()
            check_compression_state_per_model(
                self,
                submodels,
                expected_num_weight_nodes_per_model,
                expected_fake_nodes_per_model,
            )

    def test_exporters_cli_int4_with_local_model_and_default_config(self):
        with TemporaryDirectory() as tmpdir:
            pt_model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES["falcon-40b"])
            # overload for matching with default configuration
            pt_model.save_pretrained(tmpdir)
            with open(Path(tmpdir) / "config.json", "r") as f:
                config = json.load(f)
                config["_name_or_path"] = "tiiuae/falcon-7b-instruct"

            with open(Path(tmpdir) / "config.json", "w") as wf:
                json.dump(config, wf)

            subprocess.run(
                f"optimum-cli export openvino --model {tmpdir} --task text-generation-with-past --weight-format int4 {tmpdir}",
                shell=True,
                check=True,
            )

            model = OVModelForCausalLM.from_pretrained(tmpdir)
            rt_info = model.model.get_rt_info()
            self.assertTrue("nncf" in rt_info)
            self.assertTrue("weight_compression" in rt_info["nncf"])
            model_weight_compression_config = rt_info["nncf"]["weight_compression"]

            default_config = _DEFAULT_4BIT_CONFIGS["tiiuae/falcon-7b-instruct"]
            bits = default_config.pop("bits", None)
            self.assertEqual(bits, 4)

            sym = default_config.pop("sym", False)
            default_config["mode"] = f'int{bits}_{"sym" if sym else "asym"}'

            quant_method = default_config.pop("quant_method", None)
            default_config["awq"] = quant_method == "awq"
            default_config["gptq"] = quant_method == "gptq"

            default_config.pop("dataset", None)

            for key, value in default_config.items():
                self.assertIn(key, model_weight_compression_config)
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
            model = OVSentenceTransformer.from_pretrained(tmpdir, compile=False)
            self.assertFalse("last_hidden_state" in model.output_names)

    def test_exporters_cli_open_clip(self):
        model_id = MODEL_NAMES["open-clip"]
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {model_id} --framework pt {tmpdir}",
                shell=True,
                check=True,
            )
            model_vision = eval(_HEAD_TO_AUTOMODELS["open_clip_vision"]).from_pretrained(tmpdir, compile=False)
            model_text = eval(_HEAD_TO_AUTOMODELS["open_clip_text"]).from_pretrained(tmpdir, compile=False)
            self.assertTrue("image_features" in model_vision.output_names)
            self.assertTrue("text_features" in model_text.output_names)

            model = eval(_HEAD_TO_AUTOMODELS["open_clip"]).from_pretrained(tmpdir, compile=False)
            self.assertTrue("text_features" in model.text_model.output_names)
            self.assertTrue("image_features" in model.visual_model.output_names)

    def test_export_openvino_with_missed_weight_format(self):
        # Test that exception is raised when some compression parameter is given, but weight format is not.
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(subprocess.CalledProcessError) as exc_info:
                subprocess.run(
                    f"optimum-cli export openvino --model {MODEL_NAMES['gpt2']} --task text-generation --sym {tmpdir}",
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            self.assertIn(
                "Some compression parameters are provided, but the weight format is not specified.",
                str(exc_info.exception.stderr),
            )

    def test_export_openvino_with_custom_variant(self):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model katuni4ka/tiny-stable-diffusion-torch-custom-variant --variant custom {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS["stable-diffusion"]).from_pretrained(tmpdir, compile=False)
            for component in ["text_encoder", "tokenizer", "unet", "vae_encoder", "vae_decoder"]:
                self.assertIsNotNone(getattr(model, component))
