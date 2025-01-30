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
from typing import Dict, List, Tuple

from parameterized import parameterized
from transformers import AutoModelForCausalLM
from utils_tests import (
    _ARCHITECTURES_TO_EXPECTED_INT8,
    MODEL_NAMES,
    compare_num_quantized_nodes_per_model,
    get_num_quantized_nodes,
)

from optimum.exporters.openvino.__main__ import main_export
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
    ]

    if is_transformers_version(">=", "4.45"):
        SUPPORTED_ARCHITECTURES.extend(
            [("text-to-image", "stable-diffusion-3"), ("text-to-image", "flux"), ("inpainting", "flux-fill")]
        )
    EXPECTED_NUMBER_OF_TOKENIZER_MODELS = {
        "gpt2": 2 if is_tokenizers_version("<", "0.20") or is_openvino_version(">=", "2024.5") else 0,
        "t5": 0,  # no .model file in the repository
        "albert": 0,  # not supported yet
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
    }

    SUPPORTED_SD_HYBRID_ARCHITECTURES = [
        ("stable-diffusion", 72, 195),
        ("stable-diffusion-xl", 84, 331),
        ("latent-consistency", 50, 135),
    ]

    if is_transformers_version(">=", "4.45"):
        SUPPORTED_SD_HYBRID_ARCHITECTURES.append(("stable-diffusion-3", 9, 65))
        SUPPORTED_SD_HYBRID_ARCHITECTURES.append(("flux", 7, 56))

    SUPPORTED_QUANTIZATION_ARCHITECTURES = [
        (
            "automatic-speech-recognition",
            "whisper",
            "int8",
            "--dataset librispeech --num-samples 1 --smooth-quant-alpha 0.9 --trust-remote-code",
            (14, 22, 21) if is_transformers_version("<=", "4.36.0") else (14, 22, 25),
            (14, 21, 17) if is_transformers_version("<=", "4.36.0") else (14, 22, 18),
        ),
        (
            "text-generation",
            "llama",
            "f8e4m3",
            "--dataset wikitext2 --smooth-quant-alpha 0.9 --trust-remote-code",
            (13,),
            (16,),
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
                    [{"int8": 14, "int4": 16}, {"int8": 9}, {"int8": 1}],
                ),
                (
                    "image-text-to-text",
                    "llava_next",
                    'int4 --group-size 16 --ratio 0.8 --sensitivity-metric "hessian_input_activation" '
                    "--dataset contextual --num-samples 1",
                    [{"int8": 6, "int4": 24}, {"int8": 9}, {"int8": 1}],
                ),
                (
                    "image-text-to-text",
                    "nanollava",
                    "int4 --group-size 8 --ratio 0.8 --trust-remote-code",
                    [{"int8": 16, "int4": 14}, {"int8": 15}, {"int8": 1}],
                ),
                (
                    "image-text-to-text",
                    "nanollava",
                    'int4 --group-size 8 --ratio 0.8 --sensitivity-metric "mean_activation_variance" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 16, "int4": 14}, {"int8": 15}, {"int8": 1}],
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
                    [{"int8": 10, "int4": 20}, {"int8": 26}, {"int8": 1}, {"int8": 6}],
                ),
                (
                    "image-text-to-text",
                    "minicpmv",
                    'int4 --group-size 4 --ratio 0.8 --sensitivity-metric "mean_activation_magnitude" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 8, "int4": 22}, {"int8": 26}, {"int8": 1}, {"int8": 6}],
                ),
                (
                    "image-text-to-text",
                    "internvl2",
                    "int4 --group-size 4 --ratio 0.8 --trust-remote-code",
                    [{"int8": 8, "int4": 22}, {"int8": 11}, {"int8": 1}],
                ),
                (
                    "image-text-to-text",
                    "internvl2",
                    'int4 --group-size 4 --ratio 0.8 --sensitivity-metric "mean_activation_magnitude" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 8, "int4": 22}, {"int8": 11}, {"int8": 1}],
                ),
                (
                    "image-text-to-text",
                    "phi3_v",
                    "int4 --group-size 4 --ratio 0.8 --trust-remote-code",
                    [{"int8": 8, "int4": 10}, {"int8": 7}, {"int8": 1}, {"int8": 2}],
                ),
                (
                    "image-text-to-text",
                    "phi3_v",
                    'int4 --group-size 4 --ratio 0.8 --sensitivity-metric "mean_activation_magnitude" '
                    "--dataset contextual --num-samples 1 --trust-remote-code",
                    [{"int8": 4, "int4": 14}, {"int8": 7}, {"int8": 1}, {"int8": 2}],
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

            if task.startswith("text2text-generation"):
                models = [model.encoder, model.decoder]
                if task.endswith("with-past") and not model.decoder.stateful:
                    models.append(model.decoder_with_past)
            elif model_type.startswith("stable-diffusion") or model_type.startswith("flux"):
                models = [model.unet or model.transformer, model.vae_encoder, model.vae_decoder]
                models.append(model.text_encoder if model_type == "stable-diffusion" else model.text_encoder_2)
            elif task.startswith("image-text-to-text"):
                models = [model.language_model, model.vision_embeddings]
            else:
                models = [model]

            expected_int8 = _ARCHITECTURES_TO_EXPECTED_INT8[model_type]
            for i, model in enumerate(models):
                _, num_weight_nodes = get_num_quantized_nodes(model)
                self.assertEqual(expected_int8[i], num_weight_nodes["int8"])

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
            num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(
                model.unet if model.unet is not None else model.transformer
            )
            self.assertEqual(expected_int8_nodes, num_weight_nodes["int8"])
            self.assertEqual(expected_fake_nodes, num_fake_nodes)

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

            submodels = []
            if task == "text-generation-with-past":
                submodels = [model]
            elif task == "image-text-to-text":
                submodels = [model.lm_model, model.vision_embeddings_model, model.text_embeddings_model]
                submodels += [getattr(model, part) for part in model.additional_parts]

            compare_num_quantized_nodes_per_model(self, submodels, expected_num_weight_nodes_per_model)

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
        expected_fake_nodes: Tuple[int],
        expected_low_precision_nodes: Tuple[int],
    ):
        with TemporaryDirectory() as tmpdir:
            subprocess.run(
                f"optimum-cli export openvino --model {MODEL_NAMES[model_type]} --quant-mode {quant_mode} {option} {tmpdir}",
                shell=True,
                check=True,
            )
            model = eval(_HEAD_TO_AUTOMODELS[task]).from_pretrained(tmpdir)

            models = [model]
            if task == "automatic-speech-recognition":
                models = [model.encoder, model.decoder]
                if model.decoder_with_past is not None:
                    models.append(model.decoder_with_past)
                else:
                    expected_fake_nodes = expected_fake_nodes[:-1]
            self.assertEqual(len(expected_fake_nodes), len(models))
            for i, model in enumerate(models):
                num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(model)
                self.assertEqual(expected_fake_nodes[i], num_fake_nodes)
                self.assertEqual(expected_low_precision_nodes[i], num_weight_nodes[quant_mode])

    def test_exporters_cli_int4_with_local_model_and_default_config(self):
        with TemporaryDirectory() as tmpdir:
            pt_model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES["falcon-40b"])
            # overload for matching with default configuration
            pt_model.config._name_or_path = "tiiuae/falcon-7b-instruct"
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
