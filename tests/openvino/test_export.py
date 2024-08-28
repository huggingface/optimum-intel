# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import torch
from parameterized import parameterized
from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from utils_tests import MODEL_NAMES

from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED
from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.openvino import export_from_model, main_export
from optimum.exporters.tasks import TasksManager
from optimum.intel import (
    OVLatentConsistencyModelPipeline,
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForCustomTasks,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForPix2Struct,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLImg2ImgPipeline,
    OVStableDiffusionXLPipeline,
)
from optimum.intel.openvino.modeling_base import OVBaseModel
from optimum.intel.utils.import_utils import _transformers_version
from optimum.utils.save_utils import maybe_load_preprocessors


class ExportModelTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = {
        "bert": OVModelForMaskedLM,
        "pix2struct": OVModelForPix2Struct,
        "t5": OVModelForSeq2SeqLM,
        "bart": OVModelForSeq2SeqLM,
        "gpt2": OVModelForCausalLM,
        "distilbert": OVModelForQuestionAnswering,
        "albert": OVModelForSequenceClassification,
        "vit": OVModelForImageClassification,
        "roberta": OVModelForTokenClassification,
        "wav2vec2": OVModelForAudioClassification,
        "whisper": OVModelForSpeechSeq2Seq,
        "blenderbot": OVModelForFeatureExtraction,
        "stable-diffusion": OVStableDiffusionPipeline,
        "stable-diffusion-xl": OVStableDiffusionXLPipeline,
        "stable-diffusion-xl-refiner": OVStableDiffusionXLImg2ImgPipeline,
        "latent-consistency": OVLatentConsistencyModelPipeline,
    }

    GENERATIVE_MODELS = ("pix2struct", "t5", "bart", "gpt2", "whisper")

    def _openvino_export(
        self,
        model_type: str,
        compression_option: Optional[str] = None,
        stateful: bool = True,
        patch_16bit_model: bool = False,
    ):
        auto_model = self.SUPPORTED_ARCHITECTURES[model_type]
        task = auto_model.export_feature
        model_name = MODEL_NAMES[model_type]
        library_name = TasksManager.infer_library_from_model(model_name)
        loading_kwargs = {"attn_implementation": "eager"} if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED else {}

        if library_name == "timm":
            model_class = TasksManager.get_model_class_for_task(task, library=library_name)
            model = model_class(f"hf_hub:{model_name}", pretrained=True, exportable=True)
            TasksManager.standardize_model_attributes(model_name, model, library_name=library_name)
        else:
            model = auto_model.auto_model_class.from_pretrained(model_name, **loading_kwargs)

        if getattr(model.config, "model_type", None) == "pix2struct":
            preprocessors = maybe_load_preprocessors(model_name)
        else:
            preprocessors = None

        supported_tasks = (task, task + "-with-past") if "text-generation" in task else (task,)
        for supported_task in supported_tasks:
            with TemporaryDirectory() as tmpdirname:
                export_from_model(
                    model=model,
                    output=Path(tmpdirname),
                    task=supported_task,
                    preprocessors=preprocessors,
                    compression_option=compression_option,
                    stateful=stateful,
                )

                use_cache = supported_task.endswith("-with-past")
                ov_model = auto_model.from_pretrained(tmpdirname, use_cache=use_cache)
                self.assertIsInstance(ov_model, OVBaseModel)

                if "text-generation" in task:
                    self.assertEqual(ov_model.use_cache, use_cache)

                if task == "text-generation":
                    self.assertEqual(ov_model.stateful, stateful and use_cache)
                    self.assertEqual(
                        ov_model.model.get_rt_info()["optimum"]["transformers_version"], _transformers_version
                    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_export(self, model_type: str):
        self._openvino_export(model_type)

    @parameterized.expand(GENERATIVE_MODELS)
    def test_export_with_custom_gen_config(self, model_type):
        auto_model = self.SUPPORTED_ARCHITECTURES[model_type]
        task = auto_model.export_feature
        model_name = MODEL_NAMES[model_type]
        loading_kwargs = {"attn_implementation": "eager"} if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED else {}

        model = auto_model.auto_model_class.from_pretrained(model_name, **loading_kwargs)

        model.generation_config.top_k = 42
        model.generation_config.do_sample = True

        if getattr(model.config, "model_type", None) == "pix2struct":
            preprocessors = maybe_load_preprocessors(model_name)
        else:
            preprocessors = None

        supported_tasks = (task, task + "-with-past") if "text-generation" in task else (task,)
        for supported_task in supported_tasks:
            with TemporaryDirectory() as tmpdirname:
                export_from_model(
                    model=model,
                    output=Path(tmpdirname),
                    task=supported_task,
                    preprocessors=preprocessors,
                )

                use_cache = supported_task.endswith("-with-past")
                ov_model = auto_model.from_pretrained(tmpdirname, use_cache=use_cache)
                self.assertIsInstance(ov_model, OVBaseModel)
                self.assertTrue(ov_model.can_generate())
                self.assertTrue(ov_model.generation_config is not None)
                self.assertIsInstance(ov_model.generation_config, GenerationConfig)
                self.assertTrue(ov_model.generation_config.top_k == 42)

                # check that generate config remains after repeated saving
                with TemporaryDirectory() as tmpdirname2:
                    ov_model.save_pretrained(tmpdirname2)
                    ov_model = auto_model.from_pretrained(tmpdirname2, use_cache=use_cache)
                    self.assertIsInstance(ov_model, OVBaseModel)
                    self.assertTrue(ov_model.can_generate())
                    self.assertTrue(ov_model.generation_config is not None)
                    self.assertIsInstance(ov_model.generation_config, GenerationConfig)
                    self.assertTrue(ov_model.generation_config.top_k == 42)

    def test_export_fp16_model(self):
        auto_model = self.SUPPORTED_ARCHITECTURES["gpt2"]
        task = auto_model.export_feature
        model_name = MODEL_NAMES["gpt2"]
        model = auto_model.auto_model_class.from_pretrained(model_name, torch_dtype=torch.float16)
        stateful = True

        for supported_task in [task, task + "with-past"]:
            with TemporaryDirectory() as tmpdirname:
                export_from_model(
                    model=model,
                    output=Path(tmpdirname),
                    task=task,
                    preprocessors=None,
                    patch_16bit_model=True,
                    stateful=stateful,
                )
                use_cache = supported_task.endswith("-with-past")
                ov_model = auto_model.from_pretrained(tmpdirname, use_cache=use_cache)
                self.assertIsInstance(ov_model, OVBaseModel)
                self.assertEqual(ov_model.use_cache, use_cache)
                self.assertEqual(ov_model.stateful, stateful and use_cache)
                self.assertEqual(
                    ov_model.model.get_rt_info()["optimum"]["transformers_version"], _transformers_version
                )


class CustomExportModelTest(unittest.TestCase):
    def test_custom_export_config_model(self):
        class BertOnnxConfigWithPooler(BertOnnxConfig):
            @property
            def outputs(self):
                if self.task == "feature-extraction-with-pooler":
                    common_outputs = {}
                    common_outputs["last_hidden_state"] = {0: "batch_size", 1: "sequence_length"}
                    common_outputs["pooler_output"] = {0: "batch_size"}
                else:
                    common_outputs = super().outputs

                return common_outputs

        base_task = "feature-extraction"
        custom_task = f"{base_task}-with-pooler"
        model_id = "sentence-transformers/all-MiniLM-L6-v2"

        config = AutoConfig.from_pretrained(model_id)
        custom_export_configs = {"model": BertOnnxConfigWithPooler(config, task=custom_task)}

        with TemporaryDirectory() as tmpdirname:
            main_export(
                model_name_or_path=model_id,
                custom_export_configs=custom_export_configs,
                library_name="transformers",
                output=Path(tmpdirname),
                task=base_task,
            )

            ov_model = OVModelForCustomTasks.from_pretrained(tmpdirname)

            self.assertIsInstance(ov_model, OVBaseModel)
            self.assertTrue(ov_model.output_names == {"last_hidden_state": 0, "pooler_output": 1})

    def test_export_custom_model(self):
        model_id = "hf-internal-testing/tiny-random-BertModel"
        word_embedding_model = models.Transformer(model_id, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=256,
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

        with TemporaryDirectory() as tmpdirname:
            export_from_model(model, output=tmpdirname, task="feature-extraction")
            ov_model = OVModelForCustomTasks.from_pretrained(tmpdirname)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        with torch.no_grad():
            model_outputs = model(tokens)

        ov_outputs = ov_model(**tokens)
        self.assertTrue(torch.allclose(ov_outputs.token_embeddings, model_outputs.token_embeddings, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.sentence_embedding, model_outputs.sentence_embedding, atol=1e-4))
