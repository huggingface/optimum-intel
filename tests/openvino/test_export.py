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

from parameterized import parameterized
from utils_tests import MODEL_NAMES

from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED
from optimum.exporters.openvino import export_from_model
from optimum.exporters.tasks import TasksManager
from optimum.intel import (
    OVLatentConsistencyModelPipeline,
    OVModelForAudioClassification,
    OVModelForCausalLM,
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

    def _openvino_export(
        self,
        model_type: str,
        compression_option: Optional[str] = None,
        stateful: bool = True,
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

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_export(self, model_type: str):
        self._openvino_export(model_type)
