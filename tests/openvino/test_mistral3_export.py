# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from types import SimpleNamespace
from unittest.mock import patch

from optimum.exporters.openvino.__main__ import main_export
from optimum.exporters.openvino.model_configs import Mistral3OpenVINOConfig, Ministral3OpenVINOConfig
from optimum.exporters.openvino.model_patcher import Mistral3ImageEmbeddingsModelPatcher
from optimum.exporters.openvino.utils import MULTI_MODAL_TEXT_GENERATION_MODELS
from optimum.exporters.tasks import TasksManager


class Mistral3OpenVINORegistrationTest(unittest.TestCase):
    def test_mistral3_tasks_manager_registration(self):
        supported_tasks = TasksManager.get_supported_tasks_for_model_type(
            "mistral3", exporter="openvino", library_name="transformers"
        )

        self.assertEqual(
            sorted(supported_tasks.keys()),
            ["image-text-to-text", "text-generation", "text-generation-with-past"],
        )
        self.assertIs(supported_tasks["image-text-to-text"].func, Mistral3OpenVINOConfig)
        self.assertIs(supported_tasks["text-generation"].func, Mistral3OpenVINOConfig)
        self.assertIs(supported_tasks["text-generation-with-past"].func, Mistral3OpenVINOConfig)

    def test_ministral3_tasks_manager_registration(self):
        supported_tasks = TasksManager.get_supported_tasks_for_model_type(
            "ministral3", exporter="openvino", library_name="transformers"
        )

        self.assertEqual(
            sorted(supported_tasks.keys()),
            [
                "feature-extraction",
                "feature-extraction-with-past",
                "text-classification",
                "text-generation",
                "text-generation-with-past",
            ],
        )
        self.assertIs(supported_tasks["text-generation-with-past"].func, Ministral3OpenVINOConfig)

    def test_mistral3_image_embeddings_patcher_is_importable(self):
        self.assertIs(Mistral3ImageEmbeddingsModelPatcher, Mistral3ImageEmbeddingsModelPatcher)

    def test_mistral3_is_marked_as_multimodal_text_generation_model(self):
        self.assertIn("mistral3", MULTI_MODAL_TEXT_GENERATION_MODELS)

    def test_main_export_uses_image_text_loader_for_mistral3_text_generation(self):
        config = SimpleNamespace(
            model_type="mistral3",
            torch_dtype=None,
            architectures=["Mistral3ForConditionalGeneration"],
        )
        sentinel = RuntimeError("stop after checking task redirection")

        for task in ("text-generation", "text-generation-with-past"):
            with self.subTest(task=task), patch(
                "optimum.exporters.openvino.__main__.AutoConfig.from_pretrained", return_value=config
            ), patch(
                "optimum.exporters.openvino.__main__.TasksManager.get_model_from_task", side_effect=sentinel
            ) as get_model_from_task:
                with self.assertRaises(RuntimeError) as caught:
                    main_export(
                        model_name_or_path="stub-model",
                        output=Path("tests/openvino/mistral3-export-stub"),
                        task=task,
                        library_name="transformers",
                        local_files_only=True,
                    )

                self.assertIs(caught.exception, sentinel)
                self.assertEqual(get_model_from_task.call_args.args[0], "image-text-to-text")
