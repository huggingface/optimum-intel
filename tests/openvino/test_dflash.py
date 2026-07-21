#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import json
import unittest
from pathlib import Path

import nncf
import openvino as ov
from parameterized import parameterized
from transformers import AutoModelForCausalLM
from utils_tests import MODEL_NAMES

from optimum.exporters.openvino import export_from_model
from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING
from optimum.intel.openvino.utils import TemporaryDirectory
from optimum.intel.utils.import_utils import is_transformers_version


class DFlashExportTest(unittest.TestCase):
    def _assert_hidden_state_rt_info_is_valid(self, model):
        def find_output_by_locator(model, locator):
            matches = [op for op in model.get_ops() if op.get_friendly_name() == locator["producer"]]
            if len(matches) != 1:
                raise AssertionError(f"Producer {locator['producer']!r} resolved to {len(matches)} OpenVINO nodes")
            output_index = locator["output_index"]
            if not isinstance(output_index, int) or output_index < 0 or output_index >= len(matches[0].outputs()):
                raise AssertionError(f"Producer {locator['producer']!r} has no output {output_index}")
            return matches[0].output(output_index)

        self.assertTrue(model.has_rt_info(["hidden_states_decoder_layers"]))
        annotation = json.loads(model.get_rt_info()["hidden_states_decoder_layers"].value)
        self.assertIsInstance(annotation, dict)
        self.assertIn("layers", annotation)
        locators = annotation["layers"]
        self.assertTrue(locators)
        self.assertEqual(set(locators), {str(layer_id) for layer_id in range(len(locators))})

        resolved_outputs = set()
        for layer_id in range(len(locators)):
            locator = locators[str(layer_id)]
            self.assertIsInstance(locator, dict)
            self.assertIsInstance(locator.get("producer"), str)
            self.assertIsInstance(locator.get("output_index"), int)
            identity = (locator["producer"], locator["output_index"])
            self.assertNotIn(identity, resolved_outputs)
            find_output_by_locator(model, locator)
            resolved_outputs.add(identity)
        return locators

    @parameterized.expand(("llama", "gemma4", "qwen3_5", "qwen3_5_moe", "gpt_oss"))
    def test_export_hidden_state_locators_for_representative_decoder_models(self, model_type):
        if model_type == "gemma4" and not is_transformers_version(">=", "5.5.0"):
            self.skipTest("Gemma4 hidden-state locator coverage requires Transformers >= 5.5.0")
        if model_type in {"qwen3_5", "qwen3_5_moe"} and not is_transformers_version(">=", "5.2.0"):
            self.skipTest("Qwen3.5 hidden-state locator coverage requires Transformers >= 5.2.0")

        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            annotated_dir = tmpdirname / "annotated"
            if model_type == "gemma4":
                model = MODEL_TYPE_TO_CLS_MAPPING[model_type].auto_model_class.from_pretrained(
                    MODEL_NAMES[model_type]
                ).language_model
            else:
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[model_type])
            export_from_model(
                model=model,
                output=annotated_dir,
                task="text-generation",
                preprocessors=None,
                stateful=False,
            )

            annotated_model = ov.Core().read_model(annotated_dir / "openvino_model.xml")
            self._assert_hidden_state_rt_info_is_valid(annotated_model)

    def test_hidden_state_locators_survive_weight_compression(self):
        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            annotated_dir = tmpdirname / "annotated"
            export_from_model(
                model=AutoModelForCausalLM.from_pretrained(MODEL_NAMES["qwen3_5_moe"]),
                output=annotated_dir,
                task="text-generation",
                preprocessors=None,
                stateful=False,
            )
            xml_path = annotated_dir / "openvino_model.xml"
            original_model = ov.Core().read_model(xml_path)
            layer_ids = set(self._assert_hidden_state_rt_info_is_valid(original_model))
            for mode, kwargs in (
                (nncf.CompressWeightsMode.INT8_ASYM, {}),
                (nncf.CompressWeightsMode.INT4_ASYM, {"all_layers": True, "group_size": -1}),
            ):
                with self.subTest(mode=mode):
                    compressed_model = nncf.compress_weights(ov.Core().read_model(xml_path), mode=mode, **kwargs)
                    locators = self._assert_hidden_state_rt_info_is_valid(compressed_model)
                    self.assertEqual(set(locators), layer_ids)

