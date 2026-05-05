# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import openvino as ov
from transformers import AutoModelForCausalLM
from utils_tests import MODEL_NAMES

from optimum.exporters.openvino import export_from_model


HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers"


class HiddenStateAnnotationExportTest(unittest.TestCase):
    def test_export_hidden_state_annotations_without_extra_outputs(self):
        for task in ("text-generation", "text-generation-with-past"):
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES["gpt2"])
            with self.subTest(task=task), TemporaryDirectory() as tmpdirname:
                export_from_model(
                    model=model,
                    output=Path(tmpdirname),
                    task=task,
                    preprocessors=None,
                    stateful=False,
                )

                ov_model = ov.Core().read_model(Path(tmpdirname) / "openvino_model.xml")
                output_names = set().union(*(output.get_names() for output in ov_model.outputs))
                self.assertNotIn("last_hidden_state", output_names)
                self.assertFalse(any(name.startswith("ov.hidden_states.") for name in output_names))

                rt_info = ov_model.get_rt_info()
                self.assertIn(HIDDEN_STATES_RT_INFO_KEY, rt_info)
                annotation = json.loads(rt_info[HIDDEN_STATES_RT_INFO_KEY].value)
                self.assertEqual(annotation["version"], 1)
                self.assertEqual(len(annotation["layers"]), model.config.num_hidden_layers)

                graph_tensor_names = set()
                for op in ov_model.get_ops():
                    for output in op.outputs():
                        graph_tensor_names.update(output.get_names())
                for tensor_name in annotation["layers"].values():
                    self.assertIn(tensor_name, graph_tensor_names)
