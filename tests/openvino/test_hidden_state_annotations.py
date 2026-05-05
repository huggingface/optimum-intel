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

import numpy as np
import openvino as ov
import torch
from transformers import AutoModelForCausalLM
from utils_tests import MODEL_NAMES

from optimum.exporters.openvino import export_from_model


HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers"


def _find_output_by_tensor_name(model, tensor_name):
    for op in model.get_ops():
        for output in op.outputs():
            if tensor_name in output.get_names():
                return output
    raise AssertionError(f"Tensor {tensor_name} was not found in the OpenVINO graph")


def _add_model_output(model, output, output_name):
    output.get_tensor().add_names({output_name})
    if hasattr(model, "add_output"):
        model.add_output(output)
    else:
        model.add_outputs([output])


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

    def test_annotated_hidden_state_output_matches_pytorch(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES["gpt2"])
        model.eval()

        with TemporaryDirectory() as tmpdirname:
            export_from_model(
                model=model,
                output=Path(tmpdirname),
                task="text-generation",
                preprocessors=None,
                stateful=False,
            )

            core = ov.Core()
            ov_model = core.read_model(Path(tmpdirname) / "openvino_model.xml")
            annotation = json.loads(ov_model.get_rt_info()[HIDDEN_STATES_RT_INFO_KEY].value)
            layer_idx = 0
            output_name = "decoder_layer_0_hidden_state"
            hidden_state_output = _find_output_by_tensor_name(ov_model, annotation["layers"][str(layer_idx)])
            _add_model_output(ov_model, hidden_state_output, output_name)

            input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                torch_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )

            compiled_model = core.compile_model(ov_model, "CPU")
            ov_inputs = {}
            for input_port in compiled_model.inputs:
                input_name = input_port.get_any_name()
                if input_name == "input_ids":
                    ov_inputs[input_name] = input_ids.numpy()
                elif input_name == "attention_mask":
                    ov_inputs[input_name] = attention_mask.numpy()
                elif input_name == "position_ids":
                    ov_inputs[input_name] = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
                elif input_name == "token_type_ids":
                    ov_inputs[input_name] = np.zeros(input_ids.shape, dtype=np.int64)
                else:
                    self.fail(f"Unexpected OpenVINO model input: {input_name}")

            infer_result = compiled_model(ov_inputs)
            ov_output_port = next(output for output in compiled_model.outputs if output_name in output.get_names())
            np.testing.assert_allclose(
                infer_result[ov_output_port],
                torch_outputs.hidden_states[layer_idx + 1].detach().numpy(),
                rtol=1e-4,
                atol=1e-4,
            )
