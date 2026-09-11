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
import numpy as np
import openvino as ov
import torch
from parameterized import parameterized
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from utils_tests import MODEL_NAMES

from optimum.exporters.openvino import export_from_model, main_export
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

    def _export_and_assert_hidden_state_locators(self, model_type, model_class, task, model_filename):
        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            annotated_dir = tmpdirname / "annotated"
            model = model_class.from_pretrained(MODEL_NAMES[model_type])
            export_from_model(
                model=model,
                output=annotated_dir,
                task=task,
                preprocessors=None,
                stateful=False,
            )

            annotated_model = ov.Core().read_model(annotated_dir / model_filename)
            self._assert_hidden_state_rt_info_is_valid(annotated_model)

    @parameterized.expand(("qwen3", "qwen3_moe"))
    def test_export_hidden_state_locators_for_representative_decoder_models(self, model_type):
        self._export_and_assert_hidden_state_locators(
            model_type, AutoModelForCausalLM, "text-generation", "openvino_model.xml"
        )

    @parameterized.expand(("qwen3_5", "qwen3_5_moe", "gemma4"))
    def test_export_hidden_state_locators_for_representative_multi_modal_models(self, model_type):
        if model_type in {"qwen3_5", "qwen3_5_moe"} and not (
            is_transformers_version(">=", "5.2.0") and is_transformers_version("<=", "5.2.99")
        ):
            self.skipTest("Qwen3.5 hidden-state locator coverage requires Transformers >= 5.2.0 and <= 5.2.99")
        if model_type == "gemma4" and not is_transformers_version(">=", "5.5.0"):
            self.skipTest("Gemma 4 hidden-state locator coverage requires Transformers >= 5.5.0")

        self._export_and_assert_hidden_state_locators(
            model_type, AutoModelForImageTextToText, "image-text-to-text", "openvino_language_model.xml"
        )

    def test_dflash2_export_produces_backbone_and_selector(self):
        model_id = MODEL_NAMES["qwen3_dflash2"]
        config = AutoConfig.from_pretrained(model_id)
        dflash_config = config.dflash_config
        num_drafted_tokens = dflash_config["block_size"] - 1

        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            main_export(
                model_name_or_path=model_id,
                task="text-generation-with-past",
                trust_remote_code=True,
                convert_tokenizer=False,
                output=tmpdirname,
            )

            core = ov.Core()
            backbone = core.read_model(tmpdirname / "openvino_model.xml")
            selector = core.read_model(tmpdirname / "openvino_dflash2_selector_model.xml")

            self.assertEqual(
                {inp.get_any_name() for inp in backbone.inputs},
                {"inputs_embeds", "hidden_states", "position_ids", "attention_mask", "beam_idx"},
            )
            self.assertEqual([out.get_any_name() for out in backbone.outputs], ["last_hidden_state"])

            # The selector's walk over the block is unrolled at trace time, so the drafted-token
            # axis must come out static while the batch axis stays dynamic.
            selector_inputs = {inp.get_any_name(): inp.get_partial_shape() for inp in selector.inputs}
            self.assertEqual(set(selector_inputs), {"last_hidden_state", "logits", "anchor_ids"})
            for name in ("last_hidden_state", "logits"):
                self.assertTrue(selector_inputs[name][0].is_dynamic)
                self.assertEqual(selector_inputs[name][1].get_length(), num_drafted_tokens)
            self.assertEqual(
                [out.get_any_name() for out in selector.outputs], ["draft_token_ids", "candidate_token_ids"]
            )

            for model in (backbone, selector):
                self.assertEqual(model.get_rt_info()["dflash_mode"].value, "True")
                rt_info = model.get_rt_info()["dflash"]
                self.assertEqual(rt_info["version"].value, "2")
                self.assertEqual(rt_info["block_size"].value, str(dflash_config["block_size"]))
                self.assertEqual(rt_info["selector_top_k"].value, str(dflash_config["selector_top_k"]))
                self.assertEqual(
                    rt_info["target_layer_ids"].value, ",".join(map(str, dflash_config["target_layer_ids"]))
                )

            self._assert_dflash2_matches_torch(model_id, config, tmpdirname, core)

    def _assert_dflash2_matches_torch(self, model_id, config, exported_dir, core):
        from optimum.exporters.openvino.model_patcher import Qwen3DFlash2ForCausalLM

        torch.manual_seed(0)
        block_size = config.dflash_config["block_size"]
        context_length = 3 * block_size
        num_target_layers = len(config.dflash_config["target_layer_ids"])

        reference = Qwen3DFlash2ForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        target_hidden = torch.randn(1, context_length, num_target_layers * config.hidden_size)
        inputs_embeds = torch.randn(1, block_size, config.hidden_size)
        position_ids = torch.arange(context_length + block_size)[None]
        anchor_ids = torch.randint(0, config.vocab_size, (1,))
        logits = torch.randn(1, block_size - 1, config.vocab_size)

        with torch.no_grad():
            expected_hidden = reference(
                inputs_embeds=inputs_embeds,
                hidden_states=target_hidden,
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state
            expected_selection = reference.candidate_selector(expected_hidden, logits, anchor_ids)

        backbone = core.compile_model(exported_dir / "openvino_model.xml", "CPU").create_infer_request()
        actual_hidden = backbone.infer(
            {
                "inputs_embeds": inputs_embeds.numpy(),
                "hidden_states": target_hidden.numpy(),
                "position_ids": position_ids.numpy(),
                "attention_mask": np.ones((1, context_length + block_size), dtype=np.int64),
                "beam_idx": np.zeros(1, dtype=np.int32),
            }
        )["last_hidden_state"]
        np.testing.assert_allclose(actual_hidden, expected_hidden.numpy(), rtol=1e-4, atol=1e-4)

        selector = core.compile_model(exported_dir / "openvino_dflash2_selector_model.xml", "CPU")
        actual_selection = selector(
            {
                "last_hidden_state": expected_hidden.numpy(),
                "logits": logits.numpy(),
                "anchor_ids": anchor_ids.numpy(),
            }
        )
        np.testing.assert_array_equal(
            actual_selection["draft_token_ids"], expected_selection["draft_token_ids"].numpy()
        )

    def test_hidden_state_locators_survive_weight_compression(self):
        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            annotated_dir = tmpdirname / "annotated"
            export_from_model(
                model=AutoModelForCausalLM.from_pretrained(MODEL_NAMES["qwen3"]),
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
