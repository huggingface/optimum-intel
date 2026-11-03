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
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
from utils_tests import MODEL_NAMES

from optimum.exporters.openvino import export_from_model, main_export
from optimum.intel.openvino import OVAssistantForCausalLM, OVConfig, OVModelForVisualCausalLM
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

    def test_export_muse_glimmer_assistant_draft_contract(self):
        if not is_transformers_version(">=", "5.15.0"):
            self.skipTest("MuseGlimmer requires Transformers >= 5.15.0")

        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            main_export(
                model_name_or_path=MODEL_NAMES["muse_glimmer_assistant"],
                output=tmpdirname,
                task="text-generation-with-past",
            )
            model = ov.Core().read_model(tmpdirname / "openvino_model.xml")

            # The drafter borrows the target's embedding and lm_head, so it takes
            # embeddings in and emits hidden states out - never token ids or logits.
            input_names = {inp.get_any_name() for inp in model.inputs}
            self.assertEqual(
                input_names, {"inputs_embeds", "hidden_states", "position_ids", "attention_mask", "beam_idx"}
            )
            self.assertEqual({out.get_any_name() for out in model.outputs}, {"last_hidden_state"})

            self.assertTrue(model.has_rt_info(["dflash_mode"]))
            self.assertTrue(model.has_rt_info(["dflash", "block_size"]))
            self.assertTrue(model.has_rt_info(["dflash", "mask_token_id"]))
            target_layer_ids = model.get_rt_info(["dflash", "target_layer_ids"]).value.split(",")
            self.assertEqual(len(target_layer_ids), 2)

            # `hidden_states` is the target's states at every target layer, concatenated.
            hidden_size = model.input("inputs_embeds").get_partial_shape()[2].get_length()
            context_width = model.input("hidden_states").get_partial_shape()[2].get_length()
            self.assertEqual(context_width, hidden_size * len(target_layer_ids))

    def test_muse_glimmer_dflash_speculative_decoding_is_lossless(self):
        if not is_transformers_version(">=", "5.15.0"):
            self.skipTest("MuseGlimmer requires Transformers >= 5.15.0")

        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            target_dir, draft_dir = tmpdirname / "target", tmpdirname / "draft"
            # Export uncompressed. The fixture is randomly initialized, so its logits sit
            # within ~1e-3 of each other; weight compression then lets the (unavoidable)
            # numeric difference between verifying a block and decoding token by token
            # flip an argmax tie, which would fail this check for reasons that have
            # nothing to do with speculation. Real weights are not this degenerate.
            fp32 = OVConfig(dtype="fp32")
            main_export(
                model_name_or_path=MODEL_NAMES["muse_glimmer"],
                output=target_dir,
                task="image-text-to-text",
                ov_config=fp32,
            )
            main_export(
                model_name_or_path=MODEL_NAMES["muse_glimmer_assistant"],
                output=draft_dir,
                task="text-generation-with-past",
                ov_config=fp32,
            )

            model = OVModelForVisualCausalLM.from_pretrained(target_dir)
            assistant = OVAssistantForCausalLM.from_pretrained(draft_dir)
            processor = AutoProcessor.from_pretrained(MODEL_NAMES["muse_glimmer"], padding_side="left")

            image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8))
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in this image?"}]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")

            baseline = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            speculative = model.generate(**inputs, max_new_tokens=24, do_sample=False, assistant_model=assistant)

            # Greedy DFlash is lossless: the accepted prefix plus the bonus token always
            # reproduces the target's own greedy continuation.
            self.assertEqual(baseline.shape, speculative.shape)
            self.assertTrue(torch.equal(baseline, speculative))

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
