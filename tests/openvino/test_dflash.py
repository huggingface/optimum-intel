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
import os
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import nncf
import numpy as np
import openvino as ov
import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from utils_tests import MODEL_NAMES, get_dflash2_model_path

from optimum.exporters.openvino import export_from_model
from optimum.exporters.openvino.__main__ import main_export
from optimum.exporters.openvino.convert import _get_submodels_and_export_configs, export_models
from optimum.exporters.openvino.dflash_utils import parse_and_validate_dflash_config
from optimum.exporters.openvino.model_configs import DFlash2SelectorOpenVINOConfig, Qwen3OpenVINOConfig
from optimum.exporters.openvino.model_patcher import (
    DFlash2CandidateSelector,
    GroupedDynamicCausalConv,
    Qwen3DFlash2ForCausalLM,
    Qwen3DFlashForCausalLM,
)
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

    def test_v1_export_remains_single_ir(self):
        config = Qwen3Config.from_pretrained(MODEL_NAMES["qwen3"])
        config.architectures = ["DFlashDraftModel"]
        config.dflash_config = {
            "mask_token_id": config.vocab_size - 1,
            "target_layer_ids": list(range(min(2, config.num_hidden_layers))),
        }
        model = Qwen3DFlashForCausalLM(config)
        _, parts, stateful = _get_submodels_and_export_configs(
            model,
            task="text-generation-with-past",
            monolith=False,
            custom_export_configs={},
            custom_architecture=False,
            _variant="default",
            library_name="transformers",
            stateful=True,
        )
        self.assertEqual(list(parts), ["model"])
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            export_models(
                parts,
                output_dir,
                output_names=["openvino_model.xml"],
                input_shapes={"batch_size": 1, "sequence_length": 3},
                stateful=stateful,
                library_name="transformers",
            )
            draft_ir = ov.Core().read_model(output_dir / "openvino_model.xml")
            self.assertTrue(draft_ir.has_rt_info(["dflash_mode"]))
            self.assertEqual(
                draft_ir.get_rt_info()["dflash"]["mask_token_id"].value,
                str(config.dflash_config["mask_token_id"]),
            )
            self.assertEqual(
                draft_ir.get_rt_info()["dflash"]["target_layer_ids"].value,
                ",".join(map(str, config.dflash_config["target_layer_ids"])),
            )
            self.assertFalse(draft_ir.has_rt_info(["dflash", "version"]))
            self.assertFalse((output_dir / "openvino_selector_model.xml").exists())


@unittest.skipUnless(is_transformers_version(">=", "4.57"), "DFlash-2 requires Transformers 4.57 or newer")
class DFlash2ExportTest(unittest.TestCase):
    @classmethod
    def _model_path(cls):
        return get_dflash2_model_path()

    @classmethod
    def _config(cls):
        return Qwen3Config.from_pretrained(cls._model_path())

    @classmethod
    def _model(cls):
        return Qwen3DFlash2ForCausalLM.from_pretrained(cls._model_path()).eval()

    @staticmethod
    def _backbone_inputs(config, num_draft_tokens, context_length=3):
        block_length = num_draft_tokens + 1
        return {
            "inputs_embeds": torch.randn(1, block_length, config.hidden_size),
            "hidden_states": torch.randn(
                1,
                context_length,
                config.hidden_size * len(config.dflash_config["target_layer_ids"]),
            ),
            "position_ids": torch.arange(context_length + block_length).reshape(1, -1),
            "attention_mask": torch.ones(1, context_length + block_length),
        }

    @staticmethod
    def _reference_convolution(hidden_states, dynamic_kernel, base_kernel, group_size):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_groups = hidden_size // group_size
        hidden_groups = hidden_states.reshape(batch_size, sequence_length, num_groups, group_size)
        output = torch.zeros_like(hidden_groups)
        for position in range(sequence_length):
            for tap in range(base_kernel.shape[0]):
                if position < tap:
                    continue
                values = hidden_groups[:, position - tap]
                kernel = base_kernel[tap].reshape(num_groups, group_size)
                kernel = kernel + dynamic_kernel[:, position, tap].unsqueeze(-1)
                output[:, position] += values * kernel
        return output.reshape_as(hidden_states)

    @staticmethod
    def _constant_count_with_shape(model, shape):
        return sum(
            op.get_type_name() == "Constant" and list(op.get_output_shape(0)) == list(shape) for op in model.get_ops()
        )

    def test_checkpoint_keys(self):
        config = self._config()
        model, loading_info = Qwen3DFlash2ForCausalLM.from_pretrained(
            self._model_path(),
            output_loading_info=True,
        )
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])

        parameter_names = set(dict(model.named_parameters()))
        expected_names = {
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
            "candidate_selector.hidden_projection.weight",
        }
        for layer_idx in range(config.num_hidden_layers):
            expected_names.update(
                {
                    f"layers.{layer_idx}.attention_conv.base_kernel",
                    f"layers.{layer_idx}.attention_conv.kernel_projection.weight",
                    f"layers.{layer_idx}.mlp_conv.base_kernel",
                    f"layers.{layer_idx}.mlp_conv.kernel_projection.weight",
                }
            )
        self.assertTrue(expected_names.issubset(parameter_names))

    def test_config_normalization_and_validation(self):
        config = self._config()
        flat_config = deepcopy(config)
        flat_values = dict(flat_config.dflash_config)
        flat_config.dflash_config = {}
        for name, value in flat_values.items():
            setattr(flat_config, name, value)
        flat_dflash = parse_and_validate_dflash_config(flat_config)
        self.assertEqual(flat_dflash.version, 2)
        self.assertEqual(flat_dflash.values, flat_values)

        zero_softcap_config = deepcopy(config)
        zero_softcap_config.dflash_config["final_logit_softcapping"] = 0.0
        self.assertEqual(
            parse_and_validate_dflash_config(zero_softcap_config).values["final_logit_softcapping"],
            0.0,
        )

        negative_softcap_config = deepcopy(config)
        negative_softcap_config.dflash_config["final_logit_softcapping"] = -1.0
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            parse_and_validate_dflash_config(negative_softcap_config)

        zero_multiplier_config = deepcopy(config)
        zero_multiplier_config.dflash_config["output_multiplier"] = 0.0
        with self.assertRaisesRegex(ValueError, "finite positive"):
            parse_and_validate_dflash_config(zero_multiplier_config)

        missing_config = deepcopy(config)
        del missing_config.dflash_config["selector_rank"]
        with self.assertRaisesRegex(ValueError, "selector_rank"):
            DFlash2SelectorOpenVINOConfig(missing_config)

        invalid_group_config = deepcopy(config)
        invalid_group_config.dflash_config["conv_group_size"] = 7
        with self.assertRaisesRegex(ValueError, "positive divisor"):
            Qwen3OpenVINOConfig(invalid_group_config)

        invalid_target_layer_config = deepcopy(config)
        invalid_target_layer_config.dflash_config["target_layer_ids"] = [config.num_target_layers]
        with self.assertRaisesRegex(ValueError, "smaller than num_target_layers"):
            parse_and_validate_dflash_config(invalid_target_layer_config)

        invalid_mask_config = deepcopy(config)
        invalid_mask_config.dflash_config["mask_token_id"] = config.vocab_size
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            parse_and_validate_dflash_config(invalid_mask_config)

        duplicate_layers_config = deepcopy(config)
        duplicate_layers_config.dflash_config["target_layer_ids"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "unique non-negative"):
            parse_and_validate_dflash_config(duplicate_layers_config)

        conflicting_config = deepcopy(config)
        conflicting_config.selector_rank = config.dflash_config["selector_rank"] + 1
        with self.assertRaisesRegex(ValueError, "conflicting nested and flat"):
            parse_and_validate_dflash_config(conflicting_config)

        wrong_model_type_config = deepcopy(config)
        wrong_model_type_config.model_type = "muse_glimmer"
        with self.assertRaisesRegex(ValueError, "only Qwen3"):
            parse_and_validate_dflash_config(wrong_model_type_config)

        wrong_architecture_config = deepcopy(config)
        wrong_architecture_config.architectures = ["Qwen3ForCausalLM", "DFlash2DraftModel"]
        with self.assertRaisesRegex(ValueError, "architectures"):
            parse_and_validate_dflash_config(wrong_architecture_config)

        v1_config = deepcopy(config)
        v1_config.architectures = ["DFlashDraftModel"]
        v1_config.dflash_config = {
            "mask_token_id": config.dflash_config["mask_token_id"],
            "target_layer_ids": list(config.dflash_config["target_layer_ids"]),
        }
        v1_dflash = parse_and_validate_dflash_config(v1_config)
        self.assertEqual(v1_dflash.version, 1)
        self.assertEqual(v1_dflash.values, v1_config.dflash_config)

        invalid_v1_config = deepcopy(v1_config)
        invalid_v1_config.dflash_config["target_layer_ids"] = [-1]
        with self.assertRaisesRegex(ValueError, "unique non-negative"):
            parse_and_validate_dflash_config(invalid_v1_config)

    def test_target_annotations_cover_dflash2_target_layers(self):
        draft_config = self._config()
        target_config = deepcopy(draft_config)
        target_config.architectures = ["Qwen3ForCausalLM"]
        target_config.num_hidden_layers = max(draft_config.dflash_config["target_layer_ids"]) + 1
        target_model = Qwen3ForCausalLM(target_config).eval()
        with TemporaryDirectory() as tmpdirname:
            target_dir = Path(tmpdirname) / "target"
            export_from_model(
                model=target_model,
                output=target_dir,
                task="text-generation",
                preprocessors=None,
                stateful=False,
            )
            target_ir = ov.Core().read_model(target_dir / "openvino_model.xml")
            annotation = json.loads(target_ir.get_rt_info()["hidden_states_decoder_layers"].value)
            for layer_id in draft_config.dflash_config["target_layer_ids"]:
                locator = annotation["layers"][str(layer_id)]
                matches = [op for op in target_ir.get_ops() if op.get_friendly_name() == locator["producer"]]
                self.assertEqual(len(matches), 1)
                self.assertLess(locator["output_index"], len(matches[0].outputs()))

    def test_grouped_dynamic_convolution_reference_parity(self):
        config = self._config()
        convolution = GroupedDynamicCausalConv(config).eval()
        for sequence_length in (2, 4, 7):
            with self.subTest(sequence_length=sequence_length):
                hidden_states = torch.randn(2, sequence_length, config.hidden_size)
                projected = convolution.kernel_projection(hidden_states).reshape(
                    2,
                    sequence_length,
                    2,
                    convolution.kernel_size,
                    convolution.num_groups,
                )
                prepared, output_kernel = convolution.prepare(hidden_states)
                reference_prepared = self._reference_convolution(
                    hidden_states,
                    projected[:, :, 0],
                    convolution.base_kernel[0],
                    convolution.group_size,
                )
                torch.testing.assert_close(prepared, reference_prepared)
                torch.testing.assert_close(output_kernel, projected[:, :, 1])

                branch_output = torch.randn_like(hidden_states)
                finished = convolution.finish(branch_output, output_kernel)
                reference_finished = self._reference_convolution(
                    branch_output,
                    projected[:, :, 1],
                    convolution.base_kernel[1],
                    convolution.group_size,
                )
                torch.testing.assert_close(finished, reference_finished)

    def test_selector_lattice_and_path_reference_parity(self):
        config = self._config()
        config.dflash_config["selector_rank"] = 1
        config.dflash_config["selector_top_k"] = 2
        selector = DFlash2CandidateSelector(config).eval()
        with torch.no_grad():
            selector.predecessor_codebook.zero_()
            selector.successor_codebook.zero_()
            selector.hidden_projection.weight.zero_()
            selector.hidden_projection.weight[0, 0] = 1
            selector.predecessor_codebook[0, 0] = 1
            selector.predecessor_codebook[2, 0] = 2
            selector.successor_codebook[2, 0] = 3
            selector.successor_codebook[4, 0] = 2

        candidate_ids = torch.tensor([[[1, 2], [3, 4]]])
        unary_logits = torch.tensor([[[5.0, 4.0], [5.0, 4.0]]])
        draft_hidden_states = torch.zeros(1, 2, config.hidden_size)
        draft_hidden_states[..., 0] = 1
        anchor_token_ids = torch.tensor([0])
        edge_scores = selector(candidate_ids, unary_logits, draft_hidden_states, anchor_token_ids)

        reference_scores = torch.empty_like(edge_scores)
        predecessor_ids = torch.tensor([[[0, 0], [1, 2]]])
        for position in range(2):
            for predecessor_index in range(2):
                predecessor = predecessor_ids[0, position, predecessor_index]
                for candidate_index in range(2):
                    candidate = candidate_ids[0, position, candidate_index]
                    correction = (
                        selector.predecessor_codebook[predecessor, 0]
                        * draft_hidden_states[0, position, 0]
                        * selector.successor_codebook[candidate, 0]
                    )
                    reference_scores[0, position, predecessor_index, candidate_index] = (
                        unary_logits[0, position, candidate_index] + correction
                    )
        torch.testing.assert_close(edge_scores, reference_scores)
        self.assertEqual(edge_scores.dtype, torch.float32)

        previous_index = 0
        path = []
        selected_rows = []
        for position in range(candidate_ids.shape[1]):
            selected_row = edge_scores[:, position, previous_index]
            selected_rows.append(selected_row)
            previous_index = selected_row.argmax(dim=-1).item()
            path.append(candidate_ids[0, position, previous_index].item())
        self.assertEqual(path, [2, 4])
        self.assertEqual(unary_logits.argmax(dim=-1).tolist(), [[0, 0]])
        for selected_row in selected_rows:
            probabilities = torch.softmax(selected_row, dim=-1)
            torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))

        selector = selector.to(torch.bfloat16)
        bf16_scores = selector(
            candidate_ids,
            unary_logits,
            draft_hidden_states.to(torch.bfloat16),
            anchor_token_ids,
        )
        self.assertEqual(bf16_scores.dtype, torch.float32)

    def test_two_ir_export_dynamic_numerical_parity_and_selectorless_backbone(self):
        torch.manual_seed(1)
        model = self._model()
        config = model.config
        _, parts, stateful = _get_submodels_and_export_configs(
            model,
            task="text-generation",
            monolith=False,
            custom_export_configs={},
            custom_architecture=False,
            _variant="default",
            library_name="transformers",
            stateful=False,
        )
        self.assertEqual(list(parts), ["model", "selector_model"])
        self.assertEqual(stateful, [False, False])

        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            export_models(
                parts,
                output_dir,
                output_names=["openvino_model.xml", "openvino_selector_model.xml"],
                input_shapes={"batch_size": 1, "sequence_length": 3},
                stateful=stateful,
                library_name="transformers",
            )
            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            selector_ir = core.read_model(output_dir / "openvino_selector_model.xml")
            self.assertEqual({item.any_name for item in draft_ir.outputs}, {"last_hidden_state"})
            self.assertEqual({item.any_name for item in selector_ir.outputs}, {"edge_scores"})
            self.assertEqual(
                {item.any_name for item in selector_ir.inputs},
                {"candidate_ids", "unary_logits", "draft_hidden_states", "anchor_token_ids"},
            )
            self.assertEqual(draft_ir.get_rt_info()["dflash"]["version"].value, "2")
            for name in ("input_embedding_scale", "output_multiplier", "final_logit_softcapping"):
                self.assertEqual(
                    float(draft_ir.get_rt_info()["dflash"][name].value),
                    config.dflash_config[name],
                )
            self.assertEqual(selector_ir.get_rt_info()["dflash_selector"]["score_semantics"].value, "unary_inclusive")
            self.assertFalse(draft_ir.has_rt_info(["hidden_states_decoder_layers"]))
            self.assertFalse(selector_ir.has_rt_info(["hidden_states_decoder_layers"]))

            codebook_shape = (config.vocab_size, config.dflash_config["selector_rank"])
            self.assertEqual(self._constant_count_with_shape(draft_ir, codebook_shape), 0)
            self.assertGreaterEqual(self._constant_count_with_shape(selector_ir, codebook_shape), 2)

            compiled_draft = core.compile_model(draft_ir, "CPU")
            compiled_selector = core.compile_model(selector_ir, "CPU")
            output_head = torch.randn(config.hidden_size, config.vocab_size)
            for num_draft_tokens in (2, 5):
                with self.subTest(num_draft_tokens=num_draft_tokens):
                    backbone_inputs = self._backbone_inputs(config, num_draft_tokens)
                    with torch.no_grad():
                        torch_hidden = model(**backbone_inputs, use_cache=False).last_hidden_state
                    ov_hidden = compiled_draft({name: value.numpy() for name, value in backbone_inputs.items()})[0]
                    torch.testing.assert_close(
                        torch.from_numpy(np.array(ov_hidden)),
                        torch_hidden,
                        rtol=1e-4,
                        atol=1e-5,
                    )
                    ov_logits = torch.from_numpy(np.array(ov_hidden)) @ output_head
                    torch_logits = torch_hidden @ output_head
                    torch.testing.assert_close(ov_logits, torch_logits, rtol=1e-4, atol=1e-5)
                    self.assertTrue(
                        torch.equal(
                            ov_logits.topk(config.dflash_config["selector_top_k"], dim=-1).indices,
                            torch_logits.topk(config.dflash_config["selector_top_k"], dim=-1).indices,
                        )
                    )

                    candidate_ids = torch.randint(
                        config.vocab_size,
                        (1, num_draft_tokens, config.dflash_config["selector_top_k"]),
                    )
                    unary_logits = torch.randn(1, num_draft_tokens, config.dflash_config["selector_top_k"])
                    selector_inputs = {
                        "candidate_ids": candidate_ids,
                        "unary_logits": unary_logits,
                        "draft_hidden_states": torch_hidden,
                        "anchor_token_ids": torch.tensor([1]),
                    }
                    with torch.no_grad():
                        torch_edges = parts["selector_model"][0](**selector_inputs)
                    ov_edges = compiled_selector({name: value.numpy() for name, value in selector_inputs.items()})[0]
                    torch.testing.assert_close(torch.from_numpy(np.array(ov_edges)), torch_edges)

            if "GPU" in core.available_devices:
                gpu_draft = core.compile_model(draft_ir, "GPU")
                gpu_selector = core.compile_model(selector_ir, "GPU")
                for num_draft_tokens in (2, 5):
                    backbone_inputs = self._backbone_inputs(config, num_draft_tokens)
                    gpu_hidden = np.array(
                        gpu_draft({name: value.numpy() for name, value in backbone_inputs.items()})[0]
                    )
                    self.assertEqual(gpu_hidden.shape, (1, num_draft_tokens, config.hidden_size))
                    self.assertTrue(np.isfinite(gpu_hidden).all())

                    top_k = config.dflash_config["selector_top_k"]
                    selector_inputs = {
                        "candidate_ids": np.random.randint(
                            config.vocab_size,
                            size=(1, num_draft_tokens, top_k),
                            dtype=np.int64,
                        ),
                        "unary_logits": np.random.randn(1, num_draft_tokens, top_k).astype(np.float32),
                        "draft_hidden_states": gpu_hidden,
                        "anchor_token_ids": np.array([1], dtype=np.int64),
                    }
                    gpu_edges = np.array(gpu_selector(selector_inputs)[0])
                    self.assertEqual(gpu_edges.shape, (1, num_draft_tokens, top_k, top_k))
                    self.assertTrue(np.isfinite(gpu_edges).all())

            (output_dir / "openvino_selector_model.xml").unlink()
            (output_dir / "openvino_selector_model.bin").unlink()
            self.assertFalse((output_dir / "openvino_selector_model.xml").exists())
            selectorless_draft = core.compile_model(core.read_model(output_dir / "openvino_model.xml"), "CPU")
            self.assertEqual({item.any_name for item in selectorless_draft.outputs}, {"last_hidden_state"})
            selectorless_inputs = self._backbone_inputs(config, num_draft_tokens=3)
            with torch.no_grad():
                selectorless_reference = model(**selectorless_inputs, use_cache=False).last_hidden_state
            selectorless_output = selectorless_draft(
                {name: value.numpy() for name, value in selectorless_inputs.items()}
            )[0]
            torch.testing.assert_close(
                torch.from_numpy(np.array(selectorless_output)),
                selectorless_reference,
                rtol=1e-4,
                atol=1e-5,
            )

    def test_stateful_draft_compression(self):
        with TemporaryDirectory() as tmpdirname:
            stateful_dir = Path(tmpdirname) / "stateful"
            main_export(
                self._model_path(),
                stateful_dir,
                task="text-generation-with-past",
                stateful=True,
                trust_remote_code=True,
            )
            core = ov.Core()
            draft_ir = core.read_model(stateful_dir / "openvino_model.xml")
            selector_ir = core.read_model(stateful_dir / "openvino_selector_model.xml")
            self.assertGreater(len(draft_ir.get_variables()), 0)
            self.assertEqual(len(selector_ir.get_variables()), 0)

            compressed_draft = nncf.compress_weights(draft_ir, mode=nncf.CompressWeightsMode.INT8_ASYM)
            self.assertTrue(compressed_draft.has_rt_info(["dflash", "version"]))
            self.assertTrue(selector_ir.has_rt_info(["dflash_selector", "dflash_version"]))

    def test_explicit_cache_export_contract(self):
        """Explicit cache ports are an exporter contract; the DFlash CB runner uses the stateful form."""
        with TemporaryDirectory() as tmpdirname:
            explicit_cache_dir = Path(tmpdirname) / "explicit_cache"
            main_export(
                self._model_path(),
                explicit_cache_dir,
                task="text-generation-with-past",
                stateful=False,
                trust_remote_code=True,
            )
            explicit_draft = ov.Core().read_model(explicit_cache_dir / "openvino_model.xml")
            self.assertTrue(any(item.any_name.startswith("past_key_values") for item in explicit_draft.inputs))
            self.assertTrue(any(item.any_name.startswith("present") for item in explicit_draft.outputs))
            self.assertTrue((explicit_cache_dir / "openvino_selector_model.xml").exists())

    def test_dynamo_export_preserves_dflash2_input_contract(self):
        model = self._model()
        _, parts, stateful = _get_submodels_and_export_configs(
            model,
            task="text-generation",
            monolith=False,
            custom_export_configs={},
            custom_architecture=False,
            _variant="default",
            library_name="transformers",
            stateful=False,
        )
        with TemporaryDirectory() as tmpdirname, patch.dict(os.environ, {"OPENVINO_DYNAMO_EXPORT": "true"}):
            output_dir = Path(tmpdirname)
            export_models(
                parts,
                output_dir,
                output_names=["openvino_model.xml", "openvino_selector_model.xml"],
                input_shapes={"batch_size": 2, "sequence_length": 3},
                stateful=stateful,
                library_name="transformers",
            )
            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            selector_ir = core.read_model(output_dir / "openvino_selector_model.xml")
            self.assertEqual(
                [(item.any_name, len(item.partial_shape), item.element_type) for item in draft_ir.inputs],
                [
                    ("inputs_embeds", 3, ov.Type.f32),
                    ("hidden_states", 3, ov.Type.f32),
                    ("position_ids", 2, ov.Type.i64),
                    ("attention_mask", 2, ov.Type.i64),
                ],
            )
            self.assertEqual(
                [item.any_name for item in selector_ir.inputs],
                ["candidate_ids", "unary_logits", "draft_hidden_states", "anchor_token_ids"],
            )

    def test_bf16_two_ir_export_keeps_fp32_runtime_contract(self):
        model = self._model().to(torch.bfloat16)
        _, parts, stateful = _get_submodels_and_export_configs(
            model,
            task="text-generation",
            monolith=False,
            custom_export_configs={},
            custom_architecture=False,
            _variant="default",
            library_name="transformers",
            stateful=False,
        )
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            export_models(
                parts,
                output_dir,
                output_names=["openvino_model.xml", "openvino_selector_model.xml"],
                input_shapes={"batch_size": 1, "sequence_length": 3},
                stateful=stateful,
                patch_16bit_model=True,
                library_name="transformers",
            )
            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            selector_ir = core.read_model(output_dir / "openvino_selector_model.xml")
            self.assertEqual(
                {item.any_name: item.element_type for item in draft_ir.inputs},
                {
                    "inputs_embeds": ov.Type.f32,
                    "hidden_states": ov.Type.f32,
                    "position_ids": ov.Type.i64,
                    "attention_mask": ov.Type.i64,
                },
            )
            self.assertEqual(selector_ir.output("edge_scores").element_type, ov.Type.f32)
            codebook_shape = [model.config.vocab_size, model.config.dflash_config["selector_rank"]]
            codebook_types = [
                op.get_output_element_type(0)
                for op in selector_ir.get_ops()
                if op.get_type_name() == "Constant" and list(op.get_output_shape(0)) == codebook_shape
            ]
            self.assertEqual(codebook_types, [ov.Type.bf16, ov.Type.bf16])
            core.compile_model(draft_ir, "CPU")
            core.compile_model(selector_ir, "CPU")
