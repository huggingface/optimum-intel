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
import math
import os
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import nncf
import numpy as np
import openvino as ov
import torch
from openvino._offline_transformations import paged_attention_transformation
from parameterized import parameterized
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from utils_tests import MODEL_NAMES, get_dflash2_model_path, get_dspark_model_path

from optimum.exporters.openvino import export_from_model
from optimum.exporters.openvino.__main__ import infer_task, main_export
from optimum.exporters.openvino.convert import _get_submodels_and_export_configs, export_models
from optimum.exporters.openvino.dflash_utils import (
    MARKOV_HEAD_DEFAULT_TOP_K,
    DSparkConfig,
    parse_and_validate_dflash_config,
    parse_and_validate_dspark_config,
)
from optimum.exporters.openvino.model_configs import Qwen3OpenVINOConfig
from optimum.exporters.openvino.model_patcher import (
    DFlash2CandidateSelector,
    DSparkVanillaMarkovHead,
    DSparkVanillaMarkovHeadForExport,
    GroupedDynamicCausalConv,
    Qwen3DFlash2ForCausalLM,
    Qwen3DFlashForCausalLM,
    Qwen3DSparkForCausalLM,
    _dflash_block_tap_masks,
)
from optimum.intel.openvino.utils import TemporaryDirectory
from optimum.intel.utils.import_utils import is_transformers_version
from optimum.utils.input_generators import DummyTextInputGenerator


def _per_row_inputs(config, context_length, block_length, committed_length):
    """Per-row draft inputs over the new rows [context delta ; block] of one sequence."""
    num_rows = context_length + block_length
    dflash_config = getattr(config, "dflash_config", None) or {}
    num_target_layers = len(dflash_config.get("target_layer_ids") or config.target_layer_ids)
    inputs_embeds = torch.randn(1, num_rows, config.hidden_size)
    inputs_embeds[:, :context_length] = 0
    hidden_states = torch.randn(1, num_rows, config.hidden_size * num_target_layers)
    hidden_states[:, context_length:] = 0
    return {
        "inputs_embeds": inputs_embeds,
        "hidden_states": hidden_states,
        "position_ids": torch.arange(committed_length, committed_length + num_rows).reshape(1, -1),
        "attention_mask": torch.ones(1, committed_length + num_rows, dtype=torch.int64),
        "token_type_ids": torch.tensor([[0] * context_length + [1] * block_length]),
    }


def _export_draft(model, output_dir, task="text-generation", stateful=False, patch_16bit_model=False, batch_size=1):
    """Export a draft, and its Markov head model if it has one, as openvino_<part>.xml files."""
    _, parts, stateful = _get_submodels_and_export_configs(
        model,
        task=task,
        monolith=False,
        custom_export_configs={},
        custom_architecture=False,
        _variant="default",
        library_name="transformers",
        stateful=stateful,
    )
    export_models(
        parts,
        output_dir,
        output_names=[f"openvino_{name}.xml" for name in parts],
        input_shapes={"batch_size": batch_size, "sequence_length": 3},
        stateful=stateful,
        patch_16bit_model=patch_16bit_model,
        library_name="transformers",
    )
    return parts


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
            locators = self._assert_hidden_state_rt_info_is_valid(annotated_model)
            self.assertEqual(len(locators), model.config.get_text_config().num_hidden_layers)

    @parameterized.expand(("qwen3", "qwen3_moe"))
    def test_export_hidden_state_locators_for_representative_decoder_models(self, model_type):
        self._export_and_assert_hidden_state_locators(
            model_type, AutoModelForCausalLM, "text-generation", "openvino_model.xml"
        )

    @parameterized.expand(("qwen3_5", "qwen3_5_moe", "gemma4", "minicpmv4_7"))
    def test_export_hidden_state_locators_for_representative_multi_modal_models(self, model_type):
        if model_type in {"qwen3_5", "qwen3_5_moe"} and not (
            is_transformers_version(">=", "5.2.0") and is_transformers_version("<=", "5.2.99")
        ):
            self.skipTest("Qwen3.5 hidden-state locator coverage requires Transformers >= 5.2.0 and <= 5.2.99")
        if model_type == "gemma4" and not is_transformers_version(">=", "5.5.0"):
            self.skipTest("Gemma 4 hidden-state locator coverage requires Transformers >= 5.5.0")
        if model_type == "minicpmv4_7" and not is_transformers_version(">=", "5.18.0"):
            self.skipTest("MiniCPM-V 4.7 hidden-state locator coverage requires Transformers >= 5.18.0")

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
            self.assertFalse(draft_ir.has_rt_info(["dflash", "markov_rank"]))
            self.assertFalse((output_dir / "openvino_markov_head_model.xml").exists())


class DSparkExportTest(unittest.TestCase):
    TOP_K = 4

    @staticmethod
    def _model(sample_from_anchor=True):
        return Qwen3DSparkForCausalLM.from_pretrained(get_dspark_model_path(sample_from_anchor)).eval()

    @staticmethod
    def _dspark_config(**fields):
        config = Qwen3Config(vocab_size=32)
        config.architectures = ["Qwen3DSparkModel"]
        config.mask_token_id = config.vocab_size - 1
        config.target_layer_ids = [0, 1]
        for name, value in fields.items():
            setattr(config, name, value)
        return config

    @staticmethod
    def _reference_greedy_block(markov_head, base_logits, anchor_token_ids, top_k=None):
        """DeepSpec VanillaMarkov.sample_block_tokens(temperature=0), optionally restricted to the Top-K
        backbone candidates of every position as in vLLM's gathered Top-K Markov sampling."""
        tokens = []
        previous_token_ids = anchor_token_ids
        for position in range(base_logits.shape[1]):
            step_logits = base_logits[:, position] + markov_head(previous_token_ids)
            if top_k is not None:
                candidate_ids = base_logits[:, position].topk(top_k, dim=-1).indices
                restricted = torch.full_like(step_logits, float("-inf"))
                step_logits = restricted.scatter(-1, candidate_ids, step_logits.gather(-1, candidate_ids))
            previous_token_ids = step_logits.argmax(dim=-1)
            tokens.append(previous_token_ids)
        return torch.stack(tokens, dim=1)

    @staticmethod
    def _packed_blocks(block_lengths, vocab_size):
        """Draft logits of several blocks packed back to back, one anchor per block, and the block offsets."""
        draft_logits = torch.randn(sum(block_lengths), vocab_size) * 4
        anchor_token_ids = torch.randint(vocab_size, (len(block_lengths),))
        subsequence_begins = torch.tensor([0, *block_lengths]).cumsum(dim=0)
        return {
            "draft_logits": draft_logits,
            "anchor_token_ids": anchor_token_ids,
            "subsequence_begins": subsequence_begins,
        }

    @staticmethod
    def _reference_edges(markov_w1, markov_w2, draft_logits, anchor_token_id, top_k):
        """Top-K lattice of one draft block: the anchor precedes its first row."""
        unary_logits, candidate_ids = draft_logits.topk(top_k, dim=-1)
        predecessor_ids = torch.cat([torch.full((1, top_k), int(anchor_token_id)), candidate_ids[:-1]])
        edge_scores = markov_w1[predecessor_ids] @ markov_w2[candidate_ids].transpose(-1, -2)
        return edge_scores + unary_logits.unsqueeze(-2), candidate_ids

    @staticmethod
    def _markov_weights(model, config, element_type=None):
        shapes = ([config.vocab_size, config.markov_rank], [config.markov_rank, config.vocab_size])
        return [
            op
            for op in model.get_ops()
            if op.get_type_name() == "Constant"
            and list(op.get_output_shape(0)) in shapes
            and (element_type is None or op.get_output_element_type(0) == element_type)
        ]

    def test_config_parsing_and_validation(self):
        self.assertEqual(
            parse_and_validate_dspark_config(self._dspark_config(markov_rank=8, markov_head_type="vanilla")),
            DSparkConfig(markov_rank=8, markov_head_type="vanilla", sample_from_anchor=True),
        )
        self.assertEqual(
            parse_and_validate_dspark_config(self._dspark_config(markov_rank=8)).markov_head_type,
            "vanilla",
        )
        self.assertEqual(parse_and_validate_dspark_config(self._dspark_config()).markov_rank, 0)
        self.assertEqual(
            parse_and_validate_dspark_config(self._dspark_config(markov_rank=0, markov_head_type="gated")).markov_rank,
            0,
        )
        nested_config = self._dspark_config(dflash_config={"markov_rank": 8, "sample_from_anchor": False})
        self.assertEqual(
            parse_and_validate_dspark_config(nested_config),
            DSparkConfig(markov_rank=8, markov_head_type="vanilla", sample_from_anchor=False),
        )
        self.assertEqual(parse_and_validate_dspark_config(nested_config).candidate_position_offset, 1)

        for head_type in ("gated", "rnn"):
            with self.subTest(head_type=head_type), self.assertRaisesRegex(ValueError, "not supported"):
                parse_and_validate_dspark_config(self._dspark_config(markov_rank=8, markov_head_type=head_type))
        for fields in (
            {"markov_rank": -1},
            {"markov_rank": True},
            {"markov_rank": 8, "sample_from_anchor": "true"},
            {"markov_rank": 8, "dflash_config": {"markov_rank": 16}},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                parse_and_validate_dspark_config(self._dspark_config(**fields))

        config = self._dspark_config(markov_rank=8)
        self.assertEqual(Qwen3OpenVINOConfig(config).candidate_position_offset, 0)
        masked_anchor_config = self._dspark_config(markov_rank=8, sample_from_anchor=False)
        self.assertEqual(Qwen3OpenVINOConfig(masked_anchor_config).candidate_position_offset, 1)
        with self.assertRaisesRegex(ValueError, "not supported"):
            Qwen3OpenVINOConfig(self._dspark_config(markov_rank=8, markov_head_type="rnn"))

    def test_markov_rank_zero_draft_exports_a_plain_dflash_package(self):
        # DeepSpec publishes its DFlash drafts as DSpark checkpoints with markov_rank=0.
        config = AutoConfig.from_pretrained(MODEL_NAMES["qwen3_deepspec_dflash"])
        self.assertEqual(config.architectures, ["Qwen3DSparkModel"])
        self.assertEqual(config.markov_rank, 0)
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            main_export(
                MODEL_NAMES["qwen3_deepspec_dflash"],
                output=output_dir,
                task="text-generation",
            )
            self.assertEqual([path.name for path in output_dir.glob("openvino_*model.xml")], ["openvino_model.xml"])
            draft_ir = ov.Core().read_model(output_dir / "openvino_model.xml")
            self.assertEqual(draft_ir.get_rt_info()["dflash_mode"].value, "True")
            draft_rt_info = draft_ir.get_rt_info()["dflash"]
            self.assertEqual(
                {name: draft_rt_info[name].value for name in draft_rt_info},
                {
                    "mask_token_id": str(config.mask_token_id),
                    "target_layer_ids": ",".join(map(str, config.target_layer_ids)),
                    "candidate_position_offset": "0",
                },
            )

    @staticmethod
    def _keep_computation(markov_head_ir, input_names, output_names, top_k=None):
        """Keep one computation of the Markov head model and optionally replace K, as runtimes do at load time."""
        markov_head_ir = markov_head_ir.clone()
        computation = ov.Model(
            [markov_head_ir.output(name).get_node() for name in output_names],
            [markov_head_ir.input(name).get_node() for name in input_names],
        )
        if top_k is not None:
            (topk,) = [op for op in computation.get_ops() if op.get_type_name() == "TopK"]
            k_type = topk.input_value(1).get_element_type()
            topk.input(1).replace_source_output(
                ov.opset13.constant(np.array(top_k, dtype=k_type.to_dtype())).output(0)
            )
            computation.validate_nodes_and_infer_types()
        return computation

    def test_export_package_and_markov_head_computations(self):
        model = self._model()
        config = model.config
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            parts = _export_draft(model, output_dir)
            self.assertEqual(list(parts), ["model", "markov_head_model"])
            markov_head = parts["markov_head_model"][0]

            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            markov_head_ir = core.read_model(output_dir / "openvino_markov_head_model.xml")
            draft_rt_info = draft_ir.get_rt_info()["dflash"]
            self.assertEqual(draft_rt_info["candidate_position_offset"].value, "0")
            self.assertEqual(draft_rt_info["markov_rank"].value, str(config.markov_rank))
            self.assertEqual(draft_rt_info["markov_head_type"].value, "vanilla")
            self.assertEqual(self._markov_weights(draft_ir, config), [])

            markov_rt_info = markov_head_ir.get_rt_info()["dflash"]
            expected_rt_info = {
                "top_k": str(MARKOV_HEAD_DEFAULT_TOP_K),
                "markov_rank": str(config.markov_rank),
                "markov_head_type": "vanilla",
                "vocab_size": str(config.vocab_size),
            }
            self.assertEqual({name: markov_rt_info[name].value for name in markov_rt_info}, expected_rt_info)
            self.assertFalse(markov_head_ir.has_rt_info(["dflash_mode"]))
            self.assertFalse(markov_head_ir.has_rt_info(["hidden_states_decoder_layers"]))
            self.assertEqual(
                [(item.any_name, item.element_type) for item in markov_head_ir.inputs],
                [
                    ("prev_token_ids", ov.Type.i64),
                    ("draft_logits", ov.Type.f32),
                    ("anchor_token_ids", ov.Type.i64),
                    ("subsequence_begins", ov.Type.i64),
                ],
            )
            self.assertEqual(
                {item.any_name: item.element_type for item in markov_head_ir.outputs},
                {"markov_bias": ov.Type.f32, "edge_scores": ov.Type.f32, "candidate_ids": ov.Type.i64},
            )
            self.assertEqual(len([op for op in markov_head_ir.get_ops() if op.get_type_name() == "TopK"]), 1)
            # Both computations read the same two Markov tables, and the serialized weights hold each once.
            table_bytes = config.vocab_size * config.markov_rank * ov.Type.f32.size
            self.assertGreaterEqual(len(self._markov_weights(markov_head_ir, config)), 2)
            self.assertLess((output_dir / "openvino_markov_head_model.bin").stat().st_size, 3 * table_bytes)

            markov_step_ir = self._keep_computation(markov_head_ir, ["prev_token_ids"], ["markov_bias"])
            self.assertFalse(any(op.get_type_name() == "TopK" for op in markov_step_ir.get_ops()))
            prev_token_ids = torch.tensor([0, 5, config.vocab_size - 1])
            ov_bias = core.compile_model(markov_step_ir, "CPU")({"prev_token_ids": prev_token_ids.numpy()})[0]
            with torch.no_grad():
                reference_bias = markov_head.markov_bias(prev_token_ids)
            torch.testing.assert_close(torch.from_numpy(np.array(ov_bias)), reference_bias, rtol=1e-4, atol=1e-4)

            # Blocks of different lengths in one call match the lattices of the blocks scored one by one.
            block_lengths = [3, 1, 4]
            lattice_inputs = self._packed_blocks(block_lengths, config.vocab_size)
            markov_w1 = markov_head.markov_head.markov_w1.weight.detach()
            markov_w2 = markov_head.markov_head.markov_w2.weight.detach()
            for top_k in (None, self.TOP_K):
                with self.subTest(top_k=top_k):
                    lattice_ir = self._keep_computation(
                        markov_head_ir, list(lattice_inputs), ["edge_scores", "candidate_ids"], top_k
                    )
                    expected_top_k = top_k or MARKOV_HEAD_DEFAULT_TOP_K
                    self.assertEqual(lattice_ir.output("candidate_ids").partial_shape[1], expected_top_k)
                    compiled_lattice = core.compile_model(lattice_ir, "CPU")
                    ov_outputs = compiled_lattice({name: value.numpy() for name, value in lattice_inputs.items()})
                    ov_candidate_ids = torch.from_numpy(np.array(ov_outputs[compiled_lattice.output("candidate_ids")]))
                    ov_edges = torch.from_numpy(np.array(ov_outputs[compiled_lattice.output("edge_scores")]))
                    begins = lattice_inputs["subsequence_begins"].tolist()
                    for block, (begin, end) in enumerate(zip(begins[:-1], begins[1:])):
                        expected_edges, expected_ids = self._reference_edges(
                            markov_w1,
                            markov_w2,
                            lattice_inputs["draft_logits"][begin:end],
                            lattice_inputs["anchor_token_ids"][block],
                            expected_top_k,
                        )
                        self.assertTrue(torch.equal(ov_candidate_ids[begin:end], expected_ids))
                        torch.testing.assert_close(ov_edges[begin:end], expected_edges, rtol=1e-4, atol=1e-4)

    def test_dense_walk_and_lattice_reproduce_deepspec_markov_sampling(self):
        torch.manual_seed(4)
        vocab_size, markov_rank, num_draft_tokens = 24, 4, 6
        config = self._dspark_config(vocab_size=vocab_size, markov_rank=markov_rank)
        markov_head = DSparkVanillaMarkovHead(vocab_size, markov_rank).eval()
        torch.nn.init.normal_(markov_head.markov_w1.weight)
        torch.nn.init.normal_(markov_head.markov_w2.weight)
        base_logits = torch.randn(1, num_draft_tokens, vocab_size) * 2
        anchor_token_ids = torch.tensor([5])

        with torch.no_grad():
            reference_tokens = self._reference_greedy_block(markov_head, base_logits, anchor_token_ids)
            self.assertFalse(torch.equal(base_logits.argmax(dim=-1), reference_tokens))

            # Dense runtime walk: one dense-step call per draft position on the previously drafted token.
            markov_head_for_export = DSparkVanillaMarkovHeadForExport(markov_head, config, top_k=vocab_size)
            previous_token_ids = anchor_token_ids
            dense_tokens = []
            for position in range(num_draft_tokens):
                step_logits = base_logits[:, position] + markov_head_for_export.markov_bias(previous_token_ids)
                previous_token_ids = step_logits.argmax(dim=-1)
                dense_tokens.append(previous_token_ids)
            self.assertTrue(torch.equal(torch.stack(dense_tokens, dim=1), reference_tokens))

            for top_k in (vocab_size, 7):
                with self.subTest(top_k=top_k):
                    edge_scores, candidate_ids = markov_head_for_export.score_candidates(
                        base_logits[0], anchor_token_ids, torch.tensor([0, num_draft_tokens]), top_k
                    )
                    predecessor_ids = torch.cat([anchor_token_ids.expand(1, top_k), candidate_ids[:-1]])
                    for position in range(num_draft_tokens):
                        corrected_logits = base_logits[0, position] + markov_head(predecessor_ids[position])
                        expected_edges = corrected_logits.gather(-1, candidate_ids[position].expand(top_k, -1))
                        torch.testing.assert_close(edge_scores[position], expected_edges, rtol=1e-5, atol=1e-5)

                    lattice_tokens = []
                    previous_index = 0
                    for position in range(num_draft_tokens):
                        previous_index = int(edge_scores[position, previous_index].argmax())
                        lattice_tokens.append(int(candidate_ids[position, previous_index]))
                    expected_tokens = self._reference_greedy_block(markov_head, base_logits, anchor_token_ids, top_k)
                    self.assertEqual(lattice_tokens, expected_tokens[0].tolist())
                    if top_k == vocab_size:
                        self.assertEqual(lattice_tokens, reference_tokens[0].tolist())

    def test_bf16_markov_head_model_keeps_checkpoint_weights_and_fp32_outputs(self):
        model = self._model().to(torch.bfloat16)
        config = model.config
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            _export_draft(model, output_dir, patch_16bit_model=True)
            core = ov.Core()
            markov_head_ir = core.read_model(output_dir / "openvino_markov_head_model.xml")
            self.assertEqual(markov_head_ir.output("markov_bias").element_type, ov.Type.f32)
            self.assertEqual(markov_head_ir.input("draft_logits").element_type, ov.Type.f32)
            self.assertEqual(markov_head_ir.output("edge_scores").element_type, ov.Type.f32)
            self.assertEqual(self._markov_weights(markov_head_ir, config, ov.Type.f32), [])
            self.assertGreaterEqual(len(self._markov_weights(markov_head_ir, config, ov.Type.bf16)), 2)

            markov_w1 = model.markov_head.markov_w1.weight.float()
            markov_w2 = model.markov_head.markov_w2.weight.float()
            compiled_markov_head = core.compile_model(markov_head_ir, "CPU")
            lattice_inputs = self._packed_blocks([3], config.vocab_size)
            ov_outputs = compiled_markov_head(
                {
                    "prev_token_ids": np.array([3], dtype=np.int64),
                    **{name: value.numpy() for name, value in lattice_inputs.items()},
                }
            )
            ov_bias = torch.from_numpy(np.array(ov_outputs[compiled_markov_head.output("markov_bias")]))
            torch.testing.assert_close(ov_bias, markov_w1[[3]] @ markov_w2.T, rtol=1e-4, atol=1e-4)

            expected_edges, expected_ids = self._reference_edges(
                markov_w1,
                markov_w2,
                lattice_inputs["draft_logits"],
                lattice_inputs["anchor_token_ids"][0],
                MARKOV_HEAD_DEFAULT_TOP_K,
            )
            candidate_ids = torch.from_numpy(np.array(ov_outputs[compiled_markov_head.output("candidate_ids")]))
            self.assertTrue(torch.equal(candidate_ids, expected_ids))
            ov_edges = torch.from_numpy(np.array(ov_outputs[compiled_markov_head.output("edge_scores")]))
            torch.testing.assert_close(ov_edges, expected_edges, rtol=1e-4, atol=1e-4)


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
        return _per_row_inputs(config, context_length, num_draft_tokens + 1, committed_length=0)

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

    def test_task_is_inferred_for_every_draft_architecture(self):
        for model_path in (MODEL_NAMES["qwen3_dflash"], self._model_path(), get_dspark_model_path()):
            with self.subTest(model_path=model_path):
                self.assertEqual(
                    infer_task("auto", model_path, library_name="transformers"), "text-generation-with-past"
                )

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
            Qwen3OpenVINOConfig(missing_config)

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

    def test_grouped_dynamic_convolution_reference_parity(self):
        config = self._config()
        convolution = GroupedDynamicCausalConv(config).eval()
        with torch.no_grad():
            convolution.base_kernel.copy_(torch.randn_like(convolution.base_kernel))
        context_length = 3
        for block_length in (2, 4, 7):
            with self.subTest(block_length=block_length):
                # two sequences of rows [context ; block]; the convolution sees only their block rows, packed back
                # to back, so a tap must not reach from the second block into the first
                token_type_ids = torch.tensor([[0] * context_length + [1] * block_length] * 2)
                block_rows = torch.nonzero(token_type_ids.flatten() == 1).squeeze(-1)
                block_tap_masks = _dflash_block_tap_masks(token_type_ids, block_rows, convolution.kernel_size)
                hidden_states = torch.randn(2, block_length, config.hidden_size)
                projected = convolution.kernel_projection(hidden_states).reshape(
                    2,
                    block_length,
                    2,
                    convolution.kernel_size,
                    convolution.num_groups,
                )
                prepared, output_kernel = convolution.prepare(hidden_states.flatten(0, 1), block_tap_masks)
                reference_prepared = self._reference_convolution(
                    hidden_states,
                    projected[:, :, 0],
                    convolution.base_kernel[0],
                    convolution.group_size,
                )
                torch.testing.assert_close(prepared.view_as(hidden_states), reference_prepared)
                torch.testing.assert_close(output_kernel, projected[:, :, 1].reshape_as(output_kernel))

                branch_output = torch.randn_like(hidden_states)
                finished = convolution.finish(branch_output.flatten(0, 1), output_kernel, block_tap_masks)
                reference_finished = self._reference_convolution(
                    branch_output,
                    projected[:, :, 1],
                    convolution.base_kernel[1],
                    convolution.group_size,
                )
                torch.testing.assert_close(finished.view_as(hidden_states), reference_finished)

    def test_selector_edge_scores_reference_parity(self):
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

    def test_markov_head_scores_packed_blocks_like_the_candidate_selector(self):
        torch.manual_seed(2)
        model = self._model()
        config = model.config
        dflash_config = config.dflash_config
        top_k = dflash_config["selector_top_k"]
        markov_head = model.markov_head_for_export(top_k)
        block_lengths = [3, 1, 4]
        draft_logits = torch.randn(sum(block_lengths), config.vocab_size) * 4
        draft_hidden_states = torch.randn(sum(block_lengths), config.hidden_size)
        anchor_token_ids = torch.randint(config.vocab_size, (len(block_lengths),))
        subsequence_begins = torch.tensor([0, *block_lengths]).cumsum(dim=0)
        softcap = dflash_config["final_logit_softcapping"]
        with torch.no_grad():
            outputs = markov_head(draft_logits, draft_hidden_states, anchor_token_ids, subsequence_begins)
            begins = subsequence_begins.tolist()
            for block, (begin, end) in enumerate(zip(begins[:-1], begins[1:])):
                unary_logits, candidate_ids = draft_logits[begin:end].topk(top_k, dim=-1)
                unary_logits = torch.tanh(unary_logits * dflash_config["output_multiplier"] / softcap) * softcap
                expected_edges = model.candidate_selector(
                    candidate_ids[None],
                    unary_logits[None],
                    draft_hidden_states[None, begin:end],
                    anchor_token_ids[block : block + 1],
                )[0]
                self.assertTrue(torch.equal(outputs.candidate_ids[begin:end], candidate_ids))
                torch.testing.assert_close(outputs.edge_scores[begin:end], expected_edges)

    def test_markov_head_export_dynamic_numerical_parity(self):
        torch.manual_seed(1)
        model = self._model()
        config = model.config
        dflash_config = config.dflash_config
        top_k = dflash_config["selector_top_k"]
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            parts = _export_draft(model, output_dir)
            self.assertEqual(list(parts), ["model", "markov_head_model"])
            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            markov_head_ir = core.read_model(output_dir / "openvino_markov_head_model.xml")
            self.assertEqual({item.any_name for item in draft_ir.outputs}, {"last_hidden_state"})
            self.assertEqual(
                {item.any_name: item.element_type for item in markov_head_ir.inputs},
                {
                    "draft_logits": ov.Type.f32,
                    "draft_hidden_states": ov.Type.f32,
                    "anchor_token_ids": ov.Type.i64,
                    "subsequence_begins": ov.Type.i64,
                },
            )
            self.assertEqual(
                {item.any_name: item.element_type for item in markov_head_ir.outputs},
                {"edge_scores": ov.Type.f32, "candidate_ids": ov.Type.i64},
            )

            draft_rt_info = draft_ir.get_rt_info()["dflash"]
            self.assertEqual(draft_rt_info["version"].value, "2")
            self.assertEqual(draft_rt_info["candidate_position_offset"].value, "1")
            self.assertEqual(draft_rt_info["markov_rank"].value, str(dflash_config["selector_rank"]))
            self.assertEqual(draft_rt_info["markov_head_type"].value, "dflash2")
            for name in ("input_embedding_scale", "output_multiplier", "final_logit_softcapping"):
                self.assertEqual(float(draft_rt_info[name].value), dflash_config[name])
            markov_rt_info = markov_head_ir.get_rt_info()["dflash"]
            self.assertEqual(
                {name: markov_rt_info[name].value for name in markov_rt_info},
                {
                    "top_k": str(top_k),
                    "markov_rank": str(dflash_config["selector_rank"]),
                    "markov_head_type": "dflash2",
                    "vocab_size": str(config.vocab_size),
                },
            )
            self.assertFalse(markov_head_ir.has_rt_info(["dflash_mode"]))
            self.assertFalse(draft_ir.has_rt_info(["hidden_states_decoder_layers"]))
            self.assertFalse(markov_head_ir.has_rt_info(["hidden_states_decoder_layers"]))
            self.assertEqual(len([op for op in markov_head_ir.get_ops() if op.get_type_name() == "TopK"]), 1)

            codebook_shape = (config.vocab_size, dflash_config["selector_rank"])
            self.assertEqual(self._constant_count_with_shape(draft_ir, codebook_shape), 0)
            self.assertGreaterEqual(self._constant_count_with_shape(markov_head_ir, codebook_shape), 2)

            compiled_markov_head = core.compile_model(markov_head_ir, "CPU")
            output_head = torch.randn(config.hidden_size, config.vocab_size)
            context_length = 3
            for num_draft_tokens in (2, 5):
                with self.subTest(num_draft_tokens=num_draft_tokens):
                    backbone_inputs = self._backbone_inputs(config, num_draft_tokens, context_length)
                    with torch.no_grad():
                        torch_hidden = model(**backbone_inputs, use_cache=False).last_hidden_state

                    # the runtime scores the candidate rows, the block rows after the anchor row
                    draft_hidden_states = torch_hidden[0, context_length + 1 :]
                    markov_head_inputs = {
                        "draft_logits": draft_hidden_states @ output_head,
                        "draft_hidden_states": draft_hidden_states,
                        "anchor_token_ids": torch.tensor([1]),
                        "subsequence_begins": torch.tensor([0, num_draft_tokens]),
                    }
                    with torch.no_grad():
                        reference = parts["markov_head_model"][0](**markov_head_inputs)
                    ov_outputs = compiled_markov_head(
                        {name: value.numpy() for name, value in markov_head_inputs.items()}
                    )
                    ov_edges = torch.from_numpy(np.array(ov_outputs[compiled_markov_head.output("edge_scores")]))
                    torch.testing.assert_close(ov_edges, reference.edge_scores, rtol=1e-4, atol=1e-4)
                    ov_candidate_ids = np.array(ov_outputs[compiled_markov_head.output("candidate_ids")])
                    self.assertTrue(torch.equal(torch.from_numpy(ov_candidate_ids), reference.candidate_ids))

            if "GPU" in core.available_devices:
                gpu_draft = core.compile_model(draft_ir, "GPU")
                gpu_markov_head = core.compile_model(markov_head_ir, "GPU")
                for num_draft_tokens in (2, 5):
                    backbone_inputs = self._backbone_inputs(config, num_draft_tokens, context_length)
                    gpu_hidden = np.array(
                        gpu_draft({name: value.numpy() for name, value in backbone_inputs.items()})[0]
                    )
                    self.assertEqual(gpu_hidden.shape, (1, context_length + num_draft_tokens + 1, config.hidden_size))
                    self.assertTrue(np.isfinite(gpu_hidden).all())

                    gpu_outputs = gpu_markov_head(
                        {
                            "draft_logits": np.random.randn(num_draft_tokens, config.vocab_size).astype(np.float32),
                            "draft_hidden_states": gpu_hidden[0, context_length + 1 :],
                            "anchor_token_ids": np.array([1], dtype=np.int64),
                            "subsequence_begins": np.array([0, num_draft_tokens], dtype=np.int64),
                        }
                    )
                    gpu_edges = np.array(gpu_outputs[gpu_markov_head.output("edge_scores")])
                    gpu_candidate_ids = np.array(gpu_outputs[gpu_markov_head.output("candidate_ids")])
                    self.assertTrue(np.issubdtype(gpu_candidate_ids.dtype, np.integer))
                    self.assertEqual(gpu_candidate_ids.shape, (num_draft_tokens, top_k))
                    self.assertEqual(gpu_edges.shape, (num_draft_tokens, top_k, top_k))
                    self.assertTrue(np.isfinite(gpu_edges).all())

    def test_stateful_export_keeps_the_markov_head_stateless(self):
        with TemporaryDirectory() as tmpdirname:
            stateful_dir = Path(tmpdirname) / "stateful"
            main_export(
                self._model_path(),
                stateful_dir,
                task="text-generation-with-past",
                stateful=True,
            )
            core = ov.Core()
            draft_ir = core.read_model(stateful_dir / "openvino_model.xml")
            markov_head_ir = core.read_model(stateful_dir / "openvino_markov_head_model.xml")
            self.assertGreater(len(draft_ir.get_variables()), 0)
            self.assertEqual(len(markov_head_ir.get_variables()), 0)

    def test_explicit_cache_export_contract(self):
        """Explicit cache ports remain an exporter option; GenAI converts the stateful draft to PagedAttention."""
        with TemporaryDirectory() as tmpdirname:
            explicit_cache_dir = Path(tmpdirname) / "explicit_cache"
            main_export(
                self._model_path(),
                explicit_cache_dir,
                task="text-generation-with-past",
                stateful=False,
            )
            explicit_draft = ov.Core().read_model(explicit_cache_dir / "openvino_model.xml")
            self.assertTrue(any(item.any_name.startswith("past_key_values") for item in explicit_draft.inputs))
            self.assertTrue(any(item.any_name.startswith("present") for item in explicit_draft.outputs))
            self.assertTrue((explicit_cache_dir / "openvino_markov_head_model.xml").exists())

    def test_dynamo_export_preserves_dflash2_input_contract(self):
        model = self._model()
        with TemporaryDirectory() as tmpdirname, patch.dict(os.environ, {"OPENVINO_DYNAMO_EXPORT": "true"}):
            output_dir = Path(tmpdirname)
            _export_draft(model, output_dir, batch_size=2)
            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            markov_head_ir = core.read_model(output_dir / "openvino_markov_head_model.xml")
            self.assertEqual(
                [(item.any_name, len(item.partial_shape), item.element_type) for item in draft_ir.inputs],
                [
                    ("inputs_embeds", 3, ov.Type.f32),
                    ("hidden_states", 3, ov.Type.f32),
                    ("position_ids", 2, ov.Type.i64),
                    ("attention_mask", 2, ov.Type.i64),
                    ("token_type_ids", 2, ov.Type.i64),
                ],
            )
            self.assertEqual(
                [item.any_name for item in markov_head_ir.inputs],
                ["draft_logits", "draft_hidden_states", "anchor_token_ids", "subsequence_begins"],
            )

    def test_bf16_markov_head_export_keeps_fp32_runtime_contract(self):
        model = self._model().to(torch.bfloat16)
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            _export_draft(model, output_dir, patch_16bit_model=True)
            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            markov_head_ir = core.read_model(output_dir / "openvino_markov_head_model.xml")
            self.assertEqual(
                {item.any_name: item.element_type for item in draft_ir.inputs},
                {
                    "inputs_embeds": ov.Type.f32,
                    "hidden_states": ov.Type.f32,
                    "position_ids": ov.Type.i64,
                    "attention_mask": ov.Type.i64,
                    "token_type_ids": ov.Type.i64,
                },
            )
            self.assertEqual(markov_head_ir.input("draft_hidden_states").element_type, ov.Type.f32)
            self.assertEqual(markov_head_ir.output("edge_scores").element_type, ov.Type.f32)
            codebook_shape = [model.config.vocab_size, model.config.dflash_config["selector_rank"]]
            codebook_types = [
                op.get_output_element_type(0)
                for op in markov_head_ir.get_ops()
                if op.get_type_name() == "Constant" and list(op.get_output_shape(0)) == codebook_shape
            ]
            self.assertEqual(codebook_types, [ov.Type.bf16, ov.Type.bf16])
            core.compile_model(draft_ir, "CPU")
            core.compile_model(markov_head_ir, "CPU")


_SLIDING_WINDOW = 6
# GenAI pages the KV cache in blocks of 32 tokens on CPU
_CPU_KV_BLOCK_SIZE = 32
# (context delta, block) rows per draft step of two sequences drafted in one call: the first step of the first
# sequence spans two cache blocks, every step reaches past the sliding window, and only the context rows in
# front of each block keep the blocks of different sequences apart
_DRAFT_STEPS = (((34, 4), (3, 4), (1, 4)), ((5, 4), (1, 4), (2, 4)))
# PagedAttentionExtension input ports
_PA_KEY_CACHE, _PA_SLIDING_WINDOW, _PA_TOKEN_TYPE_IDS = 3, 10, 25


@unittest.skipUnless(is_transformers_version(">=", "4.57"), "DFlash-2 requires Transformers 4.57 or newer")
class DFlashPagedAttentionTest(unittest.TestCase):
    """The draft IR of every DFlash variant runs on PagedAttention, the DFlash backend of OpenVINO GenAI."""

    VARIANTS = (("dflash",), ("dflash2",), ("dspark",))

    @staticmethod
    def _model(variant):
        config = Qwen3Config.from_pretrained(MODEL_NAMES["qwen3"])
        num_target_layers = config.num_hidden_layers
        config.num_hidden_layers = 3
        config.num_target_layers = num_target_layers
        # grouped-query attention, as in the released drafts, so the converted graph covers the K/V head expansion
        config.num_key_value_heads = 1
        config.layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
        config.use_sliding_window = True
        config.sliding_window = _SLIDING_WINDOW
        config.is_causal = False
        target_layer_ids = [0, num_target_layers - 1]
        mask_token_id = config.vocab_size - 1
        if variant == "dflash":
            config.architectures = ["DFlashDraftModel"]
            config.dflash_config = {"mask_token_id": mask_token_id, "target_layer_ids": target_layer_ids}
            model_class = Qwen3DFlashForCausalLM
        elif variant == "dflash2":
            config.architectures = ["DFlash2DraftModel"]
            config.dflash_config = {
                "conv_group_size": 8 if config.hidden_size % 8 == 0 else 1,
                "conv_kernel_size": 3,
                "mask_token_id": mask_token_id,
                "selector_rank": 8,
                "selector_top_k": 4,
                "target_layer_ids": target_layer_ids,
            }
            model_class = Qwen3DFlash2ForCausalLM
        else:
            config.architectures = ["Qwen3DSparkModel"]
            config.markov_rank = 8
            config.mask_token_id = mask_token_id
            config.target_layer_ids = target_layer_ids
            model_class = Qwen3DSparkForCausalLM
        torch.manual_seed(0)
        model = model_class(config).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                # the default base kernel only passes the current row, which would hide taps reaching other rows
                if name.endswith("base_kernel"):
                    parameter.copy_(torch.randn_like(parameter))
        return model

    @staticmethod
    def _reference_outputs(model, steps):
        """Outputs of consecutive draft steps, dropping each block from the cache afterwards as past_lens does.

        The draft builds the cache itself, and its sliding layers must keep the whole past: the mask applies the window.
        """
        cache, outputs = None, []
        with torch.no_grad():
            for inputs in steps:
                output = model(**inputs, past_key_values=cache, use_cache=True)
                outputs.append(output.last_hidden_state)
                cache = output.past_key_values
                cache.crop(cache.get_seq_length() - int(inputs["token_type_ids"].sum()))
        return outputs

    @staticmethod
    def _export_draft(model, output_dir):
        """Export traced with a single block row, the trace in which a one-row value could fix a shape."""
        generate = DummyTextInputGenerator.generate

        def generate_one_block_row(generator, input_name, *args, **kwargs):
            if input_name != "token_type_ids":
                return generate(generator, input_name, *args, **kwargs)
            token_type_ids = torch.zeros(generator.batch_size, generator.sequence_length, dtype=torch.int64)
            token_type_ids[:, -1] = 1
            return token_type_ids

        with patch.object(DummyTextInputGenerator, "generate", generate_one_block_row):
            _export_draft(model, output_dir, task="text-generation-with-past", stateful=True)
        return output_dir / "openvino_model.xml"

    @staticmethod
    def _source_constant(output):
        node = output.get_node()
        while node.get_type_name() != "Constant":
            node = node.input_value(0).get_node()
        return node.get_data()

    def _assert_paged_attention_conversion(self, model, config):
        ops = Counter(op.get_type_name() for op in model.get_ops())
        self.assertEqual(ops["ScaledDotProductAttention"], config.num_hidden_layers)
        paged_attention_transformation(model)
        ops = Counter(op.get_type_name() for op in model.get_ops())
        self.assertEqual(ops["ScaledDotProductAttention"], 0)
        self.assertEqual(ops["PagedAttentionExtension"], config.num_hidden_layers)
        self.assertEqual(ops["ReadValue"] + ops["Assign"], 0)

        token_type_ids = model.input("token_type_ids").get_node()
        for op in model.get_ops():
            if op.get_type_name() != "PagedAttentionExtension":
                continue
            layer_idx = int(op.input_value(_PA_KEY_CACHE).get_any_name().split(".")[-1])
            with self.subTest(layer=layer_idx):
                # the block attends bidirectionally only if token_type_ids reaches PagedAttention
                token_type_source = op.input_value(_PA_TOKEN_TYPE_IDS).get_node()
                if token_type_source.get_type_name() == "Convert":
                    token_type_source = token_type_source.input_value(0).get_node()
                self.assertIs(token_type_source, token_type_ids)
                expected_window = _SLIDING_WINDOW if config.layer_types[layer_idx] == "sliding_attention" else 0
                self.assertEqual(self._source_constant(op.input_value(_PA_SLIDING_WINDOW)).item(), expected_window)

    def _compile_paged_draft(self, draft_path, config, num_blocks):
        """The draft converted to PagedAttention and compiled on CPU, with zeroed caches of num_blocks blocks."""
        core = ov.Core()
        paged_model = core.read_model(draft_path)
        self._assert_paged_attention_conversion(paged_model, config)
        properties = {"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32"}
        compiled = core.compile_model(paged_model, "CPU", properties)
        request = compiled.create_infer_request()
        cache_names = set()
        for port in compiled.inputs:
            name = port.get_any_name()
            if name.startswith(("key_cache.", "value_cache.")):
                shape = [num_blocks] + [dim.get_length() for dim in list(port.get_partial_shape())[1:]]
                cache = ov.Tensor(port.get_element_type(), shape)
                cache.data[...] = 0
                request.set_tensor(name, cache)
                cache_names.add(name)
        return request, cache_names

    @staticmethod
    def _paged_feed(sequence_inputs, committed_lengths, block_tables):
        """Token-major inputs of one PagedAttention call over the new rows of several sequences.

        past_lens excludes the previous block of each sequence, so the call overwrites it.
        """

        def rows(name):
            return torch.cat([inputs[name][0] for inputs in sequence_inputs])

        num_rows = [inputs["token_type_ids"].shape[1] for inputs in sequence_inputs]
        return {
            "inputs_embeds": rows("inputs_embeds").numpy(),
            "hidden_states": rows("hidden_states")[:, None].numpy(),
            "position_ids": rows("position_ids").numpy(),
            "token_type_ids": rows("token_type_ids")[:, None].numpy(),
            "past_lens": np.array(committed_lengths, dtype=np.int32),
            "subsequence_begins": np.cumsum([0, *num_rows], dtype=np.int32),
            "block_indices": np.concatenate(block_tables).astype(np.int32),
            "block_indices_begins": np.cumsum([0, *map(len, block_tables)], dtype=np.int32),
            "max_context_len": np.array(max(map(sum, zip(committed_lengths, num_rows))), dtype=np.int32),
        }

    @parameterized.expand(VARIANTS)
    def test_per_row_draft_reads_each_input_only_in_its_rows(self, variant):
        model = self._model(variant)
        torch.manual_seed(2)
        inputs = _per_row_inputs(model.config, context_length=5, block_length=4, committed_length=0)
        is_block = inputs["token_type_ids"][..., None] == 1
        ignored_rows_changed = dict(
            inputs,
            inputs_embeds=torch.where(is_block, inputs["inputs_embeds"], torch.randn_like(inputs["inputs_embeds"])),
            hidden_states=torch.where(is_block, torch.randn_like(inputs["hidden_states"]), inputs["hidden_states"]),
        )
        context_changed = dict(
            inputs,
            hidden_states=torch.where(is_block, inputs["hidden_states"], torch.randn_like(inputs["hidden_states"])),
        )
        with torch.no_grad():
            reference = model(**inputs, use_cache=False).last_hidden_state[:, is_block[0, :, 0]]
            ignored_output = model(**ignored_rows_changed, use_cache=False).last_hidden_state[:, is_block[0, :, 0]]
            context_output = model(**context_changed, use_cache=False).last_hidden_state[:, is_block[0, :, 0]]
        torch.testing.assert_close(ignored_output, reference)
        self.assertFalse(torch.allclose(context_output, reference))

    @parameterized.expand(VARIANTS)
    def test_paged_backend_matches_pytorch(self, variant):
        model = self._model(variant)
        config = model.config
        torch.manual_seed(3)
        sequences = []
        for steps in _DRAFT_STEPS:
            committed_length, inputs = 0, []
            for context_length, block_length in steps:
                inputs.append(_per_row_inputs(config, context_length, block_length, committed_length))
                committed_length += context_length
            sequences.append(inputs)
        references = [self._reference_outputs(model, inputs) for inputs in sequences]
        blocks_per_sequence = max(
            math.ceil((sum(context for context, _ in steps) + max(block for _, block in steps)) / _CPU_KV_BLOCK_SIZE)
            for steps in _DRAFT_STEPS
        )
        with TemporaryDirectory() as tmpdirname:
            draft_path = self._export_draft(model, Path(tmpdirname))
            request, cache_names = self._compile_paged_draft(draft_path, config, blocks_per_sequence * len(sequences))
            committed_lengths = [0] * len(sequences)
            for step in range(len(_DRAFT_STEPS[0])):
                step_inputs = [inputs[step] for inputs in sequences]
                block_tables = [
                    sequence * blocks_per_sequence
                    + np.arange(math.ceil((committed + inputs["token_type_ids"].shape[1]) / _CPU_KV_BLOCK_SIZE))
                    for sequence, (committed, inputs) in enumerate(zip(committed_lengths, step_inputs))
                ]
                feed = self._paged_feed(step_inputs, committed_lengths, block_tables)
                self.assertEqual({port.get_any_name() for port in request.model_inputs}, set(feed) | cache_names)
                output = torch.from_numpy(np.array(request.infer(feed)["last_hidden_state"]))
                # every row of every sequence comes out, in row order
                reference = torch.cat([outputs[step] for outputs in references], dim=1)
                torch.testing.assert_close(
                    output.reshape(reference.shape), reference, rtol=1e-4, atol=1e-5, msg=f"step {step}"
                )
                committed_lengths = [
                    committed + steps[step][0] for committed, steps in zip(committed_lengths, _DRAFT_STEPS)
                ]
