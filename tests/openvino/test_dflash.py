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
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from utils_tests import MODEL_NAMES, get_dspark_model_path

from optimum.exporters.openvino import export_from_model
from optimum.exporters.openvino.__main__ import main_export
from optimum.exporters.openvino.convert import _get_submodels_and_export_configs, export_models
from optimum.exporters.openvino.dflash_utils import (
    DSparkConfig,
    parse_and_validate_dspark_config,
    validate_dspark_top_k,
)
from optimum.exporters.openvino.model_configs import Qwen3OpenVINOConfig
from optimum.exporters.openvino.model_patcher import (
    DSparkMarkovCandidateScorerForExport,
    DSparkMarkovStepForExport,
    DSparkVanillaMarkovHead,
    Qwen3DSparkForCausalLM,
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


class DSparkExportTest(unittest.TestCase):
    TOP_K = 4

    @staticmethod
    def _model(sample_from_anchor=True):
        return Qwen3DSparkForCausalLM.from_pretrained(get_dspark_model_path(sample_from_anchor)).eval()

    @staticmethod
    def _dspark_config(**fields):
        config = Qwen3Config(vocab_size=32)
        config.architectures = ["Qwen3DSparkModel"]
        config.target_layer_ids = [0, 1]
        for name, value in fields.items():
            setattr(config, name, value)
        return config

    @staticmethod
    def _backbone_inputs(config, block_length, context_length=3):
        return {
            "inputs_embeds": torch.randn(1, block_length, config.hidden_size),
            "hidden_states": torch.randn(1, context_length, config.hidden_size * len(config.target_layer_ids)),
            "position_ids": torch.arange(context_length + block_length).reshape(1, -1),
            "attention_mask": torch.ones(1, context_length + block_length),
        }

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
    def _markov_weights(model, config, element_type=None):
        shapes = ([config.vocab_size, config.markov_rank], [config.markov_rank, config.vocab_size])
        return [
            op
            for op in model.get_ops()
            if op.get_type_name() == "Constant"
            and list(op.get_output_shape(0)) in shapes
            and (element_type is None or op.get_output_element_type(0) == element_type)
        ]

    @staticmethod
    def _export(model, output_dir, patch_16bit_model=False, dspark_top_k=None):
        _, parts, stateful = _get_submodels_and_export_configs(
            model,
            task="text-generation",
            monolith=False,
            custom_export_configs={},
            custom_architecture=False,
            _variant="default",
            library_name="transformers",
            stateful=False,
            dspark_top_k=dspark_top_k,
        )
        export_models(
            parts,
            output_dir,
            output_names=[f"openvino_{name}.xml" for name in parts],
            input_shapes={"batch_size": 1, "sequence_length": 3},
            stateful=stateful,
            patch_16bit_model=patch_16bit_model,
            library_name="transformers",
        )
        return parts

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
        self.assertEqual(validate_dspark_top_k(config, self.TOP_K), self.TOP_K)
        self.assertEqual(validate_dspark_top_k(config, config.vocab_size), config.vocab_size)
        for top_k in (0, config.vocab_size + 1, True, 2.0):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                validate_dspark_top_k(config, top_k)
        with self.assertRaisesRegex(ValueError, "markov_rank > 0"):
            validate_dspark_top_k(self._dspark_config(), self.TOP_K)
        for non_dspark in (None, Qwen3Config(vocab_size=32)):
            with self.subTest(config=non_dspark), self.assertRaisesRegex(ValueError, "only supported for DSpark"):
                validate_dspark_top_k(non_dspark, self.TOP_K)

        self.assertEqual(Qwen3OpenVINOConfig(config).candidate_position_offset, 0)
        masked_anchor_config = self._dspark_config(markov_rank=8, sample_from_anchor=False)
        self.assertEqual(Qwen3OpenVINOConfig(masked_anchor_config).candidate_position_offset, 1)
        with self.assertRaisesRegex(ValueError, "not supported"):
            Qwen3OpenVINOConfig(self._dspark_config(markov_rank=8, markov_head_type="rnn"))

    def test_top_k_option_requires_a_dspark_markov_draft(self):
        with TemporaryDirectory() as tmpdirname:
            with self.assertRaisesRegex(ValueError, "only supported for DSpark"):
                main_export(MODEL_NAMES["qwen3"], output=tmpdirname, task="text-generation", dspark_top_k=self.TOP_K)
            with self.assertRaisesRegex(ValueError, "markov_rank > 0"):
                main_export(
                    MODEL_NAMES["qwen3_deepspec_dflash"],
                    output=tmpdirname,
                    task="text-generation",
                    dspark_top_k=self.TOP_K,
                )

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
                trust_remote_code=True,
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

    def test_backbone_drops_anchor_row_only_for_masked_anchor_layout(self):
        torch.manual_seed(5)
        model = self._model()
        inputs = self._backbone_inputs(model.config, block_length=4)
        with torch.no_grad():
            anchor_rows = model(**inputs, use_cache=False).last_hidden_state
            model.sample_from_anchor = False
            masked_anchor_rows = model(**inputs, use_cache=False).last_hidden_state
        self.assertEqual(anchor_rows.shape[1], 4)
        torch.testing.assert_close(masked_anchor_rows, anchor_rows[:, 1:])

        masked_anchor_model = self._model(sample_from_anchor=False)
        self.assertFalse(masked_anchor_model.sample_from_anchor)
        _, parts, _ = _get_submodels_and_export_configs(
            masked_anchor_model,
            task="text-generation",
            monolith=False,
            custom_export_configs={},
            custom_architecture=False,
            _variant="default",
            library_name="transformers",
            stateful=False,
        )
        self.assertEqual(parts["model"][1].candidate_position_offset, 1)

    def test_dense_export_package_and_markov_step_parity(self):
        model = self._model()
        config = model.config
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            parts = self._export(model, output_dir)
            self.assertEqual(list(parts), ["model", "markov_step_model"])
            self.assertFalse((output_dir / "openvino_candidate_scorer_model.xml").exists())

            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            markov_step_ir = core.read_model(output_dir / "openvino_markov_step_model.xml")
            draft_rt_info = draft_ir.get_rt_info()["dflash"]
            self.assertEqual(draft_rt_info["candidate_position_offset"].value, "0")
            self.assertEqual(draft_rt_info["markov_rank"].value, str(config.markov_rank))
            self.assertEqual(draft_rt_info["markov_head_type"].value, "vanilla")
            self.assertEqual(self._markov_weights(draft_ir, config), [])

            markov_rt_info = markov_step_ir.get_rt_info()["dflash"]
            expected_rt_info = {
                "interface": "markov_step_v1",
                "markov_rank": str(config.markov_rank),
                "markov_head_type": "vanilla",
                "vocab_size": str(config.vocab_size),
            }
            self.assertEqual({name: markov_rt_info[name].value for name in markov_rt_info}, expected_rt_info)
            self.assertFalse(markov_step_ir.has_rt_info(["dflash_mode"]))
            self.assertFalse(markov_step_ir.has_rt_info(["hidden_states_decoder_layers"]))
            self.assertEqual(
                {item.any_name: item.element_type for item in markov_step_ir.inputs},
                {"prev_token_ids": ov.Type.i64},
            )
            self.assertEqual(
                {item.any_name: item.element_type for item in markov_step_ir.outputs},
                {"markov_bias": ov.Type.f32},
            )
            self.assertGreaterEqual(len(self._markov_weights(markov_step_ir, config)), 2)

            compiled_markov_step = core.compile_model(markov_step_ir, "CPU")
            for prev_token_ids in (torch.tensor([7]), torch.tensor([0, 5, config.vocab_size - 1])):
                with self.subTest(batch_size=prev_token_ids.numel()):
                    with torch.no_grad():
                        reference = model.markov_head(prev_token_ids)
                    ov_bias = compiled_markov_step({"prev_token_ids": prev_token_ids.numpy()})[0]
                    torch.testing.assert_close(torch.from_numpy(np.array(ov_bias)), reference, rtol=1e-4, atol=1e-4)

    def test_top_k_export_replaces_markov_step_with_candidate_scorer(self):
        model = self._model()
        config = model.config
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            parts = self._export(model, output_dir, dspark_top_k=self.TOP_K)
            self.assertEqual(list(parts), ["model", "candidate_scorer_model"])
            self.assertFalse((output_dir / "openvino_markov_step_model.xml").exists())

            core = ov.Core()
            candidate_scorer_ir = core.read_model(output_dir / "openvino_candidate_scorer_model.xml")
            candidate_scorer_rt_info = candidate_scorer_ir.get_rt_info()["dflash"]
            expected_rt_info = {
                "interface": "topk_lattice_v1",
                "top_k": str(self.TOP_K),
                "markov_rank": str(config.markov_rank),
                "markov_head_type": "vanilla",
                "vocab_size": str(config.vocab_size),
            }
            self.assertEqual(
                {name: candidate_scorer_rt_info[name].value for name in candidate_scorer_rt_info}, expected_rt_info
            )
            self.assertFalse(candidate_scorer_ir.has_rt_info(["dflash_mode"]))
            self.assertEqual(
                {item.any_name: item.element_type for item in candidate_scorer_ir.inputs},
                {"draft_logits": ov.Type.f32, "anchor_token_ids": ov.Type.i64},
            )
            self.assertEqual({item.any_name for item in candidate_scorer_ir.outputs}, {"edge_scores", "candidate_ids"})
            self.assertEqual(candidate_scorer_ir.output("edge_scores").element_type, ov.Type.f32)
            self.assertEqual(candidate_scorer_ir.output("candidate_ids").element_type, ov.Type.i64)
            self.assertTrue(any(op.get_type_name() == "TopK" for op in candidate_scorer_ir.get_ops()))

            compiled_candidate_scorer = core.compile_model(candidate_scorer_ir, "CPU")
            for num_draft_tokens in (1, 4):
                with self.subTest(num_draft_tokens=num_draft_tokens):
                    candidate_scorer_inputs = {
                        "draft_logits": torch.randn(1, num_draft_tokens, config.vocab_size) * 4,
                        "anchor_token_ids": torch.tensor([11]),
                    }
                    with torch.no_grad():
                        reference = parts["candidate_scorer_model"][0](**candidate_scorer_inputs)
                    ov_outputs = compiled_candidate_scorer(
                        {name: value.numpy() for name, value in candidate_scorer_inputs.items()}
                    )
                    ov_candidate_ids = np.array(ov_outputs[compiled_candidate_scorer.output("candidate_ids")])
                    ov_edges = np.array(ov_outputs[compiled_candidate_scorer.output("edge_scores")])
                    self.assertTrue(torch.equal(torch.from_numpy(ov_candidate_ids), reference.candidate_ids))
                    torch.testing.assert_close(torch.from_numpy(ov_edges), reference.edge_scores, rtol=1e-4, atol=1e-4)

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

            # Dense runtime walk: one step-graph call per draft position on the previously drafted token.
            markov_step = DSparkMarkovStepForExport(markov_head, config)
            previous_token_ids = anchor_token_ids
            dense_tokens = []
            for position in range(num_draft_tokens):
                step_logits = base_logits[:, position] + markov_step(previous_token_ids).markov_bias
                previous_token_ids = step_logits.argmax(dim=-1)
                dense_tokens.append(previous_token_ids)
            self.assertTrue(torch.equal(torch.stack(dense_tokens, dim=1), reference_tokens))

            for top_k in (vocab_size, 7):
                with self.subTest(top_k=top_k):
                    candidate_scorer = DSparkMarkovCandidateScorerForExport(markov_head, config, top_k)
                    lattice = candidate_scorer(base_logits, anchor_token_ids)
                    edge_scores, candidate_ids = lattice.edge_scores[0], lattice.candidate_ids[0]
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

    def test_bf16_markov_head_models_keep_checkpoint_weights_and_fp32_outputs(self):
        model = self._model().to(torch.bfloat16)
        config = model.config
        top_k_model = self._model().to(torch.bfloat16)
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            self._export(model, output_dir / "dense", patch_16bit_model=True)
            self._export(top_k_model, output_dir / "top_k", patch_16bit_model=True, dspark_top_k=self.TOP_K)
            core = ov.Core()
            markov_step_ir = core.read_model(output_dir / "dense" / "openvino_markov_step_model.xml")
            candidate_scorer_ir = core.read_model(output_dir / "top_k" / "openvino_candidate_scorer_model.xml")
            self.assertEqual(markov_step_ir.output("markov_bias").element_type, ov.Type.f32)
            self.assertEqual(candidate_scorer_ir.input("draft_logits").element_type, ov.Type.f32)
            self.assertEqual(candidate_scorer_ir.output("edge_scores").element_type, ov.Type.f32)
            for markov_model in (markov_step_ir, candidate_scorer_ir):
                self.assertEqual(self._markov_weights(markov_model, config, ov.Type.f32), [])
                self.assertGreaterEqual(len(self._markov_weights(markov_model, config, ov.Type.bf16)), 2)

            markov_w1 = model.markov_head.markov_w1.weight.float()
            markov_w2 = model.markov_head.markov_w2.weight.float()
            compiled_markov_step = core.compile_model(markov_step_ir, "CPU")
            ov_bias = np.array(compiled_markov_step({"prev_token_ids": np.array([3], dtype=np.int64)})[0])
            torch.testing.assert_close(torch.from_numpy(ov_bias), markov_w1[[3]] @ markov_w2.T, rtol=1e-4, atol=1e-4)

            compiled_candidate_scorer = core.compile_model(candidate_scorer_ir, "CPU")
            draft_logits = torch.randn(1, 3, config.vocab_size) * 4
            anchor_token_ids = torch.tensor([11])
            ov_outputs = compiled_candidate_scorer(
                {"draft_logits": draft_logits.numpy(), "anchor_token_ids": anchor_token_ids.numpy()}
            )
            candidate_ids = torch.from_numpy(np.array(ov_outputs[compiled_candidate_scorer.output("candidate_ids")]))
            self.assertTrue(torch.equal(candidate_ids, draft_logits.topk(self.TOP_K, dim=-1).indices))
            predecessor_ids = torch.cat([anchor_token_ids.expand(1, 1, self.TOP_K), candidate_ids[:, :-1]], dim=1)
            expected_edges = markov_w1[predecessor_ids] @ markov_w2[candidate_ids].transpose(-1, -2)
            expected_edges += draft_logits.gather(-1, candidate_ids).unsqueeze(-2)
            ov_edges = torch.from_numpy(np.array(ov_outputs[compiled_candidate_scorer.output("edge_scores")]))
            torch.testing.assert_close(ov_edges, expected_edges, rtol=1e-4, atol=1e-4)
