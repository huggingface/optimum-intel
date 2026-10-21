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
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from utils_tests import F32_CONFIG, MODEL_NAMES

from optimum.exporters.openvino import export_from_model, main_export
from optimum.exporters.openvino.__main__ import load_native_draft_model
from optimum.exporters.openvino.convert import _get_submodels_and_export_configs, export_models
from optimum.exporters.openvino.dflash_utils import (
    MARKOV_HEAD_DEFAULT_TOP_K,
    DSparkConfig,
    parse_and_validate_dflash_config,
    parse_and_validate_dspark_config,
)
from optimum.exporters.openvino.model_configs import Qwen3OpenVINOConfig
from optimum.exporters.openvino.model_patcher import (
    DSparkVanillaMarkovHead,
    DSparkVanillaMarkovHeadForExport,
    Qwen3DFlash2ForCausalLM,
    Qwen3DSparkForCausalLM,
    _dflash_block_tap_masks,
)
from optimum.intel.openvino import OVAssistantForCausalLM, OVConfig, OVModelForVisualCausalLM
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
                input_names,
                {"inputs_embeds", "hidden_states", "position_ids", "attention_mask", "token_type_ids", "beam_idx"},
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

            model = OVModelForVisualCausalLM.from_pretrained(target_dir, ov_config=F32_CONFIG)
            assistant = OVAssistantForCausalLM.from_pretrained(draft_dir, ov_config=F32_CONFIG)
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

    @parameterized.expand(
        (
            ("qwen3_dflash", {"candidate_position_offset": "1"}),
            # DeepSpec publishes its DFlash drafts as DSpark checkpoints with markov_rank=0.
            ("qwen3_deepspec_dflash", {"block_size": "7", "candidate_position_offset": "0"}),
        )
    )
    def test_draft_without_markov_head_exports_a_plain_dflash_package(self, model_name, expected_rt_info):
        dflash_config = parse_and_validate_dflash_config(AutoConfig.from_pretrained(MODEL_NAMES[model_name])).values
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            main_export(MODEL_NAMES[model_name], output=output_dir, task="text-generation", trust_remote_code=True)
            self.assertEqual([path.name for path in output_dir.glob("openvino_*model.xml")], ["openvino_model.xml"])
            draft_ir = ov.Core().read_model(output_dir / "openvino_model.xml")
            self.assertEqual(draft_ir.get_rt_info()["dflash_mode"].value, "True")
            draft_rt_info = draft_ir.get_rt_info()["dflash"]
            self.assertEqual(
                {name: draft_rt_info[name].value for name in draft_rt_info},
                {
                    "mask_token_id": str(dflash_config["mask_token_id"]),
                    "target_layer_ids": ",".join(map(str, dflash_config["target_layer_ids"])),
                    **expected_rt_info,
                },
            )


class DSparkExportTest(unittest.TestCase):
    TOP_K = 4

    @staticmethod
    def _model(sample_from_anchor=True):
        if is_transformers_version("<", "5"):
            raise unittest.SkipTest("DSpark drafts require transformers>=5")
        return Qwen3DSparkForCausalLM.from_pretrained(
            MODEL_NAMES["minicpmv4_7_dspark"], sample_from_anchor=sample_from_anchor
        ).eval()

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

    def test_candidate_rows_follow_the_anchor_layout(self):
        # The backbone emits every row in both layouts; the layout only moves the candidate rows, which the
        # runtime finds through dflash/candidate_position_offset.
        torch.manual_seed(5)
        for sample_from_anchor, expected_offset in ((True, 0), (False, 1)):
            with self.subTest(sample_from_anchor=sample_from_anchor):
                model = self._model(sample_from_anchor)
                inputs = _per_row_inputs(model.config, context_length=3, block_length=4, committed_length=0)
                with torch.no_grad():
                    last_hidden_state = model(**inputs, use_cache=False).last_hidden_state
                self.assertEqual(last_hidden_state.shape[1], 7)
                _, parts, _ = _get_submodels_and_export_configs(
                    model,
                    task="text-generation",
                    monolith=False,
                    custom_export_configs={},
                    custom_architecture=False,
                    _variant="default",
                    library_name="transformers",
                    stateful=False,
                )
                self.assertEqual(parts["model"][1].candidate_position_offset, expected_offset)

    def test_export_package_and_markov_head_parity(self):
        model = self._model()
        config = model.config
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            parts = _export_draft(model, output_dir)
            self.assertEqual(list(parts), ["model", "markov_head_model"])

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
                "interface": "markov_head_v1",
                "top_k": str(MARKOV_HEAD_DEFAULT_TOP_K),
                "markov_rank": str(config.markov_rank),
                "markov_head_type": "vanilla",
                "vocab_size": str(config.vocab_size),
            }
            self.assertEqual({name: markov_rt_info[name].value for name in markov_rt_info}, expected_rt_info)
            self.assertFalse(markov_head_ir.has_rt_info(["dflash_mode"]))
            self.assertFalse(markov_head_ir.has_rt_info(["hidden_states_decoder_layers"]))
            self.assertEqual(
                {item.any_name: item.element_type for item in markov_head_ir.inputs},
                {
                    "prev_token_ids": ov.Type.i64,
                    "draft_logits": ov.Type.f32,
                    "anchor_token_ids": ov.Type.i64,
                    "subsequence_begins": ov.Type.i64,
                },
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

            compiled_markov_head = core.compile_model(markov_head_ir, "CPU", F32_CONFIG)
            for prev_token_ids, block_lengths in ((torch.tensor([7]), [1]), (torch.tensor([0, 5, 9]), [4, 2])):
                with self.subTest(num_prev_tokens=prev_token_ids.numel(), block_lengths=block_lengths):
                    markov_head_inputs = {
                        "prev_token_ids": prev_token_ids,
                        **_packed_blocks(block_lengths, config.vocab_size),
                    }
                    with torch.no_grad():
                        reference = parts["markov_head_model"][0](**markov_head_inputs)
                    ov_outputs = compiled_markov_head(
                        {name: value.numpy() for name, value in markov_head_inputs.items()}
                    )
                    for name in ("markov_bias", "edge_scores"):
                        ov_output = torch.from_numpy(np.array(ov_outputs[compiled_markov_head.output(name)]))
                        torch.testing.assert_close(ov_output, reference[name], rtol=1e-4, atol=1e-4)
                    ov_candidate_ids = np.array(ov_outputs[compiled_markov_head.output("candidate_ids")])
                    self.assertTrue(torch.equal(torch.from_numpy(ov_candidate_ids), reference.candidate_ids))

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

    def test_markov_head_model_splits_into_dense_step_and_packed_top_k_lattice(self):
        model = self._model()
        config = model.config
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            markov_head = _export_draft(model, output_dir)["markov_head_model"][0]
            core = ov.Core()
            markov_head_ir = core.read_model(output_dir / "openvino_markov_head_model.xml")

            markov_step_ir = self._keep_computation(markov_head_ir, ["prev_token_ids"], ["markov_bias"])
            self.assertFalse(any(op.get_type_name() == "TopK" for op in markov_step_ir.get_ops()))
            prev_token_ids = torch.tensor([0, 5, config.vocab_size - 1])
            compiled_markov_step = core.compile_model(markov_step_ir, "CPU", F32_CONFIG)
            ov_bias = compiled_markov_step({"prev_token_ids": prev_token_ids.numpy()})[0]
            with torch.no_grad():
                reference_bias = markov_head.markov_bias(prev_token_ids)
            torch.testing.assert_close(torch.from_numpy(np.array(ov_bias)), reference_bias, rtol=1e-4, atol=1e-4)

            # Blocks of different lengths in one call match the lattices of the blocks scored one by one.
            block_lengths = [3, 1, 4]
            lattice_inputs = _packed_blocks(block_lengths, config.vocab_size)
            markov_w1 = markov_head.markov_head.markov_w1.weight.detach()
            markov_w2 = markov_head.markov_head.markov_w2.weight.detach()
            for top_k in (None, self.TOP_K):
                with self.subTest(top_k=top_k):
                    lattice_ir = self._keep_computation(
                        markov_head_ir, list(lattice_inputs), ["edge_scores", "candidate_ids"], top_k
                    )
                    expected_top_k = top_k or MARKOV_HEAD_DEFAULT_TOP_K
                    self.assertEqual(lattice_ir.output("candidate_ids").partial_shape[1], expected_top_k)
                    compiled_lattice = core.compile_model(lattice_ir, "CPU", F32_CONFIG)
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
            compiled_markov_head = core.compile_model(markov_head_ir, "CPU", F32_CONFIG)
            lattice_inputs = _packed_blocks([3], config.vocab_size)
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
    @staticmethod
    def _config():
        return Qwen3Config.from_pretrained(MODEL_NAMES["qwen3_dflash2"])

    @staticmethod
    def _model():
        return Qwen3DFlash2ForCausalLM.from_pretrained(MODEL_NAMES["qwen3_dflash2"]).eval()

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
    def _reference_edges(model, draft_logits, draft_hidden_states, anchor_token_id, top_k):
        """Top-K lattice of one draft block: the anchor precedes its first row, and the hidden state of every row
        gates its transitions."""
        dflash_config = model.config.dflash_config
        selector = model.candidate_selector
        logits, candidate_ids = draft_logits.topk(top_k, dim=-1)
        softcap = dflash_config["final_logit_softcapping"]
        unary_scores = torch.tanh(logits * dflash_config["output_multiplier"] / softcap) * softcap
        predecessor_ids = torch.cat([torch.full((1, top_k), int(anchor_token_id)), candidate_ids[:-1]])
        transition_scores = torch.einsum(
            "tir,tr,tjr->tij",
            selector.predecessor_codebook[predecessor_ids],
            selector.hidden_projection(draft_hidden_states),
            selector.successor_codebook[candidate_ids],
        )
        return transition_scores + unary_scores.unsqueeze(-2), candidate_ids

    def test_checkpoint_keys(self):
        # The tiny draft has the tensor layout of the published DFlash-2 checkpoints.
        _, loading_info = Qwen3DFlash2ForCausalLM.from_pretrained(
            MODEL_NAMES["qwen3_dflash2"], output_loading_info=True
        )
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])

    def test_config_normalization(self):
        config = self._config()
        dflash = parse_and_validate_dflash_config(config)
        self.assertEqual(dflash.version, 2)
        # block_size is not a DFlash-2 field, so it is only read from dflash_config.
        flat_config = deepcopy(config)
        flat_config.dflash_config = {"block_size": config.dflash_config["block_size"]}
        for name, value in config.dflash_config.items():
            if name != "block_size":
                setattr(flat_config, name, value)
        self.assertEqual(parse_and_validate_dflash_config(flat_config).values, dflash.values)

        zero_softcap_config = deepcopy(config)
        zero_softcap_config.dflash_config["final_logit_softcapping"] = 0.0
        self.assertEqual(parse_and_validate_dflash_config(zero_softcap_config).values["final_logit_softcapping"], 0.0)

        v1_config = deepcopy(config)
        v1_config.architectures = ["DFlashDraftModel"]
        self.assertEqual(parse_and_validate_dflash_config(v1_config).version, 1)

    # The tiny draft has 2 target layers and a vocabulary of 151936 tokens.
    @parameterized.expand(
        (
            ("negative_softcap", {}, {"final_logit_softcapping": -1.0}, "finite non-negative"),
            ("zero_multiplier", {}, {"output_multiplier": 0.0}, "finite positive"),
            ("missing_selector_rank", {}, {"selector_rank": None}, "selector_rank"),
            ("conv_group_size", {}, {"conv_group_size": 7}, "positive divisor"),
            ("target_layer_id", {}, {"target_layer_ids": [2]}, "smaller than num_target_layers"),
            ("duplicate_target_layer_ids", {}, {"target_layer_ids": [0, 0]}, "unique non-negative"),
            ("mask_token_id", {}, {"mask_token_id": 151936}, "mask_token_id"),
            ("flat_conflict", {"selector_rank": 9}, {}, "conflicting nested and flat"),
            ("model_type", {"model_type": "muse_glimmer"}, {}, "only Qwen3"),
            ("architectures", {"architectures": ["Qwen3ForCausalLM", "DFlash2DraftModel"]}, {}, "architectures"),
        )
    )
    def test_invalid_config_is_rejected(self, _, attributes, dflash_fields, error):
        config = self._config()
        for name, value in attributes.items():
            setattr(config, name, value)
        for name, value in dflash_fields.items():
            if value is None:
                del config.dflash_config[name]
            else:
                config.dflash_config[name] = value
        with self.assertRaisesRegex(ValueError, error):
            parse_and_validate_dflash_config(config)

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
        model = self._model()
        convolution = model.layers[0].attention_conv
        context_length = 3
        for block_length in (2, 4, 7):
            with self.subTest(block_length=block_length):
                # two sequences of rows [context ; block]; the convolution sees only their block rows, packed back
                # to back, so a tap must not reach from the second block into the first
                token_type_ids = torch.tensor([[0] * context_length + [1] * block_length] * 2)
                block_rows = torch.nonzero(token_type_ids.flatten() == 1).squeeze(-1)
                block_tap_masks = _dflash_block_tap_masks(token_type_ids, block_rows, convolution.kernel_size)
                hidden_states = torch.randn(2, block_length, model.config.hidden_size)
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

    def test_export_package_and_markov_head_computations(self):
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

            draft_rt_info = draft_ir.get_rt_info()["dflash"]
            self.assertEqual(
                {name: draft_rt_info[name].value for name in draft_rt_info},
                {
                    "version": "2",
                    "mask_token_id": str(dflash_config["mask_token_id"]),
                    "target_layer_ids": ",".join(map(str, dflash_config["target_layer_ids"])),
                    "block_size": str(dflash_config["block_size"]),
                    "candidate_position_offset": "1",
                    "gpu_min_activations_scale_factor": "32.0",
                    "input_embedding_scale": str(dflash_config["input_embedding_scale"]),
                    "output_multiplier": str(dflash_config["output_multiplier"]),
                    "final_logit_softcapping": str(dflash_config["final_logit_softcapping"]),
                    "markov_rank": str(dflash_config["selector_rank"]),
                    "markov_head_type": "dflash2",
                },
            )
            markov_rt_info = markov_head_ir.get_rt_info()["dflash"]
            self.assertEqual(
                {name: markov_rt_info[name].value for name in markov_rt_info},
                {
                    "interface": "markov_head_v1",
                    "top_k": str(top_k),
                    "markov_rank": str(dflash_config["selector_rank"]),
                    "markov_head_type": "dflash2",
                    "vocab_size": str(config.vocab_size),
                },
            )
            self.assertFalse(markov_head_ir.has_rt_info(["dflash_mode"]))
            self.assertFalse(draft_ir.has_rt_info(["hidden_states_decoder_layers"]))
            self.assertFalse(markov_head_ir.has_rt_info(["hidden_states_decoder_layers"]))

            self.assertEqual({item.any_name for item in draft_ir.outputs}, {"last_hidden_state"})
            self.assertEqual(
                [(item.any_name, item.element_type) for item in markov_head_ir.inputs],
                [
                    ("draft_logits", ov.Type.f32),
                    ("draft_hidden_states", ov.Type.f32),
                    ("anchor_token_ids", ov.Type.i64),
                    ("subsequence_begins", ov.Type.i64),
                ],
            )
            self.assertEqual(
                {item.any_name: item.element_type for item in markov_head_ir.outputs},
                {"edge_scores": ov.Type.f32, "candidate_ids": ov.Type.i64},
            )
            self.assertEqual(len([op for op in markov_head_ir.get_ops() if op.get_type_name() == "TopK"]), 1)

            def codebooks(model_ir):
                codebook_shape = [config.vocab_size, dflash_config["selector_rank"]]
                return [
                    op
                    for op in model_ir.get_ops()
                    if op.get_type_name() == "Constant" and list(op.get_output_shape(0)) == codebook_shape
                ]

            self.assertEqual(codebooks(draft_ir), [])
            self.assertGreaterEqual(len(codebooks(markov_head_ir)), 2)

            # Blocks of different lengths in one call match the lattices of the blocks scored one by one.
            lattice_inputs = _packed_blocks([3, 1, 4], config.vocab_size)
            lattice_inputs["draft_hidden_states"] = torch.randn(
                len(lattice_inputs["draft_logits"]), config.hidden_size
            )
            compiled_markov_head = core.compile_model(markov_head_ir, "CPU", F32_CONFIG)
            ov_outputs = compiled_markov_head({name: value.numpy() for name, value in lattice_inputs.items()})
            ov_candidate_ids = torch.from_numpy(np.array(ov_outputs[compiled_markov_head.output("candidate_ids")]))
            ov_edges = torch.from_numpy(np.array(ov_outputs[compiled_markov_head.output("edge_scores")]))
            begins = lattice_inputs["subsequence_begins"].tolist()
            for block, (begin, end) in enumerate(zip(begins[:-1], begins[1:])):
                with torch.no_grad():
                    expected_edges, expected_ids = self._reference_edges(
                        model,
                        lattice_inputs["draft_logits"][begin:end],
                        lattice_inputs["draft_hidden_states"][begin:end],
                        lattice_inputs["anchor_token_ids"][block],
                        top_k,
                    )
                self.assertTrue(torch.equal(ov_candidate_ids[begin:end], expected_ids))
                torch.testing.assert_close(ov_edges[begin:end], expected_edges, rtol=1e-4, atol=1e-4)

    def test_bf16_markov_head_model_keeps_checkpoint_weights_and_fp32_outputs(self):
        model = self._model().to(torch.bfloat16)
        config = model.config
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            _export_draft(model, output_dir, patch_16bit_model=True)
            core = ov.Core()
            draft_ir = core.read_model(output_dir / "openvino_model.xml")
            markov_head_ir = core.read_model(output_dir / "openvino_markov_head_model.xml")
            self.assertEqual(markov_head_ir.input("draft_logits").element_type, ov.Type.f32)
            self.assertEqual(markov_head_ir.input("draft_hidden_states").element_type, ov.Type.f32)
            self.assertEqual(markov_head_ir.output("edge_scores").element_type, ov.Type.f32)
            codebook_shape = [config.vocab_size, config.dflash_config["selector_rank"]]
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
# (context delta, block) rows per draft step: the first step spans two cache blocks, and every step
# reaches past the sliding window
_DRAFT_STEPS = ((34, 4), (3, 4), (1, 4), (2, 4))
# PagedAttentionExtension input ports
_PA_KEY_CACHE, _PA_SLIDING_WINDOW, _PA_TOKEN_TYPE_IDS = 3, 10, 25


@unittest.skipUnless(is_transformers_version(">=", "4.57"), "DFlash-2 requires Transformers 4.57 or newer")
class DFlashPagedAttentionTest(unittest.TestCase):
    """The draft IR of every DFlash variant runs on PagedAttention, the DFlash backend of OpenVINO GenAI."""

    # A DSpark draft has the DFlash backbone. The DFlash-2 draft has grouped-query attention, as the released drafts,
    # so the converted graph covers the K/V head expansion, and base kernels that reach the previous rows.
    VARIANTS = (("qwen3_dflash",), ("qwen3_dflash2",))

    @staticmethod
    def _model(model_name):
        model_path = MODEL_NAMES[model_name]
        return load_native_draft_model(
            AutoConfig.from_pretrained(model_path).architectures[0],
            model_path,
            layer_types=["sliding_attention", "full_attention"],
            use_sliding_window=True,
            sliding_window=_SLIDING_WINDOW,
        ).eval()

    @staticmethod
    def _draft_steps(config):
        torch.manual_seed(1)
        steps, committed_length = [], 0
        for context_length, block_length in _DRAFT_STEPS:
            steps.append(_per_row_inputs(config, context_length, block_length, committed_length))
            committed_length += context_length
        return steps

    @staticmethod
    def _reference_outputs(model, steps):
        """Outputs of consecutive draft steps, dropping each block from the cache afterwards as past_lens does."""
        cache = DynamicCache()
        outputs = []
        with torch.no_grad():
            for inputs in steps:
                outputs.append(model(**inputs, past_key_values=cache, use_cache=True).last_hidden_state)
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
    def test_per_row_draft_reads_each_input_only_in_its_rows(self, model_name):
        model = self._model(model_name)
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
    def test_paged_backend_matches_pytorch(self, model_name):
        model = self._model(model_name)
        config = model.config
        steps = self._draft_steps(config)
        references = self._reference_outputs(model, steps)
        max_length = sum(context for context, _ in _DRAFT_STEPS) + max(block for _, block in _DRAFT_STEPS)
        with TemporaryDirectory() as tmpdirname:
            draft_path = self._export_draft(model, Path(tmpdirname))
            request, cache_names = self._compile_paged_draft(
                draft_path, config, math.ceil(max_length / _CPU_KV_BLOCK_SIZE)
            )
            committed_length = 0
            for step, ((context_length, block_length), inputs, reference) in enumerate(
                zip(_DRAFT_STEPS, steps, references)
            ):
                used_blocks = math.ceil((committed_length + context_length + block_length) / _CPU_KV_BLOCK_SIZE)
                feed = self._paged_feed([inputs], [committed_length], [np.arange(used_blocks)])
                self.assertEqual({port.get_any_name() for port in request.model_inputs}, set(feed) | cache_names)
                output = torch.from_numpy(np.array(request.infer(feed)["last_hidden_state"]))
                # every row comes out, one per token
                torch.testing.assert_close(
                    output.reshape(reference.shape), reference, rtol=1e-4, atol=1e-5, msg=f"step {step}"
                )
                committed_length += context_length

    @parameterized.expand(VARIANTS)
    def test_paged_backend_drafts_several_sequences_in_one_call(self, model_name):
        model = self._model(model_name)
        config = model.config
        # (context delta, block) rows of two sequences per call; only the context rows in front of each
        # block keep the blocks of different sequences apart
        sequence_steps = (((34, 4), (3, 4), (1, 4)), ((5, 4), (1, 4), (2, 4)))
        torch.manual_seed(3)
        sequences = []
        for steps in sequence_steps:
            committed_length, inputs = 0, []
            for context_length, block_length in steps:
                inputs.append(_per_row_inputs(config, context_length, block_length, committed_length))
                committed_length += context_length
            sequences.append(inputs)
        references = [self._reference_outputs(model, inputs) for inputs in sequences]
        blocks_per_sequence = max(
            math.ceil((sum(context for context, _ in steps) + max(block for _, block in steps)) / _CPU_KV_BLOCK_SIZE)
            for steps in sequence_steps
        )
        with TemporaryDirectory() as tmpdirname:
            draft_path = self._export_draft(model, Path(tmpdirname))
            request, _ = self._compile_paged_draft(draft_path, config, blocks_per_sequence * len(sequences))
            committed_lengths = [0] * len(sequences)
            for step in range(len(sequence_steps[0])):
                step_inputs = [inputs[step] for inputs in sequences]
                block_tables = [
                    sequence * blocks_per_sequence
                    + np.arange(math.ceil((committed + inputs["token_type_ids"].shape[1]) / _CPU_KV_BLOCK_SIZE))
                    for sequence, (committed, inputs) in enumerate(zip(committed_lengths, step_inputs))
                ]
                feed = self._paged_feed(step_inputs, committed_lengths, block_tables)
                output = torch.from_numpy(np.array(request.infer(feed)["last_hidden_state"]))
                # the rows of all sequences, concatenated in row order
                reference = torch.cat([outputs[step] for outputs in references], dim=1)
                torch.testing.assert_close(
                    output.reshape(reference.shape), reference, rtol=1e-4, atol=1e-5, msg=f"step {step}"
                )
                committed_lengths = [
                    committed + steps[step][0] for committed, steps in zip(committed_lengths, sequence_steps)
                ]
