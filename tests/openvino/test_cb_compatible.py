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

import unittest
from collections import Counter
from pathlib import Path

import openvino as ov
from openvino._offline_transformations import paged_attention_transformation
from parameterized import parameterized
from transformers import AutoConfig
from utils_tests import MODEL_NAMES, OPENVINO_DEVICE, REMOTE_CODE_MODELS, get_supported_model_for_library

from optimum.exporters.openvino import main_export
from optimum.intel.openvino.utils import TemporaryDirectory
from optimum.intel.utils.import_utils import is_qwen_tts_available


# The submodels of each architecture that continuous batching can serve, i.e. whose stateful
# ScaledDotProductAttention blocks the SDPA -> PagedAttention transformation converts. Only these
# are checked: submodels without a key/value cache - encoders, embedding tables, a codec - have no
# state to page, and the transformation rejects them by design.
ARCH_TO_CB_COMPATIBLE_SUBMODELS = {
    "qwen3_asr": {
        "task": "automatic-speech-recognition",
        "submodels": ("language_model",),
    },
    "ministral3": {
        "task": "image-text-to-text",
        "submodels": ("language_model",),
    },
    "qwen3_tts": {
        "task": "text-to-audio",
        "submodels": ("talker_model", "code_predictor_model"),
    },
    # hybrid Gated DeltaNet + full attention language model with MoE: besides SDPA -> PagedAttention, the linear
    # attention layers become PagedGatedDeltaNet / PagedCausalConv1D
    "minicpmv4_7": {
        "task": "image-text-to-text",
        "submodels": ("language_model",),
    },
    # dense hybrid Gated DeltaNet + full attention language model
    "minicpm_ocr": {
        "task": "image-text-to-text",
        "submodels": ("language_model",),
    },
    # hybrid short conv + full / sliding window attention: besides SDPA -> PagedAttention (with the sliding window
    # extracted from the attention mask), the conv layers become PagedCausalConv1D
    "lfm3": {
        "task": "text-generation-with-past",
        "submodels": ("model",),
        "paged_causal_conv1d": True,
        "sliding_window": True,
    },
    "lfm3_vl": {
        "task": "image-text-to-text",
        "submodels": ("language_model",),
        "paged_causal_conv1d": True,
        "sliding_window": True,
    },
    # the language model of LFM3-Omni is the LFM3 one with an extra hidden_states output (for the depth decoder)
    "lfm3_omni": {
        "task": "image-text-to-text",
        "submodels": ("language_model",),
        "paged_causal_conv1d": True,
        "sliding_window": True,
    },
}

# Architectures exported through an out-of-tree library are not part of the transformers set, so
# they are kept explicitly and gated on that library being installed.
_NON_TRANSFORMERS_ARCHS = {"qwen3_tts": is_qwen_tts_available()}

ARCH_TO_CB_COMPATIBLE_SUBMODELS = {
    arch: expected
    for arch, expected in ARCH_TO_CB_COMPATIBLE_SUBMODELS.items()
    if arch in get_supported_model_for_library("transformers") or _NON_TRANSFORMERS_ARCHS.get(arch, False)
}

# The inputs through which a PagedAttention graph receives the paged cache and its layout.
_PAGED_ATTENTION_INPUTS = (
    "past_lens",
    "subsequence_begins",
    "block_indices",
    "block_indices_begins",
    "max_context_len",
)


class OVContinuousBatchingCompatibilityTest(unittest.TestCase):
    @parameterized.expand(list(ARCH_TO_CB_COMPATIBLE_SUBMODELS.keys()), skip_on_empty=True)
    def test_cb_compatible(self, model_arch):
        expected = ARCH_TO_CB_COMPATIBLE_SUBMODELS[model_arch]
        core = ov.Core()
        with TemporaryDirectory() as tmpdir:
            main_export(
                model_name_or_path=MODEL_NAMES[model_arch],
                output=tmpdir,
                task=expected["task"],
                trust_remote_code=model_arch in REMOTE_CODE_MODELS,
            )
            config = AutoConfig.from_pretrained(tmpdir, trust_remote_code=model_arch in REMOTE_CODE_MODELS)
            # the layer types of the language model (LFM3-Omni nests it in text_config)
            config = config.get_text_config()
            for submodel in expected["submodels"]:
                with self.subTest(submodel=submodel):
                    model = core.read_model(Path(tmpdir) / f"openvino_{submodel}.xml")
                    label = f"{model_arch}/{submodel}"
                    self._assert_paged_attention_applicable(core, model, label)
                    if expected.get("paged_causal_conv1d"):
                        self._assert_paged_causal_conv1d(model, config, label)
                    if expected.get("sliding_window"):
                        self._assert_sliding_window(model, config, label)

    def _assert_paged_causal_conv1d(self, model, config, label):
        # every short conv layer keeps its state in a paged conv state table
        num_conv_layers = config.layer_types.count("conv")
        ops = Counter(op.get_type_name() for op in model.get_ops())
        self.assertEqual(
            ops["PagedCausalConv1D"], num_conv_layers, f"{label}: not every conv became PagedCausalConv1D"
        )
        input_names = {name for port in model.inputs for name in port.get_names()}
        for layer in range(num_conv_layers):
            self.assertIn(f"conv_state_table.{layer}", input_names, f"{label}: missing conv state table {layer}")

    def _assert_sliding_window(self, model, config, label):
        # PagedAttention input 10 is the sliding window: 0 for full attention layers, the window size otherwise
        attention_layer_types = [layer_type for layer_type in config.layer_types if layer_type != "conv"]
        expected = [config.sliding_window if t == "sliding_attention" else 0 for t in attention_layer_types]
        windows = []
        for op in model.get_ordered_ops():
            if op.get_type_name() == "PagedAttentionExtension":
                window = op.input_value(10).get_node()
                # the extracted window is a constant, possibly wrapped by shape / precision adjustments
                while window.get_type_name() in ("Convert", "Squeeze", "Unsqueeze", "Reshape"):
                    window = window.input_value(0).get_node()
                self.assertEqual(
                    window.get_type_name(), "Constant", f"{label}: the sliding window of {op.get_friendly_name()}"
                )
                windows.append(int(window.get_data().flatten()[0]))
        self.assertEqual(windows, expected, f"{label}: unexpected PagedAttention sliding windows")

    def _assert_paged_attention_applicable(self, core, model, label):
        num_sdpa = Counter(op.get_type_name() for op in model.get_ops())["ScaledDotProductAttention"]
        self.assertGreater(num_sdpa, 0, f"{label}: no ScaledDotProductAttention to convert")

        try:
            paged_attention_transformation(model)
        except Exception as exc:
            self.fail(f"{label}: SDPA -> PagedAttention transformation is not applicable: {exc}")

        ops = Counter(op.get_type_name() for op in model.get_ops())
        self.assertEqual(ops["ScaledDotProductAttention"], 0, f"{label}: SDPA blocks left unconverted")
        self.assertEqual(
            ops["PagedAttentionExtension"], num_sdpa, f"{label}: not every SDPA became a PagedAttention block"
        )
        # The paged cache replaces the stateful one entirely. The state operations are what is checked:
        # the transformation removes them but leaves their variables registered on the model.
        self.assertEqual(ops["ReadValue"] + ops["Assign"], 0, f"{label}: key/value cache state left in the graph")

        input_names = {name for port in model.inputs for name in port.get_names()}
        for name in _PAGED_ATTENTION_INPUTS:
            self.assertIn(name, input_names, f"{label}: missing PagedAttention input `{name}`")
        for layer in range(num_sdpa):
            for cache in (f"key_cache.{layer}", f"value_cache.{layer}"):
                self.assertIn(cache, input_names, f"{label}: missing paged cache input `{cache}`")

        # The converted graph has to be one a device can actually run.
        core.compile_model(model, OPENVINO_DEVICE)
