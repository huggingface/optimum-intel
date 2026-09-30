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
from utils_tests import MODEL_NAMES, OPENVINO_DEVICE, REMOTE_CODE_MODELS, get_supported_model_for_library

from optimum.exporters.openvino import main_export
from optimum.intel.openvino.utils import TemporaryDirectory
from optimum.intel.utils.import_utils import is_qwen_tts_available


# The submodels of each architecture that continuous batching can serve, i.e. whose stateful
# ScaledDotProductAttention blocks the SDPA -> PagedAttention transformation converts. Only these
# are checked: submodels without a key/value cache - encoders, embedding tables, a codec - have no
# state to page, and the transformation rejects them by design.
ARCH_TO_CB_COMPATIBLE_SUBMODELS = {
    "qwen3_tts": {
        "task": "text-to-audio",
        "submodels": ("talker_model", "code_predictor_model"),
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
            for submodel in expected["submodels"]:
                with self.subTest(submodel=submodel):
                    model = core.read_model(Path(tmpdir) / f"openvino_{submodel}.xml")
                    self._assert_paged_attention_applicable(core, model, f"{model_arch}/{submodel}")

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
