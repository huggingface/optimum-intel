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
from types import SimpleNamespace

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import torch

from optimum.intel.utils.import_utils import is_diffusers_available


@unittest.skipIf(not is_diffusers_available(), "diffusers is required")
class OVModelConnectorsOutputNamesTest(unittest.TestCase):
    """`OVModelConnectors.forward` must find its outputs when a port carries several tensor names.

    Ports get extra names e.g. after `reshape()` or when an IR is re-saved; `port.names` is an unordered set.
    """

    OUTPUTS = ["video_text_embedding", "audio_text_embedding", "connector_attention_mask"]

    def _connectors(self, tmp_dir, extra_names):
        from optimum.intel.openvino.modeling_diffusion import OVModelConnectors

        hidden = ops.parameter([1, 4, 6], np.float32, name="text_encoder_hidden_states")
        mask = ops.parameter([1, 4], np.float32, name="attention_mask")
        results = []
        for i, name in enumerate(self.OUTPUTS):
            src = mask if name == "connector_attention_mask" else hidden
            node = ops.multiply(src, ops.constant(np.float32(i + 1)))
            node.output(0).get_tensor().set_names({name, *(f"{name}_alias_{k}" for k in range(extra_names))})
            results.append(node)
        model = ov.Model(results, [hidden, mask])

        (Path(tmp_dir) / "connectors").mkdir()
        (Path(tmp_dir) / "connectors" / "config.json").write_text(json.dumps({}))
        parent = SimpleNamespace(_compile_only=False, ov_config={}, model_save_dir=tmp_dir, _device="CPU")
        return OVModelConnectors(model, parent, "connectors")

    def test_outputs_with_several_names(self):
        hidden = torch.randn(1, 4, 6)
        mask = torch.ones(1, 4)
        for extra_names in (0, 8):
            with self.subTest(extra_names=extra_names), TemporaryDirectory() as tmp_dir:
                video, audio, out_mask = self._connectors(tmp_dir, extra_names)(hidden, mask)
                torch.testing.assert_close(video, hidden * 1)
                torch.testing.assert_close(audio, hidden * 2)
                torch.testing.assert_close(out_mask, mask * 3)
