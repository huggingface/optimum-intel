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

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


try:
    import openvino  # noqa: F401

    _OPENVINO_AVAILABLE = True
except ImportError:
    _OPENVINO_AVAILABLE = False


@unittest.skipUnless(_OPENVINO_AVAILABLE, "OpenVINO is required to import the OpenVINO exporter")
class SaveModelErrorTest(unittest.TestCase):
    def test_wraps_save_model_error_with_actionable_message(self):
        from optimum.exporters.openvino.convert import _save_model

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "openvino_model.xml"
            native_error = RuntimeError("basic_ios::clear: iostream error")
            with patch("optimum.exporters.openvino.convert.save_model", side_effect=native_error):
                with self.assertRaises(RuntimeError) as ctx:
                    _save_model(
                        model=MagicMock(),
                        path=target,
                        ov_config=None,
                        library_name="transformers",
                        config=MagicMock(),
                    )

            message = str(ctx.exception)
            self.assertIn(str(target), message)
            self.assertIn("basic_ios::clear: iostream error", message)
            # The original exception must be preserved as the cause.
            self.assertIs(ctx.exception.__cause__, native_error)


if __name__ == "__main__":
    unittest.main()
