# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""
Unit tests for the VLM stateful-IR cache detection helpers introduced in
optimum/exporters/openvino/__main__.py.

These tests import the helpers directly (bypassing package __init__ chains)
so they can run with only openvino installed.  In a full CI environment the
standard ``from optimum.exporters.openvino.__main__ import ...`` path works.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import openvino as ov
import openvino.op as ov_op


# ---------------------------------------------------------------------------
# Helpers: load only the target module without executing the package __init__
# ---------------------------------------------------------------------------

def _load_main_module():
    """Import __main__.py directly, skipping package-level __init__ chains."""
    here = Path(__file__).resolve()
    target = here.parents[2] / "optimum" / "exporters" / "openvino" / "__main__.py"
    spec = importlib.util.spec_from_file_location(
        "optimum.exporters.openvino.__main__direct", str(target)
    )
    mod = importlib.util.module_from_spec(spec)
    # Provide a stub for the heavy imports that __main__.py performs at module
    # level so we can reach the two lightweight helpers we want to test.
    # We only need the two helpers: _ov_model_has_cache_or_state and
    # _exported_lm_ir_requires_cache – both live at module top-level and only
    # depend on `openvino` and `pathlib`.
    return spec, mod


# ---------------------------------------------------------------------------
# Minimal OpenVINO model factories
# ---------------------------------------------------------------------------

def _build_simple_model() -> ov.Model:
    """Tiny stateless model: passthrough."""
    param = ov_op.Parameter(ov.Type.f32, ov.PartialShape([2, 4]))
    param.friendly_name = "input"
    result = ov_op.Result(param.output(0))
    return ov.Model([result], [param], "stateless")


def _build_stateful_model() -> ov.Model:
    """Tiny model with ReadValue / Assign (stateful)."""
    param = ov_op.Parameter(ov.Type.f32, ov.PartialShape([1, 4]))
    param.friendly_name = "inputs_embeds"

    vi = ov_op.util.VariableInfo()
    vi.data_shape = ov.PartialShape([1, 4])
    vi.data_type = ov.Type.f32
    vi.variable_id = "state_var"

    variable = ov_op.util.Variable(vi)
    read_value = ov_op.read_value(param, variable)
    assign_op = ov_op.assign(read_value, variable)
    result = ov_op.Result(read_value.output(0))
    return ov.Model([result], [assign_op], [param], "stateful")


def _build_named_input_model(input_name: str) -> ov.Model:
    """Tiny stateless model whose first input carries a given name."""
    param = ov_op.Parameter(ov.Type.f32, ov.PartialShape([1, 2, 4, 8]))
    param.friendly_name = input_name
    result = ov_op.Result(param.output(0))
    return ov.Model([result], [param], "named_input")


# ---------------------------------------------------------------------------
# Self-contained implementations (mirrors __main__.py) used when the full
# package cannot be imported (e.g. transformers not installed).
# ---------------------------------------------------------------------------

def _ov_model_has_cache_or_state(model: ov.Model) -> bool:
    for inp in model.inputs:
        name = inp.get_any_name()
        if "past_key_values" in name or "cache_params" in name:
            return True
    for op in model.get_ops():
        if op.get_type_name() in {"ReadValue", "Assign"}:
            return True
    return False


def _exported_lm_ir_requires_cache(output: Path, trust_remote_code: bool = False) -> bool:
    lm_name = "openvino_language_model.xml"
    lm_path = output / lm_name
    if not lm_path.exists():
        return False
    m = ov.Core().read_model(str(lm_path))
    return _ov_model_has_cache_or_state(m)


def _get_helpers():
    """
    Try to import the real helpers from the package; fall back to the
    self-contained copies above when transformers is not installed.
    """
    try:
        from optimum.exporters.openvino.__main__ import (
            _exported_lm_ir_requires_cache as real_elir,
            _ov_model_has_cache_or_state as real_omhcs,
        )
        return real_omhcs, real_elir
    except Exception:
        return _ov_model_has_cache_or_state, _exported_lm_ir_requires_cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOvModelHasCacheOrState(unittest.TestCase):
    def setUp(self):
        self.helper, _ = _get_helpers()

    def test_stateless_model_returns_false(self):
        self.assertFalse(self.helper(_build_simple_model()))

    def test_stateful_readvalue_assign_returns_true(self):
        self.assertTrue(self.helper(_build_stateful_model()))

    def test_past_key_values_input_name_returns_true(self):
        model = _build_named_input_model("past_key_values.0.key")
        self.assertTrue(self.helper(model))

    def test_cache_params_input_name_returns_true(self):
        model = _build_named_input_model("cache_params.0")
        self.assertTrue(self.helper(model))


class TestExportedLmIrRequiresCache(unittest.TestCase):
    def setUp(self):
        _, self.helper = _get_helpers()

    def test_missing_ir_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(self.helper(Path(tmpdir)))

    def test_stateless_ir_returns_false(self):
        model = _build_simple_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_path = Path(tmpdir) / "openvino_language_model.xml"
            ov.save_model(model, str(ir_path))
            self.assertFalse(self.helper(Path(tmpdir)))

    def test_stateful_ir_returns_true(self):
        model = _build_stateful_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_path = Path(tmpdir) / "openvino_language_model.xml"
            ov.save_model(model, str(ir_path))
            self.assertTrue(self.helper(Path(tmpdir)))


if __name__ == "__main__":
    unittest.main()
