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
import unittest
from unittest.mock import patch

from optimum.commands.export import transformers_version as tv

TINY_GPT2 = "optimum-intel-internal-testing/tiny-random-gpt2"


class TransformersVersionResolverTest(unittest.TestCase):
    def _resolve(self, *, current, requested="auto", bounds=(None, None), config_version=None, model_type="arch"):
        with (
            patch.object(tv, "_transformers_version", current),
            patch.object(tv, "_export_config_bounds", return_value=bounds),
            patch.object(tv, "_read_model_config", return_value=(model_type, config_version)),
        ):
            return tv.resolve_transformers_specifier("model", requested, None)

    def test_explicit_version_equal_to_current_is_noop(self):
        self.assertIsNone(self._resolve(current="4.57.6", requested="4.57.6"))

    def test_explicit_version_different_pins_exactly(self):
        self.assertEqual(self._resolve(current="4.57.6", requested="4.55.0"), "transformers==4.55.0")

    def test_explicit_version_ignores_resolution(self):
        # Bounds / config are irrelevant when an explicit version is requested.
        self.assertEqual(
            self._resolve(current="4.57.6", requested="5.2.0", bounds=(None, "4.53.3"), config_version="5.2.0"),
            "transformers==5.2.0",
        )

    def test_auto_in_range_is_noop(self):
        self.assertIsNone(self._resolve(current="4.55.0", bounds=("4.50.0", "4.57.0")))

    def test_auto_below_min_returns_range(self):
        self.assertEqual(self._resolve(current="4.49.0", bounds=("4.51.0", None)), "transformers>=4.51.0")

    def test_auto_above_max_returns_range(self):
        self.assertEqual(self._resolve(current="4.57.6", bounds=(None, "4.53.3")), "transformers<=4.53.3")

    def test_auto_min_and_max_returns_full_range(self):
        self.assertEqual(
            self._resolve(current="4.49.0", bounds=("4.51.0", "4.55.4")),
            "transformers>=4.51.0,<=4.55.4",
        )

    def test_auto_config_floor_newer_pins_exactly(self):
        self.assertEqual(
            self._resolve(current="4.57.6", bounds=(None, None), config_version="5.9.0"),
            "transformers==5.9.0",
        )

    def test_auto_config_floor_older_is_noop(self):
        self.assertIsNone(self._resolve(current="4.57.6", bounds=(None, None), config_version="4.40.0"))

    def test_auto_no_signal_is_noop(self):
        self.assertIsNone(self._resolve(current="4.57.6", bounds=(None, None), config_version=None))

    def test_auto_conflict_raises(self):
        with self.assertRaises(ValueError):
            self._resolve(current="4.57.6", bounds=(None, "4.53.3"), config_version="5.2.0")

    def test_export_config_bounds_normalizes_version_objects(self):
        from packaging.version import Version

        from optimum.exporters.tasks import TasksManager

        class FakeConfig:
            MIN_TRANSFORMERS_VERSION = Version("4.51.0")
            MAX_TRANSFORMERS_VERSION = None

        with patch.object(
            TasksManager,
            "_SUPPORTED_MODEL_TYPE",
            {"fake": {"openvino": {"text-generation": FakeConfig}}},
        ):
            self.assertEqual(tv._export_config_bounds("fake"), ("4.51.0", None))

    def test_real_model_config_read(self):
        # Integration over the real config.json read for a known tiny model.
        model_type, config_version = tv._read_model_config(TINY_GPT2, None)
        self.assertEqual(model_type, "gpt2")


class MaybeSwitchTransformersVersionTest(unittest.TestCase):
    def test_no_request_is_noop(self):
        with patch.object(tv, "resolve_transformers_specifier") as resolve:
            tv.maybe_switch_transformers_version("model", None, None)
            resolve.assert_not_called()

    def test_reexec_guard_satisfied_is_noop(self):
        with (
            patch.dict("os.environ", {tv._REEXEC_GUARD: "1"}),
            patch.object(tv, "resolve_transformers_specifier", return_value=None),
            patch("os.execve") as execve,
        ):
            tv.maybe_switch_transformers_version("model", "auto", None)
            execve.assert_not_called()

    def test_reexec_guard_still_unsatisfied_raises(self):
        with (
            patch.dict("os.environ", {tv._REEXEC_GUARD: "1"}),
            patch.object(tv, "resolve_transformers_specifier", return_value="transformers>=5.2.0,<=5.2.99"),
        ):
            with self.assertRaises(RuntimeError):
                tv.maybe_switch_transformers_version("model", "auto", None)

    def test_no_switch_needed_does_not_exec(self):
        with (
            patch.object(tv, "resolve_transformers_specifier", return_value=None),
            patch("os.execve") as execve,
        ):
            tv.maybe_switch_transformers_version("model", "auto", None)
            execve.assert_not_called()

    def test_missing_uv_raises(self):
        with (
            patch.object(tv, "resolve_transformers_specifier", return_value="transformers==4.55.0"),
            patch("shutil.which", return_value=None),
        ):
            with self.assertRaises(RuntimeError):
                tv.maybe_switch_transformers_version("model", "4.55.0", None)

    def test_reexec_invokes_uv_with_specifier(self):
        with (
            patch.object(tv, "resolve_transformers_specifier", return_value="transformers==4.55.0"),
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch("os.execve") as execve,
        ):
            tv.maybe_switch_transformers_version("model", "4.55.0", None)
            execve.assert_called_once()
            program, args, env = execve.call_args[0]
            self.assertEqual(program, "/usr/bin/uv")
            self.assertIn("transformers==4.55.0", args)
            self.assertIn("--with", args)
            self.assertEqual(env[tv._REEXEC_GUARD], "1")
