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

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_disk_utils():
    """Import the helper module.

    The preferred import goes through the package. When the full OpenVINO stack is not installed we fall back to
    loading the (dependency-free) module directly from disk so that this unit test can run in isolation.
    """
    try:
        from optimum.exporters.openvino import disk_utils
    except ImportError:
        module_path = Path(__file__).resolve().parents[2] / "optimum" / "exporters" / "openvino" / "disk_utils.py"
        spec = importlib.util.spec_from_file_location("optimum_exporters_openvino_disk_utils", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        disk_utils = module
    return disk_utils


disk_utils = _load_disk_utils()


class HumanReadableSizeTest(unittest.TestCase):
    def test_formats_bytes_binary_units(self):
        self.assertEqual(disk_utils.human_readable_size(0), "0.0 B")
        self.assertEqual(disk_utils.human_readable_size(1024), "1.0 KB")
        self.assertEqual(disk_utils.human_readable_size(1536), "1.5 KB")
        self.assertEqual(disk_utils.human_readable_size(5 * 1024**3), "5.0 GB")


class GetFilesystemInfoTest(unittest.TestCase):
    def test_returns_expected_keys_for_existing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info = disk_utils.get_filesystem_info(tmpdir)
            self.assertEqual(set(info) >= {"path", "fstype", "mount_point", "read_only"}, True)
            self.assertTrue(Path(info["path"]).exists())
            self.assertIn(info["read_only"], (True, False, None))


class CheckOutputPathTest(unittest.TestCase):
    def test_creates_missing_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b"
            disk_utils.check_output_path(nested)
            self.assertTrue(nested.is_dir())

    def test_raises_when_parent_is_a_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "not_a_dir"
            file_path.touch()
            with self.assertRaises(RuntimeError) as ctx:
                disk_utils.check_output_path(file_path / "sub")
            self.assertIn(str(file_path / "sub"), str(ctx.exception))

    def test_raises_when_not_enough_free_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            free = shutil.disk_usage(tmpdir).free
            with self.assertRaises(RuntimeError) as ctx:
                disk_utils.check_output_path(tmpdir, required_bytes=free + 1)
            self.assertIn("free space", str(ctx.exception).lower())

    def test_no_raise_when_enough_free_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            disk_utils.check_output_path(tmpdir, required_bytes=1)

    def test_raises_on_fat32_when_single_file_exceeds_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info = {"path": str(tmpdir), "fstype": "vfat", "mount_point": "/mnt", "read_only": False}
            with patch.object(disk_utils, "get_filesystem_info", return_value=info):
                with self.assertRaises(RuntimeError) as ctx:
                    disk_utils.check_output_path(tmpdir, required_bytes=disk_utils.FAT32_MAX_FILE_SIZE + 1)
            self.assertIn("FAT32", str(ctx.exception))

    def test_no_raise_on_fat32_for_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info = {"path": str(tmpdir), "fstype": "vfat", "mount_point": "/mnt", "read_only": False}
            with patch.object(disk_utils, "get_filesystem_info", return_value=info):
                disk_utils.check_output_path(tmpdir, required_bytes=1024)


class GetExportTmpdirTest(unittest.TestCase):
    def test_defaults_to_output_parent(self):
        output = Path("/somewhere/models/qwen-int4")
        self.assertEqual(disk_utils.get_export_tmpdir(output, env={}), str(output.parent))

    def test_defaults_to_current_dir_without_parent(self):
        self.assertEqual(disk_utils.get_export_tmpdir("model", env={}), ".")

    def test_env_override_wins(self):
        env = {disk_utils.OPTIMUM_OPENVINO_TMPDIR_ENV: "/scratch"}
        self.assertEqual(disk_utils.get_export_tmpdir(Path("/somewhere/models/qwen-int4"), env=env), "/scratch")


class FormatSaveErrorTest(unittest.TestCase):
    def test_includes_path_and_original_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "openvino_model.xml"
            message = disk_utils.format_save_error(target, RuntimeError("basic_ios::clear: iostream error"))
        self.assertIn(str(target), message)
        self.assertIn("basic_ios::clear: iostream error", message)

    def test_mentions_read_only_filesystem(self):
        info = {"path": "/mnt", "fstype": "ext4", "mount_point": "/mnt", "read_only": True}
        with patch.object(disk_utils, "get_filesystem_info", return_value=info):
            message = disk_utils.format_save_error(Path("/mnt/openvino_model.xml"), RuntimeError("boom"))
        self.assertIn("read-only", message)

    def test_mentions_fat32_file_size_limit(self):
        info = {"path": "/mnt", "fstype": "vfat", "mount_point": "/mnt", "read_only": False}
        with patch.object(disk_utils, "get_filesystem_info", return_value=info):
            message = disk_utils.format_save_error(Path("/mnt/openvino_model.xml"), RuntimeError("boom"))
        self.assertIn("FAT32", message)


if __name__ == "__main__":
    unittest.main()
