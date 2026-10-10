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
"""Filesystem helpers used to give actionable errors when exporting models to OpenVINO IR.

These helpers only rely on the standard library so they can be used (and tested) without the full OpenVINO stack.
The OpenVINO serialization is performed by a native ``std::ofstream`` and, when the write fails, only surfaces an
opaque ``basic_ios::clear: iostream error``. The utilities below translate that into a message describing the most
likely causes (disk full, read-only mount, FAT32 per-file size limit, ...).
"""

import os
import shutil
from pathlib import Path
from typing import Optional


# A single file cannot exceed 4 GiB - 1 byte on a FAT32 volume.
FAT32_MAX_FILE_SIZE = 4 * 1024**3 - 1

# Environment variable letting users override where the CLI stores the temporary export/quantization directory.
OPTIMUM_OPENVINO_TMPDIR_ENV = "OPTIMUM_OPENVINO_TMPDIR"


def get_export_tmpdir(output, env=None) -> str:
    """Return the parent directory used for the temporary directory of the OpenVINO export CLI.

    By default the temporary directory is created next to the final output so that it lives on the same filesystem
    (avoiding a size-limited ``/tmp`` and a costly cross-device move). The location can be overridden with the
    ``OPTIMUM_OPENVINO_TMPDIR`` environment variable.
    """
    env = os.environ if env is None else env
    override = env.get(OPTIMUM_OPENVINO_TMPDIR_ENV)
    if override:
        return override
    output = Path(output)
    return str(output.parent) if str(output.parent) else "."


def human_readable_size(num_bytes: float) -> str:
    """Format a size in bytes using binary units (KB, MB, GB, ...)."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1024.0 or unit == "PB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _unescape_mount_field(field: str) -> str:
    """Decode the octal escapes used by ``/proc/mounts`` (e.g. ``\\040`` for a space)."""
    return field.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def get_filesystem_info(path) -> dict:
    """Return information about the filesystem containing ``path``.

    On Linux the mount table is parsed to report the filesystem type and whether it is mounted read-only. On other
    platforms (or when the information cannot be determined) ``fstype``, ``mount_point`` and ``read_only`` are set to
    ``None``.
    """
    resolved = Path(path).resolve()
    info = {"path": str(resolved), "fstype": None, "mount_point": None, "read_only": None}

    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            mounts = f.read().splitlines()
    except OSError:
        return info

    best_mount_point = None
    best_fstype = None
    best_read_only = None
    resolved_str = str(resolved)
    for line in mounts:
        fields = line.split()
        if len(fields) < 4:
            continue
        mount_point = _unescape_mount_field(fields[1])
        fstype = fields[2]
        options = fields[3].split(",")
        if resolved_str == mount_point or resolved_str.startswith(mount_point.rstrip("/") + "/"):
            if best_mount_point is None or len(mount_point) > len(best_mount_point):
                best_mount_point = mount_point
                best_fstype = fstype
                best_read_only = "ro" in options

    if best_mount_point is not None:
        info["mount_point"] = best_mount_point
        info["fstype"] = best_fstype
        info["read_only"] = best_read_only
    return info


def check_output_path(path, required_bytes: Optional[int] = None, purpose: str = "export") -> None:
    """Validate that ``path`` (a directory) can be used as an export destination.

    The directory (and its parents) is created if needed, then a small probe file is written and removed to detect
    unwritable or read-only destinations before starting a long export. When ``required_bytes`` is provided it is
    interpreted as the size of the largest file that will be written and compared against the free space, and checked
    against the FAT32 per-file size limit.

    Raises:
        RuntimeError: with an actionable message when the directory cannot be written to.
    """
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"The {purpose} directory '{path}' could not be created: {exc}. "
            "Check that the path is valid and that you have write permissions."
        ) from exc

    probe = path / f".optimum_write_test_{os.getpid()}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"The {purpose} directory '{path}' is not writable: {exc}. "
            "Check the permissions and that the filesystem is not mounted read-only."
        ) from exc

    info = get_filesystem_info(path)
    if info["read_only"]:
        raise RuntimeError(
            f"The {purpose} directory '{path}' is located on '{info['mount_point']}', which is mounted read-only. "
            "Remount the filesystem read-write or choose another output directory."
        )

    if required_bytes is not None:
        if info["fstype"] == "vfat" and required_bytes > FAT32_MAX_FILE_SIZE:
            raise RuntimeError(
                f"The {purpose} directory '{path}' is on a FAT32 filesystem, which limits a single file to 4 GiB, "
                f"but the export may write a file of up to {human_readable_size(required_bytes)}. "
                "Use a filesystem without this limit (ext4, NTFS, exFAT) or export to a local disk first."
            )
        try:
            free = shutil.disk_usage(path).free
        except OSError:
            return
        if free < required_bytes:
            raise RuntimeError(
                f"Not enough free space to {purpose} to '{path}': {human_readable_size(free)} available but "
                f"{human_readable_size(required_bytes)} required. Free up space or choose another output directory."
            )


def format_save_error(path, error: Exception) -> str:
    """Build an actionable error message for a failed OpenVINO model serialization.

    ``path`` is the model path that was being written (e.g. ``openvino_model.xml``). The message includes the original
    error, the free space of the target filesystem and filesystem-specific hints.
    """
    path = Path(path)
    parent = path.parent if str(path.parent) else Path(".")
    info = get_filesystem_info(parent)

    lines = [
        f"Failed to save the OpenVINO model to '{path}': {error}",
        "",
        "The OpenVINO serialization could not write the model files. The most common causes are:",
        "  - not enough free disk space on the target filesystem;",
        "  - the target directory or filesystem being read-only;",
        "  - a per-file size limit of the target filesystem (e.g. 4 GiB on FAT32).",
    ]

    try:
        free = human_readable_size(shutil.disk_usage(parent).free)
        location = info.get("mount_point") or str(parent)
        lines.append(f"Free space reported for '{location}': {free}.")
    except OSError:
        pass

    if info.get("read_only"):
        lines.append(
            f"'{info['mount_point']}' is mounted read-only: remount it read-write or choose another output directory."
        )
    if info.get("fstype") == "vfat":
        lines.append(
            "The target filesystem is FAT32, which limits a single file to 4 GiB while the OpenVINO '.bin' file can be "
            "larger. Use ext4, NTFS or exFAT, or export to a local disk first."
        )

    lines.append(
        "Note that exports using an explicit quantization (e.g. '--weight-format int4') first write the model to a "
        "temporary directory (controlled by the 'TMPDIR'/'TEMP' environment variables, default '/tmp'). Make sure that "
        "temporary location also has enough free space, for example by setting 'TMPDIR' to a directory on a large "
        "filesystem."
    )
    return "\n".join(lines)
