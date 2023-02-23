#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

import importlib.util
import operator as op
import sys
from typing import Any, List, Union

from packaging.version import Version, parse


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}


_transformers_available = importlib.util.find_spec("transformers") is not None
_transformers_version = "N/A"
if _transformers_available:
    try:
        _transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        _transformers_available = False


_neural_compressor_available = importlib.util.find_spec("neural_compressor") is not None
_neural_compressor_version = "N/A"
if _neural_compressor_available:
    try:
        _neural_compressor_version = importlib_metadata.version("neural_compressor")
    except importlib_metadata.PackageNotFoundError:
        _neural_compressor_available = False


_openvino_available = importlib.util.find_spec("openvino") is not None
_openvino_version = "N/A"
if _openvino_available:
    try:
        _openvino_version = importlib_metadata.version("openvino")
    except importlib_metadata.PackageNotFoundError:
        _openvino_available = False


_nncf_available = importlib.util.find_spec("nncf") is not None
_nncf_version = "N/A"
if _nncf_available:
    try:
        _nncf_version = importlib_metadata.version("nncf")
    except importlib_metadata.PackageNotFoundError:
        _nncf_available = False


_diffusers_available = importlib.util.find_spec("diffusers") is not None
_diffusers_version = "N/A"
if _diffusers_available:
    try:
        _diffusers_version = importlib_metadata.version("diffusers")
    except importlib_metadata.PackageNotFoundError:
        _diffusers_available = False


def is_transformers_available():
    return _transformers_available


def is_neural_compressor_available():
    return _neural_compressor_available


def is_openvino_available():
    return _openvino_available


def is_nncf_available():
    return _nncf_available


def is_diffusers_available():
    return _diffusers_available


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compare a library version to some requirement using a given operation.

    Arguments:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def is_transformers_version(operation: str, version: str):
    """
    Compare the current Transformers version to a given reference with an operation.
    """
    if not _transformers_available:
        return False
    return compare_versions(parse(_transformers_version), operation, version)


def is_neural_compressor_version(operation: str, version: str):
    """
    Compare the current Neural Compressor version to a given reference with an operation.
    """
    if not _neural_compressor_available:
        return False
    return compare_versions(parse(_neural_compressor_version), operation, version)


def is_openvino_version(operation: str, version: str):
    """
    Compare the current OpenVINO version to a given reference with an operation.
    """
    if not _openvino_available:
        return False
    return compare_versions(parse(_openvino_version), operation, version)


def is_diffusers_version(operation: str, version: str):
    """
    Compare the current diffusers version to a given reference with an operation.
    """
    if not _diffusers_available:
        return False
    return compare_versions(parse(_diffusers_version), operation, version)
