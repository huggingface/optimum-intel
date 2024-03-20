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
import logging
import operator as op
import sys
from collections import OrderedDict
from typing import Union

from packaging.version import Version, parse


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


logger = logging.getLogger(__name__)

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_optimum_version = importlib_metadata.version("optimum")

_transformers_available = importlib.util.find_spec("transformers") is not None
_transformers_version = "N/A"
if _transformers_available:
    try:
        _transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        _transformers_available = False


_torch_available = importlib.util.find_spec("torch") is not None
_torch_version = "N/A"
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False


_neural_compressor_available = importlib.util.find_spec("neural_compressor") is not None
_neural_compressor_version = "N/A"
if _neural_compressor_available:
    try:
        _neural_compressor_version = importlib_metadata.version("neural_compressor")
    except importlib_metadata.PackageNotFoundError:
        _neural_compressor_available = False


_ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None
_ipex_version = "N/A"
if _ipex_available:
    try:
        _ipex_version = importlib_metadata.version("intel_extension_for_pytorch")
    except importlib_metadata.PackageNotFoundError:
        _ipex_available = False

_openvino_available = importlib.util.find_spec("openvino") is not None
_openvino_version = "N/A"
if _openvino_available:
    try:
        from openvino.runtime import get_version

        version = get_version()
        # avoid invalid format
        if "-" in version:
            ov_major_version, dev_info = version.split("-", 1)
            commit_id = dev_info.split("-")[0]
            version = f"{ov_major_version}-{commit_id}"
        _openvino_version = version
    except ImportError:
        _openvino_available = False

_openvino_tokenizers_available = importlib.util.find_spec("openvino_tokenizers") is not None and _openvino_available
_openvino_tokenizers_version = "N/A"
if _openvino_tokenizers_available:
    try:
        _openvino_tokenizers_version = importlib_metadata.version("openvino_tokenizers")
    except importlib_metadata.PackageNotFoundError:
        _openvino_tokenizers_available = False

if _openvino_tokenizers_available and _openvino_tokenizers_version != "N/A":
    _compatible_openvino_version = next(
        (
            requirement.split("==")[-1]
            for requirement in importlib_metadata.requires("openvino-tokenizers")
            if requirement.startswith("openvino==")
        ),
        "",
    )
    _openvino_tokenizers_available = _compatible_openvino_version == ov_major_version
    if not _openvino_tokenizers_available:
        logger.warning(
            "OpenVINO Tokenizer version is not compatible with OpenVINO version. "
            f"Installed OpenVINO version: {ov_major_version},"
            f"OpenVINO Tokenizers requires {_compatible_openvino_version}. "
            f"OpenVINO Tokenizers models will not be added during export."
        )

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


_safetensors_version = "N/A"
_safetensors_available = importlib.util.find_spec("safetensors") is not None
if _safetensors_available:
    try:
        _safetensors_version = importlib_metadata.version("safetensors")
    except importlib_metadata.PackageNotFoundError:
        _safetensors_available = False


_timm_available = importlib.util.find_spec("timm") is not None
_timm_version = "N/A"
if _timm_available:
    try:
        _timm_version = importlib_metadata.version("timm")
    except importlib_metadata.PackageNotFoundError:
        _timm_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None
_datasets_version = "N/A"

if _datasets_available:
    try:
        _datasets_version = importlib_metadata.version("datasets")
    except importlib_metadata.PackageNotFoundError:
        _datasets_available = False


_accelerate_available = importlib.util.find_spec("accelerate") is not None
_accelerate_version = "N/A"

if _accelerate_available:
    try:
        _accelerate_version = importlib_metadata.version("accelerate")
    except importlib_metadata.PackageNotFoundError:
        _accelerate_available = False


def is_transformers_available():
    return _transformers_available


def is_neural_compressor_available():
    return _neural_compressor_available


def is_ipex_available():
    return _ipex_available


def is_openvino_available():
    return _openvino_available


def is_openvino_tokenizers_available():
    return _openvino_tokenizers_available


def is_nncf_available():
    return _nncf_available


def is_diffusers_available():
    return _diffusers_available


def is_safetensors_available():
    return _safetensors_available


def is_timm_available():
    return _timm_available


def is_datasets_available():
    return _datasets_available


def is_accelerate_available():
    return _accelerate_available


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


def is_optimum_version(operation: str, version: str):
    return compare_versions(parse(_optimum_version), operation, version)


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


def is_torch_version(operation: str, version: str):
    """
    Compare the current torch version to a given reference with an operation.
    """
    if not _torch_available:
        return False

    import torch

    return compare_versions(parse(parse(torch.__version__).base_version), operation, version)


def is_ipex_version(operation: str, version: str):
    """
    Compare the current ipex version to a given reference with an operation.
    """
    if not _ipex_available:
        return False
    return compare_versions(parse(_ipex_version), operation, version)


def is_timm_version(operation: str, version: str):
    """
    Compare the current timm version to a given reference with an operation.
    """
    if not _timm_available:
        return False
    return compare_versions(parse(_timm_version), operation, version)


DIFFUSERS_IMPORT_ERROR = """
{0} requires the diffusers library but it was not found in your environment. You can install it with pip:
`pip install diffusers`. Please note that you may need to restart your runtime after installation.
"""

IPEX_IMPORT_ERROR = """
{0} requires the ipex library but it was not found in your environment. You can install it with pip:
`pip install intel_extension_for_pytorch`. Please note that you may need to restart your runtime after installation.
"""

NNCF_IMPORT_ERROR = """
{0} requires the nncf library but it was not found in your environment. You can install it with pip:
`pip install nncf`. Please note that you may need to restart your runtime after installation.
"""

OPENVINO_IMPORT_ERROR = """
{0} requires the openvino library but it was not found in your environment. You can install it with pip:
`pip install openvino`. Please note that you may need to restart your runtime after installation.
"""

NEURAL_COMPRESSOR_IMPORT_ERROR = """
{0} requires the neural-compressor library but it was not found in your environment. You can install it with pip:
`pip install neural-compressor`. Please note that you may need to restart your runtime after installation.
"""

DATASETS_IMPORT_ERROR = """
{0} requires the datasets library but it was not found in your environment. You can install it with pip:
`pip install datasets`. Please note that you may need to restart your runtime after installation.
"""

ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library but it was not found in your environment. You can install it with pip:
`pip install accelerate`. Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("diffusers", (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)),
        ("ipex", (is_ipex_available, IPEX_IMPORT_ERROR)),
        ("nncf", (is_nncf_available, NNCF_IMPORT_ERROR)),
        ("openvino", (is_openvino_available, OPENVINO_IMPORT_ERROR)),
        ("neural_compressor", (is_neural_compressor_available, NEURAL_COMPRESSOR_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


# Copied from: https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/utils/import_utils.py#L1041
class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)
