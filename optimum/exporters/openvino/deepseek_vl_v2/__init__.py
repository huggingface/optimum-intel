# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bundled DeepSeek-VL2 (``deepseek_vl_v2``) modeling/processing code.

The Hub checkpoints of the ``deepseek_vl_v2`` family (e.g.
``deepseek-ai/deepseek-vl2-tiny``) ship neither an ``auto_map`` nor any
``*.py`` modeling files, and the architecture is not natively supported by
``transformers``. The upstream ``deepseek_vl2`` package targets an old
``transformers`` release and cannot be installed alongside the pinned one.

These vendored files are a compatibility-patched copy of that upstream code so
the architecture can be exported to OpenVINO with the installed ``transformers``
release. Importing :func:`register` registers the config/model classes with the
``transformers`` Auto* factories so ``AutoConfig``/``AutoModelForCausalLM``/
``AutoModelForImageTextToText`` resolve the ``deepseek_vl_v2`` ``model_type`` for
the unmodified Hub weights.
"""

_REGISTERED = False


def register():
    """Register the ``deepseek_vl_v2`` classes with the transformers Auto* API.

    Idempotent: importing the modeling module performs the registration, so
    repeated calls are cheap and safe.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    # Importing the modeling module triggers the AutoConfig/AutoModel
    # registrations defined at its module scope.
    from . import modeling_deepseek_vl_v2  # noqa: F401

    # Register the processor with the ``AutoProcessor`` factory so that it can
    # be resolved from ``processor_config.json`` (which declares
    # ``processor_class = "DeepseekVLV2Processor"``) shipped by the real Hub
    # checkpoints. This lets the OpenVINO exporter save the processor alongside
    # the exported model and lets downstream tools reload it from the export
    # directory without remote code. It is kept out of the bundled modeling
    # module so that copying the modeling file as remote code does not pull in
    # the processing/conversation modules.
    try:
        from transformers import AutoProcessor

        from .modeling_deepseek_vl_v2 import DeepseekVLV2Config
        from .processing_deepseek_vl_v2 import DeepseekVLV2Processor

        try:
            AutoProcessor.register(DeepseekVLV2Config, DeepseekVLV2Processor)
        except (ValueError, KeyError):
            pass
    except Exception:
        pass

    _REGISTERED = True


def get_processor_class():
    from .processing_deepseek_vl_v2 import DeepseekVLV2Processor

    return DeepseekVLV2Processor
