#  Copyright 2024 The HuggingFace Team. All rights reserved.
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

# ruff: noqa

import logging
from typing import Union

import torch
from torch import nn
from transformers import add_start_docstrings
from transformers.pipelines import Pipeline
from transformers.utils import is_ipex_available

from ...exporters.tasks import TasksManager
from ..generation.modeling import jit_trace
from .modeling_base import (
    IPEXModel,
    IPEXModelForCausalLM,
    IPEXModelForMaskedLM,
    IPEXModelForSequenceClassification,
    IPEXModelForTokenClassification,
    IPEXModelForQuestionAnswering,
)


from .utils import _HEAD_TO_AUTOMODELS


logger = logging.getLogger(__name__)

IPEX_NOT_AVAILABLE_ERROR_MSG = (
    "Intel PyTorch Extensions was not found."
    "please make sure you've installed the package or run "
    "pip install intel_extension_for_pytorch"
)

if is_ipex_available():
    import intel_extension_for_pytorch as ipex


class _ModelFallbackWrapper:
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        try:
            return self._optimized(*args, **kwargs)
        except Exception:
            return self._default(*args, **kwargs)

    def __getattr__(self, item):
        if not item.startswith("__"):
            return getattr(self._default, item)
        else:
            return self.item


@add_start_docstrings(
    """
    inference_mode is an Intel specific context-manager analogous to PyTorch's inference_mode to use for inference
    workload on Intel CPUs, especially Intel Xeon Scalable CPUs.
    """,
)
class inference_mode:
    __slots__ = ("_model", "_dtype", "_graph_mode", "_verbose", "_original", "_jit")

    def __init__(
        self,
        model: Union[nn.Module, Pipeline],
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        Args:
            model (`torch.nn.Module` or `transformers.Pipeline`):
                The model or pipeline instance to optimize.
            dtype (`torch.dtype = torch.float32`), *optional*):
                The data type used to do the computation.
                Acceptable type are `torch.float32` (default) and `torch.bfloat16`.
                Please note `torch.bfloat16` requires `avx512_bf16` instructions set as present on
                4th Generation of Intel Xeon Scalable CPUs (Sapphire Rapids).
            jit (`boolean = False`, *optional*):
                Enable jit to accelerate inference speed
        """
        logger.warning(
            "`inference_mode` is deprecated and will be removed in v1.18.0. Use `pipeline` to load and export your model to TorchScript instead."
        )

        if not is_ipex_available():
            raise ImportError(IPEX_NOT_AVAILABLE_ERROR_MSG)

        self._model = model
        self._dtype = dtype
        self._graph_mode = False  # Let's keep for future use when it doesn't hang anymore
        self._original = None

        if "jit" in kwargs:
            logger.warning(
                "`jit` is deprecated and will be removed in a future version. Use `IPEXModel` to load and export your model to TorchScript instead."
            )
        self._jit = kwargs.pop("jit", False)

    def __enter__(self):
        if self._model.framework == "pt":
            with torch.inference_mode():
                try:
                    ipex.enable_onednn_fusion(True)

                    self._original = self._model.model if isinstance(self._model, Pipeline) else self._model
                    model = ipex.optimize(
                        self._original,
                        dtype=self._dtype,
                        graph_mode=self._graph_mode,
                        level="O1",
                        auto_kernel_selection=True,
                    )
                    if self._jit:
                        use_cache = getattr(self._original.config, "use_cache", False)
                        task = (
                            self._model.task
                            if isinstance(self._model, Pipeline)
                            else TasksManager._infer_task_from_model_or_model_class(model)
                        )
                        if task in _HEAD_TO_AUTOMODELS:
                            model = jit_trace(model, task, use_cache)
                            auto_model_class = eval(_HEAD_TO_AUTOMODELS[task])
                            model = auto_model_class(model, self._original.config, use_cache=use_cache)

                    # Enable automatic mixed precision (AMP) if we are going to target `bfloat16`
                    with torch.cpu.amp.autocast(enabled=self._dtype == torch.bfloat16):
                        if isinstance(self._model, Pipeline):
                            # Patching model with the new one
                            self._model.model = _ModelFallbackWrapper(model, self._original)
                            return self._model
                        return model

                except RuntimeError:
                    return self._model
        else:
            return self._model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = self._original
