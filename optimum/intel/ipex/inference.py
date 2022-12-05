import intel_extension_for_pytorch as ipex
import torch
from torch import nn
from transformers.pipelines import Pipeline
from typing import Union


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


class inference_mode:
    __slots__ = ("_model", "_dtype", "_graph_mode", "_verbose", "_original")

    def __init__(
        self,
        model: Union[nn.Module, Pipeline],
        dtype: torch.dtype = torch.float32,
        verbose: bool = False
    ):
        self._model = model
        self._verbose = ipex.utils.verbose.VERBOSE_ON if verbose else ipex.utils.verbose.VERBOSE_OFF
        self._dtype = dtype
        self._graph_mode = False  # Let's keep for future use when it doesn't hang anymore
        self._original = None

    def __enter__(self):

        with torch.inference_mode():
            with ipex.verbose(self._verbose):
                ipex.enable_onednn_fusion(True)
                if isinstance(self._model, Pipeline):
                    self._original = self._model.model

                    model = ipex.optimize(
                        self._model.model,
                        dtype=self._dtype,
                        graph_mode=self._graph_mode,
                        level="O1",
                        auto_kernel_selection=True
                    )

                    # Patching model with the new one
                    self._model.model = _ModelFallbackWrapper(model, self._original)
                    return self._model
                else:
                    self._original = self._model
                    model = ipex.optimize(
                        self._model,
                        dtype=self._dtype,
                        graph_mode=self._graph_mode,
                        level="O1",
                        auto_kernel_selection=True
                    )

                    return model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = self._original
