from typing import Union

import torch
from torch import nn
from transformers import add_start_docstrings
from transformers.pipelines import Pipeline
from transformers.utils import is_ipex_available


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
    __slots__ = ("_model", "_dtype", "_graph_mode", "_verbose", "_original")

    def __init__(self, model: Union[nn.Module, Pipeline], dtype: torch.dtype = torch.float32, verbose: bool = False):
        """
        Args:
            model (`torch.nn.Module` or `transformers.Pipeline`):
                The model or pipeline instance to optimize.
            dtype (`torch.dtype = torch.float32`), *optional*):
                The data type used to do the computation.
                Acceptable type are `torch.float32` (default) and `torch.bfloat16`.
                Please note `torch.bfloat16` requires `avx512_bf16` instructions set as present on
                4th Generation of Intel Xeon Scalable CPUs (Sapphire Rapids).
            verbose (`boolean = False`, *optional*):
                Enable IPEx verbose output to see the kernels and optimizations applied.
        """
        if not is_ipex_available():
            raise ImportError(IPEX_NOT_AVAILABLE_ERROR_MSG)

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
                        auto_kernel_selection=True,
                    )

                    # Enable automatic mixed precision (AMP) if we are going to target `bfloat16`
                    with torch.cpu.amp.autocast(enabled=(self._dtype == torch.bfloat16)):
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
                        auto_kernel_selection=True,
                    )

                    # Enable automatic mixed precision (AMP) if we are going to target `bfloat16`
                    with torch.cpu.amp.autocast(enabled=(self._dtype == torch.bfloat16)):
                        return model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = self._original
