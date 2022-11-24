import intel_extension_for_pytorch as ipex
import torch
from torch import nn
from transformers.pipelines import Pipeline
from typing import Union


class inference_mode:
    __slots__ = ("_model", "_verbose", "_use_onednn", "_original")

    def __init__(self, model: Union[nn.Module, Pipeline], verbose: bool = False, use_onednn: bool = True):
        self._model = model
        self._verbose = ipex.utils.verbose.VERBOSE_ON if verbose else ipex.utils.verbose.VERBOSE_OFF
        self._use_onednn = use_onednn
        self._original = None

    def __enter__(self):

        with torch.inference_mode():
            with ipex.verbose(self._verbose):
                ipex.enable_onednn_fusion(True)
                if isinstance(self._model, Pipeline):
                    self._original = self._model.model
                    model = ipex.optimize(
                        self._model.model.to(memory_format=torch.channels_last),
                        dtype=torch.float32,
                        level="O1",
                        auto_kernel_selection=True
                    )
                    self._model.model = model  # Patching model with the new one
                    return self._model
                else:
                    self._original = self._model
                    model = ipex.optimize(
                        self._model.to(memory_format=torch.channels_last),
                        dtype=torch.float32,
                        level="O1",
                        auto_kernel_selection=True
                    )
                    return model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = self._original
