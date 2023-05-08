import inspect
import logging
from typing import Tuple, Union

import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel, add_start_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.pipelines import Pipeline
from transformers.utils import is_ipex_available

from optimum.exporters.tasks import TasksManager

from ..utils.constant import _TASK_ALIASES


logger = logging.getLogger(__name__)

IPEX_NOT_AVAILABLE_ERROR_MSG = (
    "Intel PyTorch Extensions was not found."
    "please make sure you've installed the package or run "
    "pip install intel_extension_for_pytorch"
)

if is_ipex_available():
    import intel_extension_for_pytorch as ipex


def ordered_inputs(inputs: str, model: PreTrainedModel):
    """
    Order input dict and convert input dict to tuple since jit traced model only support tuple input.
    """
    if hasattr(model, "forward"):
        sig = inspect.signature(model.forward)
    else:
        sig = inspect.signature(model.call)

    return tuple(
        inputs[key]
        for key in sig.parameters
        if inputs.get(key, None) is not None and not isinstance(inputs.get(key, None), bool)
    )


def prepare_jit_inputs(model: PreTrainedModel, task: str):
    """
    Prepare tuple inputs for jit trace model
    """
    task = _TASK_ALIASES.get(task, task)
    if "generation" in task and hasattr(model.config, "use_cache") and model.config.use_cache:
        task += "-with-past"
    onnx_config_class = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model=model,
        task=task,
    )
    onnx_config = onnx_config_class(model.config)
    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
    inputs = ordered_inputs(dummy_inputs, model)

    return inputs


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


class _ModelGenerationWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        try:
            trace_graph_inputs = ordered_inputs(kwargs, self._default)
            if args:
                trace_graph_inputs = args + trace_graph_inputs
            trace_graph_inputs = tuple(trace_graph_inputs)
            outputs = self._optimized(*trace_graph_inputs)
            lm_logits = outputs[0]
            past_key_values = outputs[1]
            fixed_output = CausalLMOutputWithPast(
                loss=None,
                logits=lm_logits,
                past_key_values=past_key_values,
                hidden_states=None,
                attentions=None,
            )
            return fixed_output
        except Exception:
            return self._default(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)


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
        verbose: bool = False,
        jit: bool = False,
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
        self._jit = jit

    def __enter__(self):
        if self._model.framework == "pt":
            with torch.inference_mode():
                try:
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
                            with torch.cpu.amp.autocast(enabled=(self._dtype == torch.bfloat16)), torch.no_grad():
                                if self._model.tokenizer is not None and self._jit:
                                    try:
                                        jit_inputs = prepare_jit_inputs(self._model.model, self._model.task)
                                        model = torch.jit.trace(model, jit_inputs, strict=False)
                                        model = torch.jit.freeze(model)
                                        model(*jit_inputs)
                                        model(*jit_inputs)
                                    except Exception as e:
                                        logger.warning(f"failed to use PyTorch jit mode due to: {e}.")
                                # Patching model with the new one
                                if self._model.task == "text-generation":
                                    self._model.model = _ModelGenerationWrapper(model, self._original)
                                else:
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
                except RuntimeError:
                    return self._model
        else:
            return self._model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = self._original
