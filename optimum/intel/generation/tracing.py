import inspect

import torch
from transformers import PreTrainedModel

from optimum.exporters import TasksManager


def prepare_jit_inputs(model: PreTrainedModel, task: str, use_cache: bool = True):
    signature = inspect.signature(model.forward) if hasattr(model, "forward") else inspect.signature(model.call)
    onnx_config_class = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
    onnx_config = onnx_config_class(model.config, use_past=use_cache)
    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
    model_inputs = {key: dummy_inputs[key] for key in signature.parameters if dummy_inputs.get(key, None) is not None}
    return model_inputs


def jit_trace(model: PreTrainedModel, task: str, use_cache: bool = True):
    model.config.return_dict = False
    model_inputs = prepare_jit_inputs(model, task, use_cache)
    if use_cache:
        traced_model = torch.jit.trace(model, example_inputs=tuple(model_inputs.values()))
    else:
        traced_model = torch.jit.trace(model, example_kwarg_inputs=model_inputs)
    traced_model = torch.jit.freeze(traced_model.eval())
    traced_model(**model_inputs)
    traced_model(**model_inputs)

    return traced_model
