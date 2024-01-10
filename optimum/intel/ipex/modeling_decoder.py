import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from transformers import AutoModelForCausalLM, PretrainedConfig

from ..generation.modeling import BaseModelForCausalLM, jit_trace
from .modeling_base import IPEXModel


logger = logging.getLogger(__name__)


class IPEXModelForCausalLM(IPEXModel, BaseModelForCausalLM):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    forward = BaseModelForCausalLM.forward
    generate = BaseModelForCausalLM.generate
    can_generate = BaseModelForCausalLM.can_generate

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        IPEXModel.__init__(self, model, config)
        BaseModelForCausalLM.__init__(self, model, config, model_save_dir, use_cache, **kwargs)

    @classmethod
    def apply_jit_optimize(cls, model, task, use_cache, support_ipex_transformers):
        if not support_ipex_transformers:
            return jit_trace(model, task, use_cache)
        else:
            # from intel_extension_for_pytorch.transformers.optimize import get_dummy_input
            # dummy_jit_inputs = get_dummy_input(task, model) # From ipex
            # model = torch.jit.trace(model, example_input_kwargs=dummy_jit_inputs)
            return model
