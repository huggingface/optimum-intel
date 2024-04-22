import torch
import intel_extension_for_pytorch
from typing import List
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)

from .xpu.utils import update_patcher_info_on_xpu

def update_patcher_info_on_cpu(model_name):
    pass


class _IPEXPatcher:
    def __init__(self):
        self.op_patch_list: List = []
        self.function_convert_list: List = []
        self.class_convert_list: List = []

    def update_op_list(self, op_list):
        self.op_patch_list.extend(op_list)

    def update_function_convert_list(self, function_converts):
        self.function_convert_list.extend(function_converts)

    def update_class_convert_list(self, class_converts):
        self.class_convert_list.extend(class_converts)

    def patch_op_recursive(self, model):

        def patch_op(model, target_m, new_op_name, new_op):
            for name, sub_m in model.named_children():
                if isinstance(sub_m, target_m):
                    setattr(sub_m, new_op_name, new_op)
                patch_op(sub_m, target_m, new_op_name, new_op)

        for op_patch in self.op_patch_list:
            target_m, new_op_name, new_op = op_patch
            new_op_inst = new_op(model)
            patch_op(model, target_m, new_op_name, new_op_inst)

    def convert_function_recursive(self, model):

        def convert_functions(m, target_m, new_function_name, new_function):
            for _, sub_m in m.named_children():
                if isinstance(sub_m, target_m):
                    bound_method = new_function.__get__(sub_m, sub_m.__class__)
                    setattr(m, new_function_name, bound_method)
                convert_functions(sub_m, target_m, new_function_name, new_function)

        for function_convert in self.function_convert_list:
            target_m, new_function_name, new_function = function_convert
            convert_functions(model, target_m, new_function_name, new_function)

    def convert_class_recursive(self, model):

        def convert_class(m, target_m, new_class, config, distributed=False):
            for name, sub_m in m.named_children():
                if isinstance(sub_m, target_m):
                    new_m = new_class(sub_m, config, distributed)
                    setattr(m, name, new_m)
                convert_class(sub_m, target_m, new_class, config, distributed)

        for class_convert in self.class_convert_list:
            target_m, new_class, config, distributed = class_convert
            convert_class(model, target_m, new_class, config, distributed)

    def retrive_patch_info(self, model_name, device):
        if device.device_type == "xpu":
            update_patcher_info_on_xpu(model_name)
        elif device.device_type == "cpu":
            update_patcher_info_on_cpu(model_name)
        else:
            raise RuntimeError(f"Optimum-intel only support CPU and XPU device optimization. But we find this model on {device}.")

    def patch_model(self, model):
        # if isinstance(model, LlamaForCausalLM):
        self.retrive_patch_info(model.__class__.name, model.device)
        self.patch_op_recursive(model)
        self.convert_function_recursive(model)
        self.convert_class_recursive(model)

