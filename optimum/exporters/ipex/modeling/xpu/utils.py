
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.ModuleReplacer import ModuleReplacer

def update_patcher_info_on_xpu(patcher, model_name):
    patch_info = ModuleReplacer.get_patch_info_from_model(model_name)
    op_patch_list, function_convert_list, class_convert_list = patch_info
    patcher.update_op_list(op_patch_list)
    patcher.update_function_convert_list(function_convert_list)
    patcher.update_class_convert_list(class_convert_list)