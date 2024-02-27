from intel_extension_for_pytorch.llm.modules import (
    ApplyRotaryEmbedding,
    IndirectAccessKVCache,
    linear2SiluMul,
    linearAdd,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaRMSNorm,
)

from .llama_functions import (
    llama_attn_forward,
    llama_decoder_layer_forward,
    llama_layer_norm_forward,
    llama_model_forward,
)


def convert_func(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)


def convert_functions(m, target_m, new_function_name, new_function):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            convert_func(sub_m, new_function_name, new_function)
        convert_functions(sub_m, target_m, new_function_name, new_function)


def patch_op(m, target_m, op_name, op):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            setattr(sub_m, op_name, op)
        patch_op(sub_m, target_m, op_name, op)


def unpatch_op(m, target_m, op_name):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            delattr(sub_m, op_name)
        unpatch_op(sub_m, target_m, op_name)


def patch_linear(m, target_m, linear_name, linear_class, attr_list, attr_list_2=None, distributed=None):
    if attr_list_2:
        for name, sub_m in m.named_children():
            if isinstance(sub_m, target_m):
                attr_1 = sub_m
                attr_2 = sub_m
                for target_attr in attr_list:
                    attr_1 = getattr(attr_1, target_attr)
                for target_attr in attr_list_2:
                    attr_2 = getattr(attr_2, target_attr)
                setattr(sub_m, linear_name, linear_class(attr_1, attr_2))
            patch_linear(sub_m, target_m, linear_name, linear_class, attr_list, attr_list_2)
    else:
        if isinstance(linear_class, linearAdd) and distributed:
            return
        for name, sub_m in m.named_children():
            if isinstance(sub_m, target_m):
                attr = sub_m
                for target_attr in attr_list:
                    attr = getattr(attr, target_attr)
                setattr(sub_m, linear_name, linear_class(attr))
            patch_linear(sub_m, target_m, linear_name, linear_class, attr_list)


class ModelPatcher:
    def __init__(self, model, ipex_ops=None, ipex_functions=None, ipex_linears=None, original_functions=None):
        self.model = model
        self.ipex_ops = ipex_ops or []
        self.ipex_functions = ipex_functions or []
        self.ipex_linears = ipex_linears or []
        self.original_functions = original_functions or []

    def patch_ops(self):
        for module, op_name, op in self.ipex_ops:
            patch_op(self.model, module, op_name, op)

    def unpatch_ops(self):
        for module, op_name, op in self.ipex_ops:
            unpatch_op(self.model, module, op_name)

    def patch_functions(self):
        for module, func_name, func in self.ipex_functions:
            convert_functions(self.model, module, func_name, func)

    def unpatch_functions(self):
        for module, func_name, func in self.original_functions:
            convert_functions(self.model, module, func_name, func)

    def patch_linears(self):
        for module, linear_name, linear_class, attr_list, attr_list_2, distributed in self.ipex_linears:
            patch_linear(self.model, module, linear_name, linear_class, attr_list, attr_list_2, distributed)

    def unpatch_linears(self):
        for module, linear_name, linear_class, attr_list, attr_list_2, distributed in self.ipex_linears:
            unpatch_op(self.model, module, linear_name)

    def __enter__(self):
        self.patch_ops()
        self.patch_functions()
        self.patch_linears()
        return self.model

    def __exit__(self, *args, **kwargs):
        self.unpatch_ops()
        self.unpatch_functions()
        self.unpatch_linears()
        return self.model

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class LlamaModelPatcher(ModelPatcher):
    def __init__(self, model):
        super().__init__(model)

        ipex_rope = ApplyRotaryEmbedding(
            model.config.max_position_embeddings,
            model.config.hidden_size // model.config.num_attention_heads,
            model.config.rope_theta,
            model.config.architectures[0],
        )
        ipex_scale_dot_product = IndirectAccessKVCache(text_max_length=model.config.max_position_embeddings)
        self.ipex_ops = [
            (LlamaAttention, "ipex_rope", ipex_rope),
            (LlamaAttention, "ipex_scale_dot_product", ipex_scale_dot_product),
        ]
        self.ipex_functions = [
            (LlamaModel, "forward", llama_model_forward),
            (LlamaAttention, "forward", llama_attn_forward),
            (LlamaRMSNorm, "forward", llama_layer_norm_forward),
            (LlamaDecoderLayer, "forward", llama_decoder_layer_forward),
        ]

        self.ipex_linears = [
            (LlamaDecoderLayer, "mha_linear_add", linearAdd, ["self_attn", "o_proj"], None, None),
            (LlamaDecoderLayer, "mlp_linear_add", linearAdd, ["mlp", "down_proj"], None, None),
            (LlamaDecoderLayer, "linear_silu_mul", linear2SiluMul, ["mlp", "gate_proj"], ["mlp", "up_proj"], None),
        ]

        self.original_functions = [
            (LlamaModel, "forward", model.model.forward),
            (LlamaAttention, "forward", model.model.layers[0].self_attn.forward),
            (LlamaRMSNorm, "forward", model.model.norm.forward),
            (LlamaDecoderLayer, "forward", model.model.layers[0].forward),
        ]
