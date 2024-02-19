from intel_extension_for_pytorch.llm.modules import ApplyRotaryEmbedding, IndirectKVCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)

from .llama_functions import (
    _IPEXLlamaDecoderLayerRef,
    llama_attn_forward,
    llama_layer_norm_forward,
    llama_model_forward,
    prepare_inputs_for_generation,
)


def convert_func(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)


def convert_functions(m, target_m, new_function_name, new_function):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            convert_func(sub_m, new_function_name, new_function)
        convert_functions(sub_m, target_m, new_function_name, new_function)


def convert_class(m, target_m, new_class, config, distributed=False):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            new_m = new_class(sub_m, config, distributed)
            setattr(m, name, new_m)
        convert_class(sub_m, target_m, new_class, config, distributed)


def patch_op(m, target_m, new_op_name, new_op):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            setattr(sub_m, new_op_name, new_op)
        patch_op(sub_m, target_m, new_op_name, new_op)


def export_llama_model(model):
    ipex_rope = ApplyRotaryEmbedding(
        model.config.max_position_embeddings,
        model.config.hidden_size // model.config.num_attention_heads,
        model.config.rope_theta,
        model.config.architectures[0],
    )
    ipex_scale_dot_product = IndirectKVCache(text_max_length=model.config.max_position_embeddings)
    patch_op(model, LlamaAttention, "ipex_rope", ipex_rope)
    patch_op(model, LlamaAttention, "ipex_scale_dot_product", ipex_scale_dot_product)

    convert_func(model, "prepare_inputs_for_generation", prepare_inputs_for_generation)
    convert_functions(model, LlamaModel, "forward", llama_model_forward)
    convert_functions(model, LlamaAttention, "forward", llama_attn_forward)
    convert_functions(model, LlamaRMSNorm, "forward", llama_layer_norm_forward)

    convert_class(model, LlamaDecoderLayer, _IPEXLlamaDecoderLayerRef, model.config)
    return model


def export_model(model):
    if isinstance(model, LlamaForCausalLM):
        export_llama_model(model)
        return True

    return False
