from transformers.models import llama
from intel_extension_for_pytorch.llm.modules import linearAdd, linear2SiluMul, ApplyRotaryEmbedding, IndirectKVCache
from .llama_forward import llama_attn_forward


def convert_func(m, func_name, new_function):
    bound_method = new_function.__get__(m, m.__class__)
    setattr(m, func_name, bound_method)

def convert_functions(m, target_m, new_function_name, new_function):
    for _, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            convert_func(sub_m, new_function_name, new_function)
        convert_functions(sub_m, target_m, new_function_name, new_function)

def patch_op(m, target_m, new_op_name, new_op):
    for name, sub_m in m.named_children():
        if isinstance(sub_m, target_m):
            setattr(sub_m, new_op_name, new_op)
        patch_op(sub_m, target_m, new_op_name, new_op)


class LlamaModelPatcher:
    def __init__(
        self,
        model: "PreTrainedModel",
    ):
        self._model = model

        self.orig_forward_name = "forward" if hasattr(self._model, "forward") else "call"
        self.orig_forward = getattr(self._model, self.orig_forward_name)

        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    def export_model(self, model):
        ipex_rope = ApplyRotaryEmbedding(
                            model.config.max_position_embeddings,
                            model.config.hidden_size // model.config.num_attention_heads,
                            model.config.rope_theta,
                            model.config.architectures[0],
                        )
        ipex_scale_dot_product = IndirectKVCache(text_max_length=model.config.max_position_embeddings)
        patch_op(model, llama.modeling_llama.LlamaAttention, "ipex_rope", ipex_rope)
        patch_op(model, llama.modeling_llama.LlamaAttention, "ipex_scale_dot_product", ipex_scale_dot_product)

        convert_functions(model, llama.modeling_llama.LlamaAttention, "forward", llama_attn_forward)
        return model

