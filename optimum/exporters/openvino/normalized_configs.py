from optimum.utils import NormalizedTextConfig

from .base import register_normalized_config


@register_normalized_config("chatglm")
class ChatGLM2NormalizedConfig(NormalizedTextConfig):
    NUM_LAYERS = "num_layers"
    VOCAB_SIZE = "padded_vocab_size"
