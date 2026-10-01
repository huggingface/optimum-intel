from pathlib import Path
from typing import Union

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLVideoProcessor,
    Qwen3OmniForConditionalGeneration,
    WhisperFeatureExtractor,
)
from transformers.models.qwen3_omni.configuration_qwen3_omni import Qwen3OmniConfig
from transformers.models.qwen3_omni.processing_qwen3_omni import Qwen3OmniProcessor


_HIDDEN: int = 64
_HEAD_DIM: int = 32
_NUM_HEADS: int = 2
_NUM_KV_HEADS: int = 2
_NUM_LAYERS: int = 2
_INTERMEDIATE: int = 128
_MROPE_SECTION: list[int] = [8, 4, 4]
_PIXEL_BOUND: int = 16 * 28 * 28
# The processor overrides the saved video size with these at call time, so saving the same bounds
# keeps consumers that read processor_config.json (OpenVINO GenAI) on the same frame grid.
_VIDEO_MIN_PIXELS: int = 128 * 32 * 32
_VIDEO_MAX_PIXELS: int = 768 * 32 * 32
# Codec special ids default to 4196-4205 and must fall inside the talker vocab.
_TALKER_VOCAB: int = 5120

_SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
]

# The talker finds each chat turn by the role token right after <|im_start|>, so every role must
# encode to a single token. They stay non-special so they survive skip_special_tokens decoding.
_ROLE_TOKENS: list[str] = ["system", "user", "assistant"]

_EXTRA_TOKEN_ATTRS: dict[str, str] = {
    "image_token": "<|image_pad|>",
    "audio_token": "<|audio_pad|>",
    "video_token": "<|video_pad|>",
    "vision_bos_token": "<|vision_start|>",
    "vision_eos_token": "<|vision_end|>",
    "audio_bos_token": "<|audio_start|>",
    "audio_eos_token": "<|audio_end|>",
}


def _rope_scaling() -> dict[str, object]:
    return {"mrope_section": _MROPE_SECTION, "rope_type": "default"}


def _build_config(tokenizer: Qwen2TokenizerFast) -> Qwen3OmniConfig:
    text_kwargs = {
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": _NUM_HEADS,
        "num_key_value_heads": _NUM_KV_HEADS,
        "head_dim": _HEAD_DIM,
    }
    token_id = tokenizer.convert_tokens_to_ids
    # The talker runs get_rope_index on thinker-vocab ids, so it needs the same multimodal ids.
    multimodal_ids = {
        "audio_token_id": token_id("<|audio_pad|>"),
        "image_token_id": token_id("<|image_pad|>"),
        "video_token_id": token_id("<|video_pad|>"),
        "audio_start_token_id": token_id("<|audio_start|>"),
        "vision_start_token_id": token_id("<|vision_start|>"),
        # The transformers default of 25 is a Qwen2.5-Omni leftover; the real checkpoint uses 13.
        "position_id_per_seconds": 13,
    }

    return Qwen3OmniConfig(
        thinker_config={
            **multimodal_ids,
            "vision_end_token_id": token_id("<|vision_end|>"),
            "user_token_id": token_id("user"),
            "text_config": {
                **text_kwargs,
                # Sized to the tokenizer so every generated id decodes to text.
                "vocab_size": len(tokenizer),
                "rope_scaling": _rope_scaling(),
            },
            "audio_config": {
                "d_model": _HIDDEN,
                "encoder_layers": _NUM_LAYERS,
                "encoder_attention_heads": _NUM_HEADS,
                "encoder_ffn_dim": _INTERMEDIATE,
                "num_mel_bins": 16,
                "output_dim": _HIDDEN,
                # The processor sizes audio placeholders for 100-frame chunks (n_window=50) whatever the
                # config says, so any other value leaves the encoder and the prompt disagreeing.
                "n_window": 50,
                "n_window_infer": 800,
                "conv_chunksize": 10,
                "downsample_hidden_size": 32,
            },
            "vision_config": {
                "hidden_size": _HIDDEN,
                "depth": _NUM_LAYERS,
                "num_heads": _NUM_HEADS,
                "intermediate_size": _INTERMEDIATE,
                "out_hidden_size": _HIDDEN,
                "deepstack_visual_indexes": [0],
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
            },
        },
        talker_config={
            **multimodal_ids,
            "text_config": {
                **text_kwargs,
                "vocab_size": _TALKER_VOCAB,
                "rope_scaling": _rope_scaling(),
            },
            "code_predictor_config": {
                **text_kwargs,
                "vocab_size": 128,
                "num_code_groups": 4,
            },
            "thinker_hidden_size": _HIDDEN,
            "num_code_groups": 4,
            "accept_hidden_layer": 1,
            "spatial_merge_size": 2,
            # Native generate does `speaker_id.get(speaker.lower())` (default speaker "Ethan"), and
            # OpenVINO GenAI disables speech without speakers, so this must be a non-empty dict.
            "speaker_id": {"chelsie": 1, "ethan": 2, "aiden": 3},
        },
        code2wav_config={
            "hidden_size": _HIDDEN,
            "intermediate_size": _INTERMEDIATE,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": _NUM_HEADS,
            "num_key_value_heads": _NUM_KV_HEADS,
            "codebook_size": _TALKER_VOCAB,
            "num_quantizers": 4,
            "decoder_dim": _HIDDEN,
            "upsample_rates": (2, 2, 2, 2),
            "upsampling_ratios": (2, 2),
            "sliding_window": 8,
        },
        enable_audio_output=True,
        im_start_token_id=token_id("<|im_start|>"),
        im_end_token_id=token_id("<|im_end|>"),
        system_token_id=token_id("system"),
        user_token_id=token_id("user"),
        assistant_token_id=token_id("assistant"),
        tts_pad_token_id=token_id("<tts_pad>"),
        tts_bos_token_id=token_id("<tts_text_bos>"),
        tts_eos_token_id=token_id("<tts_text_eod>"),
    )


def _build_tokenizer() -> Qwen2TokenizerFast:
    # An empty BPE vocab makes openvino_tokenizers fail with IndexError on vocab[0], so seed it with
    # the byte-level base vocab.
    base_vocab = {token: idx for idx, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tok_obj = Tokenizer(models.BPE(vocab=base_vocab, merges=[]))
    tok_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_obj.decoder = decoders.ByteLevel()
    tok_obj.add_special_tokens([AddedToken(token, special=True) for token in _SPECIAL_TOKENS])
    tok_obj.add_tokens(_ROLE_TOKENS + [f"tok{i}" for i in range(500)])

    # extra_special_tokens registers the modality attributes so they survive a save/reload round-trip;
    # plain setattr does not.
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        extra_special_tokens=_EXTRA_TOKEN_ATTRS,
    )
    for token in _SPECIAL_TOKENS + _ROLE_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1, f"{token!r} must encode to one token, got {ids}"
    return tokenizer


_CHAT_TEMPLATE: str = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% elif content['type'] == 'video' %}"
    "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
    "{% elif content['type'] == 'audio' %}"
    "{{ '<|audio_start|><|audio_pad|><|audio_end|>' }}"
    "{% elif content['type'] == 'text' %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def _build_processor(tokenizer: Qwen2TokenizerFast) -> Qwen3OmniProcessor:
    # patch_size, merge_size and temporal_patch_size must match vision_config, or the vision tower
    # receives patches of the wrong width.
    return Qwen3OmniProcessor(
        image_processor=Qwen2VLImageProcessor(
            min_pixels=_PIXEL_BOUND, max_pixels=_PIXEL_BOUND, patch_size=16, merge_size=2, temporal_patch_size=2
        ),
        video_processor=Qwen2VLVideoProcessor(
            min_pixels=_VIDEO_MIN_PIXELS,
            max_pixels=_VIDEO_MAX_PIXELS,
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
        ),
        feature_extractor=WhisperFeatureExtractor(feature_size=16),
        tokenizer=tokenizer,
        chat_template=_CHAT_TEMPLATE,
    )


def generate(output_dir: Union[str, Path]) -> None:
    tokenizer = _build_tokenizer()
    model = Qwen3OmniForConditionalGeneration(_build_config(tokenizer))
    model.eval()
    model.save_pretrained(output_dir)
    # legacy_serialization=False writes a unified processor_config.json. The legacy path writes each
    # sub-processor to preprocessor_config.json, where the feature extractor overwrites the image
    # processor.
    _build_processor(tokenizer).save_pretrained(output_dir, legacy_serialization=False)
