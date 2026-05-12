import json
import tempfile
from pathlib import Path
from typing import Union

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    WhisperFeatureExtractor,
)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor


_HIDDEN: int = 64
_HEAD_DIM: int = 32
_NUM_HEADS: int = 2
_NUM_KV_HEADS: int = 2
_NUM_LAYERS: int = 2
_INTERMEDIATE: int = 128
_MROPE_SECTION: list[int] = [8, 4, 4]
_PIXEL_BOUND: int = 16 * 28 * 28

_SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_bos|>",
    "<|audio_eos|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|AUDIO|>",
    "<|IMAGE|>",
    "<|VIDEO|>",
]

_EXTRA_TOKEN_ATTRS: dict[str, str] = {
    "image_token": "<|IMAGE|>",
    "audio_token": "<|AUDIO|>",
    "video_token": "<|VIDEO|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "audio_bos_token": "<|audio_bos|>",
    "audio_eos_token": "<|audio_eos|>",
}


def _rope_scaling() -> dict[str, object]:
    return {"mrope_section": _MROPE_SECTION, "rope_type": "default"}


def _build_config() -> Qwen3OmniMoeConfig:
    # MoE thinker/talker text configs don't accept head_dim positionally, but forward it through
    # **kwargs to the PretrainedConfig base, which persists it on the config instance.
    text_kwargs = {
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": _NUM_HEADS,
        "num_key_value_heads": _NUM_KV_HEADS,
        "head_dim": _HEAD_DIM,
    }
    code_predictor_kwargs = {**text_kwargs}

    return Qwen3OmniMoeConfig(
        thinker_config={
            "text_config": {
                **text_kwargs,
                "vocab_size": 152064,
                "rope_scaling": _rope_scaling(),
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "decoder_sparse_step": 1,
                "norm_topk_prob": True,
            },
            "audio_config": {
                "d_model": _HIDDEN,
                "encoder_layers": _NUM_LAYERS,
                "encoder_attention_heads": _NUM_HEADS,
                "encoder_ffn_dim": _INTERMEDIATE,
                "num_mel_bins": 16,
                "output_dim": _HIDDEN,
                "n_window": 4,
                "n_window_infer": 16,
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
            "text_config": {
                **text_kwargs,
                "vocab_size": 256,
                "rope_scaling": _rope_scaling(),
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "decoder_sparse_step": 1,
                "norm_topk_prob": False,
            },
            "code_predictor_config": {
                **code_predictor_kwargs,
                "vocab_size": 128,
                "num_code_groups": 4,
            },
            "thinker_hidden_size": _HIDDEN,
            "num_code_groups": 4,
            "accept_hidden_layer": 1,
            "spatial_merge_size": 2,
        },
        code2wav_config={
            "hidden_size": _HIDDEN,
            "intermediate_size": _INTERMEDIATE,
            "num_hidden_layers": _NUM_LAYERS,
            "num_attention_heads": _NUM_HEADS,
            "num_key_value_heads": _NUM_KV_HEADS,
            "codebook_size": 32,
            "num_quantizers": 4,
            "decoder_dim": _HIDDEN,
            "upsample_rates": (2, 2, 2, 2),
            "upsampling_ratios": (2, 2),
            "sliding_window": 8,
        },
        enable_audio_output=True,
    )


def _build_tokenizer() -> Qwen2TokenizerFast:
    tok_obj = Tokenizer(models.BPE())
    tok_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_obj.decoder = decoders.ByteLevel()
    tok_obj.add_tokens(_SPECIAL_TOKENS + [f"tok{i}" for i in range(500)])

    # extra_special_tokens persists the multimodal token attrs through save/load via
    # _set_model_specific_special_tokens, so AutoTokenizer restores them on reload.
    return Qwen2TokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        extra_special_tokens=dict(_EXTRA_TOKEN_ATTRS),
    )


_CHAT_TEMPLATE: str = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% elif content['type'] == 'audio' %}"
    "{{ '<|audio_bos|><|AUDIO|><|audio_eos|>' }}"
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


def _build_processor() -> Qwen3OmniMoeProcessor:
    return Qwen3OmniMoeProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=_PIXEL_BOUND, max_pixels=_PIXEL_BOUND, patch_size=16),
        video_processor=Qwen2VLVideoProcessor(min_pixels=_PIXEL_BOUND, max_pixels=_PIXEL_BOUND),
        feature_extractor=WhisperFeatureExtractor(feature_size=16),
        tokenizer=_build_tokenizer(),
        chat_template=_CHAT_TEMPLATE,
    )


def _merge_image_processor_config(output_dir: Path) -> None:
    # Processor.save_pretrained makes feature_extractor overwrite image_processor in the shared
    # preprocessor_config.json; qwen3_omni_moe lacks an IMAGE_PROCESSOR_MAPPING_NAMES entry, so we
    # merge both configs and retain image_processor_type for AutoImageProcessor dispatch.
    # transformers 5.x renamed preprocessor_config.json -> processor_config.json; accept both.
    preprocessor_path = output_dir / "preprocessor_config.json"
    if not preprocessor_path.exists():
        processor_path = output_dir / "processor_config.json"
        if processor_path.exists():
            preprocessor_path.write_text(processor_path.read_text())
    with preprocessor_path.open() as f:
        feature_cfg = json.load(f)

    image_proc = Qwen2VLImageProcessor(min_pixels=_PIXEL_BOUND, max_pixels=_PIXEL_BOUND, patch_size=16)
    with tempfile.TemporaryDirectory() as scratch:
        image_proc.save_pretrained(scratch)
        with (Path(scratch) / "preprocessor_config.json").open() as f:
            image_cfg = json.load(f)

    merged = {**image_cfg, **feature_cfg, "image_processor_type": image_cfg["image_processor_type"]}
    tmp_path = preprocessor_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(merged, f, indent=2)
    tmp_path.replace(preprocessor_path)


def generate(output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    model = Qwen3OmniMoeForConditionalGeneration(_build_config())
    model.eval()
    model.save_pretrained(output_dir)
    _build_processor().save_pretrained(output_dir)
    _merge_image_processor_config(output_dir)
