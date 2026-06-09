# Used by test_transformations.py to dynamically select the right class per architecture

ARCH_TO_MODEL_CLASS = {
    # text-generation
    "afmoe": "OVModelForCausalLM",
    "gpt2": "OVModelForCausalLM",
    "llama": "OVModelForCausalLM",
    "mistral": "OVModelForCausalLM",
    "qwen2": "OVModelForCausalLM",
    "qwen3": "OVModelForCausalLM",
    "lfm2": "OVModelForCausalLM",
    "lfm2_moe": "OVModelForCausalLM",
    "qwen3_moe": "OVModelForCausalLM",
    "llama4": "OVModelForCausalLM",
    # image-text-to-text
    "llava": "OVModelForVisualCausalLM",
    "qwen3_5_moe": "OVModelForVisualCausalLM",
    "gemma4_moe": "OVModelForVisualCausalLM",
    # text-to-image / text-to-video
    "stable-diffusion": "OVDiffusionPipeline",
    # automatic-speech-recognition
    "whisper": "OVModelForSpeechSeq2Seq",
    # text2text-generation
    "bart": "OVModelForSeq2SeqLM",
    # feature-extraction
    "bert": "OVModelForFeatureExtraction",
    "electra": "OVModelForFeatureExtraction",
    # zero-shot-image-classification
    "clip": "OVModelForZeroShotImageClassification",
    "siglip": "OVModelForZeroShotImageClassification",
}
