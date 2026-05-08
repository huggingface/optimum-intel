#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import importlib.util
import json
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import openvino as ov
import torch

from optimum.exporters.tasks import TasksManager
from optimum.intel.utils.import_utils import is_transformers_version


def _create_tiny_kokoro_model():
    """Generate a tiny random Kokoro TTS model for testing and return its local path.

    Falls back to the original Hub id if the `kokoro` package is not installed.
    Result is cached on disk under the system temp dir, so subsequent calls are cheap.
    """
    output_dir = Path(tempfile.gettempdir()) / "optimum_intel_tiny_random_kokoro"
    config_file = output_dir / "config.json"
    weights_file = output_dir / "tiny-kokoro-random.pth"
    voice_file = output_dir / "voices" / "tiny_voice.pt"
    if config_file.exists() and weights_file.exists() and voice_file.exists():
        return str(output_dir)

    from transformers import AlbertConfig

    from kokoro.istftnet import Decoder
    from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder

    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = (
        ";:,.!?-—()'\"/ "
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "əɚɝɪʊʌæɑɔɛɜɒɹɾθðŋʃʒʤʧˈˌ"
        "àáâãäåèéêëìíîïòóôõöùúûüýÿ"
    )
    deduped = []
    seen = set()
    for ch in symbols:
        if ch not in seen:
            deduped.append(ch)
            seen.add(ch)
        if len(deduped) >= 177:
            break
    vocab = {ch: i + 1 for i, ch in enumerate(deduped)}

    config = {
        "model_type": "kokoro",
        "export_model_type": "kokoro",
        "hidden_dim": 512,
        "style_dim": 128,
        "n_token": 178,
        "n_layer": 1,
        "dim_in": 512,
        "n_mels": 80,
        "max_dur": 50,
        "dropout": 0.2,
        "text_encoder_kernel_size": 3,
        "plbert": {
            "hidden_size": 128,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "num_hidden_layers": 2,
            "dropout": 0.1,
        },
        "istftnet": {
            "upsample_kernel_sizes": [20, 12],
            "upsample_rates": [10, 6],
            "gen_istft_hop_size": 5,
            "gen_istft_n_fft": 20,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_initial_channel": 512,
        },
        "vocab": vocab,
        "multispeaker": True,
        "max_conv_dim": 512,
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
    bert_encoder = torch.nn.Linear(config["plbert"]["hidden_size"], config["hidden_dim"])
    predictor = ProsodyPredictor(
        style_dim=config["style_dim"],
        d_hid=config["hidden_dim"],
        nlayers=config["n_layer"],
        max_dur=config["max_dur"],
        dropout=config["dropout"],
    )
    text_encoder = TextEncoder(
        channels=config["hidden_dim"],
        kernel_size=config["text_encoder_kernel_size"],
        depth=config["n_layer"],
        n_symbols=config["n_token"],
    )
    decoder = Decoder(
        dim_in=config["hidden_dim"],
        style_dim=config["style_dim"],
        dim_out=config["n_mels"],
        **config["istftnet"],
    )

    torch.save(
        {
            "bert": bert.state_dict(),
            "bert_encoder": bert_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "decoder": decoder.state_dict(),
        },
        weights_file,
    )

    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(256, dtype=torch.float32), voice_file)

    return str(output_dir)


SEED = 42

F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}

TENSOR_ALIAS_TO_TYPE = {"pt": torch.Tensor, "np": np.ndarray}

OPENVINO_DEVICE = os.getenv("OPENVINO_TEST_DEVICE", "CPU")

MODEL_NAMES = {
    "afmoe": "optimum-intel-internal-testing/tiny-random-trinity",
    "albert": "optimum-intel-internal-testing/tiny-random-albert",
    "aquila": "optimum-intel-internal-testing/tiny-random-aquilachat",
    "aquila2": "optimum-intel-internal-testing/tiny-random-aquila2",
    "arcee": "optimum-intel-internal-testing/tiny-random-ArceeForCausalLM",
    "arctic": "optimum-intel-internal-testing/tiny-random-snowflake",
    "audio-spectrogram-transformer": "optimum-intel-internal-testing/tiny-random-ast",
    "bge": "optimum-intel-internal-testing/bge-small-en-v1.5",
    "beit": "optimum-intel-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "optimum-intel-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-BartModel",
    "baichuan2": "optimum-intel-internal-testing/tiny-random-baichuan2",
    "baichuan2-13b": "optimum-intel-internal-testing/tiny-random-baichuan2-13b",
    "bigbird_pegasus": "optimum-intel-internal-testing/tiny-random-bigbird_pegasus",
    "biogpt": "optimum-intel-internal-testing/tiny-random-BioGptForCausalLM",
    "bitnet": "optimum-intel-internal-testing/tiny-random-bitnet",
    "blenderbot-small": "optimum-intel-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "optimum-intel-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "optimum-intel-internal-testing/tiny-random-BloomModel",
    "camembert": "optimum-intel-internal-testing/tiny-random-camembert",
    "clip": "optimum-intel-internal-testing/tiny-random-CLIPModel",
    "convbert": "optimum-intel-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "cohere": "optimum-intel-internal-testing/tiny-random-CohereForCausalLM",
    "cohere2": "optimum-intel-internal-testing/tiny-random-aya-base",
    "chatglm": "optimum-intel-internal-testing/tiny-random-chatglm",
    "chatglm4": "optimum-intel-internal-testing/tiny-random-chatglm4",
    "codegen": "optimum-intel-internal-testing/tiny-random-CodeGenForCausalLM",
    "codegen2": "optimum-intel-internal-testing/tiny-random-codegen2",
    "data2vec-text": "optimum-intel-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec-vision": "optimum-intel-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec-audio": "optimum-intel-internal-testing/tiny-random-Data2VecAudioModel",
    "dbrx": "optimum-intel-internal-testing/tiny-random-dbrx",
    "deberta": "optimum-intel-internal-testing/tiny-random-deberta",
    "deberta-v2": "optimum-intel-internal-testing/tiny-random-DebertaV2Model",
    "decilm": "optimum-intel-internal-testing/tiny-random-decilm",
    "deepseek": "optimum-intel-internal-testing/tiny-random-deepseek-v3",
    "deit": "optimum-intel-internal-testing/tiny-random-DeiTModel",
    "convnext": "optimum-intel-internal-testing/tiny-random-convnext",
    "convnextv2": "optimum-intel-internal-testing/tiny-random-ConvNextV2Model",
    "distilbert": "optimum-intel-internal-testing/tiny-random-distilbert",
    "distilbert-ov": "optimum-intel-internal-testing/ov-tiny-random-distilbert",
    "donut": "optimum-internal-testing/tiny-random-VisionEncoderDecoderModel-donut",
    "donut-swin": "optimum-intel-internal-testing/tiny-random-DonutSwinModel",
    "detr": "optimum-intel-internal-testing/tiny-random-DetrModel",
    "electra": "optimum-intel-internal-testing/tiny-random-electra",
    "esm": "optimum-intel-internal-testing/tiny-random-EsmModel",
    "exaone": "optimum-intel-internal-testing/tiny-random-exaone",
    "exaone4": "optimum-intel-internal-testing/tiny-random-exaone4",
    "gemma": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    "gemma2": "optimum-intel-internal-testing/tiny-random-gemma2",
    "got_ocr2": "optimum-intel-internal-testing/tiny-random-got-ocr2-hf",
    "gemma3_text": "optimum-intel-internal-testing/tiny-random-gemma3-text",
    "gemma3": "optimum-intel-internal-testing/tiny-random-gemma3",
    "gemma4": "optimum-intel-internal-testing/tiny-random-gemma4",
    "gemma4_moe": "optimum-intel-internal-testing/tiny-random-gemma4-moe",
    "falcon": "optimum-intel-internal-testing/really-tiny-falcon-testing",
    "falcon-40b": "optimum-intel-internal-testing/tiny-random-falcon-40b",
    "falcon_mamba": "optimum-intel-internal-testing/tiny-falcon-mamba",
    "flaubert": "optimum-intel-internal-testing/tiny-random-flaubert",
    "flux": "optimum-intel-internal-testing/tiny-random-flux",
    "flux-fill": "optimum-intel-internal-testing/tiny-random-flux-fill",
    "gpt_bigcode": "optimum-intel-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "optimum-intel-internal-testing/tiny-random-gpt2",
    "gpt2-with-cache-ov": "optimum-intel-internal-testing/ov-tiny-random-gpt2-with-cache",
    "gpt2-without-cache-ov": "optimum-intel-internal-testing/ov-tiny-random-gpt2-without-cache",
    "gpt_neo": "optimum-intel-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "optimum-intel-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gpt_neox_japanese": "optimum-intel-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM",
    "gpt_oss": "optimum-intel-internal-testing/tiny-GptOssForCausalLM",
    "gpt_oss_mxfp4": "optimum-intel-internal-testing/tiny-random-gpt-oss-mxfp4",
    "gptj": "optimum-intel-internal-testing/tiny-random-GPTJModel",
    "granite": "optimum-intel-internal-testing/tiny-random-granite",
    "granitemoe": "optimum-intel-internal-testing/tiny-random-granite-moe",
    "granitemoehybrid": "optimum-intel-internal-testing/tiny-random-granitemoehybrid",
    "hubert": "optimum-intel-internal-testing/tiny-random-HubertModel",
    "hunyuan_v1_dense": "optimum-intel-internal-testing/tiny-random-hunyuan-v1-dense",
    "ibert": "optimum-intel-internal-testing/tiny-random-ibert",
    "idefics3": "optimum-intel-internal-testing/tiny-random-Idefics3ForConditionalGeneration",
    "internlm": "optimum-intel-internal-testing/tiny-random-internlm",
    "internlm2": "optimum-intel-internal-testing/tiny-random-internlm2",
    "internvl_chat": "optimum-intel-internal-testing/tiny-random-internvl2",
    "jais": "optimum-intel-internal-testing/tiny-random-jais",
    "kokoro": _create_tiny_kokoro_model(),
    "levit": "optimum-intel-internal-testing/tiny-random-LevitModel",
    "lfm2": "optimum-intel-internal-testing/tiny-random-lfm2",
    "lfm2_moe": "optimum-intel-internal-testing/tiny-random-lfm2-moe",
    "longt5": "optimum-intel-internal-testing/tiny-random-longt5",
    "llama": "optimum-intel-internal-testing/tiny-random-LlamaForCausalLM",
    "llama_awq": "optimum-intel-internal-testing/tiny-random-LlamaForCausalLM",
    "llama4": "optimum-intel-internal-testing/tiny-random-llama4",
    "llava": "optimum-intel-internal-testing/tiny-random-llava",
    "llava_next": "optimum-intel-internal-testing/tiny-random-llava-next",
    "llava_next_mistral": "optimum-intel-internal-testing/tiny-random-llava-next-mistral",
    "llava_next_video": "optimum-intel-internal-testing/tiny-random-llava-next-video",
    "m2m_100": "optimum-intel-internal-testing/tiny-random-m2m_100",
    "olmo2": "optimum-intel-internal-testing/tiny-random-olmo2",
    "opt": "optimum-intel-internal-testing/tiny-random-OPTModel",
    "opt125m": "optimum-intel-internal-testing/opt-125m",
    "opt_gptq": "optimum-intel-internal-testing/opt-125m-gptq-4bit",
    "maira2": "optimum-intel-internal-testing/tiny-random-maira2",
    "mamba": "optimum-intel-internal-testing/tiny-mamba",
    "marian": "optimum-intel-internal-testing/tiny-marian-en-de",
    "mbart": "optimum-intel-internal-testing/tiny-random-mbart",
    "minicpm": "optimum-intel-internal-testing/tiny-random-minicpm",
    "minicpm3": "optimum-intel-internal-testing/tiny-random-minicpm3",
    "minicpmv": "optimum-intel-internal-testing/tiny-random-minicpmv-2_6",
    "minicpmo": "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6",
    "mistral": "optimum-intel-internal-testing/tiny-random-mistral",
    "mistral-nemo": "optimum-intel-internal-testing/tiny-random-mistral-nemo",
    "mixtral": "optimum-intel-internal-testing/tiny-mixtral",
    "mixtral_awq": "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit",
    "mobilebert": "optimum-intel-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "optimum-intel-internal-testing/mobilenet_v1_0.75_192",
    "mobilenet_v2": "optimum-intel-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "optimum-intel-internal-testing/tiny-random-mobilevit",
    "mpt": "optimum-intel-internal-testing/tiny-random-MptForCausalLM",
    "mpnet": "optimum-intel-internal-testing/tiny-random-MPNetModel",
    "mt5": "optimum-intel-internal-testing/mt5-tiny-random",
    "llava-qwen2": "optimum-intel-internal-testing/tiny-random-nanollava",
    "nanollava_vision_tower": "optimum-intel-internal-testing/tiny-random-siglip",
    "nystromformer": "optimum-intel-internal-testing/tiny-random-NystromformerModel",
    "olmo": "optimum-intel-internal-testing/tiny-random-olmo-hf",
    "orion": "optimum-intel-internal-testing/tiny-random-orion",
    "pegasus": "optimum-intel-internal-testing/tiny-random-pegasus",
    "perceiver_text": "optimum-intel-internal-testing/tiny-random-language_perceiver",
    "perceiver_vision": "optimum-intel-internal-testing/tiny-random-vision_perceiver_conv",
    "persimmon": "optimum-intel-internal-testing/tiny-random-PersimmonForCausalLM",
    "pix2struct": "optimum-intel-internal-testing/pix2struct-tiny-random",
    "phi": "optimum-intel-internal-testing/tiny-random-PhiForCausalLM",
    "phi3": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
    "phimoe": "optimum-intel-internal-testing/phi-3.5-moe-tiny-random",
    "phi3_v": "optimum-intel-internal-testing/tiny-random-phi3-vision",
    "phi4mm": "optimum-intel-internal-testing/tiny-random-phi-4-multimodal",
    "poolformer": "optimum-intel-internal-testing/tiny-random-PoolFormerModel",
    "qwen": "optimum-intel-internal-testing/tiny-random-qwen",
    "qwen2": "optimum-intel-internal-testing/tiny-dummy-qwen2",
    "qwen2_moe": "optimum-intel-internal-testing/tiny-random-qwen1.5-moe",
    "qwen2_vl": "optimum-intel-internal-testing/tiny-random-qwen2vl",
    "qwen2_5_vl": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
    "qwen3": "optimum-intel-internal-testing/tiny-random-qwen3",
    "qwen3_moe": "optimum-intel-internal-testing/tiny-random-qwen3moe",
    "qwen3_vl": "optimum-intel-internal-testing/tiny-random-qwen3-vl",
    "qwen3_next": "optimum-intel-internal-testing/tiny-random-qwen3-next",
    "qwen3_5": "optimum-intel-internal-testing/tiny-random-qwen3.5",
    "qwen3_5_moe": "optimum-intel-internal-testing/tiny-random-qwen3.5-moe",
    "rembert": "optimum-intel-internal-testing/tiny-random-rembert",
    "resnet": "optimum-intel-internal-testing/tiny-random-resnet",
    "roberta": "optimum-intel-internal-testing/tiny-random-roberta",
    "roformer": "optimum-intel-internal-testing/tiny-random-roformer",
    "segformer": "optimum-intel-internal-testing/tiny-random-SegformerModel",
    "sentence-transformers-bert": "optimum-intel-internal-testing/stsb-bert-tiny-safetensors",
    "sam": "optimum-intel-internal-testing/sam-vit-tiny-random",
    "smolvlm": "optimum-intel-internal-testing/tiny-random-smolvlm2",
    "speecht5": "optimum-intel-internal-testing/tiny-random-SpeechT5ForTextToSpeech",
    "speech_to_text": "optimum-intel-internal-testing/tiny-random-Speech2TextModel",
    "squeezebert": "optimum-intel-internal-testing/tiny-random-squeezebert",
    "stable-diffusion": "optimum-intel-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-with-safety-checker": "optimum-intel-internal-testing/tiny-random-stable-diffusion-with-safety-checker",
    "stable-diffusion-with-custom-variant": "optimum-intel-internal-testing/tiny-stable-diffusion-torch-custom-variant",
    "stable-diffusion-with-textual-inversion": "optimum-intel-internal-testing/tiny-stable-diffusion-with-textual-inversion",
    "stable-diffusion-openvino": "optimum-intel-internal-testing/tiny-stable-diffusion-openvino",
    "stable-diffusion-xl": "optimum-intel-internal-testing/tiny-random-stable-diffusion-xl",
    "stable-diffusion-xl-refiner": "optimum-intel-internal-testing/tiny-random-stable-diffusion-xl-refiner",
    "stable-diffusion-3": "optimum-intel-internal-testing/stable-diffusion-3-tiny-random",
    "stablelm": "optimum-intel-internal-testing/tiny-random-StableLmForCausalLM",
    "starcoder2": "optimum-intel-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "siglip": "optimum-intel-internal-testing/tiny-random-SiglipModel",
    "latent-consistency": "optimum-intel-internal-testing/tiny-random-latent-consistency",
    "sew": "optimum-intel-internal-testing/tiny-random-SEWModel",
    "sew-d": "optimum-intel-internal-testing/sew-d-tiny-100k-ft-ls100h",
    "swin": "optimum-intel-internal-testing/tiny-random-SwinModel",
    "swin-window": "optimum-intel-internal-testing/tiny-random-swin-patch4-window7-224",
    "t5": "optimum-intel-internal-testing/tiny-random-t5",
    "trocr": "optimum-intel-internal-testing/trocr-small-handwritten",
    "unispeech": "optimum-intel-internal-testing/tiny-random-unispeech",
    "unispeech-sat": "optimum-intel-internal-testing/tiny-random-UnispeechSatModel",
    "vit": "optimum-intel-internal-testing/tiny-random-vit",
    "vit-with-attentions": "optimum-intel-internal-testing/vit-with-attentions",
    "vit-with-hidden-states": "optimum-intel-internal-testing/vit-with-hidden_states",
    "vision-encoder-decoder": "optimum-intel-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
    "wavlm": "optimum-intel-internal-testing/tiny-random-WavlmModel",
    "wav2vec2": "optimum-intel-internal-testing/wav2vec2-random-tiny-classifier",
    "wav2vec2-hf": "optimum-intel-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "optimum-intel-internal-testing/tiny-random-wav2vec2-conformer",
    "whisper": "optimum-intel-internal-testing/tiny-random-whisper",
    "xlm": "optimum-intel-internal-testing/tiny-random-xlm",
    "xlm-roberta": "optimum-intel-internal-testing/tiny-random-xlm-roberta",
    "xglm": "optimum-intel-internal-testing/tiny-random-XGLMForCausalLM",
    "xverse": "optimum-intel-internal-testing/tiny-random-xverse",
    "glm4": "optimum-intel-internal-testing/tiny-random-glm4",
    "glm": "optimum-intel-internal-testing/tiny-random-glm-edge",
    "open-clip": "optimum-intel-internal-testing/tiny-open-clip-model",
    "open-clip-ov": "optimum-intel-internal-testing/tiny-open-clip-model",
    "st-bert": "optimum-intel-internal-testing/all-MiniLM-L6-v2",
    "st-mpnet": "optimum-intel-internal-testing/all-mpnet-base-v2",
    "sana": "optimum-intel-internal-testing/tiny-random-sana",
    "sana-sprint": "optimum-intel-internal-testing/tiny-random-sana-sprint",
    "ltx-video": "optimum-intel-internal-testing/tiny-random-ltx-video",
    "zamba2": "optimum-intel-internal-testing/tiny-random-zamba2",
    "qwen3_eagle3": "AngelSlim/Qwen3-1.7B_eagle3",
}

EAGLE3_MODELS = {"qwen3_eagle3": ("AngelSlim/Qwen3-1.7B_eagle3", "Qwen/Qwen3-1.7B")}

_ARCHITECTURES_TO_EXPECTED_INT8 = {
    "afmoe": {"model": 16},
    "bert": {"model": 68 if is_transformers_version("<", "5") else 70},
    "roberta": {"model": 68},
    "albert": {"model": 84},
    "vit": {"model": 64},
    "blenderbot": {"model": 70 if is_transformers_version("<", "5") else 72},
    "cohere2": {"model": 30},
    "gpt2": {"model": 44},
    "granitemoehybrid": {"model": 118},
    "wav2vec2": {"model": 34},
    "distilbert": {"model": 66},
    "t5": {
        "encoder": 64,
        "decoder": 104 if is_transformers_version("<", "5") else 106,
        "decoder_with_past": 84 if is_transformers_version("<", "5") else 86,
    },
    "stable-diffusion": {
        "unet": 242,
        "vae_decoder": 42,
        "vae_encoder": 34,
        "text_encoder": 64,
    },
    "stable-diffusion-xl": {
        "unet": 366,
        "vae_decoder": 42,
        "vae_encoder": 34,
        "text_encoder": 64,
        "text_encoder_2": 66,
    },
    "stable-diffusion-xl-refiner": {
        "unet": 366,
        "vae_decoder": 42,
        "vae_encoder": 34,
        "text_encoder_2": 66,
    },
    "open-clip": {
        "text_model": 20,
        "visual_model": 28,
    },
    "stable-diffusion-3": {
        "transformer": 66,
        "vae_decoder": 58,
        "vae_encoder": 42,
        "text_encoder": 30,
        "text_encoder_2": 30,
        "text_encoder_3": 32,
    },
    "flux": {
        "transformer": 56,
        "vae_decoder": 28,
        "vae_encoder": 24,
        "text_encoder": 64,
        "text_encoder_2": 64,
    },
    "flux-fill": {
        "transformer": 56,
        "vae_decoder": 28,
        "vae_encoder": 24,
        "text_encoder": 64,
        "text_encoder_2": 64,
    },
    "llava": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 9,
    },
    "llava_next": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 9,
    },
    "minicpmv": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 26,
        "resampler_model": 6,
    },
    "llava_next_video": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 7,
        "vision_resampler_model": 0,
        "multi_modal_projector_model": 2,
    },
    "llava-qwen2": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 15,
    },
    "qwen2_vl": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 1,
        "vision_embeddings_merger_model": 10,
    },
    "qwen3_vl": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 1,
        "vision_embeddings_merger_model": 32,
        "vision_embeddings_pos_model": 1,
    },
    "qwen3_5": {
        "lm_model": 70,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 1,
        "vision_embeddings_merger_model": 10,
        "vision_embeddings_pos_model": 1,
    },
    "qwen3_5_moe": {
        "lm_model": 110,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 1,
        "vision_embeddings_merger_model": 10,
        "vision_embeddings_pos_model": 1,
    },
    "sana": {
        "transformer": 58,
        "vae_decoder": 28,
        "vae_encoder": 28,
        "text_encoder": 16,
    },
    "ltx-video": {
        "transformer": 34,
        "vae_decoder": 28,
        "vae_encoder": 28,
        "text_encoder": 64,
    },
    "sam": {
        "vision_encoder": 150,
        "prompt_encoder_mask_decoder": 98,
    },
    "speecht5": {
        "encoder": 28,
        "decoder": 52,
        "postnet": 10,
        "vocoder": 80,
    },
    "kokoro": {"model": 352},
    "clip": {"model": 130},
    "mamba": {"model": 324 if is_transformers_version("==", "5.0") else 322},
    "falcon_mamba": {"model": 164 if is_transformers_version("==", "5.0") else 162},
    "minicpmo": {
        "lm_model": 16,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 8,
        "resampler_model": 6,
    },
    "zamba2": {"model": 44},
    "exaone4": {"model": 16},
    "lfm2": {"model": 52 if is_transformers_version("<", "5") else 54},
    "lfm2_moe": {"model": 46},
    "hunyuan_v1_dense": {"model": 32},
    "qwen3_eagle3": {"model": 20},
    "qwen3_next": {"model": 100},
    "gemma4": {
        "lm_model": 54,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 10,
        "text_embeddings_per_layer_model": 1,
    },
    "gemma4_moe": {
        "lm_model": 48,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 10,
        "text_embeddings_per_layer_model": 0,
    },
}

TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

REMOTE_CODE_MODELS = (
    "chatglm",
    "minicpm",
    "baichuan2",
    "baichuan2-13b",
    "jais",
    "qwen",
    "internlm2",
    "orion",
    "aquila",
    "aquila2",
    "xverse",
    "internlm",
    "codegen2",
    "arctic",
    "chatglm4",
    "exaone",
    "exaone4",
    "decilm",
    "minicpm3",
    "deepseek",
    "qwen3_eagle3",
)

if is_transformers_version("<", "5"):
    REMOTE_CODE_MODELS += ("afmoe",)


def get_num_quantized_nodes(model):
    num_fake_nodes = 0
    types_map = {
        "i8": "int8",
        "u8": "int8",
        "i4": "int4",
        "u4": "int4",
        "f4e2m1": "f4e2m1",
        "f8e8m0": "f8e8m0",
        "nf4": "nf4",
        "f8e4m3": "f8e4m3",
        "f8e5m2": "f8e5m2",
    }
    num_weight_nodes = dict.fromkeys(types_map.values(), 0)
    ov_model = model if isinstance(model, ov.Model) else model.model
    for elem in ov_model.get_ops():
        if "FakeQuantize" in elem.name:
            num_fake_nodes += 1
        if "FakeConvert" in elem.name:
            num_fake_nodes += 1
        for i in range(elem.get_output_size()):
            type_name = elem.get_output_element_type(i).get_type_name()
            if type_name in types_map:
                name = types_map[type_name]
                num_weight_nodes[name] += 1
    return num_fake_nodes, num_weight_nodes


@contextmanager
def mock_torch_cuda_is_available(to_patch):
    original_is_available = torch.cuda.is_available
    if to_patch:
        torch.cuda.is_available = lambda: True
    try:
        yield
    finally:
        if to_patch:
            torch.cuda.is_available = original_is_available


@contextmanager
def patch_awq_for_inference(to_patch):
    orig_gemm_forward = None
    if to_patch:
        # patch GEMM module to allow inference without CUDA GPU
        from awq.modules.linear.gemm import WQLinearMMFunction
        from awq.utils.packing_utils import dequantize_gemm

        def new_forward(
            ctx,
            x,
            qweight,
            qzeros,
            scales,
            w_bit=4,
            group_size=128,
            bias=None,
            out_features=0,
        ):
            ctx.out_features = out_features

            out_shape = x.shape[:-1] + (out_features,)
            x = x.to(torch.float16)
            out = dequantize_gemm(qweight.to(torch.int32), qzeros.to(torch.int32), scales, w_bit, group_size)
            out = torch.matmul(x, out.to(x.dtype))

            out = out + bias if bias is not None else out
            out = out.reshape(out_shape)

            if len(out.shape) == 2:
                out = out.unsqueeze(0)
            return out

        orig_gemm_forward = WQLinearMMFunction.forward
        WQLinearMMFunction.forward = new_forward
    try:
        yield
    finally:
        if orig_gemm_forward is not None:
            WQLinearMMFunction.forward = orig_gemm_forward


def check_compression_state_per_model(
    test_case: unittest.TestCase,
    models: Dict[str, ov.Model],
    expected_num_weight_nodes_per_model: Dict[str, Dict[str, int]],
    expected_num_fake_nodes_per_model: Optional[Dict[str, int]] = None,
):
    test_case.assertEqual(len(models), len(expected_num_weight_nodes_per_model))
    actual_num_weights_per_model = {}
    actual_num_fake_nodes_per_model = {}
    for ov_model_name, ov_model in models.items():
        expected_num_weight_nodes = expected_num_weight_nodes_per_model[ov_model_name]
        num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(ov_model)
        expected_num_weight_nodes.update(dict.fromkeys(set(num_weight_nodes) - set(expected_num_weight_nodes), 0))

        actual_num_weights_per_model[ov_model_name] = num_weight_nodes
        actual_num_fake_nodes_per_model[ov_model_name] = num_fake_nodes

        test_case.assertFalse(ov_model.has_rt_info(["runtime_options", "KV_CACHE_PRECISION"]))

    # Check weight nodes
    test_case.assertEqual(expected_num_weight_nodes_per_model, actual_num_weights_per_model)

    # Check fake nodes
    if expected_num_fake_nodes_per_model is not None:
        test_case.assertEqual(expected_num_fake_nodes_per_model, actual_num_fake_nodes_per_model)


def get_num_sdpa(model):
    ov_model = model if isinstance(model, ov.Model) else model.model
    num_sdpa = 0
    for op in ov_model.get_ops():
        if op.type_info.name == "ScaledDotProductAttention":
            num_sdpa += 1
    return num_sdpa


TEST_NAME_TO_MODEL_TYPE = {
    "aquila2": "aquila",
    "baichuan2": "baichuan",
    "baichuan2-13b": "baichuan",
    "chatglm4": "chatglm",
    "codegen2": "codegen",
    "falcon-40b": "falcon",
    "gpt_oss_mxfp4": "gpt_oss",
    "llama_awq": "llama",
    "llava_next_mistral": "llava_next",
    "mistral-nemo": "mistral",
    "mixtral_awq": "mixtral",
    "nanollava_vision_tower": "siglip",
    "opt125m": "opt",
    "opt_gptq": "opt",
    "perceiver_text": "perceiver",
    "perceiver_vision": "perceiver",
    "swin-window": "swin",
    "vit-with-attentions": "vit",
    "vit-with-hidden-states": "vit",
    "wav2vec2-hf": "wav2vec2",
}


def get_supported_model_for_library(library_name):
    valid_model = set()
    supported_model_type = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]

    for model_type in supported_model_type:
        if supported_model_type[model_type].get("openvino"):
            export_config = next(iter(supported_model_type[model_type]["openvino"].values()))

            min_transformers = str(getattr(export_config.func, "MIN_TRANSFORMERS_VERSION", "0"))
            max_transformers = str(getattr(export_config.func, "MAX_TRANSFORMERS_VERSION", "999"))

            if is_transformers_version(">=", min_transformers) and is_transformers_version("<=", max_transformers):
                valid_model.add(model_type)

    return valid_model


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3
