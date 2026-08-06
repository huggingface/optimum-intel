#  Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Tests for the experimental ``optimum-cli export openvino-hf`` path (Transformers-native exporter)."""

import gc
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openvino
import openvino_genai
import openvino_tokenizers  # noqa: F401 — registers the OV frontend extension so tokenizer/detokenizer IR reads
import torch
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    set_seed,
)
from utils_tests import F32_CONFIG

from optimum.exporters.openvino_hf import export_openvino_hf
from optimum.exporters.openvino_hf.export import _load_processor, _task_from_architecture
from optimum.exporters.openvino_hf.inputs import _build_sample_inputs
from optimum.intel.openvino import (
    OVModelForAudioClassification,
    OVModelForAudioFrameClassification,
    OVModelForAudioXVector,
    OVModelForCausalLM,
    OVModelForCTC,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForImageTextToText,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVModelForVisualCausalLM,
    OVModelForZeroShotImageClassification,
)


# Tiny fixtures live in one consolidated repo under `<model_type>/<ClassName>` (freshly built, correctly
# weight-tied, with a matching shrunk tokenizer/processor). `_model` resolves a test `model_type` to
# `(model_id, subfolder)` into that repo. Every tested arch must be present — a missing
# `TINY_MODEL_SUBFOLDERS` entry raises `KeyError` (fix by pushing the arch and adding it), no silent fallback.
TINY_MODELS_REPO = "IlyasMoutawwakil/tiny-random-models"

TINY_MODEL_SUBFOLDERS = {
    "afmoe": "afmoe/AfmoeForCausalLM",
    "albert": "albert/AlbertModel",
    "altclip": "altclip/AltCLIPModel",
    "apertus": "apertus/ApertusForCausalLM",
    "arcee": "arcee/ArceeForCausalLM",
    "axk1": "axk1/AXK1ForCausalLM",
    "axk2": "axk2/AXK2ForCausalLM",
    "bamba": "bamba/BambaForCausalLM",
    "bart": "bart/BartForConditionalGeneration",
    "beit": "beit/BeitForImageClassification",
    "bert": "bert/BertModel",
    "bert-generation": "bert-generation/BertGenerationDecoder",
    "big_bird": "big_bird/BigBirdForCausalLM",
    "bigbird_pegasus": "bigbird_pegasus/BigBirdPegasusForConditionalGeneration",
    "biogpt": "biogpt/BioGptForCausalLM",
    "bit": "bit/BitForImageClassification",
    "bitnet": "bitnet/BitNetForCausalLM",
    "blenderbot": "blenderbot/BlenderbotForConditionalGeneration",
    "blenderbot-small": "blenderbot-small/BlenderbotSmallForConditionalGeneration",
    "bloom": "bloom/BloomForCausalLM",
    "blt": "blt/BltForCausalLM",
    "chinese_clip": "chinese_clip/ChineseCLIPModel",
    "clip": "clip/CLIPModel",
    "clipseg": "clipseg/CLIPSegModel",
    "codegen": "codegen/CodeGenForCausalLM",
    "cohere": "cohere/CohereForCausalLM",
    "cohere2": "cohere2/Cohere2ForCausalLM",
    "cohere2_moe": "cohere2_moe/Cohere2MoeForCausalLM",
    "convbert": "convbert/ConvBertModel",
    "convnext": "convnext/ConvNextForImageClassification",
    "convnextv2": "convnextv2/ConvNextV2ForImageClassification",
    "ctrl": "ctrl/CTRLLMHeadModel",
    "cvt": "cvt/CvtForImageClassification",
    "cwm": "cwm/CwmForCausalLM",
    "data2vec-audio": "data2vec-audio/Data2VecAudioForSequenceClassification",
    "data2vec-text": "data2vec-text/Data2VecTextModel",
    "data2vec-vision": "data2vec-vision/Data2VecVisionForImageClassification",
    "deberta": "deberta/DebertaModel",
    "deberta-v2": "deberta-v2/DebertaV2Model",
    "deepseek_v2": "deepseek_v2/DeepseekV2ForCausalLM",
    "deepseek_v3": "deepseek_v3/DeepseekV3ForCausalLM",
    "deepseek_v32": "deepseek_v32/DeepseekV32ForCausalLM",
    "deepseek_v4": "deepseek_v4/DeepseekV4ForCausalLM",
    "deit": "deit/DeiTForImageClassification",
    "detr": "detr/DetrForObjectDetection",
    "diffllama": "diffllama/DiffLlamaForCausalLM",
    "dinov2": "dinov2/Dinov2ForImageClassification",
    "dinov2_with_registers": "dinov2_with_registers/Dinov2WithRegistersForImageClassification",
    "distilbert": "distilbert/DistilBertModel",
    "doge": "doge/DogeForCausalLM",
    "donut-swin": "donut-swin/DonutSwinForImageClassification",
    "dots1": "dots1/Dots1ForCausalLM",
    "efficientnet": "efficientnet/EfficientNetForImageClassification",
    "electra": "electra/ElectraModel",
    "ernie": "ernie/ErnieForCausalLM",
    "ernie4_5": "ernie4_5/Ernie4_5ForCausalLM",
    "ernie4_5_moe": "ernie4_5_moe/Ernie4_5_MoeForCausalLM",
    "esm": "esm/EsmModel",
    "eurobert": "eurobert/EuroBertForMaskedLM",
    "exaone4": "exaone4/Exaone4ForCausalLM",
    "exaone_moe": "exaone_moe/ExaoneMoeForCausalLM",
    "falcon": "falcon/FalconForCausalLM",
    "falcon_h1": "falcon_h1/FalconH1ForCausalLM",
    "falcon_mamba": "falcon_mamba/FalconMambaForCausalLM",
    "flaubert": "flaubert/FlaubertModel",
    "flex_olmo": "flex_olmo/FlexOlmoForCausalLM",
    "fnet": "fnet/FNetForMaskedLM",
    "focalnet": "focalnet/FocalNetForImageClassification",
    "fsmt": "fsmt/FSMTForConditionalGeneration",
    "funnel": "funnel/FunnelForMaskedLM",
    "gemma": "gemma/GemmaForCausalLM",
    "gemma2": "gemma2/Gemma2ForCausalLM",
    "gemma3": "gemma3/Gemma3ForConditionalGeneration",
    "gemma3_text": "gemma3_text/Gemma3ForCausalLM",
    "gemma3n": "gemma3n/Gemma3nForConditionalGeneration",
    "gemma3n_text": "gemma3n_text/Gemma3nForCausalLM",
    "gemma4": "gemma4_text/Gemma4ForCausalLM",
    "gemma4_unified": "gemma4_unified/Gemma4UnifiedForConditionalGeneration",
    "gemma4_unified_text": "gemma4_unified_text/Gemma4UnifiedForCausalLM",
    "git": "git/GitForCausalLM",
    "glm": "glm/GlmForCausalLM",
    "glm4": "glm4/Glm4ForCausalLM",
    "glm4_moe": "glm4_moe/Glm4MoeForCausalLM",
    "glm4_moe_lite": "glm4_moe_lite/Glm4MoeLiteForCausalLM",
    "glm_moe_dsa": "glm_moe_dsa/GlmMoeDsaForCausalLM",
    "got_ocr2": "got_ocr2/GotOcr2ForConditionalGeneration",
    "gpt2": "gpt2/GPT2LMHeadModel",
    "gpt_bigcode": "gpt_bigcode/GPTBigCodeForCausalLM",
    "gpt_neo": "gpt_neo/GPTNeoForCausalLM",
    "gpt_neox": "gpt_neox/GPTNeoXForCausalLM",
    "gpt_neox_japanese": "gpt_neox_japanese/GPTNeoXJapaneseForCausalLM",
    "gpt_oss": "gpt_oss/GptOssForCausalLM",
    "gptj": "gptj/GPTJForCausalLM",
    "granite": "granite/GraniteForCausalLM",
    "granite_swa": "granite_swa/GraniteSWAForCausalLM",
    "granitemoe": "granitemoe/GraniteMoeForCausalLM",
    "granitemoe_swa": "granitemoe_swa/GraniteMoeSWAForCausalLM",
    "granitemoehybrid": "granitemoehybrid/GraniteMoeHybridForCausalLM",
    "granitemoeshared": "granitemoeshared/GraniteMoeSharedForCausalLM",
    "helium": "helium/HeliumForCausalLM",
    "hgnet_v2": "hgnet_v2/HGNetV2ForImageClassification",
    "hiera": "hiera/HieraForImageClassification",
    "hrm_text": "hrm_text/HrmTextForCausalLM",
    "hubert": "hubert/HubertForSequenceClassification",
    "hunyuan_v1_dense": "hunyuan_v1_dense/HunYuanDenseV1ForCausalLM",
    "hunyuan_v1_moe": "hunyuan_v1_moe/HunYuanMoEV1ForCausalLM",
    "hy_v3": "hy_v3/HYV3ForCausalLM",
    "hyperclovax": "hyperclovax/HyperCLOVAXForCausalLM",
    "ibert": "ibert/IBertModel",
    "idefics3": "idefics3/Idefics3ForConditionalGeneration",
    "ijepa": "ijepa/IJepaForImageClassification",
    "jais2": "jais2/Jais2ForCausalLM",
    "jamba": "jamba/JambaForCausalLM",
    "jetmoe": "jetmoe/JetMoeForCausalLM",
    "jina_embeddings_v3": "jina_embeddings_v3/JinaEmbeddingsV3ForMaskedLM",
    "laguna": "laguna/LagunaForCausalLM",
    "layoutlm": "layoutlm/LayoutLMForMaskedLM",
    "levit": "levit/LevitForImageClassification",
    "lfm2": "lfm2/Lfm2ForCausalLM",
    "lfm2_moe": "lfm2_moe/Lfm2MoeForCausalLM",
    "llama": "llama/LlamaForCausalLM",
    "llava": "llava/LlavaForConditionalGeneration",
    "llava_next": "llava_next/LlavaNextForConditionalGeneration",
    "longt5": "longt5/LongT5ForConditionalGeneration",
    "luke": "luke/LukeForMaskedLM",
    "m2m_100": "m2m_100/M2M100ForConditionalGeneration",
    "mamba": "mamba/MambaForCausalLM",
    "mamba2": "mamba2/Mamba2ForCausalLM",
    "marian": "marian/MarianMTModel",
    "mbart": "mbart/MBartForConditionalGeneration",
    "megatron-bert": "megatron-bert/MegatronBertForCausalLM",
    "mellum": "mellum/MellumForCausalLM",
    "metaclip_2": "metaclip_2/MetaClip2ForImageClassification",
    "mimo_v2_flash": "mimo_v2_flash/MiMoV2FlashForCausalLM",
    "minicpm3": "minicpm3/MiniCPM3ForCausalLM",
    "minimax": "minimax/MiniMaxForCausalLM",
    "minimax_m2": "minimax_m2/MiniMaxM2ForCausalLM",
    "ministral": "ministral/MinistralForCausalLM",
    "ministral3": "ministral3/Ministral3ForCausalLM",
    "mistral": "mistral/MistralForCausalLM",
    "mistral-nemo": "mistral/MistralForCausalLM",
    "mixtral": "mixtral/MixtralForCausalLM",
    "mobilebert": "mobilebert/MobileBertModel",
    "mobilenet_v1": "mobilenet_v1/MobileNetV1ForImageClassification",
    "mobilenet_v2": "mobilenet_v2/MobileNetV2ForImageClassification",
    "mobilevit": "mobilevit/MobileViTForSemanticSegmentation",
    "mobilevitv2": "mobilevitv2/MobileViTV2ForImageClassification",
    "modernbert": "modernbert/ModernBertForMaskedLM",
    "modernbert-decoder": "modernbert-decoder/ModernBertDecoderForCausalLM",
    "modernvbert": "modernvbert/ModernVBertForMaskedLM",
    "moshi": "moshi/MoshiForCausalLM",
    "mpnet": "mpnet/MPNetModel",
    "mpt": "mpt/MptForCausalLM",
    "mra": "mra/MraForMaskedLM",
    "mt5": "mt5/MT5ForConditionalGeneration",
    "mvp": "mvp/MvpForCausalLM",
    "nanochat": "nanochat/NanoChatForCausalLM",
    "nemotron": "nemotron/NemotronForCausalLM",
    "nemotron_h": "nemotron_h/NemotronHForCausalLM",
    "nomic_bert": "nomic_bert/NomicBertForMaskedLM",
    "nystromformer": "nystromformer/NystromformerModel",
    "olmo": "olmo/OlmoForCausalLM",
    "olmo2": "olmo2/Olmo2ForCausalLM",
    "olmo3": "olmo3/Olmo3ForCausalLM",
    "olmo_hybrid": "olmo_hybrid/OlmoHybridForCausalLM",
    "olmoe": "olmoe/OlmoeForCausalLM",
    "openai-gpt": "openai-gpt/OpenAIGPTLMHeadModel",
    "opt": "opt/OPTForCausalLM",
    "pegasus": "pegasus/PegasusForConditionalGeneration",
    "pegasus_x": "pegasus_x/PegasusXForConditionalGeneration",
    "perceiver_text": "perceiver/PerceiverForMaskedLM",
    "phi": "phi/PhiForCausalLM",
    "phi3": "phi3/Phi3ForCausalLM",
    "phi3-longrope": "phi3/Phi3ForCausalLM",
    "phi4_multimodal": "phi4_multimodal/Phi4MultimodalForCausalLM",
    "phimoe": "phimoe/PhimoeForCausalLM",
    "pix2struct": "pix2struct/Pix2StructForConditionalGeneration",
    "plbart": "plbart/PLBartForCausalLM",
    "poolformer": "poolformer/PoolFormerForImageClassification",
    "pp_lcnet": "pp_lcnet/PPLCNetForImageClassification",
    "prophetnet": "prophetnet/ProphetNetForCausalLM",
    "pvt": "pvt/PvtForImageClassification",
    "pvt_v2": "pvt_v2/PvtV2ForImageClassification",
    "qwen2": "qwen2/Qwen2ForCausalLM",
    "qwen2_5_vl": "qwen2_5_vl/Qwen2_5_VLForConditionalGeneration",
    "qwen2_moe": "qwen2_moe/Qwen2MoeForCausalLM",
    "qwen2_vl": "qwen2_vl/Qwen2VLForConditionalGeneration",
    "qwen3": "qwen3/Qwen3ForCausalLM",
    "qwen3_5_moe_text": "qwen3_5_moe_text/Qwen3_5MoeForCausalLM",
    "qwen3_5_text": "qwen3_5_text/Qwen3_5ForCausalLM",
    "qwen3_moe": "qwen3_moe/Qwen3MoeForCausalLM",
    "qwen3_next": "qwen3_next/Qwen3NextForCausalLM",
    "qwen3_vl": "qwen3_vl/Qwen3VLForConditionalGeneration",
    "regnet": "regnet/RegNetForImageClassification",
    "rembert": "rembert/RemBertModel",
    "resnet": "resnet/ResNetForImageClassification",
    "roberta": "roberta/RobertaModel",
    "roberta-prelayernorm": "roberta-prelayernorm/RobertaPreLayerNormForCausalLM",
    "roformer": "roformer/RoFormerModel",
    "rwkv": "rwkv/RwkvForCausalLM",
    "seed_oss": "seed_oss/SeedOssForCausalLM",
    "segformer": "segformer/SegformerForSemanticSegmentation",
    "sew": "sew/SEWForSequenceClassification",
    "sew-d": "sew-d/SEWDForSequenceClassification",
    "shieldgemma2": "shieldgemma2/ShieldGemma2ForImageClassification",
    "siglip": "siglip/SiglipModel",
    "smollm3": "smollm3/SmolLM3ForCausalLM",
    "smolvlm": "smolvlm/SmolVLMForConditionalGeneration",
    "solar_open": "solar_open/SolarOpenForCausalLM",
    "speech_to_text": "speech_to_text/Speech2TextForConditionalGeneration",
    "squeezebert": "squeezebert/SqueezeBertModel",
    "stablelm": "stablelm/StableLmForCausalLM",
    "starcoder2": "starcoder2/Starcoder2ForCausalLM",
    "swiftformer": "swiftformer/SwiftFormerForImageClassification",
    "swin": "swin/SwinForImageClassification",
    "swinv2": "swinv2/Swinv2ForImageClassification",
    "t5": "t5/T5ForConditionalGeneration",
    "t5gemma": "t5gemma/T5GemmaForConditionalGeneration",
    "tipsv2": "tipsv2/Tipsv2Model",
    "trocr": "trocr/TrOCRForCausalLM",
    "umt5": "umt5/UMT5ForConditionalGeneration",
    "unispeech": "unispeech/UniSpeechForSequenceClassification",
    "unispeech-sat": "unispeech-sat/UniSpeechSatForSequenceClassification",
    "vaultgemma": "vaultgemma/VaultGemmaForCausalLM",
    "videoprism": "videoprism/VideoPrismClipModel",
    "vision-encoder-decoder": "vision-encoder-decoder/VisionEncoderDecoderModel",
    "vit": "vit/ViTForImageClassification",
    "vit_msn": "vit_msn/ViTMSNForImageClassification",
    "wav2vec2-bert": "wav2vec2-bert/Wav2Vec2BertForSequenceClassification",
    "wav2vec2-conformer": "wav2vec2-conformer/Wav2Vec2ConformerForSequenceClassification",
    "wav2vec2-hf": "wav2vec2/Wav2Vec2Model",
    "wavlm": "wavlm/WavLMForSequenceClassification",
    "whisper": "whisper/WhisperForConditionalGeneration",
    "xglm": "xglm/XGLMForCausalLM",
    "xlm": "xlm/XLMModel",
    "xlm-roberta-xl": "xlm-roberta-xl/XLMRobertaXLForCausalLM",
    "xlnet": "xlnet/XLNetLMHeadModel",
    "xlstm": "xlstm/xLSTMForCausalLM",
    "xmod": "xmod/XmodForCausalLM",
    "yoso": "yoso/YosoForMaskedLM",
    "youtu": "youtu/YoutuForCausalLM",
    "zamba": "zamba/ZambaForCausalLM",
    "zamba2": "zamba2/Zamba2ForCausalLM",
    "zaya": "zaya/ZayaForCausalLM",
}


# A few models can't produce a self-contained random-tiny fixture (their processor won't serialize
# standalone), so point them at a small real checkpoint whose processor does load — the same approach
# optimum-intel uses elsewhere via `HUB_MODEL_NAMES`. `(repo_id, subfolder)`.
TINY_MODEL_OVERRIDES = {
    # timm's `TimmWrapperImageProcessor` is built from the timm `config.json` at load time and never
    # written as a standalone `preprocessor_config.json`, so a saved tiny fixture has no processor.
    "timm_wrapper": ("timm/mobilenetv3_small_100.lamb_in1k", ""),
    # VLMs / image-to-text whose randomly-generated tiny fixtures can't carry a coherent multimodal
    # processor (chat template + image-token coordination + patch-grid), plus whisper (timestamp gencfg):
    # use optimum-intel's own purpose-built tiny checkpoints, which are coherent. (Verified: each exports.)
    "idefics3": ("optimum-intel-internal-testing/tiny-random-Idefics3ForConditionalGeneration", ""),
    "smolvlm": ("optimum-intel-internal-testing/tiny-random-smolvlm2", ""),
    "qwen2_vl": ("optimum-intel-internal-testing/tiny-random-qwen2vl", ""),
    "qwen2_5_vl": ("optimum-intel-internal-testing/tiny-random-qwen2.5-vl", ""),
    "qwen3_vl": ("optimum-intel-internal-testing/tiny-random-qwen3-vl", ""),
    "llava_next": ("optimum-intel-internal-testing/tiny-random-llava-next", ""),
    "got_ocr2": ("optimum-intel-internal-testing/tiny-random-got-ocr2-hf", ""),
    "pix2struct": ("optimum-intel-internal-testing/pix2struct-tiny-random", ""),
    "whisper": ("optimum-intel-internal-testing/tiny-random-whisper", ""),
    "afmoe": ("optimum-intel-internal-testing/tiny-random-trinity", ""),
}


def _model(model_type):
    if model_type in TINY_MODEL_OVERRIDES:
        return TINY_MODEL_OVERRIDES[model_type]
    return TINY_MODELS_REPO, TINY_MODEL_SUBFOLDERS[model_type]


# Greedy, fixed length: makes Transformers and the OpenVINO runtimes directly token-comparable.
GEN_KWARGS = {"max_new_tokens": 10, "min_new_tokens": 10, "do_sample": False, "num_beams": 1}


class _ExportMixin:
    # model_types whose tiny weights give near-tied logits: the greedy argmax flips between Transformers
    # and OpenVINO on floating-point noise (and run-to-run), so a token match/mismatch is uninformative.
    # Every export is still checked on the logits (see `_assert_parity`); this set only exempts the extra
    # token-parity assertion. GenAI pipelines expose no logits, so their tests fall back to checking output
    # is produced (`_assert_genai_parity`), backed by the matching OVModel test's logit check.
    NEAR_TIED_ARCHITECTURES = {
        "pix2struct",
        "whisper",
        "hunyuan_v1_dense",
        "glm4",
        "glm4_moe_lite",
        "granitemoeshared",
        "gemma3_text",
        "gemma3n_text",
        "gemma4",
        "gemma4_unified_text",
    }

    def _next_token_logits(self, model, inputs):
        """Next-token logits from one forward over the given context, as a flat float32 array.

        Must be called *before* any `generate()` on the OpenVINO model: its stateful KV-cache persists
        across calls, so a forward run after a generate reads stale state (a wrong-looking divergence).
        """
        forward_inputs = dict(inputs)
        if model.config.is_encoder_decoder:
            gen = model.generation_config
            candidates = (gen.decoder_start_token_id, model.config.decoder_start_token_id, gen.bos_token_id)
            start_id = next((tok for tok in candidates if tok is not None), 0)
            forward_inputs["decoder_input_ids"] = torch.tensor([[start_id]])
        with torch.no_grad():
            return model(**forward_inputs).logits[:, -1, :].to(torch.float32).numpy().ravel()

    def _assert_parity(self, model_type, reference_logits, ov_logits, reference_ids, ov_ids):
        """Assert the export matches Transformers, on two complementary signals:

        - **Logits (always):** the exported next-token logits must tightly track Transformers — the
          numeric fidelity floor every export must clear. Relative L2 error is scale-invariant (logit
          magnitudes vary per model, so a fixed absolute tolerance would need per-model tuning) and a
          direct error magnitude; a faithful export is ~bit-exact (≈0), so the bar is tight.
        - **Greedy tokens (when meaningful):** `generate()` also exercises the multi-step decode loop /
          stateful KV-cache that a single forward can't reach. Skipped for near-tied weights, where the
          argmax flips on float noise (and greedy collapses), so a token match/mismatch says nothing.
        """
        rel_error = float(np.linalg.norm(reference_logits - ov_logits) / (np.linalg.norm(reference_logits) + 1e-12))
        self.assertLess(
            rel_error, 0.02, f"{model_type}: exported logits diverge from Transformers (relL2={rel_error:.4f})"
        )
        if model_type not in self.NEAR_TIED_ARCHITECTURES:
            self.assertEqual(reference_ids, ov_ids, f"{model_type}: Transformers and OpenVINO greedy tokens differ")

    def _assert_genai_parity(self, model_type, reference, actual, message):
        """GenAI pipelines expose no logits, so for near-tied weights only verify they produce output;
        the same IR's numerics are validated on logits by the matching OVModel test."""
        if model_type in self.NEAR_TIED_ARCHITECTURES:
            self.assertTrue(len(actual) > 0, f"{model_type}: OpenVINO GenAI produced no output")
        else:
            self.assertEqual(reference, actual, message)

    def _assert_exported(self, output):
        """Every exported component parses/type-checks as OpenVINO IR, and the config is saved."""
        output = Path(output)
        xmls = sorted(output.glob("openvino_*.xml"))
        self.assertTrue(xmls, f"no OpenVINO IR produced in {output}")
        core = openvino.Core()
        for xml in xmls:
            core.read_model(str(xml))
        self.assertTrue((output / "config.json").exists(), "model config not saved")


class OVHfExporterTest(_ExportMixin, unittest.TestCase):
    """Broad export + IR-parse sweep across every modality — this exporter is meant to cover almost any
    model ``torch.export`` can trace. Reuses the tiny fixtures the ``openvino`` exporter tests ship."""

    SUPPORTED_ARCHITECTURES = [
        # causal LM (text-generation) — decoder-only families across many architectures
        ("gpt2", "text-generation"),
        ("gpt_neo", "text-generation"),
        ("gpt_neox", "text-generation"),
        ("gptj", "text-generation"),
        ("gpt_bigcode", "text-generation"),
        ("gpt_oss", "text-generation"),
        ("bloom", "text-generation"),
        ("opt", "text-generation"),
        ("xglm", "text-generation"),
        ("mpt", "text-generation"),
        ("codegen", "text-generation"),
        ("starcoder2", "text-generation"),
        ("llama", "text-generation"),
        ("mistral", "text-generation"),
        ("mistral-nemo", "text-generation"),
        ("mixtral", "text-generation"),
        ("qwen2", "text-generation"),
        ("qwen2_moe", "text-generation"),
        ("qwen3", "text-generation"),
        ("qwen3_moe", "text-generation"),
        ("gemma", "text-generation"),
        ("gemma2", "text-generation"),
        ("granite", "text-generation"),
        ("granitemoe", "text-generation"),
        ("olmo", "text-generation"),
        ("olmo2", "text-generation"),
        ("cohere", "text-generation"),
        ("cohere2", "text-generation"),
        ("phi", "text-generation"),
        ("phi3", "text-generation"),
        ("stablelm", "text-generation"),
        ("glm", "text-generation"),
        ("glm4", "text-generation"),
        ("arcee", "text-generation"),
        ("biogpt", "text-generation"),
        ("smollm3", "text-generation"),
        ("falcon", "text-generation"),
        ("afmoe", "text-generation"),
        ("exaone4", "text-generation"),
        ("gemma3_text", "text-generation"),
        ("gemma4", "text-generation"),
        ("granitemoehybrid", "text-generation"),
        ("hunyuan_v1_dense", "text-generation"),
        ("lfm2_moe", "text-generation"),
        ("qwen3_next", "text-generation"),
        ("mamba", "text-generation"),
        ("falcon_mamba", "text-generation"),
        ("bitnet", "text-generation"),
        # seq2seq (text2text-generation) — encoder + stateful decoder
        ("t5", "text2text-generation"),
        ("mt5", "text2text-generation"),
        ("longt5", "text2text-generation"),
        ("bart", "text2text-generation"),
        ("mbart", "text2text-generation"),
        ("marian", "text2text-generation"),
        ("pegasus", "text2text-generation"),
        ("bigbird_pegasus", "text2text-generation"),
        ("blenderbot", "text2text-generation"),
        ("blenderbot-small", "text2text-generation"),
        ("m2m_100", "text2text-generation"),
        # text encoders (feature-extraction exercises the base model; task-specific heads are covered by
        # OVHfEncoderTest below)
        ("bert", "feature-extraction"),
        ("albert", "feature-extraction"),
        ("roberta", "feature-extraction"),
        ("distilbert", "feature-extraction"),
        ("electra", "feature-extraction"),
        ("deberta", "feature-extraction"),
        ("deberta-v2", "feature-extraction"),
        ("mobilebert", "feature-extraction"),
        ("xlm", "feature-extraction"),
        ("flaubert", "feature-extraction"),
        ("ibert", "feature-extraction"),
        ("mpnet", "feature-extraction"),
        ("nystromformer", "feature-extraction"),
        ("rembert", "feature-extraction"),
        ("roformer", "feature-extraction"),
        ("squeezebert", "feature-extraction"),
        ("esm", "feature-extraction"),
        ("convbert", "feature-extraction"),
        ("data2vec-text", "feature-extraction"),
        ("perceiver_text", "fill-mask"),
        # image
        ("vit", "image-classification"),
        ("beit", "image-classification"),
        ("deit", "image-classification"),
        ("convnext", "image-classification"),
        ("convnextv2", "image-classification"),
        ("resnet", "image-classification"),
        ("levit", "image-classification"),
        ("mobilenet_v1", "image-classification"),
        ("mobilenet_v2", "image-classification"),
        ("mobilevit", "image-classification"),
        ("poolformer", "image-classification"),
        ("swin", "image-classification"),
        ("shieldgemma2", "image-classification"),
        ("data2vec-vision", "image-classification"),
        ("donut-swin", "image-classification"),
        # image, dense-prediction heads (export + IR parse; no dedicated OVModel runtime class)
        ("segformer", "semantic-segmentation"),
        ("mobilevit", "semantic-segmentation"),
        ("detr", "object-detection"),
        # image-text dual encoders (CLIP-family zero-shot image classification)
        ("clip", "zero-shot-image-classification"),
        ("siglip", "zero-shot-image-classification"),
        # image-to-text (vision-encoder-decoder: vision encoder + stateful text decoder)
        ("vision-encoder-decoder", "image-to-text"),
        ("pix2struct", "image-to-text"),
        # audio
        ("wav2vec2-hf", "audio-classification"),
        ("hubert", "audio-classification"),
        ("wavlm", "audio-classification"),
        ("data2vec-audio", "audio-classification"),
        ("sew", "audio-classification"),
        ("sew-d", "audio-classification"),
        ("unispeech", "audio-classification"),
        ("unispeech-sat", "audio-classification"),
        ("wav2vec2-conformer", "audio-classification"),
        ("whisper", "automatic-speech-recognition"),
        ("speech_to_text", "automatic-speech-recognition"),
        # --- high-confidence fixtures added in bulk (export-only coverage) ---
        # causal LM (text-generation), auto-added from the tiny-model registry
        ("apertus", "text-generation"),
        ("axk1", "text-generation"),
        ("axk2", "text-generation"),
        ("bamba", "text-generation"),
        ("bert-generation", "text-generation"),
        ("big_bird", "text-generation"),
        ("blt", "text-generation"),
        ("cohere2_moe", "text-generation"),
        ("ctrl", "text-generation"),
        ("cwm", "text-generation"),
        ("deepseek_v2", "text-generation"),
        ("deepseek_v3", "text-generation"),
        ("deepseek_v32", "text-generation"),
        ("deepseek_v4", "text-generation"),
        ("diffllama", "text-generation"),
        ("doge", "text-generation"),
        ("dots1", "text-generation"),
        ("ernie", "text-generation"),
        ("ernie4_5", "text-generation"),
        ("ernie4_5_moe", "text-generation"),
        ("exaone_moe", "text-generation"),
        ("falcon_h1", "text-generation"),
        ("flex_olmo", "text-generation"),
        ("gemma3n", "text-generation"),
        ("gemma3n_text", "text-generation"),
        ("gemma4_unified", "text-generation"),
        ("gemma4_unified_text", "text-generation"),
        ("git", "text-generation"),
        ("glm4_moe", "text-generation"),
        ("glm4_moe_lite", "text-generation"),
        ("glm_moe_dsa", "text-generation"),
        ("gpt_neox_japanese", "text-generation"),
        ("granite_swa", "text-generation"),
        ("granitemoe_swa", "text-generation"),
        ("granitemoeshared", "text-generation"),
        ("helium", "text-generation"),
        ("hrm_text", "text-generation"),
        ("hunyuan_v1_moe", "text-generation"),
        ("hy_v3", "text-generation"),
        ("hyperclovax", "text-generation"),
        ("jais2", "text-generation"),
        ("jamba", "text-generation"),
        ("jetmoe", "text-generation"),
        ("laguna", "text-generation"),
        ("lfm2", "text-generation"),
        ("mamba2", "text-generation"),
        ("megatron-bert", "text-generation"),
        ("mellum", "text-generation"),
        ("mimo_v2_flash", "text-generation"),
        ("minicpm3", "text-generation"),
        ("minimax", "text-generation"),
        ("minimax_m2", "text-generation"),
        ("ministral", "text-generation"),
        ("ministral3", "text-generation"),
        ("modernbert-decoder", "text-generation"),
        ("moshi", "text-generation"),
        ("mvp", "text-generation"),
        ("nanochat", "text-generation"),
        ("nemotron", "text-generation"),
        ("nemotron_h", "text-generation"),
        ("olmo3", "text-generation"),
        ("olmo_hybrid", "text-generation"),
        ("olmoe", "text-generation"),
        ("openai-gpt", "text-generation"),
        ("phi4_multimodal", "text-generation"),
        ("plbart", "text-generation"),
        ("prophetnet", "text-generation"),
        ("qwen3_5_moe_text", "text-generation"),
        ("qwen3_5_text", "text-generation"),
        ("roberta-prelayernorm", "text-generation"),
        ("rwkv", "text-generation"),
        ("seed_oss", "text-generation"),
        ("solar_open", "text-generation"),
        ("trocr", "text-generation"),
        ("vaultgemma", "text-generation"),
        ("xlm-roberta-xl", "text-generation"),
        ("xlnet", "text-generation"),
        ("xlstm", "text-generation"),
        ("xmod", "text-generation"),
        ("youtu", "text-generation"),
        ("zamba", "text-generation"),
        ("zamba2", "text-generation"),
        ("zaya", "text-generation"),
        # seq2seq (text2text-generation), auto-added
        ("fsmt", "text2text-generation"),
        ("pegasus_x", "text2text-generation"),
        ("t5gemma", "text2text-generation"),
        ("umt5", "text2text-generation"),
        # masked LM (fill-mask), auto-added
        ("eurobert", "fill-mask"),
        ("fnet", "fill-mask"),
        ("funnel", "fill-mask"),
        ("jina_embeddings_v3", "fill-mask"),
        ("layoutlm", "fill-mask"),
        ("luke", "fill-mask"),
        ("modernbert", "fill-mask"),
        ("modernvbert", "fill-mask"),
        ("mra", "fill-mask"),
        ("nomic_bert", "fill-mask"),
        ("yoso", "fill-mask"),
        # image classification, auto-added
        ("bit", "image-classification"),
        ("cvt", "image-classification"),
        ("dinov2", "image-classification"),
        ("dinov2_with_registers", "image-classification"),
        ("efficientnet", "image-classification"),
        ("focalnet", "image-classification"),
        ("hgnet_v2", "image-classification"),
        ("hiera", "image-classification"),
        ("ijepa", "image-classification"),
        ("metaclip_2", "image-classification"),
        ("mobilevitv2", "image-classification"),
        ("pp_lcnet", "image-classification"),
        ("pvt", "image-classification"),
        ("pvt_v2", "image-classification"),
        ("regnet", "image-classification"),
        ("swiftformer", "image-classification"),
        ("swinv2", "image-classification"),
        ("timm_wrapper", "image-classification"),
        ("vit_msn", "image-classification"),
        # zero-shot image classification, auto-added
        ("altclip", "zero-shot-image-classification"),
        ("chinese_clip", "zero-shot-image-classification"),
        ("clipseg", "zero-shot-image-classification"),
        ("tipsv2", "zero-shot-image-classification"),
        ("videoprism", "zero-shot-image-classification"),
        # audio classification, auto-added
        ("wav2vec2-bert", "audio-classification"),
    ]

    # Models skipped in this sweep, with the reason (genuine OpenVINO-conversion / export gaps, or a
    # broken tiny-fixture tokenizer that can't build a sample).
    UNSUPPORTED = {
        "mamba": "Tiny fixture's tokenizer yields empty ids (the model itself exports fine).",
        "falcon_mamba": "Same broken-tokenizer fixture as `mamba` (the model itself exports fine).",
        "bitnet": "Fails in `from_pretrained` before export: Transformers' automatic weight conversion "
        "errors on the fixture's packed 1.58-bit weights.",
        "ibert": "torch.export can't trace I-BERT's custom quant autograd Functions (np.frexp on a "
        "traced tensor, then fake tensors lifted into graph constants from QuantEmbedding).",
        # No tiny processor: composite processors whose sub-components can't be default-constructed.
        "videoprism": "No tiny processor: `VideoPrismProcessor` is a composite whose `video_processor` "
        "sub-component can't be default-constructed for a random fixture.",
        "wav2vec2-bert": "No tiny processor: `Wav2Vec2BertProcessor` is a composite (feature-extractor + "
        "tokenizer) with no default construction and no resolvable checkpoint for the fixture.",
        "deepseek_v4": "Exports and the in-memory model compiles, but OpenVINO's own save->read "
        "round-trip fails (`read_model` -> `map::at`) on its DeepSeek-Sparse-Attention decode graph "
        "(manual-softmax attention + a heavy ScatterND/TopK/Sigmoid indexer). The serialized XML is "
        "structurally sound (layer ids, ports, edges, names all validate) yet the IR reader can't "
        "rebuild it — an OpenVINO serializer bug, not the exported graph. deepseek_v2/v3/v32 cover "
        "the family.",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_export(self, model_type, task):
        if model_type in self.UNSUPPORTED:
            self.skipTest(self.UNSUPPORTED[model_type])
        with tempfile.TemporaryDirectory() as tmp:
            model_id, subfolder = _model(model_type)
            self._assert_exported(export_openvino_hf(model_id, tmp, task=task, subfolder=subfolder))

    def test_task_inference_mapping(self):
        # `task="auto"` maps `config.architectures[0]` to a task. Unit-tested here (no download) since
        # some randomly-initialised fixtures omit `architectures`; real checkpoints set it.
        self.assertEqual(_task_from_architecture("LlamaForCausalLM"), "text-generation")
        self.assertEqual(_task_from_architecture("T5ForConditionalGeneration"), "text2text-generation")
        self.assertEqual(
            _task_from_architecture("LlavaForConditionalGeneration", has_vision_config=True), "image-text-to-text"
        )
        # Speech encoder-decoders end in ForConditionalGeneration but must route to ASR, not text2text.
        self.assertEqual(_task_from_architecture("WhisperForConditionalGeneration"), "automatic-speech-recognition")
        self.assertEqual(_task_from_architecture("VisionEncoderDecoderModel", has_vision_config=True), "image-to-text")
        self.assertEqual(_task_from_architecture("CLIPModel"), "zero-shot-image-classification")
        self.assertEqual(_task_from_architecture("ViTForImageClassification"), "image-classification")
        self.assertEqual(_task_from_architecture("BertForMaskedLM"), "fill-mask")
        self.assertEqual(_task_from_architecture("Wav2Vec2ForCTC"), "ctc")
        self.assertEqual(_task_from_architecture("SegformerForSemanticSegmentation"), "semantic-segmentation")
        self.assertEqual(_task_from_architecture("DetrForObjectDetection"), "object-detection")
        self.assertEqual(_task_from_architecture("SomethingUnknown"), "feature-extraction")

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                f"optimum-cli export openvino-hf --model {_model('gpt2')[0]} --subfolder {_model('gpt2')[1]} "
                f"--task text-generation {tmp}",
                shell=True,
                check=True,
                capture_output=True,
            )
            self._assert_exported(tmp)


class OVHfCausalLMTest(_ExportMixin, unittest.TestCase):
    """Text generation: the unified stateful decode drives both OpenVINO GenAI and OVModelForCausalLM,
    matching Transformers greedy decoding token-for-token."""

    # Every causal family whose export runs in OpenVINO GenAI's LLMPipeline with greedy token parity
    # (verified by sweep). GQA + sliding-window families (mistral, mixtral, gpt_oss) are covered since
    # the exporter stopped baking the sliding-cache eviction and repeat_kv 5-D expand. gptj still diverges
    # numerically; glm4 exports/runs but its tiny fixture ships no lm_head weight (untied), so each load
    # randomly re-inits it and parity is non-deterministic.
    GENAI_ARCHITECTURES = [
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "bloom",
        "opt",
        "xglm",
        "mpt",
        "codegen",
        "falcon",
        "starcoder2",
        "llama",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "granite",
        "granitemoe",
        "olmo",
        "olmo2",
        "cohere",
        "cohere2",
        "phi",
        "phi3",
        "phi3-longrope",
        "phimoe",
        "stablelm",
        "glm",
        "mistral",
        "mistral-nemo",
        "mixtral",
        "gpt_oss",
        "arcee",
        "biogpt",
        "smollm3",
        "afmoe",
        "exaone4",
        "granitemoehybrid",
        "hunyuan_v1_dense",
        "gptj",
        "glm4",
        "apertus",
        "axk1",
        "cohere2_moe",
        "cwm",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "diffllama",
        "dots1",
        "ernie4_5",
        "exaone_moe",
        "flex_olmo",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "gpt_neox_japanese",
        "granite_swa",
        "granitemoe_swa",
        "granitemoeshared",
        "helium",
        "hunyuan_v1_moe",
        "hy_v3",
        "jais2",
        "jetmoe",
        "laguna",
        "mellum",
        "mimo_v2_flash",
        "minicpm3",
        "nanochat",
        "nemotron",
        "olmo3",
        "olmoe",
        "qwen3_5_moe_text",
        "qwen3_5_text",
        "seed_oss",
        "solar_open",
        "vaultgemma",
        "youtu",
        "gemma3_text",
        "gemma3n_text",
        "gemma4",
        "gemma4_unified_text",
    ]

    # Every causal family whose export runs in OVModelForCausalLM with greedy token parity (verified by
    # sweep). Its runtime contract differs from GenAI's — it accepts the alibi / extra-input models GenAI
    # rejects (bloom, mpt, codegen, falcon, gpt_neo, xglm). GQA + sliding-window families now pass here
    # too; gptj and olmo still diverge numerically, and glm4's tiny fixture has no lm_head weight (untied)
    # so its parity is non-deterministic.
    OVMODEL_ARCHITECTURES = [
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "bloom",
        "opt",
        "xglm",
        "mpt",
        "codegen",
        "falcon",
        "starcoder2",
        "llama",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "granite",
        "granitemoe",
        "olmo2",
        "cohere",
        "cohere2",
        "phi",
        "phi3",
        "phi3-longrope",
        "phimoe",
        "stablelm",
        "glm",
        "mistral",
        "mistral-nemo",
        "mixtral",
        "gpt_oss",
        "arcee",
        "biogpt",
        "smollm3",
        "afmoe",
        "exaone4",
        "granitemoehybrid",
        "hunyuan_v1_dense",
        "gptj",
        "glm4",
        "apertus",
        "axk1",
        "cohere2_moe",
        "cwm",
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "diffllama",
        "dots1",
        "ernie4_5",
        "exaone_moe",
        "flex_olmo",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "gpt_neox_japanese",
        "granite_swa",
        "granitemoe_swa",
        "granitemoeshared",
        "helium",
        "hunyuan_v1_moe",
        "hy_v3",
        "jais2",
        "jetmoe",
        "laguna",
        "mellum",
        "mimo_v2_flash",
        "minicpm3",
        "nanochat",
        "nemotron",
        "olmo3",
        "olmoe",
        "seed_oss",
        "solar_open",
        "vaultgemma",
        "youtu",
        "ctrl",
        "olmo",
        "gemma3_text",
        "gemma3n_text",
        "gemma4",
        "gemma4_unified_text",
    ]

    def test_genai_layout(self):
        # Text generation is exported in the OpenVINO GenAI layout: one stateful `openvino_model.xml`
        # (the unified multi-token decode) + the OpenVINO tokenizer/detokenizer + generation config.
        with tempfile.TemporaryDirectory() as tmp:
            model_id, subfolder = _model("gpt2")
            output = Path(export_openvino_hf(model_id, tmp, task="text-generation", subfolder=subfolder))
            for name in (
                "openvino_model.xml",
                "openvino_tokenizer.xml",
                "openvino_detokenizer.xml",
                "generation_config.json",
            ):
                self.assertTrue((output / name).exists(), f"{name} missing from GenAI layout")

    @parameterized.expand(GENAI_ARCHITECTURES)
    def test_genai_pipeline(self, model_type):
        model_id, subfolder = _model(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            # fp16=False so the exported IR is fp32; combined with the f32 inference hint below this
            # matches Transformers' fp32 reference bit-for-bit enough for greedy token parity.
            export_openvino_hf(model_id, tmp, task="text-generation", fp16=False, subfolder=subfolder)

            set_seed(42)
            reference = AutoModelForCausalLM.from_pretrained(model_id, subfolder=subfolder).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder)
            inputs = tokenizer("Paris is the capital of", return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS).squeeze()[input_len:].tolist()

            pipe = openvino_genai.LLMPipeline(tmp, "CPU", **F32_CONFIG)
            genai_ids = pipe.generate(
                openvino.Tensor(inputs["input_ids"].numpy()), apply_chat_template=False, **GEN_KWARGS
            ).tokens[0]

            del pipe
            del reference
            gc.collect()

            self._assert_genai_parity(
                model_type, reference_ids, genai_ids, "Transformers and OpenVINO GenAI tokens differ"
            )

    @parameterized.expand(OVMODEL_ARCHITECTURES)
    def test_ovmodel_causal_lm(self, model_type):
        # The same text-generation export also drives optimum-intel's OVModelForCausalLM (not only GenAI),
        # and greedy decoding matches Transformers token-for-token (fp32 export + f32 inference hint).
        model_id, subfolder = _model(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="text-generation", fp16=False, subfolder=subfolder)

            set_seed(42)
            reference = AutoModelForCausalLM.from_pretrained(model_id, subfolder=subfolder).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder)
            inputs = tokenizer("Paris is the capital of", return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]

            ov_model = OVModelForCausalLM.from_pretrained(tmp, ov_config=F32_CONFIG)
            # Logits before generate — the stateful OV cache must be clean (see `_next_token_logits`).
            reference_logits = self._next_token_logits(reference, inputs)
            ov_logits = self._next_token_logits(ov_model, inputs)

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS).squeeze()[input_len:].tolist()
            ov_ids = ov_model.generate(**inputs, **GEN_KWARGS).squeeze()[input_len:].tolist()

            self._assert_parity(model_type, reference_logits, ov_logits, reference_ids, ov_ids)

            del ov_model
            del reference
            gc.collect()


class OVHfEncoderTest(_ExportMixin, unittest.TestCase):
    """Single-``openvino_model.xml`` tasks that must load and run in optimum-intel's own runtime classes
    (not just parse as IR), across the different task heads and modalities."""

    # (model_type, task, OVModel runtime class, modality)
    RUNTIME_ARCHITECTURES = [
        ("bert", "feature-extraction", OVModelForFeatureExtraction, "text"),
        ("bert", "fill-mask", OVModelForMaskedLM, "text"),
        ("albert", "text-classification", OVModelForSequenceClassification, "text"),
        ("roberta", "token-classification", OVModelForTokenClassification, "text"),
        ("distilbert", "question-answering", OVModelForQuestionAnswering, "text"),
        ("vit", "image-classification", OVModelForImageClassification, "image"),
        ("hubert", "audio-classification", OVModelForAudioClassification, "audio"),
        ("wav2vec2-hf", "ctc", OVModelForCTC, "audio"),
        ("wav2vec2-hf", "audio-frame-classification", OVModelForAudioFrameClassification, "audio"),
        ("wav2vec2-hf", "audio-xvector", OVModelForAudioXVector, "audio"),
        ("clip", "zero-shot-image-classification", OVModelForZeroShotImageClassification, "image-text"),
        ("siglip", "zero-shot-image-classification", OVModelForZeroShotImageClassification, "image-text"),
    ]

    @parameterized.expand(RUNTIME_ARCHITECTURES)
    def test_ovmodel_runtime(self, model_type, task, ov_class, modality):
        model_id, subfolder = _model(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task=task, fp16=False, subfolder=subfolder)
            model = ov_class.from_pretrained(tmp)
            # Reuse the exporter's own sample builder, then keep only the ports this IR declares.
            processor = _load_processor(model_id, modality, trust_remote_code=False, subfolder=subfolder)
            sample = _build_sample_inputs(processor, modality)
            outputs = model(**{k: v for k, v in sample.items() if k in model.input_names})
            self.assertTrue(any(v is not None for v in outputs.values()), "OVModel produced no output tensor")


class OVHfSeq2SeqTest(_ExportMixin, unittest.TestCase):
    """Encoder-decoder models exported as a stateless encoder + stateful decoder, driving
    OVModelForSeq2SeqLM / OVModelForSpeechSeq2Seq (token parity) and OpenVINO GenAI's WhisperPipeline."""

    # Encoder-decoder families whose stateful enc-dec export matches Transformers greedy (verified by
    # sweep). longt5 is excluded: it exports, but its block-local / transient-global attention bakes
    # shapes that fail at runtime in OVModelForSeq2SeqLM.
    SEQ2SEQ_ARCHITECTURES = [
        "t5",
        "mt5",
        "bart",
        "mbart",
        "marian",
        "pegasus",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        "m2m_100",
    ]

    @parameterized.expand(SEQ2SEQ_ARCHITECTURES)
    def test_ovmodel_seq2seq(self, model_type):
        model_id, subfolder = _model(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="text2text-generation", fp16=False, subfolder=subfolder)

            reference = AutoModelForSeq2SeqLM.from_pretrained(model_id, subfolder=subfolder).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder)
            inputs = tokenizer("translate English to German: the house is nice", return_tensors="pt")

            ov_model = OVModelForSeq2SeqLM.from_pretrained(tmp, ov_config=F32_CONFIG)
            reference_logits = self._next_token_logits(reference, inputs)
            ov_logits = self._next_token_logits(ov_model, inputs)

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS)[0].tolist()
            ov_ids = ov_model.generate(**inputs, **GEN_KWARGS)[0].tolist()

            self._assert_parity(model_type, reference_logits, ov_logits, reference_ids, ov_ids)

            del ov_model
            del reference
            gc.collect()

    # Speech seq2seq: an audio encoder + stateful text decoder, driven by OVModelForSpeechSeq2Seq.
    # whisper's encoder takes no padding mask; speech_to_text adds one. Both feed log-mel `input_features`
    # from the feature extractor.
    SPEECH_SEQ2SEQ_ARCHITECTURES = ["whisper", "speech_to_text"]

    @parameterized.expand(SPEECH_SEQ2SEQ_ARCHITECTURES)
    def test_ovmodel_speech_seq2seq(self, model_type):
        model_id, subfolder = _model(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="automatic-speech-recognition", fp16=False, subfolder=subfolder)

            reference = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, subfolder=subfolder).eval()
            processor = AutoProcessor.from_pretrained(model_id, subfolder=subfolder)
            # A real signal, not silence: zero audio drives the tiny models to NaN logits (log-mel of
            # silence), which makes both the parity check and greedy decode meaningless.
            sample_rate = 16000
            audio = (0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, sample_rate, endpoint=False))).astype(np.float32)
            features = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

            ov_model = OVModelForSpeechSeq2Seq.from_pretrained(tmp, ov_config=F32_CONFIG)
            reference_logits = self._next_token_logits(reference, features)
            ov_logits = self._next_token_logits(ov_model, features)

            with torch.no_grad():
                reference_ids = reference.generate(**features, **GEN_KWARGS)[0].tolist()
            ov_ids = ov_model.generate(**features, **GEN_KWARGS)[0].tolist()

            self._assert_parity(model_type, reference_logits, ov_logits, reference_ids, ov_ids)

            del ov_model
            del reference
            gc.collect()

    # Image encoder-decoders: a vision encoder + stateful text decoder — the same split as text seq2seq
    # with image in place of text. Both load through the canonical OVModelForImageTextToText, which
    # dispatches Pix2Struct (flattened_patches) to its own subclass; vit-gpt2 feeds `pixel_values`. Both
    # inputs come straight from the image processor.
    IMAGE_TO_TEXT_ARCHITECTURES = ["vision-encoder-decoder", "pix2struct"]

    @parameterized.expand(IMAGE_TO_TEXT_ARCHITECTURES)
    def test_ovmodel_image_to_text(self, model_type):
        model_id, subfolder = _model(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="image-to-text", fp16=False, subfolder=subfolder)

            reference = AutoModelForImageTextToText.from_pretrained(model_id, subfolder=subfolder).eval()
            encoder_inputs = AutoImageProcessor.from_pretrained(model_id, subfolder=subfolder)(
                images=[Image.new("RGB", (224, 224))], return_tensors="pt"
            )

            ov_model = OVModelForImageTextToText.from_pretrained(tmp, ov_config=F32_CONFIG)
            reference_logits = self._next_token_logits(reference, encoder_inputs)
            ov_logits = self._next_token_logits(ov_model, encoder_inputs)

            with torch.no_grad():
                reference_ids = reference.generate(**encoder_inputs, **GEN_KWARGS)[0].tolist()
            ov_ids = ov_model.generate(**encoder_inputs, **GEN_KWARGS)[0].tolist()

            self._assert_parity(model_type, reference_logits, ov_logits, reference_ids, ov_ids)

            del ov_model
            del reference
            gc.collect()

    def test_genai_whisper(self):
        # The same speech export runs in OpenVINO GenAI's WhisperPipeline (needs the OpenVINO
        # tokenizer/detokenizer, resolved from the Whisper processor), transcribing the same audio as
        # Transformers greedy.
        model_id, subfolder = _model("whisper")
        sample_rate = 16000
        audio = (0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, sample_rate, endpoint=False))).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="automatic-speech-recognition", fp16=False, subfolder=subfolder)

            reference = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, subfolder=subfolder).eval()
            processor = AutoProcessor.from_pretrained(model_id, subfolder=subfolder)
            features = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

            with torch.no_grad():
                reference_ids = reference.generate(**features, **GEN_KWARGS)[0]
                reference_text = processor.tokenizer.decode(reference_ids, skip_special_tokens=True)

            pipe = openvino_genai.WhisperPipeline(tmp, "CPU", **F32_CONFIG)
            genai_text = pipe.generate(audio.tolist(), **GEN_KWARGS).texts[0]

            del pipe
            del reference
            gc.collect()

            self._assert_genai_parity(
                "whisper", reference_text, genai_text, "Transformers and OpenVINO GenAI WhisperPipeline text differ"
            )


class OVHfVLMTest(_ExportMixin, unittest.TestCase):
    """VLMs exported as text-embeddings + vision-embeddings + stateful language-model, driving
    OVModelForVisualCausalLM (token parity) and OpenVINO GenAI's VLMPipeline. Vision fusion is
    per-architecture (`_VLM_VISION_FUSERS`); each registered family is exercised here."""

    # Families with a registered decomposition + input builder (OVModelForVisualCausalLM token parity).
    # qwen2_vl / qwen2_5_vl / qwen3_vl use bespoke decomposers (`_VLM_DECOMPOSERS`): two-stage vision +
    # 3D M-RoPE, and for qwen3_vl also a pos-embedding IR + deepstack features injected in the LM.
    VLM_ARCHITECTURES = [
        "llava",
        "llava_next",
        "got_ocr2",
        "idefics3",
        "smolvlm",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
    ]
    # GenAI VLMPipeline parity. The qwen mergers name their hidden-states input `hidden_states` and (for
    # qwen3_vl) their outputs `last_hidden_state`/`deepstack_feature_lists` — the tensor names GenAI's
    # C++ pipeline feeds/reads; qwen2_5_vl adds window-attention inputs, qwen3_vl an i64 pos-embed index.
    GENAI_VLM_ARCHITECTURES = ["llava", "qwen2_vl", "qwen2_5_vl", "qwen3_vl"]
    # Families that export + run on both runtimes but whose tiny fixtures are too degenerate (near-tied
    # logits) for greedy token parity — verified to run, not to match. gemma3 uses a bespoke decomposer
    # whose language model consumes `token_type_ids` for bidirectional image-token attention.
    RUNS_ONLY_VLM_ARCHITECTURES = ["gemma3"]

    @parameterized.expand(VLM_ARCHITECTURES + RUNS_ONLY_VLM_ARCHITECTURES)
    def test_ovmodel_vlm(self, model_type):
        model_id, subfolder = _model(model_type)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="image-text-to-text", fp16=False, subfolder=subfolder)

            reference = AutoModelForImageTextToText.from_pretrained(model_id, subfolder=subfolder).eval()
            processor = _load_processor(model_id, "multimodal", trust_remote_code=False, subfolder=subfolder)
            # Same per-architecture sample the exporter builds, so the image-token count is correct.
            inputs = _build_sample_inputs(processor, "multimodal", reference)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS)[0].tolist()

            ov_model = OVModelForVisualCausalLM.from_pretrained(tmp, ov_config=F32_CONFIG)
            ov_ids = ov_model.generate(**inputs, **GEN_KWARGS)[0].tolist()

            del ov_model
            del reference
            gc.collect()

            if model_type in self.RUNS_ONLY_VLM_ARCHITECTURES:
                # Degenerate fixture (near-tied logits): verify it runs and decodes the requested tokens,
                # not token parity.
                self.assertEqual(len(ov_ids), input_len + GEN_KWARGS["max_new_tokens"], "no tokens generated")
            else:
                self.assertEqual(reference_ids, ov_ids, "Transformers and OVModelForVisualCausalLM tokens differ")

    @parameterized.expand(GENAI_VLM_ARCHITECTURES + RUNS_ONLY_VLM_ARCHITECTURES)
    def test_genai_vlm(self, model_type):
        # The same VLM export runs in OpenVINO GenAI's VLMPipeline (needs the OpenVINO
        # tokenizer/detokenizer + `preprocessor_config.json`), and its generated text matches
        # Transformers greedy. The runtime's per-arch `preprocess_inputs` builds the reference prompt so
        # the image-token expansion matches what GenAI does with `apply_chat_template=True`.
        from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING

        model_id, subfolder = _model(model_type)
        image = Image.new("RGB", (56, 56))
        prompt = "A photo of a cat sitting on a"
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="image-text-to-text", fp16=False, subfolder=subfolder)

            config = AutoConfig.from_pretrained(model_id, subfolder=subfolder)
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder)
            processor = AutoProcessor.from_pretrained(model_id, subfolder=subfolder)
            model_cls = MODEL_TYPE_TO_CLS_MAPPING[config.model_type]
            inputs = model_cls.preprocess_inputs(
                text=prompt, image=image, tokenizer=tokenizer, processor=processor, config=config
            )
            full_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            reference = AutoModelForImageTextToText.from_pretrained(model_id, subfolder=subfolder).eval()
            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS)
            reference_text = tokenizer.decode(reference_ids[0], skip_special_tokens=True)[len(full_prompt) :].strip()

            pipe = openvino_genai.VLMPipeline(tmp, "CPU", **F32_CONFIG)
            genai_text = (
                pipe.generate(
                    prompt, images=[openvino.Tensor(np.array(image))], apply_chat_template=True, **GEN_KWARGS
                )
                .texts[0]
                .strip()
            )

            del pipe
            del reference
            gc.collect()

            if model_type in self.RUNS_ONLY_VLM_ARCHITECTURES:
                # Degenerate fixture (near-tied logits): verify it runs and produces text, not parity.
                self.assertTrue(genai_text, "OpenVINO GenAI VLMPipeline produced no text")
            else:
                self.assertEqual(reference_text, genai_text, "Transformers and OpenVINO GenAI VLMPipeline text differ")
