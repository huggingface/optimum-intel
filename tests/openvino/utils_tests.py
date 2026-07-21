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
from huggingface_hub import constants, scan_cache_dir

import optimum.exporters.openvino  # noqa: F401 (registers OpenVINO export configs in TasksManager)
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

    from kokoro.istftnet import Decoder
    from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
    from transformers import AlbertConfig

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


_GLM_EDGE_V_ORIGINAL_MODEL_ID = "zai-org/glm-edge-v-2b"
_GLM_EDGE_V_CACHE_MARKER = "tiny_glm_edge_v.v1"
_GLM_EDGE_V_REMOTE_CODE_FILES = ["configuration_glm.py", "modeling_glm.py", "siglip.py"]
_GLM_EDGE_V_ASSET_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "generation_config.json",
]


def _glm_edge_v_num_image_tokens(image_size: int, patch_size: int) -> int:
    """Vision tokens = stride-2-conv patch grid squared + boi + eoi (see siglip.Adapter)."""
    side = image_size // patch_size  # SigLIP patch grid side
    side = side // 2  # adapter conv kernel=2 stride=2
    return side * side + 2


def _glm_edge_v_patch_remote_code(modeling_path: str) -> None:
    """Repair transformers>=4.48 generation-plumbing incompatibilities in the
    shipped GLM-Edge-V remote code.

    The original remote code detects the prefill step via `if not past_key_values:`
    (an empty `DynamicCache` is truthy in transformers>=4.48) and uses the legacy
    `prepare_inputs_for_generation` protocol. These are generation-plumbing fixes
    only; the architecture (model_type, GlmForCausalLM, SigLIP vision tower,
    adapter, weights, forward math) is unchanged.
    """
    with open(modeling_path) as f:
        src = f.read()

    old_prefill = (
        '        if (input_ids is None) ^ (inputs_embeds is not None):\n'
        '            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")\n'
        "\n"
        "        if not past_key_values:\n"
    )
    new_prefill = (
        '        if (input_ids is None) ^ (inputs_embeds is not None):\n'
        '            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")\n'
        "\n"
        "        _past_len = past_key_values.get_seq_length() if isinstance(past_key_values, Cache) else (\n"
        "            len(past_key_values) if past_key_values else 0\n"
        "        )\n"
        "        _is_prefill = _past_len == 0\n"
        "\n"
        "        if _is_prefill:\n"
    )
    if old_prefill in src:
        src = src.replace(old_prefill, new_prefill, 1)

        old_embed = (
            "        if inputs_embeds is None:\n"
            "            if past_key_values:\n"
            "                inputs_embeds = self.embed_tokens(input_ids[:, -1:])\n"
            "            else:\n"
            "                inputs_embeds = self.embed_tokens(input_ids)\n"
        )
        new_embed = (
            "        if inputs_embeds is None:\n"
            "            if not _is_prefill:\n"
            "                inputs_embeds = self.embed_tokens(input_ids[:, -1:])\n"
            "            else:\n"
            "                inputs_embeds = self.embed_tokens(input_ids)\n"
        )
        if old_embed in src:
            src = src.replace(old_embed, new_embed, 1)

        old_prep_start = "    def prepare_inputs_for_generation(\n"
        old_prep_end = '            "use_cache": use_cache,\n        }\n'
        if old_prep_start in src and old_prep_end in src:
            p0 = src.index(old_prep_start)
            p1 = src.index(old_prep_end, p0) + len(old_prep_end)
            new_prep = (
                "    def prepare_inputs_for_generation(\n"
                "        self,\n"
                "        input_ids: torch.LongTensor,\n"
                "        pixel_values: Optional[torch.Tensor] = torch.zeros([1, 1, 1, 3, 672, 672]),\n"
                "        past_key_values: Optional[torch.Tensor] = None,\n"
                "        attention_mask: Optional[torch.Tensor] = None,\n"
                "        position_ids: Optional[torch.Tensor] = None,\n"
                "        use_cache: Optional[bool] = None,\n"
                "        cache_position: Optional[torch.Tensor] = None,\n"
                "        **kwargs,\n"
                "    ) -> dict:\n"
                "        model_inputs = GenerationMixin.prepare_inputs_for_generation(\n"
                "            self,\n"
                "            input_ids,\n"
                "            past_key_values=past_key_values,\n"
                "            attention_mask=attention_mask,\n"
                "            position_ids=position_ids,\n"
                "            use_cache=use_cache,\n"
                "            cache_position=cache_position,\n"
                "            **kwargs,\n"
                "        )\n"
                '        model_inputs["pixel_values"] = pixel_values\n'
                "        return model_inputs\n"
            )
            src = src[:p0] + new_prep + src[p1:]

        upd_start = "    def _update_model_kwargs_for_generation(\n"
        if upd_start in src:
            u0 = src.index(upd_start)
            after = src.index("\n    def ", u0 + 1)
            src = src[:u0] + src[after + 1 :]

    with open(modeling_path, "w") as f:
        f.write(src)


def _create_tiny_glm_edge_v_model():
    """Build a tiny, architecture-faithful random GLM-Edge-V model for testing.

    GLM-Edge-V (`zai-org/glm-edge-v-2b`) is an image-text-to-text VLM shipped with
    trust-remote-code: model_type "glm", architectures ["GlmForCausalLM"], a SigLIP
    vision tower + GLU adapter bridged into a GLM text decoder. Only the config,
    remote code and processor/tokenizer assets are downloaded (never the original
    weights); random weights are instantiated through the real architecture. The
    vision->text image-token contract is preserved by rewriting the chat template
    to insert exactly `num_image_tokens` `<|begin_of_image|>` placeholders.

    The result is cached on disk under the system temp dir, so subsequent calls are
    cheap. Falls back to the Hub id when the assets cannot be fetched.
    """
    import copy
    import shutil

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM

    output_dir = Path(tempfile.gettempdir()) / "optimum_intel_tiny_random_glm_edge_v"
    marker_file = output_dir / "_tiny_marker.json"

    hidden_size = 64
    image_size = 56
    vision_hidden_size = 64

    def _cache_is_valid():
        if not (marker_file.exists() and (output_dir / "config.json").exists()):
            return False
        try:
            with open(marker_file) as f:
                m = json.load(f)
            with open(output_dir / "config.json") as f:
                cfg = json.load(f)
        except Exception:
            return False
        if m.get("marker") != _GLM_EDGE_V_CACHE_MARKER:
            return False
        if cfg.get("model_type") != "glm" or cfg.get("architectures") != ["GlmForCausalLM"]:
            return False
        if cfg.get("hidden_size") != hidden_size:
            return False
        if cfg.get("vision_config", {}).get("image_size") != image_size:
            return False
        if not any(fn.endswith((".safetensors", ".bin")) for fn in os.listdir(output_dir)):
            return False
        return True

    if _cache_is_valid():
        return str(output_dir)

    try:
        src_dir = snapshot_download(
            _GLM_EDGE_V_ORIGINAL_MODEL_ID,
            allow_patterns=["config.json"] + _GLM_EDGE_V_REMOTE_CODE_FILES + _GLM_EDGE_V_ASSET_FILES,
        )
    except Exception:
        # No network / gated access: fall back to the Hub id (test may be skipped).
        return _GLM_EDGE_V_ORIGINAL_MODEL_ID

    with open(os.path.join(src_dir, "config.json")) as f:
        orig_cfg = json.load(f)
    orig_vc = orig_cfg["vision_config"]

    patch_size = orig_vc["patch_size"]
    num_image_tokens = _glm_edge_v_num_image_tokens(image_size, patch_size)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiny = copy.deepcopy(orig_cfg)
    tiny.update(
        dict(
            hidden_size=hidden_size,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            partial_rotary_factor=orig_cfg["partial_rotary_factor"],
            vocab_size=orig_cfg["vocab_size"],
            max_position_embeddings=orig_cfg["max_position_embeddings"],
            torch_dtype="float32",
        )
    )
    tiny["vision_config"] = dict(
        hidden_size=vision_hidden_size,
        image_size=image_size,
        intermediate_size=128,
        model_type="siglip_vision_model",
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=patch_size,
        torch_dtype="float32",
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(tiny, f, indent=2)

    for fn in _GLM_EDGE_V_REMOTE_CODE_FILES + _GLM_EDGE_V_ASSET_FILES:
        s = os.path.join(src_dir, fn)
        if os.path.exists(s):
            shutil.copy(s, output_dir / fn)

    _glm_edge_v_patch_remote_code(str(output_dir / "modeling_glm.py"))

    prep_path = output_dir / "preprocessor_config.json"
    with open(prep_path) as f:
        prep = json.load(f)
    prep["size"] = {"height": image_size, "width": image_size}
    prep["image_size"] = image_size
    with open(prep_path, "w") as f:
        json.dump(prep, f, indent=2)

    tok_cfg_path = output_dir / "tokenizer_config.json"
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)
    tmpl = tok_cfg.get("chat_template", "")
    if "range(578)" in tmpl:
        tmpl = tmpl.replace("range(578)", f"range({num_image_tokens})")
    tok_cfg["chat_template"] = tmpl
    tok_cfg["image_size"] = image_size
    with open(tok_cfg_path, "w") as f:
        json.dump(tok_cfg, f, indent=2)

    torch.manual_seed(SEED)
    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.float32)

    # Widen weight variance so the tiny random model does not collapse to a single
    # repeated output token (needed for meaningful HF-vs-OpenVINO comparison).
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.normal_(mean=0.0, std=0.6)
            elif "norm" in name:
                p.fill_(1.0)

    model.save_pretrained(output_dir, safe_serialization=True)

    with open(marker_file, "w") as f:
        json.dump(
            {
                "marker": _GLM_EDGE_V_CACHE_MARKER,
                "num_image_tokens": num_image_tokens,
                "hidden_size": hidden_size,
                "image_size": image_size,
            },
            f,
            indent=2,
        )

    return str(output_dir)


SEED = 42

F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}

TENSOR_ALIAS_TO_TYPE = {"pt": torch.Tensor, "np": np.ndarray}

OPENVINO_DEVICE = os.getenv("OPENVINO_TEST_DEVICE", "CPU")

HUB_MODEL_NAMES = {
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
    "gemma3n": "optimum-intel-internal-testing/tiny-random-gemma3n",
    "gemma3n_text": "optimum-intel-internal-testing/tiny-random-gemma3n-text",
    "gemma4": "optimum-intel-internal-testing/tiny-random-gemma4",
    "gemma4_moe": "optimum-intel-internal-testing/tiny-random-gemma4-moe",
    "gemma4_unified": "optimum-intel-internal-testing/tiny-random-gemma4-unified",
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
    "phi3-longrope": "optimum-intel-internal-testing/tiny-random-phi3-longrope",
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
    "qwen3_vl_embedding": "optimum-intel-internal-testing/tiny-random-qwen3-vl-embedding",
    "qwen3_omni_moe": "optimum-intel-internal-testing/tiny-random-qwen3-omni",
    "qwen3_next": "optimum-intel-internal-testing/tiny-random-qwen3-next",
    "qwen3_5": "optimum-intel-internal-testing/tiny-random-qwen3.5",
    "qwen3_5_moe": "optimum-intel-internal-testing/tiny-random-qwen3.5-moe",
    "qwen3_asr": "optimum-intel-internal-testing/tiny-random-qwen3-asr",
    "fun_asr": "optimum-intel-internal-testing/tiny-random-fun-asr",
    "rembert": "optimum-intel-internal-testing/tiny-random-rembert",
    "resnet": "optimum-intel-internal-testing/tiny-random-resnet",
    "roberta": "optimum-intel-internal-testing/tiny-random-roberta",
    "roformer": "optimum-intel-internal-testing/tiny-random-roformer",
    "segformer": "optimum-intel-internal-testing/tiny-random-SegformerModel",
    "sentence-transformers-bert": "optimum-intel-internal-testing/stsb-bert-tiny-safetensors",
    "sam": "optimum-intel-internal-testing/sam-vit-tiny-random",
    "smollm3": "optimum-intel-internal-testing/tiny-random-smollm3",
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
    "glm_edge_v": _create_tiny_glm_edge_v_model(),
    "open-clip": "optimum-intel-internal-testing/tiny-open-clip-model",
    "open-clip-ov": "optimum-intel-internal-testing/tiny-open-clip-model",
    "st-bert": "optimum-intel-internal-testing/all-MiniLM-L6-v2",
    "st-mpnet": "optimum-intel-internal-testing/all-mpnet-base-v2",
    "sana": "optimum-intel-internal-testing/tiny-random-sana",
    "sana-sprint": "optimum-intel-internal-testing/tiny-random-sana-sprint",
    "ltx-video": "optimum-intel-internal-testing/tiny-random-ltx-video",
    "ltx2": "optimum-intel-internal-testing/tiny-random-ltx2",
    "zamba2": "optimum-intel-internal-testing/tiny-random-zamba2",
    "qwen3_eagle3": "AngelSlim/Qwen3-1.7B_eagle3",
    "flux.2-klein": "optimum-intel-internal-testing/tiny-random-flux.2-klein",
    "qwen3_vl_eagle3": "optimum-intel-internal-testing/tiny-random-qwen3-vl-eagle3",
    "videochat_flash_qwen": "optimum-intel-internal-testing/tiny-videochat-flash-qwen",
}


def _resolve_cached_model_paths(model_names: dict) -> dict:
    try:
        if not os.path.exists(constants.HF_HUB_CACHE):
            return model_names

        repo_id_to_local_paths = {
            repo.repo_id: str(next(iter(repo.revisions)).snapshot_path)
            for repo in scan_cache_dir().repos
            if repo.revisions
        }
        return {k: repo_id_to_local_paths.get(v, v) for k, v in model_names.items()}
    except Exception:
        return model_names


MODEL_NAMES = _resolve_cached_model_paths(HUB_MODEL_NAMES)

EAGLE3_MODELS = {"qwen3_eagle3": ("AngelSlim/Qwen3-1.7B_eagle3", "Qwen/Qwen3-1.7B")}

# VLM-based Eagle3 draft models (AngelSlim Eagle3LlamaForCausalLM architecture).
# These use Qwen3-VL MRoPE and target VLM models for speculative decoding.
# Only used in the decoder test (not genai, since the VLM target needs image-text-to-text export).
EAGLE3_VLM_MODELS = {
    "qwen3_vl_eagle3": (
        "optimum-intel-internal-testing/tiny-random-qwen3-vl-eagle3",
        "optimum-intel-internal-testing/tiny-random-qwen3-vl-layer10",
    ),
}

_ARCHITECTURES_TO_EXPECTED_INT8 = {
    "afmoe": {"model": 16},
    "bert": {"model": 68 if is_transformers_version("<", "5") or is_transformers_version(">=", "5.5") else 70},
    "roberta": {"model": 68},
    "albert": {"model": 84},
    "vit": {"model": 64},
    "blenderbot": {"model": 70 if is_transformers_version("<", "5") or is_transformers_version(">=", "5.5") else 72},
    "cohere2": {"model": 30},
    "gpt2": {"model": 44},
    "granitemoehybrid": {"model": 118},
    "wav2vec2": {"model": 34},
    "distilbert": {"model": 66},
    "t5": {
        "encoder": 64,
        "decoder": 104 if is_transformers_version("<", "5") or is_transformers_version(">=", "5.5") else 106,
        "decoder_with_past": 84 if is_transformers_version("<", "5") or is_transformers_version(">=", "5.5") else 86,
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
    "flux.2-klein": {
        "transformer": 74,
        "vae_decoder": 60,
        "vae_encoder": 44,
        "text_encoder": 394,
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
    "glm_edge_v": {
        "lm_model": 26,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 18,
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
    "qwen3_vl_embedding": {
        "lm_model": 28,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 1,
        "vision_embeddings_merger_model": 104,
        "vision_embeddings_pos_model": 1,
    },
    "videochat_flash_qwen": {
        "lm_model": 30,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 5,
        "vision_projection_model": 2,
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
    "qwen3_omni_moe": {
        "lm_model": 34,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 13,
        "vision_embeddings_pos_model": 1,
        "audio_encoder_model": 18,
        "talker_model": 25,
        "talker_text_embeddings_model": 1,
        "talker_projections_model": 4,
        "code_predictor_model": 16,
        "code2wav_model": 53,
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
    "ltx2": {
        "transformer": 34,
        "vae_decoder": 28,
        "vae_encoder": 28,
        "text_encoder": 64,
        "connectors": 10,
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
    "qwen3_vl_eagle3": {"model": 18},
    "qwen3_next": {"model": 100},
    "gemma3n": {
        "lm_model": 72,
        "text_embeddings_model": 3,
        "vision_embeddings_model": 53,
        "text_embeddings_per_layer_model": 1,
    },
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
    "smollm3": {"model": 30},
    "gemma4_unified": {
        "lm_model": 56,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 3,
    },
    "qwen3_asr": {
        "encoder": 36,
        "decoder": 30,
        "decoder_with_past": 30,
    },
    "fun_asr": {
        "encoder": 46,
        "decoder": 30,
        "decoder_with_past": 30,
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
    "qwen3_vl_eagle3",
    "qwen3_asr",
    "fun_asr",
    "videochat_flash_qwen",
    "glm_edge_v",
)

if is_transformers_version("<", "5"):
    REMOTE_CODE_MODELS += ("afmoe",)


ARCH_TO_MODEL_CLASS = {
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
    "llava": "OVModelForVisualCausalLM",
    "qwen3_5_moe": "OVModelForVisualCausalLM",
    "gemma4_moe": "OVModelForVisualCausalLM",
    "gemma4_unified": "OVModelForVisualCausalLM",
    "qwen3_omni_moe": "OVModelForMultimodalLM",
    "stable-diffusion": "OVDiffusionPipeline",
    "whisper": "OVModelForSpeechSeq2Seq",
    "bart": "OVModelForSeq2SeqLM",
    "bert": "OVModelForFeatureExtraction",
    "electra": "OVModelForFeatureExtraction",
    "clip": "OVModelForZeroShotImageClassification",
    "siglip": "OVModelForZeroShotImageClassification",
}


SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED = [
    "bart",
    "musicgen",
    "whisper",
]


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
    "gemma4_moe": "gemma4",
    "glm_edge_v": "glm",
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
    "qwen3_vl_embedding": "qwen3_vl",
    "swin-window": "swin",
    "vit-with-attentions": "vit",
    "vit-with-hidden-states": "vit",
    "wav2vec2-hf": "wav2vec2",
}


def get_transformers_versions(model_type, library_name="transformers"):
    supported_model_type = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]
    export_config = next(iter(supported_model_type[model_type]["openvino"].values()))
    min_transformers = str(getattr(export_config.func, "MIN_TRANSFORMERS_VERSION", "0"))
    max_transformers = str(getattr(export_config.func, "MAX_TRANSFORMERS_VERSION", "999"))
    return min_transformers, max_transformers


def is_model_type_transformers_compatible(model_type, library_name="transformers"):
    min_transformers, max_transformers = get_transformers_versions(model_type, library_name)
    return is_transformers_version(">=", min_transformers) and is_transformers_version("<=", max_transformers)


def get_supported_model_for_library(library_name):
    valid_model = set()
    supported_model_type = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]

    for model_type in supported_model_type:
        if supported_model_type[model_type].get("openvino"):
            if is_model_type_transformers_compatible(model_type, library_name):
                valid_model.add(model_type)

    return valid_model


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3
