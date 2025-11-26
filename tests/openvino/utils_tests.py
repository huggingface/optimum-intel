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
import os
import time
import unittest
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np
import openvino as ov
import torch

from optimum.exporters.tasks import TasksManager
from optimum.intel.utils.import_utils import is_nncf_version, is_openvino_version, is_transformers_version


SEED = 42

F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}

TENSOR_ALIAS_TO_TYPE = {"pt": torch.Tensor, "np": np.ndarray}

OPENVINO_DEVICE = os.getenv("OPENVINO_TEST_DEVICE", "CPU")

MODEL_NAMES = {
    "albert": "optimum-intel-internal-testing/tiny-random-albert",
    "aquila": "optimum-intel-internal-testing/tiny-random-aquilachat",
    "aquila2": "optimum-intel-internal-testing/tiny-random-aquila2",
    "arcee": "optimum-intel-internal-testing/tiny-random-ArceeForCausalLM",
    "arctic": "optimum-intel-internal-testing/tiny-random-snowflake",
    "audio-spectrogram-transformer": "optimum-intel-internal-testing/tiny-random-ast",
    "bge": "optimum-intel-internal-testing/bge-small-en-v1.5",
    "beit": "optimum-intel-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "optimum-intel-internal-testing/tiny-random-bert",
    "bart": "optimum-intel-internal-testing/tiny-random-bart",
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
    "donut": "optimum-intel-internal-testing/tiny-doc-qa-vision-encoder-decoder",
    "donut-swin": "optimum-intel-internal-testing/tiny-random-DonutSwinModel",
    "detr": "optimum-intel-internal-testing/tiny-random-DetrModel",
    "electra": "optimum-intel-internal-testing/tiny-random-electra",
    "encoder-decoder": "optimum-internal-testing/tiny-random-encoder-decoder-gpt2-bert",
    "esm": "optimum-intel-internal-testing/tiny-random-EsmModel",
    "exaone": "optimum-intel-internal-testing/tiny-random-exaone",
    "gemma": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    "gemma2": "optimum-intel-internal-testing/tiny-random-gemma2",
    "got_ocr2": "optimum-intel-internal-testing/tiny-random-got-ocr2-hf",
    "gemma3_text": "optimum-intel-internal-testing/tiny-random-gemma3-text",
    "gemma3": "optimum-intel-internal-testing/tiny-random-gemma3",
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
    "helium": "hf-internal-testing/tiny-random-HeliumForCausalLM",
    "hubert": "optimum-intel-internal-testing/tiny-random-HubertModel",
    "ibert": "optimum-intel-internal-testing/tiny-random-ibert",
    "idefics3": "optimum-intel-internal-testing/tiny-random-Idefics3ForConditionalGeneration",
    "internlm": "optimum-intel-internal-testing/tiny-random-internlm",
    "internlm2": "optimum-intel-internal-testing/tiny-random-internlm2",
    "internvl_chat": "optimum-intel-internal-testing/tiny-random-internvl2",
    "jais": "optimum-intel-internal-testing/tiny-random-jais",
    "levit": "optimum-intel-internal-testing/tiny-random-LevitModel",
    "lfm2": "optimum-intel-internal-testing/tiny-random-lfm2",
    "longt5": "hf-internal-testing/tiny-random-LongT5Model",
    "llama": "optimum-intel-internal-testing/tiny-random-LlamaForCausalLM",
    "llama_awq": "optimum-intel-internal-testing/tiny-random-LlamaForCausalLM",
    "llama4": "optimum-intel-internal-testing/tiny-random-llama4",
    "llama4_text": "trl-internal-testing/tiny-Llama4ForCausalLM",
    "llava": "optimum-intel-internal-testing/tiny-random-llava",
    "llava_next": "optimum-intel-internal-testing/tiny-random-llava-next",
    "llava_next_mistral": "optimum-intel-internal-testing/tiny-random-llava-next-mistral",
    "llava_next_video": "optimum-intel-internal-testing/tiny-random-llava-next-video",
    "m2m_100": "optimum-intel-internal-testing/tiny-random-m2m_100",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
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
    "nemotron": "badaoui/tiny-random-NemotronForCausalLM",
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
    "phi3moe": "optimum-intel-internal-testing/phi-3.5-moe-tiny-random",
    "phi3_v": "optimum-intel-internal-testing/tiny-random-phi3-vision",
    "phi4mm": "optimum-intel-internal-testing/tiny-random-phi-4-multimodal",
    "phi4_multimodal": "echarlaix/tiny-random-phi-4-multimodal",
    "poolformer": "optimum-intel-internal-testing/tiny-random-PoolFormerModel",
    "qwen": "optimum-intel-internal-testing/tiny-random-qwen",
    "qwen2": "optimum-intel-internal-testing/tiny-dummy-qwen2",
    "qwen2_moe": "optimum-intel-internal-testing/tiny-random-qwen1.5-moe",
    "qwen2_vl": "optimum-intel-internal-testing/tiny-random-qwen2vl",
    "qwen2_5_vl": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
    "qwen3": "optimum-intel-internal-testing/tiny-random-qwen3",
    "qwen3_moe": "optimum-intel-internal-testing/tiny-random-qwen3moe",
    "resnet": "optimum-intel-internal-testing/tiny-random-resnet",
    "roberta": "optimum-intel-internal-testing/tiny-random-roberta",
    "roformer": "optimum-intel-internal-testing/tiny-random-roformer",
    "segformer": "optimum-intel-internal-testing/tiny-random-SegformerModel",
    "sentence-transformers-bert": "optimum-intel-internal-testing/stsb-bert-tiny-safetensors",
    "sam": "optimum-intel-internal-testing/sam-vit-tiny-random",
    "smollm3": "optimum-internal-testing/tiny-random-SmolLM3ForCausalLM",
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
    "xlm-roberta": "optimum-intel-internal-testing/tiny-xlm-roberta",
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
}


_ARCHITECTURES_TO_EXPECTED_INT8 = {
    "bert": {"model": 68},
    "roberta": {"model": 68},
    "albert": {"model": 84},
    "vit": {"model": 64},
    "blenderbot": {"model": 70},
    "gpt2": {"model": 44},
    "wav2vec2": {"model": 34},
    "distilbert": {"model": 66},
    "t5": {
        "encoder": 64,
        "decoder": 104,
        "decoder_with_past": 84,
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
    "sana": {
        "transformer": 58,
        "vae_decoder": 28,
        "vae_encoder": 28,
        "text_encoder": 16 if is_nncf_version(">", "2.17") else 18,
    },
    "ltx-video": {
        "transformer": 34,
        "vae_decoder": 28,
        "vae_encoder": 28,
        "text_encoder": 64,
    },
    "sam": {
        "vision_encoder": 102 if is_openvino_version("<", "2025.2.0") else 150,
        "prompt_encoder_mask_decoder": 100 if is_nncf_version("<=", "2.18") else 98,
    },
    "speecht5": {
        "encoder": 28,
        "decoder": 52,
        "postnet": 10,
        "vocoder": 80,
    },
    "clip": {"model": 130},
    "mamba": {"model": 386},
    "falcon_mamba": {"model": 194},
    "minicpmo": {
        "lm_model": 16,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 8,
        "resampler_model": 6,
    },
    "zamba2": {"model": 44},
    "lfm2": {"model": 52},
}

TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


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
