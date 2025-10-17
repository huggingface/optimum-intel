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
from typing import Dict, Optional, Union

import numpy as np
import openvino as ov
import torch

from optimum.exporters.tasks import TasksManager
from optimum.intel.openvino.modeling_base import OVBaseModel
from optimum.intel.utils.import_utils import is_nncf_version, is_openvino_version, is_transformers_version


SEED = 42

F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}

TENSOR_ALIAS_TO_TYPE = {"pt": torch.Tensor, "np": np.ndarray}

OPENVINO_DEVICE = os.getenv("OPENVINO_TEST_DEVICE", "CPU")

MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "aquila": "katuni4ka/tiny-random-aquilachat",
    "aquila2": "katuni4ka/tiny-random-aquila2",
    "arcee": "onnx-internal-testing/tiny-random-ArceeForCausalLM",
    "arctic": "katuni4ka/tiny-random-snowflake",
    "audio-spectrogram-transformer": "Ericwang/tiny-random-ast",
    "bge": "BAAI/bge-small-en-v1.5",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "baichuan2": "katuni4ka/tiny-random-baichuan2",
    "baichuan2-13b": "katuni4ka/tiny-random-baichuan2-13b",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "biogpt": "hf-tiny-model-private/tiny-random-BioGptForCausalLM",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "clip": "hf-tiny-model-private/tiny-random-CLIPModel",
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "cohere": "hf-internal-testing/tiny-random-CohereForCausalLM",
    "chatglm": "katuni4ka/tiny-random-chatglm2",
    "chatglm4": "katuni4ka/tiny-random-glm4",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "codegen2": "katuni4ka/tiny-random-codegen2",
    "data2vec-text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec-vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec-audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "dbrx": "katuni4ka/tiny-random-dbrx",
    "deberta": "hf-internal-testing/tiny-random-deberta",
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "decilm": "optimum-internal-testing/tiny-random-decilm",
    "deepseek": "katuni4ka/tiny-random-deepseek-v3",
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "convnextv2": "hf-internal-testing/tiny-random-ConvNextV2Model",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "donut": "fxmarty/tiny-doc-qa-vision-encoder-decoder",
    "donut-swin": "hf-internal-testing/tiny-random-DonutSwinModel",
    "detr": "hf-internal-testing/tiny-random-DetrModel",
    "electra": "hf-internal-testing/tiny-random-electra",
    "esm": "hf-internal-testing/tiny-random-EsmModel",
    "exaone": "katuni4ka/tiny-random-exaone",
    "gemma": "fxmarty/tiny-random-GemmaForCausalLM",
    "gemma2": "katuni4ka/tiny-random-gemma2",
    "got_ocr2": "katuni4ka/tiny-random-got-ocr2-hf",
    "gemma3_text": "katuni4ka/tiny-random-gemma3-text",
    "gemma3": "katuni4ka/tiny-random-gemma3",
    "falcon": "fxmarty/really-tiny-falcon-testing",
    "falcon-40b": "katuni4ka/tiny-random-falcon-40b",
    "falcon-mamba": "rkazants/tiny-falcon-mamba",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "flux": "katuni4ka/tiny-random-flux",
    "flux-fill": "katuni4ka/tiny-random-flux-fill",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gpt_neox_japanese": "hf-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM",
    "gpt_oss": "trl-internal-testing/tiny-GptOssForCausalLM",
    "gpt_oss_mxfp4": "echarlaix/tiny-random-gpt-oss-mxfp4",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "granite": "katuni4ka/tiny-random-granite",
    "granite-moe": "katuni4ka/tiny-random-granite-moe",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "ibert": "hf-internal-testing/tiny-random-ibert",
    "idefics3": "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration",
    "internlm": "katuni4ka/tiny-random-internlm",
    "internlm2": "katuni4ka/tiny-random-internlm2",
    "internvl_chat": "katuni4ka/tiny-random-internvl2",
    "jais": "katuni4ka/tiny-random-jais",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "longt5": "hf-internal-testing/tiny-random-longt5",
    "llama": "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    "llama_awq": "HuggingFaceH4/tiny-random-LlamaForCausalLM",
    "llama4": "hf-internal-testing/tiny-random-llama4",
    "llava": "katuni4ka/tiny-random-llava",
    "llava_next": "katuni4ka/tiny-random-llava-next",
    "llava_next_mistral": "optimum-internal-testing/tiny-random-llava-next-mistral",
    "llava_next_video": "katuni4ka/tiny-random-llava-next-video",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "opt125m": "facebook/opt-125m",
    "opt_gptq": "ybelkada/opt-125m-gptq-4bit",
    "maira2": "optimum-internal-testing/tiny-random-maira2",
    "mamba": "rkazants/tiny-mamba",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "minicpm": "katuni4ka/tiny-random-minicpm",
    "minicpm3": "katuni4ka/tiny-random-minicpm3",
    "minicpmv": "katuni4ka/tiny-random-minicpmv-2_6",
    "minicpmo": "rkazants/tiny-random-MiniCPM-o-2_6",
    "mistral": "echarlaix/tiny-random-mistral",
    "mistral-nemo": "katuni4ka/tiny-random-mistral-nemo",
    "mixtral": "TitanML/tiny-mixtral",
    "mixtral_awq": "katuni4ka/tiny-mixtral-AWQ-4bit",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "mt5": "stas/mt5-tiny-random",
    "llava-qwen2": "katuni4ka/tiny-random-nanollava",
    "nanollava_vision_tower": "katuni4ka/tiny-random-siglip",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "olmo": "katuni4ka/tiny-random-olmo-hf",
    "orion": "katuni4ka/tiny-random-orion",
    "pegasus": "hf-internal-testing/tiny-random-pegasus",
    "perceiver_text": "hf-internal-testing/tiny-random-language_perceiver",
    "perceiver_vision": "hf-internal-testing/tiny-random-vision_perceiver_conv",
    "persimmon": "hf-internal-testing/tiny-random-PersimmonForCausalLM",
    "pix2struct": "fxmarty/pix2struct-tiny-random",
    "phi": "echarlaix/tiny-random-PhiForCausalLM",
    "phi3": "Xenova/tiny-random-Phi3ForCausalLM",
    "phi3-moe": "katuni4ka/phi-3.5-moe-tiny-random",
    "phi3_v": "katuni4ka/tiny-random-phi3-vision",
    "phi4mm": "katuni4ka/tiny-random-phi-4-multimodal",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "qwen": "katuni4ka/tiny-random-qwen",
    "qwen2": "fxmarty/tiny-dummy-qwen2",
    "qwen2_moe": "katuni4ka/tiny-random-qwen1.5-moe",
    "qwen2_vl": "katuni4ka/tiny-random-qwen2vl",
    "qwen2_5_vl": "optimum-internal-testing/tiny-random-qwen2.5-vl",
    "qwen3": "katuni4ka/tiny-random-qwen3",
    "qwen3_moe": "katuni4ka/tiny-random-qwen3moe",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "sentence-transformers-bert": "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    "sam": "fxmarty/sam-vit-tiny-random",
    "smolvlm": "katuni4ka/tiny-random-smolvlm2",
    "speecht5": "hf-internal-testing/tiny-random-SpeechT5ForTextToSpeech",
    "speech_to_text": "hf-internal-testing/tiny-random-Speech2TextModel",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-openvino": "hf-internal-testing/tiny-stable-diffusion-openvino",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "stable-diffusion-xl-refiner": "echarlaix/tiny-random-stable-diffusion-xl-refiner",
    "stable-diffusion-3": "yujiepan/stable-diffusion-3-tiny-random",
    "stablelm": "hf-internal-testing/tiny-random-StableLmForCausalLM",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "siglip": "katuni4ka/tiny-random-SiglipModel",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew-d": "asapp/sew-d-tiny-100k-ft-ls100h",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "swin-window": "yujiepan/tiny-random-swin-patch4-window7-224",
    "t5": "hf-internal-testing/tiny-random-t5",
    "trocr": "microsoft/trocr-small-handwritten",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech-sat": "hf-internal-testing/tiny-random-UnispeechSatModel",
    "vit": "hf-internal-testing/tiny-random-vit",
    "vit-with-attentions": "IlyasMoutawwakil/vit-with-attentions",
    "vit-with-hidden-states": "IlyasMoutawwakil/vit-with-hidden_states",
    "vision-encoder-decoder": "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
    "wavlm": "hf-internal-testing/tiny-random-WavlmModel",
    "wav2vec2": "anton-l/wav2vec2-random-tiny-classifier",
    "wav2vec2-hf": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "whisper": "nikita-savelyev-intel/tiny-random-whisper",
    "xlm": "hf-internal-testing/tiny-random-xlm",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
    "xglm": "hf-internal-testing/tiny-random-XGLMForCausalLM",
    "xverse": "katuni4ka/tiny-random-xverse",
    "glm4": "snake7gun/tiny-random-glm4",
    "glm": "katuni4ka/tiny-random-glm-edge",
    "open-clip": "hf-internal-testing/tiny-open-clip-model",
    "open-clip-ov": "zofinka/tiny-open-clip-model",
    "st-bert": "sentence-transformers/all-MiniLM-L6-v2",
    "st-mpnet": "sentence-transformers/all-mpnet-base-v2",
    "sana": "katuni4ka/tiny-random-sana",
    "sana-sprint": "katuni4ka/tiny-random-sana-sprint",
    "ltx-video": "katuni4ka/tiny-random-ltx-video",
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
        "prompt_encoder_mask_decoder": 100,
    },
    "speecht5": {
        "encoder": 28,
        "decoder": 52,
        "postnet": 10,
        "vocoder": 80,
    },
    "clip": {"model": 130},
    "mamba": {"model": 386},
    "falcon-mamba": {"model": 194},
    "minicpmo": {
        "lm_model": 16,
        "text_embeddings_model": 1,
        "vision_embeddings_model": 8,
        "resampler_model": 6,
    },
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
    models: Dict[str, Union[ov.Model, OVBaseModel]],
    expected_num_weight_nodes_per_model: Dict[str, Dict[str, int]],
    expected_num_fake_nodes_per_model: Optional[Dict[str, int]] = None,
):
    test_case.assertEqual(len(models), len(expected_num_weight_nodes_per_model))
    actual_num_weights_per_model = {}
    actual_num_fake_nodes_per_model = {}
    for submodel_name, submodel in models.items():
        expected_num_weight_nodes = expected_num_weight_nodes_per_model[submodel_name]
        ov_model = submodel if isinstance(submodel, ov.Model) else submodel.model
        num_fake_nodes, num_weight_nodes = get_num_quantized_nodes(ov_model)
        expected_num_weight_nodes.update(dict.fromkeys(set(num_weight_nodes) - set(expected_num_weight_nodes), 0))

        actual_num_weights_per_model[submodel_name] = num_weight_nodes
        actual_num_fake_nodes_per_model[submodel_name] = num_fake_nodes

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
    "falcon-mamba": "falcon_mamba",
    "falcon-40b": "falcon",
    "gpt_oss_mxfp4": "gpt_oss",
    "granite-moe": "granitemoe",
    "llama_awq": "llama",
    "llava_next_mistral": "llava_next",
    "mistral-nemo": "mistral",
    "mixtral_awq": "mixtral",
    "nanollava_vision_tower": "siglip",
    "opt125m": "opt",
    "opt_gptq": "opt",
    "perceiver_text": "perceiver",
    "perceiver_vision": "perceiver",
    "phi3-moe": "phimoe",
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
