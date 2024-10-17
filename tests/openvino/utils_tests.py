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

import numpy as np
import openvino as ov
import torch


MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "aquila": "katuni4ka/tiny-random-aquilachat",
    "aquila2": "katuni4ka/tiny-random-aquila2",
    "audio_spectrogram_transformer": "Ericwang/tiny-random-ast",
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
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "cohere": "hf-internal-testing/tiny-random-CohereForCausalLM",
    "chatglm": "katuni4ka/tiny-random-chatglm2",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "codegen2": "katuni4ka/tiny-random-codegen2",
    "data2vec_text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec_vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec_audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "dbrx": "katuni4ka/tiny-random-dbrx",
    "deberta": "hf-internal-testing/tiny-random-deberta",
    "deberta_v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "decilm": "katuni4ka/tiny-random-decilm",
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "convnextv2": "hf-internal-testing/tiny-random-ConvNextV2Model",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "donut": "fxmarty/tiny-doc-qa-vision-encoder-decoder",
    "donut-swin": "hf-internal-testing/tiny-random-DonutSwinModel",
    "detr": "hf-internal-testing/tiny-random-DetrModel",
    "electra": "hf-internal-testing/tiny-random-electra",
    "exaone": "katuni4ka/tiny-random-exaone",
    "gemma": "fxmarty/tiny-random-GemmaForCausalLM",
    "gemma2": "katuni4ka/tiny-random-gemma2",
    "falcon": "fxmarty/really-tiny-falcon-testing",
    "falcon-40b": "katuni4ka/tiny-random-falcon-40b",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gpt_neox_japanese": "hf-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "ibert": "hf-internal-testing/tiny-random-ibert",
    "internlm": "katuni4ka/tiny-random-internlm",
    "internlm2": "katuni4ka/tiny-random-internlm2",
    "jais": "katuni4ka/tiny-random-jais",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "longt5": "hf-internal-testing/tiny-random-longt5",
    "llama": "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    "llama_awq": "HuggingFaceH4/tiny-random-LlamaForCausalLM",
    "llama_gptq": "hf-internal-testing/TinyLlama-1.1B-Chat-v0.3-GPTQ",
    "llava": "trl-internal-testing/tiny-random-LlavaForConditionalGeneration",
    "llava_next": "katuni4ka/tiny-random-llava-next",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "opt125m": "facebook/opt-125m",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "minicpm": "katuni4ka/tiny-random-minicpm",
    "mistral": "echarlaix/tiny-random-mistral",
    "mistral-nemo": "katuni4ka/tiny-random-mistral-nemo",
    "mixtral": "TitanML/tiny-mixtral",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "mt5": "stas/mt5-tiny-random",
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
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "qwen": "katuni4ka/tiny-random-qwen",
    "qwen2": "fxmarty/tiny-dummy-qwen2",
    "qwen2-moe": "katuni4ka/tiny-random-qwen1.5-moe",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "sentence-transformers-bert": "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    "speech_to_text": "hf-internal-testing/tiny-random-Speech2TextModel",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-openvino": "hf-internal-testing/tiny-stable-diffusion-openvino",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "stable-diffusion-xl-refiner": "echarlaix/tiny-random-stable-diffusion-xl-refiner",
    "stablelm": "hf-internal-testing/tiny-random-StableLmForCausalLM",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew_d": "asapp/sew-d-tiny-100k-ft-ls100h",
    "arctic": "katuni4ka/tiny-random-snowflake",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "swin-window": "yujiepan/tiny-random-swin-patch4-window7-224",
    "t5": "hf-internal-testing/tiny-random-t5",
    "trocr": "microsoft/trocr-small-handwritten",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech_sat": "hf-internal-testing/tiny-random-UnispeechSatModel",
    "vit": "hf-internal-testing/tiny-random-vit",
    "vit-with-attentions": "IlyasMoutawwakil/vit-with-attentions",
    "vit-with-hidden-states": "IlyasMoutawwakil/vit-with-hidden_states",
    "vision-encoder-decoder": "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
    "wavlm": "hf-internal-testing/tiny-random-WavlmModel",
    "wav2vec2": "anton-l/wav2vec2-random-tiny-classifier",
    "wav2vec2-hf": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "whisper": "openai/whisper-tiny.en",
    "xlm": "hf-internal-testing/tiny-random-xlm",
    "xlm_roberta": "hf-internal-testing/tiny-xlm-roberta",
    "xglm": "hf-internal-testing/tiny-random-XGLMForCausalLM",
    "xverse": "katuni4ka/tiny-random-xverse",
    "glm4": "katuni4ka/tiny-random-glm4",
    "open-clip": "hf-internal-testing/tiny-open-clip-model",
    "open-clip-ov": "zofinka/tiny-open-clip-model",
}


TENSOR_ALIAS_TO_TYPE = {
    "pt": torch.Tensor,
    "np": np.ndarray,
}

SEED = 42

_ARCHITECTURES_TO_EXPECTED_INT8 = {
    "bert": (68,),
    "roberta": (68,),
    "albert": (84,),
    "vit": (64,),
    "blenderbot": (70,),
    "gpt2": (44,),
    "wav2vec2": (34,),
    "distilbert": (66,),
    "t5": (64, 104, 84),
    "stable-diffusion": (242, 34, 42, 64),
    "stable-diffusion-xl": (366, 34, 42, 66),
    "stable-diffusion-xl-refiner": (366, 34, 42, 66),
    "open-clip": (20, 28),
    "llava": (30, 18, 2),
}


def get_num_quantized_nodes(model):
    num_fake_quantize = 0
    num_weight_nodes = {
        "int8": 0,
        "int4": 0,
        "f4e2m1": 0,
        "f8e8m0": 0,
    }
    ov_model = model if isinstance(model, ov.Model) else model.model
    for elem in ov_model.get_ops():
        if "FakeQuantize" in elem.name:
            num_fake_quantize += 1
        for i in range(elem.get_output_size()):
            type_name = elem.get_output_element_type(i).get_type_name()
            if type_name in ["i8", "u8"]:
                num_weight_nodes["int8"] += 1
            if type_name in ["i4", "u4"]:
                num_weight_nodes["int4"] += 1
            if type_name == "f4e2m1":
                num_weight_nodes["f4e2m1"] += 1
            if type_name == "f8e8m0":
                num_weight_nodes["f8e8m0"] += 1
    return num_fake_quantize, num_weight_nodes
