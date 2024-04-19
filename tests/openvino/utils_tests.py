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
import torch


MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "audio_spectrogram_transformer": "Ericwang/tiny-random-ast",
    "bge": "BAAI/bge-small-en-v1.5",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "baichuan2": "katuni4ka/tiny-random-baichuan2",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "chatglm": "katuni4ka/tiny-random-chatglm2",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "data2vec_text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec_vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec_audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "deberta": "hf-internal-testing/tiny-random-deberta",
    "deberta_v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "hf-internal-testing/tiny-random-deit",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "donut": "fxmarty/tiny-doc-qa-vision-encoder-decoder",
    "electra": "hf-internal-testing/tiny-random-electra",
    "gemma": "fxmarty/tiny-random-GemmaForCausalLM",
    "falcon": "fxmarty/really-tiny-falcon-testing",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "ibert": "hf-internal-testing/tiny-random-ibert",
    "internlm2": "katuni4ka/tiny-random-internlm2",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "longt5": "hf-internal-testing/tiny-random-longt5",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "llama_gptq": "hf-internal-testing/TinyLlama-1.1B-Chat-v0.3-GPTQ",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "opt125m": "facebook/opt-125m",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "minicpm": "katuni4ka/tiny-random-minicpm",
    "mistral": "echarlaix/tiny-random-mistral",
    "mixtral": "TitanML/tiny-mixtral",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "stas/mt5-tiny-random",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "olmo": "katuni4ka/tiny-random-olmo",
    "orion": "katuni4ka/tiny-random-orion",
    "pegasus": "hf-internal-testing/tiny-random-pegasus",
    "pix2struct": "fxmarty/pix2struct-tiny-random",
    "phi": "echarlaix/tiny-random-PhiForCausalLM",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "qwen": "katuni4ka/tiny-random-qwen",
    "qwen2": "Qwen/Qwen1.5-0.5B",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "sentence-transformers-bert": "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    "speech_to_text": "hf-internal-testing/tiny-random-Speech2TextModel",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "stable-diffusion-xl-refiner": "echarlaix/tiny-random-stable-diffusion-xl-refiner",
    "stablelm": "hf-internal-testing/tiny-random-StableLmForCausalLM",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew_d": "asapp/sew-d-tiny-100k-ft-ls100h",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
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
}

_ARCHITECTURES_TO_EXPECTED_INT4_INT8 = {"opt125m": (62, 86)}


def get_num_quantized_nodes(ov_model):
    num_fake_quantize = 0
    num_int8 = 0
    num_int4 = 0
    for elem in ov_model.model.get_ops():
        if "FakeQuantize" in elem.name:
            num_fake_quantize += 1
        for i in range(elem.get_output_size()):
            if elem.get_output_element_type(i).get_type_name() in ["i8", "u8"]:
                num_int8 += 1
            if elem.get_output_element_type(i).get_type_name() in ["i4", "u4"]:
                num_int4 += 1
    return num_fake_quantize, num_int8, num_int4
