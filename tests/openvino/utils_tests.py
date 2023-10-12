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
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot-small": "hf-internal-testing/tiny-random-BlenderbotModel",
    "blenderbot": "hf-internal-testing/tiny-random-blenderbot",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "convbert": "hf-internal-testing/tiny-random-ConvBertForSequenceClassification",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "data2vec_text": "hf-internal-testing/tiny-random-Data2VecTextModel",
    "data2vec_vision": "hf-internal-testing/tiny-random-Data2VecVisionModel",
    "data2vec_audio": "hf-internal-testing/tiny-random-Data2VecAudioModel",
    "deberta": "hf-internal-testing/tiny-random-deberta",
    "deberta_v2": "hf-internal-testing/tiny-random-DebertaV2Model",
    "deit": "hf-internal-testing/tiny-random-deit",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "distilbert": "hf-internal-testing/tiny-random-distilbert",
    "electra": "hf-internal-testing/tiny-random-electra",
    "flaubert": "hf-internal-testing/tiny-random-flaubert",
    # "gpt_bigcode": "bigcode/tiny_starcoder_py",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "ibert": "hf-internal-testing/tiny-random-ibert",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "longt5": "hf-internal-testing/tiny-random-longt5",
    "llama": "fxmarty/tiny-llama-fast-tokenizer",
    "m2m_100": "hf-internal-testing/tiny-random-m2m_100",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "marian": "sshleifer/tiny-marian-en-de",  # hf-internal-testing ones are broken
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet_v1": "google/mobilenet_v1_0.75_192",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "stas/mt5-tiny-random",
    "nystromformer": "hf-internal-testing/tiny-random-NystromformerModel",
    "pegasus": "hf-internal-testing/tiny-random-pegasus",
    "pix2struct": "fxmarty/pix2struct-tiny-random",
    "poolformer": "hf-internal-testing/tiny-random-PoolFormerModel",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "segformer": "hf-internal-testing/tiny-random-SegformerModel",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "stable-diffusion-xl-refiner": "echarlaix/tiny-random-stable-diffusion-xl-refiner",
    "sew": "hf-internal-testing/tiny-random-SEWModel",
    "sew_d": "hf-internal-testing/tiny-random-SEWDModel",
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "t5": "hf-internal-testing/tiny-random-t5",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech_sat": "hf-internal-testing/tiny-random-UnispeechSatModel",
    "vit": "hf-internal-testing/tiny-random-vit",
    "wavlm": "hf-internal-testing/tiny-random-WavlmModel",
    "wav2vec2": "anton-l/wav2vec2-random-tiny-classifier",
    "wav2vec2-hf": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",
    "xlm": "hf-internal-testing/tiny-random-xlm",
    "xlm_roberta": "hf-internal-testing/tiny-xlm-roberta",
}


TENSOR_ALIAS_TO_TYPE = {
    "pt": torch.Tensor,
    "np": np.ndarray,
}

SEED = 42


_ARCHITECTURES_TO_EXPECTED_INT8 = {
    "bert": (34,),
    "roberta": (34,),
    "albert": (42,),
    "vit": (31,),
    "blenderbot": (35,),
    "gpt2": (22,),
    "wav2vec2": (15,),
    "distilbert": (33,),
    "t5": (32, 52, 42),
    "stable-diffusion": (74, 4, 4, 32),
    "stable-diffusion-xl": (148, 4, 4, 33),
    "stable-diffusion-xl-refiner": (148, 4, 4, 33),
}


def get_num_quantized_nodes(ov_model):
    num_fake_quantize = 0
    num_int8 = 0
    for elem in ov_model.model.get_ops():
        if "FakeQuantize" in elem.name:
            num_fake_quantize += 1
        for i in range(elem.get_output_size()):
            if "8" in elem.get_output_element_type(i).get_type_name():
                num_int8 += 1
    return num_fake_quantize, num_int8
