#  Copyright 2024 The HuggingFace Team. All rights reserved.
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

import time
from transformers import is_torch_xpu_available


IS_XPU_AVAILABLE = is_torch_xpu_available(check_device=True)

MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-albert",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-bert",
    "bart": "hf-internal-testing/tiny-random-bart",
    "blenderbot": "hf-internal-testing/tiny-random-BlenderbotModel",
    "bloom": "hf-internal-testing/tiny-random-BloomModel",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "electra": "hf-internal-testing/tiny-random-electra",
    "falcon": "Intel/tiny-random-falcon",
    "gpt_bigcode": "hf-internal-testing/tiny-random-GPTBigCodeModel",
    "gpt2": "Intel/tiny-random-gpt2",
    "gpt_neo": "hf-internal-testing/tiny-random-GPTNeoModel",
    "gpt_neox": "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "llama2": "Intel/tiny-random-llama2",
    "marian": "sshleifer/tiny-marian-en-de",
    "mbart": "hf-internal-testing/tiny-random-mbart",
    "mistral": "echarlaix/tiny-random-mistral",
    "mobilenet_v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "mt5": "stas/mt5-tiny-random",
    "opt": "hf-internal-testing/tiny-random-OPTModel",
    "phi": "echarlaix/tiny-random-PhiForCausalLM",
    "qwen2": "Jiqing/tiny-random-Qwen2",
    "resnet": "hf-internal-testing/tiny-random-resnet",
    "roberta": "hf-internal-testing/tiny-random-roberta",
    "roformer": "hf-internal-testing/tiny-random-roformer",
    "st-bert": "sentence-transformers-testing/stsb-bert-tiny-safetensors",
    "st-mpnet": "sentence-transformers/all-mpnet-base-v2",
    "t5": "hf-internal-testing/tiny-random-t5",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "vit": "hf-internal-testing/tiny-random-vit",
    "wav2vec2": "anton-l/wav2vec2-random-tiny-classifier",
    "xlm": "hf-internal-testing/tiny-random-xlm",
    "patched_falcon": "Intel/tiny-random-falcon_ipex_model",
    "patched_gpt2": "Intel/tiny-random-gpt2_ipex_model",
    "patched_llama2": "Intel/tiny-random-llama2_ipex_model",
    "patched_qwen2": "Jiqing/tiny-random-Qwen2_ipex_model",
}


class Timer(object):
    def __enter__(self):
        self.elapsed = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.elapsed) * 1e3
