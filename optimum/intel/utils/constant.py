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

_TASK_ALIASES = {
    "sequence-classification": "text-classification",
    "sentiment-analysis": "text-classification",
    "zero-shot-classification": "text-classification",
    "default": "feature-extraction",
    "masked-lm": "fill-mask",
    "causal-lm": "text-generation",
    "seq2seq-lm": "text2text-generation",
    "summarization": "text2text-generation",
    "translation": "text2text-generation",
}

_TASK_LEGACY = {
    "text-classification": "sequence-classification",
    "fill-mask": "masked-lm",
    "text-generation": "causal-lm",
    "text2text-generation": "seq2seq-lm",
    "summarization": "seq2seq-lm",
    "translation": "seq2seq-lm",
}
