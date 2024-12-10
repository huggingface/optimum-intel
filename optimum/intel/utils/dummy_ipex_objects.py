# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .import_utils import DummyObject, requires_backends


class IPEXModel(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForSequenceClassification(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForTokenClassification(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForMaskedLM(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForCausalLM(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForSeq2SeqLM(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForQuestionAnswering(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForImageClassification(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXModelForAudioClassification(metaclass=DummyObject):
    _backends = ["ipex"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex"])


class IPEXSentenceTransformer(metaclass=DummyObject):
    _backends = ["ipex", "sentence_transformers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["ipex", "sentence_transformers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["ipex", "sentence_transformers"])
