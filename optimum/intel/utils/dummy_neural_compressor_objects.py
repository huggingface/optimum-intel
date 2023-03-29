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


class INCModel(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForCausalLM(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForMaskedLM(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForMultipleChoice(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForQuestionAnswering(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForSeq2SeqLM(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForSequenceClassification(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForTokenClassification(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCModelForVision2Seq(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCQuantizer(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCSeq2SeqTrainer(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCTrainer(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])


class INCConfig(metaclass=DummyObject):
    _backends = ["neural_compressor"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["neural_compressor"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["neural_compressor"])
