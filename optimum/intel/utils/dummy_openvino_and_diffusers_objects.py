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


class OVStableDiffusionPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusionImg2ImgPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusionInpaintPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusionXLPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusionXLImg2ImgPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusionXLInpaintPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVLatentConsistencyModelPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVLatentConsistencyModelImg2ImgPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVLTXPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVDiffusionPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVPipelineForText2Image(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVPipelineForImage2Image(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVPipelineForText2Video(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVPipelineForInpainting(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusion3Img2ImgPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusion3Pipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVStableDiffusion3InpaintPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVFluxPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVFluxImg2ImgPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVFluxInpaintPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVFluxFillPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVSanaPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])


class OVSanaSprintPipeline(metaclass=DummyObject):
    _backends = ["openvino", "diffusers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["openvino", "diffusers"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["openvino", "diffusers"])
