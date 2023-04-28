#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

import torch
from diffusers import StableDiffusionPipeline
from neural_compressor.utils.pytorch import load

from ..utils.constant import DIFFUSION_WEIGHTS_NAME, WEIGHTS_NAME
from ..utils.import_utils import _torch_version, is_torch_version
from .configuration import INCConfig


class INCStableDiffusionPipeline(StableDiffusionPipeline):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super(INCStableDiffusionPipeline, cls).from_pretrained(*args, low_cpu_mem_usage=False, **kwargs)
        components = set(model.config.keys()).intersection({"vae", "text_encoder", "unet"})
        for name in components:
            component = getattr(model, name, None)
            name_or_path = ""
            if hasattr(component, "_internal_dict"):
                name_or_path = component._internal_dict["_name_or_path"]
            elif hasattr(component, "name_or_path"):
                name_or_path = component.name_or_path
            if os.path.isdir(name_or_path):
                folder_contents = os.listdir(name_or_path)
                file_name = DIFFUSION_WEIGHTS_NAME if DIFFUSION_WEIGHTS_NAME in folder_contents else WEIGHTS_NAME
                state_dict_path = os.path.join(name_or_path, file_name)
                if os.path.exists(state_dict_path) and INCConfig.CONFIG_NAME in folder_contents:
                    msg = None
                    inc_config = INCConfig.from_pretrained(name_or_path)
                    if not is_torch_version("==", inc_config.torch_version):
                        msg = f"Quantized model was obtained with torch version {inc_config.torch_version} but {_torch_version} was found."
                    state_dict = torch.load(state_dict_path, map_location="cpu")
                    if "best_configure" in state_dict and state_dict["best_configure"] is not None:
                        try:
                            load(state_dict_path, component)
                        except Exception as e:
                            if msg is not None:
                                e.args += (msg,)
                            raise
        return model
