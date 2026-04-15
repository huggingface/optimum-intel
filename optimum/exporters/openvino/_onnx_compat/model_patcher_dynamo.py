# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import torch
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings

from .model_patcher import ModelPatcher, is_transformers_version


def patched_vit_patch_embedding_forward(
    self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
) -> torch.Tensor:
    # _batch_size, num_channels, height, width = pixel_values.shape
    # Unexpted error.
    # TypeError: cond must be a bool, but got <class 'torch.Tensor'>?
    # torch._check(
    #    num_channels == self.num_channels,
    #    lambda: (
    #        "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
    #        f" Expected {self.num_channels} but got {num_channels}."
    #    )
    # )
    # This check fails if dynamic shapes are not properly set up.
    # Let's drop it.
    # if not interpolate_pos_encoding:
    #    torch._check(
    #        height == self.image_size[0] and width == self.image_size[1],
    #        lambda:(
    #            f"Input image size ({height}*{width}) doesn't match model"
    #            f" ({self.image_size[0]}*{self.image_size[1]})."
    #        )
    #    )
    embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
    return embeddings


class ViTForImageClassificationPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()

        if is_transformers_version(">=", "4.36.0"):
            self.original_forward = ViTPatchEmbeddings.forward
            ViTPatchEmbeddings.forward = patched_vit_patch_embedding_forward

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

        if is_transformers_version(">=", "4.36.0"):
            ViTPatchEmbeddings.forward = self.original_forward
