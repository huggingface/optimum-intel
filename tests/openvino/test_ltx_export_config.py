# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import unittest
from types import SimpleNamespace

import torch

from optimum.exporters.openvino.model_configs import LTXVaeDecoderOpenVINOConfig, LTXVaeDummyInputGenerator
from optimum.utils.normalized_config import NormalizedConfig


class LTXVideoExportConfigTestCase(unittest.TestCase):
    @staticmethod
    def _make_decoder_export_config(timestep_conditioning: bool) -> LTXVaeDecoderOpenVINOConfig:
        # Use a lightweight instance because these tests only validate `inputs` schema logic.
        export_config = LTXVaeDecoderOpenVINOConfig.__new__(LTXVaeDecoderOpenVINOConfig)
        export_config._normalized_config = SimpleNamespace(
            config=SimpleNamespace(timestep_conditioning=timestep_conditioning)
        )
        return export_config

    def test_ltx_vae_decoder_inputs_include_timestep_when_conditioning_enabled(self):
        export_config = self._make_decoder_export_config(timestep_conditioning=True)

        self.assertIn("latent_sample", export_config.inputs)
        self.assertIn("timestep", export_config.inputs)
        self.assertEqual(export_config.inputs["timestep"], {0: "batch_size"})

    def test_ltx_vae_decoder_inputs_skip_timestep_when_conditioning_disabled(self):
        export_config = self._make_decoder_export_config(timestep_conditioning=False)

        self.assertIn("latent_sample", export_config.inputs)
        self.assertNotIn("timestep", export_config.inputs)

    def test_ltx_vae_dummy_generator_produces_expected_shapes(self):
        normalized_config = NormalizedConfig(SimpleNamespace(in_channels=3, latent_channels=8))
        generator = LTXVaeDummyInputGenerator(
            task="semantic-segmentation",
            normalized_config=normalized_config,
            batch_size=2,
            num_channels=3,
            width=4,
            height=5,
            num_frames=6,
        )

        sample = generator.generate("sample")
        latent_sample = generator.generate("latent_sample")
        timestep = generator.generate("timestep")

        self.assertEqual(tuple(sample.shape), (2, 3, 6, 5, 4))
        self.assertEqual(tuple(latent_sample.shape), (2, 8, 6, 5, 4))
        self.assertEqual(tuple(timestep.shape), (2,))
        self.assertTrue(torch.is_floating_point(timestep))
