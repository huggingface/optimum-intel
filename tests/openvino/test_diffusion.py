#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

import random
import tempfile
import unittest
from typing import Dict

import numpy as np
import PIL
import pytest
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import floats_tensor
from openvino.runtime.ie_api import CompiledModel
from parameterized import parameterized
from transformers.testing_utils import slow
from utils_tests import MODEL_NAMES, SEED

from optimum.intel import (
    OVLatentConsistencyModelPipeline,
    OVStableDiffusionImg2ImgPipeline,
    OVStableDiffusionInpaintPipeline,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLImg2ImgPipeline,
    OVStableDiffusionXLPipeline,
)
from optimum.intel.openvino.modeling_diffusion import (
    OVModelTextEncoder,
    OVModelUnet,
    OVModelVaeDecoder,
    OVModelVaeEncoder,
)
from optimum.intel.utils.import_utils import is_diffusers_version
from optimum.utils.import_utils import is_onnxruntime_available


F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}


def _generate_inputs(batch_size=1):
    inputs = {
        "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }
    return inputs


def _create_image(height=128, width=128, batch_size=1, channel=3, input_type="pil"):
    if input_type == "pil":
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        ).resize((width, height))
    elif input_type == "np":
        image = np.random.rand(height, width, channel)
    elif input_type == "pt":
        image = torch.rand((channel, height, width))

    return [image] * batch_size


def to_np(image):
    if isinstance(image[0], PIL.Image.Image):
        return np.stack([np.array(i) for i in image], axis=0)
    elif isinstance(image, torch.Tensor):
        return image.cpu().numpy().transpose(0, 2, 3, 1)
    return image


class OVStableDiffusionPipelineBaseTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("stable-diffusion",)
    MODEL_CLASS = OVStableDiffusionPipeline
    TASK = "text-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_num_images_per_prompt(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False)
        pipeline.to("cpu")
        pipeline.compile()
        self.assertEqual(pipeline.vae_scale_factor, 2)
        self.assertEqual(pipeline.vae_decoder.config["latent_channels"], 4)
        self.assertEqual(pipeline.unet.config["in_channels"], 4)
        batch_size, height = 2, 128
        for width in [64, 128]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
            for num_images in [1, 3]:
                outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
                self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_callback(self, model_arch: str):
        MODEL_NAMES[model_arch]

        def callback_fn(step: int, timestep: int, latents: np.ndarray) -> None:
            callback_fn.has_been_called = True
            callback_fn.number_of_steps += 1

        pipeline = self.MODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)
        callback_fn.has_been_called = False
        callback_fn.number_of_steps = 0
        inputs = self.generate_inputs(height=64, width=64)
        pipeline(**inputs, callback=callback_fn, callback_steps=1)
        self.assertTrue(callback_fn.has_been_called)
        self.assertEqual(callback_fn.number_of_steps, inputs["num_inference_steps"])

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_shape(self, model_arch: str):
        height, width, batch_size = 128, 64, 1
        pipeline = self.MODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)

        if self.TASK == "image-to-image":
            input_types = ["np", "pil", "pt"]
        elif self.TASK == "text-to-image":
            input_types = ["np"]
        else:
            input_types = ["pil"]

        for input_type in input_types:
            if self.TASK == "image-to-image":
                inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, input_type=input_type)
            else:
                inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
            for output_type in ["np", "pil", "latent"]:
                inputs["output_type"] = output_type
                outputs = pipeline(**inputs).images
                if output_type == "pil":
                    self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
                elif output_type == "np":
                    self.assertEqual(outputs.shape, (batch_size, height, width, 3))
                else:
                    self.assertEqual(
                        outputs.shape,
                        (batch_size, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor),
                    )

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = _generate_inputs(batch_size)
        inputs["height"] = height
        inputs["width"] = width
        return inputs


class OVStableDiffusionImg2ImgPipelineTest(OVStableDiffusionPipelineBaseTest):
    SUPPORTED_ARCHITECTURES = ("stable-diffusion",)
    MODEL_CLASS = OVStableDiffusionImg2ImgPipeline
    TASK = "image-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_diffusers_pipeline(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        height, width, batch_size = 128, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        inputs["prompt"] = "A painting of a squirrel eating a burger"
        inputs["image"] = floats_tensor((batch_size, 3, height, width), rng=random.Random(SEED))
        np.random.seed(0)
        output = pipeline(**inputs).images[0, -3:, -3:, -1]
        # https://github.com/huggingface/diffusers/blob/v0.17.1/tests/pipelines/stable_diffusion/test_onnx_stable_diffusion_img2img.py#L71
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        self.assertTrue(np.allclose(output.flatten(), expected_slice, atol=1e-1))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False, dynamic_shapes=False)
        batch_size, num_images, height, width = 2, 3, 128, 64
        pipeline.half()
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images)
        for _height in [height, height + 16]:
            inputs = self.generate_inputs(height=_height, width=width, batch_size=batch_size)
            outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
            self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    def generate_inputs(self, height=128, width=128, batch_size=1, input_type="np"):
        inputs = _generate_inputs(batch_size)
        inputs["image"] = _create_image(height=height, width=width, batch_size=batch_size, input_type=input_type)
        inputs["strength"] = 0.75
        return inputs


class OVStableDiffusionPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("stable-diffusion",)
    MODEL_CLASS = OVStableDiffusionPipeline
    TASK = "text-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_diffusers(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        ov_pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_pipeline.text_encoder, OVModelTextEncoder)
        self.assertIsInstance(ov_pipeline.vae_encoder, OVModelVaeEncoder)
        self.assertIsInstance(ov_pipeline.vae_decoder, OVModelVaeDecoder)
        self.assertIsInstance(ov_pipeline.unet, OVModelUnet)
        self.assertIsInstance(ov_pipeline.config, Dict)

        pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAMES[model_arch])
        pipeline.safety_checker = None
        batch_size, num_images_per_prompt, height, width = 1, 2, 64, 64

        latents = ov_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            ov_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
        }

        for output_type in ["latent", "np"]:
            ov_outputs = ov_pipeline(latents=latents, output_type=output_type, **kwargs).images
            self.assertIsInstance(ov_outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images
            # Compare model outputs
            self.assertTrue(np.allclose(ov_outputs, outputs, atol=1e-4))

        # Compare model devices
        self.assertEqual(pipeline.device.type, ov_pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_image_reproducibility(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        inputs = _generate_inputs()
        height, width = 64, 64
        np.random.seed(0)
        ov_outputs_1 = pipeline(**inputs, height=height, width=width)
        np.random.seed(0)
        ov_outputs_2 = pipeline(**inputs, height=height, width=width)
        ov_outputs_3 = pipeline(**inputs, height=height, width=width)
        # Compare model outputs
        self.assertTrue(np.array_equal(ov_outputs_1.images[0], ov_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ov_outputs_1.images[0], ov_outputs_3.images[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False)
        batch_size, num_images, height, width = 3, 4, 128, 64
        pipeline.half()
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images)
        self.assertFalse(pipeline.is_dynamic)
        pipeline.compile()
        # Verify output shapes requirements not matching the static model doesn't impact the final outputs
        for _height in [height, height + 16]:
            inputs = _generate_inputs(batch_size)
            outputs = pipeline(**inputs, num_images_per_prompt=num_images, height=_height, width=width).images
            self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_height_width_properties(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        batch_size, num_images, height, width = 2, 4, 128, 64
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False, dynamic_shapes=True)
        self.assertTrue(pipeline.is_dynamic)
        self.assertEqual(pipeline.height, -1)
        self.assertEqual(pipeline.width, -1)
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images)
        self.assertFalse(pipeline.is_dynamic)
        self.assertEqual(pipeline.height, height)
        self.assertEqual(pipeline.width, width)


class OVStableDiffusionInpaintPipelineTest(OVStableDiffusionPipelineBaseTest):
    SUPPORTED_ARCHITECTURES = ("stable-diffusion",)
    MODEL_CLASS = OVStableDiffusionInpaintPipeline
    TASK = "inpaint"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.skipIf(not is_onnxruntime_available(), "this test requires onnxruntime")
    def test_compare_diffusers_pipeline(self, model_arch: str):
        from optimum.onnxruntime import ORTStableDiffusionInpaintPipeline

        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, ov_config=F32_CONFIG)
        batch_size, num_images, height, width = 1, 1, 64, 64
        latents = pipeline.prepare_latents(
            batch_size * num_images,
            pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )
        inputs = self.generate_inputs(height=height, width=width)

        inputs["image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        ).resize((width, height))

        inputs["mask_image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        ).resize((width, height))

        outputs = pipeline(**inputs, latents=latents).images
        self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

        ort_pipeline = ORTStableDiffusionInpaintPipeline.from_pretrained(model_id, export=True)
        ort_outputs = ort_pipeline(**inputs, latents=latents).images
        self.assertTrue(np.allclose(outputs, ort_outputs, atol=1e-1))

        expected_slice = np.array([0.4692, 0.5260, 0.4005, 0.3609, 0.3259, 0.4676, 0.5593, 0.4728, 0.4411])
        self.assertTrue(np.allclose(outputs[0, -3:, -3:, -1].flatten(), expected_slice, atol=1e-1))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False, dynamic_shapes=False)
        batch_size, num_images, height, width = 1, 3, 128, 64
        pipeline.half()
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images)
        for _height in [height, height + 16]:
            inputs = self.generate_inputs(height=_height, width=width, batch_size=batch_size)
            outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
            self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = super(OVStableDiffusionInpaintPipelineTest, self).generate_inputs(height, width, batch_size)
        inputs["image"] = _create_image(height=height, width=width, batch_size=1, input_type="pil")[0]
        inputs["mask_image"] = _create_image(height=height, width=width, batch_size=1, input_type="pil")[0]
        return inputs


class OVtableDiffusionXLPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("stable-diffusion-xl",)
    MODEL_CLASS = OVStableDiffusionXLPipeline
    PT_MODEL_CLASS = StableDiffusionXLPipeline
    TASK = "text-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_diffusers(self, model_arch: str):
        ov_pipeline = self.MODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_pipeline.text_encoder, OVModelTextEncoder)
        self.assertIsInstance(ov_pipeline.text_encoder_2, OVModelTextEncoder)
        self.assertIsInstance(ov_pipeline.vae_encoder, OVModelVaeEncoder)
        self.assertIsInstance(ov_pipeline.vae_decoder, OVModelVaeDecoder)
        self.assertIsInstance(ov_pipeline.unet, OVModelUnet)
        self.assertIsInstance(ov_pipeline.config, Dict)

        pipeline = self.PT_MODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        batch_size, num_images_per_prompt, height, width = 2, 3, 64, 128
        latents = ov_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            ov_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
        }

        for output_type in ["latent", "np"]:
            ov_outputs = ov_pipeline(latents=latents, output_type=output_type, **kwargs).images

            self.assertIsInstance(ov_outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images

            # Compare model outputs
            self.assertTrue(np.allclose(ov_outputs, outputs, atol=1e-4))
        # Compare model devices
        self.assertEqual(pipeline.device.type, ov_pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_image_reproducibility(self, model_arch: str):
        pipeline = self.MODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True)

        # Verify every subcomponent is compiled by default
        for component in {"unet", "vae_encoder", "vae_decoder", "text_encoder", "text_encoder_2"}:
            self.assertIsInstance(getattr(pipeline, component).request, CompiledModel)

        batch_size, num_images_per_prompt, height, width = 2, 3, 64, 128
        inputs = _generate_inputs(batch_size)
        np.random.seed(0)
        ov_outputs_1 = pipeline(**inputs, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        np.random.seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            pipeline = self.MODEL_CLASS.from_pretrained(tmp_dir)
        ov_outputs_2 = pipeline(**inputs, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        ov_outputs_3 = pipeline(**inputs, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        self.assertTrue(np.array_equal(ov_outputs_1.images[0], ov_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ov_outputs_1.images[0], ov_outputs_3.images[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False)
        batch_size, num_images, height, width = 3, 4, 128, 64
        pipeline.half()
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images)
        self.assertFalse(pipeline.is_dynamic)
        pipeline.compile()

        for _height in [height, height + 16]:
            inputs = _generate_inputs(batch_size)
            outputs = pipeline(**inputs, num_images_per_prompt=num_images, height=_height, width=width).images
            self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))


class OVStableDiffusionXLImg2ImgPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("stable-diffusion-xl", "stable-diffusion-xl-refiner")
    MODEL_CLASS = OVStableDiffusionXLImg2ImgPipeline
    PT_MODEL_CLASS = StableDiffusionXLImg2ImgPipeline
    TASK = "image-to-image"

    def test_inference(self):
        model_id = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, ov_config=F32_CONFIG)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            pipeline = self.MODEL_CLASS.from_pretrained(tmp_dir, ov_config=F32_CONFIG)

        batch_size, height, width = 1, 128, 128
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        inputs["image"] = floats_tensor((batch_size, 3, height, width), rng=random.Random(SEED))
        np.random.seed(0)
        output = pipeline(**inputs).images[0, -3:, -3:, -1]
        expected_slice = np.array([0.5683, 0.5121, 0.4767, 0.5253, 0.5072, 0.5462, 0.4766, 0.4279, 0.4855])
        self.assertTrue(np.allclose(output.flatten(), expected_slice, atol=1e-3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False, dynamic_shapes=False)
        batch_size, num_images, height, width = 2, 3, 128, 64
        pipeline.half()
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images)
        for _height in [height, height + 16]:
            inputs = self.generate_inputs(height=_height, width=width, batch_size=batch_size)
            outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
            self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    def generate_inputs(self, height=128, width=128, batch_size=1, input_type="np"):
        inputs = _generate_inputs(batch_size)
        inputs["image"] = _create_image(height=height, width=width, batch_size=batch_size, input_type=input_type)
        inputs["strength"] = 0.75
        return inputs


class OVLatentConsistencyModelPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("latent-consistency",)
    MODEL_CLASS = OVLatentConsistencyModelPipeline
    TASK = "text-to-image"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.skipIf(is_diffusers_version("<=", "0.21.4"), "not supported with this diffusers version")
    def test_compare_to_diffusers(self, model_arch: str):
        ov_pipeline = self.MODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_pipeline.text_encoder, OVModelTextEncoder)
        self.assertIsInstance(ov_pipeline.vae_encoder, OVModelVaeEncoder)
        self.assertIsInstance(ov_pipeline.vae_decoder, OVModelVaeDecoder)
        self.assertIsInstance(ov_pipeline.unet, OVModelUnet)
        self.assertIsInstance(ov_pipeline.config, Dict)

        from diffusers import LatentConsistencyModelPipeline

        pipeline = LatentConsistencyModelPipeline.from_pretrained(MODEL_NAMES[model_arch])
        batch_size, num_images_per_prompt, height, width = 2, 3, 64, 128
        latents = ov_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            ov_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_scale": 8.5,
        }

        for output_type in ["latent", "np"]:
            ov_outputs = ov_pipeline(latents=latents, output_type=output_type, **kwargs).images
            self.assertIsInstance(ov_outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images

            # Compare model outputs
            self.assertTrue(np.allclose(ov_outputs, outputs, atol=1e-4))
        # Compare model devices
        self.assertEqual(pipeline.device.type, ov_pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    @unittest.skipIf(is_diffusers_version("<=", "0.21.4"), "not supported with this diffusers version")
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True, compile=False, dynamic_shapes=False)
        batch_size, num_images, height, width = 3, 4, 128, 64
        pipeline.half()
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images)
        self.assertFalse(pipeline.is_dynamic)
        pipeline.compile()

        for _height in [height, height + 16]:
            inputs = _generate_inputs(batch_size)
            outputs = pipeline(**inputs, num_images_per_prompt=num_images, height=_height, width=width).images
            self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.skipIf(is_diffusers_version("<=", "0.21.4"), "not supported with this diffusers version")
    def test_safety_checker(self, model_arch: str):
        ov_pipeline = self.MODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], export=True, ov_config=F32_CONFIG)
        self.assertIsInstance(ov_pipeline.text_encoder, OVModelTextEncoder)
        self.assertIsInstance(ov_pipeline.vae_encoder, OVModelVaeEncoder)
        self.assertIsInstance(ov_pipeline.vae_decoder, OVModelVaeDecoder)
        self.assertIsInstance(ov_pipeline.unet, OVModelUnet)
        self.assertIsInstance(ov_pipeline.config, Dict)

        from diffusers import LatentConsistencyModelPipeline
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        pipeline = LatentConsistencyModelPipeline.from_pretrained(
            MODEL_NAMES[model_arch], safety_checker=safety_checker
        )

        batch_size, num_images_per_prompt, height, width = 2, 3, 64, 128
        latents = ov_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            ov_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_scale": 8.5,
        }

        for output_type in ["latent", "np"]:
            ov_outputs = ov_pipeline(latents=latents, output_type=output_type, **kwargs).images
            self.assertIsInstance(ov_outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images

            # Compare model outputs
            self.assertTrue(np.allclose(ov_outputs, outputs, atol=1e-4))
        # Compare model devices
        self.assertEqual(pipeline.device.type, ov_pipeline.device)
