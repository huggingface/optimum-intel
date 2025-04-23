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

import json
import logging
import unittest
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import load_image
from parameterized import parameterized
from utils_tests import MODEL_NAMES, SEED

from optimum.intel.openvino import (
    OVDiffusionPipeline,
    OVPipelineForImage2Image,
    OVPipelineForInpainting,
    OVPipelineForText2Image,
)
from optimum.intel.openvino.utils import TemporaryDirectory
from optimum.intel.utils.import_utils import is_diffusers_version, is_transformers_version
from optimum.utils.testing_utils import require_diffusers


def get_generator(framework, seed):
    if framework == "np":
        return np.random.RandomState(seed)
    elif framework == "pt":
        return torch.Generator().manual_seed(seed)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def _generate_prompts(batch_size=1):
    inputs = {
        "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }
    return inputs


def _generate_images(height=128, width=128, batch_size=1, channel=3, input_type="pil"):
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


class OVPipelineForText2ImageTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl", "latent-consistency"]
    NEGATIVE_PROMPT_SUPPORT_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl", "latent-consistency"]
    if is_transformers_version(">=", "4.40.0"):
        SUPPORTED_ARCHITECTURES.extend(["stable-diffusion-3", "flux", "sana"])
        NEGATIVE_PROMPT_SUPPORT_ARCHITECTURES.extend(["stable-diffusion-3"])
    if is_diffusers_version(">=", "0.33.0"):
        SUPPORTED_ARCHITECTURES.extend(["sana-sprint"])
    CALLBACK_SUPPORT_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl", "latent-consistency"]

    OVMODEL_CLASS = OVPipelineForText2Image
    AUTOMODEL_CLASS = AutoPipelineForText2Image

    TASK = "text-to-image"

    def generate_inputs(self, height=128, width=128, batch_size=1, model_type=None):
        inputs = _generate_prompts(batch_size=batch_size)

        inputs["height"] = height
        inputs["width"] = width

        if model_type == "sana-sprint":
            inputs["num_inference_steps"] = 2

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(f"does not appear to have a file named {self.OVMODEL_CLASS.config_name}", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_ov_pipeline_class_dispatch(self, model_arch: str):
        auto_cls = self.AUTOMODEL_CLASS if "sana" not in model_arch else DiffusionPipeline
        auto_pipeline = auto_cls.from_pretrained(MODEL_NAMES[model_arch])
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        self.assertEqual(ov_pipeline.auto_model_class, auto_pipeline.__class__)

        auto_pipeline = DiffusionPipeline.from_pretrained(MODEL_NAMES[model_arch])
        ov_pipeline = OVDiffusionPipeline.from_pretrained(MODEL_NAMES[model_arch])

        self.assertEqual(ov_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        for batch_size in [1, 3]:
            for height in [64, 128]:
                for width in [64, 128]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(
                            height=height, width=width, batch_size=batch_size, model_type=model_arch
                        )
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        auto_cls = self.AUTOMODEL_CLASS if "sana" not in model_arch else DiffusionPipeline
        diffusers_pipeline = auto_cls.from_pretrained(MODEL_NAMES[model_arch])

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type
            if "sana" in model_arch:
                # resolution binning will lead to resize output to standard resolution and back that can interpolate floating-point deviations
                inputs["use_resolution_binning"] = False
            ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            np.testing.assert_allclose(ov_output, diffusers_output, atol=6e-3, rtol=1e-2)

        # test on inputs nondivisible on 64
        height, width, batch_size = 96, 96, 1

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type
            if "sana" in model_arch:
                # resolution binning will lead to resize output to standard resolution and back that can interpolate floating-point deviations
                inputs["use_resolution_binning"] = False

            ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ov_output, diffusers_output, atol=6e-3, rtol=1e-2)

    @parameterized.expand(CALLBACK_SUPPORT_ARCHITECTURES)
    @require_diffusers
    def test_callback(self, model_arch: str):
        height, width, batch_size = 64, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        class Callback:
            def __init__(self):
                self.has_been_called = False
                self.number_of_steps = 0

            def __call__(self, *args, **kwargs) -> None:
                self.has_been_called = True
                self.number_of_steps += 1

        ov_callback = Callback()
        auto_callback = Callback()

        ov_pipe = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        auto_cls = self.AUTOMODEL_CLASS if model_arch != "sana" else DiffusionPipeline
        auto_pipe = auto_cls.from_pretrained(MODEL_NAMES[model_arch])

        # callback_steps=1 to trigger callback every step
        ov_pipe(**inputs, callback=ov_callback, callback_steps=1)
        auto_pipe(**inputs, callback=auto_callback, callback_steps=1)

        self.assertTrue(ov_callback.has_been_called)
        self.assertTrue(auto_callback.has_been_called)
        self.assertEqual(auto_callback.number_of_steps, ov_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 128, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)
        if "sana" in model_arch:
            inputs["use_resolution_binning"] = False

        for output_type in ["pil", "np", "pt", "latent"]:
            inputs["output_type"] = output_type
            outputs = pipeline(**inputs).images
            if output_type == "pil":
                self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
            elif output_type == "np":
                self.assertEqual(outputs.shape, (batch_size, height, width, 3))
            elif output_type == "pt":
                self.assertEqual(outputs.shape, (batch_size, 3, height, width))
            else:
                if model_arch != "flux":
                    out_channels = (
                        pipeline.unet.config.out_channels
                        if pipeline.unet is not None
                        else pipeline.transformer.config.out_channels
                    )
                    self.assertEqual(
                        outputs.shape,
                        (
                            batch_size,
                            out_channels,
                            height // pipeline.vae_scale_factor,
                            width // pipeline.vae_scale_factor,
                        ),
                    )
                else:
                    packed_height = height // pipeline.vae_scale_factor // 2
                    packed_width = width // pipeline.vae_scale_factor // 2
                    channels = pipeline.transformer.config.in_channels
                    self.assertEqual(outputs.shape, (batch_size, packed_height * packed_width, channels))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)
        if "sana" in model_arch:
            inputs["use_resolution_binning"] = False

        for generator_framework in ["np", "pt"]:
            ov_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ov_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ov_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertFalse(np.array_equal(ov_outputs_1.images[0], ov_outputs_3.images[0]))
            np.testing.assert_allclose(ov_outputs_1.images[0], ov_outputs_2.images[0], atol=1e-4, rtol=1e-2)

    @parameterized.expand(NEGATIVE_PROMPT_SUPPORT_ARCHITECTURES)
    def test_negative_prompt(self, model_arch: str):
        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        negative_prompt = ["This is a negative prompt"]
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        images_1 = pipeline(**inputs, negative_prompt=negative_prompt, generator=get_generator("pt", SEED)).images
        prompt = inputs.pop("prompt")

        if model_arch == "stable-diffusion-xl":
            (
                inputs["prompt_embeds"],
                inputs["negative_prompt_embeds"],
                inputs["pooled_prompt_embeds"],
                inputs["negative_pooled_prompt_embeds"],
            ) = pipeline.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=1,
                device=torch.device("cpu"),
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        elif model_arch == "stable-diffusion-3":
            (
                inputs["prompt_embeds"],
                inputs["negative_prompt_embeds"],
                inputs["pooled_prompt_embeds"],
                inputs["negative_pooled_prompt_embeds"],
            ) = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                num_images_per_prompt=1,
                device=torch.device("cpu"),
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        else:
            inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = pipeline.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=1,
                device=torch.device("cpu"),
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

        images_2 = pipeline(**inputs, generator=get_generator("pt", SEED)).images

        np.testing.assert_allclose(images_1, images_2, atol=1e-4, rtol=1e-2)

    @parameterized.expand(["stable-diffusion", "latent-consistency"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "katuni4ka/tiny-random-stable-diffusion-with-safety-checker", subfolder="safety_checker"
        )

        pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)

        self.assertIsInstance(pipeline.safety_checker, StableDiffusionSafetyChecker)
        self.assertIsInstance(ov_pipeline.safety_checker, StableDiffusionSafetyChecker)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED))
        diffusers_output = pipeline(**inputs, generator=get_generator("pt", SEED))

        ov_nsfw_content_detected = ov_output.nsfw_content_detected
        diffusers_nsfw_content_detected = diffusers_output.nsfw_content_detected

        self.assertTrue(ov_nsfw_content_detected is not None)
        self.assertTrue(diffusers_nsfw_content_detected is not None)
        self.assertEqual(ov_nsfw_content_detected, diffusers_nsfw_content_detected)

        ov_images = ov_output.images
        diffusers_images = diffusers_output.images

        np.testing.assert_allclose(ov_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @require_diffusers
    def test_load_and_save_pipeline_with_safety_checker(self):
        model_id = "katuni4ka/tiny-random-stable-diffusion-with-safety-checker"
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(model_id)
        self.assertTrue(ov_pipeline.safety_checker is not None)
        self.assertIsInstance(ov_pipeline.safety_checker, StableDiffusionSafetyChecker)
        with TemporaryDirectory() as tmpdirname:
            ov_pipeline.save_pretrained(tmpdirname)
            for subdir in [
                "text_encoder",
                "tokenizer",
                "unet",
                "vae_encoder",
                "vae_decoder",
                "scheduler",
                "feature_extractor",
            ]:
                subdir_path = Path(tmpdirname) / subdir
                self.assertTrue(subdir_path.is_dir())
            # check that config contains original model classes
            pipeline_config = Path(tmpdirname) / "model_index.json"
            self.assertTrue(pipeline_config.exists())
            with pipeline_config.open("r") as f:
                config = json.load(f)
                for key in ["unet", "vae", "text_encoder"]:
                    model_lib, model_class = config[key]
                    self.assertTrue(model_lib in ["diffusers", "transformers"])
                    self.assertFalse(model_class.startswith("OV"))
            loaded_pipeline = self.OVMODEL_CLASS.from_pretrained(tmpdirname)
            for component in ["text_encoder", "unet", "vae_encoder", "vae_decoder"]:
                config = getattr(getattr(ov_pipeline, component), "config", None)
                if config is not None:
                    loaded_config = getattr(getattr(loaded_pipeline, component), "config")
                    self.assertDictEqual(
                        config, loaded_config, f"Expected config:\n{config}\nLoaded config:|n{loaded_config}"
                    )
            self.assertTrue(loaded_pipeline.safety_checker is not None)
            self.assertIsInstance(loaded_pipeline.safety_checker, StableDiffusionSafetyChecker)
            del loaded_pipeline
        del ov_pipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_height_width_properties(self, model_arch: str):
        batch_size, height, width, num_images_per_prompt = 2, 128, 64, 4
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch], export=True, compile=False, dynamic_shapes=True
        )

        self.assertTrue(ov_pipeline.is_dynamic)
        self.assertEqual(ov_pipeline.batch_size, -1)
        self.assertEqual(ov_pipeline.height, -1)
        self.assertEqual(ov_pipeline.width, -1)

        ov_pipeline.reshape(
            batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt
        )

        self.assertFalse(ov_pipeline.is_dynamic)
        expected_batch = batch_size * num_images_per_prompt
        if (
            ov_pipeline.unet is not None
            and "timestep_cond" not in {inputs.get_any_name() for inputs in ov_pipeline.unet.model.inputs}
        ) or (
            ov_pipeline.transformer is not None
            and "txt_ids" not in {inputs.get_any_name() for inputs in ov_pipeline.transformer.model.inputs}
        ):
            expected_batch *= 2
        self.assertEqual(
            ov_pipeline.batch_size,
            expected_batch,
        )
        self.assertEqual(ov_pipeline.height, height)
        self.assertEqual(ov_pipeline.width, width)

    @require_diffusers
    def test_textual_inversion(self):
        model_id = "katuni4ka/tiny-random-stable-diffusion-with-safety-checker"
        ti_id = "katuni4ka/textual_inversion_cat"

        inputs = self.generate_inputs()
        inputs["prompt"] = "A <cat-toy> backpack"

        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(model_id, safety_checker=None)
        diffusers_pipeline.load_textual_inversion(ti_id)

        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(model_id, compile=False, safety_checker=None)
        ov_pipeline.load_textual_inversion(ti_id)

        diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images
        ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images

        np.testing.assert_allclose(ov_output, diffusers_output, atol=1e-4, rtol=1e-2)

    @require_diffusers
    def test_load_custom_weight_variant(self):
        model_id = "katuni4ka/tiny-stable-diffusion-torch-custom-variant"
        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(model_id, variant="custom")
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(model_id, compile=False, variant="custom")
        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED))
        diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED))
        ov_images = ov_output.images
        diffusers_images = diffusers_output.images

        np.testing.assert_allclose(ov_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_static_shape_image_generation(self, model_arch):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], compile=False)
        pipeline.reshape(batch_size=1, height=32, width=32)
        pipeline.compile()
        # generation with incompatible size
        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)
        inputs["output_type"] = "pil"
        from optimum.intel.openvino.modeling_diffusion import logger as diffusers_logger

        with self.assertLogs(diffusers_logger, logging.WARN) as warning_log:
            image = pipeline(**inputs).images[0]
            self.assertTrue(
                any(
                    "Incompatible width argument provided" in log or "Incompatible height argument provided" in log
                    for log in warning_log.output
                )
            )
        self.assertTupleEqual(image.size, (32, 32))
        # generation without height / width provided
        inputs.pop("height")
        inputs.pop("width")
        image = pipeline(**inputs).images[0]
        self.assertTupleEqual(image.size, (32, 32))


class OVPipelineForImage2ImageTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl", "latent-consistency"]
    if is_transformers_version(">=", "4.40.0"):
        SUPPORTED_ARCHITECTURES.append("stable-diffusion-3")
        SUPPORTED_ARCHITECTURES.append("flux")

    AUTOMODEL_CLASS = AutoPipelineForImage2Image
    OVMODEL_CLASS = OVPipelineForImage2Image

    TASK = "image-to-image"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="pil", model_type=None):
        inputs = _generate_prompts(batch_size=batch_size)

        inputs["image"] = _generate_images(
            height=height, width=width, batch_size=batch_size, channel=channel, input_type=input_type
        )

        if model_type in ["flux", "stable-diffusion-3"]:
            inputs["height"] = height
            inputs["width"] = width

        inputs["strength"] = 0.75

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(f"does not appear to have a file named {self.OVMODEL_CLASS.config_name}", str(context.exception))

    @parameterized.expand(list(SUPPORTED_ARCHITECTURES))
    @require_diffusers
    def test_ov_pipeline_class_dispatch(self, model_arch: str):
        auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        self.assertEqual(ov_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        for batch_size in [1, 3]:
            for height in [64, 128]:
                for width in [64, 128]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(
                            height=height, width=width, batch_size=batch_size, model_type=model_arch
                        )
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(["stable-diffusion", "stable-diffusion-xl", "latent-consistency"])
    @require_diffusers
    def test_callback(self, model_arch: str):
        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        class Callback:
            def __init__(self):
                self.has_been_called = False
                self.number_of_steps = 0

            def __call__(self, *args, **kwargs) -> None:
                self.has_been_called = True
                self.number_of_steps += 1

        ov_pipe = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ov_callback = Callback()
        auto_callback = Callback()
        # callback_steps=1 to trigger callback every step
        ov_pipe(**inputs, callback=ov_callback, callback_steps=1)
        auto_pipe(**inputs, callback=auto_callback, callback_steps=1)

        self.assertTrue(ov_callback.has_been_called)
        self.assertEqual(ov_callback.number_of_steps, auto_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 128, 64, 1

        for input_type in ["pil", "np", "pt"]:
            inputs = self.generate_inputs(
                height=height, width=width, batch_size=batch_size, input_type=input_type, model_type=model_arch
            )

            for output_type in ["pil", "np", "pt", "latent"]:
                inputs["output_type"] = output_type
                outputs = pipeline(**inputs).images
                if output_type == "pil":
                    self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
                elif output_type == "np":
                    self.assertEqual(outputs.shape, (batch_size, height, width, 3))
                elif output_type == "pt":
                    self.assertEqual(outputs.shape, (batch_size, 3, height, width))
                else:
                    if model_arch != "flux":
                        out_channels = (
                            pipeline.unet.config.out_channels
                            if pipeline.unet is not None
                            else pipeline.transformer.config.out_channels
                        )
                        self.assertEqual(
                            outputs.shape,
                            (
                                batch_size,
                                out_channels,
                                height // pipeline.vae_scale_factor,
                                width // pipeline.vae_scale_factor,
                            ),
                        )
                    else:
                        packed_height = height // pipeline.vae_scale_factor // 2
                        packed_width = width // pipeline.vae_scale_factor // 2
                        channels = pipeline.transformer.config.in_channels
                        self.assertEqual(outputs.shape, (batch_size, packed_height * packed_width, channels))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        height, width, batch_size = 128, 128, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        for output_type in ["latent", "np", "pt"]:
            print(output_type)
            inputs["output_type"] = output_type

            ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ov_output, diffusers_output, atol=6e-3, rtol=1e-2)

        # test generation when input resolution nondevisible on 64
        height, width, batch_size = 96, 96, 1

        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        for output_type in ["latent", "np", "pt"]:
            print(output_type)
            inputs["output_type"] = output_type

            ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ov_output, diffusers_output, atol=6e-3, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        for generator_framework in ["np", "pt"]:
            ov_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ov_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ov_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertFalse(np.array_equal(ov_outputs_1.images[0], ov_outputs_3.images[0]))
            np.testing.assert_allclose(ov_outputs_1.images[0], ov_outputs_2.images[0], atol=1e-4, rtol=1e-2)

    @parameterized.expand(["stable-diffusion", "latent-consistency"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "katuni4ka/tiny-random-stable-diffusion-with-safety-checker", subfolder="safety_checker"
        )

        pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)

        self.assertIsInstance(pipeline.safety_checker, StableDiffusionSafetyChecker)
        self.assertIsInstance(ov_pipeline.safety_checker, StableDiffusionSafetyChecker)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_type=model_arch)

        ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED))
        diffusers_output = pipeline(**inputs, generator=get_generator("pt", SEED))

        ov_nsfw_content_detected = ov_output.nsfw_content_detected
        diffusers_nsfw_content_detected = diffusers_output.nsfw_content_detected

        self.assertTrue(ov_nsfw_content_detected is not None)
        self.assertTrue(diffusers_nsfw_content_detected is not None)
        self.assertEqual(ov_nsfw_content_detected, diffusers_nsfw_content_detected)

        ov_images = ov_output.images
        diffusers_images = diffusers_output.images

        np.testing.assert_allclose(ov_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_height_width_properties(self, model_arch: str):
        batch_size, height, width, num_images_per_prompt = 2, 128, 64, 4
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch], export=True, compile=False, dynamic_shapes=True
        )

        self.assertTrue(ov_pipeline.is_dynamic)
        self.assertEqual(ov_pipeline.batch_size, -1)
        self.assertEqual(ov_pipeline.height, -1)
        self.assertEqual(ov_pipeline.width, -1)

        ov_pipeline.reshape(
            batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt
        )

        self.assertFalse(ov_pipeline.is_dynamic)
        expected_batch = batch_size * num_images_per_prompt
        if (
            ov_pipeline.unet is not None
            and "timestep_cond" not in {inputs.get_any_name() for inputs in ov_pipeline.unet.model.inputs}
        ) or (
            ov_pipeline.transformer is not None
            and "txt_ids" not in {inputs.get_any_name() for inputs in ov_pipeline.transformer.model.inputs}
        ):
            expected_batch *= 2
        self.assertEqual(ov_pipeline.batch_size, expected_batch)
        self.assertEqual(ov_pipeline.height, height)
        self.assertEqual(ov_pipeline.width, width)

    @require_diffusers
    def test_textual_inversion(self):
        model_id = "katuni4ka/tiny-random-stable-diffusion-with-safety-checker"
        ti_id = "katuni4ka/textual_inversion_cat"

        inputs = self.generate_inputs(model_type="stable-diffusion")
        inputs["prompt"] = "A <cat-toy> backpack"

        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(model_id, safety_checker=None)
        diffusers_pipeline.load_textual_inversion(ti_id)

        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(model_id, compile=False, safety_checker=None)
        ov_pipeline.load_textual_inversion(ti_id)

        diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images
        ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images

        np.testing.assert_allclose(ov_output, diffusers_output, atol=1e-4, rtol=1e-2)


class OVPipelineForInpaintingTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl"]

    if is_transformers_version(">=", "4.40.0"):
        SUPPORTED_ARCHITECTURES.append("stable-diffusion-3")
        SUPPORTED_ARCHITECTURES.append("flux")
        SUPPORTED_ARCHITECTURES.append("flux-fill")

    AUTOMODEL_CLASS = AutoPipelineForInpainting
    OVMODEL_CLASS = OVPipelineForInpainting

    TASK = "inpainting"

    def generate_inputs(self, height=128, width=128, batch_size=1, channel=3, input_type="pil", model_arch=""):
        inputs = _generate_prompts(batch_size=batch_size)

        inputs["image"] = _generate_images(
            height=height, width=width, batch_size=batch_size, channel=channel, input_type=input_type
        )
        inputs["mask_image"] = _generate_images(
            height=height, width=width, batch_size=batch_size, channel=1, input_type=input_type
        )

        if model_arch != "flux-fill":
            inputs["strength"] = 0.75
        inputs["height"] = height
        inputs["width"] = width

        return inputs

    @require_diffusers
    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn(f"does not appear to have a file named {self.OVMODEL_CLASS.config_name}", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_ov_pipeline_class_dispatch(self, model_arch: str):
        if model_arch != "flux-fill":
            auto_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        else:
            from diffusers import FluxFillPipeline

            auto_pipeline = FluxFillPipeline.from_pretrained(MODEL_NAMES[model_arch])
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        self.assertEqual(ov_pipeline.auto_model_class, auto_pipeline.__class__)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_num_images_per_prompt(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        for batch_size in [1, 3]:
            for height in [64, 128]:
                for width in [64, 128]:
                    for num_images_per_prompt in [1, 3]:
                        inputs = self.generate_inputs(
                            height=height, width=width, batch_size=batch_size, model_arch=model_arch
                        )
                        outputs = pipeline(**inputs, num_images_per_prompt=num_images_per_prompt).images
                        self.assertEqual(outputs.shape, (batch_size * num_images_per_prompt, height, width, 3))

    @parameterized.expand(["stable-diffusion", "stable-diffusion-xl"])
    @require_diffusers
    def test_callback(self, model_arch: str):
        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        class Callback:
            def __init__(self):
                self.has_been_called = False
                self.number_of_steps = 0

            def __call__(self, *args, **kwargs) -> None:
                self.has_been_called = True
                self.number_of_steps += 1

        ov_pipe = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        auto_pipe = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        ov_callback = Callback()
        auto_callback = Callback()
        # callback_steps=1 to trigger callback every step
        ov_pipe(**inputs, callback=ov_callback, callback_steps=1)
        auto_pipe(**inputs, callback=auto_callback, callback_steps=1)

        self.assertTrue(ov_callback.has_been_called)
        self.assertEqual(ov_callback.number_of_steps, auto_callback.number_of_steps)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_shape(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 128, 64, 1

        for input_type in ["pil", "np", "pt"]:
            inputs = self.generate_inputs(
                height=height, width=width, batch_size=batch_size, input_type=input_type, model_arch=model_arch
            )

            for output_type in ["pil", "np", "pt", "latent"]:
                inputs["output_type"] = output_type
                outputs = pipeline(**inputs).images
                if output_type == "pil":
                    self.assertEqual((len(outputs), outputs[0].height, outputs[0].width), (batch_size, height, width))
                elif output_type == "np":
                    self.assertEqual(outputs.shape, (batch_size, height, width, 3))
                elif output_type == "pt":
                    self.assertEqual(outputs.shape, (batch_size, 3, height, width))
                else:
                    if not model_arch.startswith("flux"):
                        out_channels = (
                            pipeline.unet.config.out_channels
                            if pipeline.unet is not None
                            else pipeline.transformer.config.out_channels
                        )
                        self.assertEqual(
                            outputs.shape,
                            (
                                batch_size,
                                out_channels,
                                height // pipeline.vae_scale_factor,
                                width // pipeline.vae_scale_factor,
                            ),
                        )
                    else:
                        packed_height = height // pipeline.vae_scale_factor // 2
                        packed_width = width // pipeline.vae_scale_factor // 2
                        channels = (
                            pipeline.transformer.config.in_channels
                            if model_arch != "flux-fill"
                            else pipeline.transformer.out_channels
                        )
                        self.assertEqual(outputs.shape, (batch_size, packed_height * packed_width, channels))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_compare_to_diffusers_pipeline(self, model_arch: str):
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        if model_arch != "flux-fill":
            diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])
        else:
            from diffusers import FluxFillPipeline

            diffusers_pipeline = FluxFillPipeline.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_arch=model_arch)

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ov_output, diffusers_output, atol=6e-3, rtol=1e-2)

        # test generation when input resolution nondevisible on 64
        height, width, batch_size = 96, 96, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_arch=model_arch)

        for output_type in ["latent", "np", "pt"]:
            inputs["output_type"] = output_type

            ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images
            diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images

            np.testing.assert_allclose(ov_output, diffusers_output, atol=6e-3, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_diffusers
    def test_image_reproducibility(self, model_arch: str):
        pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch])

        height, width, batch_size = 64, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size, model_arch=model_arch)

        for generator_framework in ["np", "pt"]:
            ov_outputs_1 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ov_outputs_2 = pipeline(**inputs, generator=get_generator(generator_framework, SEED))
            ov_outputs_3 = pipeline(**inputs, generator=get_generator(generator_framework, SEED + 1))

            self.assertFalse(np.array_equal(ov_outputs_1.images[0], ov_outputs_3.images[0]))
            np.testing.assert_allclose(ov_outputs_1.images[0], ov_outputs_2.images[0], atol=1e-4, rtol=1e-2)

    @parameterized.expand(["stable-diffusion"])
    @require_diffusers
    def test_safety_checker(self, model_arch: str):
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "katuni4ka/tiny-random-stable-diffusion-with-safety-checker", subfolder="safety_checker"
        )

        pipeline = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], safety_checker=safety_checker)

        self.assertIsInstance(pipeline.safety_checker, StableDiffusionSafetyChecker)
        self.assertIsInstance(ov_pipeline.safety_checker, StableDiffusionSafetyChecker)

        height, width, batch_size = 32, 64, 1
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)

        ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED))
        diffusers_output = pipeline(**inputs, generator=get_generator("pt", SEED))

        ov_nsfw_content_detected = ov_output.nsfw_content_detected
        diffusers_nsfw_content_detected = diffusers_output.nsfw_content_detected

        self.assertTrue(ov_nsfw_content_detected is not None)
        self.assertTrue(diffusers_nsfw_content_detected is not None)
        self.assertEqual(ov_nsfw_content_detected, diffusers_nsfw_content_detected)

        ov_images = ov_output.images
        diffusers_images = diffusers_output.images

        np.testing.assert_allclose(ov_images, diffusers_images, atol=1e-4, rtol=1e-2)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_height_width_properties(self, model_arch: str):
        batch_size, height, width, num_images_per_prompt = 2, 128, 64, 4
        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch], export=True, compile=False, dynamic_shapes=True
        )

        self.assertTrue(ov_pipeline.is_dynamic)
        self.assertEqual(ov_pipeline.batch_size, -1)
        self.assertEqual(ov_pipeline.height, -1)
        self.assertEqual(ov_pipeline.width, -1)

        ov_pipeline.reshape(
            batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt
        )

        self.assertFalse(ov_pipeline.is_dynamic)
        expected_batch = batch_size * num_images_per_prompt
        if (
            ov_pipeline.unet is not None
            and "timestep_cond" not in {inputs.get_any_name() for inputs in ov_pipeline.unet.model.inputs}
        ) or (
            ov_pipeline.transformer is not None
            and "txt_ids" not in {inputs.get_any_name() for inputs in ov_pipeline.transformer.model.inputs}
        ):
            expected_batch *= 2
        self.assertEqual(
            ov_pipeline.batch_size,
            expected_batch,
        )
        self.assertEqual(ov_pipeline.height, height)
        self.assertEqual(ov_pipeline.width, width)

    @require_diffusers
    def test_textual_inversion(self):
        model_id = "katuni4ka/tiny-random-stable-diffusion-with-safety-checker"
        ti_id = "katuni4ka/textual_inversion_cat"

        inputs = self.generate_inputs()
        inputs["prompt"] = "A <cat-toy> backpack"

        diffusers_pipeline = self.AUTOMODEL_CLASS.from_pretrained(model_id, safety_checker=None)
        diffusers_pipeline.load_textual_inversion(ti_id)

        ov_pipeline = self.OVMODEL_CLASS.from_pretrained(model_id, compile=False, safety_checker=None)
        ov_pipeline.load_textual_inversion(ti_id)

        diffusers_output = diffusers_pipeline(**inputs, generator=get_generator("pt", SEED)).images
        ov_output = ov_pipeline(**inputs, generator=get_generator("pt", SEED)).images

        np.testing.assert_allclose(ov_output, diffusers_output, atol=1e-4, rtol=1e-2)
