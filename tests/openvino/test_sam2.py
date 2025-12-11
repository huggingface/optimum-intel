import unittest
from types import SimpleNamespace

from functools import partial

from optimum.exporters.openvino.model_configs import (
    Sam2VideoMaskDecoderOpenVINOConfig,
    Sam2VideoPromptEncoderOpenVINOConfig,
    Sam2VideoVisionEncoderOpenVINOConfig,
)
from optimum.exporters.tasks import TasksManager  # type: ignore[attr-defined]
from optimum.intel.openvino.configuration import _DEFAULT_4BIT_WQ_CONFIGS


class _DummySam2VideoModel:
    def __init__(self):
        self.config = SimpleNamespace(model_type="sam2_video")


class Sam2VideoRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.model = _DummySam2VideoModel()

    def test_vision_encoder_config_registered(self):
        ctor = TasksManager.get_exporter_config_constructor(
            model=self.model,
            exporter="openvino",
            library_name="transformers",
            task="feature-extraction",
            model_type="sam2video_vision_encoder",
        )
        if isinstance(ctor, partial):
            self.assertIs(ctor.func, Sam2VideoVisionEncoderOpenVINOConfig)
        else:
            self.assertIs(ctor, Sam2VideoVisionEncoderOpenVINOConfig)

    def test_prompt_encoder_config_registered(self):
        ctor = TasksManager.get_exporter_config_constructor(
            model=self.model,
            exporter="openvino",
            library_name="transformers",
            task="feature-extraction",
            model_type="sam2video_prompt_encoder",
        )
        if isinstance(ctor, partial):
            self.assertIs(ctor.func, Sam2VideoPromptEncoderOpenVINOConfig)
        else:
            self.assertIs(ctor, Sam2VideoPromptEncoderOpenVINOConfig)

    def test_mask_decoder_config_registered(self):
        ctor = TasksManager.get_exporter_config_constructor(
            model=self.model,
            exporter="openvino",
            library_name="transformers",
            task="image-segmentation",
            model_type="sam2video_mask_decoder",
        )
        if isinstance(ctor, partial):
            self.assertIs(ctor.func, Sam2VideoMaskDecoderOpenVINOConfig)
        else:
            self.assertIs(ctor, Sam2VideoMaskDecoderOpenVINOConfig)


class Sam2QuantDefaultsTest(unittest.TestCase):
    def test_default_4bit_quant_config_registered(self):
        self.assertIn("facebook/sam2.1-hiera-small", _DEFAULT_4BIT_WQ_CONFIGS)
        config = _DEFAULT_4BIT_WQ_CONFIGS["facebook/sam2.1-hiera-small"]
        self.assertEqual(config.get("bits"), 4)
        self.assertEqual(config.get("group_size"), 32)
        self.assertFalse(config.get("sym"))
        self.assertEqual(config.get("ratio"), 1.0)
