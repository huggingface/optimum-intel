import inspect
import unittest


class Flux2KleinOpenVINOSupportTest(unittest.TestCase):
    def test_pipeline_import(self):
        from optimum.intel.openvino.modeling_diffusion import OVFlux2KleinPipeline

        self.assertEqual(OVFlux2KleinPipeline.__name__, "OVFlux2KleinPipeline")

    def test_pipeline_registration(self):
        from optimum.intel.openvino import modeling_diffusion
        from optimum.intel.openvino.modeling_diffusion import OVFlux2KleinPipeline

        source = inspect.getsource(modeling_diffusion)
        self.assertIn("SUPPORTED_OV_PIPELINES.append(OVFlux2KleinPipeline)", source)
        self.assertIn('OV_TEXT2IMAGE_PIPELINES_MAPPING["flux2-klein"] = OVFlux2KleinPipeline', source)

        if getattr(modeling_diffusion, "Flux2KleinPipeline", object) is not object:
            self.assertIn(OVFlux2KleinPipeline, modeling_diffusion.SUPPORTED_OV_PIPELINES)
            self.assertIs(modeling_diffusion.OV_TEXT2IMAGE_PIPELINES_MAPPING["flux2-klein"], OVFlux2KleinPipeline)

    def test_model_config_import(self):
        from optimum.exporters.openvino.model_configs import Flux2KleinTransformerOpenVINOConfig

        self.assertEqual(Flux2KleinTransformerOpenVINOConfig.__name__, "Flux2KleinTransformerOpenVINOConfig")

    def test_qwen3_diffusion_patcher_import(self):
        from optimum.exporters.openvino.model_patcher import Qwen3DiffusionTextEncoderModelPatcher

        self.assertEqual(Qwen3DiffusionTextEncoderModelPatcher.__name__, "Qwen3DiffusionTextEncoderModelPatcher")

    def test_export_routing_source_contains_flux2_klein(self):
        from optimum.exporters.openvino.convert import get_diffusion_models_for_export_ext

        source = inspect.getsource(get_diffusion_models_for_export_ext)
        self.assertIn('is_flux2_klein = pipeline.__class__.__name__ == "Flux2KleinPipeline"', source)
        self.assertIn("get_flux2_klein_models_for_export", source)


if __name__ == "__main__":
    unittest.main()
