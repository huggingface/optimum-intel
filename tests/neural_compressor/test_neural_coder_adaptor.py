import unittest
from argparse import Namespace

from neural_coder.launcher import Launcher
from optimum.intel.neural_compressor.neural_coder_adaptor import NeuralCoderAdaptor


class NeuralCoderAdaptorTest(unittest.TestCase):
    def test_string_type(self):
        dynamic_api = NeuralCoderAdaptor.default_quant_dynamic
        static_api = NeuralCoderAdaptor.default_quant_static
        self.assertEqual(type(dynamic_api), type(""))
        self.assertEqual(type(static_api), type(""))

    def test_launcher(self):
        args = Namespace(
            opt="",
            approach="auto",
            config="",
            bench=False,
            enable=True,
            script="run_glue_source.py",
        )

        modular_pattern = {}

        modular_pattern["pytorch_inc_static_quant_fx"] = NeuralCoderAdaptor.default_quant_static
        modular_pattern["pytorch_inc_dynamic_quant"] = NeuralCoderAdaptor.default_quant_dynamic
        modular_pattern["inc_auto"] = NeuralCoderAdaptor.default_quant_dynamic

        Launcher.execute(args, use_modular=True, modular_pattern=modular_pattern, use_inc=False)

        # determine if the code optimization is correct
        import filecmp

        self.assertEqual(True, filecmp.cmp("run_glue_target.py", "run_glue_source_optimized.py"))
