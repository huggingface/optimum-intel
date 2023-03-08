import unittest
from argparse import Namespace

import requests
from neural_coder.launcher import Launcher
from optimum.intel.neural_compressor.neural_coder_adaptor import NeuralCoderAdaptor


class NeuralCoderAdaptorTest(unittest.TestCase):
    def test_string_type(self):
        dynamic_api = NeuralCoderAdaptor.default_quant_dynamic
        static_api = NeuralCoderAdaptor.default_quant_static
        self.assertEqual(type(dynamic_api), type(""))
        self.assertEqual(type(static_api), type(""))

    def test_launcher(self):
        # clone latest run_glue.py from transformers repo
        url = "https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/text-classification/run_glue.py"
        r = requests.get(url)
        f = open("run_glue.py", "wb")
        f.write(r.content)

        args = Namespace(
            opt="",
            approach="auto",
            config="",
            bench=False,
            enable=True,
            script="run_glue.py",
        )

        modular_pattern = {}

        modular_pattern["pytorch_inc_static_quant_fx"] = NeuralCoderAdaptor.default_quant_static
        modular_pattern["pytorch_inc_dynamic_quant"] = NeuralCoderAdaptor.default_quant_dynamic
        modular_pattern["inc_auto"] = NeuralCoderAdaptor.default_quant_dynamic

        Launcher.execute(args, use_modular=True, modular_pattern=modular_pattern, use_inc=False)

        # to-add: execute "run_glue_optimized.py" and see if the saved model can correctly perform inference?
        self.assertEqual(0, 0)
