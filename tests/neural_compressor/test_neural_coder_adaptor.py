import contextlib
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
        f.close()

        args = Namespace(
            opt="",
            approach="auto",
            config="",
            bench=False,
            enable=False,
            script="run_glue.py",
            script_args="--model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result",
        )

        modular_pattern = {}

        modular_pattern["pytorch_inc_static_quant_fx"] = NeuralCoderAdaptor.default_quant_static
        modular_pattern["pytorch_inc_dynamic_quant"] = NeuralCoderAdaptor.default_quant_dynamic
        modular_pattern["inc_auto"] = NeuralCoderAdaptor.default_quant_dynamic

        execution_status = ""

        # execute launcher-optimized "run_glue.py" and see if it runs finely
        try:
            Launcher.execute(args, use_modular=True, modular_pattern=modular_pattern, use_inc=False)
            print("Execution of optimized code has succeeded.")
            execution_status = "pass"
        except Exception as e:
            print("Execution of optimized code has failed.")
            print("Error: ", e)
            execution_status = "fail"

        self.assertEqual(execution_status, "pass")
