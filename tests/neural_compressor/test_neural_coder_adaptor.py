import os
import subprocess
import tempfile
import unittest

import requests
from optimum.intel.neural_compressor.neural_coder_adaptor import NeuralCoderAdaptor


class NeuralCoderAdaptorTest(unittest.TestCase):
    def test_string_type(self):
        dynamic_api = NeuralCoderAdaptor.default_quant_dynamic
        static_api = NeuralCoderAdaptor.default_quant_static
        self.assertIsInstance(dynamic_api, str)
        self.assertIsInstance(static_api, str)

    def test_cli(self):
        # clone latest run_glue.py from transformers repo
        url = "https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/text-classification/run_glue.py"

        with tempfile.TemporaryDirectory() as tempdir:
            script_path = os.path.join(tempdir, "run_glue.py")
            r = requests.get(url)
            f = open(script_path, "wb")
            f.write(r.content)
            f.close()

            # First export a tiny encoder, decoder only and encoder-decoder
            export_commands = [
                f"optimum-intel-cli inc quantize {script_path} --model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir {tempdir}/bert",
                f"optimum-intel-cli inc quantize {script_path} --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english --task_name sst2 --do_eval --output_dir {tempdir}/distilbert",
            ]

            for export in export_commands:
                subprocess.run(export, shell=True, check=True)
