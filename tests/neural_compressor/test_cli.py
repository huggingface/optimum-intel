import subprocess
import tempfile
import unittest

from optimum.intel.neural_compressor.neural_coder_adaptor import NeuralCoderAdaptor


class NeuralCoderAdaptorTest(unittest.TestCase):
    def test_string_type(self):
        dynamic_api = NeuralCoderAdaptor.default_quant_dynamic
        static_api = NeuralCoderAdaptor.default_quant_static
        self.assertIsInstance(dynamic_api, str)
        self.assertIsInstance(static_api, str)

    @unittest.skip(reason="INC is going to deprecate, skip this failed test")
    def test_cli(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # TODO : enable
            # script_path = os.path.join(tempdir, "run_glue.py")
            # r = requests.get(url)
            # f = open(script_path, "wb")
            # f.write(r.content)
            # f.close()

            export_commands = [
                f"optimum-cli inc quantize --model hf-internal-testing/tiny-random-bert --output {tempdir}/bert --task fill-mask",
                f"optimum-cli inc quantize --model hf-internal-testing/tiny-random-distilbert --output {tempdir}/distilbert --task text-classification",
                f"optimum-cli inc quantize --model hf-internal-testing/tiny-random-gpt2 --output {tempdir}/gpt2 --task text-generation",
                f"optimum-cli inc quantize --model distilbert-base-cased-distilled-squad --output {tempdir}/distilbert_squad",
            ]

            for export in export_commands:
                subprocess.run(export, shell=True, check=True)
