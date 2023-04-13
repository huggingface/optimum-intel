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

    def test_cli(self):
        # clone latest run_glue.py from transformers repo

        with tempfile.TemporaryDirectory() as tempdir:
            # script_path = os.path.join(tempdir, "run_glue.py")
            # r = requests.get(url)
            # f = open(script_path, "wb")
            # f.write(r.content)
            # f.close()

            export_commands = [
                f"optimum-cli inc quantize --model hf-internal-testing/tiny-random-bert --output {tempdir}/bert --task fill-mask",
                f"optimum-cli inc quantize --model hf-internal-testing/tiny-random-distilbert --output {tempdir}/distilbert --task text-classification",
                f"optimum-cli inc quantize --model hf-internal-testing/tiny-random-gpt2 --output {tempdir}/gpt2 --task text-generation",
                # f"optimum-cli inc quantize {script_path} --model_name_or_path hf-internal-testing/tiny-random-bert --task_name mrpc --do_eval --output_dir {tempdir}/bert_mrpc",
                # f"optimum-cli inc quantize {script_path} --model_name_or_path hf-internal-testing/tiny-random-distilbert --task_name sst2 --do_eval --output_dir {tempdir}/distilbert_sst2",
            ]

            for export in export_commands:
                subprocess.run(export, shell=True, check=True)
