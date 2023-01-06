import unittest

from optimum.intel.neural_compressor.neural_coder_adaptor import NeuralCoderAdaptor


class NeuralCoderAdaptorTest(unittest.TestCase):
    def test_string_type(self):
        dynamic_api = NeuralCoderAdaptor.default_quant_dynamic
        static_api = NeuralCoderAdaptor.default_quant_static
        self.assertEqual(type(dynamic_api), type(""))
        self.assertEqual(type(static_api), type(""))
