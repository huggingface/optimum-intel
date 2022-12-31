from neural_coder.launcher import Launcher

from .neural_compressor.neural_coder_adaptor import NeuralCoderAdaptor


args = Launcher.parse_args()

modular_pattern = {}

modular_pattern["pytorch_inc_static_quant_fx"] = NeuralCoderAdaptor.default_quant_static
modular_pattern["pytorch_inc_dynamic_quant"] = NeuralCoderAdaptor.default_quant_dynamic
modular_pattern["inc_auto"] = NeuralCoderAdaptor.default_quant_dynamic

Launcher.execute(args, use_modular=True, modular_pattern=modular_pattern)
