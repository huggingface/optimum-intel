from neural_coder.launcher import Launcher

from .neural_coder_adaptor import NeuralCoderAdaptor


def _quantize(args):
    modular_pattern = {
        "pytorch_inc_static_quant_fx": NeuralCoderAdaptor.default_quant_static,
        "pytorch_inc_dynamic_quant": NeuralCoderAdaptor.default_quant_dynamic,
        "inc_auto": NeuralCoderAdaptor.default_quant_dynamic,
    }

    Launcher.execute(args, use_modular=True, modular_pattern=modular_pattern, use_inc=False)
