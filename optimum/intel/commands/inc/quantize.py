from argparse import REMAINDER


def parse_args_inc_quantize(parser):
    parser.add_argument("-o", "--opt", type=str, default="", help="optimization feature to enable")

    parser.add_argument("-a", "--approach", type=str, default="auto", help="quantization approach (strategy)")

    parser.add_argument("--config", type=str, default="", help="quantization configuration file path")

    parser.add_argument(
        "-b", "--bench", default=False, action="store_true", help="conduct auto_quant benchmark instead of enable"
    )

    parser.add_argument(
        "-e", "--enable", default=False, action="store_true", help="only do enable, not overwrite or run program"
    )

    # positional
    parser.add_argument(
        "script",
        type=str,
        help="The full path to the script to be launched. " "followed by all the arguments for the script",
    )

    # script args
    parser.add_argument("script_args", nargs=REMAINDER)


from neural_coder.launcher import Launcher

from ...neural_compressor.neural_coder_adaptor import NeuralCoderAdaptor


class INCQuantizeCommand:
    def __init__(self, args):
        self.args = args

    def run(self):
        modular_pattern = {}

        modular_pattern["pytorch_inc_static_quant_fx"] = NeuralCoderAdaptor.default_quant_static
        modular_pattern["pytorch_inc_dynamic_quant"] = NeuralCoderAdaptor.default_quant_dynamic
        modular_pattern["inc_auto"] = NeuralCoderAdaptor.default_quant_dynamic

        Launcher.execute(self.args, use_modular=True, modular_pattern=modular_pattern, use_inc=False)
