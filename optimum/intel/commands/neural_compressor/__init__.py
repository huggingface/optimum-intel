import sys
from argparse import ArgumentParser

from .. import BaseOptimumIntelCLICommand
from .quantize import INCQuantizeCommand, parse_args_inc_quantize


def inc_quantize_factory(args):
    return INCQuantizeCommand(args)


class INCCommand(BaseOptimumIntelCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        inc_parser = parser.add_parser("inc", help="INC quantize utilities.")
        inc_sub_parsers = inc_parser.add_subparsers()

        quantize_parser = inc_sub_parsers.add_parser("quantize", help="INC dynamic quantization.")

        parse_args_inc_quantize(quantize_parser)

        quantize_parser.set_defaults(func=inc_quantize_factory)

    def run(self):
        raise NotImplementedError()
