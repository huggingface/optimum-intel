# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from argparse import REMAINDER
from typing import TYPE_CHECKING, Optional

from ...neural_compressor.launcher import _quantize
from ..base import BaseOptimumCLICommand, CommandInfo


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_inc_quantize(parser: "ArgumentParser"):
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


class INCQuantizeCommand(BaseOptimumCLICommand):
    def __init__(
        self,
        subparsers: "_SubParsersAction",
        args: Optional["Namespace"] = None,
        command: Optional["CommandInfo"] = None,
        from_defaults_factory: bool = False,
        parser: Optional["ArgumentParser"] = None,
    ):
        super().__init__(
            subparsers, args=args, command=command, from_defaults_factory=from_defaults_factory, parser=parser
        )
        self.args_string = " ".join(sys.argv[3:])

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_inc_quantize(parser)

    def run(self):
        _quantize(self.args)
