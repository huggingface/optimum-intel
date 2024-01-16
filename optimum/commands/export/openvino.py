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
"""Defines the command line for the export with OpenVINO."""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...exporters import TasksManager
from ..base import BaseOptimumCLICommand, CommandInfo


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_openvino(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store the generated OV model."
    )
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
            f" {str(TasksManager.get_all_tasks())}. For decoder models, use `xxx-with-past` to export the model using past key values in the decoder."
        ),
    )
    optional_group.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    optional_group.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        default=None,
        help=(
            "The framework to use for the export. If not provided, will attempt to use the local checkpoint's original framework or what is available in the environment."
        ),
    )
    optional_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which "
            "you have read the code, as it will execute on your local machine arbitrary code present in the model repository."
        ),
    )
    optional_group.add_argument(
        "--pad-token-id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it."
        ),
    )
    optional_group.add_argument("--fp16", action="store_true", help="Compress weights to fp16")
    optional_group.add_argument("--int8", action="store_true", help="Compress weights to int8")
    optional_group.add_argument(
        "--weight-format",
        type=str,
        choices=["fp32", "fp16", "int8", "int4_sym_g128", "int4_asym_g128", "int4_sym_g64", "int4_asym_g64"],
        default=None,
        help=(
            "The weight format of the exporting model, e.g. f32 stands for float32 weights, f16 - for float16 weights, i8 - INT8 weights, int4_* - for INT4 compressed weights."
        ),
    )
    optional_group.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help=(
            "Compression ratio between primary and backup precision. In the case of INT4, NNCF evaluates layer sensitivity and keeps the most impactful layers in INT8"
            "precision (by default 20%% in INT8). This helps to achieve better accuracy after weight compression."
        ),
    )
    optional_group.add_argument(
        "--disable-stateful",
        action="store_true",
        help=(
            "Disable stateful converted models, stateless models will be generated instead. Stateful models are produced by default when this key is not used. "
            "In stateful models all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. "
            "If --disable-stateful option is used, it may result in sub-optimal inference performance. "
            "Use it when you intentionally want to use a stateless model, for example, to be compatible with existing "
            "OpenVINO native inference code that expects kv-cache inputs and outputs in the model."
        ),
    )


class OVExportCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="openvino", help="Export PyTorch models to OpenVINO IR.")

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
        return parse_args_openvino(parser)

    def run(self):
        from ...exporters.openvino.__main__ import main_export

        if self.args.fp16:
            logger.warning(
                "`--fp16` option is deprecated and will be removed in a future version. Use `--weight-format` instead."
            )
            self.args.weight_format = "fp16"
        if self.args.int8:
            logger.warning(
                "`--int8` option is deprecated and will be removed in a future version. Use `--weight-format` instead."
            )
            self.args.weight_format = "int8"

        # TODO : add input shapes
        main_export(
            model_name_or_path=self.args.model,
            output=self.args.output,
            task=self.args.task,
            framework=self.args.framework,
            cache_dir=self.args.cache_dir,
            trust_remote_code=self.args.trust_remote_code,
            pad_token_id=self.args.pad_token_id,
            compression_option=self.args.weight_format,
            compression_ratio=self.args.ratio,
            stateful=not self.args.disable_stateful,
            # **input_shapes,
        )
