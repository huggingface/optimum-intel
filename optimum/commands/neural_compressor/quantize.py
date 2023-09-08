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
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...exporters import TasksManager
from ..base import BaseOptimumCLICommand, CommandInfo


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_inc_quantize(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the repository where the model to quantize is located.",
    )
    required_group.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to the directory where to store generated quantized model.",
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
            f" {str(TasksManager.get_all_tasks())}."
        ),
    )


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
        from neural_compressor.config import PostTrainingQuantConfig

        from ...intel.neural_compressor import INCQuantizer

        save_dir = self.args.output
        model_id = self.args.model
        task = self.args.task

        if save_dir == model_id:
            raise ValueError("The output directory must be different than the directory hosting the model.")

        if task == "auto":
            try:
                task = TasksManager.infer_task_from_model(model_id)
            except Exception as e:
                return (
                    f"### Error: {e}. Please pass explicitely the task as it could not be infered.",
                    None,
                )

        model = TasksManager.get_model_from_task(task, model_id)

        quantization_config = PostTrainingQuantConfig(approach="dynamic")
        quantizer = INCQuantizer.from_pretrained(model)
        quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)
