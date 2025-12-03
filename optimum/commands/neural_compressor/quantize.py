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

from pathlib import Path
from typing import TYPE_CHECKING

from optimum.commands.base import BaseOptimumCLICommand
from optimum.utils.constant import ALL_TASKS


if TYPE_CHECKING:
    from argparse import ArgumentParser


def parse_args_inc_quantize(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m",
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
            "The task to export the model for. If not specified, the task will be auto-inferred from the model's metadata or files. "
            "For tasks that generate text, add the `xxx-with-past` suffix to export the model using past key values caching. "
            f"Available tasks depend on the model, but are among the following list: {ALL_TASKS}."
        ),
    )


class INCQuantizeCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_inc_quantize(parser)

    def run(self):
        from neural_compressor.config import PostTrainingQuantConfig

        from optimum.exporters.tasks import TasksManager
        from optimum.intel.neural_compressor import INCQuantizer

        save_dir = self.args.output
        model_id = self.args.model
        task = self.args.task

        if save_dir == model_id:
            raise ValueError("The output directory must be different than the directory hosting the model.")

        if task == "auto":
            try:
                task = TasksManager.infer_task_from_model(model_id)
            except Exception as e:
                raise ValueError(
                    "The task could not be inferred automatically. Please provide the task using the --task argument."
                ) from e

        model = TasksManager.get_model_from_task(task, model_id)

        quantization_config = PostTrainingQuantConfig(approach="dynamic")
        quantizer = INCQuantizer.from_pretrained(model)
        quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)
