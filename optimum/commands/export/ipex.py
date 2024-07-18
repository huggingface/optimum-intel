# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Defines the command line for the export with IPEX."""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from ...exporters import TasksManager
from ..base import BaseOptimumCLICommand, CommandInfo


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace, _SubParsersAction


def parse_args_ipex(parser: "ArgumentParser"):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store the generated IPEX model."
    )
    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model."
        ),
    )
    optional_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which "
            "you have read the code, as it will execute on your local machine arbitrary code present in the model repository."
        ),
    )
    optional_group.add_argument(
        "--revision",
        default=None,
        help="The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.",
    )
    optional_group.add_argument(
        "--token",
        default=None,
        help="The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).",
    )
    optional_group.add_argument(
        "--cache_dir",
        type=str,
        default=HUGGINGFACE_HUB_CACHE,
        help="Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.",
    )
    optional_group.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can specify the folder name here.",
    )
    optional_group.add_argument(
        "--local_files_only",
        type=bool,
        default=False,
        help="Whether or not to only look at local files (i.e., do not try to download the model).",
    )
    optional_group.add_argument(
        "--force_download",
        type=bool,
        default=False,
        help="Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.",
    )
    optional_group.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        help="float16 or bfloat16 or float32: load in a specified dtype, ignoring the model’s config.torch_dtype if one exists. If not specified, the model will get loaded in float32.",
    )


class IPEXExportCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="ipex", help="Export PyTorch models to IPEX IR.")

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
        return parse_args_ipex(parser)

    def run(self):
        import torch

        from optimum.intel.ipex.utils import _HEAD_TO_AUTOMODELS

        if self.args.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.args.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        model_kwargs = {
            "revision": self.args.revision,
            "token": self.args.token,
            "cache_dir": self.args.cache_dir,
            "subfolder": self.args.subfolder,
            "local_files_only": self.args.local_files_only,
            "force_download": self.args.force_download,
            "commit_hash": self.args.commit_hash,
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.args.trust_remote_code,
        }

        task = TasksManager.infer_task_from_model(self.args.model) if self.args.task == "auto" else self.args.task
        if task not in _HEAD_TO_AUTOMODELS:
            raise ValueError(f"{task} is not supported, please choose from {_HEAD_TO_AUTOMODELS}")

        model_class = _HEAD_TO_AUTOMODELS[task]
        model = eval(model_class).from_pretrained(self.args.model, **model_kwargs)
        model.save_pretrained(self.args.output)
