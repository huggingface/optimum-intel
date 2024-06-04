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

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from ...exporters import TasksManager
from ...intel.utils.import_utils import DIFFUSERS_IMPORT_ERROR, is_diffusers_available
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
    optional_group.add_argument(
        "--cache_dir", type=str, default=HUGGINGFACE_HUB_CACHE, help="Path indicating where to store cache."
    )
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
        choices=["fp32", "fp16", "int8", "int4", "int4_sym_g128", "int4_asym_g128", "int4_sym_g64", "int4_asym_g64"],
        default=None,
        help=(
            "The weight format of the exporting model, e.g. f32 stands for float32 weights, f16 - for float16 weights, i8 - INT8 weights, int4_* - for INT4 compressed weights."
        ),
    )
    optional_group.add_argument(
        "--ratio",
        type=float,
        default=None,
        help=(
            "Compression ratio between primary and backup precision. In the case of INT4, NNCF evaluates layer sensitivity and keeps the most impactful layers in INT8"
            "precision (by default 20%% in INT8). This helps to achieve better accuracy after weight compression."
        ),
    )
    optional_group.add_argument(
        "--sym",
        action="store_true",
        default=None,
        help=("Whether to apply symmetric quantization"),
    )
    optional_group.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=("The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization."),
    )
    optional_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "The dataset used for data-aware compression or quantization with NNCF. "
            "You can use the one from the list ['wikitext2','c4','c4-new','ptb','ptb-new'] for LLLMs "
            "or ['conceptual_captions','laion/220k-GPT4Vision-captions-from-LIVIS','laion/filtered-wit'] for diffusion models."
        ),
    )
    optional_group.add_argument(
        "--all-layers",
        action="store_true",
        default=None,
        help=(
            "Whether embeddings and last MatMul layers should be compressed to INT4. If not provided an weight "
            "compression is applied, they are compressed to INT8."
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
    optional_group.add_argument(
        "--disable-convert-tokenizer",
        action="store_true",
        help="Do not add converted tokenizer and detokenizer OpenVINO models.",
    )
    optional_group.add_argument(
        "--convert-tokenizer",
        action="store_true",
        help="[Deprecated] Add converted tokenizer and detokenizer with OpenVINO Tokenizers.",
    )

    optional_group.add_argument(
        "--library",
        type=str,
        choices=["transformers", "diffusers", "timm", "sentence_transformers"],
        default=None,
        help=("The library on the model. If not provided, will attempt to infer the local checkpoint's library"),
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
        from ...intel.openvino.configuration import _DEFAULT_4BIT_CONFIGS, OVConfig

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

        if self.args.weight_format is None:
            ov_config = None
        elif self.args.weight_format in {"fp16", "fp32"}:
            ov_config = OVConfig(dtype=self.args.weight_format)
        else:
            is_int8 = self.args.weight_format == "int8"

            # For int4 quantization if not parameter is provided, then use the default config if exist
            if (
                not is_int8
                and self.args.ratio is None
                and self.args.group_size is None
                and self.args.sym is None
                and self.args.all_layers is None
                and self.args.model in _DEFAULT_4BIT_CONFIGS
            ):
                quantization_config = _DEFAULT_4BIT_CONFIGS[self.args.model]
            else:
                quantization_config = {
                    "bits": 8 if is_int8 else 4,
                    "ratio": 1 if is_int8 else (self.args.ratio or 0.8),
                    "sym": self.args.sym or False,
                    "group_size": -1 if is_int8 else self.args.group_size,
                    "all_layers": None if is_int8 else self.args.all_layers,
                }

            if self.args.weight_format in {"int4_sym_g128", "int4_asym_g128", "int4_sym_g64", "int4_asym_g64"}:
                logger.warning(
                    f"--weight-format {self.args.weight_format} is deprecated, possible choices are fp32, fp16, int8, int4"
                )
                quantization_config["sym"] = "asym" not in self.args.weight_format
                quantization_config["group_size"] = 128 if "128" in self.args.weight_format else 64
            quantization_config["dataset"] = self.args.dataset
            ov_config = OVConfig(quantization_config=quantization_config)

        library_name = TasksManager.infer_library_from_model(self.args.model, library_name=self.args.library)
        if library_name == "sentence_transformers" and self.args.library is None:
            logger.warning(
                "Library name is not specified. There are multiple possible variants: `sentence_transformers`, `transformers`."
                "`transformers` will be selected. If you want to load your model with the `sentence-transformers` library instead, please set --library sentence_transformers"
            )
            library_name = "transformers"

        if self.args.convert_tokenizer:
            logger.warning("`--convert-tokenizer` option is deprecated. Tokenizer will be converted by default.")

        if (
            library_name == "diffusers"
            and ov_config
            and ov_config.quantization_config
            and ov_config.quantization_config.dataset is not None
        ):
            if not is_diffusers_available():
                raise ValueError(DIFFUSERS_IMPORT_ERROR.format("Export of diffusers models"))

            from diffusers import DiffusionPipeline

            diffusers_config = DiffusionPipeline.load_config(self.args.model)
            class_name = diffusers_config.get("_class_name", None)

            if class_name == "LatentConsistencyModelPipeline":
                from optimum.intel import OVLatentConsistencyModelPipeline

                model_cls = OVLatentConsistencyModelPipeline

            elif class_name == "StableDiffusionXLPipeline":
                from optimum.intel import OVStableDiffusionXLPipeline

                model_cls = OVStableDiffusionXLPipeline
            elif class_name == "StableDiffusionPipeline":
                from optimum.intel import OVStableDiffusionPipeline

                model_cls = OVStableDiffusionPipeline
            else:
                raise NotImplementedError(f"Quantization in hybrid mode isn't supported for class {class_name}.")

            model = model_cls.from_pretrained(
                self.args.model, export=True, quantization_config=ov_config.quantization_config
            )
            model.save_pretrained(self.args.output)

            if self.args.disable_convert_tokenizer:
                return

            # avoid import when using other exporters (IPEX, INC)
            from ...exporters.openvino.convert import export_tokenizer

            output = Path(self.args.output)
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is not None:
                export_tokenizer(tokenizer, output / "tokenizer")

            tokenizer_2 = getattr(model, "tokenizer_2", None)
            if tokenizer_2 is not None:
                export_tokenizer(tokenizer_2, output / "tokenizer_2")
        else:
            # TODO : add input shapes
            main_export(
                model_name_or_path=self.args.model,
                output=self.args.output,
                task=self.args.task,
                framework=self.args.framework,
                cache_dir=self.args.cache_dir,
                trust_remote_code=self.args.trust_remote_code,
                pad_token_id=self.args.pad_token_id,
                ov_config=ov_config,
                stateful=not self.args.disable_stateful,
                convert_tokenizer=not self.args.disable_convert_tokenizer,
                library_name=library_name,
                # **input_shapes,
            )
