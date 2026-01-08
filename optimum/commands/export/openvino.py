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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from optimum.commands.base import BaseOptimumCLICommand, CommandInfo
from optimum.utils.constant import ALL_TASKS


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from argparse import ArgumentParser


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
            "The task to export the model for. If not specified, the task will be auto-inferred from the model's metadata or files. "
            "For tasks that generate text, add the `xxx-with-past` suffix to export the model using past key values caching. "
            f"Available tasks depend on the model, but are among the following list: {ALL_TASKS}."
        ),
    )
    optional_group.add_argument(
        "--framework",
        type=str,
        choices=["pt"],
        default="pt",
        help="The framework to use for the export. Defaults to 'pt' for PyTorch. ",
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
        "--weight-format",
        type=str,
        choices=["fp32", "fp16", "int8", "int4", "mxfp4", "nf4", "cb4"],
        default=None,
        help=(
            "The weight format of the exported model. Option 'cb4' represents a codebook with 16 fixed fp8 values in E4M3 format."
        ),
    )
    optional_group.add_argument(
        "--quant-mode",
        type=str,
        choices=["int8", "f8e4m3", "f8e5m2", "cb4_f8e4m3", "int4_f8e4m3", "int4_f8e5m2"],
        default=None,
        help=(
            "Quantization precision mode. This is used for applying full model quantization including activations. "
        ),
    )
    optional_group.add_argument(
        "--library",
        type=str,
        choices=["transformers", "diffusers", "timm", "sentence_transformers", "open_clip"],
        default=None,
        help="The library used to load the model before export. If not provided, will attempt to infer the local checkpoint's library",
    )
    optional_group.add_argument(
        "--cache_dir",
        type=str,
        default=HUGGINGFACE_HUB_CACHE,
        help="The path to a directory in which the downloaded model should be cached if the standard cache should not be used.",
    )
    optional_group.add_argument(
        "--pad-token-id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess it."
        ),
    )
    optional_group.add_argument(
        "--variant",
        type=str,
        default=None,
        help=("If specified load weights from variant filename."),
    )
    optional_group.add_argument(
        "--ratio",
        type=float,
        default=None,
        help=(
            "A parameter used when applying 4-bit quantization to control the ratio between 4-bit and 8-bit quantization. If set to 0.8, 80%% of the layers will be quantized to int4 "
            "while 20%% will be quantized to int8. This helps to achieve better accuracy at the sacrifice of the model size and inference latency. Default value is 1.0. "
            "Note: If dataset is provided, and the ratio is less than 1.0, then data-aware mixed precision assignment will be applied."
        ),
    )
    optional_group.add_argument(
        "--sym",
        action="store_true",
        default=None,
        help=(
            "Whether to apply symmetric quantization. This argument is related to integer-typed --weight-format and --quant-mode options. "
            "In case of full or mixed quantization (--quant-mode) symmetric quantization will be applied to weights in any case, so only activation quantization "
            "will be affected by --sym argument. For weight-only quantization (--weight-format) --sym argument does not affect backup precision. "
            "Examples: (1) --weight-format int8 --sym => int8 symmetric quantization of weights; "
            "(2) --weight-format int4 => int4 asymmetric quantization of weights; "
            "(3) --weight-format int4 --sym --backup-precision int8_asym => int4 symmetric quantization of weights with int8 asymmetric backup precision; "
            "(4) --quant-mode int8 --sym => weights and activations are quantized to int8 symmetric data type; "
            "(5) --quant-mode int8 => activations are quantized to int8 asymmetric data type, weights -- to int8 symmetric data type; "
            "(6) --quant-mode int4_f8e5m2 --sym => activations are quantized to f8e5m2 data type, weights -- to int4 symmetric data type."
        ),
    )
    optional_group.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=("The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization."),
    )
    optional_group.add_argument(
        "--group-size-fallback",
        type=str,
        choices=["error", "ignore", "adjust"],
        default=None,
        help=(
            "Specifies how to handle operations that do not support the given group size. Possible values are: "
            "`error`: raise an error if the given group size is not supported by a node, this is the default behavior; "
            "`ignore`: skip nodes that cannot be compressed with the given group size; "
            "`adjust`: adjust the group size to the maximum supported value for each problematic node, if there is no "
            "valid value greater than or equal to 32, then the node is quantized to the backup precision which is "
            "int8_asym by default. "
        ),
    )
    optional_group.add_argument(
        "--backup-precision",
        type=str,
        choices=["none", "int8_sym", "int8_asym"],
        default=None,
        help=(
            "Defines a backup precision for mixed-precision weight compression. Only valid for 4-bit weight formats. "
            "If not provided, backup precision is int8_asym. 'none' stands for original floating-point precision of "
            "the model weights, in this case weights are retained in their original precision without any "
            "quantization. 'int8_sym' stands for 8-bit integer symmetric quantization without zero point. 'int8_asym' "
            "stands for 8-bit integer asymmetric quantization with zero points per each quantization group."
        ),
    )
    optional_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "The dataset used for data-aware compression or quantization with NNCF. Can be a dataset name (e.g., 'wikitext2') "
            "or a string with options (e.g., 'wikitext2:seq_len=128'). The only currently supported option is `seq_len` which represents a length of an input sample sequence (sentence). "
            "For language models you can use the one from the list ['auto','wikitext2','c4','c4-new','gsm8k']. With 'auto' the "
            "dataset will be collected from model's generations. "
            "For diffusion models it should be on of ['conceptual_captions',"
            "'laion/220k-GPT4Vision-captions-from-LIVIS','laion/filtered-wit']. "
            "For visual language models the dataset must be set to 'contextual'. "
            "Note: if none of the data-aware compression algorithms are selected and ratio parameter is omitted or "
            "equals 1.0, the dataset argument will not have an effect on the resulting model."
            "Note: for text generation task, datasets with English texts such as 'wikitext2','gsm8k','c4' or 'c4-new' usually "
            "work fine even for non-English models."
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
        "--awq",
        action="store_true",
        default=None,
        help=(
            "Whether to apply AWQ algorithm. AWQ improves generation quality of INT4-compressed LLMs. If dataset is "
            "provided, a data-aware activation-based version of the algorithm will be executed, which requires "
            "additional time. Otherwise, data-free AWQ will be applied which relies on per-column magnitudes of "
            "weights instead of activations. Note: it is possible that there will be no matching patterns in the model "
            "to apply AWQ, in such case it will be skipped."
        ),
    )
    optional_group.add_argument(
        "--scale-estimation",
        action="store_true",
        default=None,
        help=(
            "Indicates whether to apply a scale estimation algorithm that minimizes the L2 error between the original "
            "and compressed layers. Providing a dataset is required to run scale estimation. Please note, that "
            "applying scale estimation takes additional memory and time."
        ),
    )
    optional_group.add_argument(
        "--gptq",
        action="store_true",
        default=None,
        help=(
            "Indicates whether to apply GPTQ algorithm that optimizes compressed weights in a layer-wise fashion to "
            "minimize the difference between activations of a compressed and original layer. Please note, that "
            "applying GPTQ takes additional memory and time."
        ),
    )
    optional_group.add_argument(
        "--lora-correction",
        action="store_true",
        default=None,
        help=(
            "Indicates whether to apply LoRA Correction algorithm. When enabled, this algorithm introduces low-rank "
            "adaptation layers in the model that can recover accuracy after weight compression at some cost of "
            "inference latency. Please note, that applying LoRA Correction algorithm takes additional memory and time."
        ),
    )
    optional_group.add_argument(
        "--sensitivity-metric",
        type=str,
        default=None,
        help=(
            "The sensitivity metric for assigning quantization precision to layers. It can be one of the following: "
            "['weight_quantization_error', 'hessian_input_activation', 'mean_activation_variance', "
            "'max_activation_variance', 'mean_activation_magnitude']."
        ),
    )
    optional_group.add_argument(
        "--quantization-statistics-path",
        type=str,
        default=None,
        help=(
            "Directory path to dump/load data-aware weight-only quantization statistics. This is useful when running "
            "data-aware quantization multiple times on the same model and dataset to avoid recomputing statistics. "
            "This option is applicable exclusively for weight-only quantization. Please note that the statistics depend "
            "on the dataset, so if you change the dataset, you should also change the statistics path to avoid confusion."
        ),
    )
    optional_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The maximum number of samples to take from the dataset for quantization.",
    )
    optional_group.add_argument(
        "--disable-stateful",
        action="store_true",
        help=(
            "Disable stateful converted models, stateless models will be generated instead. Stateful models are produced by default when this key is not used. "
            "In stateful models all kv-cache inputs and outputs are hidden in the model and are not exposed as model inputs and outputs. "
            "If --disable-stateful option is used, it may result in sub-optimal inference performance. "
            "Use it when you intentionally want to use a stateless model, for example, to be compatible with existing "
            "OpenVINO native inference code that expects KV-cache inputs and outputs in the model."
        ),
    )
    optional_group.add_argument(
        "--disable-convert-tokenizer",
        action="store_true",
        help="Do not add converted tokenizer and detokenizer OpenVINO models.",
    )
    optional_group.add_argument(
        "--smooth-quant-alpha",
        type=float,
        default=None,
        help=(
            "SmoothQuant alpha parameter that improves the distribution of activations before MatMul layers and "
            "reduces quantization error. Valid only when activations quantization is enabled."
        ),
    )
    optional_group.add_argument(
        "--model-kwargs",
        type=json.loads,
        help=("Any kwargs passed to the model forward, or used to customize the export for a given model."),
    )


def no_compression_parameter_provided(args):
    from optimum.exporters.openvino.__main__ import _no_compression_parameter_provided

    logger.warning(
        "Calling `no_compression_parameter_provided` from `openvino.py` is deprecated and will be removed in future versions."
        "Please call it from `exporters.openvino.__main__` as `_no_compression_parameter_provided` instead."
    )
    return _no_compression_parameter_provided(
        args.ratio,
        args.group_size,
        args.sym,
        args.all_layers,
        args.dataset,
        args.num_samples,
        args.awq,
        args.scale_estimation,
        args.gptq,
        args.lora_correction,
        args.sensitivity_metric,
        args.backup_precision,
    )


def no_quantization_parameter_provided(args):
    from optimum.exporters.openvino.__main__ import _no_quantization_parameter_provided

    logger.warning(
        "Calling `no_quantization_parameter_provided` from `openvino.py` is deprecated and will be removed in future versions."
        "Please call it from `exporters.openvino.__main__` as `_no_quantization_parameter_provided` instead."
    )
    return _no_quantization_parameter_provided(
        args.sym,
        args.dataset,
        args.num_samples,
        args.smooth_quant_alpha,
    )


class OVExportCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="openvino", help="Export PyTorch models to OpenVINO IR.")

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_openvino(parser)

    def run(self):
        from ...exporters.openvino.__main__ import (
            _no_compression_parameter_provided,
            _no_quantization_parameter_provided,
            main_export,
            main_quantize,
            prepare_quantization_config,
        )
        from ...exporters.openvino.utils import _merge_move
        from ...intel.openvino.configuration import OVConfig
        from ...intel.openvino.utils import TemporaryDirectory

        if self.args.weight_format is None and self.args.quant_mode is None:
            ov_config = None
            no_compression_parameter_was_provided = _no_compression_parameter_provided(
                self.args.ratio,
                self.args.group_size,
                self.args.sym,
                self.args.all_layers,
                self.args.dataset,
                self.args.num_samples,
                self.args.awq,
                self.args.scale_estimation,
                self.args.gptq,
                self.args.lora_correction,
                self.args.sensitivity_metric,
                self.args.backup_precision,
            )
            no_quantization_parameter_was_provided = _no_quantization_parameter_provided(
                self.args.sym, self.args.dataset, self.args.num_samples, self.args.smooth_quant_alpha
            )
            if not no_compression_parameter_was_provided or self.args.quantization_statistics_path is not None:
                raise ValueError(
                    "Some compression parameters are provided, but the weight format is not specified. "
                    "Please provide it with --weight-format argument."
                )
            if not no_quantization_parameter_was_provided:
                raise ValueError(
                    "Some quantization parameters are provided, but the quantization mode is not specified. "
                    "Please provide it with --quant-mode argument."
                )
        elif self.args.weight_format in {"fp16", "fp32"}:
            ov_config = OVConfig(dtype=self.args.weight_format)
        else:
            ov_config = OVConfig(dtype="auto")
        temporary_directory = None
        original_output = None
        if (
            self.args.weight_format is not None
            and self.args.weight_format not in {"fp16", "fp32"}
            or self.args.quant_mode is not None
        ):
            # In case some quantization argument is provided, we export to a temporary directory first.
            # This is to avoid confusion in the case when quantization unexpectedly fails, and an intermediate
            # floating point model ends up at the target location.
            original_output = Path(self.args.output)
            temporary_directory = TemporaryDirectory()
            output = Path(temporary_directory.name)
        else:
            output = Path(self.args.output)

        try:
            # TODO : add input shapes
            # TODO(@echarlaix): Consider replacing `ov_config` argument with `dtype` argument
            main_export(
                model_name_or_path=self.args.model,
                output=output,
                task=self.args.task,
                framework=self.args.framework,
                cache_dir=self.args.cache_dir,
                trust_remote_code=self.args.trust_remote_code,
                pad_token_id=self.args.pad_token_id,
                ov_config=ov_config,
                stateful=not self.args.disable_stateful,
                convert_tokenizer=not self.args.disable_convert_tokenizer,
                library_name=self.args.library,
                variant=self.args.variant,
                model_kwargs=self.args.model_kwargs,
                # **input_shapes,
            )

            quantization_config = prepare_quantization_config(
                output=output,
                model_name_or_path=self.args.model,
                task=self.args.task,
                library_name=self.args.library,
                cache_dir=self.args.cache_dir,
                trust_remote_code=self.args.trust_remote_code,
                weight_format=self.args.weight_format,
                quant_mode=self.args.quant_mode,
                ratio=self.args.ratio,
                sym=self.args.sym,
                group_size=self.args.group_size,
                all_layers=self.args.all_layers,
                dataset=self.args.dataset,
                num_samples=self.args.num_samples,
                awq=self.args.awq,
                sensitivity_metric=self.args.sensitivity_metric,
                scale_estimation=self.args.scale_estimation,
                gptq=self.args.gptq,
                lora_correction=self.args.lora_correction,
                quantization_statistics_path=self.args.quantization_statistics_path,
                backup_precision=self.args.backup_precision,
                group_size_fallback=self.args.group_size_fallback,
                smooth_quant_alpha=self.args.smooth_quant_alpha,
            )
            if quantization_config:
                main_quantize(
                    model_name_or_path=self.args.model,
                    task=self.args.task,
                    library_name=self.args.library,
                    quantization_config=quantization_config,
                    output=output,
                    cache_dir=self.args.cache_dir,
                    trust_remote_code=self.args.trust_remote_code,
                    model_kwargs=self.args.model_kwargs,
                )
            if original_output is not None:
                # If exported to temporary directory, move exported model to the original output directory
                original_output.mkdir(parents=True, exist_ok=True)
                _merge_move(output, original_output)
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()


def prepare_wc_config(args, default_configs):
    from optimum.exporters.openvino.__main__ import _prepare_wc_config

    logger.warning(
        "Calling `prepare_wc_config` from openvino.py is deprecated and will be removed in future versions."
        "Please call it from `exporters.openvino.__main__` as `_prepare_wc_config` instead."
    )
    return _prepare_wc_config(
        args.weight_format,
        args.ratio,
        args.sym,
        args.group_size,
        args.all_layers,
        args.dataset,
        args.num_samples,
        args.awq,
        args.sensitivity_metric,
        args.scale_estimation,
        args.gptq,
        args.lora_correction,
        args.quantization_statistics_path,
        args.backup_precision,
        args.group_size_fallback,
        default_configs,
    )


def prepare_q_config(args):
    from optimum.exporters.openvino.__main__ import _prepare_q_config

    logger.warning(
        "Calling `prepare_q_config` from openvino.py is deprecated and will be removed in future versions."
        "Please call it from `exporters.openvino.__main__` as `_prepare_q_config` instead."
    )
    return _prepare_q_config(
        args.quant_mode,
        args.sym,
        args.dataset,
        args.num_samples,
        args.smooth_quant_alpha,
    )
