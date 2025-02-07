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
from ...intel.utils.modeling_utils import _infer_library_from_model_name_or_path
from ...utils.save_utils import maybe_load_preprocessors
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
        "--weight-format",
        type=str,
        choices=["fp32", "fp16", "int8", "int4", "mxfp4", "nf4"],
        default=None,
        help="The weight format of the exported model.",
    )
    optional_group.add_argument(
        "--quant-mode",
        type=str,
        choices=["int8", "f8e4m3", "f8e5m2"],
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
        help=("Whether to apply symmetric quantization"),
    )
    optional_group.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=("The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization."),
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
            "The dataset used for data-aware compression or quantization with NNCF. "
            "For language models you can use the one from the list ['auto','wikitext2','c4','c4-new']. With 'auto' the "
            "dataset will be collected from model's generations. "
            "For diffusion models it should be on of ['conceptual_captions',"
            "'laion/220k-GPT4Vision-captions-from-LIVIS','laion/filtered-wit']. "
            "For visual language models the dataset must be set to 'contextual'. "
            "Note: if none of the data-aware compression algorithms are selected and ratio parameter is omitted or "
            "equals 1.0, the dataset argument will not have an effect on the resulting model."
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
            "Whether to apply AWQ algorithm. AWQ improves generation quality of INT4-compressed LLMs, but requires "
            "additional time for tuning weights on a calibration dataset. To run AWQ, please also provide a dataset "
            "argument. Note: it is possible that there will be no matching patterns in the model to apply AWQ, in such "
            "case it will be skipped."
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


def no_compression_parameter_provided(args):
    return all(
        (
            it is None
            for it in (
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
        )
    )


def no_quantization_parameter_provided(args):
    return all(
        (
            it is None
            for it in (
                args.sym,
                args.dataset,
                args.num_samples,
                args.smooth_quant_alpha,
            )
        )
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
        from ...exporters.openvino.__main__ import infer_task, main_export, maybe_convert_tokenizers
        from ...exporters.openvino.utils import save_preprocessors
        from ...intel.openvino.configuration import _DEFAULT_4BIT_CONFIG, OVConfig, get_default_int4_config

        if self.args.library is None:
            # TODO: add revision, subfolder and token to args
            library_name = _infer_library_from_model_name_or_path(
                model_name_or_path=self.args.model, cache_dir=self.args.cache_dir
            )
            if library_name == "sentence_transformers":
                logger.warning(
                    "Library name is not specified. There are multiple possible variants: `sentence_transformers`, `transformers`."
                    "`transformers` will be selected. If you want to load your model with the `sentence-transformers` library instead, please set --library sentence_transformers"
                )
                library_name = "transformers"
        else:
            library_name = self.args.library

        if self.args.weight_format is None and self.args.quant_mode is None:
            ov_config = None
            if not no_compression_parameter_provided(self.args):
                raise ValueError(
                    "Some compression parameters are provided, but the weight format is not specified. "
                    "Please provide it with --weight-format argument."
                )
            if not no_quantization_parameter_provided(self.args):
                raise ValueError(
                    "Some quantization parameters are provided, but the quantization mode is not specified. "
                    "Please provide it with --quant-mode argument."
                )
        elif self.args.weight_format in {"fp16", "fp32"}:
            ov_config = OVConfig(dtype=self.args.weight_format)
        elif self.args.weight_format is not None:
            # For int4 quantization if no parameter is provided, then use the default config if exists
            if no_compression_parameter_provided(self.args) and self.args.weight_format == "int4":
                quantization_config = get_default_int4_config(self.args.model)
            else:
                is_int8 = self.args.weight_format == "int8"
                quantization_config = {
                    "bits": 8 if is_int8 else 4,
                    "ratio": 1.0 if is_int8 else (self.args.ratio or _DEFAULT_4BIT_CONFIG["ratio"]),
                    "sym": self.args.sym or False,
                    "group_size": -1 if is_int8 else self.args.group_size,
                    "all_layers": None if is_int8 else self.args.all_layers,
                    "dataset": self.args.dataset,
                    "num_samples": self.args.num_samples,
                    "quant_method": "awq" if self.args.awq else "default",
                    "sensitivity_metric": self.args.sensitivity_metric,
                    "scale_estimation": self.args.scale_estimation,
                    "gptq": self.args.gptq,
                    "lora_correction": self.args.lora_correction,
                    "weight_format": self.args.weight_format,
                    "backup_precision": self.args.backup_precision,
                }

            if quantization_config.get("dataset", None) is not None:
                quantization_config["trust_remote_code"] = self.args.trust_remote_code
            ov_config = OVConfig(quantization_config=quantization_config)
        else:
            if self.args.dataset is None:
                raise ValueError(
                    "Dataset is required for full quantization. Please provide it with --dataset argument."
                )

            quantization_config = {
                "weight_format": self.args.quant_mode,
                "activation_format": self.args.quant_mode,
                "bits": 8,
                "sym": self.args.sym or False,
                "dataset": self.args.dataset,
                "num_samples": self.args.num_samples,
                "smooth_quant_alpha": self.args.smooth_quant_alpha,
                "trust_remote_code": self.args.trust_remote_code,
            }
            ov_config = OVConfig(quantization_config=quantization_config)

        quantization_config = ov_config.quantization_config if ov_config else None
        quantize_with_dataset = quantization_config and getattr(quantization_config, "dataset", None) is not None
        task = infer_task(self.args.task, self.args.model, library_name=library_name)

        if library_name == "diffusers" and quantize_with_dataset:
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
            elif class_name == "StableDiffusion3Pipeline":
                from optimum.intel import OVStableDiffusion3Pipeline

                model_cls = OVStableDiffusion3Pipeline
            elif class_name == "FluxPipeline":
                from optimum.intel import OVFluxPipeline

                model_cls = OVFluxPipeline
            elif class_name == "SanaPipeline":
                from optimum.intel import OVSanaPipeline

                model_cls = OVSanaPipeline
            else:
                raise NotImplementedError(f"Quantization in hybrid mode isn't supported for class {class_name}.")

            model = model_cls.from_pretrained(self.args.model, export=True, quantization_config=quantization_config)
            model.save_pretrained(self.args.output)
            if not self.args.disable_convert_tokenizer:
                maybe_convert_tokenizers(library_name, self.args.output, model, task=task)
        elif (
            quantize_with_dataset
            and (task.startswith("text-generation") or task == "automatic-speech-recognition")
            or (task == "image-text-to-text" and quantization_config is not None)
        ):
            if task.startswith("text-generation"):
                from optimum.intel import OVModelForCausalLM

                model_cls = OVModelForCausalLM
            elif task == "image-text-to-text":
                from optimum.intel import OVModelForVisualCausalLM

                model_cls = OVModelForVisualCausalLM
            else:
                from optimum.intel import OVModelForSpeechSeq2Seq

                model_cls = OVModelForSpeechSeq2Seq

            # In this case, to apply quantization an instance of a model class is required
            model = model_cls.from_pretrained(
                self.args.model,
                export=True,
                quantization_config=quantization_config,
                stateful=not self.args.disable_stateful,
                trust_remote_code=self.args.trust_remote_code,
                variant=self.args.variant,
                cache_dir=self.args.cache_dir,
            )
            model.save_pretrained(self.args.output)

            preprocessors = maybe_load_preprocessors(self.args.model, trust_remote_code=self.args.trust_remote_code)
            save_preprocessors(preprocessors, model.config, self.args.output, self.args.trust_remote_code)
            if not self.args.disable_convert_tokenizer:
                maybe_convert_tokenizers(library_name, self.args.output, preprocessors=preprocessors, task=task)
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
                variant=self.args.variant,
                # **input_shapes,
            )
