#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import logging
import math
import os
import warnings
from collections import UserDict

import torch
from accelerate import init_empty_weights
from datasets import load_dataset
from neural_compressor.torch.algorithms.weight_only.modules import WeightOnlyLinear
from neural_compressor.torch.quantization import (
    AutoRoundConfig,
    GPTQConfig,
    RTNConfig,
    convert,
    prepare,
)
from neural_compressor.utils.pytorch import load
from torch.utils.data import DataLoader

from ..utils.constant import WEIGHTS_NAME
from ..utils.import_utils import (
    _autoround_available,
    _ipex_available,
)


if _autoround_available:
    from auto_round.calib_dataset import get_dataloader as get_autoround_dataloader
    from auto_round.export.export_to_itrex.model_wrapper import (
        WeightOnlyLinear as auto_round_woqlinear,
    )

if _ipex_available:
    import intel_extension_for_pytorch as ipex
logger = logging.getLogger(__name__)


CONFIG_NAME = "best_configure.yaml"
QUANTIZATION_CONFIG_NAME = "quantize_config.json"

NEURAL_COMPRESSOR_MINIMUM_VERSION = "2.1.0"
NEURAL_COMPRESSOR_WEIGHT_ONLY_MINIMUM_VERSION = "2.3.0"
IPEX_MINIMUM_VERSION = "2.1.0"
ITREX_MINIMUM_VERSION = "1.4.0"
ITREX_MINIMUM_TORCH_VERSION = "2.2.0"


_HEAD_TO_AUTOMODELS = {
    "fill-mask": "INCModelForMaskedLM",
    "text-generation": "INCModelForCausalLM",
    "text2text-generation": "INCModelForSeq2SeqLM",
    "text-classification": "INCModelForSequenceClassification",
    "token-classification": "INCModelForTokenClassification",
    "question-answering": "INCModelForQuestionAnswering",
    "multiple-choice": "INCModelForMultipleChoice",
    "stable-diffusion": "INCStableDiffusionPipeline",
    "feature-extraction": "INCModel",
}


class INCDataLoader(DataLoader):
    use_label = True

    @classmethod
    def from_pytorch_dataloader(cls, dataloader: DataLoader, use_label: bool = True):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected a PyTorch DataLoader, got: {type(dataloader)}.")
        inc_dataloader = cls(dataloader.dataset)
        cls.use_label = use_label
        for key, value in dataloader.__dict__.items():
            inc_dataloader.__dict__[key] = value
        return inc_dataloader

    def __iter__(self):
        for input in super().__iter__():
            if not isinstance(input, (dict, tuple, list, UserDict)):
                raise TypeError(f"Model calibration cannot use input of type {type(input)}.")
            label = input.get("labels") if isinstance(input, dict) else None
            if self.use_label:
                yield input, label
            else:
                yield input


def load_quantized_model(checkpoint_dir_or_file: str, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """
    Returns the quantized model, which was quantized through neural_compressor.

    Arguments:
        checkpoint_dir_or_file (`str`):
            The path to the model checkpoint containing the quantization information.
        model (`torch.nn.Module`):
            The original FP32 model.
    """
    warnings.warn("This function has been depreciated and will be removed in optimum-intel v1.9.")
    if os.path.isdir(checkpoint_dir_or_file):
        checkpoint_dir_or_file = os.path.join(
            os.path.abspath(os.path.expanduser(checkpoint_dir_or_file)), WEIGHTS_NAME
        )

    return load(checkpoint_dir_or_file, model, **kwargs)


DTYPE_BITS_MAPPING = {
    "int4": 4,
    "int4_fullrange": 4,
    "int4_clip": 4,
    "int8": 8,
}


def convert_dtype_str2torch(str_dtype):
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16":
        return torch.float16
    elif str_dtype == "bf16":
        return torch.bfloat16
    else:
        assert False, "Unsupported str dtype {} to torch dtype".format(str_dtype)


def replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    device="cpu",
    empty_weights=False,
):
    if modules_to_not_convert is None:
        # output_layer is chatglm last layer name
        # embed_out is dolly_v2 last layer name
        modules_to_not_convert = ["lm_head", "output_layer", "embed_out"]
    model, is_replaced = _replace_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        device=device,
        empty_weights=empty_weights,
    )

    if not is_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


def _replace_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    is_replaced=False,
    device="cpu",
    empty_weights=False,
):
    """Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfully or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        is_removed = False
        use_optimum_format = getattr(module, "use_optimum_format", False) or quantization_config.weight_dtype not in [
            "int4_fullrange"
        ]

        if (
            isinstance(module, torch.nn.Linear)
            or isinstance(module, WeightOnlyLinear)
            or (_autoround_available and isinstance(module, auto_round_woqlinear))
            or (_ipex_available and isinstance(module, ipex.nn.utils._weight_prepack._IPEXLinear))
        ) and (name not in modules_to_not_convert):
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features
                    if device == "cpu" or device == torch.device("cpu") or device == "auto":
                        from intel_extension_for_transformers.transformers.llm.quantization.nn.modules import (
                            QuantizedLinearQBits,
                        )

                        use_optimum_format = getattr(module, "use_optimum_format", False)
                        model._modules[name] = QuantizedLinearQBits(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compute_dtype=quantization_config.compute_dtype,
                            compress_statistics=False,
                            weight_dtype=quantization_config.weight_dtype,
                            bits=quantization_config.bits,
                            scale_dtype=quantization_config.scale_dtype,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme,
                            compression_dtype=getattr(module, "compression_dtype", torch.int32),
                            compression_dim=getattr(module, "compression_dim", 1),
                            device=device,
                            use_optimum_format=use_optimum_format,
                        )
                    elif device == "xpu" or device == torch.device("xpu"):
                        from intel_extension_for_pytorch.nn.utils._quantize_convert import (
                            WeightOnlyQuantizedLinear as ipex_linear,
                        )

                        model._modules[name] = ipex_linear(
                            in_features,
                            out_features,
                            module.bias is not None,
                            compute_dtype=quantization_config.compute_dtype,
                            compress_statistics=False,
                            weight_dtype=quantization_config.weight_dtype,
                            scale_dtype=quantization_config.scale_dtype,
                            blocksize=quantization_config.group_size,
                            scheme=quantization_config.scheme,
                            compression_dtype=torch.int32,
                            compression_dim=getattr(module, "compression_dim", 1),
                            device=device,
                            use_optimum_format=getattr(module, "use_optimum_format", True),
                        )
                        if quantization_config.quant_method.value == "gptq":
                            g_idx = getattr(
                                module,
                                "g_idx",
                                torch.zeros(in_features, dtype=torch.int32).to(device),
                            )
                        else:
                            g_idx = None
                        model._modules[name].set_scales_zps_gidx(
                            (
                                module.scales
                                if hasattr(module, "scales")
                                else torch.ones(
                                    (
                                        math.ceil(in_features / quantization_config.group_size),
                                        out_features,
                                    ),
                                    dtype=convert_dtype_str2torch(quantization_config.compute_dtype),
                                    device=torch.device(device),
                                )
                            ),
                            module.qzeros if hasattr(module, "qzeros") else None,
                            g_idx,
                        )
                    else:
                        raise Exception("{} device Unsupported weight only quantization!".format(device))

                    is_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
                if quantization_config.use_ipex:
                    pass
                elif device == "cpu" or device == torch.device("cpu") or device == "auto":
                    int_weight = module.unpack_tensor_with_numpy(module.qweight)
                    scales = module.scales
                    zeros = module.qzeros if hasattr(module, "qzeros") else None

                    model._modules[name].set_weights_bias(
                        int_weight,
                        scales,
                        zeros,
                        module.g_idx if hasattr(module, "g_idx") else None,
                        quantization_config,
                        bias=None if module.bias is None else module.bias.data,
                    )
                else:
                    if not hasattr(module, "qweight"):
                        n_pack = 32 // DTYPE_BITS_MAPPING[quantization_config.weight_dtype]
                        weight = torch.zeros(
                            (math.ceil(in_features / n_pack), out_features),
                            dtype=torch.int32,
                            device=torch.device(device),
                        )
                    model._modules[name].set_weights_bias(
                        module.qweight.data if hasattr(module, "qweight") else weight,
                        None if module.bias is None else module.bias.data,
                    )
                    del module
                    gc.collect()
                    is_removed = True

        if not is_removed and len(list(module.children())) > 0:  # pylint: disable=E1101
            _, is_replaced = _replace_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                is_replaced=is_replaced,
                device=device,
                empty_weights=empty_weights,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, is_replaced


def default_run_fn(model, tokenizer, dataset, max_length=512, n_samples=100, batch_size=8, algo="rtn"):
    from torch.utils.data import DataLoader

    if isinstance(dataset, (str, bytes, os.PathLike)):
        calib_dataset = load_dataset(dataset, split="train")
    calib_dataset = calib_dataset.shuffle(seed=42)
    if tokenizer is None:
        logger.error("Please provide the tokenizer in quantization_config.")
        exit(0)

    def tokenize_function(examples):
        if "text" in examples:
            example = tokenizer(examples["text"])
        else:
            logger.error(
                "Please check dataset prompt identifier," + " NeelNanda/pile-10k is default used calibration dataset."
            )
            exit(0)
        return example

    tokenized_dataset = calib_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    tokenized_dataset = tokenized_dataset.filter(lambda x: x["input_ids"].shape[-1] >= max_length)

    def collate_batch(batch):
        input_ids_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            if len(input_ids) >= max_length:
                input_ids = input_ids[:max_length]
                input_ids_padded.append(input_ids)
            else:
                continue
        assert (
            input_ids_padded != []
        ), "The dataset does not have data that meets the required input length. Please reduce seq_len."
        return torch.vstack(input_ids_padded)

    calib_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    total_cnt = 0
    for i, (input_ids) in enumerate(calib_dataloader):
        if total_cnt + input_ids.shape[0] > n_samples:
            input_ids = input_ids[: n_samples - total_cnt, ...]
        total_cnt += input_ids.shape[0]
        if total_cnt >= n_samples:
            break

        try:
            model(
                input_ids=input_ids,
            )
        except ValueError:
            pass


@torch.no_grad()
def run_fn_for_autoround(model, dataloader):
    for data in dataloader:
        if isinstance(data, tuple) or isinstance(data, list):
            model(*data)
        elif isinstance(data, dict):
            model(**data)
        else:
            model(data)


def convert_to_weight_only_quantized_model(model, config, device="cpu"):
    if device == "xpu" or device == torch.device("xpu"):
        assert hasattr(torch, "xpu") and torch.xpu.is_available(), "There is no xpu device in this system!"

    orig_dtype = torch.float32
    for param in model.parameters():
        orig_dtype = param.dtype
        if orig_dtype != torch.float32:
            model.to(dtype=torch.float32)
        break

    if config.bits == 8:
        dtype = "int8"
    elif config.bits == 4:
        dtype = "int4"
    else:
        raise ValueError("Please check quantization config bits number.")
    # mapping to INC config
    if config.quant_method.value == "rtn":
        quant_config = RTNConfig(
            dtype=dtype,
            bits=config.bits,
            use_sym=config.sym,
            group_size=config.group_size,
            use_layer_wise=config.use_layer_wise,
        )
        model = prepare(model, quant_config)
        model = convert(model)
    elif config.quant_method.value == "gptq":
        model.seqlen = config.seq_len
        quant_config = GPTQConfig(
            dtype=dtype,
            bits=config.bits,
            use_sym=config.sym,
            group_size=config.group_size,
            use_layer_wise=config.use_layer_wise,
            act_order=config.desc_act,
            percdamp=config.damp_percent,
            block_size=config.block_size,
            static_groups=config.static_groups,
        )
        logger.info(f"Do GPTQ algorithm with config {quant_config}")
        run_fn = default_run_fn
        run_args = (
            config.tokenizer,
            config.dataset,
            config.seq_len,  # max_length
            config.num_samples,  # n_samples
            config.batch_size,  # batch_size
            config.quant_method.value,  # algo
        )
        model = prepare(model=model, quant_config=quant_config)
        run_fn(model, *run_args)
        model = convert(model)
    elif config.quant_method.value == "autoround":
        quant_config = AutoRoundConfig(
            dtype=dtype,
            bits=config.bits,
            use_sym=config.sym,
            group_size=config.group_size,
            enable_quanted_input=not config.disable_quanted_input,
            lr=config.lr,
            minmax_lr=config.minmax_lr,
            seqlen=config.seq_len,
            n_samples=config.n_samples,
            iters=config.iters,
            scale_dtype=config.scale_dtype,
        )
        logger.info(f"Do AutoRound algorithm with config {quant_config}")
        dataloader = get_autoround_dataloader(
            tokenizer=config.tokenizer,
            seqlen=config.seq_len,
            dataset_name=config.dataset,
            seed=42,
            bs=config.batch_size,
            n_samples=config.num_samples,
        )
        run_fn = run_fn_for_autoround
        run_args = (dataloader,)
        model = prepare(model=model, quant_config=quant_config)
        run_fn(model, *run_args)
        model = convert(model)
    else:
        assert False, "The Supported algorithm are RTN, GPTQ, AUTOROUND"

    if device == "xpu" or device == torch.device("xpu"):
        logger.warning("The recommended ipex version is higher than 2.3.10 for xpu device.")

    model.eval()
    # INC attribute conflicted with transformers when use nf4/int8 training.
    del model.is_quantized
    q_model = replace_linear(model, None, None, config, device=device)

    if orig_dtype != torch.float32:
        q_model.to(dtype=orig_dtype)

    return q_model.to(device)
