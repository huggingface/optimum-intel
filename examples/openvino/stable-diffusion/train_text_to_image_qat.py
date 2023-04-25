#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import itertools
import logging
import math
import os
import random
import tempfile
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline, LMSDiscreteScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from nncf import NNCFConfig
from nncf.common.logging import nncf_logger
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.layer_utils import CompressionParameter
from openvino._offline_transformations import apply_moc_transformations, compress_quantize_weights_transformation
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from optimum.exporters.onnx import export_models, get_stable_diffusion_models_for_export
from optimum.intel import OVStableDiffusionPipeline
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)


random.seed(42)
logger = get_logger(__name__)
nncf_logger.setLevel(logging.INFO)


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def pokemon_preprocess_train(examples, train_transforms, tokenize_captions, image_column="image"):
    image = examples[image_column]
    examples["pixel_values"] = train_transforms(image.convert("RGB"))
    examples["input_ids"] = tokenize_captions(examples)
    return examples


def get_pil_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")


# Many of the images in laion2B dataset are unavailable
# This is a workaround to substitute such images with a backup or cached available examples
BACKUP_PAIR = (
    get_pil_from_url(
        "https://thumbs.dreamstime.com/t/altai-mountains-mountain-lake-russia-siberia-chuya-ridge-49130812.jpg"
    ),
    "Altai mountains Stock Photography",
)
AVAILABLE_EXAMPLES = []


def laion2B_preprocess_train(examples, train_transforms, tokenize_captions, image_column="URL"):
    url = examples[image_column]
    try:
        image = get_pil_from_url(url)
        AVAILABLE_EXAMPLES.append((url, examples["TEXT"]))
    except Exception:
        logger.info(f"Can't load image from url: {url}, using cache with size: {len(AVAILABLE_EXAMPLES)}")
        if len(AVAILABLE_EXAMPLES) > 0:
            backup_id = random.randint(0, len(AVAILABLE_EXAMPLES) - 1)
            backup_example = AVAILABLE_EXAMPLES[backup_id]
            try:
                image = get_pil_from_url(backup_example[0])
                examples["TEXT"] = backup_example[1]
            except Exception:
                logger.info(f"Can't load image from cached url: {backup_example[0]}, using backup")
                image = BACKUP_PAIR[0].copy()
                examples["TEXT"] = BACKUP_PAIR[1]
        else:
            logger.info(f"Can't load image from url: {url}, using backup")
            image = BACKUP_PAIR[0].copy()
            examples["TEXT"] = BACKUP_PAIR[1]

    examples["pixel_values"] = train_transforms(image)
    examples["input_ids"] = tokenize_captions(examples)
    return examples


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": {
        "columns": ("image", "text"),
        "preprocess_fn": pokemon_preprocess_train,
        "streaming": False,
    },
    "laion/laion2B-en": {
        "columns": ("URL", "TEXT"),
        "preprocess_fn": laion2B_preprocess_train,
        "streaming": True,
    },
    "laion/laion2B-en-aesthetic": {
        "columns": ("URL", "TEXT"),
        "preprocess_fn": laion2B_preprocess_train,
        "streaming": True,
    },
}


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAQUnet:
    """
    Exponential Moving Average of unets weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = param.clone()
                tmp = tmp.to(s_param.device)
                # tmp = self.decay * (s_param - param.clone.to(s_param.device))
                tmp.sub_(s_param)
                tmp.mul_(self.decay)
                tmp.neg_()
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion 8-bit Quantization for OpenVINO")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lambdalabs/pokemon-blip-captions",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-quantized",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--noise_scheduler",
        type=str,
        default=None,
        choices=["DDIM", "DDPM", "LMSDiscrete"],
        help="The noise scheduler for the Diffusion pipiline used for training.",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=0.00085,
        help="Beta min value for noise scheduler.",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=0.012,
        help="BetaMax value for noise scheduler.",
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="scaled_linear",
        help="Beta schedule type",
    )
    parser.add_argument(
        "--noise_schedule_steps",
        type=int,
        default=1000,
        help=("The noise scheduler max train timestemps"),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=15000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--ema_device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Whether to use EMA model and where to store the EMA model.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=15000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--opt_init_steps",
        type=int,
        default=300,
        help=("Max number of initialization steps for quantization before the actual fine-tuning."),
    )
    parser.add_argument(
        "--opt_init_type",
        type=str,
        default="mean_min_max",
        choices=["min_max", "mean_min_max", "threesigma"],
        help="They way how to estimate activation quantization paramters at the initializatin step before QAT.",
    )
    parser.add_argument(
        "--tune_quantizers_only",
        action="store_true",
        default=False,
        help="Whether to train quantization parameters only.",
    )
    parser.add_argument("--use_kd", action="store_true", help="Use Knowledge Distillation to boost accuracy.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    return args


def get_noise_scheduler(args):
    scheduler_args = {
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "beta_schedule": args.beta_schedule,
        "num_train_timesteps": args.noise_schedule_steps,
    }
    if args.noise_scheduler == "DDIM":
        noise_scheduler = DDIMScheduler(**scheduler_args)
    elif args.noise_scheduler == "DDPM":
        noise_scheduler = DDPMScheduler(**scheduler_args)
    elif args.noise_scheduler == "LMSDiscrete":
        noise_scheduler = LMSDiscreteScheduler(**scheduler_args)
    else:
        raise ValueError(f"Unknown noise schedule {args.noise_schedule}")
    return noise_scheduler


def export_to_onnx(pipeline, save_dir):
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder

    unet.eval().cpu()
    vae.eval().cpu()
    text_encoder.eval().cpu()

    ONNX_WEIGHTS_NAME = "model.onnx"

    output_names = [
        os.path.join(DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
        os.path.join(DIFFUSION_MODEL_UNET_SUBFOLDER, ONNX_WEIGHTS_NAME),
        os.path.join(DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
        os.path.join(DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER, ONNX_WEIGHTS_NAME),
    ]

    with torch.no_grad():
        models_and_onnx_configs = get_stable_diffusion_models_for_export(pipeline)
        pipeline.save_config(save_dir)
        export_models(
            models_and_onnx_configs=models_and_onnx_configs, output_dir=Path(save_dir), output_names=output_names
        )


def export_to_openvino(pipeline, onnx_dir, save_dir):
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(
        model_id=onnx_dir,
        from_onnx=True,
        model_save_dir=save_dir,
        tokenizer=pipeline.tokenizer,
        scheduler=pipeline.scheduler,
        feature_extractor=pipeline.feature_extractor,
        compile=False,
    )
    apply_moc_transformations(ov_pipe.unet.model, cf=False)
    compress_quantize_weights_transformation(ov_pipe.unet.model)
    ov_pipe.save_pretrained(save_dir)


class UnetInitDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.init_data = data

    def __len__(self):
        return len(self.init_data)

    def __getitem__(self, index):
        return self.init_data[index]


def prepare_nncf_init_data(pipeline, dataloader, args):
    weight_dtype = torch.float32
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    noise_scheduler = pipeline.scheduler

    nncf_init_data = []

    logger.info(f"Fetching {args.opt_init_steps} for the initialization...")
    for _, batch in tqdm(zip(range(args.opt_init_steps), itertools.islice(dataloader, 0, args.opt_init_steps))):
        with torch.no_grad():
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            nncf_init_data.append(
                (
                    torch.squeeze(noisy_latents).to("cpu"),
                    torch.squeeze(timesteps).to("cpu"),
                    torch.squeeze(encoder_hidden_states).to("cpu"),
                    0,
                )
            )
    return nncf_init_data


# The config should work for Stable Diffusion v1.4-2.1
def get_nncf_config(pipeline, dataloader, args):
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    nncf_config_dict = {
        "input_info": [
            {  # "keyword": "latent_model_input",
                "sample_size": [1, unet.config["in_channels"], unet.config["sample_size"], unet.config["sample_size"]]
            },
            {"sample_size": [1]},  # "keyword": "t",
            {  # "keyword": "encoder_hidden_states",
                "sample_size": [1, text_encoder.config.max_position_embeddings, text_encoder.config.hidden_size]
            },
        ],
        "log_dir": args.output_dir,  # The log directory for NNCF-specific logging outputs.
        "compression": [
            {
                "algorithm": "quantization",  # Specify the algorithm here.
                "preset": "mixed",
                "initializer": {
                    "range": {"num_init_samples": args.opt_init_steps, "type": args.opt_init_type},
                    "batchnorm_adaptation": {"num_bn_adaptation_samples": args.opt_init_steps},
                },
                "scope_overrides": {"activations": {"{re}.*baddbmm_0": {"mode": "symmetric"}}},
                "ignored_scopes": [
                    "{re}.*__add___[0-2]",
                    "{re}.*layer_norm_0",
                    "{re}.*Attention.*/bmm_0",
                    "{re}.*__truediv__*",
                    "{re}.*group_norm_0",
                    "{re}.*mul___[0-2]",
                    "{re}.*silu_[0-2]",
                ],
                "export_to_onnx_standard_ops": True,
            },
        ],
    }
    if args.use_kd:
        nncf_config_dict["compression"].append({"algorithm": "knowledge_distillation", "type": "mse"})  # or ""softmax

    class UnetInitDataLoader(PTInitializingDataLoader):
        def get_inputs(self, dataloader_output):
            noisy_latents = dataloader_output[0].float().to(unet.device, non_blocking=True)
            timesteps = dataloader_output[1].float().to(unet.device, non_blocking=True)
            encoder_hidden_states = dataloader_output[2].float().to(unet.device, non_blocking=True)
            return (noisy_latents, timesteps, encoder_hidden_states), {}

        def get_target(self, dataloader_output):
            return dataloader_output[0]

    nncf_config = NNCFConfig.from_dict(nncf_config_dict)
    nncf_config = register_default_init_args(nncf_config, UnetInitDataLoader(dataloader))
    return nncf_config


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    pipeline = DiffusionPipeline.from_pretrained(args.model_id)

    # Load models and create wrapper for stable diffusion
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    noise_scheduler = pipeline.scheduler
    if args.noise_scheduler:
        noise_scheduler = get_noise_scheduler(args)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset_settings = dataset_name_mapping.get(args.dataset_name, None)
    if dataset_settings is None:
        raise ValueError(
            f"Dataset {args.dataset_name} not supported. Please choose from {dataset_name_mapping.keys()}"
        )

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            streaming=dataset_settings["streaming"],
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    # 6. Get the column names for input/target.
    dataset_columns = dataset_settings["columns"]
    caption_column = dataset_columns[1]

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        caption = examples[caption_column]
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Caption column `{caption_column}` should contain either strings or lists of strings.")
        inputs = tokenizer(captions[0], max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    preprocess_fn = partial(
        dataset_settings["preprocess_fn"], train_transforms=train_transforms, tokenize_captions=tokenize_captions
    )

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=42, buffer_size=args.max_train_samples)
        # Set the training transforms
        train_dataset = dataset["train"]

    def collate_fn(examples):
        examples = [preprocess_fn(example) for example in examples]
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers
    )

    unet = accelerator.prepare(unet)
    vae.to(unet.device)
    text_encoder.to(unet.device)
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)
    orig_unet = unet  # save link to original unet model for EMA

    ## Create initialization dataset for PTQ
    nncf_init_data = prepare_nncf_init_data(pipeline, train_dataloader, args)
    init_dataloader = torch.utils.data.DataLoader(UnetInitDataset(nncf_init_data), batch_size=1, num_workers=1)
    nncf_config = get_nncf_config(pipeline, init_dataloader, args)

    # Quantize the model and initialize quantizer using init data
    compression_controller, unet = create_compressed_model(unet, nncf_config)

    statistics_unet = compression_controller.statistics()
    logger.info(statistics_unet.to_str())

    del nncf_init_data, init_dataloader
    torch.cuda.empty_cache()

    unet.train()

    if args.tune_quantizers_only:
        for p in unet.parameters():
            if not isinstance(p, CompressionParameter):
                p.requires_grad = False

    # Reinit
    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    dataset_len = args.max_train_samples if args.max_train_samples is not None else len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(dataset_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.ema_device:
        ema_unet = EMAQUnet(orig_unet.parameters())
        if args.ema_device == "cpu":
            ema_unet.to("cpu")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(dataset_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        train_loss = 0.0
        compression_controller.scheduler.epoch_step()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual and compute loss
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                compression_loss_unet = compression_controller.loss()
                loss = loss + compression_loss_unet

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.ema_device:
                    ema_unet.step(orig_unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.ema_device:
            ema_unet.copy_to(orig_unet.parameters())

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()

    # Export optimized pipline to OpenVINO
    export_unet = compression_controller.strip(do_copy=False)
    export_pipeline = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=export_unet,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        safety_checker=pipeline.safety_checker,
        feature_extractor=pipeline.feature_extractor,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        export_to_onnx(export_pipeline, tmpdirname)
        export_to_openvino(export_pipeline, tmpdirname, Path(args.output_dir) / "openvino")


if __name__ == "__main__":
    main()
