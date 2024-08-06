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

import inspect
import io
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import hp_params
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

# isort: on

import openvino
import torch
import torch.distributed as dist
import torch.nn.functional as F
from nncf import NNCFConfig
from nncf.common.logging.logger import nncf_logger, set_log_level
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.config.structures import BNAdaptationInitArgs, QuantizationRangeInitArgs
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsityController
from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerStage
from nncf.torch import create_compressed_model
from nncf.torch.composite_compression import PTCompositeCompressionAlgorithmController
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.algo import QuantizationController
from openvino._offline_transformations import (
    apply_fused_names_cleanup,
    apply_moc_transformations,
    apply_pruning_transformation,
    compress_quantize_weights_transformation,
)
from openvino.runtime import Core, PartialShape, save_model
from packaging import version
from torch import nn
from torch.onnx import export as onnx_export
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, RandomSampler
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import TrainerCallback, TrainerState
from transformers.trainer_pt_utils import get_dataloader_sampler, get_model_param_count
from transformers.trainer_utils import (
    EvalPrediction,
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    WEIGHTS_NAME,
    is_accelerate_available,
    is_apex_available,
    is_sagemaker_mp_enabled,
    logging,
)

from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfig

from ..utils.constant import _TASK_ALIASES
from ..utils.import_utils import is_transformers_version
from .configuration import OVConfig
from .quantization import OVDataLoader
from .training_args import OVTrainingArguments
from .utils import (
    MAX_ONNX_OPSET,
    MIN_ONNX_QDQ_OPSET,
    ONNX_WEIGHTS_NAME,
    OV_XML_FILE_NAME,
    use_external_data_format,
)


if is_transformers_version(">=", "4.39.0"):
    from transformers.utils import is_torch_xla_available
else:
    from transformers.utils import is_torch_tpu_available as is_torch_xla_available

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version
    from accelerate import skip_first_batches

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        pass
    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        pass


if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

core = Core()

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# NNCF Error to be shown on stdout
# set_log_level(logging.ERROR)
NNCF_LOG_FILE_NAME = "nncf_output.log"


DEFAULT_QUANTIZATION_CONFIG = {
    "algorithm": "quantization",
    "preset": "mixed",
    "overflow_fix": "disable",
    "initializer": {
        "range": {"num_init_samples": 300, "type": "mean_min_max"},
        "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
    },
    "scope_overrides": {"activations": {"{re}.*matmul_0": {"mode": "symmetric"}}},
    "ignored_scopes": [
        "{re}.*Embedding.*",
        "{re}.*add___.*",
        "{re}.*layer_norm_.*",
    ],
}


def _onnx_export_nncf_model(model: NNCFNetwork, config: OnnxConfig, output: Union[str, io.BytesIO], opset: int = None):
    # TODO: remove it when fix controller.strip(copy=True) behavior
    signature = inspect.signature(model.forward)
    signature = list(signature.parameters.keys())
    opset = opset or config.DEFAULT_ONNX_OPSET
    model_inputs = config.generate_dummy_inputs(framework="pt")
    # Create ordered inputs for the ONNX export of NNCFNetwork as keyword arguments are currently not supported
    model_inputs = tuple(model_inputs.pop(key, None) for key in signature if len(model_inputs) != 0)
    device = model.device

    def remap(value):
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        return value

    with config.patch_model_for_export(model):
        model_inputs = tree_map(remap, model_inputs)
        with torch.no_grad():
            model.eval()
            # Disable node additions to be exported in the graph
            model.nncf.disable_dynamic_graph_building()
            onnx_export(
                model,
                model_inputs,
                f=output,
                input_names=list(config.inputs.keys()),
                output_names=list(config.outputs.keys()),
                dynamic_axes=dict(chain(config.inputs.items(), config.outputs.items())),
                do_constant_folding=True,
                opset_version=opset,
            )
            model.nncf.enable_dynamic_graph_building()


class OVTrainer(Trainer):
    """
    OVTrainer enables NNCF quantization aware training.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        teacher_model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: OVTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        ov_config: Optional[OVConfig] = None,
        task: Optional[str] = None,
    ):
        self.neftune_noise_alpha = None

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        self.ov_config = ov_config
        self.task = task
        self.teacher = None
        if teacher_model is not None:
            self.teacher = teacher_model.to(args.device)
            if self.args.n_gpu > 1:
                self.teacher = torch.nn.DataParallel(self.teacher)
            self.teacher.eval()
        self.compression_controller = None

        if self.ov_config is not None:
            if self.ov_config.compression is None:
                self.ov_config.compression = DEFAULT_QUANTIZATION_CONFIG
            if (
                isinstance(self.ov_config.compression, dict)
                and "algorithm" in self.ov_config.compression
                and self.ov_config.compression["algorithm"] == "quantization"
            ):
                self.ov_config.compression["export_to_onnx_standard_ops"] = self.ov_config.save_onnx_model
            elif isinstance(self.ov_config.compression, list):
                for i, algo_config in enumerate(self.ov_config.compression):
                    if algo_config["algorithm"] == "quantization":
                        self.ov_config.compression[i]["export_to_onnx_standard_ops"] = self.ov_config.save_onnx_model

            if self.args.do_train:
                self._set_task()
                train_dataloader = self.get_train_dataloader()
                model_inputs = next(iter(train_dataloader))
                for label_name in self.label_names:
                    model_inputs.pop(label_name)
                force_batch_one = self._is_pruning_enabled()
                self.ov_config.add_input_info(model_inputs, force_batch_one)
                nncf_config = NNCFConfig.from_dict(self.ov_config.__dict__)
                nncf_config.register_extra_structs(
                    [
                        QuantizationRangeInitArgs(OVDataLoader(train_dataloader)),
                        BNAdaptationInitArgs(OVDataLoader(train_dataloader)),
                    ]
                )

                # Configure NNCF logging
                # Disable nncf logging to stdout except error
                # but to file nncf_output.log
                nncf_config["log_dir"] = args.output_dir
                nncf_log_file_handler = logging.logging.FileHandler(os.path.join(args.output_dir, NNCF_LOG_FILE_NAME))
                nncf_log_file_handler.setFormatter(logging.logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
                nncf_logger.addHandler(nncf_log_file_handler)
                set_log_level(logging.ERROR)
                nncf_logger.setLevel(logging.INFO)
                nncf_log_file_handler.setLevel(logging.INFO)

                self.compression_controller, self.model = create_compressed_model(self.model, nncf_config)
                self.model_wrapped = self.model
                # TODO : To deprecate once support transformers > 4.30.0
                self.deepspeed = None

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size

        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        is_fsdp_xla_enabled = (
            self.is_fsdp_xla_enabled if is_transformers_version(">=", "4.36.0") else self.fsdp is not None
        )
        delay_optimizer_creation = is_sagemaker_mp_enabled() or is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        if is_transformers_version("<", "4.29.0"):
            is_distributed = self.args.local_rank != -1
        else:
            from accelerate.utils import DistributedType

            is_distributed = self.args.distributed_state.distributed_type != DistributedType.NO

        if self.compression_controller is not None and is_distributed:
            self.compression_controller.distributed()

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if is_transformers_version("<", "4.36.0") and use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        self.compression_metrics = defaultdict(lambda: torch.tensor(0.0).to(args.device))
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if self.compression_controller is not None:
                # Must be called at the beginning of each training epoch to prepare the compression method
                self.compression_controller.scheduler.epoch_step()
                nncf_logger.info(
                    "\nEpoch {} |".format(epoch).join(self.compression_controller.statistics().to_str().split("\n"))
                )

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if is_transformers_version(">=", "4.36.0") and self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    if self.compression_controller is not None:
                        # Must be called at the beginning of each training step to prepare the compression method
                        self.compression_controller.scheduler.step()

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if getattr(self, "do_grad_scaling", False):
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif getattr(self, "do_grad_scaling", False):
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def compute_distillation_loss(self, inputs, student_outputs):
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        teacher_logits = teacher_outputs.logits
        student_logits = student_outputs.logits
        temperature = self.args.distillation_temperature
        return F.kl_div(
            input=F.log_softmax(student_logits / temperature, dim=-1),
            target=F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.teacher is None:
            retval = super().compute_loss(model, inputs, return_outputs)

            if return_outputs is True:
                loss, outputs = retval
            else:
                loss = retval
        else:
            task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            if self.args.n_gpu > 1:
                task_loss = task_loss.mean()
            distillation_loss = self.compute_distillation_loss(inputs, outputs)
            loss = (1 - self.args.distillation_weight) * task_loss + self.args.distillation_weight * distillation_loss

            if model.training:
                self.compression_metrics["task_loss"] = task_loss.item()
                self.compression_metrics["distillation_loss"] = distillation_loss.item()

        if self.compression_controller is not None:
            compression_loss = self.compression_controller.loss()
            loss += compression_loss
            if model.training:
                self.compression_metrics["compression_loss"] = compression_loss.item()

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if model.training:
                for key, value in self.compression_metrics.items():
                    logs[key] = value

            if self.compression_controller is not None:
                compression_stats = self.compression_controller.statistics()
                for key, value in prepare_for_tensorboard(compression_stats).items():
                    logs["compression/{0}".format(key)] = value

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if not isinstance(self.model, PreTrainedModel):
            unwrapped_model = unwrap_model(self.model)
            is_pretrained_model = isinstance(unwrapped_model, PreTrainedModel)
            if state_dict is None:
                state_dict = self.model.state_dict()
            if is_pretrained_model:
                unwrapped_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.compression_controller is not None:
            # Save the configuration containing all the parameters related to quantization
            self.ov_config.save_pretrained(output_dir)

            # Export the compressed model to the ONNX format
            output_path = os.path.join(output_dir, OV_XML_FILE_NAME)
            self.compression_controller.prepare_for_export()
            model_type = self.model.config.model_type.replace("_", "-")
            exporter_config_class = TasksManager.get_exporter_config_constructor(
                exporter="onnx",
                model=self.model,
                task=self.task,
                model_type=model_type,
            )

            if self.task == "text-generation":
                onnx_config = exporter_config_class(self.model.config, use_past=self.model.config.use_cache)
            else:
                onnx_config = exporter_config_class(self.model.config)

            num_parameters = self.model.num_parameters()
            save_as_external_data = use_external_data_format(num_parameters) or self.ov_config.save_onnx_model
            f = io.BytesIO() if not save_as_external_data else os.path.join(output_dir, ONNX_WEIGHTS_NAME)

            opset = min(onnx_config.DEFAULT_ONNX_OPSET, MAX_ONNX_OPSET)
            opset = opset if not self.ov_config.save_onnx_model else max(opset, MIN_ONNX_QDQ_OPSET)
            _onnx_export_nncf_model(self.model, onnx_config, f, opset)
            ov_model = core.read_model(f) if save_as_external_data else core.read_model(f.getvalue(), b"")

            # Prune IR if structured pruning is conducted on the model
            if self._should_apply_pruning_transform():
                try:
                    # OpenVINO IR pruning requires static-shaped input
                    ov_model = self._reshape_ir(ov_model, static_shape=True)
                    apply_moc_transformations(ov_model, cf=False)
                    if self._get_compression_controller_by_cls(QuantizationController) is not None:
                        compress_quantize_weights_transformation(ov_model)
                    apply_pruning_transformation(ov_model)
                    apply_fused_names_cleanup(ov_model)
                    # Reshape back to dynamic shape IR
                    ov_model = self._reshape_ir(ov_model, static_shape=False)
                except Exception as err:
                    onnx_path = Path(output_dir, ONNX_WEIGHTS_NAME).resolve()
                    if not save_as_external_data:
                        onnx_path.write_bytes(f.getvalue())
                    logger.error(
                        f"Error encountered during OpenVINO IR pruning: {err}. {onnx_path} is dumped for debugging."
                    )
                    raise
            else:
                if self._get_compression_controller_by_cls(QuantizationController) is not None:
                    compress_quantize_weights_transformation(ov_model)

            # Serialize IR xml and bin
            save_model(ov_model, output_path, compress_to_fp16=False)

    def _get_compression_controller_by_cls(
        self, controller_cls: Type[PTCompressionAlgorithmController]
    ) -> Optional[PTCompressionAlgorithmController]:
        if isinstance(self.compression_controller, controller_cls):
            return self.compression_controller
        if isinstance(self.compression_controller, PTCompositeCompressionAlgorithmController):
            for child_controller in self.compression_controller.child_ctrls:
                if isinstance(child_controller, controller_cls):
                    return child_controller
        return None

    def _should_apply_pruning_transform(self) -> bool:
        movement_controller = self._get_compression_controller_by_cls(MovementSparsityController)
        return (
            movement_controller is not None
            and movement_controller.scheduler.enable_structured_masking
            and movement_controller.scheduler.current_stage == MovementSchedulerStage.POST_WARMUP
        )

    def _reshape_ir(self, ov_model: openvino.runtime.Model, static_shape: bool) -> openvino.runtime.Model:
        new_input_cfg = {}
        input_name_vs_shape = {item["keyword"]: item["sample_size"] for item in self.ov_config.input_info}
        for input_ in ov_model.inputs:
            if static_shape is True:
                new_input_cfg[input_.any_name] = PartialShape(
                    [1] + input_name_vs_shape[input_.any_name][1:]
                )  # use batch size of 1 for static shape IR
            else:
                new_input_cfg[input_.any_name] = PartialShape([-1] * len(input_.partial_shape))
        ov_model.reshape(new_input_cfg)
        return ov_model

    def _set_task(self):
        if self.task is None:
            raise ValueError("The model task defining the model topology needs to be specified for the ONNX export.")
        self.task = _TASK_ALIASES.get(self.task, self.task)

    def _is_pruning_enabled(compression: Union[Dict, List, None]):
        if isinstance(compression, dict) and compression["algorithm"] == "movement_pruning":
            return True
        if isinstance(compression, list):
            for algo_config in compression:
                if algo_config["algorithm"] == "movement_pruning":
                    return True
        return False
