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

import copy
import math
import os
import sys
import time
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.distributed as dist

# from packaging import version
from torch import nn
from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init
from transformers.file_utils import WEIGHTS_NAME

# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params
from transformers.modeling_utils import get_parameter_dtype, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import HPSearchBackend, ShardedDDPOption, TrainOutput, has_length, speed_metrics
from transformers.utils import is_sagemaker_mp_enabled, logging

from neural_compressor.experimental.export import torch_to_fp32_onnx, torch_to_int8_onnx

from ..utils.import_utils import is_neural_compressor_version
from .utils import MIN_QDQ_ONNX_OPSET, ONNX_WEIGHTS_NAME, TRAINING_ARGS_NAME


if TYPE_CHECKING:
    import optuna


__version__ = "4.22.2"


logger = logging.get_logger(__name__)


from itertools import chain
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

from neural_compressor import training
from neural_compressor.conf.pythonic_config import _BaseQuantizationConfig
from neural_compressor.model.torch_model import PyTorchModel
from optimum.exporters import TasksManager


class INCTrainer(Trainer):
    """
    INCTrainer enables Intel Neural Compression quantization aware training, pruning and distillation.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        quantization_config: Optional[_BaseQuantizationConfig] = None,
        pruning_config: Optional[_BaseQuantizationConfig] = None,
        distillation_config: Optional[_BaseQuantizationConfig] = None,
        task: Optional[str] = None,
        save_onnx_model: bool = False,
    ):
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

        if self.args.device.type == "cuda" and not is_neural_compressor_version(">", "2.0.0"):
            logger.warning(
                "Neural Compressor version must be > 2.0.0 to train on CUDA devices. "
                "Please upgrade Neural Compressor or train your model on CPU devices instead."
            )

        inc_config = []
        self.task = task
        self.quantization_config = quantization_config
        self.pruning_config = pruning_config
        self.distillation_config = distillation_config
        self._compression_manager = None
        self.save_onnx_model = save_onnx_model

        # Attach dtype and architecture to the config
        self.dtype = "int8" if quantization_config is not None else str(get_parameter_dtype(self.model)).split(".")[1]
        self.model.config.torch_dtype = self.dtype
        self.model.config.framework = "pytorch_fx"
        self.model.config.backend = "default"
        self.model.config.architectures = [self.model.__class__.__name__]

        self._set_signature_columns_if_needed()

        for config in [quantization_config, pruning_config, distillation_config]:
            if config is not None:
                inc_config.append(config)

        if len(inc_config) >= 1 and self.args.do_train:
            inc_config = inc_config if len(inc_config) > 1 else inc_config.pop()
            self._compression_manager = training.prepare_compression(self.model, confs=inc_config)
            self.model = self._compression_manager.model.model
            self.model_wrapped = self.model

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
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
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
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

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

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
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
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
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if self._compression_manager is not None:
            self._compression_manager.callbacks.on_train_begin()

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if self._compression_manager is not None:
                self._compression_manager.callbacks.on_epoch_begin(epoch)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
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
                    if self._compression_manager is not None:
                        self._compression_manager.callbacks.on_step_begin(step)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    if self._compression_manager is not None:
                        self._compression_manager.callbacks.on_before_optimizer_step()

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if self._compression_manager is not None:
                        self._compression_manager.callbacks.on_after_optimizer_step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self._compression_manager is not None:
                        self._compression_manager.callbacks.on_step_end()

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
            if self._compression_manager is not None:
                self._compression_manager.callbacks.on_epoch_end()

            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.

            if args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self._compression_manager is not None:
            self._compression_manager.callbacks.on_train_end()

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False, save_onnx_model=None):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """
        save_onnx_model = save_onnx_model if save_onnx_model is not None else self.save_onnx_model

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            self._save(output_dir=output_dir, save_onnx_model=save_onnx_model)

    def _save(self, output_dir: Optional[str] = None, state_dict=None, save_onnx_model=False):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        if os.path.isfile(output_dir):
            logger.error(f"Provided path ({output_dir}) should be a directory, not a file")
            return

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)

        # Save the config
        if self.model.config is not None:
            self.model.config.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Save the model
        if state_dict is None:
            state_dict = self.model.state_dict()
            if self._compression_manager is not None and hasattr(self._compression_manager.model, "q_config"):
                state_dict["best_configure"] = self._compression_manager.model.q_config
        torch.save(state_dict, output_model_file)

        # Export the compressed model to the ONNX format
        if save_onnx_model:
            self._set_task()
            model_type = self.model.config.model_type.replace("_", "-")
            model_name = getattr(self.model, "name", None)
            onnx_config_class = TasksManager.get_exporter_config_constructor(
                exporter="onnx", model=self.model, task=self.task, model_type=model_type, model_name=model_name
            )
            onnx_config = onnx_config_class(self.model.config)
            output_onnx_path = os.path.join(output_dir, ONNX_WEIGHTS_NAME)

            signature_columns = copy.deepcopy(self._signature_columns)
            self._signature_columns = list(
                set(self._signature_columns) - set(["label", "label_ids"] + self.label_names)
            )
            self._signature_columns = signature_columns

            self.model.eval()
            self._onnx_export(self.model, onnx_config, output_onnx_path)

        logger.info(f"Model weights saved in {output_model_file}")

    def _set_task(self):
        if self.task is None:
            raise ValueError("The model task defining the model topology needs to be specified for the ONNX export.")
        elif self.task in ["sentiment-analysis", "text-classification", "zero-shot-classification"]:
            self.task = "sequence-classification"
        elif self.task in ["feature-extraction", "fill-mask"]:
            self.task = "default"

    def _onnx_export(self, model: nn.Module, config: "OnnxConfig", output_path: str):
        opset = min(config.DEFAULT_ONNX_OPSET, MIN_QDQ_ONNX_OPSET)
        dynamic_axes = {name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())}
        inputs = config.generate_dummy_inputs(framework="pt")
        device = model.device
        inputs = dict((k, v.to(device)) for k, v in inputs.items())

        if self.dtype == "int8":
            torch_to_int8_onnx(
                fp32_model=self._compression_manager.fp32_model.model.to(device),
                int8_model=model,
                q_config=self._compression_manager.model.q_config,
                save_path=output_path,
                example_inputs=inputs,
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                input_names=list(config.inputs.keys()),
                output_names=list(config.outputs.keys()),
            )
        else:
            torch_to_fp32_onnx(
                fp32_model=model,
                save_path=output_path,
                example_inputs=inputs,
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                input_names=list(config.inputs.keys()),
                output_names=list(config.outputs.keys()),
                do_constant_folding=True,
            )

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        signature_columns += ["teacher_logits"]

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        return dataset.remove_columns(ignored_columns)

    @staticmethod
    def _get_logits(model_outputs):
        output_names = ["logits", "start_logits", "end_logits"]
        return tuple(model_outputs.get(name) for name in output_names if name in model_outputs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        teacher_outputs = inputs.pop("teacher_logits", None)
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.distillation_config is not None:
            student_outputs = self._get_logits(outputs)
            if teacher_outputs is not None:
                if len(teacher_outputs.shape) == 3 and teacher_outputs.shape[1] == 2:
                    teacher_outputs = tuple(teacher_outputs.transpose(1, 0))
            else:
                self.distillation_config.teacher_model.eval()
                self.distillation_config.teacher_model.to(model.device)
                teacher_outputs = self.distillation_config.teacher_model(**inputs)
                teacher_outputs = self._get_logits(teacher_outputs)

            if teacher_outputs is not None:
                distillation_loss = self.compute_distillation_loss(student_outputs, teacher_outputs)
                loss *= self._compression_manager.callbacks.callbacks.criterion.loss_weights[0]
                loss += distillation_loss * self._compression_manager.callbacks.callbacks.criterion.loss_weights[1]
                loss /= sum(self._compression_manager.callbacks.callbacks.criterion.loss_weights)

                if isinstance(outputs, dict):
                    outputs["loss"] = loss
                else:
                    outputs[0] = loss

        return (loss, outputs) if return_outputs else loss

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.model.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

    @staticmethod
    def _get_logits(model_outputs):
        output_names = ["logits", "start_logits", "end_logits"]
        return tuple(model_outputs.get(name) for name in output_names if name in model_outputs)

    def compute_distillation_loss(self, student_outputs, teacher_outputs):
        """
        How the distillation loss is computed given the student and teacher outputs.
        """
        distillation_loss = None
        temperature = self._compression_manager.callbacks.callbacks.criterion.temperature
        for student_output, teacher_output in zip(student_outputs, teacher_outputs):
            student_output = student_output / temperature
            teacher_output = teacher_output / temperature
            loss = self._compression_manager.callbacks.callbacks.criterion.teacher_student_loss_cal(
                student_output, teacher_output
            )
            distillation_loss = loss if distillation_loss is None else distillation_loss + loss
        distillation_loss *= temperature**2
        return distillation_loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if self.quantization_config is not None:
            logger.warning("Evaluation of quantized models is not supported by the CUDA backend.")
            self.model.to("cpu")
        if getattr(self.model.config, "backend", None) == "ipex":
            self.args.use_ipex = False
            self.args.bf16 = False
            self.use_cpu_amp = False
        if (
            getattr(self.model.config, "torch_dtype", None) == "int8"
            and getattr(self.model.config, "framework", None) in {"pytorch", "pytorch_fx"}
            and self.use_cpu_amp
        ):
            logger.warn(
                f"{self.model.config.framework} quantized model doesn't support BFloat16 input, setting `use_cpu_amp` to False."
            )
            self.use_cpu_amp = False
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def predict(self, *args, **kwargs):
        if self.quantization_config is not None:
            logger.warning("Evaluation of quantized models is not supported by the CUDA backend.")
            self.model.to("cpu")
        return super().predict(*args, **kwargs)

    def get_model_sparsity(self):
        sparsity = 0.0
        if self._compression_manager is not None:
            sparsity = self._compression_manager.model.report_sparsity()[-1]
        return sparsity


class IncTrainer(INCTrainer):
    # Warning at import time
    warnings.warn(
        "The class `IncTrainer` has been depreciated and will be removed in optimum-intel v1.7, please use "
        "`INCTrainer` instead.",
    )
