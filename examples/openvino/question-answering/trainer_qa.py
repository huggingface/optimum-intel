# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
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
"""
A subclass of `OVTrainer` specific to Question-Answering tasks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.trainer_utils import PredictionOutput

from optimum.intel import OVTrainer


class QuestionAnsweringOVTrainer(OVTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)

    def compute_distillation_loss(self, inputs, student_outputs):
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        temperature = self.args.distillation_temperature
        distilliation_loss_start = F.kl_div(
            input=F.log_softmax(student_outputs.start_logits / temperature, dim=-1),
            target=F.softmax(teacher_outputs.start_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
        distilliation_loss_end = F.kl_div(
            input=F.log_softmax(student_outputs.end_logits / temperature, dim=-1),
            target=F.softmax(teacher_outputs.end_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature**2)
        return (distilliation_loss_start + distilliation_loss_end) / 2.0

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.teacher is None:
            retval = super().compute_loss(model, inputs, return_outputs)

            if return_outputs is True:
                loss, outputs = retval
            else:
                loss = retval
        else:
            # compute_loss is not used as QA distillation requires custom handling for outputs
            # Using compute_loss incurs excessive computational footprint
            outputs = self.model(**inputs)

            task_loss_start = self.criterion(outputs.start_logits, inputs["start_positions"])
            task_loss_end = self.criterion(outputs.end_logits, inputs["end_positions"])
            task_loss = (task_loss_start + task_loss_end) / 2.0

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
