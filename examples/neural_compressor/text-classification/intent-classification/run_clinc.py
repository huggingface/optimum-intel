#!/usr/bin/env python
# coding=utf-8
#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

"""Finetuning the library models for sequence classification on Clinc."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import copy
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from neural_compressor import (
    DistillationConfig,
    PostTrainingQuantConfig,
    QuantizationAwareTrainingConfig,
    WeightPruningConfig,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from optimum.intel.neural_compressor import INCModel, INCQuantizer, INCTrainer


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_DISABLED"] = "true"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.20.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/neural_compressor/text-classification/intent-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="clinc_oos", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="plus", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a dataset name, a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert validation_extension == train_extension, (
                "`validation_file` should have the same extension (csv or json) as `train_file`."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    apply_quantization: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    quantization_approach: str = field(
        default="dynamic",
        metadata={"help": "Quantization approach. Supported approach are static, dynamic and aware_training."},
    )
    num_calibration_samples: int = field(
        default=50,
        metadata={"help": "Number of examples to use for the calibration step resulting from static quantization."},
    )
    apply_pruning: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply pruning."},
    )
    target_sparsity: float = field(
        default=0.1,
        metadata={"help": "Targeted sparsity when pruning the model."},
    )
    start_epoch: int = field(
        default=0,
        metadata={"help": "Epoch for which the pruning process will start."},
    )
    end_epoch: Optional[int] = field(
        default=None,
        metadata={"help": "Epoch for which the pruning process will end."},
    )
    pruning_approach: str = field(
        default="basic_magnitude",
        metadata={"help": "Pruning approach. Supported approach is basic_magnitude."},
    )
    apply_distillation: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply distillation."},
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=False, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    verify_loading: bool = field(
        default=False,
        metadata={"help": "Whether or not to verify the loading of the quantized model."},
    )
    # TODO : remove
    metric: Optional[str] = field(
        default=None,
        metadata={"help": "Metric used for the tuning strategy."},
    )
    tolerance_criterion: Optional[float] = field(
        default=None,
        metadata={"help": "Performance tolerance when optimizing the model."},
    )
    verify_loading: bool = field(
        default=False,
        metadata={"help": "Whether or not to verify the loading of the quantized model."},
    )
    examples_duplicate_ratio: float = field(
        default=100,
        metadata={
            "help": "How much times to duplicate the sentences dataset to construct the dataset "
            "of randomly paired two sentences for distillation of the SetFit model."
        },
    )


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output["last_hidden_state"]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SetFitModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(model, "config"):
            self.config = model.config

    def forward(self, input_ids, attention_mask, token_type_ids=None, *args, **kwargs):
        if token_type_ids is not None:
            model_output = self.model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
        else:
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask)
        return sentence_embeddings


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, optim_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, optim_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    class SetFitModelTraining(torch.nn.Module):
        def __init__(self, model, tokenizer):
            super().__init__()
            self.model = SetFitModel(model)
            self.tokenizer = tokenizer
            if hasattr(model, "config"):
                self.config = model.config

        def forward(self, sentences=None, *args, **kwargs):
            if not (isinstance(sentences, (tuple, list)) and len(sentences) == 2):
                raise ValueError("sentences should be a tuple or a list with 2 sentences string.")
            inputs = self.tokenizer(
                sentences[0] + sentences[1],
                padding=padding,
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            embeddings = self.model(**inputs)
            length = len(embeddings) // 2
            return {"logits": torch.cosine_similarity(embeddings[:length], embeddings[length:]), "loss": 0}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (examples["text"],)
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        result["label"] = examples["intent"]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        preprocessed_datasets = raw_datasets.map(
            preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache
        )
        preprocessed_datasets = preprocessed_datasets.remove_columns(["text", "intent"])

    if training_args.do_train or (optim_args.apply_quantization and optim_args.quantization_approach == "static"):
        if "train" not in preprocessed_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = preprocessed_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "test" not in preprocessed_datasets:
            raise ValueError("--do_eval requires a test dataset")
        eval_dataset = preprocessed_datasets["test"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(preds, labels):
        return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    quantization_config = None
    pruning_config = None
    distillation_config = None

    if not optim_args.apply_quantization and not optim_args.apply_pruning and not optim_args.apply_distillation:
        raise ValueError("No optimization activated.")

    if not training_args.do_train and (
        optim_args.apply_distillation
        or optim_args.apply_pruning
        or (optim_args.apply_quantization and optim_args.quantization_approach == "aware_training")
    ):
        raise ValueError("`do_train` must be set to True.")
    if optim_args.apply_quantization:
        supported_approach = {"static", "dynamic", "aware_training"}
        if optim_args.quantization_approach not in supported_approach:
            raise ValueError(
                f"Unknown quantization approach. Supported approach are {supported_approach}."
                f"{optim_args.quantization_approach} was given."
            )
        if optim_args.quantization_approach == "aware_training":
            quantization_config = QuantizationAwareTrainingConfig()
        else:
            quantization_config = PostTrainingQuantConfig(approach=optim_args.quantization_approach)

    if optim_args.apply_pruning:
        if optim_args.end_epoch is None:
            end_epoch = training_args.num_train_epochs
        else:
            end_epoch = min(optim_args.end_epoch, training_args.num_train_epochs - 1)

        pruning_config = WeightPruningConfig(
            start_epoch=optim_args.start_epoch,
            end_epoch=end_epoch,
            target_sparsity=optim_args.target_sparsity,
            prune_type=optim_args.pruning_approach,
        )

    if optim_args.apply_distillation:
        if optim_args.teacher_model_name_or_path is None:
            raise ValueError("A teacher model is needed to apply distillation.")

        teacher_model = AutoModel.from_pretrained(
            optim_args.teacher_model_name_or_path,
            from_tf=bool(".ckpt" in optim_args.teacher_model_name_or_path),
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            optim_args.teacher_model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        teacher_model = SetFitModelTraining(teacher_model, teacher_tokenizer)
        teacher_model.to(training_args.device)

        teacher_tokenizer = AutoTokenizer.from_pretrained(optim_args.teacher_model_name_or_path, use_fast=True)
        if teacher_tokenizer.vocab != tokenizer.vocab:
            raise ValueError("Teacher model and student model should have same tokenizer.")

        distillation_config = DistillationConfig(teacher_model=teacher_model)

        examples = raw_datasets["train"]["text"]
        if data_args.max_train_samples is not None:
            examples_duplicate = raw_datasets["train"].select(range(data_args.max_train_samples))["text"]
        else:
            examples_duplicate_ratio = optim_args.examples_duplicate_ratio
            examples_duplicate = []
            for i in range(int(examples_duplicate_ratio)):
                examples_duplicate.extend(examples)
            examples_duplicate.extend(
                examples[: int(len(examples) * (examples_duplicate_ratio - int(examples_duplicate_ratio)))]
            )
        shuffled_examples_duplicate = copy.deepcopy(examples_duplicate)
        distillation_dataset = list(zip(examples_duplicate, shuffled_examples_duplicate))

        def sentences_data_collator(sentences_pairs):
            return {"sentences": [[sp[i] for sp in sentences_pairs] for i in range(len(sentences_pairs[0]))]}

        train_dataset = distillation_dataset

    # Initialize our Trainer
    trainer = INCTrainer(
        model=model,
        task="default",
        quantization_config=quantization_config if optim_args.quantization_approach == "aware_training" else None,
        pruning_config=pruning_config,
        distillation_config=distillation_config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # if optim_args.apply_distillation:
        # data_collator = trainer.data_collator
        # trainer.data_collator = sentences_data_collator

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # if optim_args.apply_distillation:
        # trainer.data_collator = data_collator

    if optim_args.apply_quantization and optim_args.quantization_approach in {"static", "dynamic"}:
        model = trainer.model if isinstance(trainer.model, PreTrainedModel) else trainer.model._model
        quantizer = INCQuantizer.from_pretrained(model)
        if optim_args.quantization_approach == "static":
            num_calibration_samples = min(len(train_dataset), optim_args.num_calibration_samples)
            train_dataset = train_dataset.select(range(num_calibration_samples))
            quantization_config.calibration_sampling_size = num_calibration_samples

        quantizer.quantize(
            quantization_config=quantization_config,
            save_directory=training_args.output_dir,
            calibration_dataset=train_dataset if optim_args.quantization_approach == "static" else None,
            batch_size=training_args.per_device_train_batch_size,
        )

    if optim_args.apply_quantization and optim_args.verify_loading:
        loaded_model = INCModel.from_pretrained(training_args.output_dir)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        with torch.no_grad():
            original_model_outputs = quantizer._quantized_model(**tokens)
            quantized_model_outputs = loaded_model(**tokens)
            if torch.allclose(original_model_outputs.pooler_output, quantized_model_outputs.pooler_output, atol=1e-4):
                logger.info("The quantized model was successfully loaded.")
            else:
                logger.warning("The quantized model was not successfully loaded.")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
