import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import transformers
from datasets import load_metric
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import HfArgumentParser, TrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel
from transformers.utils import check_min_version

from optimum.intel.neural_compressor import IncOptimizer, IncQuantizationConfig, IncQuantizationMode, IncQuantizer
from optimum.intel.neural_compressor.quantization import IncQuantizedModelForVision2Seq


os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

logger = logging.getLogger(__name__)


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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
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
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    max_calibration_samples: Optional[int] = field(
        default=100, metadata={"help": ("The Number of samples to calibration quantization sacle and zero point.")}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": ("Number of samples to evaluation.")},
    )

    datasets_dir: str = field(default=None, metadata={"help": "The input testing data path."})

    def __post_init__(self):
        if self.datasets_dir is None:
            raise ValueError("Need a dataset path")


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    apply_quantization: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    quantization_approach: Optional[str] = field(
        default=None,
        metadata={"help": "Quantization approach. Supported approach are static and dynamic."},
    )
    quantization_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing the YAML configuration file used to control the quantization and "
            "tuning behavior."
        },
    )
    tolerance_criterion: Optional[float] = field(
        default=None,
        metadata={"help": "Performance tolerance when optimizing the model."},
    )
    verify_loading: bool = field(
        default=False,
        metadata={"help": "Whether or not to verify the loading of the quantized model."},
    )
    warmup_iter: int = field(
        default=5,
        metadata={"help": "For benchmark measurement only."},
    )


def main():
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
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    class IAMDataset(Dataset):
        def __init__(self, root_dir, processor, max_target_length=128, max_samples=None):
            self.samples_dir = os.path.join(root_dir, "image")
            samples_file = os.path.join(root_dir, "gt_test.txt")
            df = pd.read_fwf(samples_file, header=None)
            df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
            if max_samples is not None and max_samples < len(df):
                df.drop(labels=range(max_samples, len(df)), inplace=True)
            del df[2]
            self.df = df
            self.processor = processor
            self.max_target_length = max_target_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            # get file name + text
            file_name = self.df["file_name"][idx]
            text = self.df["text"][idx]
            # some file names end with jp instead of jpg, the two lines below fix this
            if file_name.endswith("jp"):
                file_name = file_name + "g"
            # prepare image (i.e. resize + normalize)
            image = Image.open(os.path.join(self.samples_dir, file_name)).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            # add labels (input_ids) by encoding the text
            labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
            # important: make sure that PAD tokens are ignored by the loss function
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
            return encoding

    processor = TrOCRProcessor.from_pretrained(model_args.model_name_or_path)
    test_dataset = IAMDataset(
        root_dir=data_args.datasets_dir, processor=processor, max_samples=data_args.max_eval_samples
    )
    test_dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size)

    device = torch.device("cpu")

    model = VisionEncoderDecoderModel.from_pretrained(model_args.model_name_or_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.encoder.num_beams = data_args.num_beams
    model.to(device)

    cer = load_metric("cer")

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self, name, fmt=":f"):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)

    def eval_func(model, iters=None):
        batch_time = AverageMeter("Time", ":6.3f")

        for i, batch in enumerate(test_dataloader):
            # predict using generate
            pixel_values = batch["pixel_values"].to(device)
            # setting channels_last
            pixel_values = pixel_values.to(memory_format=torch.channels_last)
            if i >= optim_args.warmup_iter:
                start = time.time()
            outputs = model.generate(pixel_values, num_beams=data_args.num_beams)
            # measure elapsed time
            if i >= optim_args.warmup_iter:
                batch_time.update(time.time() - start)

            # decode
            pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(labels, skip_special_tokens=True)

            # add batch to metric
            cer.add_batch(predictions=pred_str, references=label_str)
            if iters is not None and i >= iters:
                break
        score = cer.compute()
        print("Batch size = %d" % training_args.per_device_eval_batch_size)
        if training_args.per_device_eval_batch_size == 1:
            print("Latency: %.3f ms" % (batch_time.avg * 1000))
        if batch_time.avg == 0:
            print("The time of evaluation is 0, Please check it.")
        else:
            print("Throughput: %.3f images/sec" % (training_args.per_device_eval_batch_size / batch_time.avg))

        # TODO: this should also be done with the ProgressMeter
        print("cer: {score:.5f}".format(score=(score)))

        return score

    if optim_args.apply_quantization:
        default_config = os.path.join(
            os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), "config"
        )
        q8_config = IncQuantizationConfig.from_pretrained(
            optim_args.quantization_config if optim_args.quantization_config is not None else default_config,
            config_file_name="quantization.yml",
            cache_dir=model_args.cache_dir,
        )
        # Set metric tolerance if specified
        if optim_args.tolerance_criterion is not None:
            q8_config.set_tolerance(optim_args.tolerance_criterion)
        # Set quantization approach if specified
        if optim_args.quantization_approach is not None:
            supported_approach = {"static", "dynamic"}
            if optim_args.quantization_approach not in supported_approach:
                raise ValueError(
                    "Unknown quantization approach. Supported approach are " + ", ".join(supported_approach)
                )
            quant_approach = getattr(IncQuantizationMode, optim_args.quantization_approach.upper()).value
            q8_config.set_config("quantization.approach", quant_approach)
        quant_approach = IncQuantizationMode(q8_config.get_config("quantization.approach"))
        # torch FX used for post-training quantization and quantization aware training
        # dynamic quantization will be added when torch FX is more mature
        if quant_approach != IncQuantizationMode.DYNAMIC:
            q8_config.set_config("model.framework", "pytorch_fx")
        if quant_approach == IncQuantizationMode.STATIC:
            q8_config.set_config("quantization.calibration.sampling_size", [data_args.max_calibration_samples])
        q8_config.set_config("tuning.accuracy_criterion.higher_is_better", False)
        quantizer = IncQuantizer(q8_config, eval_func=eval_func, calib_dataloader=test_dataloader)
        optimizer = IncOptimizer(model, quantizer=quantizer)
        q_model = optimizer.fit()
        result_optimized_model = eval_func(q_model, 20)

        # Save the resulting model and its corresponding configuration in the given directory
        optimizer.save_pretrained(training_args.output_dir)

    if optim_args.apply_quantization and optim_args.verify_loading:
        print("loading int8 model...")
        loaded_model = IncQuantizedModelForVision2Seq.from_pretrained(training_args.output_dir)
        loaded_model.eval()

        print("Running evaluation on reloaded model...")
        result_loaded_model = eval_func(loaded_model, 20)
        print("Character error rate on test set:", result_loaded_model)
        if result_loaded_model != result_optimized_model:
            logger.error("The quantized model was not successfully loaded.")
        else:
            logger.info(f"The quantized model was successfully loaded.")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
