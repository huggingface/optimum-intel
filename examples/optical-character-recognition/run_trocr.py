import argparse
import os
import time

import pandas as pd
import torch
from datasets import load_metric
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from optimum.intel.neural_compressor import IncOptimizer, IncQuantizationConfig, IncQuantizationMode, IncQuantizer
from optimum.intel.neural_compressor.quantization import IncQuantizedModelForVision2Seq


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument(
    "-t", "--tune", dest="tune", action="store_true", help="tune best int8 model on calibration dataset"
)
parser.add_argument("-w", "--warmup_iter", default=5, type=int, help="For benchmark measurement only.")
parser.add_argument(
    "-b",
    "--batch_size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument("-d", "--datasets_dir", default="IAM", type=str, help="Path to the IAM datasets")
parser.add_argument(
    "-m",
    "--model_name_or_path",
    default=None,
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "-q",
    "--quantization_config",
    default=None,
    type=str,
    help="Path to the directory containing the YAML configuration file used to "
    "control the quantization and tuning behavior.",
)
parser.add_argument(
    "--quantization_approach",
    default="static",
    type=str,
    help="Quantization approach. Supported approach are static, dynamic and aware_training.",
)
parser.add_argument(
    "-c",
    "--cache_dir",
    default=None,
    type=str,
    help="Path to directory to store the pretrained models downloaded from " "huggingface.co.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    default="saved_results",
    type=str,
    help="Path to directory to store the resulting model and its corresponding configuration",
)
parser.add_argument(
    "--tolerance_criterion", default=0.15, type=float, help="Performance tolerance when optimizing the model."
)
parser.add_argument(
    "-v",
    "--verify_loading",
    dest="verify_loading",
    action="store_true",
    help="Whether or not to verify the loading of the quantized model and run benchmark.",
)

args = parser.parse_args()
print(args)

test_file = os.path.join(args.datasets_dir, "gt_test.txt")
test_dir = os.path.join(args.datasets_dir, "image")
df = pd.read_fwf(test_file, header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
del df[2]
df.head()


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
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
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


processor = TrOCRProcessor.from_pretrained(args.model_name_or_path)
test_dataset = IAMDataset(root_dir=test_dir, df=df, processor=processor)

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained(args.model_name_or_path)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
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
        if i >= args.warmup_iter:
            start = time.time()
        outputs = model.generate(pixel_values)
        # measure elapsed time
        if i >= args.warmup_iter:
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
    print("Batch size = %d" % args.batch_size)
    if args.batch_size == 1:
        print("Latency: %.3f ms" % (batch_time.avg * 1000))
    print("Throughput: %.3f images/sec" % (args.batch_size / batch_time.avg))

    # TODO: this should also be done with the ProgressMeter
    print("cer: {score:.5f}".format(score=(score)))

    return score


if args.tune:
    torch.backends.quantized.engine = "onednn"
    default_config = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), "config")
    q8_config = IncQuantizationConfig.from_pretrained(
        args.quantization_config if args.quantization_config is not None else default_config,
        config_file_name="quantization.yml",
        cache_dir=args.cache_dir,
    )
    # Set metric tolerance if specified
    if args.tolerance_criterion is not None:
        q8_config.set_tolerance(args.tolerance_criterion)
    # Set quantization approach if specified
    if args.quantization_approach is not None:
        supported_approach = {"static", "dynamic", "aware_training"}
        if args.quantization_approach not in supported_approach:
            raise ValueError("Unknown quantization approach. Supported approach are " + ", ".join(supported_approach))
        quant_approach = getattr(IncQuantizationMode, args.quantization_approach.upper()).value
        q8_config.set_config("quantization.approach", quant_approach)
    quant_approach = IncQuantizationMode(q8_config.get_config("quantization.approach"))
    # torch FX used for post-training quantization and quantization aware training
    # dynamic quantization will be added when torch FX is more mature
    if quant_approach != IncQuantizationMode.DYNAMIC:
        q8_config.set_config("model.framework", "pytorch_fx")
    q8_config.set_config("tuning.accuracy_criterion.higher_is_better", False)
    quantizer = IncQuantizer(q8_config, eval_func=eval_func, calib_dataloader=test_dataloader)
    optimizer = IncOptimizer(model, quantizer=quantizer)
    q_model = optimizer.fit()
    optimizer.save_pretrained(args.output_dir)

if args.verify_loading:
    torch.backends.quantized.engine = "onednn"
    print("loading int8 model...")
    model = IncQuantizedModelForVision2Seq.from_pretrained(args.output_dir)
    model.eval()

print("Running evaluation...")
final_score = eval_func(model, 20)

print("Character error rate on test set:", final_score)
