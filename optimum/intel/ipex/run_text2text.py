import argparse
import time

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)
from transformers.utils import ContextManagers

from optimum.intel import inference_mode


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str)
parser.add_argument("--dtype", default="bf16", type=str)
parser.add_argument("--jit", default=False, type=bool)
parser.add_argument("--use_ipex", default=False, type=bool)

args = parser.parse_args()

kwargs = {}
if args.dtype == "bf16":
    kwargs["torch_dtype"] = torch.bfloat16
else:
    kwargs["torch_dtype"] = torch.float32
kwargs["use_cache"] = True
kwargs["low_cpu_mem_usage"] = True
kwargs["return_dict"] = True
model_id = args.model_id
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()
# model = model.to(memory_format=torch.channels_last)

if kwargs["torch_dtype"] == torch.bfloat16:
    model.to(torch.bfloat16)

input_seq = (
    "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"
)
input_length = len(tokenizer(input_seq)["input_ids"])

print("Input sequence is: ")
print(input_seq)

generate_kwargs = {"max_length": 64, "min_length": 8, "do_sample": False, "num_beams": 4, "num_beam_groups": 1}
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
batch_size = generate_kwargs.get("num_beams", 1)


def run_pipeline(generator, num_batches=10):
    for i in range(num_batches):
        pre = time.time()
        out = generator(input_seq, **generate_kwargs)
        print(f"origin model infer costs {time.time()-pre} seconds")
        print(out)
        real_output_token_num = len(tokenizer.batch_encode_plus([out[0]["generated_text"]])["input_ids"][0])
        print(f"Real output tokens: {real_output_token_num}")


init_context = []
if kwargs["torch_dtype"] == torch.bfloat16:
    init_context.append(torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16))


init_context.append(torch.inference_mode())
with ContextManagers(init_context):
    run_pipeline(generator)
init_context.pop()


if args.jit:
    generator.model.config.return_dict = False
with ContextManagers(init_context), inference_mode(
    generator,
    dtype=kwargs["torch_dtype"],
    verbose=False,
    jit=args.jit,
    use_ipex=args.use_ipex,
    input_length=input_length,
    batch_size=batch_size,
) as trace_pipe:
    run_pipeline(trace_pipe)


generator.model = torch.compile(model)
print(generator.model)
generator.model.config.return_dict = True
init_context.append(torch.inference_mode())
with ContextManagers(init_context):
    run_pipeline(generator)
init_context.pop()
generator.model = model
