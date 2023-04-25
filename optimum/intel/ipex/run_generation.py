import argparse
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from transformers.utils import ContextManagers

from optimum.intel import inference_mode


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str)
parser.add_argument("--dtype", default="fp32", type=str)
parser.add_argument("--jit", default=False, type=bool)
parser.add_argument("--use_ipex", default=False, type=bool)
parser.add_argument("--enable_amp_autocast", default=False, type=bool)

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
model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()
# model = model.to(memory_format=torch.channels_last)

# if kwargs["torch_dtype"] == torch.bfloat16:
#     model.to(torch.bfloat16)

input_seq = "Given the context please answer the question. Context: Berlin is the capital of Germany. Paris is the capital of France.; Question: What is the capital of Germany?; Answer:"

print("Input sequence is: ")
print(input_seq)

# generate_kwargs = {'return_full_text': False, 'max_new_tokens': 128, 'num_return_sequences': 1, 'num_beams': 4, 'do_sample': False, 'min_new_tokens': 32, 'no_repeat_ngram_size': 2, 'early_stopping': True, 'forced_eos_token_id': [13, 30, 0]}
generate_kwargs = {
    "max_new_tokens": 32,
    "do_sample": False,
    "num_beams": 4,
    "num_beam_groups": 1,
    "return_full_text": False,
    "no_repeat_ngram_size": 2,
}
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def run_pipeline(generator, num_batches=5):
    for i in range(num_batches):
        pre = time.time()
        out = generator(input_seq, **generate_kwargs)
        print(f"origin model infer costs {time.time()-pre} seconds")
        print(out)
        real_output_token_num = len(tokenizer.batch_encode_plus([out[0]["generated_text"]])["input_ids"][0])
        print(f"Real output tokens: {real_output_token_num}")


with torch.cpu.amp.autocast(enabled=args.enable_amp_autocast), torch.inference_mode():
    run_pipeline(generator)


with torch.cpu.amp.autocast(enabled=args.enable_amp_autocast), inference_mode(generator, dtype=torch.bfloat16, verbose=False,jit=args.jit) as trace_pipe:
    run_pipeline(trace_pipe)

