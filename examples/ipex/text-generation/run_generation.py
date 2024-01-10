import torch
from transformers import AutoTokenizer, pipeline

from optimum.intel.ipex.modeling_decoder import IPEXModelForCausalLM


model_id = "gpt2"
model = IPEXModelForCausalLM.from_pretrained(model_id, export=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(text_generator("This is an example input"))
