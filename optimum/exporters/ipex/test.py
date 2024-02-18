import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from optimum.intel import IPEXModelForCausalLM


model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "gpt2"
model = IPEXModelForCausalLM.from_pretrained(model_id, export=True, torch_dtype=torch.bfloat16)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# from intel_extension_for_pytorch.transformers import optimize_transformers
# model = optimize_transformers(model, dtype=torch.bfloat16, inplace=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(text_generator("This is an example input"))