import torch
from transformers import AutoTokenizer, pipeline

from optimum.intel.ipex.modeling_base import IPEXModelForSequenceClassification


model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = IPEXModelForSequenceClassification.from_pretrained(model_id, export=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
text_classifer = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(text_classifer("This movie is disgustingly good !"))
