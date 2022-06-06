<p align="center">
    <img src="readme_logo.png" />
</p>

# Optimum Intel

🤗 Optimum Intel is the interface between the 🤗 Transformers library and the different tools and libraries provided by Intel to accelerate end-to-end pipelines on Intel architectures.

Intel [Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) (INC) is an open-source library enabling the usage of the most popular compression techniques such as quantization, pruning and knowledge distillation. It supports automatic accuracy-driven tuning strategies in order for users to easily obtain the best quantized model. The users can easily apply static, dynamic and aware-training quantization approaches giving an expected accuracy criteria. It also supports different weight pruning techniques enabling the creation of pruned model giving a predefined sparsity target.

## Install
To install the latest release of this package:

`pip install optimum[intel]`

Optimum Intel is a fast-moving project, and you may want to install from source.

`pip install git+https://github.com/huggingface/optimum-intel.git`


## Running the examples

There are a number of examples provided in the `examples` directory.

Please install the requirements for every example:

```
cd <example-folder>
pip install -r requirements.txt
```

## How to use it?

Here is an example on how to combine magnitude pruning with dynamic quantization while fine-tuning a DistilBERT on the sst-2 task.
Note that quantization is currently only supported for CPUs (only CPU backends are available), so we will not be utilizing GPUs / CUDA in this example.

```python
import os
import yaml
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
from optimum.intel.neural_compressor import IncOptimizer, IncPruner, IncQuantizer, IncTrainer
from optimum.intel.neural_compressor.configuration import IncPruningConfig, IncQuantizationConfig
from optimum.intel.neural_compressor.quantization import IncQuantizedModelForSequenceClassification

# The model we wish to apply pruning and quantization on
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# The targeted sparsity is chosen to be 10%
target_sparsity = 0.1
# The output directory where the model checkpoint will be saved
save_dir = "/tmp/quantized_pruned_distilbert_sst2_2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
metric = load_metric("glue", "sst2")
dataset = load_dataset("glue", "sst2")
dataset = dataset.map(
    lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
)
train_dataset = dataset["train"].select(range(256))
eval_dataset = dataset["validation"].select(range(256))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    return result

config_path = "echarlaix/distilbert-sst2-inc-dynamic-quantization-magnitude-pruning-0.1"
# Loaad our quantization configuration detailing the quantization we wish to apply
quantization_config = IncQuantizationConfig.from_pretrained(config_path, config_file_name="quantization.yml")
# Loaad our quantization configuration detailing the pruning we wish to apply
pruning_config = IncPruningConfig.from_pretrained(config_path, config_file_name="prune.yml")

# Initialize our IncTrainer
trainer = IncTrainer(
    model=model,
    args=TrainingArguments(save_dir, num_train_epochs=3.0),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

def train_func(model):
    trainer.model_wrapped = model
    trainer.model = model
    _ = trainer.train(pruner)
    return trainer.model

def eval_func(model):
    trainer.model = model
    metrics = trainer.evaluate()
    return metrics.get("eval_accuracy")

inc_quantizer = IncQuantizer(model, quantization_config, eval_func=eval_func)
quantizer = inc_quantizer.fit()
inc_pruner = IncPruner(model, pruning_config, eval_func=eval_func, train_func=train_func)
pruner = inc_pruner.fit()
inc_optimizer = IncOptimizer(model, quantizer=quantizer, pruner=pruner)
opt_model = inc_optimizer.fit()

# Save the resulting model and final configuration (needed to load the model)
trainer.save_model(save_dir)
with open(os.path.join(save_dir, "best_configure.yaml"), "w") as f:
    yaml.dump(opt_model.tune_cfg, f, default_flow_style=False)

# To load the resulting model, you can do as follows :
loaded_model = IncQuantizedModelForSequenceClassification.from_pretrained(save_dir)

# To load a model hosted on the hub obtained after applying INC quantization, you can do as follows :
loaded_model_from_hub = IncQuantizedModelForSequenceClassification.from_pretrained(
    "echarlaix/distilbert-sst2-inc-dynamic-quantization-magnitude-pruning-0.1"
)
```
