<p align="center">
    <img src="readme_logo.png" />
</p>

# Optimum Intel

🤗 Optimum Intel is the interface between the 🤗 Transformers library and the different tools and libraries provided by Intel to accelerate end-to-end pipelines on Intel architectures.

Intel [Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) is an open-source library enabling the usage of the most popular compression techniques such as quantization, pruning and knowledge distillation. It supports automatic accuracy-driven tuning strategies in order for users to easily generate quantized model. The users can easily apply static, dynamic and aware-training quantization approaches while giving an expected accuracy criteria. It also supports different weight pruning techniques enabling the creation of pruned model giving a predefined sparsity target.

[OpenVINO](https://docs.openvino.ai/latest/index.html) is an open-source toolkit that enables high performance inference capabilities for Intel CPUs, GPUs, and special DL inference accelerators. It is supplied with a set of tools to optimize and quantize models. Optimum Intel provides a simple interface to optimize Transformer models, convert them to OpenVINO Intermediate Representation format and to run inference using OpenVINO.

## Installation

🤗 Optimum Intel can be installed using `pip` as follows:

```bash
python -m pip install optimum[intel]
```

Optimum Intel is a fast-moving project, and you may want to install from source.

```bash
pip install git+https://github.com/huggingface/optimum-intel.git
```

To install the latest release of this package with the corresponding required dependencies, you can do respectively:

| Accelerator                                                                                                      | Installation                                                        |
|:-----------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|
| [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) | `python -m pip install optimum[neural-compressor]`                  |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)                                                           | `python -m pip install optimum[openvino,nncf] transformers==4.23.*` |

## Running the examples

There are a number of examples provided in the `examples` directory.

Please install the requirements for every example:

```
cd <example-folder>
pip install -r requirements.txt
```

# How to use it?

## Neural Compressor

Here is an example on how to apply dynamic quantization on a DistilBERT fine-tuned on the SQuAD1.0 dataset.
Note that quantization is currently only supported for CPUs (only CPU backends are available), so we will not be utilizing GPUs / CUDA in this example.

```python
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from evaluate import evaluator
from optimum.intel.neural_compressor import IncOptimizer, IncQuantizationConfig, IncQuantizer

model_id = "distilbert-base-cased-distilled-squad"
max_eval_samples = 100
model = AutoModelForQuestionAnswering.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
eval_dataset = load_dataset("squad", split="validation").select(range(max_eval_samples))
eval = evaluator("question-answering")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def eval_func(model):
    qa_pipeline.model = model
    metrics = eval.compute(model_or_pipeline=qa_pipeline, data=eval_dataset, metric="squad")
    return metrics["f1"]

# Load the quantization configuration detailing the quantization we wish to apply
config_path = "echarlaix/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
quantization_config = IncQuantizationConfig.from_pretrained(config_path)

# Instantiate our IncQuantizer using the desired configuration and the evaluation function used
# for the INC accuracy-driven tuning strategy
quantizer = IncQuantizer(quantization_config, eval_func=eval_func)
optimizer = IncOptimizer(model, quantizer=quantizer)

# Apply dynamic quantization
quantized_model = optimizer.fit()

# Save the resulting model and its corresponding configuration in the given directory
optimizer.save_pretrained("./quantized_model")
```

To load a quantized model hosted locally or on the 🤗 hub, you can do as follows :
```python
from optimum.intel.neural_compressor.quantization import IncQuantizedModelForSequenceClassification

loaded_model_from_hub = IncQuantizedModelForSequenceClassification.from_pretrained(
    "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
)
```

You can load many more quantized models hosted on the hub under the Intel organization [`here`](https://huggingface.co/Intel).

Check out the [`examples`](https://github.com/huggingface/optimum-intel/tree/main/examples) directory for more sophisticated usage.

## OpenVINO
Below are the examples of how to use OpenVINO and its [NNCF](https://docs.openvino.ai/latest/tmo_introduction.html) framework for model optimization, quantization, and inference.

#### OpenVINO inference example:

```diff
-from transformers import AutoModelForSequenceClassification
+from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
-model = AutoModelForSequenceClassification.from_pretrained(model_id)
+model = OVModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe_cls = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "He's a dreadful magician."
outputs = pipe_cls(text)
```

#### Post-training quantization example:
```python
from functools import partial
from optimum.intel.openvino import OVQuantizer, OVModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)    
tokenizer = AutoTokenizer.from_pretrained(model_id)
def preprocess_fn(examples, tokenizer):
    return tokenizer(
        examples["sentence"], padding=True, truncation=True, max_length=128
    )

quantizer = OVQuantizer.from_pretrained(model)
calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=100,
    dataset_split="train",
    preprocess_batch=True,
)
# The directory where the quantized model will be saved
save_dir = "nncf_results"
# Apply static quantization and save the resulting model in the OpenVINO IR format
quantizer.quantize(calibration_dataset=calibration_dataset, save_directory=save_dir)
# Load the quantized model
optimized_model = OVModelForSequenceClassification.from_pretrained(save_dir)
```

#### Quantization-aware training example:

```diff
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, default_data_collator
-from transformers import Trainer
+from optimum.intel.openvino import OVConfig, OVModelForSequenceClassification, OVTrainer

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)    
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("glue", "sst2")
dataset = dataset.map(
    lambda examples: tokenizer(examples["sentence"], padding=True, truncation=True, max_length=128), batched=True
)
metric = load_metric("accuracy")
compute_metrics = lambda p: metric.compute(
    predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
)

# The directory where the quantized model will be saved
save_dir = "nncf_results"

# Load the default quantization configuration detailing the quantization we wish to apply
+ov_config = OVConfig()

-trainer = Trainer(
+trainer = OVTrainer(
    model=model,
    args=TrainingArguments(save_dir, num_train_epochs=1.0, do_train=True, do_eval=True),
    train_dataset=dataset["train"].select(range(300)),
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
+   ov_config=ov_config,
+   feature="sequence-classification",
)
train_result = trainer.train()
metrics = trainer.evaluate()
trainer.save_model()

+optimized_model = OVModelForSequenceClassification.from_pretrained(save_dir)
```

You can find more OpenVINO examples in the corresponding [Optimum Intel documentation](https://huggingface.co/docs/optimum/main/en/intel_inference).
