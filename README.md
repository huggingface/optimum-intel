<p align="center">
    <img src="readme_logo.png" />
</p>

# Optimum Intel

🤗 Optimum Intel is the interface between the 🤗 Transformers library and the different tools and libraries provided by Intel to accelerate end-to-end pipelines on Intel architectures.

Intel [Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) is an open-source library enabling the usage of the most popular compression techniques such as quantization, pruning and knowledge distillation. It supports automatic accuracy-driven tuning strategies in order for users to easily generate quantized model. The users can easily apply static, dynamic and aware-training quantization approaches while giving an expected accuracy criteria. It also supports different weight pruning techniques enabling the creation of pruned model giving a predefined sparsity target.

[OpenVINO](https://docs.openvino.ai/latest/index.html) is an open-source toolkit that enables high performance inference capabilities for Intel CPUs, GPUs, and special DL inference accelerators. It is supplied with a set of tools to optimize and quantize models. Optimum Intel provides a simple interface to optimize Transformer models, convert them to OpenVINO Intermediate Representation format and to run inference using OpenVINO.

## Installation

To install the latest release of 🤗 Optimum Intel with the corresponding required dependencies, you can use `pip` as follows:

| Accelerator                                                                                                      | Installation                                                         |
|:-----------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|
| [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) | `python -m pip install "optimum[neural-compressor]"`                 |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)                                                           | `python -m pip install "optimum[openvino,nncf]"`                     |

We recommend creating a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) and upgrading
pip with `python -m pip install --upgrade pip`.

Optimum Intel is a fast-moving project, and you may want to install from source with the following command:

```bash
python -m pip install git+https://github.com/huggingface/optimum-intel.git
```

or to install from source including dependencies:

```bash
python -m pip install "optimum-intel[extras]"@git+https://github.com/huggingface/optimum-intel.git
```

where `extras` can be one or more of `neural-compressor`, `openvino`, `nncf`.

# Quick tour

## Neural Compressor

#### Dynamic quantization:

Here is an example on how to apply dynamic quantization on a DistilBERT fine-tuned on the SQuAD1.0 dataset.
Note that quantization is currently only supported for CPUs (only CPU backends are available), so we will not be utilizing GPUs / CUDA in this example.

```python
from transformers import AutoModelForQuestionAnswering
from neural_compressor.config import PostTrainingQuantConfig
from optimum.intel.neural_compressor import INCQuantizer

model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# The directory where the quantized model will be saved
save_dir = "quantized_model"
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = PostTrainingQuantConfig(approach="dynamic")
quantizer = INCQuantizer.from_pretrained(model)
# Apply dynamic quantization and save the resulting model
quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)
```

To load a quantized model hosted locally or on the 🤗 hub, you can do as follows :
```python
from optimum.intel.neural_compressor import INCModelForSequenceClassification

# Load the PyTorch model hosted on the hub
loaded_model_from_hub = INCModelForSequenceClassification.from_pretrained(
    "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
)
```

You can load many more quantized models hosted on the hub under the Intel organization [`here`](https://huggingface.co/Intel).

## OpenVINO

Below are the examples of how to use OpenVINO and its [NNCF](https://docs.openvino.ai/latest/tmo_introduction.html) framework to accelerate inference.

#### Inference:

To load a model and run inference with OpenVINO Runtime, you can just replace your `AutoModelForXxx` class with the corresponding `OVModelForXxx` class.
If you want to load a PyTorch checkpoint, set `from_transformers=True` to convert your model to the OpenVINO IR.

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

#### Post-training static quantization:

Post-training static quantization introduces an additional calibration step where data is fed through the network in order to compute the activations quantization parameters. Here is an example on how to apply static quantization on a fine-tuned DistilBERT.

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

#### Quantization-aware training:

Quantization aware training (QAT) is applied in order to simulate the effects of quantization during training, to alleviate its effects on the model’s accuracy. Here is an example on how to fine-tune a DistilBERT model on the sst-2 task while applying quantization aware training (QAT).

```diff
import evaluate
import numpy as np
from datasets import load_dataset
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
metric = evaluate.load("glue", "sst2")
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
+   task="sequence-classification",
)
train_result = trainer.train()
metrics = trainer.evaluate()
trainer.save_model()

+optimized_model = OVModelForSequenceClassification.from_pretrained(save_dir)
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/intel/index).


## Running the examples

Check out the [`examples`](https://github.com/huggingface/optimum-intel/tree/main/examples) directory to see how 🤗 Optimum Intel can be used to optimize models and accelerate inference.

Do not forget to install requirements for every example:

```
cd <example-folder>
pip install -r requirements.txt
```
