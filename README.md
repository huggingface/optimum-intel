<p align="center">
    <img src="readme_logo.png" />
</p>

# Optimum Intel

🤗 Optimum Intel is the interface between the 🤗 Transformers library and the different tools and libraries provided by Intel to accelerate end-to-end pipelines on Intel architectures.

Intel [Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) is an open-source library enabling the usage of the most popular compression techniques such as quantization, pruning and knowledge distillation. It supports automatic accuracy-driven tuning strategies in order for users to easily generate quantized model. The users can easily apply static, dynamic and aware-training quantization approaches while giving an expected accuracy criteria. It also supports different weight pruning techniques enabling the creation of pruned model giving a predefined sparsity target.

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

To apply our pruning methodology, we need to create an instance of IncTrainer, which is very similar to the 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).
We will fine-tune our model for 3 epochs while applying pruning.

```diff
-from transformers import Trainer
+from optimum.intel.neural_compressor import IncTrainer

# Initialize our IncTrainer
-trainer = Trainer(
+trainer = IncTrainer(
    model=model,
    args=TrainingArguments(output_dir, num_train_epochs=3.0),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)
```

To apply our quantization and pruning methodologies, we first need to create the corresponding configuration describing how we want those methodologies to be applied :

```python
from optimum.intel.neural_compressor import IncOptimizer, IncPruner, IncQuantizer
from optimum.intel.neural_compressor.configuration import IncPruningConfig, IncQuantizationConfig

# The targeted sparsity is set to 10%
target_sparsity = 0.1
config_path = "echarlaix/distilbert-sst2-inc-dynamic-quantization-magnitude-pruning-0.1"
# Load the quantization configuration detailing the quantization we wish to apply
quantization_config = IncQuantizationConfig.from_pretrained(config_path, config_file_name="quantization.yml")
# Load the pruning configuration detailing the pruning we wish to apply
pruning_config = IncPruningConfig.from_pretrained(config_path, config_file_name="prune.yml")

# Instantiate our IncQuantizer using the desired configuration
quantizer = IncQuantizer(quantization_config, eval_func=eval_func)
# Instantiate our IncPruner using the desired configuration
pruner = IncPruner(pruning_config, eval_func=eval_func, train_func=train_func)
optimizer = IncOptimizer(model, quantizer=quantizer, pruner=pruner)
# Apply pruning and quantization 
optimized_model = optimizer.fit()

# Save the resulting model and its corresponding configuration in the given directory
optimizer.save_pretrained(output_dir)

```

To load a quantized model hosted locally or on the 🤗 hub, you can do as follows :
```python
from optimum.intel.neural_compressor.quantization import IncQuantizedModelForSequenceClassification

loaded_model_from_hub = IncQuantizedModelForSequenceClassification.from_pretrained(
    "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
)
```

You can load many more quantized models hosted on the hub under the Intel organization [`here`](https://huggingface.co/Intel).

Check out the [`examples`](https://github.com/huggingface/optimum-intel/tree/main/examples) directory for more sophisticated usage.

