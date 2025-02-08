<p align="center">
    <img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/logo/hf_intel_logo.png" />
</p>

# Optimum Intel

🤗 Optimum Intel is the interface between the 🤗 Transformers and Diffusers libraries and the different tools and libraries provided by Intel to accelerate end-to-end pipelines on Intel architectures.

[Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction) is an open-source library which provides optimizations like faster attention and operators fusion.

Intel [Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) is an open-source library enabling the usage of the most popular compression techniques such as quantization, pruning and knowledge distillation. It supports automatic accuracy-driven tuning strategies in order for users to easily generate quantized model. The users can easily apply static, dynamic and aware-training quantization approaches while giving an expected accuracy criteria. It also supports different weight pruning techniques enabling the creation of pruned model giving a predefined sparsity target.

[OpenVINO](https://docs.openvino.ai) is an open-source toolkit that enables high performance inference capabilities for Intel CPUs, GPUs, and special DL inference accelerators ([see](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html) the full list of supported devices). It is supplied with a set of tools to optimize your models with compression techniques such as quantization, pruning and knowledge distillation. Optimum Intel provides a simple interface to optimize your Transformers and Diffusers models, convert them to the OpenVINO Intermediate Representation (IR) format and run inference using OpenVINO Runtime.


## Installation

To install the latest release of 🤗 Optimum Intel with the corresponding required dependencies, you can use `pip` as follows:

| Accelerator                                                                                                      | Installation                                                         |
|:-----------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------|
| [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) | `pip install --upgrade --upgrade-strategy eager "optimum[neural-compressor]"`  |
| [OpenVINO](https://docs.openvino.ai)                                                                             | `pip install --upgrade --upgrade-strategy eager "optimum[openvino]"`           |
| [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/#introduction)                 | `pip install --upgrade --upgrade-strategy eager "optimum[ipex]"`               |

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

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

where `extras` can be one or more of `ipex`, `neural-compressor`, `openvino`, `nncf`.

# Quick tour

## Neural Compressor

Dynamic quantization can be used through the Optimum command-line interface:

```bash
optimum-cli inc quantize --model distilbert-base-cased-distilled-squad --output ./quantized_distilbert
```
Note that quantization is currently only supported for CPUs (only CPU backends are available), so we will not be utilizing GPUs / CUDA in this example.

To load a quantized model hosted locally or on the 🤗 hub, you can do as follows :
```python
from optimum.intel import INCModelForSequenceClassification

model_id = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model = INCModelForSequenceClassification.from_pretrained(model_id)
```

You can load many more quantized models hosted on the hub under the Intel organization [`here`](https://huggingface.co/Intel).

For more details on the supported compression techniques, please refer to the [documentation](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc).


## OpenVINO

Below are examples of how to use OpenVINO and its [NNCF](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/compressing-models-during-training.html) framework to accelerate inference.

#### Export:

It is also possible to export your model to the [OpenVINO IR](https://docs.openvino.ai/2024/documentation/openvino-ir-format.html) format with the CLI :

```plain
optimum-cli export openvino --model gpt2 ov_model
```

You can also apply 8-bit weight-only quantization when exporting your model : the model linear, embedding and convolution weights will be quantized to INT8, the activations will be kept in floating point precision.

```plain
optimum-cli export openvino --model gpt2 --weight-format int8 ov_model
```

Quantization in hybrid mode can be applied to Stable Diffusion pipeline during model export. This involves applying hybrid post-training quantization to the UNet model and weight-only quantization for the rest of the pipeline components. In the hybrid mode, weights in MatMul and Embedding layers are quantized, as well as activations of other layers.

```plain
optimum-cli export openvino --model stabilityai/stable-diffusion-2-1 --dataset conceptual_captions --weight-format int8 ov_model
```

To apply quantization on both weights and activations, you can find more information in the [documentation](https://huggingface.co/docs/optimum/main/en/intel/optimization_ov).

#### Inference:

To load a model and run inference with OpenVINO Runtime, you can just replace your `AutoModelForXxx` class with the corresponding `OVModelForXxx` class.


```diff
- from transformers import AutoModelForSeq2SeqLM
+ from optimum.intel import OVModelForSeq2SeqLM
  from transformers import AutoTokenizer, pipeline

  model_id = "echarlaix/t5-small-openvino"
- model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
+ model = OVModelForSeq2SeqLM.from_pretrained(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
  results = pipe("He never went out without a book under his arm, and he often came back with two.")

  [{'translation_text': "Il n'est jamais sorti sans un livre sous son bras, et il est souvent revenu avec deux."}]
```

If you want to load a PyTorch checkpoint, set `export=True` to convert your model to the OpenVINO IR.

```python
from optimum.intel import OVModelForCausalLM

model = OVModelForCausalLM.from_pretrained("gpt2", export=True)
model.save_pretrained("./ov_model")
```


#### Post-training static quantization:

Post-training static quantization introduces an additional calibration step where data is fed through the network in order to compute the activations quantization parameters. Here is an example on how to apply static quantization on a fine-tuned DistilBERT.

```python
from functools import partial
from optimum.intel import OVQuantizer, OVModelForSequenceClassification, OVConfig, OVQuantizationConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
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
ov_config = OVConfig(quantization_config=OVQuantizationConfig())
quantizer.quantize(ov_config=ov_config, calibration_dataset=calibration_dataset, save_directory=save_dir)
# Load the quantized model
optimized_model = OVModelForSequenceClassification.from_pretrained(save_dir)
```


## IPEX
To load your IPEX model, you can just replace your `AutoModelForXxx` class with the corresponding `IPEXModelForXxx` class. It will load a PyTorch checkpoint, and apply IPEX operators optimization (replaced with customized IPEX operators).
```diff
  from transformers import AutoTokenizer, pipeline
- from transformers import AutoModelForCausalLM
+ from optimum.intel import IPEXModelForCausalLM


  model_id = "gpt2"
- model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
+ model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
  results = pipe("He's a dreadful magician and")
```

For more details, please refer to the [documentation](https://intel.github.io/intel-extension-for-pytorch/#introduction).


## Running the examples

Check out the [`examples`](https://github.com/huggingface/optimum-intel/tree/main/examples) and [`notebooks`](https://github.com/huggingface/optimum-intel/tree/main/notebooks) directory to see how 🤗 Optimum Intel can be used to optimize models and accelerate inference.

Do not forget to install requirements for every example:

```
cd <example-folder>
pip install -r requirements.txt
```


## Gaudi

To train your model on [Intel Gaudi AI Accelerators (HPU)](https://docs.habana.ai/en/latest/index.html), check out [Optimum Habana](https://github.com/huggingface/optimum-habana) which provides a set of tools enabling easy model loading, training and inference on single- and multi-HPU settings for different downstream tasks. After training your model, feel free to submit it to the Intel [leaderboard](https://huggingface.co/spaces/Intel/powered_by_intel_llm_leaderboard) which is designed to evaluate, score, and rank open-source LLMs that have been pre-trained or fine-tuned on Intel Hardwares. Models submitted to the leaderboard will be evaluated on the Intel Developer Cloud. The evaluation platform consists of Gaudi Accelerators and Xeon CPUs running benchmarks from the Eleuther AI Language Model Evaluation Harness.
