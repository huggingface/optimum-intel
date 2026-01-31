<p align="center">
    <img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/intel/logo/hf_intel_logo.png" />
</p>

# Optimum Intel

🤗 [Optimum Intel](https://huggingface.co/docs/optimum-intel/en/index) is the interface between the 🤗 Transformers and Diffusers libraries and the different tools and libraries provided by [OpenVINO](https://docs.openvino.ai) to accelerate end-to-end pipelines on Intel architectures.

[OpenVINO](https://docs.openvino.ai) is an open-source toolkit that enables high performance inference capabilities for Intel CPUs, GPUs, and special DL inference accelerators ([see](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html) the full list of supported devices). It is supplied with a set of tools to optimize your models with compression techniques such as quantization, pruning and knowledge distillation. Optimum Intel provides a simple interface to optimize your Transformers and Diffusers models, convert them to the OpenVINO Intermediate Representation (IR) format and run inference using OpenVINO Runtime.


## Installation

To install the latest release of 🤗 Optimum Intel with the corresponding required dependencies, you can use `pip` as follows:

```bash
python -m pip install -U "optimum-intel[openvino]"
```

Optimum Intel is a fast-moving project with regular additions of new model support, so you may want to install from source with the following command:

```bash
python -m pip install "optimum-intel"@git+https://github.com/huggingface/optimum-intel.git
```

**Deprecation Notice:** The `extras` for `openvino` (e.g., `pip install optimum-intel[openvino,nncf]`), `nncf`, `neural-compressor`, `ipex` are **deprecated** and will be **removed in a future release**.  


## Export:

To export your model to [OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html) format, use the optimum-cli tool.
Below is an example of exporting [TinyLlama/TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1) model:

```sh
optimum-cli export openvino --model TinyLlama/TinyLlama_v1.1 ov_TinyLlama_v1_1
```

Additional information on exporting models is available in the [documentation](https://huggingface.co/docs/optimum-intel/en/openvino/export).

## Inference:

To load an exported model and run inference using Optimum Intel, use the corresponding `OVModelForXxx` class instead of `AutoModelForXxx`:

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline

model_id = "ov_TinyLlama_v1_1"
model = OVModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
results = pipe("Hey, how are you doing today?", max_new_tokens=100)
```

For more details on Optimum Intel inference, refer to the [documentation](https://huggingface.co/docs/optimum-intel/en/openvino/inference).

**Note:** Alternatively, an exported model can also be inferred using [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) framework,
that provides optimized execution methods for highly performant Generative AI.

## Quantization:

Post-training static quantization can also be applied. Here is an example on how to apply static quantization on a Whisper model using the [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) dataset for the calibration step.

```python
from optimum.intel import OVModelForSpeechSeq2Seq, OVQuantizationConfig

model_id = "openai/whisper-tiny"
q_config = OVQuantizationConfig(dtype="int8", dataset="librispeech", num_samples=50)
q_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, quantization_config=q_config)

# The directory where the quantized model will be saved
save_dir = "nncf_results"
q_model.save_pretrained(save_dir)
```

You can find more information in the [documentation](https://huggingface.co/docs/optimum-intel/en/openvino/optimization).

## Running the examples

Check out the [`notebooks`](https://github.com/huggingface/optimum-intel/tree/main/notebooks) directory to see how 🤗 Optimum Intel can be used to optimize models and accelerate inference.

Do not forget to install requirements for every example:

```sh
cd <example-folder>
pip install -r requirements.txt
```
