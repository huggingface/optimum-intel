<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🤗 Optimum Intel Notebooks
🤗[Optimum Intel](https://github.com/huggingface/optimum-intel)  is the interface between the 🤗 [Transformers](https://github.com/huggingface/transformers) library and the different tools and libraries provided by Intel to accelerate end-to-end pipelines on Intel architectures.

You can find here a list of the notebooks, leveraging the usage of the Optimum Intel library for different tools and libraries provided by Intel.

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [How to quantize a model with Intel Neural Compressor for text classification](https://github.com/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb)| Show how to apply static, dynamic and aware training quantization on a model using Intel [Neural Compressor](https://github.com/intel/neural-compressor) for any GLUE task. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/examples/text_classification_quantization_inc.ipynb)|
| [How to quantize a question answering model with OpenVINO NNCF](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb) | Show how to apply post-training quantization on a question answering model using [NNCF](https://github.com/openvinotoolkit/nncf) and to accelerate inference with OpenVINO| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb)|
