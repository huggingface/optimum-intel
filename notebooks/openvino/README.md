# 🤗 Optimum OpenVINO Notebooks

This directory contains notebooks for the OpenVINO integration in 🤗 Optimum. To
install the requirements for running all notebooks, do `pip install -r
requirements.txt`. If you do not want to install the requirements to run all the
notebooks, you can also install the requirements for a specific notebook. They
are listed at the top of each notebook file.

The notebooks have been tested with Python 3.8 and 3.10 on Ubuntu Linux.

| Notebook     |      Description      |   |   |
|:----------|:-------------|:-------------|------:|
| [How to run inference with the OpenVINO](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb) | Explains how to export your model to OpenVINO and to run inference with OpenVINO Runtime on various tasks| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/optimum-intel/blob/main/notebooks/openvino/optimum_openvino_inference.ipynb)|
| [How to quantize a question answering model with OpenVINO NNCF](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb) | Show how to apply post-training quantization on a question answering model using [NNCF](https://github.com/openvinotoolkit/nncf) and to accelerate inference with OpenVINO| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb)| [![Open in AWS Studio](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/optimum-intel/blob/main/notebooks/openvino/question_answering_quantization.ipynb)|

