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




