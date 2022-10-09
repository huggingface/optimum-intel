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

# Optical Character Recognition

The script [`run_trocr.py`](https://github.com/huggingface/optimum/blob/main/examples/optical-character-recognition/run_trocr.py)
allows us to apply different quantization approaches (such as dynamic, static and aware-training quantization) as well as pruning 
using the [Intel Neural Compressor ](https://github.com/intel/neural-compressor) library for optical character recognition tasks and [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) datasets.

Note that this case is from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb.

## Prepare datasets
```bash
wget https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz
tar xvf IAM.tar.gz
```

## Run Command
The following example applies post-training static quantization on TrOCR small fine-tuned on the IAM dataset.
```bash
python run_trocr.py \
    --model_name_or_path microsoft/trocr-small-handwritten \
    --datasets_dir IAM \
    --tune \
    --quantization_approach static \
    --verify_loading \
    --output_dir /tmp/trocr_output
```

In order to apply dynamic, static or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static` or `aware_training`.

The configuration file containing all the information related to the model quantization objectives can be specified using respectively `quantization_config`. If not specified, the default
[quantization](https://github.com/huggingface/optimum/blob/main/examples/config/quantization.yml) configuration files will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
