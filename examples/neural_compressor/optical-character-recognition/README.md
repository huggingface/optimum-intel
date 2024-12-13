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

The script [`run_ocr_post_training.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/optical-character-recognition/run_ocr_post_training.py)
allows us to apply different quantization approaches (such as dynamic and static quantization) 
using the [Intel Neural Compressor ](https://github.com/intel/neural-compressor) library for optical character recognition tasks and [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) datasets.

Note that this case is from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb, And it only support TrOCR models.

## Prepare datasets
```bash
wget https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz
tar xvf IAM.tar.gz
```

## Run Command
The following example applies post-training static quantization on TrOCR small fine-tuned on the IAM dataset. This tuning process will take a long time. If you want to get a quantized model quickly, but do not care about the accuracy, you can set --tolerance_criterion larger or --max_eval_samples small. The default value of tolerance_criterion is 0.01, and max_eval_samples is None(None means the entire datasets).
```bash
python run_ocr_post_training.py \
    --model_name_or_path microsoft/trocr-small-handwritten \
    --datasets_dir IAM \
    --apply_quantization \
    --quantization_approach dynamic \
    --verify_loading \
    --output_dir /tmp/trocr_output
```

In order to apply dynamic and static quantization, `quantization_approach` must be set to respectively `dynamic` or `static`.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
