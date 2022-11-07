<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# Translation

The script [`run_translation.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/translation/run_translation.py)
allows us to apply different quantization approaches (such as dynamic, static and aware-training quantization) as well as pruning 
using the [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor) library for translation tasks.

The following example applies post-training static quantization on a T5 model.

```bash
python run_translation.py \ 
    --model_name_or_path t5-small \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_prefix "translate English to Romanian: " \
    --apply_quantization \
    --quantization_approach static \
    --tolerance_criterion 0.7 \
    --do_eval \
    --verify_loading \
    --predict_with_generate \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir /tmp/test_translation
```

The following example fine-tunes a T5 model on the wmt16 dataset while applying magnitude pruning and then applies dynamic quantization as a second step.

```bash


python run_translation.py \ 
    --model_name_or_path t5-small \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_prefix "translate English to Romanian: " \
    --apply_quantization \
    --quantization_approach dynamic \
    --apply_pruning \
    --target_sparsity 0.1 \
    --tolerance_criterion 0.7 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --verify_loading \
    --predict_with_generate \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir /tmp/test_translation
```

In order to apply dynamic, static or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static` or `aware_training`.

The configuration file containing all the information related to the model quantization and pruning objectives can be 
specified using respectively `quantization_config` and `pruning_config`. If not specified, the default
[quantization](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/config/quantization.yml),
and [pruning](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/config/prune.yml) 
configuration files will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
