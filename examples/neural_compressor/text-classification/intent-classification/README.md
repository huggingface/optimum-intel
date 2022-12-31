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

# Intent classification 

## Clinc task

The script [`run_clinc.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/text-classification/intent-classification/run_clinc.py)
allows us to apply static quantization approach as well as distillation 
using the Intel [Neural Compressor](https://github.com/intel/neural-compressor) library for 
sequence classification task.

The following example applies post-training static quantization on a distilled MiniLM [model](https://huggingface.co/SetFit/MiniLM_L3_clinc_oos_plus_distilled) fine-tuned on the [CLINC150](https://huggingface.co/datasets/clinc_oos) dataset.

```bash
python run_clinc.py \
    --model_name_or_path SetFit/MiniLM_L3_clinc_oos_plus_distilled \
    --dataset_name clinc_oos \
    --apply_quantization \
    --quantization_approach static \
    --num_calibration_samples 50 \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/clinc_output
```

The following example applies dynamic quantization on a distilled MiniLM [model](https://huggingface.co/SetFit/MiniLM_L3_clinc_oos_plus_distilled) fine-tuned on the [CLINC150](https://huggingface.co/datasets/clinc_oos) dataset.

```bash
python run_clinc.py \
    --model_name_or_path SetFit/MiniLM_L3_clinc_oos_plus_distilled \
    --dataset_name clinc_oos \
    --apply_quantization \
    --quantization_approach dynamic \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/clinc_output
```


The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
