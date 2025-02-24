<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

## Language generation

Based on the script [`run_generation.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py).

This example also allows us to apply different quantization approaches (such as dynamic, static, The example applies post-training static quantization on a gptj model).

Example usage:
### apply_quantization with post-training static
```bash
python run_generation.py \
    --model_type=gptj \
    --model_name_or_path=EleutherAI/gpt-j-6b \
    --apply_quantization \
    --quantization_approach static\
    --smooth_quant \
    --smooth_quant_alpha 0.7
```

### Use JIT model and apply_quantization with post-training static
```bash
python run_generation.py \
    --model_type=gptj \
    --model_name_or_path=EleutherAI/gpt-j-6b \
    --apply_quantization \
    --quantization_approach static\
    --smooth_quant \
    --smooth_quant_alpha 0.7 \
    --jit
```
