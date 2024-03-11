<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

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

The original generation task only supported the PyTorch eager and graph model. By calling the `IPEXModelForCausalLM` class, we can now apply ipex optimizations to the eager and graph model for generation tasks.


Example usage:
### Use bf16 and JIT model
```bash
python run_generation.py \
    --model_name_or_path=gpt2 \
    --bf16 \
    --jit
```
