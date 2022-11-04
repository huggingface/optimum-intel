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

# text to image 

The script [`run_diffusion.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/text-to-image/run_diffusion.py)
allows us to apply different quantization approaches (such as dynamic, static) using the [Intel Neural Compressor ](https://github.com/intel/neural-compressor) library for 
text to image tasks.

The following example applies post-training static quantization on a stable-diffusion with pretrained model:[CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). In this example, we only quantized the unet model in the diffusion pipeline, and the diffusion pipeline has four models: safety_checker, text_encoder, unet, vae. 

## Prepare pretrained model
```bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
```
Now, the models are in stable-diffusion-v1-4

## Quantization
```bash
python run_diffusion.py \
    --model_name_or_path stable-diffusion-v1-4 \
    --apply_quantization \
    --quantization_approach static \
    --tolerance_criterion 100 \
    --verify_loading \
    --output_dir /tmp/diffusion_output \
    --input_text "a photo of an astronaut riding a horse on mars"
```

In order to apply dynamic or static, `quantization_approach` must be set to respectively `dynamic` or `static`.

The configuration file containing all the information related to the model quantization can be 
specified using respectively `quantization_config`. If not specified, the default
[quantization](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/text-to-image/quantization.yml),
configuration files will be used.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
