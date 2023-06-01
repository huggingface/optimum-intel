# Stable Diffusion Quantization
This example demonstrates how to apply Quantization-aware Training (QAT) from [NNCF](https://github.com/openvinotoolkit/nncf) and Token Merging method to optimize UNet model from Stable Diffusion pipeline. The optimized model and the pipeline are exported to the OpenVINO format for inference with `OVStableDiffusionPipeline` helper. The original training code was taken from the Diffusers [repository](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) and modified to support QAT.

Knowledge distillation and EMA techniques can be used to improve the model accuracy.

This example supports model tuning on the following datasets from the HuggingFace:
* [Pokemon BLIP captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
* [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en)
* [laion2B-en-aesthetic](https://huggingface.co/datasets/laion/laion2B-en-aesthetic)
* [laion-art](https://huggingface.co/datasets/laion/laion-art)
* [laion400m](https://huggingface.co/datasets/laion/laion400m)

But it can be easily extended to other datasets.
>**Note**: laion2B* datasets are being downloaded on-fly during the fine-tuning process. No need to store them locally.

## Prerequisites
* Install Optimum-Intel for OpenVINO:
```python
pip install optimum-intel[openvino]
```
* Install example requirements:
```python
pip install -r requirements.txt
```
>**Note**: The example requires `torch~=1.13` and does not work with PyTorch 2.0.

## Running pre-optimized model
* You can also run the [notebook](../../../notebooks/openvino/stable_diffusion_optimization.ipynb) to compare FP32 pipeline with the optimized versions.
* General-purpose image generation model:
```python
from optimum.intel.openvino import OVStableDiffusionPipeline

pipe = OVStableDiffusionPipeline.from_pretrained("OpenVINO/stable-diffusion-2-1-quantized", compile=False)
pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
pipe.compile()

prompt = "sailing ship in storm by Rembrandt"
output = pipe(prompt, num_inference_steps=50, output_type="pil")
output.images[0].save("result.png")
```
* Pokemon generation:
```python
from optimum.intel.openvino import OVStableDiffusionPipeline

pipe = OVStableDiffusionPipeline.from_pretrained("OpenVINO/Stable-Diffusion-Pokemon-en-quantized", compile=False)
pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
pipe.compile()

prompt = "cartoon bird"
output = pipe(prompt, num_inference_steps=50, output_type="pil")
output.images[0].save("result.png")
```

## HW Requirements for QAT
The minimal HW setup for the run is GPU with 24GB of memory.

>**NOTE**: Potentially you can set the number of training steps to 0 and it will lead to Post-Training Quantization. CPU should be enough in this case but you may need to modify the scipt.

## Run QAT:
* QAT for pokemon generation model:
```python
python train_text_to_image_qat.py \
    --ema_device="cpu" \
    --use_kd \
    --model_id="svjack/Stable-Diffusion-Pokemon-en" \
    --center_crop \
    --random_flip \
    --gradient_checkpointing \
    --dataloader_num_workers=2 \
    --dataset_name="lambdalabs/pokemon-blip-captions" \
    --max_train_steps=4096 \
    --opt_init_steps=300 \
    --output_dir=sd-quantized-pokemon
```

* QAT + Token Merging (0.5 ratio) for pokemon generation model:
```python
python train_text_to_image_qat.py \
    --ema_device="cpu" \
    --use_kd \
    --model_id="svjack/Stable-Diffusion-Pokemon-en" \
    --center_crop \
    --random_flip \
    --gradient_checkpointing \
    --dataloader_num_workers=2 \
    --dataset_name="lambdalabs/pokemon-blip-captions" \
    --max_train_steps=8000 \
    --opt_init_steps=300 \
    --tome_ratio=0.5 \
    --output_dir=sd-quantized-pokemon
```

* QAT on a laion-aesthetic dataset:
```python
python train_text_to_image_qat.py \
    --use_kd \
    --center_crop \
    --random_flip \
    --dataset_name="laion/laion2B-en-aesthetic" \
    --max_train_steps=2048  \
    --model_id="runwayml/stable-diffusion-v1-5" \
    --max_train_samples=15000 \
    --dataloader_num_workers=4 \
    --opt_init_steps=500 \
    --gradient_checkpointing \
    --tune_quantizers_only \
    --output_dir=sd-1-5-quantied-laion
```

## References
* [Optimizing Stable Diffusion for Intel CPUs with NNCF and 🤗 Optimum](https://huggingface.co/blog/train-optimize-sd-intel)