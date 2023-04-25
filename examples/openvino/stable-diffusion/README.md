# Stable Diffusion Quantization
This example demonstrates Quantization-aware Training (QAT) of Stable Diffusion using [NNCF](https://github.com/openvinotoolkit/nncf). Quantization is applyied to UNet model which is the most time-consuming element of the whole pipeline. The quantized model and the pipeline is exported to the OpenVINO format for inference with `OVStableDiffusionPipeline` helper. The original training code was taken from the Diffusers [repository](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) and modified to support QAT.

Knowledge distillation and EMA techniques can be used to improve the model accuracy.

This example supports model tuning on two datasets from the HuggingFace:
* [Pokemon BLIP captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
* [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en)
* [laion2B-en-aesthetic](https://huggingface.co/datasets/laion/laion2B-en-aesthetic)

But it can be easily extended to other datasets.
>**Note**: laion2B-en is being downloaded on-fly durint the fine-tuning process. No need to store it locally.

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
* You can also run `pokemon_generation_demo.ipynb` notebook from the folder to compare FP32 pipeline with the optimized.

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