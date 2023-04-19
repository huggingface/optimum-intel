# Stable Diffusion Quantization
This repository demonstrates Quantization-aware Training (QAT) of Stable Diffusion Unet model wich is the most time-consuming element of the whole pipeline. The quantized model is exported to the OpenVINO IR.

The expected speedup from quantization is ~1.7x (for CPUs w/ Intel DL Boost) and can very depeding on the HW.

Knowledge distillation and EMA techniques can be used to improve the model accuracy.

## Prerequisites
```python
pip install -r requirements.txt
```

Install NNCF from source:
```python
pip install git+https://github.com/openvinotoolkit/nncf.git
```

## Running pre-optimized model
```python
from optimum.intel.openvino import OVStableDiffusionPipeline

pipe = OVStableDiffusionPipeline.from_pretrained("OpenVINO/stable-diffusion-1-5-quantized", compile=False)
pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
pipe.compile()

prompt = "Super cute fluffy cat warrior in armor, photorealistic, 4K, ultra detailed, vray rendering, unreal engine"
output = pipe(prompt, num_inference_steps=50, output_type="pil")
output.images[0].save("result.png")
```

## HW Requirements for QAT
The minimal HW setup for the run is GPU with 24GB of memory.

>**NOTE**: Potentially you can set the number of training steps to 0 and it will lead to Post-Training Quantization. CPU should be enough in this case but you may need to modify the scipt.

## Run PTQ:
```python
python quantize.py --use_kd --center_crop --random_flip --dataset_name="lambdalabs/pokemon-blip-captions" --max_train_steps=0 --model_id="runwayml/stable-diffusion-v1-5"
```

On a part of "laion/laion2B-en" dateset:
```python
python quantize.py --use_kd --center_crop --random_flip --dataset_name="laion/laion2B-en" --model_id="stabilityai/stable-diffusion-2-1" --max_train_samples=800 --opt_init_steps=800 --dataloader_num_workers=6
```

>**NOTE**: You may need to paly with seed or do one more try to get good results. The results can be better if to use the same dataset that was used to train the original model.

## Run QAT:

* The best results are achieved with Knowledge Distillation and EMA techniques when tuning on a laion dataset:
```python
CUDA_VISIBLE_DEVICES=2 python quantize.py --ema_device="cpu" --use_kd --center_crop --random_flip --dataset_name="laion/laion2B-en" --max_train_steps=4096  --model_id="runwayml/stable-diffusion-v1-5" --max_train_samples=10000 --dataloader_num_workers=8 --gradient_checkpointing --output_dir=sd-1-5-quantied-laion
```

* Tune model parameters on a target dataset with QAT:
```python
python quantize.py --use_kd --ema_device="cpu" --model_id="runwayml/stable-diffusion-v1-5" --center_crop --random_flip --gradient_checkpointing --dataloader_num_workers=8 --dataset_name="lambdalabs/pokemon-blip-captions" --max_train_steps=15000
```

`--ema_device="cpu"` and `--gradient_checkpointing` are used to save GPU mememory.

* Tune only quantization parameters for a short time. You can use smaller training steps and any relevant dataset:
```python
python quantize.py --use_kd --model_id="runwayml/stable-diffusion-v1-5" --center_crop --random_flip --gradient_checkpointing --dataloader_num_workers=8 --dataset_name="laion/laion2B-en" --tune_quantizers_only --max_train_steps=256 --max_train_samples=10000 --opt_init_steps=800 --opt_init_type="min_max"
```


