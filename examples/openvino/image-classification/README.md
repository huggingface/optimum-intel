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
# Image classification

This folder contains [`run_image_classification.py`](https://github.com/huggingface/optimum/blob/main/examples/openvino/image-classification/run_image_classification.py), a script to fine-tune a 🤗 Transformers model on an image classification dataset while applying Quantization-Aware Training (QAT). QAT can be easily applied by replacing the Transformers [`Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) with the Optimum [`OVTrainer`]. Any model from our [hub](https://huggingface.co/models) can be fine-tuned and quantized, as long as the model is supported by the [`AutoModelForImageClassification`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForImageClassification) API.

### Fine-tuning ViT on the beans dataset

Here we show how to apply Quantization-Aware Training (QAT) on a fine-tuned Vision Transformer (ViT) on the beans dataset (to classify the disease type of bean leaves).

```bash
python run_image_classification.py \
    --model_name_or_path nateraw/vit-base-beans \
    --dataset_name beans \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --seed 1337 \
    --output_dir /tmp/beans_outputs/
```

On a single V100 GPU, this example takes about 1 minute and yields a quantized model with accuracy of **98.5%**.

### Joint Pruning, Quantization and Distillation (JPQD) of Swin on food101

`OVTrainer` also provides advanced optimization workflow via NNCF to structurally prune, quantize and distill. Following is an example of joint pruning, quantization and distillation on Swin-base model for food101 dataset. To enable JPQD optimization, use an alternative configuration specified with `--nncf_compression_config`. For more details on how to configure the pruning algorithm, see NNCF documentation [here](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/sparsity/movement/MovementSparsity.md).

```bash
torchrun --nproc-per-node=1 run_image_classification.py \
    --model_name_or_path microsoft/swin-base-patch4-window7-224 \
    --teacher_model_name_or_path skylord/swin-finetuned-food101 \
    --distillation_weight 0.9 \
    --ignore_mismatched_sizes \
    --dataset_name food101 \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --logging_steps 1 \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --seed 42 \
    --overwrite_output_dir \
    --output_dir /tmp/food101_outputs/ \
    --nncf_compression_config configs/swin-base-jpqd.json
```

This example results in a quantized swin-base model with ~40% sparsity in its linear layers of the transformer blocks, giving 90.7% accuracy on food101 and taking about 12.5 hours on a single V100 GPU. For launching the script on multiple GPUs specify `--nproc-per-node=<number of GPU>`. Note, that different batch size and other hyperparameters might be required to achieve the same results as on a single GPU.
