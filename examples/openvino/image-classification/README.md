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

This folder contains [`run_image_classification.py`](https://github.com/huggingface/optimum/blob/main/examples/openvino/image-classification/run_image_classification.py), a script to fine-tune a 🤗 Transformers model on an image classification dataset while applying quantization aware training (QAT). QAT can be easily applied by replacing the Transformers [`Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) with the Optimum [`OVTrainer`]. Any model from our [hub](https://huggingface.co/models) can be fine-tuned and quantized, as long as the model is supported by the [`AutoModelForImageClassification`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForImageClassification) API.

### Fine-tuning ViT on the beans dataset

Here we show how to apply quantization aware training (QAT) on a fine-tuned Vision Transformer (ViT) on the beans dataset (to classify the disease type of bean leaves).

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