<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Audio classification examples

This folder contains [`run_audio_classification.py`](https://github.com/huggingface/optimum/blob/main/examples/openvino/audio-classification/run_audio_classification.py), a script to fine-tune a 🤗 Transformers model on the 🗣️ [Keyword Spotting subset](https://huggingface.co/datasets/superb#ks) of the SUPERB dataset while applying Quantization-Aware Training (QAT). QAT can be easily applied by replacing the Transformers [`Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) with the Optimum [`OVTrainer`]. Any model from our [hub](https://huggingface.co/models) can be fine-tuned and quantized, as long as the model is supported by the [`AutoModelForAudioClassification`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForAudioClassification) API.

### Fine-tuning Wav2Vec2 on Keyword Spotting with QAT

The following command shows how to fine-tune [Wav2Vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the 🗣️ [Keyword Spotting subset](https://huggingface.co/datasets/superb#ks) of the SUPERB dataset with Quantization-Aware Training (QAT). The `OVTrainer` uses a default quantization configuration which should work in many cases, but we can also customize the algorithm details. Here, we quantize the Wav2Vec2-base model with a custom configuration file specified by `--nncf_compression_config`. For more details on the quantization configuration, see NNCF documentation [here](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md).

```bash
python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --nncf_compression_config configs/wav2vec2-base-qat.json \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir /tmp/qat-wav2vec2-base-ft-keyword-spotting \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 64 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 42
```

On a single V100 GPU, this script should run in ~45 minutes and yield a quantized model with accuracy of **97.5%**.

### Joint Pruning, Quantization and Distillation (JPQD) of Wav2Vec2 on Keyword Spotting

`OVTrainer` also provides advanced optimization workflow via NNCF to structurally prune, quantize and distill. Following is an example of joint pruning, quantization and distillation on Wav2Vec2-base model for keyword spotting task. To enable JPQD optimization, use an alternative configuration specified with `--nncf_compression_config`. For more details on how to configure the pruning algorithm, see NNCF documentation [here](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/sparsity/movement/MovementSparsity.md).

```bash
torchrun --nproc-per-node=1 run_audio_classification.py \
    --model_name_or_path anton-l/wav2vec2-base-ft-keyword-spotting \
    --teacher_model_name_or_path anton-l/wav2vec2-base-ft-keyword-spotting \
    --nncf_compression_config configs/wav2vec2-base-jpqd.json \
    --freeze_feature_encoder False \
    --distillation_weight 0.9 \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir /tmp/jpqd-wav2vec2-base-ft-keyword-spotting \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_eval \
    --do_train  \
    --fp16 \
    --optim adamw_torch \
    --learning_rate 7e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.5 \
    --num_train_epochs 12 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 64 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --seed 42
```

This script should take about 3 hours on a single V100 GPU and produce a quantized Wav2Vec2-base model with ~60% structured sparsity in its linear layers. The model accuracy should converge to about 97.5%. For launching the script on multiple GPUs specify `--nproc-per-node=<number of GPU>`. Note, that different batch size and other hyperparameters might be required to achieve the same results as on a single GPU.
