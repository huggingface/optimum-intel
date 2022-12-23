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

The following examples showcase how to fine-tune `Wav2Vec2` for audio classification using PyTorch.

Speech recognition models that have been pretrained in unsupervised fashion on audio data alone,
*e.g.* [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html),
[HuBERT](https://huggingface.co/transformers/main/model_doc/hubert.html),
[XLSR-Wav2Vec2](https://huggingface.co/transformers/main/model_doc/xlsr_wav2vec2.html), have shown to require only
very little annotated data to yield good performance on speech classification datasets.

## Single-GPU

The following command shows how to fine-tune [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the 🗣️ [Keyword Spotting subset](https://huggingface.co/datasets/superb#ks) of the SUPERB dataset.

```bash
python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir wav2vec2-base-ft-keyword-spotting \
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
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 0 \
    --push_to_hub
```

On a single V100 GPU (16GB), this script should run in ~14 minutes and yield accuracy of **98.26%**.

👀 See the results here: [anton-l/wav2vec2-base-ft-keyword-spotting](https://huggingface.co/anton-l/wav2vec2-base-ft-keyword-spotting)

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.

### Joint Pruning, Quantization and Distillation (JPQD) for Wav2Vec2 on Keyword Spotting

```bash
python run_audio_classification.py \
    --model_name_or_path anton-l/wav2vec2-base-ft-keyword-spotting \
    --teacher_model_or_path anton-l/wav2vec2-base-ft-keyword-spotting \
    --nncf_compression_config configs/wav2vec2-base-jpqd.json \
    --distillation_weight 0.9 \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir /tmp/wav2vec2_ks \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_eval \
    --do_train  \
    --fp16 \
    --optim adamw_torch \
    --learning_rate 2e-4 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 64 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 42 
```