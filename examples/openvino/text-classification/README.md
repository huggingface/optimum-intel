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

# Text classification

This folder contains [`run_glue.py`](https://github.com/huggingface/optimum/blob/main/examples/openvino/text-classification/run_glue.py), a script to fine-tune a 🤗 Transformers model on the [General Language Understanding Evaluation](https://gluebenchmark.com/) (GLUE) benchmark while applying quantization aware training (QAT). QAT can be easily applied by replacing the Transformers [`Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) with the Optimum [`OVTrainer`]. Any model from our [hub](https://huggingface.co/models) can be fine-tuned and quantized, as long as the model is supported by the [`AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification) API.

### Fine-tuning BERT on GLUE with QAT

Here is the example to apply Quantization Aware Training (QAT) on BERT-base model for Stanford Sentiment Treebank-2 (SST-2) task in GLUE benchmark.

```bash
TASK_NAME=sst2
python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --output_dir /tmp/qat-bert-base-ft-$TASK_NAME \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --optim adamw_torch \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_strategy epoch \
    --fp16 \
    --seed 42
```

On a single V100 GPU, this script should run in ~40 minutes and yield accuracy of **92.9%**.

### Joint Pruning, Quantization and Distillation (JPQD) of BERT on GLUE

`OVTrainer` also provides advanced optimization workflow via NNCF to structurally prune, quantize and distillation. Following is an example to optimize a sparse-quantized BERT-base model for SST2, distilling from a BERT-large teacher. Do take note of additional NNCF config `--nncf_compression_config`.
More on how to configure movement sparsity, see NNCF documentation [here](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/sparsity/movement/MovementSparsity.md).

To run the JPQD example, please install optimum-intel from source. This command will install or upgrade optimum-intel and all necessary dependencies:

```python -m pip install --upgrade "git+https://github.com/huggingface/optimum-intel.git#egg=optimum-intel[openvino, nncf]"
```

```bash
TASK_NAME=sst2
torchrun --nproc-per-node=1 run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --teacher_model_name_or_path yoshitomo-matsubara/bert-large-uncased-sst2 \
    --nncf_compression_config ./configs/bert-base-jpqd.json \
    --distillation_weight 0.9 \
    --output_dir /tmp/jpqd-bert-base-ft-$TASK_NAME \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --optim adamw_torch \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --fp16 \
    --seed 42
```

On a single V100 GPU, this script should run in ~1.8 hours, and yield accuracy of **92.2%** with ~40% of the weights of the Transformer blocks pruned.
For launching the script on multiple GPUs specify `--nproc-per-node=<number of GPU>`. Note, that different batch size and other hyperparameters might be required to achieve the same results as on a single GPU.
