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

# Question answering


The script [`run_qa.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/question-answering/run_qa.py)
allows us to apply different quantization approaches (such as dynamic, static and aware-training quantization) as well as pruning 
using the [Intel® Neural Compressor](https://github.com/intel/neural-compressor) library for
question answering tasks.

For quantization, Intel® Neural Compressor has integrated the [IPEX (Intel Extension for PyTorch)](https://github.com/intel/intel-extension-for-pytorch) post-training static quantization. The flag `--use_ipex` can be passed along to use IPEX backend and `--bf16 --no_cuda` will enable mixed precision.

For pruning, we support snip_momentum(default), snip_momentum_progressive, magnitude, magnitude_progressive, gradient, gradient_progressive, snip, snip_progressive and pattern_lock. You can refer to [the pruning details](https://github.com/intel/neural-compressor/tree/master/neural_compressor/pruner#pruning-types).

> **_Note:_** At present, neural_compressor only support to prune linear and conv ops. So if we set a target sparsity is 0.9, it means that the pruning op's sparsity will be 0.9, not the whole model's sparsity is 0.9. For example: the embedding ops will not be pruned in the model.

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along 
the flag `--version_2_with_negative`.

The following example applies post-training static quantization on a DistilBERT fine-tuned on the SQuAD1.0 dataset.

```bash
python run_qa_post_training.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --apply_quantization \
    --quantization_approach static \
    --num_calibration_samples 40 \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/squad_output
```

In order to apply dynamic or static, `quantization_approach` must be set to respectively `dynamic` or `static`.

The following example fine-tunes DistilBERT on the SQuAD1.0 dataset while applying knowledge distillation with quantization aware training.

```bash
python run_qa.py \
    --model_name_or_path distilbert-base-uncased \
    --dataset_name squad \
    --apply_distillation \
    --generate_teacher_logits \
    --teacher_model_name_or_path distilbert-base-uncased-distilled-squad \
    --apply_quantization \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --max_train_samples 100 \
    --verify_loading \
    --output_dir /tmp/squad_output
```

The distillation process can be accelerated by the flag `--generate_teacher_logits`, which will add an additional step where the teacher outputs will be computed and saved in the training dataset, removing the need to compute the teacher outputs at every training step.

The following example fine-tunes DistilBERT on the SQuAD1.0 dataset while applying magnitude pruning and quantization aware training.

```bash
python run_qa.py \
    --model_name_or_path distilbert-base-uncased-distilled-squad \
    --dataset_name squad \
    --apply_quantization \
    --apply_pruning \
    --target_sparsity 0.1 \
    --num_train_epochs 4 \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/squad_output
```

## Prune Once For All

The following example demonstrate the steps of reproducing Prune Once For All examples results on SQuAD1.0 dataset.
<br>
This examples will take a pre-trained DistilBERT with a sparsity of 90% and fine-tune it on the downstream task. This fine-tuning process will be decomposed into two steps. During step 1, distillation as well as pattern lock pruning are both applied during fine-tuning. During step 2, quantization aware training will be additionaly applied, to obtain a fine-tuned quantized model with the same sparsity as the pre-trained model.
<br>
For more informations of Prune Once For All, please refer to the paper [Prune Once For All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)

```bash
# for stage 1
python run_qa.py \
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --dataset_name squad \
    --apply_distillation \
    --generate_teacher_logits \
    --teacher_model_name_or_path distilbert-base-uncased-distilled-squad \
    --apply_pruning \
    --pruning_config ../config/prune_pattern_lock.yml \
    --do_train \
    --do_eval \
    --learning_rate 1.8e-4 \
    --num_train_epochs 8 \
    --max_seq_length 384 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --pad_to_max_length \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --output_dir /tmp/squad_output_stage1

# for stage 2
python run_qa.py \
    --model_name_or_path /tmp/squad_output_stage1 \
    --dataset_name squad \
    --apply_distillation \
    --generate_teacher_logits \
    --teacher_model_name_or_path distilbert-base-uncased-distilled-squad \
    --apply_pruning \
    --pruning_config ../config/prune_pattern_lock.yml \
    --apply_quantization \
    --do_train \
    --do_eval \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --pad_to_max_length \
    --warmup_ratio 0.1 \
    --weight_decay 0 \
    --verify_loading \
    --output_dir /tmp/squad_output_stage2
```

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
