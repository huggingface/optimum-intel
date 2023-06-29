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

# Text classification 

## GLUE tasks

The script [`run_glue.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/text-classification/run_glue.py)
allows us to apply different quantization approaches (such as dynamic, static and quantization-aware training) as well as pruning 
using the [Intel Neural Compressor ](https://github.com/intel/neural-compressor) library for 
sequence classification tasks such as the ones from the [GLUE benchmark](https://gluebenchmark.com/).

For pruning, we support snip_momentum(default), snip_momentum_progressive, magnitude, magnitude_progressive, gradient, gradient_progressive, snip, snip_progressive and pattern_lock. You can refer to [the pruning details](https://github.com/intel/neural-compressor/tree/master/neural_compressor/pruner#pruning-types).

> **_Note:_** At present, neural_compressor only support to prune linear and conv ops. So if we set a target sparsity is 0.9, it means that the pruning op's sparsity will be 0.9, not the whole model's sparsity is 0.9. For example: the embedding ops will not be pruned in the model.

### Post-training Quantization

The following example applies post-training static quantization on a DistilBERT fine-tuned on the sst-2 task.

```bash
python run_glue_post_training.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --apply_quantization \
    --quantization_approach static \
    --num_calibration_samples 50 \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/sst2_output
```
In order to apply dynamic or static, `quantization_approach` must be set to respectively `dynamic` or `static`.


### Distributed Tuning with Multi-node
To speedup the accuracy-aware tuning process, you can use the [distributed tuning](https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#distributed-tuning).

- Prerequisites:
    - [Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build)
    - [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html#using-pip)


> **_Note:_** User can also install Open MPI with [Conda](https://anaconda.org/conda-forge/openmpi).

In `run_glue_post_training.py`, set `quant_level` to 1 as the following statement:

```python
from neural_compressor import PostTrainingQuantConfig
quantization_config = PostTrainingQuantConfig(approach=optim_args.quantization_approach, quant_level=1)
```

> **_Note:_** Please also add the below code snippet at the beginning of the example to disable the initialization of torch distribution.
> 
>`os.environ.setdefault('OMPI_COMM_WORLD_SIZE', '-1') if os.environ.get('OMPI_COMM_WORLD_SIZE', -1) != -1 else None`

And then, modify the `run_task_in_distributed_mode.sh` according to the cluster information. Below is the explanation for each parameter.

- `<NUM_PROCESS>` is the number of processes, recommend to set it with the number of hosts plus one.

- `<MAX_NUM_THREADS>` is the number of threads, recommend to set it with the number of physical cores on one node.

- `<HOSTNAME>` is the host name, and argument --host `<HOSTNAME>`,`<HOSTNAME>`... can be replaced with `--hostfile <HOSTFILE>`, when each line in `<HOSTFILE>` is a host name.

- `-mca btl_tcp_if_include <NETWORK_INTERFACE>` is used to set the network communication interface between hosts. For example, `<NETWORK_INTERFACE>` can be set to `192.168.20.0/24` to allow MPI communication between all hosts under `192.168.20.*` network segment.


### Knowledge Distillation with Quantization Aware Training

The following example fine-tunes DistilBERT on the sst-2 task while applying knowledge distillation with quantization aware training.

```bash
python run_glue.py \
    --model_name_or_path distilbert-base-uncased \
    --task_name sst2 \
    --apply_distillation \
    --teacher_model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --apply_quantization \
    --num_train_epochs 1 \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/sst2_output
```

The following example fine-tunes DistilBERT on the sst-2 task while applying magnitude pruning:

```bash
python run_glue.py \
    --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --task_name sst2 \
    --apply_pruning \
    --target_sparsity 0.1 \
    --num_train_epochs 4 \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/sst2_output \
    --overwrite_output_dir
```

### Prune Once For All

The following example demonstrate the steps of reproducing Prune Once For All examples results on the sst-2 task.
<br>
This examples will take a pre-trained DistilBERT with a sparsity of 90% and fine-tune it on the downstream task. This fine-tuning process will be decomposed into two steps. During step 1, distillation as well as pattern lock pruning are both applied during fine-tuning. During step 2, quantization aware training will be additionaly applied, to obtain a fine-tuned quantized model with the same sparsity as the pre-trained model.
<br>
For more informations of Prune Once For All, please refer to the paper [Prune Once For All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)

```bash
# for stage 1
python run_glue_during_training_optimization.py \
    --model_name_or_path Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa \
    --task_name sst2 \
    --apply_distillation \
    --teacher_model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --apply_pruning \
    --pruning_config ../config/prune_pattern_lock.yml \
    --do_train \
    --do_eval \
    --learning_rate 1.5e-4 \
    --num_train_epochs 9 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --pad_to_max_length \
    --warmup_ratio 0 \
    --weight_decay 0 \
    --output_dir /tmp/sst2_output_stage1

# for stage 2
python run_glue_during_training_optimization.py \
    --model_name_or_path /tmp/sst2_output_stage1 \
    --task_name sst2 \
    --apply_distillation \
    --teacher_model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
    --apply_pruning \
    --pruning_config ../config/prune_pattern_lock.yml \
    --apply_quantization \
    --do_train \
    --do_eval \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --pad_to_max_length \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --verify_loading \
    --output_dir /tmp/sst2_output_stage2
```

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.
