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

# Language modeling training

The scripts [`run_clm.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/language-modeling/run_clm.py) 
and [`run_mlm.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/language-modeling/run_mlm.py)
allow us to apply different quantization approaches (such as dynamic, static, weight-only and aware-training quantization) as well as pruning 
using the [Intel Neural Compressor ](https://github.com/intel/neural-compressor) library for language modeling tasks.

The SmoothQuant methodology is also available for post-training quantization.

For pruning, we support snip_momentum(default), snip_momentum_progressive, magnitude, magnitude_progressive, gradient, gradient_progressive, snip, snip_progressive and pattern_lock. You can refer to [the pruning details](https://github.com/intel/neural-compressor/tree/master/neural_compressor/pruner#pruning-types).

> **_Note:_** At present, neural_compressor only support to prune linear and conv ops. So if we set a target sparsity is 0.9, it means that the pruning op's sparsity will be 0.9, not the whole model's sparsity is 0.9. For example: the embedding ops will not be pruned in the model.


GPT and GPT-2 are trained or fine-tuned using a causal language modeling (CLM) loss. ALBERT, BERT, DistilBERT and 
RoBERTa are trained or fine-tuned using a masked language modeling (MLM) loss, more information about the differences 
between those objectives can be found in our [model summary](https://huggingface.co/transformers/model_summary.html).


### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-Neo on WikiText-2 while first applying snip_momentum pruning and then quantization aware training.
We're using the raw WikiText-2 (no tokens were replaced before the tokenization). The loss here is that of causal language modeling (CLM). 

```bash
python run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-125M \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --apply_quantization \
    --quantization_approach aware_training \
    --apply_pruning \
    --target_sparsity 0.02 \
    --num_train_epochs 4 \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/clm_output
```

The following example shows how to apply post-training static quantization using the SmoothQuant methodology on a GPT-Neo model :
```bash
python run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-125M \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --apply_quantization \
    --quantization_approach static \
    --smooth_quant \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/clm_output
```

### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2 while applying quantization aware training and snip_momentum pruning. We're using the raw 
WikiText-2. The loss is different as BERT/RoBERTa have a bidirectional mechanism, we are therefore using the same loss 
that was used during their pre-training: masked language modeling (MLM) loss. 

```bash
python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --apply_quantization \
    --quantization_approach aware_training \
    --apply_pruning \
    --target_sparsity 0.02 \
    --num_train_epochs 4 \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/mlm_output
```

In order to apply dynamic, static, weight-only or aware-training quantization, `quantization_approach` must be set to 
respectively `dynamic`, `static`, `weight_only` or `aware_training`.

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.

> **_Note:_** `weight_only` quantization_approach requires `neural-compressor` > 3.0.
