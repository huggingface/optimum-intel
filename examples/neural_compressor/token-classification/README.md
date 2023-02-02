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

# Token classification


The script [`run_ner.py`](https://github.com/huggingface/optimum-intel/blob/main/examples/neural_compressor/token-classification/run_ner.py)
allows us to apply different quantization approaches (such as dynamic, static and aware-training quantization) as well as pruning
using the [Intel Neural Compressor ](https://github.com/intel/neural-compressor) library for token classification tasks.

For pruning, we support snip_momentum(default), snip_momentum_progressive, magnitude, magnitude_progressive, gradient, gradient_progressive, snip, snip_progressive and pattern_lock. You can refer to [the pruning details](https://github.com/intel/neural-compressor/tree/master/neural_compressor/pruner#pruning-types).

> **_Note:_** At present, neural_compressor only support to prune linear and conv ops. So if we set a target sparsity is 0.9, it means that the pruning op's sparsity will be 0.9, not the whole model's sparsity is 0.9. For example: the embedding ops will not be pruned in the model.

The following example applies post-training static quantization on a DistilBERT fine-tuned on the CoNLL-2003 task.

```bash
python run_ner_post_training.py \
    --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english \
    --dataset_name conll2003 \
    --apply_quantization \
    --quantization_approach static \
    --num_calibration_samples 50 \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/conll03_output
```
In order to apply dynamic or static, `quantization_approach` must be set to respectively `dynamic` or `static`.

The following example fine-tunes DistilBERT on the CoNLL-2003 task while applying knowledge distillation with quantization aware training.

```bash
python run_ner.py \
    --model_name_or_path distilbert-base-uncased \
    --dataset_name conll2003 \
    --apply_distillation \
    --teacher_model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english \
    --apply_quantization \
    --num_train_epochs 1 \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/conll03_output
```

The following example fine-tunes DistilBERT on the CoNLL-2003 task while applying magnitude pruning.

```bash
python run_ner.py \
    --model_name_or_path elastic/distilbert-base-uncased-finetuned-conll03-english \
    --dataset_name conll2003 \
    --apply_quantization \
    --apply_pruning \
    --target_sparsity 0.1 \
    --num_train_epochs 4 \
    --max_train_samples 100 \
    --do_train \
    --do_eval \
    --verify_loading \
    --output_dir /tmp/conll03_output
```

The flag `--verify_loading` can be passed along to verify that the resulting quantized model can be loaded correctly.