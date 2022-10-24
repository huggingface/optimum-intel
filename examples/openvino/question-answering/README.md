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
# Question answering

This folder contains [`run_qa.py`](https://github.com/huggingface/optimum/blob/main/examples/openvino/question-answering/run_qa.py), a script to fine-tune a 🤗 Transformers model on a question answering dataset while applying quantization aware training (QAT). QAT can be easily applied by replacing the Transformers [`Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) with the Optimum [`OVTrainer`].
An `QuestionAnsweringOVTrainer` is defined in [`trainer_qa.py`](https://github.com/huggingface/optimum/blob/main/examples/openvino/question-answering/trainer_qa.py), which inherits from `OVTrainer` and is adapted to perform question answering tasks evaluation.

Any model from our [hub](https://huggingface.co/models) (as long as the model supported by the [`AutoModelForQuestionAnswering`](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForQuestionAnswering) API) can be fine-tuned on a question-answering dataset (such as SQuAD, or any other QA dataset available in the `datasets` library, or your own csv/jsonlines files) as long as they are structured the same way as SQuAD. You might need to tweak the data processing inside the script if your data is structured differently.

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks).

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the flag `--version_2_with_negative`.

### Fine-tuning BERT on SQuAD1.0

Here we show how to apply quantization aware training (QAT) on a fine-tuned DistilBERT on the SQuAD1.0 dataset.

```bash
python run_qa.py \
  --model_name_or_path distilbert-base-uncased-distilled-squad \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --max_train_samples 1024 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/outputs_squad/
```