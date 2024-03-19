#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-classification",
        "question-answering",
        "audio-classification",
        "image-classification",
    ]
]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:
    import run_audio_classification
    import run_glue
    import run_image_classification
    import run_qa


class TestExamples(unittest.TestCase):
    def test_audio_classification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_audio_classification.py
                --model_name_or_path hf-internal-testing/tiny-random-Wav2Vec2Model
                --nncf_compression_config  examples/openvino/audio-classification/configs/wav2vec2-base-qat.json
                --dataset_name superb
                --dataset_config_name ks
                --max_train_samples 10
                --max_eval_samples 2
                --remove_unused_columns False
                --do_train
                --learning_rate 3e-5
                --max_length_seconds 1
                --attention_mask False
                --warmup_ratio 0.1
                --num_train_epochs 1
                --gradient_accumulation_steps 1
                --dataloader_num_workers 1
                --logging_strategy steps
                --logging_steps 1
                --evaluation_strategy epoch
                --save_strategy epoch
                --load_best_model_at_end False
                --seed 42
                --output_dir {tmp_dir}
                --overwrite_output_dir
                """.split()

            with patch.object(sys, "argv", test_args):
                run_audio_classification.main()

    def test_image_classification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_image_classification.py
                --model_name_or_path hf-internal-testing/tiny-random-ViTModel
                --dataset_name beans
                --max_train_samples 10
                --max_eval_samples 2
                --remove_unused_columns False
                --do_train
                --do_eval
                --learning_rate 2e-5
                --num_train_epochs 1
                --logging_strategy steps
                --logging_steps 1
                --evaluation_strategy epoch
                --save_strategy epoch
                --save_total_limit 1
                --seed 1337
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_image_classification.main()

    def test_text_classification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_glue.py
                --model_name_or_path hf-internal-testing/tiny-random-DistilBertForSequenceClassification
                --task_name sst2
                --max_train_samples 10
                --max_eval_samples 2
                --overwrite_output_dir
                --do_train
                --do_eval
                --max_seq_length 128
                --learning_rate 1e-5
                --optim adamw_torch
                --num_train_epochs 1
                --logging_steps 1
                --evaluation_strategy steps
                --eval_steps 1
                --save_strategy epoch
                --seed 42
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_glue.main()

    def test_question_answering(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_qa.py
                --model_name_or_path hf-internal-testing/tiny-random-DistilBertForQuestionAnswering
                --dataset_name squad
                --do_train
                --do_eval
                --max_train_samples 10
                --max_eval_samples 2
                --learning_rate 3e-5
                --num_train_epochs 1
                --max_seq_length 384
                --doc_stride 128
                --overwrite_output_dir
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_qa.main()


if __name__ == "__main__":
    unittest.main()
