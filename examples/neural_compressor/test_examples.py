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
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


os.environ["CUDA_VISIBLE_DEVICES"] = ""


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-classification",
        "question-answering",
        "token-classification",
        "multiple-choice",
        "language-modeling",
    ]
]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:
    import run_clm
    import run_glue
    import run_mlm
    import run_ner
    import run_qa
    import run_swag


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"Can't find {path}.")
    return results


class TestExamples(unittest.TestCase):
    def test_run_glue(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_glue.py
                --model_name_or_path hf-internal-testing/tiny-random-DistilBertForSequenceClassification
                --task_name sst2
                --apply_quantization
                --apply_pruning
                --target_sparsity 0.02
                --do_eval
                --do_train
                --per_device_eval_batch_size 1
                --per_device_train_batch_size 1
                --max_eval_samples 50
                --max_train_samples 4
                --num_train_epoch 2
                --learning_rate 1e-10
                --verify_loading
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_glue.main()
                get_results(tmp_dir)

    def test_run_qa(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_qa.py
                --model_name_or_path hf-internal-testing/tiny-random-DistilBertForQuestionAnswering
                --dataset_name squad
                --apply_quantization
                --apply_pruning
                --target_sparsity 0.02
                --do_eval
                --do_train
                --per_device_eval_batch_size 1
                --per_device_train_batch_size 1
                --max_eval_samples 50
                --max_train_samples 4
                --num_train_epoch 2
                --learning_rate 1e-10
                --verify_loading
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_qa.main()
                get_results(tmp_dir)

    def test_run_ner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_ner.py
                --model_name_or_path hf-internal-testing/tiny-random-RobertaForTokenClassification
                --dataset_name conll2003
                --apply_quantization
                --apply_pruning
                --target_sparsity 0.02
                --do_eval
                --do_train
                --per_device_eval_batch_size 1
                --per_device_train_batch_size 1
                --max_eval_samples 50
                --max_train_samples 4
                --num_train_epoch 2
                --learning_rate 1e-10
                --verify_loading
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_ner.main()
                get_results(tmp_dir)

    def test_run_swag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_swag.py
                --model_name_or_path hf-internal-testing/tiny-random-AlbertForMultipleChoice
                --apply_quantization
                --apply_pruning
                --target_sparsity 0.02
                --do_eval
                --do_train
                --per_device_eval_batch_size 1
                --per_device_train_batch_size 1
                --max_eval_samples 50
                --max_train_samples 4
                --num_train_epoch 2
                --learning_rate 1e-10
                --verify_loading
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_swag.main()
                get_results(tmp_dir)

    def test_run_clm(self):
        quantization_approach = "dynamic"

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_clm.py
                --model_name_or_path hf-internal-testing/tiny-random-GPT2LMHeadModel
                --dataset_name wikitext
                --dataset_config_name wikitext-2-raw-v1
                --apply_quantization
                --quantization_approach {quantization_approach}
                --apply_pruning
                --target_sparsity 0.02
                --do_eval
                --do_train
                --per_device_eval_batch_size 1
                --per_device_train_batch_size 1
                --max_eval_samples 50
                --max_train_samples 4
                --num_train_epoch 2
                --learning_rate 1e-10
                --verify_loading
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_clm.main()
                get_results(tmp_dir)

    def test_run_mlm(self):
        quantization_approach = "static"

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_args = f"""
                run_mlm.py
                --model_name_or_path hf-internal-testing/tiny-random-DistilBertForMaskedLM
                --dataset_name wikitext
                --dataset_config_name wikitext-2-raw-v1
                --apply_quantization
                --quantization_approach {quantization_approach}
                --apply_pruning
                --target_sparsity 0.02
                --do_eval
                --do_train
                --per_device_eval_batch_size 1
                --per_device_train_batch_size 1
                --max_eval_samples 50
                --max_train_samples 4
                --num_train_epoch 2
                --learning_rate 1e-10
                --verify_loading
                --output_dir {tmp_dir}
                """.split()

            with patch.object(sys, "argv", test_args):
                run_mlm.main()
                get_results(tmp_dir)


if __name__ == "__main__":
    unittest.main()
