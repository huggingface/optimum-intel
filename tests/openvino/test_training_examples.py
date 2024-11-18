#  Copyright 2023 The HuggingFace Team. All rights reserved.
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
import subprocess
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch
import torch.cuda
from parameterized import parameterized

from optimum.intel.openvino.utils import OV_XML_FILE_NAME, TemporaryDirectory


PROJECT_ROOT = Path(__file__).parents[2]
OPENVINO_EXAMPLES_PATH = PROJECT_ROOT / "examples" / "openvino"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


@dataclass
class TrainingExampleDescriptor:
    cwd: Union[Path, str]
    filename: str
    args: List[str]
    timeout: int

    def get_args_with_output_dir(self, output_dir: Union[Path, str]):
        flag = "--output_dir"
        args = self.args.copy()
        if flag in args:
            idx = args.index(flag)
            del args[idx : idx + 2]
        return [*args, flag, str(output_dir)]


TRAINING_EXAMPLE_DESCRIPTORS = {
    "text-classification-QAT": TrainingExampleDescriptor(
        cwd=OPENVINO_EXAMPLES_PATH / "text-classification",
        filename="run_glue.py",
        args=[
            "--model_name_or_path",
            "hf-internal-testing/tiny-bert",
            "--task_name",
            "sst2",
            "--do_train",
            "--do_eval",
            "--per_device_train_batch_size",
            "2",
            "--per_device_eval_batch_size",
            "8",
            "--logging_steps",
            "1",
            "--evaluation_strategy",
            "steps",
            "--eval_steps",
            "2",
            "--save_strategy",
            "steps",
            "--save_steps",
            "2",
            "--save_total_limit",
            "1",
            "--max_steps",
            "5",
            "--fp16",
            "--report_to",
            "none",
        ],
        timeout=300,
    ),
    "text-classification-JPQD": TrainingExampleDescriptor(
        cwd=OPENVINO_EXAMPLES_PATH / "text-classification",
        filename="run_glue.py",
        args=[
            "--model_name_or_path",
            "hf-internal-testing/tiny-bert",
            "--teacher_model_name_or_path",
            "hf-internal-testing/tiny-bert",
            "--nncf_compression_config",
            "./configs/bert-base-jpqd.json",
            "--task_name",
            "sst2",
            "--do_train",
            "--do_eval",
            "--per_device_train_batch_size",
            "2",
            "--per_device_eval_batch_size",
            "8",
            "--logging_steps",
            "1",
            "--evaluation_strategy",
            "steps",
            "--eval_steps",
            "2",
            "--save_strategy",
            "steps",
            "--save_steps",
            "2",
            "--save_total_limit",
            "1",
            "--max_steps",
            "5",
            "--fp16",
            "--report_to",
            "none",
        ],
        timeout=300,
    ),
}


def get_available_cuda_device_ids() -> List[int]:
    torch_device_count = torch.cuda.device_count()
    visible_devices_str = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not visible_devices_str:
        return list(range(torch_device_count))
    device_ids = list(map(int, visible_devices_str.strip().split(",")))
    if len(device_ids) != torch_device_count:
        # Cannot decide device ids since some devices in env are unavailable.
        return []
    return device_ids


class OVTrainingExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.available_cuda_device_ids = get_available_cuda_device_ids()
        self.env = os.environ.copy()

    @parameterized.expand(TRAINING_EXAMPLE_DESCRIPTORS.items())
    def test_single_card_training(self, _, desc: TrainingExampleDescriptor):
        if len(self.available_cuda_device_ids) < 1:
            self.skipTest("No enough cuda devices.")

        self.env[CUDA_VISIBLE_DEVICES] = str(self.available_cuda_device_ids[0])
        with TemporaryDirectory() as output_dir:
            args = ["torchrun", "--nproc_per_node=1", desc.filename, *desc.get_args_with_output_dir(output_dir)]
            proc = subprocess.Popen(
                args=args,
                cwd=desc.cwd,
                env=self.env.copy(),
            )
            return_code = proc.wait(desc.timeout)
            self.assertEqual(return_code, 0)
            self.assertTrue(Path(output_dir, OV_XML_FILE_NAME).is_file())

    @parameterized.expand(TRAINING_EXAMPLE_DESCRIPTORS.items())
    def test_data_parallel_training(self, _, desc: TrainingExampleDescriptor):
        if len(self.available_cuda_device_ids) < 2:
            self.skipTest("No enough cuda devices.")

        self.env[CUDA_VISIBLE_DEVICES] = ",".join(map(str, self.available_cuda_device_ids[:2]))
        with TemporaryDirectory() as output_dir:
            args = [sys.executable, desc.filename, *desc.get_args_with_output_dir(output_dir)]
            proc = subprocess.Popen(
                args=args,
                cwd=desc.cwd,
                env=self.env.copy(),
            )
            return_code = proc.wait(desc.timeout)
            self.assertEqual(return_code, 0)
            self.assertTrue(Path(output_dir, OV_XML_FILE_NAME).is_file())

    @parameterized.expand(TRAINING_EXAMPLE_DESCRIPTORS.items())
    def test_distributed_data_parallel_training(self, _, desc: TrainingExampleDescriptor):
        if len(self.available_cuda_device_ids) < 2:
            self.skipTest("No enough cuda devices.")

        self.env[CUDA_VISIBLE_DEVICES] = ",".join(map(str, self.available_cuda_device_ids[:2]))
        with TemporaryDirectory() as output_dir:
            args = [
                "torchrun",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                "--nnodes=1",
                "--nproc_per_node=2",
                desc.filename,
                *desc.get_args_with_output_dir(output_dir),
            ]
            proc = subprocess.Popen(
                args=args,
                cwd=desc.cwd,
                env=self.env.copy(),
            )
            return_code = proc.wait(desc.timeout)
            self.assertEqual(return_code, 0)
            self.assertTrue(Path(output_dir, OV_XML_FILE_NAME).is_file())
