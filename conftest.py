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

import pytest


def pytest_addoption(parser):
    parser.addoption("--use_torch_export", action="store", default="False")


def pytest_collection_modifyitems(config, items):
    if config.option.use_torch_export == "True":
        with open(
                "tests/openvino/excluded_tests_with_torch_export.txt", "r"
        ) as file:
            skipped_tests = file.readlines()
            # it is necessary to check if stripped line is not empty
            # and exclude such lines
            skipped_tests = [
                line.strip() for line in skipped_tests if line.strip()
            ]
            for item in items:
                for skipped_test in skipped_tests:
                    if skipped_test in item.nodeid:
                        item.add_marker(pytest.mark.skip("Unsupported yet with torch.export()."))
