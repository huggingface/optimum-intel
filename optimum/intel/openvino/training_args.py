#  Copyright 2024 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class OVTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to OpenVINO/NNCF-enabled training flow
    """

    distillation_weight: float = field(
        default=0.5, metadata={"help": "weightage of distillation loss, value between 0.0 to 1.0"}
    )
    distillation_temperature: float = field(default=2.0, metadata={"help": "temperature of distillation."})

    def __post_init__(self):
        super().__post_init__()
        if self.distillation_weight < 0.0 or self.distillation_weight > 1.0:
            raise ValueError("distillation_weight must be between 0.0 and 1.0")

        if self.distillation_temperature < 1:
            raise ValueError("distillation_temperature must be >= 1.0")
