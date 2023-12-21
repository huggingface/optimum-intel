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


import types

import torch


def patch_model_with_bettertransformer(model, model_config):
    try:
        model = model.to_bettertransformer()
    except Exception as e:
        print(f"[ WARNING ] Cannot apply model.to_bettertransformer because of the exception:\n{e}")
        return model

    # for better transformers we need sequence lenght to be not 1 to make a correct trace
    # patch generate_dummy_inputs in the config

    def pathed_generate_dummy_inputs(self, *args, **kwargs):
        dummy_inputs = self._original_generate_dummy_inputs(*args, **kwargs)
        if "input_ids" in dummy_inputs and dummy_inputs["input_ids"].shape[1] == 1:
            dummy_inputs["input_ids"] = torch.cat([dummy_inputs["input_ids"], dummy_inputs["input_ids"]], dim=-1)
            attention_mask = dummy_inputs["attention_mask"]
            dummy_inputs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return dummy_inputs

    model_config._original_generate_dummy_inputs = model_config.generate_dummy_inputs
    model_config.generate_dummy_inputs = types.MethodType(pathed_generate_dummy_inputs, model_config)

    return model
