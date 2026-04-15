# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from collections import defaultdict
from functools import wraps

from transformers.utils.generic import logger


try:
    # transformers>=5.2
    from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder
except ImportError:
    from transformers.utils.generic import _CAN_RECORD_REGISTRY, OutputRecorder


# This is a fixed version of transformers.utils.generic.check_model_inputs
# that fixes issues related to onnx export and tracing
# - adds support for positional args (use_cache), without which use_cache end up being passed twice
# - fixes issue with default capture_flags being None for some models
def traceable_check_model_inputs(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        use_cache = (
            kwargs["use_cache"] if kwargs.get("use_cache") is not None else getattr(self.config, "use_cache", None)
        )
        if use_cache is not None:
            if getattr(self, "gradient_checkpointing", False) and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            # Prevent passing use_cache twice
            if "use_cache" in func.__code__.co_varnames:
                use_cache_idx = func.__code__.co_varnames.index("use_cache") - 1  # minus 1 for 'self'
                if len(args) > use_cache_idx:
                    args = list(args)
                    args[use_cache_idx] = use_cache
                    args = tuple(args)
                else:
                    kwargs["use_cache"] = use_cache

        return_dict = kwargs.pop("return_dict", None)
        if return_dict is None:
            return_dict = getattr(self.config, "return_dict", True)

        all_args = kwargs.copy()
        if "kwargs" in all_args:
            for k, v in all_args["kwargs"].items():
                all_args[k] = v

        capture_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__)) or {}  # there is a weak ref for executorch

        recordable_keys = {
            f"output_{k}": all_args.get(
                f"output_{k}",
                getattr(
                    self.config,
                    f"output_{k}",
                    all_args.get("output_attentions", getattr(self.config, "output_attentions", False)),
                ),
            )
            for k in capture_flags
        }

        # We let cross attentions to be saved separately because some models add `cross-attn` layer
        # when certain conditions are met. Let's output cross attention if attentions are requested (for BC)
        if "output_attentions" in recordable_keys:
            recordable_keys["output_cross_attentions"] = recordable_keys["output_attentions"]

        collected_outputs = defaultdict(tuple)
        monkey_patched_layers = []

        # Check attention implementation is properly set for capturing attention outputs
        if recordable_keys.get("output_attentions", False):
            supported_attn = ["eager", "eager_paged", "flex_attention"]
            config_attn = getattr(self.config, "_attn_implementation", None)
            sub_configs = [getattr(self.config, key, None) for key in self.config.sub_configs]
            sub_configs_attn = [
                getattr(config, "_attn_implementation", None) for config in sub_configs if config is not None
            ]
            if config_attn not in supported_attn or any(attn not in supported_attn for attn in sub_configs_attn):
                warnings.warn(
                    f"`output_attentions=True` is not supported with `attn_implementation` other than {supported_attn}. "
                    "Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.",
                    UserWarning,
                    stacklevel=2,
                )

        def make_capture_wrapper(module, orig_forward, key, index):
            @wraps(orig_forward)
            def wrapped_forward(*args, **kwargs):
                if key == "hidden_states" and len(collected_outputs[key]) == 0:
                    collected_outputs[key] += (args[0],)
                output = orig_forward(*args, **kwargs)
                if not isinstance(output, tuple):
                    collected_outputs[key] += (output,)
                elif output[index] is not None:
                    if key not in collected_outputs:
                        collected_outputs[key] = (output[index],)
                    else:
                        collected_outputs[key] += (output[index],)
                return output

            return wrapped_forward

        if any(recordable_keys.values()):
            capture_tasks = []
            for key, layer_specs in capture_flags.items():
                if not recordable_keys.get(f"output_{key}", False):
                    continue
                if not isinstance(layer_specs, list):
                    layer_specs = [layer_specs]
                for specs in layer_specs:
                    if not isinstance(specs, OutputRecorder):
                        index = 0 if "hidden_states" in key else 1
                        class_name = None if not isinstance(specs, str) else specs
                        target_class = specs if not isinstance(specs, str) else None
                        specs = OutputRecorder(target_class=target_class, index=index, class_name=class_name)
                    capture_tasks.append((key, specs))

            for name, module in self.named_modules():
                for key, specs in capture_tasks:
                    # The second check is for multimodals where only backbone layer suffix is available
                    if (specs.target_class is not None and isinstance(module, specs.target_class)) or (
                        specs.class_name is not None and name.endswith(specs.class_name)
                    ):
                        if specs.layer_name is not None and specs.layer_name not in name:
                            continue
                        # Monkey patch forward
                        original_forward = module.forward
                        module.forward = make_capture_wrapper(module, original_forward, key, specs.index)
                        monkey_patched_layers.append((module, original_forward))

        outputs = func(self, *args, **kwargs)
        # Restore original forward methods
        for module, original_forward in monkey_patched_layers:
            module.forward = original_forward

        # Inject collected outputs into model output
        for key in collected_outputs:
            if key == "hidden_states":
                if hasattr(outputs, "vision_hidden_states"):
                    collected_outputs[key] = collected_outputs[key][:-1]
                    collected_outputs[key] += (outputs.vision_hidden_states,)
                elif hasattr(outputs, "last_hidden_state"):
                    collected_outputs[key] = collected_outputs[key][:-1]
                    collected_outputs[key] += (outputs.last_hidden_state,)

                outputs[key] = collected_outputs[key]
            elif key == "attentions":
                if isinstance(capture_flags[key], list) and len(capture_flags[key]) == 2:
                    outputs[key] = collected_outputs[key][0::2]
                    outputs["cross_" + key] = collected_outputs[key][1::2]
                else:
                    outputs[key] = collected_outputs[key]
            else:
                outputs[key] = collected_outputs[key]
        if return_dict is False:
            outputs = outputs.to_tuple()
        return outputs

    return wrapper
