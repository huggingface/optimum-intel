#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino import Core, Tensor, Type
from openvino.preprocess import PrePostProcessor
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.generation import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateOutput, GenerationMode
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils.hub import PushToHubMixin

from optimum.utils.normalized_config import NormalizedConfigManager

from ...exporters.openvino import ensure_stateful_is_available, main_export, patch_stateful
from ...exporters.openvino.stateful import model_has_state
from ...exporters.openvino.utils import SSM_MODELS
from ..utils.import_utils import compare_versions, is_nncf_available, is_transformers_version
from ..utils.modeling_utils import MULTI_QUERY_ATTN_MODELS
from .configuration import (
    _DEFAULT_4BIT_WQ_CONFIG,
    OVConfig,
    OVWeightQuantizationConfig,
    get_default_quantization_config,
)
from .modeling import _TOKENIZER_FOR_DOC, INPUTS_DOCSTRING, MODEL_START_DOCSTRING, OVModel
from .utils import (
    ONNX_WEIGHTS_NAME,
    OV_XML_FILE_NAME,
    STR_TO_OV_TYPE,
    TemporaryDirectory,
    get_export_transformers_version,
    model_has_dynamic_inputs,
)


if is_transformers_version(">=", "4.43"):
    from transformers.cache_utils import MambaCache
else:
    MambaCache = object

if TYPE_CHECKING:
    try:
        from transformers.generation.streamers import BaseStreamer
    except Exception:
        from typing import Generator as BaseStreamer

    from transformers.modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

core = Core()


TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> inputs = tokenizer("I love this story because", return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)
    ```
    Example using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    >>> text = "I love this story because"
    >>> gen = gen_pipeline(text)
    ```
"""


# inheritage from PushToHubMixin added as workaround for transformers>=4.52.0 and nncf<=2.16.0 compatibility
# during dataset preparatioon nncf checks isinstance(model, PreTrainedModel.__bases__)
# in transformers 4.52.0 PreTrainedModel does not include GenerationMixin and this check failed for OVModelForCausalLM
# TO DO: remove it after migration on new nncf
@add_start_docstrings(
    """
    Base OVBaseDecoderModel class.
    """,
)
class OVBaseDecoderModel(OVModel, PushToHubMixin):
    def __init__(
        self,
        model: openvino.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = None,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        if dynamic_shapes is not None:
            logger.warning(
                f"`dynamic_shapes` was set to {dynamic_shapes}, but this value will be ignored as only dynamic shapes are supported."
            )

        compile_only = kwargs.get("compile_only", False)
        enable_compilation = kwargs.get("compile", True)
        kwargs["compile"] = False or compile_only  # avoid extra compilation in the base class
        if compile_only and not enable_compilation:
            raise ValueError(
                "`compile_only` mode does not support disabling compilation."
                "Please provide `compile=True` if you want to use `compile_only=True` or set `compile_only=False`"
            )

        config.is_encoder_decoder = False

        super().__init__(
            model,
            config,
            device=device,
            # dynamic_shapes set to False for compile_only=False to make sure the model is not reshaped dynamically
            dynamic_shapes=model_has_dynamic_inputs(model) if compile_only else False,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )
        self.is_dynamic = True
        use_cache = kwargs.pop("use_cache", True)
        model_has_sinks = model_has_state(self.model)
        self.use_cache = self._has_cache_inputs(model) or model_has_sinks
        stateful = kwargs.pop("stateful", None)  # stateful model only if it is converted with stateful=True
        self.stateful = model_has_sinks
        self.main_input_name = "input_ids"
        self.num_pkv = 2
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        self.key_value_output_names = [key for key in self.output_names if "present" in key]
        # Keeping the original model for serialization
        self._pkv_precision = Type.f32
        self.next_beam_idx = None
        self._past_length = 0
        self._first_iter_beam_search = False
        self._second_iter_beam_search = False
        self.update_pkv_precision()

        # reshape with dynamic shapes is needed for decoder_with_past model to be compatible with inference
        if not self.stateful and not self._compile_only:
            self.model = self._reshape(self.model, -1, -1)

        is_stateful_supported = ensure_stateful_is_available(warn=False)

        if self.use_cache and not self.stateful:
            logger.warning(
                "Provided model does not contain state. It may lead to sub-optimal performance."
                "Please reexport model with updated OpenVINO version >= 2023.3.0 calling the `from_pretrained` method with original model "
                "and `export=True` parameter"
            )

        if self.stateful:
            if stateful is None:
                stateful = is_stateful_supported
            if model_has_sinks and not is_stateful_supported:
                raise ValueError(
                    "Loaded stateful model, while OpenVINO runtime version does not support stateful model inference. "
                    "Please update OpenVINO version >= 2023.3.0 "
                    "or export the original model once again with `stateful=False` when calling the `from_pretrained` method."
                    "To export your model, simply set `export=True`."
                )

        def raise_error(model_prop, user_prop, name):
            raise ValueError(
                f"`{name}` was set to `{user_prop}` but the loaded model only supports `{name}={model_prop}`. "
                f"Please load your current model with `{name}={model_prop}` or export the original model "
                f"once again with `{name}={user_prop}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )

        if stateful is not None and stateful ^ self.stateful:
            # We cannot transform stateful model to stateless
            raise_error(self.stateful, stateful, "stateful")

        if use_cache ^ self.use_cache:
            raise_error(self.use_cache, use_cache, "use_cache")

        if self._compile_only:
            self.request = self.model.create_infer_request()

        if not self._compile_only and enable_compilation:
            self.compile()

    @staticmethod
    def _has_cache_inputs(model: openvino.Model) -> bool:
        return any("past_key_values" in key.get_any_name() for key in model.inputs)

    @staticmethod
    def _get_model_with_updated_pkv_precision(model: openvino.Model, pkv_precision: Type) -> openvino.Model:
        ppp = PrePostProcessor(model)
        for key in model.inputs:
            if "past_key_values" in key.get_any_name() and pkv_precision != key.get_element_type():
                ppp.input(key.get_any_name()).tensor().set_element_type(pkv_precision)
        for key in model.outputs:
            if "present" in key.get_any_name() and pkv_precision != key.get_element_type():
                ppp.output(key.get_any_name()).tensor().set_element_type(pkv_precision)
        return ppp.build()

    def update_pkv_precision(self, force_fp32=False):
        if not self.use_cache or self.stateful or self._compile_only:
            return
        pkv_precision = Type.f32
        if not force_fp32:
            device = self._device.upper()
            try:
                if "INFERENCE_PRECISION_HINT" in core.get_property(device, "SUPPORTED_PROPERTIES"):
                    pkv_precision = core.get_property(device, "INFERENCE_PRECISION_HINT")
            except RuntimeError:  # use default precision when get_property fails, e.g. when device is "AUTO:GPU"
                pass

            # ov_config["INFERENCE_PRECISION_HINT"] may override the prefer precision
            if self.ov_config:
                inference_precision_hint = self.ov_config.get("INFERENCE_PRECISION_HINT", "")
                if inference_precision_hint in STR_TO_OV_TYPE:
                    pkv_precision = STR_TO_OV_TYPE[inference_precision_hint]

            self.model = self._get_model_with_updated_pkv_precision(self.model, pkv_precision)
            self._pkv_precision = pkv_precision
            self.request = None
        else:
            if hasattr(self, "_pkv_precision") and self._pkv_precision != Type.f32:
                self.model = self._get_model_with_updated_pkv_precision(self.model, Type.f32)
                self._pkv_precision = Type.f32
                if self.is_dynamic and not self._compile_only:
                    self.model = self._reshape(self.model, -1, -1)
                self.request = None

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """

        if self._compile_only:
            raise ValueError(
                "`save_pretrained()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        model_to_save = (
            self.model
            if self._pkv_precision == Type.f32
            else self._get_model_with_updated_pkv_precision(self.model.clone(), Type.f32)
        )
        dst_path = os.path.join(save_directory, OV_XML_FILE_NAME)
        openvino.save_model(model_to_save, dst_path, compress_to_fp16=False)

        if self.generation_config is not None:
            try:
                self.generation_config.save_pretrained(save_directory)
            except Exception as exception:
                logger.warning(
                    f"The generation config will not be saved, saving failed with following error:\n{exception}"
                )

        self._save_openvino_config(save_directory)

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        use_cache: bool = True,
        trust_remote_code: bool = False,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting
        # would end-up removing the directory containing the underlying OpenVINO model
        cls._model_save_dir_tempdirectory_instance = save_dir

        compile_only = kwargs.pop("compile_only", False)
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False

        if task is None:
            task = cls.export_feature
            if use_cache:
                task = task + "-with-past"

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_export_config = None
        else:
            ov_export_config = OVConfig(dtype="auto")

        stateful = kwargs.pop("stateful", ensure_stateful_is_available(warn=False) and use_cache)

        torch_dtype = kwargs.pop("torch_dtype", None)

        model_loading_kwargs = {}

        if torch_dtype is not None:
            model_loading_kwargs["torch_dtype"] = torch_dtype

        variant = kwargs.pop("variant", None)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            ov_config=ov_export_config,
            stateful=stateful,
            model_loading_kwargs=model_loading_kwargs,
            library_name=cls._library_name,
            variant=variant,
        )

        if config.model_type == "phi3" and config.max_position_embeddings != getattr(
            config, "original_max_position_embeddings", config.max_position_embeddings
        ):
            config.max_position_embeddings = config.original_max_position_embeddings

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            stateful=None,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            compile_only=compile_only,
            **kwargs,
        )

    def _reshape(
        self,
        model: openvino.Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        if self._compile_only:
            raise ValueError(
                "`reshape()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        if height is not None:
            logger.warning(f"`height` set to `{height}` will be ignored during reshaping operation.")

        if width is not None:
            logger.warning(f"`width` set to `{width}` will be ignored during reshaping operation.")

        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = -1
            input_name = inputs.get_any_name()
            if input_name.startswith("past_key_values"):
                if (len(inputs.partial_shape) == 3 and input_name.endswith("value")) or (
                    self.config.model_type == "chatglm" and not hasattr(self.config, "rope_ratio")
                ):
                    shapes[inputs][1] = -1
                else:
                    shapes[inputs][2] = -1
            elif input_name.startswith("beam_idx") or input_name.startswith("cache_position"):
                shapes[inputs][0] = -1
            else:
                shapes[inputs][1] = -1
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int):
        logger.warning("Static shapes are not supported for causal language model.")
        return self

    @property
    def normalized_config(self):
        logger.warning(
            "access to normalized_config attribute is deprecated and will be removed in future versions, please use config"
        )
        return NormalizedConfigManager.get_normalized_config_class(self.config.model_type)(self.config)

    def compile(self):
        if self.request is None:
            if self._compile_only:
                self.request = self.model.create_infer_request()
            super().compile()
            self.request = self.request.create_infer_request()

    def _make_stateful(self):
        patch_stateful(self.config, self.model)
        self.stateful = True


@add_start_docstrings(
    """
    OpenVINO Model with a causal language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForCausalLM(OVBaseDecoderModel, GenerationMixin):
    export_feature = "text-generation"
    auto_model_class = AutoModelForCausalLM

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForCausalLM",
            checkpoint="gpt2",
        )
    )
    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict:
        batch_size = input_ids.shape[0]
        model_transformers_version = get_export_transformers_version(self.model, self.config)
        if self.config.model_type == "bloom" and compare_versions(model_transformers_version, "<", "4.44"):
            batch_size *= self.config.num_attention_heads

        inputs = {}
        if not self.stateful:
            if past_key_values is not None:
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                    self.config.model_type == "falcon" and self.config.new_decoder_architecture
                ):
                    if self._pkv_precision == Type.bf16:
                        # numpy does not support bf16, pretending f16, should change to bf16
                        past_key_values = tuple(
                            Tensor(past_key_value, past_key_value.shape, Type.bf16)
                            for pkv_per_layer in past_key_values
                            for past_key_value in pkv_per_layer
                        )
                    else:
                        # Flatten the past_key_values
                        past_key_values = tuple(
                            past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                        )

                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))

            # Create empty past_key_values for decoder_with_past first generation step
            elif self.use_cache:
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    if self.config.model_type == "chatglm" and not hasattr(self.config, "rope_ratio"):
                        shape[0] = 0
                        shape[1] = batch_size
                    else:
                        shape[0] = batch_size
                        if shape[2].is_dynamic:
                            shape[2] = 0
                        else:
                            shape[1] = 0
                    inputs[input_name] = Tensor(model_inputs.get_element_type(), [dim.get_length() for dim in shape])
        else:
            # past_key_values are not used explicitly, instead they are handled inside the model
            if past_key_values is None:
                # This is the first iteration in a sequence, reset all states
                if self.request is not None:
                    self.request.reset_state()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
        past_len = self._get_past_length(past_key_values)
        inputs["input_ids"] = input_ids.cpu().numpy()
        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = attention_mask.cpu().numpy()
            else:
                attention_mask = np.ones(
                    (input_ids.shape[0], input_ids.shape[1] + past_len), dtype=inputs["input_ids"].dtype
                )

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = position_ids.cpu().numpy()
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
            )

        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()
        # added as model.generate validates model inputs based on forward signature
        kwargs["token_type_ids"] = token_type_ids

        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **kwargs,
        )

        if self._first_iter_beam_search:
            inputs, duplication_indices = self._deduplicate_inputs(inputs)

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).clone().to(self.device)
        if self.stateful:
            # Need a marker to differentiate the first generate iteration from the others in
            # the first condition at the function beginning above.
            # It should be something that is not None and it should be True when converted to Boolean.
            past_key_values = ((),)
            self._past_length += input_ids.shape[1]

        if not self.stateful:
            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
                past_key_values = tuple(
                    np.copy(self.request.get_tensor(key).data) for key in self.key_value_output_names
                )
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                    self.config.model_type == "falcon" and self.config.new_decoder_architecture
                ):
                    # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
                    past_key_values = tuple(
                        past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
                    )
            else:
                past_key_values = None

        if self._first_iter_beam_search:
            logits, past_key_values = self._expand_outputs_for_generation(duplication_indices, logits, past_key_values)
            self._first_iter_beam_search = False

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    # Adapted from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        attention_mask = kwargs.get("attention_mask", None)
        use_cache = kwargs.get("use_cache", None)

        if past_key_values is not None:
            past_len = self._get_past_length(past_key_values)
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_len < input_ids.shape[1]:
                input_ids = input_ids[:, past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None and "position_ids" in self.input_names:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs, model_kwargs=model_kwargs, is_encoder_decoder=is_encoder_decoder, **kwargs
        )

        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat([position_ids, new_position_id], dim=-1)
        return model_kwargs

    def _expand_outputs_for_generation(self, indicies, logits: torch.Tensor, past_key_values: Tuple):
        batch_size = logits.shape[0]
        if indicies.shape[0] != 1:
            logits = logits[indicies]
            if past_key_values and not self.stateful:
                if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                    self.config.model_type == "falcon" and self.config.new_decoder_architecture
                ):
                    past_key_values = tuple(
                        tuple(
                            (
                                past_state[indicies]
                                if not (self.config.model_type == "chatglm" and not hasattr(self.config, "rope_ratio"))
                                else past_state[:, indicies, ...]
                            )
                            for past_state in layer_past
                        )
                        for layer_past in past_key_values
                    )
                else:
                    past_key_values = tuple([past_state[indicies] for past_state in past_key_values])
        if self.stateful:
            self.next_beam_idx = (
                self.next_beam_idx[indicies]
                if self.next_beam_idx is not None
                else np.arange(batch_size, dtype=int)[indicies]
            )
            self._second_iter_beam_search = True
        return logits, past_key_values

    def _deduplicate_inputs(self, model_inputs: Dict):
        input_ids = model_inputs["input_ids"]
        upd_model_inputs = {}
        unique_input_ids, indicies, reverse_indicies = np.unique(
            input_ids, axis=0, return_index=True, return_inverse=True
        )
        export_transformers_version = get_export_transformers_version(self.model, self.config)
        for input_name, input_tensor in model_inputs.items():
            if input_name not in ["input_ids", "beam_idx"]:
                if input_name not in self.key_value_input_names:
                    upd_model_inputs[input_name] = input_tensor[indicies]
                else:
                    shape = input_tensor.shape if isinstance(input_tensor, Tensor) else list(input_tensor.shape)
                    dtype = input_tensor.element_type if isinstance(input_tensor, Tensor) else Type(input_tensor.dtype)
                    upd_batch_size = indicies.shape[0]
                    if self.config.model_type == "bloom" and compare_versions(
                        export_transformers_version, "<", "4.44"
                    ):
                        upd_batch_size *= self.config.num_attention_heads
                    shape[
                        (
                            0
                            if not (self.config.model_type == "chatglm" and not hasattr(self.config, "rope_ratio"))
                            else 1
                        )
                    ] = upd_batch_size
                    upd_model_inputs[input_name] = Tensor(dtype, shape)
        upd_model_inputs["input_ids"] = unique_input_ids
        if "beam_idx" in model_inputs:
            beam_range = (
                unique_input_ids.shape[0] * self.config.num_attention_heads
                if (self.config.model_type == "bloom" and compare_versions(export_transformers_version, "<", "4.44"))
                else unique_input_ids.shape[0]
            )
            beam_idx = np.arange(beam_range, dtype=int)
            upd_model_inputs["beam_idx"] = beam_idx
        return upd_model_inputs, reverse_indicies

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if is_transformers_version(">=", "4.39.0"):
            _generation_config, _ = self._prepare_generation_config(generation_config, **kwargs)
            generation_mode = _generation_config.get_generation_mode(assistant_model)
        else:
            _generation_config = generation_config or self.generation_config
            generation_mode = self._get_generation_mode(_generation_config, assistant_model)

        is_beam_search = generation_mode in [
            GenerationMode.BEAM_SEARCH,
            GenerationMode.BEAM_SAMPLE,
            GenerationMode.GROUP_BEAM_SEARCH,
            GenerationMode.CONSTRAINED_BEAM_SEARCH,
        ]
        if is_beam_search:
            self._first_iter_beam_search = True
        result = super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            negative_prompt_ids,
            negative_prompt_attention_mask,
            **kwargs,
        )
        return result

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        if self.stateful:
            return self._past_length
        if self.config.model_type in MULTI_QUERY_ATTN_MODELS and not (
            self.config.model_type == "falcon" and self.config.new_decoder_architecture
        ):
            return past_key_values[0].shape[-2]
        seq_length_dim = -2
        if self.config.model_type == "chatglm" and not hasattr(self.config, "rope_ratio"):
            seq_length_dim = 0
        elif self.config.model_type == "qwen":
            seq_length_dim = 1
        # input is tuple of pairs
        if isinstance(past_key_values[0], (tuple, list)):
            return past_key_values[0][1].shape[seq_length_dim]
        # past key values comes after flattening
        return past_key_values[1].shape[seq_length_dim]

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.stateful:
            # TODO: Apply it differently based on model type
            # TODO: At least for bloom we need to replicate values for each attention head
            self.next_beam_idx = (
                np.array(beam_idx) if not self._second_iter_beam_search else self.next_beam_idx
            )  # save beam_idx to be used as an input in the next iteration
            self._second_iter_beam_search = False
            return past_key_values
        else:
            if self.config.model_type not in MULTI_QUERY_ATTN_MODELS or (
                self.config.model_type == "falcon" and self.config.new_decoder_architecture
            ):
                return tuple(
                    tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past)
                    for layer_past in past_key_values
                )
            return tuple(np.take(past_state, beam_idx, 0) for past_state in past_key_values)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        compile_only: bool = False,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        generation_config = kwargs.pop("generation_config", None)
        model_path = Path(model_id)
        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME
        file_name = file_name or default_file_name

        model_cache_path = cls._cached_file(
            model_path=model_path,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            local_files_only=local_files_only,
        )

        if not compile_only:
            model = cls.load_model(model_cache_path)
        else:
            model = cls._compile_model(
                model_cache_path, kwargs.get("device", "CPU"), kwargs.get("ov_config"), model_cache_path.parent
            )

        model_type = config.model_type
        export_transformers_version = get_export_transformers_version(model, config)
        if model_type == "bloom" and compare_versions(export_transformers_version, "<", "4.44"):
            init_cls = OVBloomForCausalLM
        elif model_type == "gpt_bigcode":
            init_cls = OVGPTBigCodeForCausalLM
        elif model_type in SSM_MODELS:
            init_cls = OVMambaForCausalLM
        else:
            init_cls = cls

        if isinstance(quantization_config, dict) and quantization_config == {"bits": 4}:
            default_config = get_default_quantization_config(config.name_or_path, weight_format="int4")
            quantization_config = cls._prepare_quantization_config(
                default_config or _DEFAULT_4BIT_WQ_CONFIG, load_in_8bit
            )
            if quantization_config.dataset is not None:
                quantization_config.trust_remote_code = kwargs.get("trust_remote_code", False)
        else:
            quantization_config = cls._prepare_quantization_config(quantization_config, load_in_8bit)
            if isinstance(quantization_config, OVWeightQuantizationConfig) and quantization_config.bits == 4:
                default_config = get_default_quantization_config(config.name_or_path, weight_format="int4")
                if default_config:
                    logger.info(
                        f"For the given model, we recommend the following `quantization_config` : {default_config}"
                    )

        enable_compilation = kwargs.pop("compile", True) and not quantization_config

        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                )
                if getattr(generation_config, "cache_implementation", None) is not None:
                    generation_config.cache_implementation = None
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )

        causal_model = init_cls(
            model=model,
            config=config,
            model_save_dir=model_cache_path.parent,
            compile=enable_compilation,
            compile_only=compile_only,
            quantization_config=quantization_config,
            generation_config=generation_config,
            **kwargs,
        )

        if quantization_config:
            if not is_nncf_available():
                raise ImportError(
                    "Quantization of the weights requires nncf, please install it with `pip install nncf`"
                )

            if compile_only:
                raise ValueError(
                    "quantization is not supported with `compile_only` mode, please initialize model without this option"
                )

            from optimum.intel.openvino.quantization import OVQuantizer

            quantizer = OVQuantizer(causal_model)
            quantization_config_copy = copy.deepcopy(quantization_config)
            quantization_config_copy.tokenizer = quantization_config.tokenizer or model_id
            quantizer.quantize(ov_config=OVConfig(quantization_config=quantization_config_copy))

        return causal_model


class OVBloomForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # only last token for input_ids if past is not None
        if past_key_values and not self.stateful:
            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)
        return super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)

    # Adapted from transformers.models.bloom.modeling_bloom.BloomForCausalLM._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called for bloom architecture.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.stateful:
            batch_size = beam_idx.shape[0]
            beam_idx = np.array(beam_idx) if not self._second_iter_beam_search else self.next_beam_idx
            indices = np.array(range(batch_size * self.config.num_attention_heads))
            indices = indices.reshape([batch_size, self.config.num_attention_heads])
            self.next_beam_idx = np.take(indices, beam_idx, 0).flatten()
            self._second_iter_beam_search = False
            return past_key_values
        else:
            standardized_past = self._convert_to_standard_cache(past_key_values, batch_size=len(beam_idx))
            reordered_past = tuple(
                (
                    np.take(layer_past[0], beam_idx, 0),
                    np.take(layer_past[1], beam_idx, 0),
                )
                for layer_past in standardized_past
            )
            return self._convert_to_bloom_cache(reordered_past)

    # Copied from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_bloom_cache
    @staticmethod
    def _convert_to_bloom_cache(past_key_value: Tuple[Tuple[torch.Tensor]]) -> Tuple[Tuple[torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape((batch_size_times_num_heads, head_dim, seq_length)),
                layer_past[1].reshape((batch_size_times_num_heads, seq_length, head_dim)),
            )
            for layer_past in past_key_value
        )

    # Adapted from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._convert_to_standard_cache
    def _convert_to_standard_cache(
        self, past_key_value: Tuple[Tuple[torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size, num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape((batch_size, num_heads, head_dim, seq_length)),
                layer_past[1].reshape((batch_size, num_heads, seq_length, head_dim)),
            )
            for layer_past in past_key_value
        )

    def _expand_outputs_for_generation(self, indicies, logits: torch.Tensor, past_key_values: Tuple):
        batch_size = logits.shape[0]
        if indicies.shape[0] != 1:
            logits = logits[indicies]
            if past_key_values and not self.stateful:
                pkv_standard = self._convert_to_standard_cache(past_key_values, batch_size)
                pkv = tuple(tuple(past_state[indicies] for past_state in layer_past) for layer_past in pkv_standard)
                past_key_values = self._convert_to_bloom_cache(pkv)

        if self.stateful:
            self.next_beam_idx = (
                self.next_beam_idx[indicies]
                if self.next_beam_idx is not None
                else np.arange(batch_size, dtype=int)[indicies]
            )
        self._second_iter_beam_search = True
        return logits, past_key_values


class OVGPTBigCodeForCausalLM(OVModelForCausalLM):
    # Adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM._reorder_cache
    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        if self.stateful:
            # save beam_idx to be used as an input in the next iteration
            self.next_beam_idx = np.array(beam_idx) if not self._second_iter_beam_search else self.next_beam_idx
            self._second_iter_beam_search = False
            return past_key_values
        else:
            return tuple(np.take(layer_past, beam_idx, 0) for layer_past in past_key_values)


class OVMambaCache(MambaCache):
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.
        max_batch_size (`int`):
            Maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        conv_states (`List[torch.Tensor]`):
            A list of convolutional state tensors for each layer, used to cache intermediate state for the convolution
            component of the Mamba model.
        ssm_states (`List[torch.Tensor]`):
            A list of state-space model (SSM) state tensors for each layer, used to cache intermediate state for the
            SSM component of the Mamba model.

    """

    def __init__(
        self,
        config: "PretrainedConfig",
        batch_size: int = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
        max_batch_size: Optional[int] = None,
        conv_states: Optional[List[torch.Tensor]] = None,
        ssm_states: Optional[List[torch.Tensor]] = None,
    ):
        self.dtype = dtype
        self.max_batch_size = batch_size or max_batch_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if conv_states is not None:
            self.conv_states = conv_states
        else:
            self.conv_states = []
            for _ in range(config.num_hidden_layers):
                conv_state: torch.Tensor = torch.zeros(
                    self.max_batch_size, self.intermediate_size, self.conv_kernel_size, device=self.device, dtype=dtype
                )
                self.conv_states.append(conv_state)

        if ssm_states is not None:
            self.ssm_states = ssm_states
        else:
            self.ssm_states: List[torch.Tensor] = []
            for _ in range(config.num_hidden_layers):
                ssm_state: torch.Tensor = torch.zeros(
                    self.max_batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=self.device,
                    dtype=dtype,
                )
                self.ssm_states.append(ssm_state)


@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of logits at the output layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[OVMambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class OVMambaForCausalLM(OVModelForCausalLM):
    """
    OpenVINO-based causal language model class designed to run models that include Mamba blocks.
    This model assumes a fixed-size Mamba context for sequential computation.
    The context for each mamba block consists of two tensors:
    1. convolutional cache tensor - conv_states: (batch_size, hidden_dim, conv_kernel)
    2. state-space model - ssm_states:  (batch_size, hidden_dim, num_state_features)
    This class supports stateful and stateless inference using OpenVINO.
    """

    def __init__(
        self,
        model: openvino.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            config=config,
            device=device,
            dynamic_shapes=dynamic_shapes,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )

        self.ssm_cache_input_names = [key for key in self.input_names if "cache_params.past.ssm" in key]
        self.conv_cache_input_names = [key for key in self.input_names if "cache_params.past.conv" in key]
        self.ssm_cache_output_names = [key for key in self.output_names if "cache_params.present.ssm" in key]
        self.conv_cache_output_names = [key for key in self.output_names if "cache_params.present.conv" in key]

    @staticmethod
    def _has_cache_inputs(model: openvino.Model) -> bool:
        return any(
            "past_key_values" in key.get_any_name() or "cache_params" in key.get_any_name() for key in model.inputs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_params=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs = {"input_ids": input_ids}
        if "cache_position" in self.input_names:
            if cache_position is None:
                # initialize it as for prefill stage
                cache_position = torch.arange(0, self.config.conv_kernel)
            inputs["cache_position"] = cache_position
        if "attention_mask" in self.input_names:
            if attention_mask is None:
                # during decoding stage it must be a tensor of ones
                attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            inputs["attention_mask"] = attention_mask

        if not self.stateful and self.ssm_cache_input_names and self.conv_cache_input_names:
            if cache_params is None:
                cache_params = OVMambaCache(self.config, input_ids.shape[0])

            ssm_cache = cache_params.ssm_states
            conv_cache = cache_params.conv_states

            inputs.update(zip(self.ssm_cache_input_names, ssm_cache))
            inputs.update(zip(self.conv_cache_input_names, conv_cache))
        else:
            if cache_params is None:
                # this is prefill step, reset all states
                if self.request is not None:
                    self.request.reset_state()
                self._past_length = 0

        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

        if self.stateful:
            self._past_length += input_ids.shape[1]
            num_states = len(self.request.query_state()) // 2
            ssm_states = [None] * num_states
            conv_states = [None] * num_states
            for state in self.request.query_state():
                if "cache_params.past.ssm" in state.name:
                    idx = int(state.name.rsplit(".", 1)[-1])
                    ssm_states[idx] = state.state.data
                elif "cache_params.past.conv" in state.name:
                    idx = int(state.name.rsplit(".", 1)[-1])
                    conv_states[idx] = state.state.data
        else:
            ssm_states = [self.request.get_tensor(key).data for key in self.ssm_cache_output_names]
            conv_states = [self.request.get_tensor(key).data for key in self.conv_cache_output_names]
        cache_params = OVMambaCache(self.config, input_ids.shape[0], conv_states=conv_states, ssm_states=ssm_states)

        return MambaOutput(logits=logits, cache_params=cache_params)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], num_new_tokens: int = 1, **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`

        if self.use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                # decoding stage so it takes the last token
                input_ids = input_ids[:, -1].unsqueeze(-1)
                # models like Mamba typically do not require an attention_mask
                # for the decoding step after the first token so use attention mask of ones
                attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
