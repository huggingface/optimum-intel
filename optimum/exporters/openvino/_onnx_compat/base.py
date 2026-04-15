# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""ONNX configuration base classes."""

from __future__ import annotations

import enum
import gc
import inspect
import itertools
import os
import re
from abc import ABC
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from transformers.utils import is_accelerate_available, is_torch_available


if is_torch_available():
    import torch.nn as nn

from optimum.exporters.base import ExporterConfig
from optimum.exporters.openvino._onnx_compat.constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from optimum.exporters.openvino._onnx_compat.model_patcher import ModelPatcher
from optimum.utils import DEFAULT_DUMMY_SHAPES, DummyInputGenerator, DummySeq2SeqPastKeyValuesGenerator, logging
from optimum.utils.doc import add_dynamic_docstring
from optimum.utils.import_utils import (
    is_onnx_available,
    is_onnxruntime_available,
    is_transformers_version,
)


if is_accelerate_available():
    from accelerate.utils import find_tied_parameters

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from optimum.exporters.openvino._onnx_compat.model_patcher import PatchingSpec


logger = logging.get_logger(__name__)


GENERATE_DUMMY_DOCSTRING = r"""
        Generates the dummy inputs necessary for tracing the model. If not explicitly specified, default input shapes are used.

        Args:
            framework (`str`, defaults to `"pt"`):
                The framework for which to create the dummy inputs.
            batch_size (`int`, defaults to {batch_size}):
                The batch size to use in the dummy inputs.
            sequence_length (`int`, defaults to {sequence_length}):
                The sequence length to use in the dummy inputs.
            num_choices (`int`, defaults to {num_choices}):
                The number of candidate answers provided for multiple choice task.
            image_width (`int`, defaults to {width}):
                The width to use in the dummy inputs for vision tasks.
            image_height (`int`, defaults to {height}):
                The height to use in the dummy inputs for vision tasks.
            num_channels (`int`, defaults to {num_channels}):
                The number of channels to use in the dummpy inputs for vision tasks.
            feature_size (`int`, defaults to {feature_size}):
                The number of features to use in the dummpy inputs for audio tasks in case it is not raw audio.
                This is for example the number of STFT bins or MEL bins.
            nb_max_frames (`int`, defaults to {nb_max_frames}):
                The number of frames to use in the dummpy inputs for audio tasks in case the input is not raw audio.
            audio_sequence_length (`int`, defaults to {audio_sequence_length}):
                The number of frames to use in the dummpy inputs for audio tasks in case the input is raw audio.

        Returns:
            `Dict`: A dictionary mapping the input names to dummy tensors in the proper framework format.
"""


class OnnxConfig(ExporterConfig, ABC):
    DEFAULT_ONNX_OPSET = 18
    VARIANTS: ClassVar[dict[str, str]] = {"default": "The default ONNX variant."}
    DEFAULT_VARIANT = "default"
    PATCHING_SPECS: list[PatchingSpec] | None = None
    _MODEL_PATCHER = ModelPatcher

    _TASK_TO_COMMON_OUTPUTS = {  # noqa: RUF012
        "audio-classification": OrderedDict({"logits": {0: "batch_size"}}),
        "audio-frame-classification": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "automatic-speech-recognition": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "audio-xvector": OrderedDict({"logits": {0: "batch_size"}, "embeddings": {0: "batch_size"}}),
        "depth-estimation": OrderedDict({"predicted_depth": {0: "batch_size", 1: "height", 2: "width"}}),
        "document-question-answering": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "feature-extraction": OrderedDict({"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}),
        "fill-mask": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "image-classification": OrderedDict({"logits": {0: "batch_size"}}),
        "image-segmentation": OrderedDict({"logits": {0: "batch_size", 2: "height", 3: "width"}}),
        "image-to-text": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "image-to-image": OrderedDict(
            {"reconstruction": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}
        ),
        "keypoint-detection": OrderedDict(
            {"heatmaps": {0: "batch_size", 1: "num_keypoints", 2: "height", 3: "width"}}
        ),
        "mask-generation": OrderedDict({"logits": {0: "batch_size"}}),
        "masked-im": OrderedDict(
            {"reconstruction" if is_transformers_version(">=", "4.29.0") else "logits": {0: "batch_size"}}
        ),
        "multiple-choice": OrderedDict({"logits": {0: "batch_size", 1: "num_choices"}}),
        "object-detection": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "num_queries"},
                "pred_boxes": {0: "batch_size", 1: "num_queries"},
            }
        ),
        "question-answering": OrderedDict(
            {
                "start_logits": {0: "batch_size", 1: "sequence_length"},
                "end_logits": {0: "batch_size", 1: "sequence_length"},
            }
        ),
        "semantic-segmentation": OrderedDict({"logits": {0: "batch_size", 1: "num_labels", 2: "height", 3: "width"}}),
        "text2text-generation": OrderedDict({"logits": {0: "batch_size", 1: "decoder_sequence_length"}}),
        "text-classification": OrderedDict({"logits": {0: "batch_size"}}),
        "text-generation": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "time-series-forecasting": OrderedDict({"prediction_outputs": {0: "batch_size"}}),
        "token-classification": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "visual-question-answering": OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}}),
        "zero-shot-image-classification": OrderedDict(
            {
                "logits_per_image": {0: "image_batch_size", 1: "text_batch_size"},
                "logits_per_text": {0: "text_batch_size", 1: "image_batch_size"},
                "text_embeds": {0: "text_batch_size"},
                "image_embeds": {0: "image_batch_size"},
            }
        ),
        "zero-shot-object-detection": OrderedDict(
            {
                "logits": {0: "batch_size", 1: "num_queries"},
                "pred_boxes": {0: "batch_size", 1: "num_queries"},
                "text_embeds": {0: "text_batch_size"},
                "image_embeds": {0: "image_batch_size"},
            }
        ),
    }

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        preprocessors: list[Any] | None = None,
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        super().__init__(config=config, task=task, int_dtype=int_dtype, float_dtype=float_dtype)

        self.variant = "default"
        self._preprocessors = preprocessors

    @property
    def variant(self) -> str:
        """For a given ONNX config, the variant of the model to export.

        This property allows to define variants of a given model, in case
        different users would like to export the model differently (with different inputs/outputs, model split in several ONNX or not, etc.).
        """
        return self._variant

    @variant.setter
    def variant(self, value: str):
        if value == "default" and hasattr(self, "DEFAULT_VARIANT"):
            value = self.DEFAULT_VARIANT
        if value not in self.VARIANTS:
            raise ValueError(f"The variant {value} is not supported for the ONNX config {self.__class__.__name__}.")
        self._variant = value

    def fix_dynamic_axes(
        self, model_path: Path, device: str = "cpu", dtype: str | None = None, input_shapes: dict | None = None
    ):
        """Fixes potential issues with dynamic axes.

        During the export, ONNX will infer some axes to be dynamic which are actually static. This method is called
        right after the export to fix such issues.

        Args:
            model_path (`Path`):
                The path of the freshly exported ONNX model.
            device (`str`, defaults to `"cpu"`):
                The device on which the model will be run. This is used to determine the ONNX Runtime provider.
            dtype (`Optional[str]`, defaults to `None`):
                The data type of the model inputs. If `None`, it will be inferred from the model inputs.
            input_shapes (`Optional[Dict[str, Any]]`, defaults to `None`):
                The shapes of the model inputs. If `None`, it will be inferred from the model inputs.
        """
        if not (is_onnx_available() and is_onnxruntime_available()):
            raise RuntimeError(
                "The onnx and onnxruntime packages are necessary to fix the dynamic shapes of the exported model. "
                "You can install them by doing: pip install onnx onnxruntime"
            )

        from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

        import onnx

        allowed_dynamic_axes = set()
        for input_ in self.inputs.values():
            allowed_dynamic_axes |= set(input_.values())
        for output in self.outputs.values():
            allowed_dynamic_axes |= set(output.values())

        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        session_options = SessionOptions()
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL  # no need to optimize here
        session = InferenceSession(model_path.as_posix(), providers=providers, sess_options=session_options)

        to_fix = []
        for output_idx, node in enumerate(session.get_outputs()):
            for idx, axis in enumerate(node.shape):
                if isinstance(axis, str) and axis not in allowed_dynamic_axes:
                    to_fix.append((output_idx, idx))

        # We branch here to avoid doing an unnecessary forward pass.
        if to_fix:
            if input_shapes is None:
                input_shapes = {}

            onnx_input_names = [inp.name for inp in session.get_inputs()]
            dummy_inputs = self.generate_dummy_inputs(framework="np", **input_shapes)
            dummy_inputs = self.generate_dummy_inputs_for_validation(dummy_inputs, onnx_input_names)
            dummy_inputs = self.rename_ambiguous_inputs(dummy_inputs)

            onnx_inputs = {}
            for name, value in dummy_inputs.items():
                if isinstance(value, (list, tuple)):
                    value = self.flatten_output_collection_property(name, value)
                    onnx_inputs.update(dict(value.items()))
                else:
                    onnx_inputs[name] = value

            for name, value in onnx_inputs.items():
                if value.dtype == np.float32 and dtype == "fp16":
                    onnx_inputs[name] = onnx_inputs[name].astype(np.float16)

            outputs = session.run(None, onnx_inputs)
            del session

            onnx_model = onnx.load(model_path.as_posix(), load_external_data=False)

            for output_idx, dim_idx in to_fix:
                dims = onnx_model.graph.output[output_idx].type.tensor_type.shape.dim
                dims[dim_idx].dim_value = outputs[output_idx].shape[dim_idx]

            onnx.save(
                onnx_model,
                model_path.as_posix(),
                convert_attribute=True,
            )
            del onnx_model
            gc.collect()

    @property
    def torch_to_onnx_input_map(self) -> dict[str, str]:
        """Dictionary mapping input names from the PyTorch model to input names from the exported ONNX model.

        Override the function when the input names and the exported ONNX input names are different.

        Returns:
            `Dict[str, str]`: A dictionary mapping the PyTorch model input names to the exported ONNX model input names.
        """
        return {}

    @property
    def torch_to_onnx_output_map(self) -> dict[str, str]:
        """Dictionary mapping output names from the PyTorch model to output names from the exported ONNX model.

        Override the function when the output names and the exported ONNX output names are different.

        Returns:
            `Dict[str, str]`: A dictionary mapping the PyTorch model output names to the exported ONNX model output names.
        """
        return {}

    def rename_ambiguous_inputs(self, inputs) -> dict[str, dict[int, str]]:
        """Updates the input names of the model to export.

        Override the function when the model input names are ambiguous or too generic.

        Returns:
            `Dict[str, Dict[int, str]]`: Updated inputs.
        """
        return inputs

    def ordered_inputs(self, model: PreTrainedModel) -> dict[str, dict[int, str]]:
        """Re-orders the inputs using the model forward pass signature.

        Args:
            model ([`transformers.PreTrainedModel`]):
                The model for which we will use the OnnxConfig.

        Returns:
            `Dict[str, Dict[int, str]]`: The properly ordered inputs.
        """
        inputs = self.inputs
        inputs = self.rename_ambiguous_inputs(inputs)

        ordered_inputs = {}
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
        else:
            sig = inspect.signature(model.call)

        for param in sig.parameters:
            param_regex = re.compile(rf"{param}(\..*)?$")
            to_insert = []
            for name, dynamic_axes in inputs.items():
                if re.match(param_regex, name):
                    to_insert.append((name, dynamic_axes))
            # TODO: figure out a smart way of re-ordering potential nested structures.
            # to_insert = sorted(to_insert, key=lambda t: t[0])
            for name, dynamic_axes in to_insert:
                name = self.torch_to_onnx_input_map.get(name, name)
                ordered_inputs[name] = dynamic_axes
        return ordered_inputs

    @classmethod
    def flatten_output_collection_property(cls, name: str, field: Iterable[Any]) -> dict[str, Any]:
        """Flattens any potential nested structure expanding the name of the field with the index of the element within the structure.

        Args:
            name (`str`):
                The name of the nested structure.
            field (`Iterable[Any]`):
                The structure to potentially flattened.

        Returns:
            `Dict[str, Any]`: Outputs with flattened structure and key mapping this new structure.

        """
        if isinstance(field[0], (list, tuple)):
            return {f"{name}.{idx}": item for idx, item in enumerate(itertools.chain.from_iterable(field))}
        else:
            return {f"{name}.{idx}": item for idx, item in enumerate(field)}

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: dict[str, Any], onnx_input_names: list[str]
    ) -> dict[str, Any]:
        """Generates inputs for ONNX Runtime using the reference model inputs.

        Override this to run inference with seq2seq
        models which have the encoder and decoder exported as separate ONNX files.

        Args:
            reference_model_inputs (`Dict[str, Tensor]`):
                Reference inputs for the model.
            onnx_input_names (`Optional[List[str]]`, defaults to `None`):
                Names of the actual inputs to the ONNX model. This argument may be required as an unused
                input to the model is automatically removed by torch.onnx.export (e.g. encoder_outputs in the decoder with past)

        Returns:
            `Dict[str, Tensor]`: The mapping holding the kwargs to provide to the model's forward function
        """
        return reference_model_inputs

    def post_process_exported_models(
        self,
        path: Path,
        models_and_onnx_configs: dict[str, tuple[PreTrainedModel, OnnxConfig]],
        onnx_files_subpaths: list[str],
    ):
        """Performs any model-specific post-processing on the exported models."""
        return models_and_onnx_configs, onnx_files_subpaths

    def patch_model_for_export(
        self, model: PreTrainedModel, model_kwargs: dict[str, Any] | None = None
    ) -> ModelPatcher:
        return self._MODEL_PATCHER(self, model, model_kwargs=model_kwargs)


class OnnxConfigWithPast(OnnxConfig, ABC):
    """Inherits from [`~exporters.onnx.OnnxConfig`]. A base class to handle the ONNX configuration of decoder-only models."""

    PAD_ATTENTION_MASK_TO_PAST: bool = False
    SUPPORTS_PAST: bool = True

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        preprocessors: list[Any] | None = None,
    ):
        self.use_past = use_past
        self.use_past_in_inputs = use_past_in_inputs

        self.is_merged = False
        self.use_cache_branch = None
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if not self.use_past_in_inputs:
            common_outputs = super().outputs
        # In the other cases, the sequence_length axis is not dynamic, always of length 1
        elif self.task == "feature-extraction":
            common_outputs = OrderedDict({"last_hidden_state": {0: "batch_size"}})
        else:
            common_outputs = OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}})
        if self.use_past:
            # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
            self.add_past_key_values(common_outputs, direction="outputs")
        return common_outputs

    @property
    def values_override(self) -> dict[str, Any] | None:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}

    @add_dynamic_docstring(text=GENERATE_DUMMY_DOCSTRING, dynamic_elements=DEFAULT_DUMMY_SHAPES)
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

        dummy_inputs = {}
        input_names = [key for key in self.inputs if not key.startswith("past_key_values")]
        if self.use_past_in_inputs and self.use_cache_branch is not False:
            input_names.append("past_key_values")

        for input_name in input_names:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = self.overwrite_shape_and_generate_input(
                        dummy_input_gen,
                        input_name,
                        framework,
                        input_shapes=kwargs,
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                )

        # refer to https://github.com/huggingface/optimum/pull/764
        if (
            self.use_past_in_inputs
            and self.PAD_ATTENTION_MASK_TO_PAST
            and self.use_cache_branch is not False
            and "attention_mask" in dummy_inputs
            and self.task == "text-generation"
        ):
            seq_len = dummy_inputs["input_ids"].shape[1]
            past_seq_len = dummy_inputs["past_key_values"][0][1].shape[-2]
            dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                dummy_inputs["attention_mask"], desired_length=past_seq_len + seq_len, dim=1
            )

        return dummy_inputs

    def overwrite_shape_and_generate_input(
        self, dummy_input_gen: DummyInputGenerator, input_name: str, framework: str, input_shapes: dict
    ):
        """The shape passed to the dummy input generator may not always be correct for all of the inputs it manages.

        This method allows
        to overwrite some shapes, and generate the dummy input. This should probably be refactored more elegantly.
        """
        # models from TextSeq2SeqOnnxConfig use decoder_input_ids as input name
        # while models from TextDecoderOnnxConfig use input_ids, hence the check for both

        # NOTE: The check `self.task != "text-generation" is added following the use of a single ONNX for both without/with KV cache, without subgraphs.
        # This overwrite may be moved to OnnxSeq2SeqConfigWithPast, but I am afraid it would break encoder-decoder models.
        if (
            self.use_past
            and self.use_past_in_inputs
            and self.use_cache_branch is not False
            and input_name in ["decoder_input_ids", "input_ids", "position_ids"]
            and self.task != "text-generation"
        ):
            sequence_length = dummy_input_gen.sequence_length
            # Use a sequence length of 1 when the KV cache is already populated.
            dummy_input_gen.sequence_length = 1
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )
            dummy_input_gen.sequence_length = sequence_length
        else:
            dummy_input = dummy_input_gen.generate(
                input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )

        return dummy_input

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        """Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.

        Args:
            inputs_or_outputs (`Dict[str, Dict[int, str]]`):
                The mapping to fill.
            direction (`str`):
                either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                output mapping, this is important for axes naming.
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: decoder_sequence_name}

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> dict[str, Any]:
        flattened_output = {}
        if name in ["present", "past_key_values"]:
            for idx, t in enumerate(field):
                self.flatten_past_key_values(flattened_output, name, idx, t)
        else:
            flattened_output = super().flatten_output_collection_property(name, field)

        return flattened_output

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: dict[str, Any], onnx_input_names: list[str]
    ) -> dict[str, Any]:
        if self.is_merged is True and self.use_cache_branch is True:
            reference_model_inputs["use_cache_branch"] = DummyInputGenerator.constant_tensor(shape=[1], value=True)
        elif self.is_merged is True and self.use_cache_branch is False:
            reference_model_inputs["use_cache_branch"] = DummyInputGenerator.constant_tensor(shape=[1], value=False)

            # We don't support optional inputs for now, so even though the non-cache branch is used,
            # dummy past key values are necessary
            batch_size = reference_model_inputs["input_ids"].shape[0]
            pkv_generator = self.DUMMY_PKV_GENERATOR_CLASS(
                task=self.task, normalized_config=self._normalized_config, sequence_length=1, batch_size=batch_size
            )
            reference_model_inputs["past_key_values"] = pkv_generator.generate(
                "past_key_values", framework="pt", int_dtype=self.int_dtype, float_dtype=self.float_dtype
            )

        return super().generate_dummy_inputs_for_validation(reference_model_inputs, onnx_input_names)


class ConfigBehavior(str, enum.Enum):
    """Specifies the behavior of the [`~exporters.onnx.base.OnnxSeq2SeqConfigWithPast`].

    - MONOLITH: the config can be used to export the whole seq2seq model as a single file.
    - ENCODER: the config can be used to export the encoder part of the seq2seq model.
    - DECODER: the config can be used to export the decoder part of the seq2seq model.
    """

    MONOLITH = "monolith"
    ENCODER = "encoder"
    DECODER = "decoder"


class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
    """Inherits from [`~exporters.onnx.OnnxConfigWithPast`]. A base class to handle the ONNX configuration of encoder-decoder models."""

    DUMMY_PKV_GENERATOR_CLASS = DummySeq2SeqPastKeyValuesGenerator

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        behavior: ConfigBehavior = ConfigBehavior.MONOLITH,
        preprocessors: list[Any] | None = None,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            preprocessors=preprocessors,
        )
        self._behavior = behavior

        if self._behavior is ConfigBehavior.ENCODER:
            self.task = "feature-extraction"
            self.use_past_in_inputs = False

    def with_behavior(
        self,
        behavior: str | ConfigBehavior,
        use_past: bool = False,
        use_past_in_inputs: bool = False,
    ) -> OnnxSeq2SeqConfigWithPast:
        """Creates a copy of the current OnnxConfig but with a different `ConfigBehavior` and `use_past` value.

        Args:
            behavior ([`ConfigBehavior`]):
                The behavior to use for the new instance.
            use_past (`bool`, defaults to `False`):
                Whether or not the ONNX config to instantiate is for a model using KV cache.
            use_past_in_inputs (`bool`, defaults to `False`):
                Whether the KV cache is to be passed as an input to the ONNX.

        Returns:
            `OnnxSeq2SeqConfigWithPast`
        """
        if isinstance(behavior, str) and not isinstance(behavior, ConfigBehavior):
            behavior = ConfigBehavior(behavior)

        onnx_config = self.__class__(
            self._config,
            task=self.task,
            int_dtype=self.int_dtype,
            float_dtype=self.float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            behavior=behavior,
            preprocessors=self._preprocessors,
        )
        onnx_config.variant = self.variant
        return onnx_config

    @property
    def torch_to_onnx_input_map(self) -> dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "decoder_attention_mask": "attention_mask",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        # Renaming the outputs axes properly.
        for name, axes_names in common_outputs.items():
            if self._behavior is ConfigBehavior.ENCODER or "encoder" in name:
                sequence_name = "encoder_sequence_length"
            else:
                sequence_name = "decoder_sequence_length"

            new_axes_names = {}
            for axis_idx, axis_name in axes_names.items():
                if "sequence" in axis_name:
                    if self.use_past_in_inputs is False or self.is_merged is True:
                        new_axes_names[axis_idx] = sequence_name
                    else:
                        # Trick to force it since ONNX sometimes infer a dynamic axis where it's not.
                        new_axes_names[axis_idx] = "1"
                else:
                    new_axes_names[axis_idx] = axis_name
            common_outputs[name] = new_axes_names

        if self.use_past:
            # When exporting decoder models with use_cache=True, both the decoder without past and with past have the KV cache as an output.
            self.add_past_key_values(common_outputs, direction="outputs")

        return common_outputs

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_decoder_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_decoder_sequence_length + decoder_sequence_length"
            name = "present"

        for i in range(self._normalized_config.decoder_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch_size", 2: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch_size", 2: decoder_sequence_name}

            if (
                self.is_merged is True
                or (self._behavior is ConfigBehavior.DECODER and not self.use_past_in_inputs)
                or direction == "inputs"
            ):
                inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch_size", 2: "encoder_sequence_length"}
                inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch_size", 2: "encoder_sequence_length"}

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        if len(t) not in [2, 4]:
            raise ValueError(
                "past_key_values to flatten should be of length 2 (self-attention only) or 4 (self and cross attention)."
            )

        flattened_output[f"{name}.{idx}.decoder.key"] = t[0]
        flattened_output[f"{name}.{idx}.decoder.value"] = t[1]
        if len(t) == 4:
            flattened_output[f"{name}.{idx}.encoder.key"] = t[2]
            flattened_output[f"{name}.{idx}.encoder.value"] = t[3]

    def post_process_exported_models(
        self,
        path: Path,
        models_and_onnx_configs: dict[str, tuple[PreTrainedModel, OnnxConfig]],
        onnx_files_subpaths: list[str],
    ):
        models_and_onnx_configs, onnx_files_subpaths = super().post_process_exported_models(
            path, models_and_onnx_configs, onnx_files_subpaths
        )
        return models_and_onnx_configs, onnx_files_subpaths

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)

        if (
            self.use_past_in_inputs
            and self.PAD_ATTENTION_MASK_TO_PAST
            and self.use_cache_branch is not False
            and "decoder_attention_mask" in dummy_inputs
        ):
            seq_len = dummy_inputs["decoder_input_ids"].shape[1]
            past_seq_len = dummy_inputs["past_key_values"][0][1].shape[-2]
            dummy_inputs["decoder_attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                dummy_inputs["decoder_attention_mask"], desired_length=past_seq_len + seq_len, dim=1
            )

        return dummy_inputs

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: dict[str, Any], onnx_input_names: list[str]
    ) -> dict[str, Any]:
        if self._behavior is ConfigBehavior.DECODER:
            if "decoder_input_ids" in reference_model_inputs:
                reference_model_inputs["input_ids"] = reference_model_inputs.pop("decoder_input_ids")
            if "attention_mask" in reference_model_inputs:
                reference_model_inputs["encoder_attention_mask"] = reference_model_inputs.pop("attention_mask")
            if "decoder_attention_mask" in reference_model_inputs:
                reference_model_inputs["attention_mask"] = reference_model_inputs.pop("decoder_attention_mask")
            if "encoder_outputs" in reference_model_inputs:
                if "encoder_hidden_states" in onnx_input_names:
                    reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]
                else:
                    reference_model_inputs.pop("encoder_outputs")

        return super().generate_dummy_inputs_for_validation(reference_model_inputs, onnx_input_names)
