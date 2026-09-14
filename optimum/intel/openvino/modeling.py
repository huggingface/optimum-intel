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
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import openvino
import torch
import transformers
from huggingface_hub import model_info
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForZeroShotImageClassification,
    PretrainedConfig,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    ImageClassifierOutput,
    MaskedLMOutput,
    ModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    XVectorOutput,
)
from transformers.models.clip.modeling_clip import CLIPOutput

from ...exporters.openvino.stateful import model_has_state
from ..utils.import_utils import is_timm_available, is_timm_version
from .configuration import OVQuantizationConfigBase
from .modeling_base import OVBaseModel
from .modeling_sam import OVSamModel
from .utils import (
    _is_timm_ov_dir,
    ensure_numpy,
)


logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"
_FEATURE_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

MODEL_START_DOCSTRING = r"""
    This model inherits from [`optimum.intel.openvino.modeling.OVBaseModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        model (`openvino.Model`): is the main class used to run OpenVINO Runtime inference.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~intel.openvino.modeling.OVBaseModel.from_pretrained`] method to load the model weights.
        device (`str`, defaults to `"CPU"`):
            The device type for which the model will be optimized for. The resulting compiled model will contains nodes specific to this device.
        dynamic_shapes (`bool`, defaults to `True`):
            All the model's dimension will be set to dynamic when set to `True`. Should be set to `False` for the model to not be dynamically reshaped by default.
        ov_config (`Optional[Dict]`, defaults to `None`):
            The dictionary containing the information related to the model compilation.
        compile (`bool`, defaults to `True`):
            Disable the model compilation during the loading step when set to `False`.
            Can be useful to avoid unnecessary compilation, in the case where the model needs to be statically reshaped, the device modified or if FP16 conversion is enabled.
"""

INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`torch.Tensor`), *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        token_type_ids (`torch.Tensor`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 1 for tokens that are **sentence A**,
            - 0 for tokens that are **sentence B**.
            [What are token type IDs?](https://huggingface.co/docs/transformers/glossary#token-type-ids)
"""

IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.Tensor`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoFeatureExtractor`](https://huggingface.co/docs/transformers/autoclass_tutorial#autofeatureextractor).
"""

AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.Tensor` of shape `({0})`):
            Float values of input raw speech waveform..
            Input values can be obtained from audio file loaded into an array using [`AutoFeatureExtractor`](https://huggingface.co/docs/transformers/autoclass_tutorial#autofeatureextractor).
"""


class OVModel(OVBaseModel):
    base_model_prefix = "openvino_model"
    auto_model_class = AutoModel

    def __init__(self, model: openvino.Model, config: transformers.PretrainedConfig = None, **kwargs):
        super().__init__(model, config, **kwargs)
        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example of sequence classification using `transformers.pipeline`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    >>> outputs = pipe("Hello, my dog is cute")
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a SequenceClassifierOutput for sequence classification tasks.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForSequenceClassification(OVModel):
    export_feature = "text-classification"
    auto_model_class = AutoModelForSequenceClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForSequenceClassification",
            checkpoint="distilbert-base-uncased-finetuned-sst-2-english",
        )
    )
    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        attention_mask: Union[torch.Tensor, np.ndarray],
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(input_ids, np.ndarray)

        input_ids = ensure_numpy(input_ids)
        attention_mask = ensure_numpy(attention_mask)
        token_type_ids = ensure_numpy(token_type_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add the token_type_ids when needed
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids if token_type_ids is not None else np.zeros_like(input_ids)

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]
        return SequenceClassifierOutput(logits=logits)


QUESTION_ANSWERING_EXAMPLE = r"""
    Example of question answering using `transformers.pipeline`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> outputs = pipe(question, text)
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a QuestionAnsweringModelOutput for extractive question-answering tasks.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForQuestionAnswering(OVModel):
    export_feature = "question-answering"
    auto_model_class = AutoModelForQuestionAnswering

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForQuestionAnswering",
            checkpoint="distilbert-base-cased-distilled-squad",
        )
    )
    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        attention_mask: Union[torch.Tensor, np.ndarray],
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(input_ids, np.ndarray)

        input_ids = ensure_numpy(input_ids)
        attention_mask = ensure_numpy(attention_mask)
        token_type_ids = ensure_numpy(token_type_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add the token_type_ids when needed
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids if token_type_ids is not None else np.zeros_like(input_ids)

        outputs = self._inference(inputs)
        start_logits = (
            torch.from_numpy(outputs["start_logits"]).to(self.device) if not np_inputs else outputs["start_logits"]
        )
        end_logits = (
            torch.from_numpy(outputs["end_logits"]).to(self.device) if not np_inputs else outputs["end_logits"]
        )
        return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)


TOKEN_CLASSIFICATION_EXAMPLE = r"""
    Example of token classification using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
    >>> outputs = pipe("My Name is Peter and I live in New York.")
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a TokenClassifierOutput for token classification tasks.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForTokenClassification(OVModel):
    export_feature = "token-classification"
    auto_model_class = AutoModelForTokenClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForTokenClassification",
            checkpoint="dslim/bert-base-NER",
        )
    )
    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        attention_mask: Union[torch.Tensor, np.ndarray],
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(input_ids, np.ndarray)

        input_ids = ensure_numpy(input_ids)
        attention_mask = ensure_numpy(attention_mask)
        token_type_ids = ensure_numpy(token_type_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add the token_type_ids when needed
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids if token_type_ids is not None else np.zeros_like(input_ids)

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]
        return TokenClassifierOutput(logits=logits)


FEATURE_EXTRACTION_EXAMPLE = r"""
    Example of feature extraction using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    >>> outputs = pipe("My Name is Peter and I live in New York.")
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a BaseModelOutput for feature extraction tasks.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForFeatureExtraction(OVModel):
    export_feature = "feature-extraction"
    auto_model_class = AutoModel

    def __init__(self, model=None, config=None, **kwargs):
        if {"token_embeddings", "sentence_embedding"}.issubset(
            {name for output in model.outputs for name in output.names}
        ):  # Sentence Transormers outputs
            raise ValueError(
                "This model is a Sentence Transformers model. Please use `OVSentenceTransformer` to load this model."
            )

        # A `feature-extraction-with-past` export keeps its KV cache inside the model, so a growing
        # sequence can be scored token by token instead of being re-read from scratch on every call.
        self.stateful = model_has_state(model)
        super().__init__(model, config, **kwargs)
        self._past_length = 0
        if self.stateful and self._compile_only:
            self.request = self.model.create_infer_request()

    def _reshape(
        self,
        model: openvino.Model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        if not self.stateful:
            return super()._reshape(model, batch_size, sequence_length, height, width)
        # `beam_idx`, added by the stateful export, is 1D and only indexes the batch
        shapes = {}
        for model_input in model.inputs:
            shapes[model_input] = model_input.get_partial_shape()
            shapes[model_input][0] = batch_size
            if model_input.get_any_name() != "beam_idx":
                shapes[model_input][1] = sequence_length
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int, height: int = None, width: int = None):
        if self.stateful and sequence_length != -1:
            logger.warning(
                "Static sequence length is not supported for a model with a KV cache, the request will be ignored."
            )
            return self
        return super().reshape(batch_size, sequence_length, height, width)

    def compile(self):
        if self.request is None:
            super().compile()
            if self.stateful:
                # an InferRequest, unlike a CompiledModel, owns the KV cache and can reset it
                self.request = self.request.create_infer_request()

    def reset_state(self):
        """
        Drops the cached prefix of a stateful model, so that the next call starts a new sequence.
        """
        if self.stateful and self.request is not None:
            self.request.reset_state()
        self._past_length = 0

    def _inference(self, inputs):
        if not self.stateful:
            return super()._inference(inputs)

        try:
            self.request.start_async(inputs, share_inputs=True)
            self.request.wait()
        except Exception as e:
            invalid_inputs_msg = self._incompatible_inputs_warning(inputs)
            if invalid_inputs_msg is not None:
                e.args += (invalid_inputs_msg,)
            raise e
        # the request owns these tensors and overwrites them on the next call
        return {name: self.request.get_tensor(name).data.copy() for name in self.output_names}

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForFeatureExtraction",
            checkpoint="sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        position_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(input_ids, np.ndarray)

        input_ids = ensure_numpy(input_ids)
        attention_mask = ensure_numpy(attention_mask)
        token_type_ids = ensure_numpy(token_type_ids)
        position_ids = ensure_numpy(position_ids)

        # On a stateful model, `past_key_values` returned by a previous call means "keep scoring the
        # same sequence"; its absence starts a new one and therefore invalidates the cached prefix.
        if self.stateful and past_key_values is None:
            self.reset_state()

        if attention_mask is None:
            attention_mask = np.ones((input_ids.shape[0], self._past_length + input_ids.shape[1]), dtype=np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add the token_type_ids when needed
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids if token_type_ids is not None else np.zeros_like(input_ids)

        if "decoder_input_ids" in self.input_names:
            inputs["decoder_input_ids"] = input_ids

        if "position_ids" in self.input_names:
            if position_ids is None:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
            # the cache already holds the prefix, only the incoming tokens are scored
            inputs["position_ids"] = position_ids[:, -input_ids.shape[1] :]

        if "beam_idx" in self.input_names:
            # feature extraction has no beam search, every sequence keeps its own cache slot
            inputs["beam_idx"] = np.arange(input_ids.shape[0], dtype=np.int32)

        outputs = self._inference(inputs)
        model_outputs = {
            name: outputs[name] if np_inputs else torch.from_numpy(outputs[name]).to(self.device)
            for name in self.output_names
        }

        if self.stateful:
            self._past_length += input_ids.shape[1]
            # a non-empty marker, so that passing it back continues this sequence, cf. OVModelForCausalLM
            model_outputs["past_key_values"] = ((),)

        # models exposing a single embedding keep the historical return type, the ones replacing it by
        # several tensors (e.g. the four classification heads of Qwen3Guard) are returned as they are
        if set(model_outputs) == {"last_hidden_state"}:
            return BaseModelOutput(last_hidden_state=model_outputs["last_hidden_state"])
        return ModelOutput(**model_outputs)

    @classmethod
    def _from_pretrained(cls, model_id: Union[str, Path], config: PretrainedConfig, *args, **kwargs):
        from .modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING

        if config.model_type == "sam":
            return OVSamModel._from_pretrained(model_id, config, *args, **kwargs)
        if config.model_type in MODEL_TYPE_TO_CLS_MAPPING.keys():
            from .modeling_visual_language import OVModelForVisualCausalLM

            return OVModelForVisualCausalLM.from_pretrained(
                model_id, config=config, export_feature="feature-extraction", *args, **kwargs
            )
        else:
            return super()._from_pretrained(model_id, config, *args, **kwargs)

    def _preprocess_quantization_config(
        self,
        quantization_config: OVQuantizationConfigBase,
        model_name_or_path: str,
    ) -> OVQuantizationConfigBase:
        if quantization_config.tokenizer is None:
            quantization_config = quantization_config.clone()
            quantization_config.tokenizer = model_name_or_path
        return quantization_config


MASKED_LM_EXAMPLE = r"""
    Example of masked language modeling using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> mask_token = tokenizer.mask_token
    >>> pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    >>> outputs = pipe("The goal of life is" + mask_token)
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a MaskedLMOutput for masked language modeling tasks.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForMaskedLM(OVModel):
    export_feature = "fill-mask"
    auto_model_class = AutoModelForMaskedLM

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForMaskedLM",
            checkpoint="roberta-base",
        )
    )
    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        attention_mask: Union[torch.Tensor, np.ndarray],
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(input_ids, np.ndarray)

        input_ids = ensure_numpy(input_ids)
        attention_mask = ensure_numpy(attention_mask)
        token_type_ids = ensure_numpy(token_type_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add the token_type_ids when needed
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids if token_type_ids is not None else np.zeros_like(input_ids)

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]
        return MaskedLMOutput(logits=logits)

    def _preprocess_quantization_config(
        self,
        quantization_config: OVQuantizationConfigBase,
        model_name_or_path: str,
    ) -> OVQuantizationConfigBase:
        if quantization_config.tokenizer is None:
            quantization_config = quantization_config.clone()
            quantization_config.tokenizer = model_name_or_path
        return quantization_config


IMAGE_CLASSIFICATION_EXAMPLE = r"""
    Example of image classification using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> model.reshape(batch_size=1, sequence_length=3, height=224, width=224)
    >>> pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> outputs = pipe(url)
    ```
    This class can also be used with [timm](https://github.com/huggingface/pytorch-image-models)
    models hosted on [HuggingFaceHub](https://huggingface.co/timm). Example:
    ```python
    >>> from transformers import pipeline
    >>> from optimum.intel.openvino.modeling_timm import TimmImageProcessor
    >>> from optimum.intel import OVModelForImageClassification

    >>> model_id = "timm/vit_tiny_patch16_224.augreg_in21k"
    >>> preprocessor = TimmImageProcessor.from_pretrained(model_id)
    >>> model = OVModelForImageClassification.from_pretrained(model_id, export=True)
    >>> pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> outputs = pipe(url)
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a ImageClassifierOutput for image classification tasks.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForImageClassification(OVModel):
    export_feature = "image-classification"
    auto_model_class = AutoModelForImageClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        config: Optional["PretrainedConfig"] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        # Fix the mismatch between timm_config and huggingface_config
        local_timm_model = _is_timm_ov_dir(model_id)
        if local_timm_model or (not os.path.isdir(model_id) and model_info(model_id).library_name == "timm"):
            if not is_timm_available():
                raise ImportError(
                    "To load a timm model, timm needs to be installed. Please install it with `pip install timm`."
                )

            if is_timm_version("<", "0.9.0"):
                raise ImportError(
                    "To load a timm model, please make sure to upgrade your `timm` version to at least 0.9.0, you can upgrade it by running `pip install --upgrade timm`"
                )

            from .modeling_timm import TimmConfig, TimmForImageClassification, TimmOpenVINOConfig

            config = TimmConfig.from_pretrained(model_id, **kwargs)
            #  If locally saved timm model, directly load
            if local_timm_model:
                return super()._from_pretrained(model_id=model_id, config=config)
            model = TimmForImageClassification.from_pretrained(model_id, **kwargs)
            openvino_config = TimmOpenVINOConfig(model.config)

            return cls._to_load(model=model, config=config, exporter_config=openvino_config, stateful=False, **kwargs)
        else:
            return super().from_pretrained(
                model_id=model_id,
                config=config,
                export=export,
                token=token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                subfolder=subfolder,
                local_files_only=local_files_only,
                task=task,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

    @add_start_docstrings_to_model_forward(
        IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + IMAGE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="OVModelForImageClassification",
            checkpoint="google/vit-base-patch16-224",
        )
    )
    def forward(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(pixel_values, np.ndarray)

        pixel_values = ensure_numpy(pixel_values)

        inputs = {
            "pixel_values": pixel_values,
        }

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]
        return ImageClassifierOutput(logits=logits)


AUDIO_CLASSIFICATION_EXAMPLE = r"""
    Example of audio classification using `transformers.pipelines`:
    ```python
    >>> from datasets import load_dataset
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)
    >>> pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
    >>> dataset = load_dataset("superb", "ks", split="test")
    >>> audio_file = dataset[3]["audio"]["array"]
    >>> outputs = pipe(audio_file)
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a SequenceClassifierOutput for audio classification tasks.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForAudioClassification(OVModel):
    export_feature = "audio-classification"
    auto_model_class = AutoModelForAudioClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + AUDIO_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="OVModelForAudioClassification",
            checkpoint="superb/hubert-base-superb-er",
        )
    )
    def forward(
        self,
        input_values: Union[torch.Tensor, np.ndarray],
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(input_values, np.ndarray)

        input_values = ensure_numpy(input_values)
        attention_mask = ensure_numpy(attention_mask)

        inputs = {
            "input_values": input_values,
        }

        # Add the attention_mask when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]
        return SequenceClassifierOutput(logits=logits)


CTC_EXAMPLE = r"""
    Example of CTC:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="np")
    >>> logits = model(**inputs).logits
    >>> predicted_ids = np.argmax(logits, axis=-1)

    >>> transcription = processor.batch_decode(predicted_ids)
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with a language modeling head on top for Connectionist Temporal Classification (CTC).
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForCTC(OVModel):
    """
    CTC model for OpenVINO.
    """

    auto_model_class = AutoModelForCTC
    export_feature = "automatic-speech-recognition"

    @add_start_docstrings_to_model_forward(
        AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + CTC_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="OVModelForCTC",
            checkpoint="facebook/hubert-large-ls960-ft",
        )
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        np_inputs = isinstance(input_values, np.ndarray)

        input_values = ensure_numpy(input_values)
        attention_mask = ensure_numpy(attention_mask)

        inputs = {
            "input_values": input_values,
        }

        # Add the attention_mask when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]
        return CausalLMOutput(logits=logits)


AUDIO_XVECTOR_EXAMPLE = r"""
    Example of Audio XVector:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>>     embeddings = model(**inputs).embeddings

    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForAudioXVector(OVModel):
    """
    Audio XVector model for OpenVINO.
    """

    auto_model_class = AutoModelForAudioXVector
    export_feature = "audio-xvector"

    @add_start_docstrings_to_model_forward(
        AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + AUDIO_XVECTOR_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="OVModelForAudioXVector",
            checkpoint="anton-l/wav2vec2-base-superb-sv",
        )
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        np_inputs = isinstance(input_values, np.ndarray)

        input_values = ensure_numpy(input_values)
        attention_mask = ensure_numpy(attention_mask)

        inputs = {
            "input_values": input_values,
        }

        # Add the attention_mask when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]
        embeddings = (
            torch.from_numpy(outputs["embeddings"]).to(self.device) if not np_inputs else outputs["embeddings"]
        )

        return XVectorOutput(logits=logits, embeddings=embeddings)


AUDIO_FRAME_CLASSIFICATION_EXAMPLE = r"""
    Example of audio frame classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model =  {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    >>>    logits = model(**inputs).logits

    >>> probabilities = torch.sigmoid(torch.as_tensor(logits)[0])
    >>> labels = (probabilities > 0.5).long()
    >>> labels[0].tolist()
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model for with a frame classification head on top for tasks like Speaker Diarization.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForAudioFrameClassification(OVModel):
    """
    Audio Frame Classification model for OpenVINO.
    """

    auto_model_class = AutoModelForAudioFrameClassification
    export_feature = "audio-frame-classification"

    @add_start_docstrings_to_model_forward(
        AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + AUDIO_FRAME_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="OVModelForAudioFrameClassification",
            checkpoint="anton-l/wav2vec2-base-superb-sd",
        )
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        np_inputs = isinstance(input_values, np.ndarray)

        input_values = ensure_numpy(input_values)
        attention_mask = ensure_numpy(attention_mask)

        inputs = {
            "input_values": input_values,
        }

        # Add the attention_mask when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        outputs = self._inference(inputs)
        logits = torch.from_numpy(outputs["logits"]).to(self.device) if not np_inputs else outputs["logits"]

        return TokenClassifierOutput(logits=logits)


CUSTOM_TASKS_EXAMPLE = """
    Example of custom tasks (e.g. a sentence transformers with a pooler head):

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("I love burritos!", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooler_output = outputs.pooler_output
    ```
"""


@add_start_docstrings(
    """
    OpenVINO Model for custom tasks. It can be used to leverage the inference acceleration for any single-file OpenVINO model, that may use custom inputs and outputs.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForCustomTasks(OVModel):
    @add_start_docstrings_to_model_forward(
        CUSTOM_TASKS_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForCustomTasks",
            checkpoint="IlyasMoutawwakil/sbert-all-MiniLM-L6-v2-with-pooler",
        )
    )
    def forward(self, **kwargs):
        expected_inputs_names = set(self.input_names)
        inputs_names = set(kwargs)

        if not expected_inputs_names.issubset(inputs_names):
            raise ValueError(
                f"Got unexpected inputs: expecting the following inputs : {', '.join(expected_inputs_names)} but got : {', '.join(inputs_names)}."
            )

        np_inputs = isinstance(next(iter(kwargs.values())), np.ndarray)
        inputs = {}
        for input_name in self.input_names:
            inputs[input_name] = kwargs.pop(input_name).cpu().numpy() if not np_inputs else kwargs.pop(input_name)

        outputs = self._inference(inputs)
        model_outputs = {}
        for key, value in outputs.items():
            key_name = next(iter(key.names))
            if "." in key_name:
                key_name = key_name.split(".")[0]
                if key_name not in model_outputs:
                    model_outputs[key_name] = []
                model_outputs[key_name].append(torch.from_numpy(value).to(self.device) if not np_inputs else value)
            else:
                model_outputs[key_name] = torch.from_numpy(value).to(self.device) if not np_inputs else value

        return ModelOutput(**model_outputs)


class OVModelForZeroShotImageClassification(OVModel):
    auto_model_class = AutoModelForZeroShotImageClassification
    export_feature = "zero-shot-image-classification"

    def forward(self, input_ids, pixel_values, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        self.compile()

        np_inputs = isinstance(input_ids, np.ndarray)

        input_ids = ensure_numpy(input_ids)
        pixel_values = ensure_numpy(pixel_values)
        attention_mask = ensure_numpy(attention_mask)

        inputs = {"input_ids": input_ids, "pixel_values": pixel_values}
        # Add the attention_mask when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask if attention_mask is not None else np.ones_like(input_ids)
        outputs = self._inference(inputs)
        logits_per_image = (
            torch.from_numpy(outputs["logits_per_image"]).to(self.device)
            if not np_inputs
            else outputs["logits_per_image"]
        )
        logits_per_text = (
            torch.from_numpy(outputs["logits_per_text"]).to(self.device)
            if not np_inputs
            else outputs["logits_per_text"]
        )
        text_embeds = (
            torch.from_numpy(outputs["text_embeds"]).to(self.device) if not np_inputs else outputs["text_embeds"]
        )
        image_embeds = (
            torch.from_numpy(outputs["image_embeds"]).to(self.device) if not np_inputs else outputs["image_embeds"]
        )

        return CLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )

    def _preprocess_quantization_config(
        self,
        quantization_config: OVQuantizationConfigBase,
        model_name_or_path: str,
    ) -> OVQuantizationConfigBase:
        if quantization_config.processor is None:
            quantization_config = quantization_config.clone()
            quantization_config.processor = model_name_or_path
        return quantization_config
