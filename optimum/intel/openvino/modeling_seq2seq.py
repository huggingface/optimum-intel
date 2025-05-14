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
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openvino
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino import Core
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    GenerationConfig,
    Pix2StructForConditionalGeneration,
    PretrainedConfig,
    WhisperForConditionalGeneration,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import http_user_agent

from ...exporters.openvino import main_export
from ...exporters.openvino.stateful import model_has_state
from .. import OVConfig, OVQuantizer
from ..utils import is_transformers_version
from .configuration import OVQuantizationConfig, OVQuantizationConfigBase, OVWeightQuantizationConfig
from .modeling_base import OVBaseModel
from .utils import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    OV_DECODER_NAME,
    OV_DECODER_WITH_PAST_NAME,
    OV_ENCODER_NAME,
    OV_TO_PT_TYPE,
    TemporaryDirectory,
    _print_compiled_model_properties,
)


core = Core()

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "AutoTokenizer"

INPUTS_DOCSTRING = r"""
    Arguments:
        encoder (`openvino.Model`):
            The OpenVINO Runtime model associated to the encoder.
        decoder (`openvino.Model`):
            The OpenVINO Runtime model associated to the decoder.
        decoder_with_past (`openvino.Model`):
            The OpenVINO Runtime model associated  to the decoder with past key values.
        config (`transformers.PretrainedConfig`):
            [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is an instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
"""

ENCODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
"""


DECODER_INPUTS_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SEQ2SEQ_MODEL_DOCSTRING = r"""
    Arguments:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""


TRANSLATION_EXAMPLE = r"""
    Example of text generation:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> inputs = tokenizer(text, return_tensors="pt")
    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
    >>> text = "He never went out without a book under his arm, and he often came back with two."
    >>> outputs = pipe(text)
    ```
"""

PIX2STRUCT_MODEL_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, seq_length, hidden_size)`):
            Flattened pixel patches. the `hidden_size` is obtained by the following formula: `hidden_size` =
            `num_channels` * `patch_size` * `patch_size`
            The process of flattening the pixel patches is done by `Pix2StructProcessor`.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.
            Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

VISION_ENCODER_DECODER_SEQ2SEQ_MODEL_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor`):
            Features extracted from an Image. This tensor should be of shape
            `(batch_size, num_channels, height, width)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

_PROCESSOR_FOR_DOC = "AutoProcessor"

PIX2STRUCT_EXAMPLE = r"""
    Example of pix2struct:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}
    >>> from PIL import Image
    >>> import requests

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
    >>> inputs = processor(images=image, text=question, return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = processor.batch_decode(gen_tokens, skip_special_tokens=True)
    ```
"""

SPEECH_SEQ2SEQ_MODEL_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor`):
            Mel features extracted from the raw speech waveform.
            `(batch_size, feature_size, encoder_sequence_length)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.intel import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor.feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> gen_tokens = model.generate(inputs=inputs.input_features)
    >>> outputs = processor.tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.intel import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> speech_recognition = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> pred = speech_recognition(ds[0]["audio"]["array"])
    ```
"""

IMAGE_TO_TEXT_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}, {tokenizer_class}
    >>> from optimum.intel import {model_class}
    >>> from PIL import Image
    >>> import requests


    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> tokenizer = {tokenizer_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> inputs = processor(image, return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, {tokenizer_class}, pipeline
    >>> from optimum.intel import {model_class}
    >>> from PIL import Image
    >>> import requests


    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> tokenizer = {tokenizer_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_to_text = pipeline("image-to-text", model=model, tokenizer=tokenizer, feature_extractor=processor, image_processor=processor)
    >>> pred = image_to_text(image)
    ```
"""


@add_start_docstrings(
    """
    Sequence-to-sequence model with a language modeling head for OpenVINO inference.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForSeq2SeqLM(OVBaseModel, GenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"
    export_feature = "text2text-generation"

    def __init__(
        self,
        encoder: openvino.Model,
        decoder: openvino.Model,
        decoder_with_past: openvino.Model = None,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        self.config = config
        self.use_cache = decoder_with_past is not None or model_has_state(decoder)
        self.model_save_dir = model_save_dir
        self._compile_only = kwargs.get("compile_only", False)
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])

        if self.is_dynamic and not self._compile_only:
            encoder = self._reshape(encoder, -1, -1, is_decoder=False)
            decoder = self._reshape(decoder, -1, -1)
            if decoder_with_past is not None:
                decoder_with_past = self._reshape(decoder_with_past, -1, -1) if self.use_cache else None

        generation_config = kwargs.get("generation_config", None)
        self.generation_config = generation_config or GenerationConfig.from_model_config(config)

        if is_transformers_version(">=", "4.44.99"):
            # some model configs may have issues with loading without parameters initialization
            try:
                misplaced_generation_parameters = self.config._get_non_default_generation_parameters()
            except (KeyError, TypeError):
                misplaced_generation_parameters = {}
            if len(misplaced_generation_parameters) > 0:
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(self.generation_config, param_name, param_value)
                    setattr(self.config, param_name, None)

        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()

        self.decoder_with_past = None
        enable_compilation = kwargs.get("compile", True)
        self.encoder = OVEncoder(encoder, parent_model=self)
        self.decoder = OVDecoder(decoder, parent_model=self)

        if self.use_cache and not model_has_state(self.decoder.model):
            self.decoder_with_past = OVDecoder(decoder_with_past, parent_model=self)
        if enable_compilation:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        try:
            self.auto_model_class.register(AutoConfig, self.__class__)
        except AttributeError:
            pass

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self.encoder.dtype or self.decoder.dtype

    @property
    def _ov_submodel_names(self) -> List[str]:
        submodel_names = ["encoder", "decoder"]
        if self.decoder_with_past is not None:
            submodel_names.append("decoder_with_past")
        return submodel_names

    @property
    def encoder_model(self) -> openvino.Model:
        logger.warning(
            "Access to the `encoder_model` attribute is deprecated and will be removed in optimum-intel v1.24, please use `encoder.model` instead"
        )
        return self.encoder.model

    @property
    def decoder_model(self) -> openvino.Model:
        logger.warning(
            "Access to the `decoder_model` attribute is deprecated and will be removed in optimum-intel v1.24, please use `decoder.model` instead"
        )
        return self.decoder.model

    @property
    def decoder_with_past_model(self) -> openvino.Model:
        logger.warning(
            "Access to the `decoder_with_past_model` attribute is deprecated and will be removed in optimum-intel v1.24, please use `decoder_with_past.model` instead"
        )
        return getattr(self.decoder_with_past, "model", None)

    @property
    def ov_submodels(self) -> Dict[str, openvino.Model]:
        return {component_name: getattr(self, component_name).model for component_name in self._ov_submodel_names}

    def _save_pretrained(self, save_directory: Union[str, Path]):
        file_names = {
            "encoder": OV_ENCODER_NAME,
            "decoder": OV_DECODER_NAME,
            "decoder_with_past": OV_DECODER_WITH_PAST_NAME,
        }
        for name, model in self.ov_submodels.items():
            dst_path = os.path.join(save_directory, file_names[name])
            openvino.save_model(model, dst_path, compress_to_fp16=False)

        self._save_openvino_config(save_directory)
        if self.generation_config is not None:
            try:
                self.generation_config.save_pretrained(save_directory)
            except Exception as exception:
                logger.warning(
                    f"The generation config will not be saved, saving failed with following error:\n{exception}"
                )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        local_files_only: bool = False,
        use_cache: bool = True,
        from_onnx: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        generation_config = kwargs.pop("generation_config", None)
        subfolder = kwargs.pop("subfolder", "")

        default_encoder_file_name = ONNX_ENCODER_NAME if from_onnx else OV_ENCODER_NAME
        default_decoder_file_name = ONNX_DECODER_NAME if from_onnx else OV_DECODER_NAME
        default_decoder_with_past_file_name = ONNX_DECODER_WITH_PAST_NAME if from_onnx else OV_DECODER_WITH_PAST_NAME
        encoder_file_name = encoder_file_name or default_encoder_file_name
        decoder_file_name = decoder_file_name or default_decoder_file_name
        decoder_with_past_file_name = decoder_with_past_file_name or default_decoder_with_past_file_name
        decoder_with_past = None

        quantization_config = cls._prepare_quantization_config(quantization_config, load_in_8bit)
        compile_only = kwargs.pop("compile_only", False)
        device = kwargs.pop("device", "CPU")
        ov_config = kwargs.pop("ov_config", None)

        # Load model from hub
        if not os.path.isdir(model_id):
            allow_patterns = {
                encoder_file_name,
                decoder_file_name,
                decoder_with_past_file_name,
                encoder_file_name.replace(".xml", ".bin"),
                decoder_file_name.replace(".xml", ".bin"),
                decoder_with_past_file_name.replace(".xml", ".bin"),
                cls.config_name,
            }

            ignore_patterns = ["*.msgpack", "*.safetensors", "*pytorch_model.bin"]
            if not from_onnx:
                ignore_patterns.extend(["*.onnx", "*.onnx_data"])

            model_save_folder = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                revision=revision,
                token=token,
                user_agent=http_user_agent,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )

            model_save_dir = Path(model_save_folder)

        else:
            model_save_dir = Path(model_id)

        file_names = {
            "encoder": model_save_dir / encoder_file_name,
            "decoder": model_save_dir / decoder_file_name,
            "decoder_with_past": model_save_dir / decoder_with_past_file_name,
        }
        if not compile_only:
            encoder = cls.load_model(file_names["encoder"], quantization_config)
            decoder = cls.load_model(file_names["decoder"], quantization_config)
            if use_cache and not model_has_state(decoder) and os.path.exists(file_names["decoder_with_past"]):
                decoder_with_past = cls.load_model(file_names["decoder_with_past"], quantization_config)
        else:
            model_kwargs = {"device": device, "ov_config": ov_config, "model_save_dir": model_save_dir}
            encoder = cls._compile_model(file_names["encoder"], **model_kwargs)
            decoder = cls._compile_model(file_names["decoder"], **model_kwargs)

            if use_cache and not model_has_state(decoder) and os.path.exists(file_names["decoder_with_past"]):
                decoder_with_past = cls._compile_model(file_names["decoder_with_past"], **model_kwargs)

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

        return cls(
            encoder=encoder,
            decoder=decoder,
            decoder_with_past=decoder_with_past,
            config=config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            generation_config=generation_config,
            device=device,
            ov_config=ov_config,
            compile_only=compile_only,
            **kwargs,
        )

    @classmethod
    def _from_transformers(
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
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting
        # would end-up removing the directory containing the underlying OpenVINO model
        cls._model_save_dir_tempdirectory_instance = save_dir

        if task is None:
            task = cls.export_feature
            if use_cache:
                task = task + "-with-past"

        compile_only = kwargs.pop("compile_only", False)
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False
        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")
        stateful = kwargs.get("stateful", True)
        variant = kwargs.pop("variant", None)

        # now we use model_kwargs only for text-to-speech models to specify vocoder
        model_kwargs = kwargs if cls.export_feature == "text-to-audio" else None

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
            ov_config=ov_config,
            stateful=stateful,
            variant=variant,
            model_kwargs=model_kwargs,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            compile_only=compile_only,
            **kwargs,
        )

    @add_start_docstrings_to_model_forward(
        SEQ2SEQ_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TRANSLATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForSeq2SeqLM",
            checkpoint="echarlaix/t5-small-openvino",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encode if needed : first prediction pass
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Decode
        if past_key_values is None or self.decoder_with_past is None:
            decoder_outputs = self.decoder(
                input_ids=(
                    decoder_input_ids[:, -1:] if past_key_values is not None and self.use_cache else decoder_input_ids
                ),
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                cache_position=cache_position,
            )
        else:
            decoder_outputs = self.decoder_with_past(
                input_ids=decoder_input_ids[:, -1:],  # Cut decoder_input_ids if past is used
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                cache_position=cache_position,
            )

        return Seq2SeqLMOutput(logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values or kwargs.get("past", None),
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self):
        return self.encoder

    def _reorder_cache(self, past, beam_idx) -> Tuple[Tuple[torch.FloatTensor]]:
        return self.decoder._reorder_cache(past, beam_idx)

    def _reshape(self, model: openvino.Model, batch_size: int, sequence_length: int, is_decoder=True):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size if not is_decoder else -1
            if inputs.get_any_name().startswith("past_key_values"):
                shapes[inputs][2] = -1
            elif inputs.get_any_name().startswith("cache_position"):
                shapes[inputs][0] = sequence_length
            elif is_decoder and not inputs.get_any_name().startswith("encoder"):
                if not inputs.get_any_name().startswith("beam_idx"):
                    shapes[inputs][1] = -1
            else:
                shapes[inputs][1] = sequence_length
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            sequence_length (`int`):
                The sequence length.
        """
        if self._compile_only:
            raise ValueError(
                "`reshape()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        logger.warning("Some part of the model's decoder do not support static shapes and will be kept dynamic.")
        self.is_dynamic = True if batch_size == -1 and sequence_length == -1 else False
        self.encoder.model = self._reshape(self.encoder.model, batch_size, sequence_length, is_decoder=False)
        self.decoder.model = self._reshape(self.decoder.model, batch_size, sequence_length)
        if self.decoder_with_past is not None:
            self.decoder_with_past.model = self._reshape(self.decoder_with_past.model, batch_size, sequence_length)

        self.clear_requests()
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """

        if self._compile_only:
            raise ValueError(
                "`half()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        for submodel in self.ov_submodels.values():
            apply_moc_transformations(submodel, cf=False)
            compress_model_transformation(submodel)

        self.clear_requests()
        return self

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        for submodel_name in self._ov_submodel_names:
            getattr(self, submodel_name).request = None

    def compile(self):
        for submodel_name in self._ov_submodel_names:
            getattr(self, submodel_name)._compile()


class OVEncoder:
    """
    Encoder model for OpenVINO inference.

    Arguments:
        request (`openvino.ie_api.InferRequest`):
            The OpenVINO inference request associated to the encoder.
    """

    def __init__(self, model: openvino.Model, parent_model: OVModelForSeq2SeqLM):
        self.model = model
        self.parent_model = parent_model
        self._comple_only = parent_model._compile_only
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.inputs}
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.main_input_name = self.parent_model.main_input_name or "input_ids"
        self.request = None if not self._comple_only else self.model

    @property
    def _device(self):
        return self.parent_model._device

    @property
    def device(self):
        return self.parent_model.device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        for dtype in self.input_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    @add_start_docstrings_to_model_forward(ENCODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        **kwargs,
    ) -> BaseModelOutput:
        self._compile()

        # Model inputs
        inputs = {self.main_input_name: input_ids if input_ids is not None else kwargs.get(self.main_input_name)}

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        # Run inference
        last_hidden_state = torch.from_numpy(
            self.request(inputs, share_inputs=True, share_outputs=True)["last_hidden_state"]
        ).to(self.device)

        return BaseModelOutput(last_hidden_state=last_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _compile(self):
        ov_config = {**self.parent_model.ov_config}
        if (
            "CACHE_DIR" not in ov_config.keys()
            and not str(self.parent_model.model_save_dir).startswith(gettempdir())
            and "gpu" in self._device.lower()
        ):
            cache_dir = Path(self.parent_model.model_save_dir).joinpath("model_cache")
            ov_config["CACHE_DIR"] = str(cache_dir)

        if self.request is None:
            logger.info(f"Compiling the encoder to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                _print_compiled_model_properties(self.request)


class OVDecoder:
    """
    Decoder model for OpenVINO inference.

    Arguments:
        request (`openvino.ie_api.InferRequest`):
            The OpenVINO inference request associated to the decoder.
        device (`torch.device`):
            The device type used by this process.
    """

    def __init__(self, model: openvino.Model, parent_model: OVModelForSeq2SeqLM):
        self.model = model
        self.parent_model = parent_model
        self._compile_only = parent_model._compile_only
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.inputs}
        self.key_value_input_names = [key for key in self.input_names if "key_values" in key]
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.key_value_output_names = [key for key in self.output_names if "key_values" in key or "present" in key]
        self.stateful = model_has_state(self.model)
        is_legacy = any("past_key_values" in key.get_any_name() for key in self.model.outputs)
        self.use_past = len(self.key_value_input_names) > 0 or self.stateful
        self.next_beam_idx = None
        self._past_length = 0

        if len(self.key_value_input_names) > 0 and not is_legacy:
            self.use_past = True
            self.num_pkv = 2
        else:
            self.use_past = False
            self.num_pkv = 4

        self.request = None if not self._compile_only else self.model.create_infer_request()

    @property
    def _device(self) -> str:
        return self.parent_model._device

    @property
    def device(self) -> torch.device:
        return self.parent_model.device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        for dtype in self.input_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    @add_start_docstrings_to_model_forward(DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        self._compile()
        # Model inputs
        inputs = {}

        if self.stateful and past_key_values is None:
            self.request.reset_state()
            self._past_length = 0
            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)

        if past_key_values is not None and not self.stateful:
            # Flatten the past_key_values
            past_key_values = tuple(
                past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
            )

            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        inputs["input_ids"] = input_ids

        # Add the encoder_attention_mask inputs when needed
        if "encoder_attention_mask" in self.input_names and encoder_attention_mask is not None:
            inputs["encoder_attention_mask"] = encoder_attention_mask

        # Add the encoder_hidden_states inputs when needed
        if "encoder_hidden_states" in self.input_names and encoder_hidden_states is not None:
            inputs["encoder_hidden_states"] = encoder_hidden_states

        if "decoder_attention_mask" in self.input_names and decoder_attention_mask is not None:
            inputs["decoder_attention_mask"] = decoder_attention_mask

        if "cache_position" in self.input_names:
            if cache_position is None:
                past_len = self._get_past_length(past_key_values)
                cache_position = np.arange(past_len, past_len + input_ids.shape[1])
            inputs["cache_position"] = cache_position

        if "beam_idx" in self.input_names:
            batch_size = input_ids.shape[0]
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=np.int32)
            )
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = torch.from_numpy(self.request.get_tensor("logits").data).clone().to(self.device)
        self._past_length += input_ids.shape[1]

        out_past_key_values = ((),)

        if not self.stateful:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the
            # self-attention layer and 2 to the cross-attention layer)
            out_past_key_values = tuple(
                np.copy(self.request.get_tensor(key).data) for key in self.key_value_output_names
            )

            # Tuple of tuple of length `n_layers`, with each tuple of length equal to:
            # * 4 for the decoder without cache (k/v of self-attention + k/v of cross-attention)
            # * 2 for the decoder with cache (k/v of self-attention as cross-attention cache is constant)
            if self.use_past is False:
                out_past_key_values = tuple(
                    out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
                )
            else:
                # grab the cross attention key/values from the inputs
                out_past_key_values = tuple(
                    out_past_key_values[i : i + self.num_pkv] + past_key_values[2 * i + 2 : 2 * i + 2 + self.num_pkv]
                    for i in range(0, len(out_past_key_values), self.num_pkv)
                )

        return Seq2SeqLMOutput(logits=logits, past_key_values=out_past_key_values)

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        if self.stateful:
            return self._past_length
        return past_key_values[0][0].shape[-2]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _compile(self):
        ov_config = {**self.parent_model.ov_config}
        if (
            "CACHE_DIR" not in ov_config.keys()
            and not str(self.parent_model.model_save_dir).startswith(gettempdir())
            and "gpu" in self._device.lower()
        ):
            cache_dir = Path(self.parent_model.model_save_dir).joinpath("model_cache")
            ov_config["CACHE_DIR"] = str(cache_dir)

        if self.request is None:
            logger.info(f"Compiling the decoder to {self._device} ...")
            compiled_model = core.compile_model(self.model, self._device, ov_config)
            self.request = compiled_model.create_infer_request()
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                _print_compiled_model_properties(compiled_model)

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        if self.stateful:
            self.next_beam_idx = np.array(beam_idx)
            return past_key_values
        else:
            reordered_past = ()
            for layer_past in past_key_values:
                # Cached cross_attention states don't have to be reordered -> they are always the same
                reordered_past += (
                    tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past[:2]) + layer_past[2:],
                )
        return reordered_past


@add_start_docstrings(
    """
    VisionEncoderDecoder Sequence-to-sequence model with a language modeling head for OpenVINO inference.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForVision2Seq(OVModelForSeq2SeqLM):
    auto_model_class = AutoModelForVision2Seq
    main_input_name = "pixel_values"
    export_feature = "image-to-text"

    def __init__(
        self,
        encoder: openvino.Model,
        decoder: openvino.Model,
        decoder_with_past: openvino.Model = None,
        config: PretrainedConfig = None,
        **kwargs,
    ):
        if config.decoder.model_type == "gpt2":
            self.no_cross_attention_cache = True
        super().__init__(encoder, decoder, decoder_with_past, config, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @add_start_docstrings_to_model_forward(
        VISION_ENCODER_DECODER_SEQ2SEQ_MODEL_DOCSTRING
        + IMAGE_TO_TEXT_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            tokenizer_class=_TOKENIZER_FOR_DOC,
            model_class="OVModelForVision2Seq",
            checkpoint="microsoft/trocr-small-handwritten",
        )
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        return super().forward(
            input_ids=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            **kwargs,
        )

    def _reshape(self, model: openvino.Model, batch_size: int, sequence_length: int, is_decoder=True):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size if not is_decoder else -1
            if is_decoder:
                if inputs.get_any_name().startswith("past_key_values"):
                    shapes[inputs][2] = -1
                elif not inputs.get_any_name().startswith("encoder") and not inputs.get_any_name().startswith(
                    "beam_idx"
                ):
                    shapes[inputs][1] = -1
        model.reshape(shapes)
        return model


@add_start_docstrings(
    """
    Pix2Struct model with a language modeling head for OpenVINO inference.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForPix2Struct(OVModelForSeq2SeqLM):
    auto_model_class = Pix2StructForConditionalGeneration
    main_input_name = "flattened_patches"
    export_feature = "image-to-text"

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ) -> Dict:
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @add_start_docstrings_to_model_forward(
        PIX2STRUCT_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + PIX2STRUCT_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="OVModelForPix2Struct",
            checkpoint="google/pix2struct-ai2d-base",
        )
    )
    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        return super().forward(
            input_ids=flattened_patches,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            **kwargs,
        )

    def _reshape(self, model: openvino.Model, batch_size: int, sequence_length: int, is_decoder=True):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size if not is_decoder else -1
            if is_decoder:
                if inputs.get_any_name().startswith("past_key_values"):
                    shapes[inputs][2] = -1
                elif not inputs.get_any_name().startswith("encoder") and not inputs.get_any_name().startswith(
                    "beam_idx"
                ):
                    shapes[inputs][1] = -1
        model.reshape(shapes)
        return model


@add_start_docstrings(
    """
    Speech Sequence-to-sequence model with a language modeling head for OpenVINO inference. This class officially supports whisper, speech_to_text.
    """,
    INPUTS_DOCSTRING,
)
class OVModelForSpeechSeq2Seq(OVModelForSeq2SeqLM):
    auto_model_class = AutoModelForSpeechSeq2Seq
    main_input_name = "input_features"
    export_feature = "automatic-speech-recognition"

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        if decoder_attention_mask is None and decoder_input_ids is not None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids).to(decoder_input_ids.device)

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    @add_start_docstrings_to_model_forward(
        SPEECH_SEQ2SEQ_MODEL_DOCSTRING
        + AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="OVModelForSpeechSeq2Seq",
            checkpoint="openai/whisper-tiny",
        )
    )
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        return super().forward(
            input_ids=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        **kwargs,
    ):
        if "WhisperForConditionalGeneration" in config.architectures:
            return _OVModelForWhisper._from_pretrained(model_id, config, **kwargs)
        else:
            return super()._from_pretrained(model_id, config, **kwargs)


class _OVModelForWhisper(OVModelForSpeechSeq2Seq, WhisperForConditionalGeneration):
    """
    Whisper implements its own generate() method.
    """

    auto_model_class = WhisperForConditionalGeneration

    # force the use of the WhisperForConditionalGeneration generate and prepare_inputs_for_generation methods
    generate = WhisperForConditionalGeneration.generate

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        load_in_8bit: bool = False,
        quantization_config: Union[dict, OVQuantizationConfigBase] = None,
        **kwargs,
    ):
        compile_only = kwargs.get("compile_only", False)

        quantization_config = cls._prepare_quantization_config(quantization_config, load_in_8bit)
        if not compile_only and isinstance(quantization_config, OVQuantizationConfig):
            model = super(OVModelForSpeechSeq2Seq, cls)._from_pretrained(
                model_id, config, load_in_8bit=False, **kwargs
            )
            quantization_config_copy = copy.deepcopy(quantization_config)
            quantization_config_copy.processor = quantization_config.processor or model_id
            OVQuantizer(model).quantize(ov_config=OVConfig(quantization_config=quantization_config_copy))
        else:
            model = super(OVModelForSpeechSeq2Seq, cls)._from_pretrained(
                model_id, config, load_in_8bit=load_in_8bit, quantization_config=quantization_config, **kwargs
            )

        return model

    class DummyWhisperModel:
        def __init__(self):
            self.encoder = self.Encoder()

        class Encoder:
            def __init__(self):
                self.conv1 = self.Conv(stride=(1,))
                self.conv2 = self.Conv(stride=(2,))

            class Conv:
                def __init__(self, stride):
                    self.stride = stride

    # a dummy model attribute that's used in the generate method to compute the input stride
    # input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
    model = DummyWhisperModel()

    # Adopeted for stateful support from https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1810
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        # Overwritten -- encoder-decoder whisper has custom logic, but it's close to the general function. Next time
        # this function needs to be touched, let's try to sort out the commonalities between the two and remove the
        # overwrite.

        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        past_length = 0
        if past_key_values is not None:
            past_length = self.decoder._get_past_length(past_key_values)

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if decoder_position_ids is not None:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                decoder_position_ids = decoder_position_ids.clone(memory_format=torch.contiguous_format)

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1] :]

        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        decoder_input_ids = decoder_input_ids.contiguous()

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "cache_position": cache_position,
        }
