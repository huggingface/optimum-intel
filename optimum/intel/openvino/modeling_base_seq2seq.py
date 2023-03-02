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
from tempfile import TemporaryDirectory
from typing import Optional, Union

import transformers
from transformers import AutoConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings, default_cache_path
from transformers.onnx import FeaturesManager, export
from transformers.onnx.utils import get_preprocessor

import openvino
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from openvino._offline_transformations import compress_model_transformation
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export_models, get_encoder_decoder_models_for_export

from ..utils.import_utils import is_transformers_version
from .modeling_base import OVBaseModel
from .utils import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    OV_DECODER_NAME,
    OV_DECODER_WITH_PAST_NAME,
    OV_ENCODER_NAME,
)


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Base OVModelForSeq2SeqLM class.
    """,
)
class OVBaseModelForSeq2SeqLM(OVBaseModel):
    export_feature = "seq2seq-lm"

    def __init__(
        self,
        encoder: openvino.runtime.Model,
        decoder: openvino.runtime.Model,
        decoder_with_past: openvino.runtime.Model = None,
        config: PretrainedConfig = None,
        **kwargs
    ):
        self.config = config
        self.use_cache = decoder_with_past is not None
        self.model_save_dir = kwargs.get("model_save_dir")
        self._device = kwargs.get("device", "CPU").upper()
        self.is_dynamic = kwargs.get("dynamic_shapes", True)
        self.preprocessors = kwargs.get("preprocessors", [])
        self.ov_config = {}
        if "GPU" in self._device:
            raise ValueError("Support of dynamic shapes for GPU devices is not yet available.")
        if self.is_dynamic:
            encoder = self._reshape(encoder, -1, -1, is_decoder=False)
            decoder = self._reshape(decoder, -1, -1)
            decoder_with_past = self._reshape(decoder_with_past, -1, -1) if self.use_cache else None
        self.encoder_model = encoder
        self.decoder_model = decoder
        self.decoder_with_past_model = decoder_with_past
        self.encoder_request = None
        self.decoder_request = None
        self.decoder_with_past_request = None
        if is_transformers_version("<=", "4.25.1"):
            self.generation_config = None
        else:
            from transformers import GenerationConfig

            self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        **kwargs
    ):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name and allows one to save the encoder model
                with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name and allows one to save the decoder model
                with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
        """
        src_files = [self.encoder_model, self.decoder_model]
        dst_file_names = [encoder_file_name or OV_ENCODER_NAME, decoder_file_name or OV_DECODER_NAME]
        if self.use_cache:
            src_files.append(self.decoder_with_past_model)
            dst_file_names.append(decoder_with_past_file_name or OV_DECODER_WITH_PAST_NAME)

        for src_file, dst_file_name in zip(src_files, dst_file_names):
            dst_path = os.path.join(save_directory, dst_file_name)
            openvino.runtime.serialize(src_file, dst_path, dst_path.replace(".xml", ".bin"))

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        local_files_only: bool = False,
        use_cache: bool = True,
        from_onnx: bool = False,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name openvino_encoder_model.xml and allows one to
                load the encoder model with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name openvino_decoder_model.xml and allows one to
                load the decoder model with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name
                openvino_decoder_with_past_model.xml, allowing to load the decoder model with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        default_encoder_file_name = ONNX_ENCODER_NAME if from_onnx else OV_ENCODER_NAME
        default_decoder_file_name = ONNX_DECODER_NAME if from_onnx else OV_DECODER_NAME
        default_decoder_with_past_file_name = ONNX_DECODER_WITH_PAST_NAME if from_onnx else OV_DECODER_WITH_PAST_NAME
        encoder_file_name = encoder_file_name or default_encoder_file_name
        decoder_file_name = decoder_file_name or default_decoder_file_name
        decoder_with_past_file_name = decoder_with_past_file_name or default_decoder_with_past_file_name

        # Load model from a local directory
        if os.path.isdir(model_id):
            if os.path.isfile(os.path.join(model_id, "ov_encoder_model.xml")):
                encoder_file_name = "ov_encoder_model.xml"
                encoder_file_name = "ov_decoder_model.xml"
                encoder_file_name = "ov_decoder_with_past_model.xml"
                logger.warning(
                    "The file names `ov_encoder_model.xml`, `ov_decoder_model.xml` and `ov_decoder_with_past_model.xml` "
                    "will be soon deprecated. Make sure to rename your file to respectively `openvino_encoder_model.xml`, "
                    "`openvino_decoder_model.xml` and `openvino_decoder_with_past_model.xml`"
                )

            encoder = cls.load_model(os.path.join(model_id, encoder_file_name))
            decoder = cls.load_model(os.path.join(model_id, decoder_file_name))
            decoder_with_past = (
                cls.load_model(os.path.join(model_id, decoder_with_past_file_name)) if use_cache else None
            )
            model_save_dir = Path(model_id)

        # Load model from hub
        else:
            model_file_names = {"encoder": encoder_file_name, "decoder": decoder_file_name}
            if use_cache:
                model_file_names["decoder_with_past"] = decoder_with_past_file_name

            # If not ONNX then OpenVINO IR : adds binary files
            if not from_onnx:
                for key in list(model_file_names.keys()):
                    model_file_names[key + "_bin"] = model_file_names[key].replace(".xml", ".bin")
            file_names = model_file_names.copy()
            try:
                for name, file_name in model_file_names.items():
                    model_cache_path = hf_hub_download(
                        repo_id=model_id,
                        filename=file_name,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    file_names[name] = model_cache_path
            except EntryNotFoundError:
                model_file_names = {"encoder": "ov_encoder_model.xml", "decoder": "ov_decoder_model.xml"}
                if use_cache:
                    model_file_names["decoder_with_past"] = "ov_decoder_with_past_model.xml"
                for key in list(model_file_names.keys()):
                    model_file_names[key + "_bin"] = model_file_names[key].replace(".xml", ".bin")
                file_names = model_file_names.copy()
                for name, file_name in model_file_names.items():
                    model_cache_path = hf_hub_download(
                        repo_id=model_id,
                        filename=file_name,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    file_names[name] = model_cache_path
                logger.warning(
                    "The file names `ov_encoder_model.xml`, `ov_decoder_model.xml` and `ov_decoder_with_past_model.xml` "
                    "will be soon deprecated. Make sure to rename your file to respectively `openvino_encoder_model.xml`, "
                    "`openvino_decoder_model.xml` and `openvino_decoder_with_past_model.xml`"
                )

            model_save_dir = Path(model_cache_path).parent
            encoder = cls.load_model(file_names["encoder"])
            decoder = cls.load_model(file_names["decoder"])
            decoder_with_past = cls.load_model(file_names["decoder_with_past"]) if use_cache else None

        return cls(
            encoder=encoder,
            decoder=decoder,
            decoder_with_past=decoder_with_past,
            config=config,
            model_save_dir=model_save_dir,
            **kwargs,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        task: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, defaults to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        encoder_file_name = os.path.join("encoder", ONNX_ENCODER_NAME)
        decoder_file_name = os.path.join("decoder", ONNX_DECODER_NAME)
        decoder_with_past_file_name = os.path.join("decoder_with_past", ONNX_DECODER_WITH_PAST_NAME)

        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        model = TasksManager.get_model_from_task(
            task,
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            config=config,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
        )

        onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        onnx_config = onnx_config_constructor(model.config, use_past=use_cache)
        models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)

        output_names = [encoder_file_name, decoder_file_name]
        if use_cache is True:
            output_names.append(decoder_with_past_file_name)

        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output_dir=save_dir_path,
            output_names=output_names,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            from_onnx=True,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            encoder_file_name=encoder_file_name,
            decoder_file_name=decoder_file_name,
            decoder_with_past_file_name=decoder_with_past_file_name,
            local_files_only=local_files_only,
            **kwargs,
        )

    def _reshape(self, model: openvino.runtime.Model, batch_size: int, sequence_length: int, is_decoder=True):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size if not is_decoder else -1
            if inputs.get_any_name().startswith("past_key_values"):
                shapes[inputs][2] = -1
            elif is_decoder and not inputs.get_any_name().startswith("encoder"):
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
        logger.warning("Some part of the model's decoder do not support static shapes and will be kept dynamic.")
        self.is_dynamic = True if batch_size == -1 and sequence_length == -1 else False
        self.encoder_model = self._reshape(self.encoder_model, batch_size, sequence_length, is_decoder=False)
        self.decoder_model = self._reshape(self.decoder_model, batch_size, sequence_length)
        if self.use_cache:
            self.decoder_with_past_model = self._reshape(self.decoder_with_past_model, batch_size, sequence_length)

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        apply_moc_transformations(self.encoder_model)
        apply_moc_transformations(self.decoder_model)
        compress_model_transformation(self.encoder_model)
        compress_model_transformation(self.decoder_model)
        if self.use_cache:
            apply_moc_transformations(self.decoder_with_past_model)
            compress_model_transformation(self.decoder_with_past_model)
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError
