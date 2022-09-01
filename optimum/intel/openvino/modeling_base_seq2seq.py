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
from typing import Optional, Union

import transformers
from transformers import AutoConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings, default_cache_path
from transformers.onnx import FeaturesManager, export
from transformers.onnx.utils import get_preprocessor

import openvino
import openvino.runtime.passes as passes
from huggingface_hub import HfApi, hf_hub_download
from optimum.onnx.configuration import DecoderOnnxConfig, EncoderOnnxConfig
from optimum.onnx.modeling_seq2seq import _DecoderWithLMhead

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
    Base OVModel class.
    """,
)
class OVBaseModelForSeq2SeqLM(OVBaseModel):

    export_feature = "seq2seq-lm"

    def __init__(
        self,
        encoder: openvino.pyopenvino.Model,
        decoder: openvino.pyopenvino.Model,
        decoder_with_past: openvino.pyopenvino.Model = None,
        config: transformers.PretrainedConfig = None,
        **kwargs
    ):
        self.config = config
        self.use_cache = decoder_with_past is not None
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self._device = kwargs.get("device", "CPU")
        self.ov_config = {"PERFORMANCE_HINT": "LATENCY"}
        self.decoder_input_names = {key.get_any_name(): idx for idx, key in enumerate(decoder.inputs)}
        self.decoder_with_past_input_names = (
            {key.get_any_name(): idx for idx, key in enumerate(decoder_with_past.inputs)} if self.use_cache else None
        )
        self.encoder_model = encoder
        self.decoder_model = decoder
        self.decoder_with_past_model = decoder_with_past
        self.encoder_request = self._create_infer_request(encoder)
        self.decoder_request = self._create_infer_request(decoder)
        self.decoder_with_past_request = self._create_infer_request(decoder_with_past) if self.use_cache else None

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
            pass_manager = passes.Manager()
            pass_manager.register_pass("Serialize", dst_path, dst_path.replace(".xml", ".bin"))
            pass_manager.run_passes(src_file)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
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
                The encoder model file name. Overwrites the default file name and allows one to load the encoder model
                with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name and allows one to load the decoder model
                with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name, allowing to load
                the decoder model with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        from_onnx = kwargs.pop("from_onnx", False)
        config_dict = kwargs.pop("config", {})
        config = PretrainedConfig.from_dict(config_dict)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cache = kwargs.pop("use_cache", True)
        # TODO: Remove with next openvino release
        if use_cache:
            logger.warning(
                "The `use_cache` argument is changed to `False`, its support will be enabled in the next release."
            )
            use_cache = False
        default_encoder_file_name = ONNX_ENCODER_NAME if from_onnx else OV_ENCODER_NAME
        default_decoder_file_name = ONNX_DECODER_NAME if from_onnx else OV_DECODER_NAME
        default_decoder_with_past_file_name = ONNX_DECODER_WITH_PAST_NAME if from_onnx else OV_DECODER_WITH_PAST_NAME

        encoder_file_name = encoder_file_name or default_encoder_file_name
        decoder_file_name = decoder_file_name or default_decoder_file_name
        decoder_with_past_file_name = decoder_with_past_file_name or default_decoder_with_past_file_name

        # Load model from a local directory
        if os.path.isdir(model_id):
            encoder_bin_file_name = (
                os.path.join(model_id, encoder_file_name.replace(".xml", ".bin")) if not from_onnx else None
            )
            decoder_bin_file_name = (
                os.path.join(model_id, decoder_file_name.replace(".xml", ".bin")) if not from_onnx else None
            )
            decoder_with_past_bin_file_name = (
                os.path.join(model_id, decoder_with_past_file_name.replace(".xml", ".bin")) if not from_onnx else None
            )

            encoder = cls.load_model(os.path.join(model_id, encoder_file_name), encoder_bin_file_name)
            decoder = cls.load_model(os.path.join(model_id, decoder_file_name), decoder_bin_file_name)
            decoder_with_past = (
                cls.load_model(os.path.join(model_id, decoder_with_past_file_name), decoder_with_past_bin_file_name)
                if use_cache
                else None
            )
            kwargs["model_save_dir"] = Path(model_id)

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
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            encoder = cls.load_model(file_names["encoder"], bin_file_name=file_names.pop("encoder_bin", None))
            decoder = cls.load_model(file_names["decoder"], bin_file_name=file_names.pop("decoder_bin", None))
            if use_cache:
                decoder_with_past = cls.load_model(
                    file_names["decoder_with_past"], bin_file_name=file_names.pop("decoder_with_past_bin", None)
                )
            else:
                decoder_with_past = None

        return cls(encoder, decoder, decoder_with_past, config=config, **kwargs)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        save_dir: Union[str, Path] = default_cache_path,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                Directory from which to load
            save_dir (`str` or `Path`):
                Directory where the exported ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        # Create a local directory to save the model
        save_dir = Path(save_dir).joinpath(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model_save_dir"] = save_dir
        use_cache = kwargs.get("use_cache", True)
        preprocessor = get_preprocessor(model_id)

        model = FeaturesManager.get_model_from_feature(cls.export_feature, model_id)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=cls.export_feature)
        onnx_config = model_onnx_config(model.config)
        onnx_opset = onnx_config.default_onnx_opset
        onnx_config_encoder = EncoderOnnxConfig(model.config, task="default")
        onnx_config_decoder = DecoderOnnxConfig(model.config, task=cls.export_feature, use_past=False)
        onnx_config_decoder_with_past = DecoderOnnxConfig(model.config, task=cls.export_feature, use_past=True)

        # Extract the encoder for ONNX export
        encoder = model.get_encoder()
        # Concatenate the decoder with the language model head for ONNX export
        decoder_with_lm_head = _DecoderWithLMhead(model)

        # Export the encoder
        export(
            preprocessor=preprocessor,
            model=encoder,
            config=onnx_config_encoder,
            opset=onnx_opset,
            output=save_dir.joinpath(ONNX_ENCODER_NAME),
        )

        # Export the decoder without the past key values
        export(
            preprocessor=preprocessor,
            model=decoder_with_lm_head,
            config=onnx_config_decoder,
            opset=onnx_opset,
            output=save_dir.joinpath(ONNX_DECODER_NAME),
        )

        # Export the decoder with the past key values
        if use_cache:
            export(
                preprocessor=preprocessor,
                model=decoder_with_lm_head,
                config=onnx_config_decoder_with_past,
                opset=onnx_opset,
                output=save_dir.joinpath(ONNX_DECODER_WITH_PAST_NAME),
            )

        kwargs["config"] = model.config.__dict__
        kwargs["from_onnx"] = True

        return cls._from_pretrained(save_dir, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
