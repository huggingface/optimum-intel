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

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from torch.nn import functional as F
from transformers import (
    CLIPConfig,
    PretrainedConfig,
)
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import ModelOutput
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers.utils import is_offline_mode

from optimum.exporters.tasks import TasksManager

from ...exporters.openvino import main_export
from ..utils.modeling_utils import _find_files_matching_pattern, _OpenClipForZeroShotImageClassification
from .configuration import OVConfig, OVWeightQuantizationConfig
from .modeling import MODEL_START_DOCSTRING, OVModel
from .utils import TemporaryDirectory


logger = logging.getLogger(__name__)


class OVModelOpenCLIPBase(OVModel):
    config_name = "open_clip_config.json"
    _library_name = "open_clip"

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @classmethod
    def _load_config(
        cls,
        config_name_or_path: Union[str, os.PathLike],
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: Optional[Union[bool, str]] = None,
        force_download: bool = False,
        subfolder: str = "",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> PretrainedConfig:
        config_path = None
        config_dir = os.path.join(config_name_or_path, subfolder)

        all_files, _ = TasksManager.get_model_files(
            config_name_or_path, subfolder=subfolder, cache_dir=cache_dir, revision=revision, token=token
        )

        transformers_config_name = "config.json"
        config_name = None
        if cls.config_name in all_files:
            config_name = cls.config_name
        elif transformers_config_name in all_files:
            config_name = transformers_config_name

        if os.path.isdir(config_dir):
            if config_name is None:
                raise OSError(
                    f"neither {cls.config_name} nor {transformers_config_name} was found in {config_dir} local folder"
                )
            config_path = os.path.join(config_dir, config_name)
        else:
            if config_name:
                config_path = hf_hub_download(
                    repo_id=config_name_or_path,
                    filename=config_name,
                    subfolder=subfolder,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            else:
                open_clip_config = _OpenClipForZeroShotImageClassification.find_config_by_hub_url(config_name_or_path)

        if config_path:
            open_clip_config = {}
            with open(config_path, "r", encoding="utf-8") as f:
                open_clip_config = json.load(f)

        model_cfg = open_clip_config.get("model_cfg", open_clip_config)

        text_config_dict = model_cfg.get("text_cfg", None) or model_cfg.get("text_config", None)
        vision_config_dict = model_cfg.get("vision_cfg", None) or model_cfg.get("vision_config", None)

        config = CLIPConfig(
            text_config_dict=text_config_dict,
            vision_config_dict=vision_config_dict,
            **open_clip_config,
        )

        return config

    # function is overloaded to avoid fails when calling the TasksManager.infer_library_from_model in OptimizedModel.from_pretrained()
    # this should be removed when open_clip library support is added to optimum
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
        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        _export = export
        try:
            if local_files_only:
                object_id = model_id.replace("/", "--")
                cached_model_dir = os.path.join(cache_dir, f"models--{object_id}")
                refs_file = os.path.join(os.path.join(cached_model_dir, "refs"), revision or "main")
                with open(refs_file) as f:
                    revision = f.read()
                model_dir = os.path.join(cached_model_dir, "snapshots", revision)
            else:
                model_dir = model_id

            ov_files = _find_files_matching_pattern(
                model_dir,
                pattern=r"(.*)?openvino(.*)?\_model\_(.*)?.xml$",
                subfolder=subfolder,
                use_auth_token=token,
                revision=revision,
            )
            _export = len(ov_files) == 0
            if _export ^ export:
                if export:
                    logger.warning(
                        f"The model {model_id} was already converted to the OpenVINO IR but got `export=True`, the model will be converted to OpenVINO once again. "
                        "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
                    _export = True
                else:
                    logger.warning(
                        f"No OpenVINO files were found for {model_id}, setting `export=True` to convert the model to the OpenVINO IR. "
                        "Don't forget to save the resulting model with `.save_pretrained()`"
                    )
        except Exception as exception:
            logger.warning(
                f"Could not infer whether the model was already converted or not to the OpenVINO IR, keeping `export={export}`.\n{exception}"
            )

        if isinstance(model_id, Path):
            model_id = model_id.as_posix()

        config_path = config if isinstance(config, (str, os.PathLike)) else model_id
        config = cls._load_config(
            config_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        from_pretrained_method = cls._from_transformers if _export else cls._from_pretrained
        return from_pretrained_method(
            model_id=model_id,
            config=config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            subfolder=subfolder,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            task=task,
            **kwargs,
        )


@add_start_docstrings(
    """
    OpenVINO Model for OpenCLIP Text model for tasks like zero-shot-image-classification.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelOpenCLIPText(OVModelOpenCLIPBase):
    _xml_model_name = "openvino_model_text.xml"
    export_feature = "feature-extraction"

    def __init__(self, model=None, config=None, tokenize_cfg=None, **kwargs):
        super().__init__(model, config, **kwargs)
        self.tokenize_cfg = tokenize_cfg

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

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        def fn_get_submodels(model):
            return {"model_text": model.text}

        custom_export_configs = {
            "model_text": {},
        }

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task or cls.export_feature,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            ov_config=ov_config,
            library_name=cls._library_name,
            framework="pt",
            fn_get_submodels=fn_get_submodels,
            custom_architecture=True,
            custom_export_configs=custom_export_configs,
        )

        config.save_pretrained(save_dir_path)
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            file_name=cls._xml_model_name,
            **kwargs,
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
        file_name: Optional[str] = "openvino_model_text.xml",
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        return super()._from_pretrained(
            model_id=model_id,
            config=config,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            from_onnx=from_onnx,
            local_files_only=local_files_only,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ):
        self.compile()

        inputs = {"text": input_ids}

        outputs = self._inference(inputs)
        return ModelOutput(text_features=torch.from_numpy(outputs["text_features"]))


@add_start_docstrings(
    """
    OpenVINO Model for OpenCLIP Vision model for tasks like zero-shot-image-classification.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelOpenCLIPVisual(OVModelOpenCLIPBase):
    _xml_model_name = "openvino_model_vision.xml"
    export_feature = "feature-extraction"

    def __init__(self, model=None, config=None, preprocess_cfg=None, **kwargs):
        super().__init__(model, config, **kwargs)
        self.preprocess_cfg = preprocess_cfg

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

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        def fn_get_submodels(model):
            return {"model_vision": model.visual}

        custom_export_configs = {
            "model_vision": {},
        }

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task or cls.export_feature,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            ov_config=ov_config,
            library_name=cls._library_name,
            framework="pt",
            fn_get_submodels=fn_get_submodels,
            custom_architecture=True,
            custom_export_configs=custom_export_configs,
        )

        config.save_pretrained(save_dir_path)
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            file_name=cls._xml_model_name,
            **kwargs,
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
        file_name: Optional[str] = "openvino_model_vision.xml",
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        return super()._from_pretrained(
            model_id=model_id,
            config=config,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            from_onnx=from_onnx,
            local_files_only=local_files_only,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )

    def forward(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ):
        self.compile()

        inputs = {"x": pixel_values}

        outputs = self._inference(inputs)
        return ModelOutput(image_features=torch.from_numpy(outputs["image_features"]))


@add_start_docstrings(
    """
    OpenVINO Model with OpenCLIP for tasks like zero-shot-image-classification.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelOpenCLIPForZeroShotImageClassification:
    export_feature = "zero-shot-image-classification"

    def __init__(
        self,
        text_model: OVModelOpenCLIPText = None,
        visual_model: OVModelOpenCLIPVisual = None,
        config=None,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.text_model = text_model
        self.visual_model = visual_model
        self.config = config
        self.logit_scale = None
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = None
        if init_logit_bias is not None:
            self.logit_bias = torch.nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, device: str):
        """
        Use the specified `device` for inference. For example: "cpu" or "gpu". `device` can
        be in upper or lower case. To speed up first inference, call `.compile()` after `.to()`.
        """
        self.text_model.to(device=device)
        self.visual_model.to(device=device)

        return self

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
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        **kwargs,
    ):
        text_model = OVModelOpenCLIPText.from_pretrained(
            model_id=model_id,
            export=export,
            config=config,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            subfolder=subfolder,
            local_files_only=local_files_only,
            task=task or cls.export_feature,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        visual_model = OVModelOpenCLIPVisual.from_pretrained(
            model_id=model_id,
            export=export,
            config=config,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            subfolder=subfolder,
            local_files_only=local_files_only,
            task=task or cls.export_feature,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        if config is None:
            config = text_model.config

        return cls(
            text_model=text_model,
            visual_model=visual_model,
            config=config,
            init_logit_scale=init_logit_scale,
            init_logit_bias=init_logit_bias,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, np.ndarray],
        pixel_values: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ):
        self.text_model.compile()
        self.visual_model.compile()

        text_model_outputs = self.text_model._inference({"text": input_ids})
        visual_model_outputs = self.visual_model._inference({"x": pixel_values})

        text_features = F.normalize(torch.from_numpy(text_model_outputs["text_features"]), dim=-1)
        image_features = F.normalize(torch.from_numpy(visual_model_outputs["image_features"]), dim=-1)

        logits_per_image = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            logits_per_image += self.logit_bias
        logits_per_text = logits_per_image.T

        return CLIPOutput(
            loss=None,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_features,
            image_embeds=image_features,
            text_model_output=text_model_outputs,
            vision_model_output=visual_model_outputs,
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        self.text_model.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        self.visual_model.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        self.config.save_pretrained(save_directory)
        return

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> str:
        self.text_model.push_to_hub(
            save_directory=save_directory, repository_id=repository_id, private=private, token=token
        )
        self.visual_model.push_to_hub(
            save_directory=save_directory, repository_id=repository_id, private=private, token=token
        )
        return

    def compile(self):
        self.text_model.compile()
        self.visual_model.compile()

    def reshape(self, batch_size: int, sequence_length: int, height: int = None, width: int = None):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            sequence_length (`int`):
                The sequence length or number of channels.
            height (`int`, *optional*):
                The image height.
            width (`int`, *optional*):
                The image width.
        """
        self.text_model.reshape(batch_size=batch_size, sequence_length=sequence_length, height=height, width=width)
        self.visual_model.reshape(batch_size=batch_size, sequence_length=sequence_length, height=height, width=width)
        return self

    def half(self):
        """
        Converts all the model weights to FP16
        """
        self.text_model.half()
        self.visual_model.half()
        return self

    def eval(self):
        return self

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        return self.text_model.can_generate() and self.visual_model.can_generate()
