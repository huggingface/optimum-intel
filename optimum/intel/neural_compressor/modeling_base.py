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
from typing import Dict, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from neural_compressor.utils.pytorch import load
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    PretrainedConfig,
    XLNetLMHeadModel,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import is_ipex_available
from transformers.utils.generic import ContextManagers

from ...exporters import TasksManager
from ...modeling_base import OptimizedModel
from ..generation.modeling import jit_trace
from ..utils.import_utils import _torch_version, is_torch_version
from ..utils.modeling_utils import patch_decoder_attention_mask
from .configuration import INCConfig
from .utils import WEIGHTS_NAME


logger = logging.getLogger(__name__)


MODEL_START_DOCSTRING = r"""
    This model check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        model (`PyTorch model`): is the main class used to run inference.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is the Model configuration class with all the parameters of the model.
        device (`str`, defaults to `"cpu"`):
            The device type for which the model will be optimized for. The resulting compiled model will contains nodes specific to this device.
"""


class INCModel(OptimizedModel):
    auto_model_class = AutoModel
    base_model_prefix = "inc_model"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        q_config: Dict = None,
        inc_config: Dict = None,
        **kwargs,
    ):
        super().__init__(model=model, config=config)

        self.inc_config = inc_config
        self._q_config = q_config
        self.model_save_dir = model_save_dir
        self.is_quantized = q_config is not None

        if getattr(self.config, "backend", None) == "ipex":
            if not is_ipex_available():
                raise ImportError(
                    "Intel PyTorch Extensions was not found, please make sure you've installed the package or run `pip install intel-extension-for-pytorch`"
                )
            # Need import intel_extension_for_pytorch for ipex model
            import intel_extension_for_pytorch as ipex

            # Just to avoid to change by ruff.
            logger.info("intel_extension_for_pytorch version is " + ipex.__version__)

        # Registers the INCModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = WEIGHTS_NAME,
        local_files_only: bool = False,
        subfolder: str = "",
        **kwargs,
    ):
        model_name_or_path = kwargs.pop("model_name_or_path", None)
        if model_name_or_path is not None:
            logger.warning("`model_name_or_path` is deprecated please use `model_id`")
            model_id = model_id or model_name_or_path

        model_path = Path(model_id)

        if model_path.is_dir():
            model_cache_path = model_path / file_name
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

        model_save_dir = Path(model_cache_path).parent
        inc_config = None
        msg = None
        try:
            inc_config = INCConfig.from_pretrained(model_id)
            if not is_torch_version("==", inc_config.torch_version):
                msg = f"Quantized model was obtained with torch version {inc_config.torch_version} but {_torch_version} was found."
                logger.warning(f"{msg}")
        except Exception:
            logger.info("Couldn't verify torch version.")

        if getattr(config, "backend", None) == "ipex" or getattr(config, "torchscript", False):
            # NOTE: Will improve to use load function when Intel Neural Compressor next 2.1 release.
            # load(model_cache_path)
            model = torch.jit.load(model_cache_path)
            model = torch.jit.freeze(model.eval())
            return cls(model, config=config, model_save_dir=model_save_dir, inc_config=inc_config, **kwargs)

        model_class = _get_model_class(config, cls.auto_model_class._model_mapping)
        # Load the state dictionary of the model to verify whether the model to get the quantization config
        state_dict = torch.load(model_cache_path, map_location="cpu")
        q_config = state_dict.get("best_configure", None)

        if q_config is None:
            model = model_class.from_pretrained(model_save_dir)
        else:
            init_contexts = [no_init_weights(_enable=True)]
            with ContextManagers(init_contexts):
                model = model_class(config)
            try:
                model = load(model_cache_path, model)
            except Exception as e:
                # For incompatible torch version check
                if msg is not None:
                    e.args += (msg,)
                raise

        return cls(
            model, config=config, model_save_dir=model_save_dir, q_config=q_config, inc_config=inc_config, **kwargs
        )

    def _save_pretrained(self, save_directory: Union[str, Path]):
        output_path = os.path.join(save_directory, WEIGHTS_NAME)

        if isinstance(self.model, torch.nn.Module):
            state_dict = self.model.state_dict()
            if self._q_config:
                state_dict["best_configure"] = self._q_config
            torch.save(state_dict, output_path)
        else:
            torch.jit.save(self.model, output_path)

        if self.inc_config:
            self.inc_config.save_pretrained(save_directory)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        if is_torch_version("<", "2.0.0"):
            raise ImportError("`torch>=2.0.0` is needed to trace your model")

        task = cls.export_feature
        kwargs.get("file_name", None)

        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
            "torch_dtype": torch_dtype,
        }

        if config.torch_dtype == "int8" or config.torch_dtype == torch.int8:
            raise ValueError("quantized model cannot be exported")

        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)

        if task == "text-generation":
            model = patch_decoder_attention_mask(model)

        traced_model, has_position_ids = jit_trace(model, task, use_cache)
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
        config.torchscript = True

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            has_position_ids=has_position_ids,
            **kwargs,
        )


class INCModelForQuestionAnswering(INCModel):
    auto_model_class = AutoModelForQuestionAnswering
    export_feature = "question-answering"


class INCModelForSequenceClassification(INCModel):
    auto_model_class = AutoModelForSequenceClassification
    export_feature = "text-classification"


class INCModelForTokenClassification(INCModel):
    auto_model_class = AutoModelForTokenClassification
    export_feature = "token-classification"


class INCModelForMultipleChoice(INCModel):
    auto_model_class = AutoModelForMultipleChoice
    export_feature = "multiple-choice"


class INCModelForSeq2SeqLM(INCModel):
    auto_model_class = AutoModelForSeq2SeqLM
    export_feature = "text2text-generation"


class INCModelForMaskedLM(INCModel):
    auto_model_class = AutoModelForMaskedLM
    export_feature = "fill-mask"


class INCModelForVision2Seq(INCModel):
    auto_model_class = AutoModelForVision2Seq
    export_feature = "image-to-text"


class INCModelForXLNetLM(INCModel):
    auto_model_class = XLNetLMHeadModel
    export_feature = "fill-mask"
