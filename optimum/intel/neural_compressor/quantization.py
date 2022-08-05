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
from enum import Enum
from typing import Callable, ClassVar, Dict, Optional, Union

import torch
from torch.quantization import add_observer_, convert
from torch.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    XLNetLMHeadModel,
)
from transformers.file_utils import cached_path, hf_bucket_url
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils.versions import require_version

import neural_compressor
from neural_compressor.adaptor.pytorch import PyTorch_FXAdaptor, _cfg_to_qconfig, _propagate_qconfig
from neural_compressor.adaptor.torch_utils.util import get_embedding_contiguous
from neural_compressor.conf.config import Quantization_Conf
from neural_compressor.experimental import Quantization
from neural_compressor.utils.pytorch import _load_int8_orchestration

from .configuration import IncOptimizedConfig, IncQuantizationConfig
from .utils import WEIGHTS_NAME, IncDataLoader, _cfgs_to_fx_cfgs


logger = logging.getLogger(__name__)
require_version("neural_compressor>=1.9.0", "To fix: pip install neural_compressor")


class IncQuantizationMode(Enum):

    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"


SUPPORTED_QUANT_MODE = set([approach.value for approach in IncQuantizationMode])


class IncQuantizer:
    def __init__(
        self,
        config: Union[str, IncQuantizationConfig],
        eval_func: Optional[Callable],
        train_func: Optional[Callable] = None,
        calib_dataloader: Optional[DataLoader] = None,
    ):
        """
        Arguments:
            config (`Union[str, IncQuantizationConfig]`):
                Path to the YAML configuration file or an instance of the class :class:`IncQuantizationConfig`, used to
                control the tuning behavior.
            eval_func (`Callable`):
                Evaluation function to evaluate the tuning objective.
            train_func (`Callable`, *optional*):
                Training function for quantization aware training approach.
            calib_dataloader (`DataLoader`, *optional*):
                DataLoader for post-training quantization calibration.
        """

        self.config = config.config if isinstance(config, IncQuantizationConfig) else Quantization_Conf(config)
        self.approach = IncQuantizationMode(self.config.usr_cfg.quantization.approach)
        self.eval_func = eval_func
        self.train_func = train_func
        if calib_dataloader is not None:
            calib_dataloader = IncDataLoader.from_pytorch_dataloader(calib_dataloader)
        self.calib_dataloader = calib_dataloader

        if self.config.usr_cfg.model.framework == "pytorch_fx":
            neural_compressor.adaptor.pytorch._cfgs_to_fx_cfgs = _cfgs_to_fx_cfgs

        self.quantization = Quantization(self.config)

        self.quantization.eval_func = self.eval_func

        if self.approach == IncQuantizationMode.STATIC:
            if self.calib_dataloader is None:
                raise ValueError("calib_dataloader must be provided for static quantization.")
            self.quantization._calib_dataloader = self.calib_dataloader

        if self.approach == IncQuantizationMode.AWARE_TRAINING:
            if self.train_func is None:
                raise ValueError("train_func must be provided for quantization aware training.")
            self.quantization.q_func = self.train_func


# Adapted from https://github.com/intel/neural-compressor/blob/master/neural_compressor/utils/pytorch.py#L96
def apply_quantization_from_config(q_config: Dict, model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply Intel Neural Compressor quantization steps on the given model.

    Arguments:
        q_config (`Dict`):
            Dictionary containing all quantization information such as approach, dtype, scheme and granularity.
        model (`torch.nn.Module`):
            Model to quantize.
    Returns:
        q_model (`torch.nn.Module`):
            Quantized model.
    """
    approach = q_config.get("approach")
    framework = q_config.get("framework")

    if approach not in SUPPORTED_QUANT_MODE:
        raise ValueError(
            "Unknown quantization approach. Supported approach are " + ", ".join(SUPPORTED_QUANT_MODE.keys())
        )

    quant_mode = IncQuantizationMode(approach)
    q_model = copy.deepcopy(model)
    q_model.eval()

    if framework == "pytorch_fx":
        op_cfgs = _cfg_to_qconfig(q_config, approach)
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, approach)

        if not q_config["fx_sub_module_list"]:
            if quant_mode == IncQuantizationMode.AWARE_TRAINING:
                q_model.train()
                q_model = prepare_qat_fx(q_model, fx_op_cfgs)
            else:
                q_model = prepare_fx(q_model, fx_op_cfgs)
            q_model = convert_fx(q_model)

        else:
            sub_module_list = q_config["fx_sub_module_list"]
            if q_config["approach"] == "quant_aware_training":
                q_model.train()
                PyTorch_FXAdaptor.prepare_sub_graph(sub_module_list, fx_op_cfgs, q_model, prefix="", is_qat=True)
            else:
                PyTorch_FXAdaptor.prepare_sub_graph(sub_module_list, fx_op_cfgs, q_model, prefix="")
            PyTorch_FXAdaptor.convert_sub_graph(sub_module_list, q_model, prefix="")

    else:
        if quant_mode == IncQuantizationMode.DYNAMIC:
            q_mapping = torch.quantization.quantization_mappings.get_default_dynamic_quant_module_mappings()
            op_cfgs = _cfg_to_qconfig(q_config, approach)
        else:
            q_mapping = torch.quantization.quantization_mappings.get_default_static_quant_module_mappings()
            op_cfgs = _cfg_to_qconfig(q_config)

        _propagate_qconfig(q_model, op_cfgs, approach=approach)

        if quant_mode != IncQuantizationMode.DYNAMIC:
            add_observer_(q_model)
        q_model = convert(q_model, mapping=q_mapping, inplace=True)

    return q_model


class IncQuantizedModel:

    TRANSFORMERS_AUTO_CLASS: ClassVar

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the"
            f"`{self.__class__.__name__}.from_pretrained(model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        inc_config: Union[IncOptimizedConfig, str] = None,
        q_model_name: Optional[str] = None,
        **kwargs
    ) -> torch.nn.Module:
        """
        Instantiate a quantized pytorch model from a given Intel Neural Compressor configuration file.
        Arguments:
            model_name_or_path (`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            inc_config (`Union[IncOptimizedConfig, str]`, *optional*):
                Configuration file containing all the information related to the model quantization.
                Can be either:
                    - an instance of the class :class:`IncOptimizedConfig`,
                    - a string valid as input to :func:`IncOptimizedConfig.from_pretrained`.
            q_model_name (`str`, *optional*):
                Name of the state dictionary located in model_name_or_path used to load the quantized model. If
                state_dict is specified, the latter will not be used.
            cache_dir (`str`, *optional*):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                State dictionary of the quantized model, if not specified q_model_name will be used to load the
                state dictionary.
        Returns:
            q_model: Quantized model.
        """
        download_kwarg_default = [
            ("cache_dir", None),
            ("force_download", False),
            ("resume_download", False),
            ("revision", None),
        ]
        download_kwargs = {name: kwargs.get(name, default_value) for (name, default_value) in download_kwarg_default}
        state_dict = kwargs.get("state_dict", None)

        config = AutoConfig.from_pretrained(model_name_or_path)
        model_class = _get_model_class(config, cls.TRANSFORMERS_AUTO_CLASS._model_mapping)
        keys_to_ignore_on_load_unexpected = copy.deepcopy(
            getattr(model_class, "_keys_to_ignore_on_load_unexpected", None)
        )
        keys_to_ignore_on_load_missing = copy.deepcopy(getattr(model_class, "_keys_to_ignore_on_load_missing", None))
        # Avoid unnecessary warnings resulting from quantized model initialization
        quantized_keys_to_ignore_on_load = [
            r"zero_point",
            r"scale",
            r"packed_params",
            r"constant",
            r"module",
            r"best_configure",
            r"max_val",
            r"min_val",
            r"eps",
            r"fake_quant_enabled",
            r"observer_enabled",
        ]
        if keys_to_ignore_on_load_unexpected is None:
            model_class._keys_to_ignore_on_load_unexpected = quantized_keys_to_ignore_on_load
        else:
            model_class._keys_to_ignore_on_load_unexpected.extend(quantized_keys_to_ignore_on_load)
        missing_keys_to_ignore_on_load = [r"weight", r"bias"]
        if keys_to_ignore_on_load_missing is None:
            model_class._keys_to_ignore_on_load_missing = missing_keys_to_ignore_on_load
        else:
            model_class._keys_to_ignore_on_load_missing.extend(missing_keys_to_ignore_on_load)

        model = model_class.from_pretrained(model_name_or_path, **kwargs)

        model_class._keys_to_ignore_on_load_unexpected = keys_to_ignore_on_load_unexpected
        model_class._keys_to_ignore_on_load_missing = keys_to_ignore_on_load_missing

        if state_dict is None:

            q_model_name = q_model_name if q_model_name is not None else WEIGHTS_NAME
            revision = download_kwargs.pop("revision", None)
            if os.path.isdir(model_name_or_path):
                state_dict_path = os.path.join(model_name_or_path, q_model_name)
            elif os.path.isfile(model_name_or_path):
                state_dict_path = model_name_or_path
            else:
                state_dict_path = hf_bucket_url(model_name_or_path, filename=q_model_name, revision=revision)

            try:
                state_dict_path = cached_path(state_dict_path, **download_kwargs)
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load config for '{model_name_or_path}'. Make sure that:\n\n - '{model_name_or_path}' is a "
                    f"correct model identifier listed on 'https://huggingface.co/models'\n\n - or "
                    f"'{model_name_or_path}' is a correct path to a directory containing a {q_model_name} file\n\n"
                )

                if revision is not None:
                    msg += (
                        f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id)  "
                        f"thatexists for this model name as listed on its model page on "
                        f"'https://huggingface.co/models'\n\n"
                    )

                raise EnvironmentError(msg)

            state_dict = torch.load(state_dict_path)

        if "best_configure" in state_dict:
            inc_config = state_dict.pop("best_configure")
        elif isinstance(inc_config, IncOptimizedConfig):
            inc_config = inc_config.config
        else:
            config_path = inc_config if inc_config is not None else model_name_or_path
            inc_config = IncOptimizedConfig.from_pretrained(config_path, **download_kwargs).config

        if "is_oneshot" in inc_config and inc_config["is_oneshot"]:
            return _load_int8_orchestration(model, inc_config, state_dict)

        q_model = apply_quantization_from_config(inc_config, model)

        q_model.load_state_dict(state_dict, strict=False)

        get_embedding_contiguous(q_model)

        return q_model


class IncQuantizedModelForQuestionAnswering(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class IncQuantizedModelForSequenceClassification(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class IncQuantizedModelForTokenClassification(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


class IncQuantizedModelForMultipleChoice(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = AutoModelForMultipleChoice


class IncQuantizedModelForSeq2SeqLM(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = AutoModelForSeq2SeqLM


class IncQuantizedModelForCausalLM(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = AutoModelForCausalLM


class IncQuantizedModelForMaskedLM(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = AutoModelForMaskedLM


class IncQuantizedModelForXLNetLM(IncQuantizedModel):

    TRANSFORMERS_AUTO_CLASS = XLNetLMHeadModel
