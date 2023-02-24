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
import inspect
import logging
import os
import warnings
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Callable, ClassVar, Dict, Optional, Union

import torch
import transformers
from datasets import Dataset, load_dataset
from packaging import version
from torch.quantization import add_observer_, convert
from torch.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    DataCollator,
    PretrainedConfig,
    PreTrainedModel,
    XLNetLMHeadModel,
    default_data_collator,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import TRANSFORMERS_CACHE, is_offline_mode
from transformers.utils.generic import ContextManagers

import neural_compressor
from huggingface_hub import HfApi, hf_hub_download
from neural_compressor.adaptor.pytorch import PyTorch_FXAdaptor, _cfg_to_qconfig, _propagate_qconfig
from neural_compressor.experimental.export import torch_to_int8_onnx
from neural_compressor.model.torch_model import IPEXModel, PyTorchModel
from neural_compressor.quantization import fit
from neural_compressor.utils.pytorch import load
from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfig
from optimum.quantization_base import OptimumQuantizer

from .configuration import IncOptimizedConfig, IncQuantizationConfig
from .utils import (
    MIN_QDQ_ONNX_OPSET,
    ONNX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    INCDataLoader,
    _cfgs_to_fx_cfgs,
    is_torch_less_than_1_13,
)


logger = logging.getLogger(__name__)

_neural_compressor_version = version.parse(version.parse(neural_compressor.__version__).base_version)

# TODO : Replace required version to 2.0.0
NEURAL_COMPRESSOR_REQUIRED_VERSION = version.parse("1.14.2")

if _neural_compressor_version < NEURAL_COMPRESSOR_REQUIRED_VERSION:
    raise ImportError(
        f"Found an incompatible version of neural-compressor. Found version {_neural_compressor_version}, "
        f"but only version {NEURAL_COMPRESSOR_REQUIRED_VERSION} is supported."
    )


class INCQuantizationMode(Enum):
    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"


SUPPORTED_QUANT_MODE = set([approach.value for approach in INCQuantizationMode])


class INCQuantizer(OptimumQuantizer):
    """
    Handle the Neural Compressor quantization process.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        eval_fn: Optional[Callable[[PreTrainedModel], int]] = None,
        calibration_fn: Optional[Callable[[PreTrainedModel], int]] = None,
        task: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Args:
            model (`torch.nn.Module`):
                The model to quantize.
            eval_fn (`Callable[[PreTrainedModel], int]`, defaults to None):
                The evaluation function to use for the accuracy driven strategy of the quantization process.
                The accuracy driven strategy will be enabled only if `eval_fn` is provided.
            task (`str`, defaults to None):
                The task defining the model topology used for the ONNX export.
            seed (`int`, defaults to 42):
                The random seed to use when shuffling the calibration dataset.
        """
        super().__init__()
        self._original_model = model
        self.eval_fn = eval_fn
        self.calibration_fn = calibration_fn
        self.task = task
        self.seed = seed
        signature = inspect.signature(self._original_model.forward)
        self._signature_columns = list(signature.parameters.keys())
        self.input_names = None
        self._quantized_model = None

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        quantization_config: "PostTrainingQuantConfig",
        save_directory: Union[str, Path],
        calibration_dataset: Dataset = None,
        batch_size: int = 8,
        data_collator: Optional[DataCollator] = None,
        remove_unused_columns: bool = True,
        **kwargs,
    ):
        """
        Quantize a model given the optimization specifications defined in `quantization_config`.

        Args:
            quantization_config (`PostTrainingQuantConfig`):
                The configuration containing the parameters related to quantization.
            save_directory (`Union[str, Path]`):
                The directory where the quantized model should be saved.
            calibration_dataset (`datasets.Dataset`, defaults to `None`):
                The dataset to use for the calibration step, needed for post-training static quantization.
            batch_size (`int`, defaults to 8):
                The number of calibration samples to load per batch.
            data_collator (`DataCollator`, defaults to `None`):
                The function to use to form a batch from a list of elements of the calibration dataset.
            remove_unused_columns (`bool`, defaults to `True`):
                Whether or not to remove the columns unused by the model forward method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_onnx_model = kwargs.pop("save_onnx_model", False)
        output_path = save_directory.joinpath(WEIGHTS_NAME)
        calibration_dataloader = None

        if INCQuantizationMode(quantization_config.approach) == INCQuantizationMode.STATIC:
            if calibration_dataset is None:
                raise ValueError("Post-training static quantization needs a calibration dataset.")
            calibration_dataloader = self._get_calibration_dataloader(
                calibration_dataset=calibration_dataset,
                batch_size=batch_size,
                remove_unused_columns=remove_unused_columns,
                data_collator=data_collator,
            )

        if isinstance(self._original_model.config, PretrainedConfig):
            self._original_model.config.backend = quantization_config.backend

        compressed_model = fit(
            self._original_model,
            conf=quantization_config,
            calib_dataloader=calibration_dataloader,
            eval_func=self.eval_fn,
            calib_func=self.calibration_fn,
        )

        if not hasattr(compressed_model, "_model") or compressed_model._model is None:
            raise RuntimeError(
                "The maximum number of trials specified has been reached and no quantized model meeting the specified"
                " accuracy tolerance has been found. Either the tolerance or the number of trials need to be increased."
            )
        if isinstance(self._original_model.config, PretrainedConfig):
            self._original_model.config.save_pretrained(save_directory)

        self._quantized_model = compressed_model._model

        if save_onnx_model:
            self._set_task()
            model_type = self._original_model.config.model_type.replace("_", "-")
            model_name = getattr(self._original_model, "name", None)
            onnx_config_class = TasksManager.get_exporter_config_constructor(
                exporter="onnx",
                model=self._original_model,
                task=self.task,
                model_type=model_type,
                model_name=model_name,
            )
            onnx_config = onnx_config_class(self._original_model.config)
            compressed_model.eval()
            output_onnx_path = save_directory.joinpath(ONNX_WEIGHTS_NAME)
            # Export the compressed model to the ONNX format
            self._onnx_export(compressed_model, onnx_config, output_onnx_path)

        # Save the quantized model
        self._save_pretrained(compressed_model, output_path)
        # TODO : Save quantization_config

    @staticmethod
    def _save_pretrained(model: Union[PyTorchModel, IPEXModel], output_path: str):
        if isinstance(model, IPEXModel):
            model._model.save(output_path)
            logger.info(f"Model weights saved to {output_path}")
            return
        state_dict = model._model.state_dict()
        if hasattr(model, "q_config"):
            state_dict["best_configure"] = model.q_config
        torch.save(state_dict, output_path)
        logger.info(f"Model weights saved to {output_path}")

    def _onnx_export(
        self,
        model: PyTorchModel,
        config: OnnxConfig,
        output_path: Union[str, Path],
    ):
        opset = min(config.DEFAULT_ONNX_OPSET, MIN_QDQ_ONNX_OPSET)
        dynamic_axes = {name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())}
        inputs = config.generate_dummy_inputs(framework="pt")
        device = model.model.device
        inputs = dict((k, v.to(device)) for k, v in inputs.items())
        torch_to_int8_onnx(
            fp32_model=self._original_model.to(device),
            int8_model=model.model,
            q_config=model.q_config,
            save_path=str(output_path),
            example_inputs=inputs,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            input_names=list(config.inputs.keys()),
            output_names=list(config.outputs.keys()),
        )

    def _set_task(self):
        if self.task is None:
            self.task = HfApi().model_info(self._original_model.config._name_or_path).pipeline_tag
            if self.task in ["sentiment-analysis", "text-classification", "zero-shot-classification"]:
                self.task = "sequence-classification"
            elif self.task in ["feature-extraction", "fill-mask"]:
                self.task = "default"
            elif self.task is None:
                raise ValueError(
                    "The task defining the model topology could not be extracted and needs to be specified for the ONNX export."
                )
        if self.task in ["seq2seq-lm", "translation", "summarization"]:
            raise ValueError(f"Seq2Seq models are currently not supported for post-training static quantization.")

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: str = "train",
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        use_auth_token: bool = False,
    ) -> Dataset:
        """
        Create the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                in generic formats and optionally a dataset script, if it requires some code to read the data files.
            num_samples (`int`, defaults to 100):
                The maximum number of samples composing the calibration dataset.
            dataset_config_name (`str`, *optional*):
                The name of the dataset configuration.
            dataset_split (`str`, defaults to `"train"`):
                Which split of the dataset to use to perform the calibration step.
            preprocess_function (`Callable`, *optional*):
                Processing function to apply to each example after loading dataset.
            preprocess_batch (`bool`, defaults to `True`):
                Whether the `preprocess_function` should be batched.
            use_auth_token (`bool`, defaults to `False`):
                Whether to use the token generated when running `transformers-cli login`.
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration step.
        """
        calibration_dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
            use_auth_token=use_auth_token,
        )

        if num_samples is not None:
            num_samples = min(num_samples, len(calibration_dataset))
            calibration_dataset = calibration_dataset.shuffle(seed=self.seed).select(range(num_samples))

        if preprocess_function is not None:
            calibration_dataset = calibration_dataset.map(preprocess_function, batched=preprocess_batch)

        return calibration_dataset

    def _get_calibration_dataloader(
        self,
        calibration_dataset: Dataset,
        batch_size: int,
        remove_unused_columns: bool,
        data_collator: Optional[DataCollator] = None,
    ) -> INCDataLoader:
        data_collator = data_collator if data_collator is not None else default_data_collator
        if remove_unused_columns:
            calibration_dataset = self._remove_unused_columns(calibration_dataset)
        self.input_names = getattr(calibration_dataset, "column_names", None)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        sampler = RandomSampler(calibration_dataset, generator=generator)
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=False,
        )

        return INCDataLoader.from_pytorch_dataloader(calibration_dataloader)

    def _remove_unused_columns(self, dataset: Dataset):
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)


# Adapted from https://github.com/intel/neural-compressor/blob/master/neural_compressor/utils/pytorch.py#L96
def _apply_quantization_from_config(q_config: Dict, model: torch.nn.Module) -> torch.nn.Module:
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

    quant_mode = INCQuantizationMode(approach)
    q_model = copy.deepcopy(model)
    q_model.eval()

    if framework == "pytorch_fx":
        op_cfgs = _cfg_to_qconfig(q_config, approach)
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, approach)

        if not q_config["fx_sub_module_list"]:
            if quant_mode == INCQuantizationMode.AWARE_TRAINING:
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
        if quant_mode == INCQuantizationMode.DYNAMIC:
            q_mapping = torch.quantization.quantization_mappings.get_default_dynamic_quant_module_mappings()
            op_cfgs = _cfg_to_qconfig(q_config, approach)
        else:
            q_mapping = torch.quantization.quantization_mappings.get_default_static_quant_module_mappings()
            op_cfgs = _cfg_to_qconfig(q_config)

        _propagate_qconfig(q_model, op_cfgs, approach=approach)

        if quant_mode != INCQuantizationMode.DYNAMIC:
            add_observer_(q_model)
        q_model = convert(q_model, mapping=q_mapping, inplace=True)

    return q_model


class INCModel:
    TRANSFORMERS_AUTO_CLASS: ClassVar = AutoModel

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the"
            f"`{self.__class__.__name__}.from_pretrained(model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, q_model_name: Optional[str] = None, **kwargs) -> torch.nn.Module:
        """
        Instantiate a quantized pytorch model from a given Intel Neural Compressor configuration file.
        Arguments:
            model_name_or_path (`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
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
            state_dict_path (`str`, *optional*):
                The path to the state dictionary of the quantized model.
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
        state_dict_path = kwargs.get("state_dict_path", None)

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

        try:
            model = model_class.from_pretrained(model_name_or_path, **kwargs)
        except AttributeError:
            init_contexts = [no_init_weights(_enable=True)]
            with ContextManagers(init_contexts):
                model = model_class(config, **kwargs)

        model_class._keys_to_ignore_on_load_unexpected = keys_to_ignore_on_load_unexpected
        model_class._keys_to_ignore_on_load_missing = keys_to_ignore_on_load_missing

        if state_dict_path is None:
            q_model_name = q_model_name if q_model_name is not None else WEIGHTS_NAME
            revision = download_kwargs.pop("revision", None)
            if os.path.isdir(model_name_or_path):
                state_dict_path = os.path.join(model_name_or_path, q_model_name)
            elif os.path.isfile(model_name_or_path):
                state_dict_path = model_name_or_path
            else:
                local_files_only = False
                if is_offline_mode():
                    logger.info("Offline mode: forcing local_files_only=True")
                    local_files_only = True
                cache_dir = download_kwargs.get("cache_dir", None)
                if cache_dir is None:
                    cache_dir = TRANSFORMERS_CACHE
                if isinstance(cache_dir, Path):
                    cache_dir = str(cache_dir)
                try:
                    state_dict_path = hf_hub_download(
                        repo_id=model_name_or_path,
                        filename=q_model_name,
                        revision=revision,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                except EnvironmentError as err:
                    logger.error(err)
                    msg = (
                        f"Can't load config for '{model_name_or_path}'. Make sure that:\n\n"
                        f"-'{model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                        f"-or '{model_name_or_path}' is a correct path to a directory containing a {q_model_name} file\n\n"
                    )

                    if revision is not None:
                        msg += (
                            f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that "
                            f"exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"
                        )

                    raise EnvironmentError(msg)

        if getattr(config, "backend", None) == "ipex":
            # NOTE: Will improve to use load function when Intel Neural Compressor next 2.1 release.
            # return load(state_dict_path)
            load_model = torch.jit.load(state_dict_path)
            load_model = torch.jit.freeze(load_model.eval())
            return load_model

        # Load the state dictionary of the model to verify whether the model is quantized or not
        state_dict = torch.load(state_dict_path)

        if "best_configure" in state_dict and state_dict["best_configure"] is not None:
            model = load(state_dict_path, model)

        return model.eval()


class INCModelForQuestionAnswering(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForQuestionAnswering


class INCModelForSequenceClassification(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForSequenceClassification


class INCModelForTokenClassification(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForTokenClassification


class INCModelForMultipleChoice(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForMultipleChoice


class INCModelForSeq2SeqLM(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForSeq2SeqLM


class INCModelForCausalLM(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForCausalLM


class INCModelForMaskedLM(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForMaskedLM


class INCModelForXLNetLM(INCModel):
    TRANSFORMERS_AUTO_CLASS = XLNetLMHeadModel


class INCModelForVision2Seq(INCModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForVision2Seq


class IncQuantizedModel(INCModel):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        warnings.warn(
            f"The class `{cls.__name__}` has been depreciated and will be removed in optimum-intel v1.7, please use "
            f"`{cls.__name__.replace('IncQuantized', 'INC')}` instead."
        )
        return super().from_pretrained(*args, **kwargs)


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


class IncQuantizedModelForVision2Seq(IncQuantizedModel):
    TRANSFORMERS_AUTO_CLASS = AutoModelForVision2Seq


class IncQuantizer(INCQuantizer):
    # Warning at import time
    warnings.warn(
        "The class `IncQuantizer` has been depreciated and will be removed in optimum-intel v1.7, please use "
        "`INCQuantizer` instead.",
    )
