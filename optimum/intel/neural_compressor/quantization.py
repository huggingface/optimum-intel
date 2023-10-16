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
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
from datasets import Dataset, load_dataset
from neural_compressor.adaptor.pytorch import PyTorch_FXAdaptor, _cfg_to_qconfig, _propagate_qconfig
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.experimental.export import torch_to_int8_onnx
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.model.torch_model import IPEXModel, PyTorchModel
from neural_compressor.quantization import fit
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    DataCollator,
    PretrainedConfig,
    PreTrainedModel,
    default_data_collator,
)

from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfig
from optimum.onnxruntime import ORTModel
from optimum.onnxruntime.modeling_decoder import ORTModelDecoder
from optimum.onnxruntime.modeling_seq2seq import ORTModelForConditionalGeneration
from optimum.onnxruntime.utils import ONNX_DECODER_NAME
from optimum.quantization_base import OptimumQuantizer

from ..utils.constant import _TASK_ALIASES, MIN_QDQ_ONNX_OPSET, ONNX_WEIGHTS_NAME, WEIGHTS_NAME
from ..utils.import_utils import (
    _ipex_version,
    _neural_compressor_version,
    is_ipex_version,
    is_neural_compressor_version,
)
from .configuration import INCConfig
from .modeling_base import (  # noqa
    INCModel,
    INCModelForMaskedLM,
    INCModelForMultipleChoice,
    INCModelForQuestionAnswering,
    INCModelForSeq2SeqLM,
    INCModelForSequenceClassification,
    INCModelForTokenClassification,
    INCModelForVision2Seq,
    INCModelForXLNetLM,
)
from .utils import INCDataLoader, _cfgs_to_fx_cfgs


logger = logging.getLogger(__name__)

NEURAL_COMPRESSOR_MINIMUM_VERSION = "2.1.0"
NEURAL_COMPRESSOR_WEIGHT_ONLY_MINIMUM_VERSION = "2.3.0"
IPEX_MINIMUM_VERSION = "2.1.0"

if is_neural_compressor_version("<", NEURAL_COMPRESSOR_MINIMUM_VERSION):
    raise ImportError(
        f"Found an incompatible version of neural-compressor. Found version {_neural_compressor_version}, "
        f"but only version {NEURAL_COMPRESSOR_MINIMUM_VERSION} or higher is supported."
    )


class INCQuantizationMode(Enum):
    DYNAMIC = "post_training_dynamic_quant"
    STATIC = "post_training_static_quant"
    AWARE_TRAINING = "quant_aware_training"
    WEIGHT_ONLY = "post_training_weight_only"


SUPPORTED_QUANT_MODE = {approach.value for approach in INCQuantizationMode}


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
        self.eval_fn = eval_fn if eval_fn is not None else lambda model: 1
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
        file_name: str = None,
        weight_only: bool = False,
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
            weight_only (`bool`, defaults to `False`):
                Whether compress weights to integer precision (4-bit by default) while keeping activations
                floating-point. Fits best for LLM footprint reduction and performance acceleration.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_onnx_model = kwargs.pop("save_onnx_model", False)

        if save_onnx_model and isinstance(self._original_model, ORTModel):
            save_onnx_model = False
            logger.warning("Model provided is an ONNX model, `save_onnx_model` is set to False")

        default_name = WEIGHTS_NAME if not isinstance(self._original_model, ORTModel) else ONNX_WEIGHTS_NAME
        calibration_dataloader = None
        self._set_task()

        if weight_only:
            # check neural-compressor version
            if is_neural_compressor_version("<", NEURAL_COMPRESSOR_WEIGHT_ONLY_MINIMUM_VERSION):
                raise ImportError(
                    f"Found an incompatible version of neural-compressor. Found version {_neural_compressor_version}, "
                    f"but only version {NEURAL_COMPRESSOR_WEIGHT_ONLY_MINIMUM_VERSION} or higher supports weight-only quantization."
                )

            # If op_type_dict of quantization_config is not defined, it will use default values for weight-only quantization:
            # {"bits": 4, "group_size": 32, "scheme": "sym", "algorithm": "RTN"}
            if isinstance(quantization_config.op_type_dict, dict) and len(quantization_config.op_type_dict) > 0:
                algo = []
                for _, val in quantization_config.op_type_dict.items():
                    algo += val.get("weight", {}).get("algorithm", ["RTN"])
            else:
                algo = ["RTN"]

            if calibration_dataset is None and ("GPTQ" in algo or "AWQ" in algo):
                raise ValueError(
                    "Weight-only quantization needs a calibration dataset for both GPTQ and AWQ methodologies."
                )

            if calibration_dataset is None:
                calibration_dataloader = None
            else:
                calibration_dataloader = self._get_calibration_dataloader(
                    calibration_dataset=calibration_dataset,
                    batch_size=batch_size,
                    remove_unused_columns=remove_unused_columns,
                    data_collator=data_collator,
                    use_label=False if "GPTQ" in algo else True,
                )

        elif INCQuantizationMode(quantization_config.approach) == INCQuantizationMode.STATIC:
            # Since PyTorch fx trace does not really require an example_inputs, only need calibration_dataset or calibration_fn here.
            if calibration_dataset is None and self.calibration_fn is None:
                raise ValueError(
                    "Post-training static quantization needs a calibration dataset or a calibration_function."
                )
            if calibration_dataset is None:
                calibration_dataloader = None
            else:
                quantization_config.calibration_sampling_size = len(calibration_dataset)
                calibration_dataloader = self._get_calibration_dataloader(
                    calibration_dataset=calibration_dataset,
                    batch_size=batch_size,
                    remove_unused_columns=remove_unused_columns,
                    data_collator=data_collator,
                )
            op_type_dict = getattr(quantization_config, "op_type_dict", None)
            if op_type_dict is None or "Embedding" not in op_type_dict:
                logger.warning("ONNX export is no supported for model with quantized embeddings")
                save_onnx_model = False

        else:
            # Disable ONNX export for dynamically quantized model as deprecated in neural-compressor>=2.2.0
            if save_onnx_model:
                logger.warning(
                    "ONNX export for dynamic quantized model is no longer supported by neural-compressor>=2.2.0. "
                    "To apply dynamic quantization on an ONNX model, you can use optimum.onnxruntime.ORTQuantizer"
                )
                save_onnx_model = False

        if (
            quantization_config.backend == "ipex"
            and is_ipex_version("<", IPEX_MINIMUM_VERSION)
            and "generation" in self.task
        ):
            raise ImportError(
                f"Found an incompatible version of intel-extension-for-pytorch. Found version {_ipex_version}, "
                f"but only version {IPEX_MINIMUM_VERSION} or higher is supported."
            )

        if isinstance(self._original_model.config, PretrainedConfig):
            self._original_model.config.backend = quantization_config.backend

        if isinstance(self._original_model, ORTModel):
            # TODO : enable seq2seq models
            if isinstance(self._original_model, ORTModelForConditionalGeneration):
                raise RuntimeError("ORTModelForConditionalGeneration not supported for quantization")

            if isinstance(self._original_model, ORTModelDecoder):
                model_or_path = self._original_model.onnx_paths
                if len(model_or_path) > 1:
                    raise RuntimeError(
                        f"Too many ONNX model files were found in {self._original_model.onnx_paths}, only `use_cache=False` is supported"
                    )
                model_or_path = str(model_or_path[0])
                default_name = ONNX_DECODER_NAME
            else:
                model_or_path = str(self._original_model.model_path)
        else:
            model_or_path = self._original_model

        compressed_model = fit(
            model_or_path,
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
            # If backend is IPEX, then the quantized model is JIT model which will drop the config attribute,
            # so need set config from original_model.
            model_config = copy.deepcopy(self._original_model.config)
            model_config.torch_dtype = "int8"
            if isinstance(compressed_model, IPEXModel):
                model_config.torchscript = True
                model_config.backend = "ipex"
            elif not isinstance(compressed_model, ONNXModel):
                compressed_model._model.config = model_config
            model_config.save_pretrained(save_directory)

        self._quantized_model = compressed_model._model

        if save_onnx_model:
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

        output_path = save_directory.joinpath(file_name or default_name)
        # Save the quantized model
        self._save_pretrained(compressed_model, output_path)
        quantization_config = INCConfig(quantization=quantization_config, save_onnx_model=save_onnx_model)
        quantization_config.save_pretrained(save_directory)

    @staticmethod
    def _save_pretrained(model: Union[PyTorchModel, IPEXModel], output_path: str):
        if isinstance(model, IPEXModel):
            model._model.save(output_path)
        elif isinstance(model, ONNXModel):
            model.save(output_path)
        else:
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
        opset = max(config.DEFAULT_ONNX_OPSET, MIN_QDQ_ONNX_OPSET)
        dynamic_axes = dict(chain(config.inputs.items(), config.outputs.items()))
        inputs = config.generate_dummy_inputs(framework="pt")
        device = model.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if is_neural_compressor_version(">", "2.2.1"):
            torch_to_int8_onnx(
                self._original_model,
                model.model,
                q_config=model.q_config,
                save_path=str(output_path),
                example_inputs=inputs,
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                input_names=list(config.inputs.keys()),
                output_names=list(config.outputs.keys()),
            )
        else:
            torch_to_int8_onnx(
                model.model,
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
            try:
                self.task = TasksManager.infer_task_from_model(self._original_model.config._name_or_path)
            except Exception as e:
                self.task = "default"
                logger.warning(
                    f"The task could not be automatically inferred and will be set to {self.task}. "
                    f"Please provide the task argument with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                )

        self.task = _TASK_ALIASES.get(self.task, self.task)

        if self.task == "text2text-generation":
            raise ValueError("Seq2Seq models are currently not supported for post-training static quantization.")

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
        use_label: Optional[bool] = True,
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

        return INCDataLoader.from_pytorch_dataloader(calibration_dataloader, use_label)

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
    from torch.quantization import add_observer_, convert
    from torch.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

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
