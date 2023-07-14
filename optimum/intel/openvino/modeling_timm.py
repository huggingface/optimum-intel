import os
import timm
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput
from collections import OrderedDict
from typing import Mapping, Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from packaging import version
from timm.models._hub import load_model_config_from_hf
from optimum.exporters import TasksManager
from transformers import PreTrainedModel, PretrainedConfig
from optimum.exporters.onnx.model_configs import ViTOnnxConfig

ExportConfigConstructor = Callable[[PretrainedConfig], "ExportConfig"]

def get_model_from_timm(
    task: str,
    model_name_or_path: Union[str, Path],
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    torch_dtype: Optional["torch.dtype"] = None,
    device: Optional[Union["torch.device", str]] = None,
    **model_kwargs,
) -> PreTrainedModel:

    kwargs = {"subfolder": subfolder, "revision": revision, "cache_dir": cache_dir, **model_kwargs}
    kwargs["torch_dtype"] = torch_dtype
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device("cpu")
    if version.parse(torch.__version__) >= version.parse("2.0"):
        with device:
            # Initialize directly in the requested device, to save allocation time. Especially useful for large
            # models to initialize on cuda device.
            model = TimmForImageClassification.from_pretrained(model_name_or_path, **kwargs)
    else:
        model = TimmForImageClassification.from_pretrained(model_name_or_path, **kwargs).to(device)
    return model

def get_timm_exporter_config_constructor(
    exporter: str,
    model: Optional["PreTrainedModel"] = None,
    task: str = "image-classification",
    exporter_config_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ExportConfigConstructor:
    # print(exporter)
    # print(task)
    assert exporter == 'onnx' and task == "image-classification"
    exporter_config_constructor = TimmOnnxConfig

    if exporter_config_kwargs is not None:
        exporter_config_constructor = partial( exporter_config_constructor, **exporter_config_kwargs)

    return exporter_config_constructor 


TasksManager.get_model_from_timm = staticmethod(get_model_from_timm)
TasksManager.get_timm_exporter_config_constructor = staticmethod(get_timm_exporter_config_constructor)

class TimmConfig(PretrainedConfig):
    model_type = "timm"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "PretrainedConfig":

        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision


        config_dict = load_model_config_from_hf(pretrained_model_name_or_path)[0]
        config_dict["num_labels"] = config_dict.pop("num_classes")
        config_dict["image_size"] = config_dict.get("input_size")[-1]

        # print(config_dict)
        return cls.from_dict(config_dict, **kwargs)

class TimmOnnxConfig(ViTOnnxConfig):
    pass

# class TimmOnnxConfig(OnnxConfig):    
#     DEFAULT_ONNX_OPSET = 12
#     torch_onnx_minimum_version = version.parse("1.11")

#     @property
#     def inputs(self) -> Mapping[str, Mapping[int, str]]:
#         return OrderedDict(
#             [
#                 ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
#             ]
#         )

#     @property
#     def atol_for_validation(self) -> float:
#         return 1e-4


class TimmPreTrainedModel(PreTrainedModel):
    config_class = TimmConfig
    base_model_prefix = "timm"
    main_input_name = "pixel_values"

class TimmModel(TimmPreTrainedModel):
    def __init__(self, config: TimmConfig, feature_only : bool = True, pretrained : bool = True, in_chans : int = 3, **kwargs):
        super().__init__(config)

        self.config = config
        if feature_only:
            self.timm_model = timm.create_model("hf-hub" + self.config.hf_hub_id,
                                           num_classes = 0,
                                           pretrained = pretrained,
                                           in_chans = in_chans)
        else:
            self.timm_model = timm.create_model("hf-hub" + self.config.hf_hub_id,
                                           num_classes = self.config.num_labels,
                                           pretrained = pretrained,
                                           in_chans = in_chans)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        config = TimmConfig.from_pretrained(model_name_or_path, **kwargs)
        return cls(config, **kwargs)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        # expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        # if pixel_values.dtype != expected_dtype:
        #     pixel_values = pixel_values.to(expected_dtype)

        model_output = self.timm_model(pixel_values)

        if not return_dict:
            return model_output

        return BaseModelOutput(
            last_hidden_state=model_output,
            hidden_states= None
        )


class TimmForImageClassification(TimmPreTrainedModel):
    def __init__(self, config: TimmConfig, num_labels: int = None, **kwargs) -> None:
        super().__init__(config)

        if num_labels:
            config.num_labels = num_labels
        self.timm = TimmModel(config, feature_only = False)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        config = TimmConfig.from_pretrained(model_name_or_path, **kwargs)
        return cls(config, **kwargs)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits = self.timm(
            pixel_values,
            return_dict=return_dict,
        )

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            return logits

        return ImageClassifierOutput(
            loss=loss,
            logits = logits.last_hidden_state,
        )

