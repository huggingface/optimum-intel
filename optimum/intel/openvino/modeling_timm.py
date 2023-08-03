import os
from pathlib import Path
from packaging import version
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Mapping, Any, Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import ViTOnnxConfig

import timm
from timm.layers.config import set_fused_attn
from timm.models._hub import load_model_config_from_hf
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput

# from .modeling_base import OVBaseModel
from .modeling import OVModelForImageClassification
from .utils import ONNX_WEIGHTS_NAME, OV_XML_FILE_NAME

set_fused_attn(False, False)
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

        return cls.from_dict(config_dict, **kwargs)


class TimmOnnxConfig(ViTOnnxConfig):
    DEFAULT_TIMM_ONNX_OPSET = 13
    outputs= OrderedDict([('logits', {0: 'batch_size'})])



class TimmPreTrainedModel(PreTrainedModel):
    config_class = TimmConfig
    base_model_prefix = "timm"
    main_input_name = "pixel_values"


class TimmModel(TimmPreTrainedModel):
    def __init__(self, 
                config: TimmConfig, 
                feature_only : bool = True, 
                pretrained : bool = True, 
                in_chans : int = 3, 
                **kwargs):
        super().__init__(config)

        self.config = config
        if feature_only:
            self.timm_model = timm.create_model("hf-hub:" + self.config.hf_hub_id,
                                           num_classes = 0,
                                           pretrained = pretrained,
                                           in_chans = in_chans)
        else:
            self.timm_model = timm.create_model("hf-hub:" + self.config.hf_hub_id,
                                           num_classes = self.config.num_labels,
                                           pretrained = pretrained,
                                           in_chans = in_chans)
        self.timm_model.eval()

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
        super().__init__(config, **kwargs)

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


class OVModelForTimm(OVModelForImageClassification):

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
        task: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        task = task or cls.export_feature

        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
            "trust_remote_code": trust_remote_code,
        }


        # model = TasksManager.get_model_from_timm(task, model_id, **model_kwargs)
        model = TimmForImageClassification.from_pretrained(model_id, **kwargs)
        onnx_config_class = TasksManager.get_timm_exporter_config_constructor(
            exporter="onnx",
            task=task,
        )

        onnx_config = onnx_config_class(model.config)
        save_dir = TemporaryDirectory()
        
        with TemporaryDirectory() as save_dir:
            save_dir_path = Path(save_dir)
            export(
                model=model,
                config=onnx_config,
                opset=onnx_config.DEFAULT_TIMM_ONNX_OPSET,
                output=save_dir_path / ONNX_WEIGHTS_NAME,
            )

            return cls._from_pretrained(
                model_id=save_dir_path,
                config=config,
                from_onnx=True,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **kwargs,
            )

    @classmethod
    def _load_config(cls, model_id,**kwargs):
        return TimmConfig.from_pretrained(model_id, **kwargs)

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     model_id: Union[str, Path],
    #     export: bool = False,
    #     force_download: bool = False,
    #     use_auth_token: Optional[str] = None,
    #     cache_dir: Optional[str] = None,
    #     subfolder: str = "",
    #     config: Optional["PretrainedConfig"] = None,
    #     local_files_only: bool = False,
    #     trust_remote_code: bool = False,
    #     revision: Optional[str] = None,
    #     **kwargs,
    # ) -> "OptimizedModel":
    #     """
    #     Returns:
    #         `OptimizedModel`: The loaded optimized model.
    #     """
    #     if isinstance(model_id, Path):
    #         model_id = model_id.as_posix()

    #     from_transformers = kwargs.pop("from_transformers", None)
    #     if from_transformers is not None:
    #         logger.warning(
    #             "The argument `from_transformers` is deprecated, and will be removed in optimum 2.0.  Use `export` instead"
    #         )
    #         export = from_transformers

    #     if len(model_id.split("@")) == 2:
    #         if revision is not None:
    #             logger.warning(
    #                 f"The argument `revision` was set to {revision} but will be ignored for {model_id.split('@')[1]}"
    #             )
    #         model_id, revision = model_id.split("@")

    #     if config is None:
    #         if os.path.isdir(os.path.join(model_id, subfolder)) and cls.config_name == CONFIG_NAME:
    #             if CONFIG_NAME in os.listdir(os.path.join(model_id, subfolder)):
    #                 config = AutoConfig.from_pretrained(os.path.join(model_id, subfolder, CONFIG_NAME))
    #             elif CONFIG_NAME in os.listdir(model_id):
    #                 config = AutoConfig.from_pretrained(os.path.join(model_id, CONFIG_NAME))
    #                 logger.info(
    #                     f"config.json not found in the specified subfolder {subfolder}. Using the top level config.json."
    #                 )
    #             else:
    #                 raise OSError(f"config.json not found in {model_id} local folder")
    #         else:
    #             config = cls._load_config(
    #                 model_id,
    #                 revision=revision,
    #                 cache_dir=cache_dir,
    #                 use_auth_token=use_auth_token,
    #                 force_download=force_download,
    #                 subfolder=subfolder,
    #             )
    #     elif isinstance(config, (str, os.PathLike)):
    #         config = cls._load_config(
    #             config,
    #             revision=revision,
    #             cache_dir=cache_dir,
    #             use_auth_token=use_auth_token,
    #             force_download=force_download,
    #             subfolder=subfolder,
    #         )

    #     if not export and trust_remote_code:
    #         logger.warning(
    #             "The argument `trust_remote_code` is to be used along with export=True. It will be ignored."
    #         )
    #     elif export and trust_remote_code is None:
    #         trust_remote_code = False

    #     from_pretrained_method = cls._from_transformers if export else cls._from_pretrained
    #     return from_pretrained_method(
    #         model_id=model_id,
    #         config=config,
    #         revision=revision,
    #         cache_dir=cache_dir,
    #         force_download=force_download,
    #         use_auth_token=use_auth_token,
    #         subfolder=subfolder,
    #         local_files_only=local_files_only,
    #         trust_remote_code=trust_remote_code,
    #         **kwargs,
    #     )