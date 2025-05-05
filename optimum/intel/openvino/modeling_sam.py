import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import openvino as ov
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation
from transformers import AutoConfig, PretrainedConfig, SamModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.sam.modeling_sam import SamImageSegmentationOutput, SamPositionalEmbedding

from ...exporters.openvino.utils import save_config
from .modeling_base import OVBaseModel, OVModelPart
from .utils import (
    ONNX_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME,
    ONNX_VISION_ENCODER_MODEL_NAME,
    OV_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME,
    OV_VISION_ENCODER_MODEL_NAME,
    TemporaryDirectory,
    model_has_dynamic_inputs,
)


logger = logging.getLogger(__name__)

core = ov.Core()


class OVSamVisionEncoder(OVModelPart):
    _model_name = "vision_encoder"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)

    def forward(self, pixel_values):
        self._compile()
        inputs = {"pixel_values": pixel_values}
        result = self.request(inputs)
        image_embeddings = result["image_embeddings"]
        image_positional_embeddings = result["image_positional_embeddings"]
        return ModelOutput(image_embeddings=image_embeddings, image_positional_embeddings=image_positional_embeddings)


class OVSamPromptEncoder(OVModelPart):
    _model_name = "prompt_encoder_mask_decoder"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)

    def forward(self, image_embeddings, image_positional_embeddings, input_points, input_labels=None):
        self._compile()
        inputs = {
            "image_embeddings": image_embeddings,
            "image_positional_embeddings": image_positional_embeddings,
            "input_points": input_points,
        }

        if input_labels is None:
            input_labels = np.ones(input_points[:, :, :, 0].shape, dtype=int)
        inputs["input_labels"] = input_labels

        result = self.request(inputs)
        iou_scores = torch.from_numpy(result["iou_scores"])
        pred_masks = torch.from_numpy(result["pred_masks"])

        return SamImageSegmentationOutput(iou_scores=iou_scores, pred_masks=pred_masks)


class OVSamModel(OVBaseModel):
    export_feature = "feature-extraction"
    auto_model_class = SamModel

    def __init__(
        self,
        vision_encoder_model: ov.Model,
        prompt_encoder_mask_decoder_model: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.config = config
        self._model_save_dir = model_save_dir
        self._device = device.upper()
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])
        self.vision_encoder_model = vision_encoder_model
        self.prompt_encoder_mask_decoder_model = prompt_encoder_mask_decoder_model
        self._compile_only = kwargs.get("compile_only", False)
        enable_compilation = kwargs.get("compile", True)
        self.vision_encoder = OVSamVisionEncoder(self.vision_encoder_model, self)
        self.prompt_encoder_mask_decoder = OVSamPromptEncoder(self.prompt_encoder_mask_decoder_model, self)

        if dynamic_shapes and not self.is_dynamic and not self._compile_only:
            self.reshape()

        if enable_compilation and not self._compile_only:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        try:
            self.auto_model_class.register(AutoConfig, self.__class__)
        except AttributeError:
            pass
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)
        self.positional_embeddings = self.get_image_wide_positional_embeddings()

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        for _, component in self.components.items():
            component.clear_requests()

    def compile(self):
        self.vision_encoder._compile()
        self.prompt_encoder_mask_decoder._compile()

    def _save_config(self, save_directory):
        """
        Saves a model configuration into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        save_config(self.config, save_directory)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        src_models = self.ov_submodels
        dst_file_names = {
            "vision_encoder_model": OV_VISION_ENCODER_MODEL_NAME,
            "prompt_encoder_mask_decoder_model": OV_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME,
        }

        for name in self._ov_submodel_names:
            model = src_models[name]
            dst_file_name = dst_file_names[name]
            dst_path = os.path.join(save_directory, dst_file_name)
            ov.save_model(model, dst_path, compress_to_fp16=False)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        vision_encoder_file_name: Optional[str] = None,
        prompt_encoder_mask_decoder_file_name: Optional[str] = None,
        local_files_only: bool = False,
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
            use_auth_token (Optional[Union[bool, str]], defaults to `None`):
                Deprecated. Please use `token` instead.
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            vision_encoder_file_name(`str`, *optional*):
                The vision encoder model file name. Overwrites the default file name openvino_vision_encoder.xml and allows one to
                load the vision encoder model with a different name.
            prompt_encoder_mask_decoder_file_name(`str`, *optional*):
                The prompt encdoer and mask decoder model file name overwriting the default file name
                openvino_prompt_encoder_mask_decoder.xml, allowing to load the decoder model with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        from_onnx = kwargs.get("from_onnx", False)

        default_vision_encoder_file_name = (
            ONNX_VISION_ENCODER_MODEL_NAME if from_onnx else OV_VISION_ENCODER_MODEL_NAME
        )
        default_prompt_encoder_mask_decoder_file_name = (
            ONNX_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME if from_onnx else OV_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME
        )
        vision_encoder_file_name = vision_encoder_file_name or default_vision_encoder_file_name
        prompt_encoder_mask_decoder_file_name = (
            prompt_encoder_mask_decoder_file_name or default_prompt_encoder_mask_decoder_file_name
        )

        model_file_names = {
            "vision_encoder_model": vision_encoder_file_name,
            "prompt_encoder_mask_decoder_model": prompt_encoder_mask_decoder_file_name,
        }

        if not from_onnx:
            model_file_names.update(
                {
                    "vision_encoder_model_bin": vision_encoder_file_name.replace(".xml", ".bin"),
                    "prompt_encoder_mask_decoder_model_bin": prompt_encoder_mask_decoder_file_name.replace(
                        ".xml", ".bin"
                    ),
                }
            )

        compile_only = kwargs.get("compile_only", False)
        if os.path.isdir(model_id):
            # Load model from a local directory
            model_save_dir = Path(model_id)
            file_names = {k: os.path.join(model_id, model_file_names[k]) for k in model_file_names}
        else:
            file_names = {}
            for name, file_name in model_file_names.items():
                model_cache_path = cls._cached_file(
                    model_path=model_id,
                    file_name=file_name,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                file_names[name] = model_cache_path
            model_save_dir = Path(model_cache_path).parent
        if not compile_only:
            vision_encoder_model = cls.load_model(file_names["vision_encoder_model"])
            prompt_encoder_model = cls.load_model(file_names["prompt_encoder_mask_decoder_model"])
        else:
            vision_encoder_model = cls._compile_model(
                file_names["vision_encoder_model"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
            prompt_encoder_model = cls._compile_model(
                file_names["prompt_encoder_mask_decoder_model"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )

        model = cls(
            vision_encoder_model=vision_encoder_model,
            prompt_encoder_mask_decoder_model=prompt_encoder_model,
            config=config,
            model_save_dir=model_save_dir,
            **kwargs,
        )

        return model

    @property
    def _ov_submodel_names(self):
        model_names = ["vision_encoder_model", "prompt_encoder_mask_decoder_model"]
        return model_names

    def reshape(self, batch_size: int = -1, point_batch_size: int = -1, num_points_per_image: int = -1):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            point_batch_size (`int`):
                The point batch size.
            num_points_per_image (`int`, *optional*):
                The number of points per image
        """
        if self._compile_only:
            raise ValueError(
                "`reshape()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        vision_encoder_shapes = {}
        for inputs in self.vision_encoder_model.inputs:
            vision_encoder_shapes[inputs] = inputs.get_partial_shape()
            vision_encoder_shapes[inputs][0] = batch_size
        self.vision_encoder_model.reshape(vision_encoder_shapes)
        self.vision_encoder.request = None
        mask_decoder_shapes = {}
        for inputs in self.prompt_encoder_mask_decoder_model.inputs:
            mask_decoder_shapes[inputs] = inputs.get_partial_shape()
            mask_decoder_shapes[inputs][0] = batch_size
            if inputs.get_any_name() in ["input_points", "input_labels"]:
                mask_decoder_shapes[inputs][1] = point_batch_size
                mask_decoder_shapes[inputs][2] = num_points_per_image
        self.prompt_encoder_mask_decoder_model.reshape(mask_decoder_shapes)
        self.prompt_encoder_mask_decoder.request = None
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        for submodel in self.ov_submodels.values():
            apply_moc_transformations(submodel, cf=False)
            compress_model_transformation(submodel)
        return self

    def forward(
        self,
        pixel_values: Optional[torch.LongTensor] = None,
        input_points: Optional[torch.LongTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either `pixel_values` or `image_embeddings` must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of `pixel_values` and `image_embeddings` can be provided.")

        if input_points is None:
            raise ValueError("`input_points` must be provided.")

        if len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )

        if pixel_values is not None:
            vision_out = self.vision_encoder(pixel_values)
            image_embeddings = vision_out.image_embeddings
            image_positional_embeddings = vision_out.image_positional_embeddings
        else:
            image_positional_embeddings = self.positional_embeddings
            # repeat with batch size
            batch_size = image_embeddings.shape[0]
            image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )
        return self.prompt_encoder_mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            input_points=input_points,
            input_labels=input_labels,
        )

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    def get_image_features(self, pixel_values, *args, **kwargs):
        return torch.from_numpy(self.vision_encoder(pixel_values).image_embeddings)

    @property
    def is_dynamic(self):
        return model_has_dynamic_inputs(self.vision_encoder_model) or model_has_dynamic_inputs(
            self.prompt_encoder_mask_decoder_model
        )
