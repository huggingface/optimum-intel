import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
import timm
import torch
from packaging import version
from timm.layers.config import set_fused_attn
from timm.models._hub import load_model_config_from_hf
from transformers import PretrainedConfig, PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import resize, to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageFeatureExtractionMixin,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.utils import TensorType

from optimum.exporters.onnx.config import VisionOnnxConfig
from optimum.utils import NormalizedVisionConfig

from .utils import _is_timm_ov_dir


set_fused_attn(False, False)


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
        if _is_timm_ov_dir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            return cls.from_json_file(config_path)

        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        config_dict = load_model_config_from_hf(pretrained_model_name_or_path)[0]
        config_dict["num_labels"] = config_dict.pop("num_classes")
        config_dict["image_size"] = config_dict.get("input_size")[-1]

        return cls.from_dict(config_dict, **kwargs)


class TimmOnnxConfig(VisionOnnxConfig):
    DEFAULT_TIMM_ONNX_OPSET = 13
    outputs = OrderedDict([("logits", {0: "batch_size"})])
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    MIN_TORCH_VERSION = version.parse("1.11")

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}


class TimmForImageClassification(PreTrainedModel):
    def __init__(self, config: TimmConfig, num_labels: int = None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if num_labels:
            config.num_labels = num_labels
        self.model = timm.create_model(
            "hf-hub:" + self.config.hf_hub_id,
            num_classes=self.config.num_labels,
            pretrained=True,
            in_chans=3,
        )
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        config = TimmConfig.from_pretrained(model_name_or_path, **kwargs)
        return cls(config, **kwargs)

    def forward(self, pixel_values: Optional[torch.Tensor] = None):
        logits = self.model(
            pixel_values,
        )

        return ImageClassifierOutput(
            logits=logits,
        )


# Adapted from ViTImageProcessor - https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/image_processing_vit.py
class TimmImageProcessor(BaseImageProcessor, ImageFeatureExtractionMixin):
    r"""
    Constructs a ViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        timm_config_dict, _ = load_model_config_from_hf(pretrained_model_name_or_path)

        _, im_h, im_w = timm_config_dict.get("input_size", [3, 224, 224])

        image_preprocess_config_dict = {
            "crop_size": {"height": im_h, "width": im_w},
            "do_center_crop": True if timm_config_dict.get("crop_mode") == "center" else False,
            "do_normalize": True,
            "do_reduce_labels": False,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": timm_config_dict.get("mean", IMAGENET_STANDARD_MEAN),
            "image_processor_type": "TimmImageProcessor",
            "image_std": timm_config_dict.get("std", IMAGENET_STANDARD_STD),
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": {"height": im_h, "width": im_w},
        }

        return cls.from_dict(image_preprocess_config_dict, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample:
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return resize(
            image, size=(size["height"], size["width"]), resample=resample, data_format=data_format, **kwargs
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size_dict, resample=resample) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
