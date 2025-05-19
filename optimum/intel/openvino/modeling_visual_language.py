import copy
import enum
import inspect
import logging
import math
import os
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino as ov
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import ModelOutput

from ...exporters.openvino import main_export
from ...exporters.openvino.stateful import ensure_stateful_is_available, model_has_input_output_name
from ...exporters.openvino.utils import save_config
from ..utils.import_utils import is_transformers_version
from .configuration import OVConfig, OVWeightQuantizationConfig
from .modeling_base import OVBaseModel, OVModelPart
from .modeling_decoder import CausalLMOutputWithPast, OVModelForCausalLM
from .utils import (
    OV_LANGUAGE_MODEL_NAME,
    OV_TEXT_EMBEDDINGS_MODEL_NAME,
    OV_VISION_EMBEDDINGS_MODEL_NAME,
    TemporaryDirectory,
)


try:
    from transformers import LlavaForConditionalGeneration
except ImportError:
    LlavaForConditionalGeneration = None

try:
    from transformers import LlavaNextForConditionalGeneration
except ImportError:
    LlavaNextForConditionalGeneration = None


if TYPE_CHECKING:
    from PIL.Image import Image

    if is_transformers_version(">=", "4.42.0"):
        from transformers.image_utils import VideoInput
    else:
        VideoInput = List[Image]


logger = logging.getLogger(__name__)

core = ov.Core()


class InputMode(enum.Enum):
    LANGUAGE = 0
    VISION = 1
    SPEECH = 2
    VISION_SPEECH = 3


class OVModelWithEmbedForCausalLM(OVModelForCausalLM):
    def __init__(
        self,
        model: ov.Model,
        text_embeds_model: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        self.model = model
        self.text_emb_model = text_embeds_model
        self.request = None
        self.text_emb_request = None
        compile_only = kwargs.get("compile_only", False)
        if compile_only:
            self.text_emb_request = self.text_emb_model
            self.request = self.model.create_infer_request()

        super().__init__(
            model, config, device, dynamic_shapes, ov_config, model_save_dir, quantization_config, **kwargs
        )

    def compile(self):
        if self.request is None:
            logger.info(f"Compiling the Language model to {self._device} ...")
            super().compile()
        self._compile_text_emb()

    def _compile_text_emb(self):
        if self.text_emb_request is None:
            logger.info(f"Compiling the Text embeddings model to {self._device} ...")
            if self._compile_only:
                self.text_emb_request = self.text_emb_model
            else:
                logger.info(f"Compiling the Text embeddings model to {self._device} ...")
                self.text_emb_request = self._compile_model(
                    self.text_emb_model, self._device, self.ov_config, self.model_save_dir
                )

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        self.request = None
        self.text_emb_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_text_emb()
        res = self.text_emb_request(input_ids, share_inputs=True)
        return res[0]

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        inputs = {}
        # past_key_values are not used explicitly, instead they are handled inside the model
        if past_key_values is None:
            # This is the first iteration in a sequence, reset all states
            if self.request is not None:
                self.request.reset_state()
                # Set initial value for the next beam_idx input that will be used at the current iteration
                # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
                self.next_beam_idx = np.arange(batch_size, dtype=int)
                self._past_length = 0
        past_len = self._get_past_length(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids if past_key_values is None else input_ids[:, -1:])

            if hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb
        inputs["inputs_embeds"] = inputs_embeds

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names or "position_ids" in self.input_names:
            if attention_mask is not None:
                attention_mask = attention_mask.cpu().numpy()
            else:
                attention_mask = np.ones((inputs_embeds.shape[0], inputs_embeds.shape[1] + past_len), dtype=int)

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = position_ids.cpu().numpy()
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
            if past_len:
                position_ids = position_ids[:, -inputs_embeds.shape[1] :]

            if self.config.model_type == "qwen2_vl" and position_ids.ndim != 3:
                position_ids = np.repeat(np.expand_dims(position_ids, 0), 3, axis=0)

            inputs["position_ids"] = position_ids

        if "token_type_ids" in self.input_names:
            if token_type_ids is None:
                token_type_ids = np.zeros(inputs_embeds.shape[:2], dtype=int)
            inputs["token_type_ids"] = token_type_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
            )

        return inputs

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        self.compile()

        inputs = self.prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        logits = self.request.get_tensor("logits").data
        logits = torch.from_numpy(logits).clone().to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class OVVisionEmbedding(OVModelPart):
    _model_name = "vision_embeddings"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.hidden_states_output_names = []
        if len(self.model.outputs) > 2:
            self.hidden_states_output_names = [
                key.get_any_name() for key in self.model.outputs[2:] if "hidden_states" in key.get_any_name()
            ]
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        if model_has_input_output_name(self.model, "images"):
            self._main_input = "images"
        elif model_has_input_output_name(self.model, "hidden_states"):
            self._main_input = "hidden_states"
        else:
            self._main_input = "pixel_values"

    def forward(self, pixel_values, **kwargs):
        self._compile()
        inputs = {self._main_input: pixel_values}
        if len(self.input_names) > 1:
            for name in self.input_names:
                if name in kwargs:
                    inputs[name] = kwargs[name]
        result = self.request(inputs)
        last_hidden_state = result[0]
        hidden_states = None
        pooler_out = None
        if len(result) > 1:
            pooler_out = result[1]
            if self.hidden_states_output_names:
                hidden_states = []
                for out in self.hidden_states_output_names:
                    hidden_states.append(result[out])
        return BaseModelOutputWithPooling(
            pooler_output=pooler_out, last_hidden_state=last_hidden_state, hidden_states=hidden_states
        )


class OVResampler(OVModelPart):
    _model_name = "resampler"

    def __init__(self, model: ov.Model, parent_model: OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}

    def forward(self, image_feature, pos_embed, key_padding_mask):
        self._compile()
        result = self.request(
            {"image_feature": image_feature, "pos_embed": pos_embed, "key_padding_mask": key_padding_mask}
        )[0]
        return result


class OVVisionProjection(OVModelPart):
    _model_name = "vision_projection"

    def forward(self, img_features):
        self._compile()
        return self.request(img_features)[0]


class OVVisionResampler(OVVisionProjection):
    _model_name = "vision_resampler"


class OVMultiModalProjector(OVVisionProjection):
    _model_name = "multi_modal_projector"


class OVAudioEmbeddings(OVModelPart):
    _model_name = "audio_embeddings"

    def forward(self, audio_signal):
        self._compile()
        return self.request(audio_signal)[0]


class OVAudioEncoder(OVModelPart):
    _model_name = "audio_encoder"

    def forward(self, audio_feature, audio_mask):
        self._compile()
        return self.request({"audio_feature": audio_feature, "audio_mask": audio_mask})[0]


MODEL_PARTS_CLS_MAPPING = {
    "resampler": OVResampler,
    "language_model": OVModelWithEmbedForCausalLM,
    "vision_embeddings": OVVisionEmbedding,
    "vision_projection": OVVisionProjection,
    "vision_resampler": OVVisionResampler,
    "multi_modal_projector": OVMultiModalProjector,
    "vision_embeddings_merger": OVVisionEmbedding,
    "audio_embeddings": OVAudioEmbeddings,
    "audio_forward_embeddings": OVAudioEmbeddings,
    "audio_encoder": OVAudioEncoder,
    "audio_vision_projection": OVAudioEmbeddings,
    "audio_speech_projection": OVAudioEmbeddings,
}


class OVModelForVisualCausalLM(OVBaseModel, GenerationMixin):
    export_feature = "image-text-to-text"
    additional_parts = []
    auto_model_class = AutoModelForCausalLM

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_embeddings: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        self.config = config
        self.use_cache = kwargs.get("use_cache", True)
        self._model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])
        self.lm_model = language_model
        self.text_embeddings_model = text_embeddings
        self.vision_embeddings_model = vision_embeddings
        self._supports_cache_class = False
        self.main_input_name = "input_ids"
        self._compile_only = kwargs.get("compile_only", False)

        for part in self.additional_parts:
            setattr(self, f"{part}_model", kwargs.get(part))

        enable_compilation = kwargs.get("compile", True)
        self.generation_config = kwargs.get("generation_config", GenerationConfig.from_model_config(config))
        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()
        self.language_model = OVModelWithEmbedForCausalLM(
            self.lm_model,
            self.text_embeddings_model,
            config=config,
            device=device,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            compile=self._compile_only or enable_compilation,
            compile_only=self._compile_only,
        )
        self.vision_embeddings = OVVisionEmbedding(self.vision_embeddings_model, self)
        for part in self.additional_parts:
            model_part = getattr(self, f"{part}_model", None)
            if model_part is not None:
                model_part = MODEL_PARTS_CLS_MAPPING[part](model_part, self)
            setattr(self, part, model_part)

        if enable_compilation and not self._compile_only:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        try:
            self.auto_model_class.register(AutoConfig, self.__class__)
        except AttributeError:
            pass

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        for _, component in self.components.items():
            component.clear_requests()

    def compile(self):
        for _, component in self.components.items():
            if isinstance(component, OVModelPart):
                component._compile()
            else:
                component.compile()

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
            "lm_model": OV_LANGUAGE_MODEL_NAME,
            "text_embeddings_model": OV_TEXT_EMBEDDINGS_MODEL_NAME,
            "vision_embeddings_model": OV_VISION_EMBEDDINGS_MODEL_NAME,
        }
        for name in self._ov_submodel_names:
            if name not in dst_file_names:
                dst_file_names[name] = f"openvino_{name}.xml"

        for name in self._ov_submodel_names:
            model = src_models[name]
            dst_file_name = dst_file_names[name]
            dst_path = os.path.join(save_directory, dst_file_name)
            ov.save_model(model, dst_path, compress_to_fp16=False)

        self._save_openvino_config(save_directory)
        if self.generation_config is not None:
            try:
                self.generation_config.save_pretrained(save_directory)
            except Exception as exception:
                logger.warning(
                    f"The generation config will not be saved, saving failed with following error:\n{exception}"
                )

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
        local_files_only: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
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
            encoder_file_name(`str`, *optional*):
                The encoder model file name. Overwrites the default file name openvino_encoder_model.xml and allows one to
                load the encoder model with a different name.
            decoder_file_name(`str`, *optional*):
                The decoder model file name. Overwrites the default file name openvino_decoder_model.xml and allows one to
                load the decoder model with a different name.
            decoder_with_past_file_name(`str`, *optional*):
                The decoder with past key values model file name overwriting the default file name
                openvino_decoder_with_past_model.xml, allowing to load the decoder model with a different name.
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

        model_file_names = {
            "language_model": OV_LANGUAGE_MODEL_NAME,
            "language_model_bin": OV_LANGUAGE_MODEL_NAME.replace(".xml", ".bin"),
            "text_embeddings": OV_TEXT_EMBEDDINGS_MODEL_NAME,
            "text_embeddings_bin": OV_TEXT_EMBEDDINGS_MODEL_NAME.replace(".xml", ".bin"),
            "vision_embeddings": OV_VISION_EMBEDDINGS_MODEL_NAME,
            "vision_embeddings_bin": OV_VISION_EMBEDDINGS_MODEL_NAME.replace(".xml", ".bin"),
        }

        model_cls = MODEL_TYPE_TO_CLS_MAPPING[config.model_type]
        for part in model_cls.additional_parts:
            model_file_names[part] = f"openvino_{part}_model.xml"
            model_file_names[part + "_bin"] = f"openvino_{part}_model.bin"
        compile_only = kwargs.get("compile_only", False)
        if os.path.isdir(model_id):
            # Load model from a local directory
            model_save_dir = Path(model_id)
            file_names = {k: os.path.join(model_id, model_file_names[k]) for k in model_file_names}
        else:
            file_names = {}
            for name, file_name in model_file_names.items():
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_name,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                file_names[name] = model_cache_path
            model_save_dir = Path(model_cache_path).parent
        if not compile_only:
            language_model = model_cls.load_model(file_names["language_model"])
            text_embeddings = model_cls.load_model(file_names["text_embeddings"])
            vision_embeddings = model_cls.load_model(file_names["vision_embeddings"])
            for part in model_cls.additional_parts:
                kwargs[part] = model_cls.load_model(file_names[part])
        else:
            language_model = model_cls._compile_model(
                file_names["language_model"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
            text_embeddings = model_cls._compile_model(
                file_names["text_embeddings"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
            vision_embeddings = model_cls._compile_model(
                file_names["vision_embeddings"],
                kwargs.get("device", "CPU"),
                kwargs.get("ov_config"),
                model_save_dir,
            )
            for part in model_cls.additional_parts:
                kwargs[part] = model_cls._compile_model(
                    file_names[part],
                    kwargs.get("device", "CPU"),
                    kwargs.get("ov_config"),
                    model_save_dir,
                )
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            kwargs["generation_config"] = generation_config
        except Exception:
            pass

        quantization_config = model_cls._prepare_quantization_config(quantization_config, load_in_8bit)
        to_quantize = not compile_only and quantization_config is not None
        if to_quantize:
            kwargs["compile"] = False

        model = model_cls(
            language_model=language_model,
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            config=config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )

        if to_quantize:
            from optimum.intel.openvino.quantization import OVQuantizer

            quantization_config_copy = copy.deepcopy(quantization_config)
            quantization_config_copy.tokenizer = quantization_config.tokenizer or model_id
            potential_processor_id = config.mm_vision_tower if isinstance(model, _OVNanoLlavaForCausalLM) else model_id
            quantization_config_copy.processor = quantization_config.processor or potential_processor_id
            OVQuantizer(model).quantize(ov_config=OVConfig(quantization_config=quantization_config_copy))

        return model

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        use_cache: bool = True,
        trust_remote_code: bool = False,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        compile_only = kwargs.pop("compile_only", False)
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting
        # would end-up removing the directory containing the underlying OpenVINO model
        cls._model_save_dir_tempdirectory_instance = save_dir

        if task is None:
            task = cls.export_feature

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            # Export in fp32 if compression won't be applied later
            ov_config = OVConfig(dtype="fp32" if load_in_8bit is False else "auto")

        stateful = kwargs.pop("stateful", ensure_stateful_is_available(warn=False) and use_cache)
        variant = kwargs.pop("variant", None)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            ov_config=ov_config,
            stateful=stateful,
            variant=variant,
        )
        config = AutoConfig.from_pretrained(save_dir_path, trust_remote_code=trust_remote_code)
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )

    @property
    def _component_names(self):
        base_components = ["language_model", "vision_embeddings"]
        additional_components = [part for part in self.additional_parts if getattr(self, part, None) is not None]
        return base_components + additional_components

    @property
    def components(self):
        return {component_name: getattr(self, component_name) for component_name in self._component_names}

    @property
    def _ov_submodel_names(self):
        model_names = ["lm_model", "text_embeddings_model", "vision_embeddings_model"]
        for part in self.additional_parts:
            if getattr(self, part, None) is not None:
                model_names.append(part + "_model")
        return model_names

    def reshape(self, batch_size: int, sequence_length: int):
        logger.warning("Static shapes are not supported for causal language model.")
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        for submodel in self.ov_submodels.values():
            apply_moc_transformations(submodel, cf=False)
            compress_model_transformation(submodel)
        return self

    def to(self, device):
        self.language_model.to(device)
        super().to(device)
        return self

    def forward(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        inputs_embeds=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        image_bound=None,
        tgt_sizes=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        images=None,
        second_per_grid_ts=None,
        token_type_ids=None,
        pixel_attention_mask=None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_pixel_values: Optional[torch.FloatTensor] = None,
        image_attention_mask=None,
        audio_input_features: Optional[torch.FloatTensor] = None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        input_mode=None,
        **kwargs,
    ):
        if pixel_values is None:
            pixel_values = images if images is not None else image_pixel_values
        inputs_embeds, attention_mask, position_ids = self.get_multimodal_embeddings(
            input_ids,
            pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            image_bound=image_bound,
            tgt_sizes=tgt_sizes,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            second_per_grid_ts=second_per_grid_ts,
            pixel_attention_mask=pixel_attention_mask,
            input_image_embeds=input_image_embeds,
            image_attention_mask=image_attention_mask,
            input_audio_embeds=input_audio_embeds if input_audio_embeds is not None else audio_input_features,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            input_mode=input_mode,
            **kwargs,
        )
        return self.language_model.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    def get_vision_embeddings(self, pixel_values, **kwargs):
        raise NotImplementedError

    def get_text_embeddings(self, input_ids, **kwargs):
        return self.language_model.embed_tokens(input_ids)

    def merge_vision_text_embeddings(
        self, vision_embeds, inputs_embeds, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        raise NotImplementedError

    def get_multimodal_embeddings(
        self, input_ids, pixel_values=None, attention_mask=None, position_ids=None, **kwargs
    ):
        inputs_embeds = self.get_text_embeddings(input_ids, **kwargs)
        if pixel_values is not None:
            vision_embeds = self.get_vision_embeddings(pixel_values, input_ids=input_ids, **kwargs)
            if vision_embeds is not None:
                inputs_embeds, attention_mask, position_ids = self.merge_vision_text_embeddings(
                    vision_embeds,
                    inputs_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )
        return inputs_embeds, attention_mask, position_ids

    # Adopted from https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llava/modeling_llava.py#L521
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = self.language_model._get_past_length(past_key_values)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and past_length + 1 > input_ids.shape[1]:
                input_discount = max(attention_mask.shape[1] - past_length, 1)
                input_ids = input_ids[:, -input_discount:]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.llava
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif getattr(self.config, "image_token_index", -1) in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            # position_ids in Gemma3 are 1-indexed
            if self.config.model_type == "gemma3":
                position_ids += 1

            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if pixel_values is None:
            pixel_values = kwargs.get("input_image_embeds", kwargs.get("images", kwargs.get("image_pixel_values")))

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
                "image_bound": kwargs.get("image_bound"),
                "tgt_sizes": kwargs.get("tgt_sizes"),
                "pixel_values_videos": kwargs.get("pixel_values_videos"),
                "image_grid_thw": kwargs.get("image_grid_thw"),
                "video_grid_thw": kwargs.get("video_grid_thw"),
                "token_type_ids": kwargs.get("token_type_ids"),
                "pixel_attetion_mask": kwargs.get("pixle_attetion_mask"),
                "image_attention_mask": kwargs.get("image_attention_mask"),
                "input_audio_embeds": kwargs.get("input_audio_embeds", kwargs.get("audio_input_features")),
                "audio_embed_sizes": kwargs.get("audio_embed_sizes"),
                "input_mode": kwargs.get("input_mode"),
            }
        )
        return model_inputs

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    @staticmethod
    @abstractmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        """
        Preprocess input instruction and an image.
        """


class _OVLlavaForCausalLM(OVModelForVisualCausalLM):
    auto_model_class = LlavaForConditionalGeneration

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_embeddings: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(
            language_model=language_model,
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            config=config,
            device=device,
            dynamic_shapes=dynamic_shapes,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )
        self._support_new_processing = hasattr(self.config, "image_seq_length")

    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        if not isinstance(pixel_values, list):
            image_features = self.vision_embeddings(pixel_values).last_hidden_state
        else:
            image_features = []
            for patch in pixel_values:
                if isinstance(patch, list):
                    patch_feats = []
                    for patch_value in patch:
                        patch_feats.append(self.vision_embeddings(np.expand_dims(patch_value, 0)).last_hidden_state)
                    patch_feats = np.concatenate(patch_feats, axis=1)
                else:
                    patch_feats = self.vision_embeddings(patch).last_hidden_state
                image_features.append(patch_feats)
            image_features = np.concatenate(image_features, 0)

        return image_features

    # Adopted from https://github.com/huggingface/transformers/blob/d7950bff82b18c823193d17d72188c5e46d06c83/src/transformers/models/llava/modeling_llava.py#L297C9-L297C45
    def merge_vision_text_embeddings(
        self,
        vision_embeds,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        legacy_processing=False,
        **kwargs,
    ):
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds

        if legacy_processing:
            pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

            num_images, num_image_patches, embed_dim = image_features.shape
            batch_size, sequence_length = input_ids.shape
            left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == self.config.image_token_index
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # Compute the maximum embed dimension
            max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
            batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
            nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
            if left_padding:
                new_token_positions += nb_image_pad[:, None]  # offset for left padding
            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

            # 3. Create the full embedding, already padded to the maximum position
            final_embedding = torch.zeros(
                batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
            final_attention_mask = torch.zeros(
                batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
            )

            # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
            # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
            final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
            final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
            # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]

            if image_to_overwrite.sum() != image_features.shape[:-1].numel():
                raise ValueError(
                    f"The input provided to the model a/pre-releasesre wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
                )

            final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
            final_attention_mask |= image_to_overwrite
            position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

            # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
            batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
            indices_to_mask = new_token_positions[batch_indices, pad_indices]

            final_embedding[batch_indices, indices_to_mask] = 0
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            image_features = image_features.to(inputs_embeds.dtype)
            final_embedding = inputs_embeds.masked_scatter(special_image_mask, image_features)
            final_attention_mask = attention_mask

        return final_embedding, final_attention_mask, position_ids

    def get_multimodal_embeddings(
        self, input_ids, pixel_values=None, attention_mask=None, position_ids=None, past_key_values=None, **kwargs
    ):
        if pixel_values is not None and self._support_new_processing and past_key_values is None:
            legacy_processing = (input_ids == self.config.image_token_index).sum(
                1
            ).max() < self.config.image_seq_length
        else:
            legacy_processing = True
        inputs_embeds, attention_mask, position_ids = super().get_multimodal_embeddings(
            input_ids, pixel_values, attention_mask, position_ids, legacy_processing=legacy_processing, **kwargs
        )

        if legacy_processing and pixel_values is not None and past_key_values is not None:
            attention_mask, position_ids = self._filter_unattended_tokens(input_ids, attention_mask, past_key_values)

        return inputs_embeds, attention_mask, position_ids

    def _filter_unattended_tokens(self, input_ids, attention_mask, past_key_values):
        # Get the target length
        target_length = input_ids.shape[1]
        past_length = self.language_model._get_past_length(past_key_values)

        extended_attention_mask = torch.ones(
            (attention_mask.shape[0], past_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
        position_ids = torch.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1
        return attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if getattr(processor, "chat_template", None) is not None:
            chat_prompt = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            if image is not None:
                chat_prompt[0]["content"].append({"type": "image"})
            prompt = processor.apply_chat_template(chat_prompt, add_generation_prompt=True, tokenize=False)
        else:
            if image is not None and "<image>" not in text:
                prompt = "<image>\n" + text
            else:
                prompt = text

        if is_transformers_version(">", "4.47.99") and getattr(processor, "patch_size", None) is None:
            if (
                getattr(config, "vision_config", None) is not None
                and getattr(config.vision_config, "patch_size", None) is not None
            ):
                processor.patch_size = config.vision_config.patch_size
            else:
                raise ValueError(
                    "Processor does not have `patch_size` attribute. Please fix the processor or provide `patch_size` in the config."
                )

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        return inputs


class _OVLlavaNextForCausalLM(_OVLlavaForCausalLM):
    auto_model_class = LlavaNextForConditionalGeneration

    # Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/modeling_llava_next.py#L655
    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image

        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    def add_image_features(
        self,
        input_ids,
        inputs_embeds,
        pixel_values,
        attention_mask,
        position_ids,
        image_sizes,
        legacy_processing,
        **kwargs,
    ):
        from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        # figure out if pixel_values is concatenated or stacked
        if pixel_values.dim() == 5:
            # stacking when input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")
        vision_embeds = self.get_vision_embeddings(pixel_values, input_ids=input_ids, **kwargs)
        if vision_embeds is not None:
            image_newline = torch.tensor(self.config.image_newline)
            image_features = torch.split(torch.from_numpy(vision_embeds), image_num_patches, dim=0)
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=image_newline,
            )
            inputs_embeds, attention_mask, position_ids = self.merge_vision_text_embeddings(
                image_features,
                inputs_embeds,
                feature_lens=feature_lens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                legacy_processing=legacy_processing,
                **kwargs,
            )
        return inputs_embeds, attention_mask, position_ids

    # Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/modeling_llava_next.py#L416
    def get_multimodal_embeddings(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        image_sizes=None,
        **kwargs,
    ):
        inputs_embeds = self.get_text_embeddings(input_ids, **kwargs)

        if pixel_values is not None and self._support_new_processing and past_key_values is None:
            legacy_processing = (input_ids == self.config.image_token_index).sum(
                1
            ).max() < self.config.image_seq_length
        else:
            legacy_processing = True

        if pixel_values is not None and pixel_values.size(0) > 0:
            inputs_embeds, attention_mask, position_ids = self.add_image_features(
                input_ids,
                inputs_embeds,
                pixel_values,
                attention_mask,
                position_ids,
                image_sizes,
                legacy_processing,
                **kwargs,
            )

        if legacy_processing and pixel_values is not None and past_key_values is not None and input_ids.shape[1] == 1:
            attention_mask, position_ids = self._filter_unattended_tokens(input_ids, attention_mask, past_key_values)

        return inputs_embeds, attention_mask, position_ids

    def merge_vision_text_embeddings(
        self,
        vision_embeds,
        inputs_embeds,
        feature_lens,
        input_ids,
        attention_mask,
        position_ids=None,
        legacy_processing=False,
        image_token_index=None,
        **kwargs,
    ):
        image_token_index = self.config.image_token_index if image_token_index is None else image_token_index
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds

        if legacy_processing:
            with torch.no_grad():
                # ! in llava 1.6, number of patches is variable
                num_images = feature_lens.size(0)
                num_image_features, embed_dim = image_features.shape
                if feature_lens.sum() != num_image_features:
                    raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
                batch_size = input_ids.shape[0]
                _left_padding = torch.any(attention_mask[:, 0] == 0)
                _right_padding = torch.any(attention_mask[:, -1] == 0)

                left_padding = True
                if batch_size > 1:
                    if _left_padding and not _right_padding:
                        left_padding = True
                    elif not _left_padding and _right_padding:
                        left_padding = False
                    elif not _left_padding and not _right_padding:
                        left_padding = True
                    else:
                        # invalid attention_mask
                        raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

                # Whether to turn off right padding
                # 1. Create a mask to know where special image tokens are
                special_image_token_mask = input_ids == image_token_index
                # special_image_token_mask: [bsz, seqlen]
                num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
                # num_special_image_tokens: [bsz]
                # Reserve for padding of num_images
                total_num_special_image_tokens = torch.sum(special_image_token_mask)
                if total_num_special_image_tokens != num_images:
                    raise ValueError(
                        f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                    )
                # Compute the maximum embed dimension
                # max_image_feature_lens is max_feature_lens per batch
                feature_lens = feature_lens.to(input_ids.device)
                feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
                feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
                embed_sequence_lengths = (
                    (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
                )
                max_embed_dim = embed_sequence_lengths.max()

                batch_indices, non_image_indices = torch.where(
                    (input_ids != image_token_index) & (attention_mask == 1)
                )
                # 2. Compute the positions where text should be written
                # Calculate new positions for text tokens in merged image-text sequence.
                # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
                # `torch.cumsum` computes how each image token shifts subsequent text token positions.
                # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
                # ! instead of special_image_token_mask * (num_image_patches - 1)
                #   special_image_token_mask * (num_feature_len - 1)
                special_image_token_mask = special_image_token_mask.long()
                special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
                new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
                if left_padding:
                    # shift right token positions so that they are ending at the same number
                    # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                    new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

                text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

            # 3. Create the full embedding, already padded to the maximum position
            final_embedding = torch.zeros(
                batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
            final_attention_mask = torch.zeros(
                batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
            )
            # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
            # set the corresponding tensors into their correct target device.
            target_device = inputs_embeds.device
            batch_indices, non_image_indices, text_to_overwrite = (
                batch_indices.to(target_device),
                non_image_indices.to(target_device),
                text_to_overwrite.to(target_device),
            )
            attention_mask = attention_mask.to(target_device)
            input_ids = input_ids.to(target_device)

            # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
            # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
            final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
            final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

            # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
            with torch.no_grad():
                image_to_overwrite = torch.full(
                    (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
                )
                image_to_overwrite[batch_indices, text_to_overwrite] = False
                embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
                embed_indices = embed_indices.expand(batch_size, max_embed_dim)
                embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

                if left_padding:
                    # exclude padding on the left
                    max_embed_dim = max_embed_dim.to(target_device)
                    val = (max_embed_dim - embed_indices) <= embed_seq_lens
                else:
                    # exclude padding on the right
                    val = embed_indices < embed_seq_lens
                image_to_overwrite &= val

                if image_to_overwrite.sum() != num_image_features:
                    raise ValueError(
                        f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                        f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                        f" the number of image given to the model is {num_images}. "
                        f"This prevents correct indexing and breaks batch generation."
                    )
            final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
            final_attention_mask |= image_to_overwrite
            position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        else:
            special_image_mask = (input_ids == image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            image_features = image_features.to(inputs_embeds.dtype)
            final_embedding = inputs_embeds.masked_scatter(special_image_mask, image_features)
            final_attention_mask = attention_mask

        return final_embedding, final_attention_mask, position_ids

    def get_text_embeddings(self, input_ids, **kwargs):
        for_inputs_embeds_ids = input_ids.clone()
        for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
        return super().get_text_embeddings(for_inputs_embeds_ids, **kwargs)


class _OVLlavaNextVideoForCausalLM(_OVLlavaNextForCausalLM):
    additional_parts = ["vision_resampler", "multi_modal_projector"]
    auto_model_class = AutoModelForVision2Seq

    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        image_features = self.vision_embeddings(pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(image_features)
        return image_features

    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_select_strategy (`str`)
                The feature selection strategy used to select the vision feature from the vision backbone.
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        from transformers.models.llava_next_video.modeling_llava_next_video import (
            get_anyres_image_grid_shape,
            unpad_image,
        )

        new_image_features = []
        feature_lens = []
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )

                if (
                    np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
                    and vision_feature_select_strategy == "default"
                ):
                    logger.warning_once(
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a"
                        " visual encoder that does not have CLS."
                    )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if getattr(processor, "chat_template", None) is not None:
            chat_prompt = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            if image is not None:
                chat_prompt[0]["content"].append({"type": "image"})
            if video is not None:
                chat_prompt[0]["content"].append({"type": "video"})
            prompt = processor.apply_chat_template(chat_prompt, add_generation_prompt=True, tokenize=False)
        else:
            prompt = text
            if image is not None and "<image>" not in prompt:
                prompt = "<image>\n" + prompt
            if video is not None and "<video>" not in prompt:
                prompt = "<video>\n" + prompt

        if is_transformers_version(">", "4.47.99") and getattr(processor, "patch_size", None) is None:
            if (
                getattr(config, "vision_config", None) is not None
                and getattr(config.vision_config, "patch_size", None) is not None
            ):
                processor.patch_size = config.vision_config.patch_size
            else:
                raise ValueError(
                    "Processor does not have `patch_size` attribute. Please fix the processor or provide `patch_size` in the config."
                )

        inputs = processor(images=image, text=prompt, videos=video, return_tensors="pt")
        return inputs

    def get_multimodal_embeddings(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        image_sizes=None,
        pixel_values_videos=None,
        **kwargs,
    ):
        inputs_embeds = self.get_text_embeddings(input_ids, **kwargs)

        if (
            pixel_values is not None
            and pixel_values.size(0) > 0
            and self._support_new_processing
            and past_key_values is None
        ):
            legacy_processing = (
                (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
            ).item()
        elif (
            pixel_values_videos is not None
            and pixel_values_videos.size(0) > 0
            and self._support_new_processing
            and past_key_values is None
        ):
            legacy_processing = (
                (input_ids == self.config.video_token_index).sum(1).max() < self.config.video_seq_length
            ).item()
        else:
            legacy_processing = True

        legacy_processing = (
            legacy_processing.item() if isinstance(legacy_processing, torch.Tensor) else legacy_processing
        )

        if pixel_values is not None and pixel_values.size(0) > 0:
            inputs_embeds, attention_mask, position_ids = self.add_image_features(
                input_ids,
                inputs_embeds,
                pixel_values,
                attention_mask,
                position_ids,
                image_sizes,
                legacy_processing,
                **kwargs,
            )

        if pixel_values_videos is not None and pixel_values_videos.size(0) > 0:
            inputs_embeds, attention_mask, position_ids = self.add_video_features(
                input_ids,
                inputs_embeds,
                pixel_values_videos,
                attention_mask,
                position_ids,
                legacy_processing=legacy_processing,
                **kwargs,
            )

        if legacy_processing and pixel_values is not None and past_key_values is not None and input_ids.shape[1] == 1:
            attention_mask, position_ids = self._filter_unattended_tokens(input_ids, attention_mask, past_key_values)

        return inputs_embeds, attention_mask, position_ids

    def add_video_features(
        self,
        input_ids,
        inputs_embeds,
        pixel_values_videos,
        attention_mask,
        position_ids,
        legacy_processing,
        **kwargs,
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L732-L751
        video_features = self.get_video_features(pixel_values_videos, input_ids)
        if video_features is not None and len(video_features) != 0:
            video_features = [feature.flatten(0, 1) for feature in video_features]
            video_feature_lens = [feature.size(0) for feature in video_features]
            video_features = torch.cat(video_features, dim=0)
            video_feature_lens = torch.tensor(video_feature_lens, dtype=torch.long, device=video_features.device)

            if legacy_processing:
                inputs_embeds, attention_mask, position_ids = self.merge_vision_text_embeddings(
                    video_features,
                    inputs_embeds,
                    video_feature_lens,
                    input_ids,
                    attention_mask,
                    position_ids,
                    legacy_processing,
                    self.config.video_token_index,
                )
            else:
                inputs_embeds = (
                    torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds
                )
                special_image_mask = (input_ids == self.config.video_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds)
                if inputs_embeds[special_image_mask].numel() != video_features.numel():
                    n_video_tokens = (input_ids == self.config.video_token_index).sum().item()
                    n_video_features = video_features.shape[0]
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, video_features)
        return inputs_embeds, attention_mask, position_ids

    def get_video_features(self, pixel_values, input_ids=None, **kwargs):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/llava_next_video/modeling_llava_next_video.py#L835
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        batch_size, frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * frames, channels, height, width)
        selected_video_features = self.vision_embeddings(pixel_values).last_hidden_state
        video_features = self.vision_resampler(selected_video_features)
        video_features = self.multi_modal_projector(video_features)
        video_features = torch.split(torch.from_numpy(video_features), frames, dim=0)
        return video_features


class _OVInternVLForCausalLM(OVModelForVisualCausalLM):
    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        image_features = self.vision_embeddings(pixel_values, **kwargs).last_hidden_state
        return image_features

    def merge_vision_text_embeddings(
        self, vision_embeds, input_embeds, input_ids, attention_mask, position_ids=None, **kwargs
    ):
        input_embeds = torch.from_numpy(input_embeds) if isinstance(input_embeds, np.ndarray) else input_embeds
        vision_embeds = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.config.img_context_token_id
        assert selected.sum() != 0
        input_embeds[selected] = vision_embeds.reshape(-1, C)

        input_embeds = input_embeds.reshape(B, N, C)
        return input_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if tokenizer is None:
            raise ValueError("Tokenizer is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=28, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            }
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size,
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image, input_size=448, max_num=12):
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        if image is not None and "<image>" not in text:
            text = "<image>\n" + text

        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
            )

        inputs = {}

        if image is not None:
            if config is None:
                raise ValueError("Config is required.")
            pixel_values = load_image(image, input_size=config.vision_config.image_size)
            num_patches = pixel_values.shape[0]
            num_image_token = int(
                (config.vision_config.image_size // config.vision_config.patch_size) ** 2
                * (config.downsample_ratio**2)
            )
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            text = text.replace("<image>", image_tokens, 1)
            inputs.update({"pixel_values": pixel_values})
        inputs.update(tokenizer(text, return_tensors="pt"))
        return inputs

    # internvl has issue with check  _get_non_default_parameters, as wrkaraund overide _prepare_generation_config
    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        using_model_generation_config = False
        if generation_config is None:
            if (
                self.generation_config._from_model_config  # 1)
                and self.generation_config._original_object_hash == hash(self.generation_config)  # 2)
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:  # 4)
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed in v5."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )",
                        UserWarning,
                    )
                    self.generation_config = new_generation_config

            generation_config = self.generation_config
            using_model_generation_config = True

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
        if not using_model_generation_config:
            if generation_config.bos_token_id is None:
                generation_config.bos_token_id = self.generation_config.bos_token_id
            if generation_config.eos_token_id is None:
                generation_config.eos_token_id = self.generation_config.eos_token_id
            if generation_config.pad_token_id is None:
                generation_config.pad_token_id = self.generation_config.pad_token_id
            if generation_config.decoder_start_token_id is None:
                generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id

        return generation_config, model_kwargs


class _OVMiniCPMVForCausalLM(OVModelForVisualCausalLM):
    additional_parts = ["resampler"]

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_embeddings: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(
            language_model,
            text_embeddings,
            vision_embeddings,
            config,
            device,
            dynamic_shapes,
            ov_config,
            model_save_dir,
            quantization_config,
            **kwargs,
        )
        self.embed_dim = self.language_model.config.hidden_size
        max_size = self.config.vision_config.image_size // self.config.vision_config.patch_size
        self._pos_embeds = torch.from_numpy(self._get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self.max_size = (max_size, max_size)

    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        tgt_sizes = kwargs["tgt_sizes"]
        pixel_values_list = pixel_values
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        for pixel_value in pixel_values_list:
            img_cnt.append(len(pixel_value))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_value])

        vision_embedding = None
        # exist image
        if all_pixel_values:
            tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool)
            for i in range(B):
                patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True
            position_ids = self._prepare_vis_position_ids(
                all_pixel_values,
                patch_attn_mask,
                tgt_sizes,
                self.config.vision_config.patch_size,
                self.config.vision_config.image_size // self.config.patch_size,
            )
            vision_embedding = torch.from_numpy(
                self.vision_embeddings(
                    pixel_values=all_pixel_values, patch_attention_mask=patch_attn_mask, position_ids=position_ids
                )[0]
            )
            vision_embedding = self.resampling(vision_embedding, tgt_sizes)

            start = 0
            for pixel_value in pixel_values_list:
                img_cnt = len(pixel_value)
                if img_cnt > 0:
                    vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                    start += img_cnt
                else:
                    vision_hidden_states.append([])
        else:  # no image
            dummy_feature = []
            for _ in range(len(pixel_values_list)):
                vision_hidden_states.append(dummy_feature)
        return vision_hidden_states

    def resampling(self, x, tgt_sizes):
        bs = x.shape[0]

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self._pos_embeds[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D
        res = torch.from_numpy(self.resampler(image_feature=x, pos_embed=pos_embed, key_padding_mask=key_padding_mask))
        return res

    def _set_2d_pos_cache(self, max_size):
        pos_embed = torch.from_numpy(self._get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self._pos_embeds = pos_embed

    def _adjust_pos_cache(self, tgt_sizes):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size)

    def _get_2d_sincos_pos_embed(self, embed_dim, image_size):
        """
        image_size: image_size or (image_height, image_width)
        return:
        pos_embed: [image_height, image_width, embed_dim]
        """
        if isinstance(image_size, int):
            grid_h_size, grid_w_size = image_size, image_size
        else:
            grid_h_size, grid_w_size = image_size[0], image_size[1]

        grid_h = np.arange(grid_h_size, dtype=np.float32)
        grid_w = np.arange(grid_w_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self._get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
        emb_w = self._get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
        return emb

    def _get_1d_sincos_pos_embed_from_grid_new(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (H, W)
        out: (H, W, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

        emb_sin = np.sin(out)  # (H, W, D/2)
        emb_cos = np.cos(out)  # (H, W, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
        return emb

    def _prepare_vis_position_ids(
        self, pixel_values, patch_attention_mask, tgt_sizes, patch_size, num_patches_per_side
    ):
        batch_size = pixel_values.size(0)
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        return position_ids

    def merge_vision_text_embeddings(
        self, vision_embeds, input_embeds, input_ids, attention_mask, position_ids=None, **kwargs
    ):
        bs = input_ids.shape[0]
        image_bound = kwargs["image_bound"]
        vllm_embedding = torch.from_numpy(input_embeds)
        for i in range(bs):
            cur_vs_hs = vision_embeds[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = image_bound[i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack([torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound])

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
        return vllm_embedding, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if getattr(processor, "chat_template", None) is not None:
            messages = [{"role": "user", "content": text if image is None else "(<image>./</image>)\n" + text}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = (
                f"<|im_start|>user\n(<image>./</image>)\n{text}<|im_end|>\n<|im_start|>assistant\n"
                if image is not None
                else text
            )
        inputs = processor([prompt], [image], return_tensors="pt")
        inputs.pop("image_sizes", None)
        return inputs


class _OVNanoLlavaForCausalLM(OVModelForVisualCausalLM):
    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        if isinstance(pixel_values, list) or pixel_values.ndim == 5:
            concat_images = torch.cat(pixel_values, dim=0) if isinstance(pixel_values, list) else pixel_values
            image_features = torch.from_numpy(self.vision_embeddings(concat_images).last_hidden_state)
            split_sizes = [image.shape[0] for image in pixel_values]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.vision_embeddings(pixel_values).last_hidden_state

        return image_features

    def get_multimodal_embeddings(
        self, input_ids, pixel_values=None, attention_mask=None, position_ids=None, **kwargs
    ):
        vision_embeds = None
        IGNORE_INDEX = -100
        IMAGE_TOKEN_INDEX = -200
        if pixel_values is None and "images" in kwargs:
            pixel_values = kwargs["images"]
        if pixel_values is not None:
            vision_embeds = self.get_vision_embeddings(pixel_values, input_ids=input_ids, **kwargs)
        if vision_embeds is None:
            inputs_embeds = torch.from_numpy(self.get_text_embeddings(input_ids))
            past_len = self.language_model._get_past_length(kwargs.get("past_key_values"))
            if attention_mask is not None and attention_mask.shape[1] < past_len + input_ids.shape[1]:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(attention_mask.shape[0], past_len + input_ids.shape[1] - attention_mask.shape[1]),
                    ],
                    dim=1,
                )
                position_ids = None
            return inputs_embeds, attention_mask, position_ids

        vision_embeds = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask.bool())
        ]
        labels = [
            cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask.bool())
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = vision_embeds[cur_image_idx]
                cur_input_embeds_1 = torch.from_numpy(self.get_text_embeddings(cur_input_ids.unsqueeze(0))[0])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = torch.from_numpy(
                self.get_text_embeddings(torch.cat(cur_input_ids_noim).unsqueeze(0))[0]
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = vision_embeds[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        return new_input_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if tokenizer is None:
            raise ValueError("Tokenizer is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if image is not None and processor is None:
            raise ValueError("Processor is required.")
        text = f"<image>\n{text}" if image is not None else text
        messages = [{"role": "user", "content": text}]
        if tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is not None:
            text_chunks = [tokenizer(chunk).input_ids for chunk in text.split("<image>")]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)
        else:
            input_ids = tokenizer(text, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        if image is not None:
            result["images"] = processor(images=[image], return_tensors="pt")["pixel_values"]
        return result


class _OVPhi3VisionForCausalLM(OVModelForVisualCausalLM):
    additional_parts = ["vision_projection"]

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_embeddings: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(
            language_model,
            text_embeddings,
            vision_embeddings,
            config,
            device,
            dynamic_shapes,
            ov_config,
            model_save_dir,
            quantization_config,
            **kwargs,
        )
        self.sub_GN = torch.tensor(self.config.sub_GN)
        self.glb_GN = torch.tensor(self.config.glb_GN)
        self.image_dim_out = self.config.img_processor["image_dim_out"]

    def get_vision_embeddings(self, pixel_values, image_sizes, **kwargs):
        num_images, num_crops, c, h, w = pixel_values.shape
        img_features = self.vision_embeddings(pixel_values.flatten(0, 1)).last_hidden_state.reshape(
            num_images, num_crops, -1, self.image_dim_out
        )
        image_features_proj = self.hd_feature_transform(img_features, image_sizes)
        return image_features_proj

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """

        image_features = torch.from_numpy(image_features)
        global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

        all_image_embeddings = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1 : 1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop)
            sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

            # [sub features, separator, global features]
            all_image_embeddings.extend(
                [
                    sub_image_features_hd_newline.squeeze(0),  # (h_crop*12*(w_crop*12+1), 4096)
                    self.glb_GN.squeeze(0),
                    global_image_features_hd_newline[i],
                ]
            )
        image_features_proj = self.vision_projection(torch.cat(all_image_embeddings, dim=0).unsqueeze(0))[0]

        return image_features_proj

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(num_images, h_crop, w_crop, H // 2, H // 2, -1)  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(num_images, h_crop * H // 2, w_crop * H // 2, 4 * C)  # n_img, h_crop*12, w_crop*12, 4096
        )

        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(num_images, h, -1, -1)  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat([image_features_hd, newline_embeddings], dim=2).reshape(
            num_images, -1, hid_dim
        )
        return image_features_hd_newline

    def get_multimodal_embeddings(
        self, input_ids, pixel_values=None, attention_mask=None, position_ids=None, image_sizes=None, **kwargs
    ):
        MAX_INPUT_ID = int(1e9)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # positions for image tokens
        positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=True)
        has_image = len(positions[0].tolist()) > 0
        input_ids = input_ids.clamp_min(0).clamp_max(self.config.vocab_size)
        inputs_embeds = torch.from_numpy(self.get_text_embeddings(input_ids, **kwargs))
        if has_image:
            vision_embeds = self.get_vision_embeddings(
                pixel_values, input_ids=input_ids, image_sizes=image_sizes, **kwargs
            )
            image_features_proj = torch.from_numpy(vision_embeds)
            inputs_embeds = inputs_embeds.index_put(positions, image_features_proj, accumulate=False)

        return inputs_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if image is not None and "<|image_1|>" not in text:
            text = "<|image_1|>\n" + text
        if getattr(processor.tokenizer, "chat_template", None) is not None:
            chat_prompt = [{"role": "user", "content": text}]
            text = processor.tokenizer.apply_chat_template(chat_prompt, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=image, text=text, return_tensors="pt")
        return inputs


@dataclass
class QWen2VLModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    rope_deltas: Optional[torch.FloatTensor] = None
    second_per_grid_ts: Optional[torch.FloatTensor] = None


class _OVQwen2VLForCausalLM(OVModelForVisualCausalLM):
    additional_parts = ["vision_embeddings_merger"]

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_embeddings: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(
            language_model=language_model,
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            config=config,
            device=device,
            dynamic_shapes=dynamic_shapes,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )
        self.rope_deltas = None  # cache rope_deltas here

        if is_transformers_version(">=", "4.45.0"):
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLForConditionalGeneration,
                VisionRotaryEmbedding,
            )

            self._rotary_pos_emb = VisionRotaryEmbedding(
                self.config.vision_config.embed_dim // self.config.vision_config.num_heads // 2
            )
            self.get_rope_index = MethodType(Qwen2VLForConditionalGeneration.get_rope_index, self)
        else:
            raise ValueError(
                f"Initialization model for {self.config.model_type} required at least transformers >= 4.45"
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    # Copied from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1602
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def get_vision_embeddings(self, pixel_values, grid_thw, **kwargs):
        hidden_states = self.vision_embeddings(pixel_values)[0]
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        attention_mask = torch.zeros((1, hidden_states.shape[0], hidden_states.shape[0]), dtype=torch.bool)
        causal_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        causal_mask.masked_fill_(torch.logical_not(attention_mask), float("-inf"))

        res = self.vision_embeddings_merger(
            pixel_values=hidden_states, attention_mask=causal_mask, rotary_pos_emb=rotary_pos_emb
        )[0]
        return res

    # Adopted from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1089
    # Use config values instead of model attributes, replace self.rotary_pos_emb -> self._rotary_pos_emb
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
                w // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
                w // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self._rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_multimodal_embeddings(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        inputs_embeds = torch.from_numpy(self.get_text_embeddings(input_ids))
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_embeds = torch.from_numpy(self.get_vision_embeddings(pixel_values, image_grid_thw))
            image_mask = input_ids == self.config.image_token_id
            inputs_embeds[image_mask] = image_embeds
        if pixel_values_videos is not None and input_ids.shape[1] != 1:
            pixel_values_videos = pixel_values_videos
            video_embeds = torch.from_numpy(self.get_vision_embeddings(pixel_values_videos, video_grid_thw))
            video_mask = input_ids == self.config.video_token_id
            inputs_embeds[video_mask] = video_embeds

        # if we get 4D attention mask we cannot calculate rope deltas anymore.
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return inputs_embeds, attention_mask, position_ids

    def forward(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        inputs_embeds=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        image_bound=None,
        tgt_sizes=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        **kwargs,
    ):
        result = super().forward(
            input_ids,
            pixel_values,
            past_key_values,
            inputs_embeds,
            image_sizes,
            attention_mask,
            position_ids,
            image_bound,
            tgt_sizes,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            rope_deltas,
            **kwargs,
        )
        final_result = QWen2VLModelOutputWithPast(
            logits=result.logits, past_key_values=result.past_key_values, rope_deltas=rope_deltas
        )
        return final_result

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})
        if video is not None:
            conversation[0]["content"].insert(0, {"type": "video"})

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=text_prompt, videos=video, return_tensors="pt")
        return inputs


class _OVQwen2_5_VLForCausalLM(OVModelForVisualCausalLM):
    additional_parts = ["vision_embeddings_merger"]

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_embeddings: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(
            language_model=language_model,
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            config=config,
            device=device,
            dynamic_shapes=dynamic_shapes,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )
        self.rope_deltas = None  # cache rope_deltas here

        class Qwen2_5_VisionRotaryEmbedding(torch.nn.Module):
            def __init__(self, dim: int, theta: float = 10000.0) -> None:
                super().__init__()
                inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            def forward(self, seqlen: int) -> torch.Tensor:
                seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(seq, self.inv_freq)
                return freqs

        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        self._rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # modified from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1546
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
            }
        )
        return model_inputs

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
                w // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
                w // self.config.vision_config.spatial_merge_size,
                self.config.vision_config.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self._rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_multimodal_embeddings(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1791-L1861
        inputs_embeds = torch.from_numpy(self.get_text_embeddings(input_ids))
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_embeds = torch.from_numpy(self.get_vision_embeddings(pixel_values, image_grid_thw))
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None and input_ids.shape[1] != 1:
            video_embeds = torch.from_numpy(self.get_vision_embeddings(pixel_values_videos, video_grid_thw))
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # if we get 4D attention mask we cannot calculate rope deltas anymore.
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return inputs_embeds, attention_mask, position_ids

    def get_vision_embeddings(self, pixel_values, grid_thw, **kwargs):
        hidden_states = self.vision_embeddings(pixel_values)[0]
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        attention_mask = torch.zeros((1, hidden_states.shape[0], hidden_states.shape[0]), dtype=torch.bool)
        causal_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        causal_mask.masked_fill_(torch.logical_not(attention_mask), float("-inf"))

        window_attention_mask = torch.zeros((1, hidden_states.shape[0], hidden_states.shape[0]), dtype=torch.bool)
        window_causal_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
        for i in range(1, len(cu_window_seqlens)):
            window_attention_mask[
                ..., cu_window_seqlens[i - 1] : cu_window_seqlens[i], cu_window_seqlens[i - 1] : cu_window_seqlens[i]
            ] = True

        window_causal_mask.masked_fill_(torch.logical_not(window_attention_mask), float("-inf"))

        res = self.vision_embeddings_merger(
            pixel_values=hidden_states,
            attention_mask=causal_mask,
            window_attention_mask=window_causal_mask,
            window_index=window_index,
            rotary_pos_emb=rotary_pos_emb,
        )[0]
        return res

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.config.vision_config.window_size
            // self.config.vision_config.spatial_merge_size
            // self.config.vision_config.patch_size
        )
        spatial_merge_unit = (
            self.config.vision_config.spatial_merge_size * self.config.vision_config.spatial_merge_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.config.vision_config.spatial_merge_size,
                grid_w // self.config.vision_config.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})
        if video is not None:
            conversation[0]["content"].insert(0, {"type": "video"})

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=text_prompt, videos=video, return_tensors="pt")
        return inputs

    # Copied from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1602
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs


class _OVMaira2ForCausalLM(_OVLlavaForCausalLM):
    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("processor is required")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if image is None:
            return processor(text=text, return_tensors="pt")
        processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
            frontal_image=image,
            phrase=text,
            return_tensors="pt",
        )
        return processed_inputs


class _OVGemma3ForCausalLM(OVModelForVisualCausalLM):
    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        return self.vision_embeddings(pixel_values).last_hidden_state

    def merge_vision_text_embeddings(
        self, vision_embeds, inputs_embeds, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-Gemma-3/src/transformers/models/gemma3/modeling_gemma3.py#L1323-L1339
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds
        if input_ids is None:
            special_image_mask = inputs_embeds == torch.from_numpy(
                self.get_text_embeddings(torch.tensor([[self.config.image_token_index]], dtype=torch.long))[0]
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds)

            image_features = image_features.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = processor(images=image, text=text_prompt, videos=video, return_tensors="pt")
        return inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        # Token type ids used only for first inference mask generation
        model_kwargs.pop("token_type_ids", None)

        return model_kwargs


class _OVGotOCR2ForCausalLM(OVModelForVisualCausalLM):
    def get_vision_embeddings(self, pixel_values, input_ids, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1 and kwargs.get("past_key_values") is not None:
            return None
        return self.vision_embeddings(pixel_values).last_hidden_state

    def merge_vision_text_embeddings(
        self, vision_embeds, inputs_embeds, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/got_ocr2/modeling_got_ocr2.py#L836-L845
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds
        n_image_tokens = (input_ids == self.config.image_token_index).sum()
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: Optional[str] = None,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("processor is required")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if image is None:
            raise ValueError("Image is required")
        processed_inputs = processor(image, return_tensors="pt")
        return processed_inputs


class _OVIdefics3ForCausalLM(OVModelForVisualCausalLM):
    def get_vision_embeddings(self, pixel_values, input_ids, **kwargs):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-SmolVLM-2/src/transformers/models/smolvlm/modeling_smolvlm.py#L899-L942
        if input_ids is not None and input_ids.shape[1] == 1 and kwargs.get("past_key_values") is not None:
            return None
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        if not any(real_images_inds):
            # no images, leave one empty image.
            real_images_inds[0] = True

        pixel_values = pixel_values[real_images_inds].contiguous()
        pixel_attention_mask = kwargs.get("pixel_attention_mask")

        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        num_patches_per_side = self.config.vision_config.image_size // patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
        batch_size_, _, max_im_h, max_im_w = pixel_values.shape
        max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(size=(batch_size_, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        return self.vision_embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, patch_position_ids=position_ids
        ).last_hidden_state

    def merge_vision_text_embeddings(
        self, vision_embeds, inputs_embeds, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-SmolVLM-2/src/transformers/models/idefics3/modeling_idefics3.py#L881
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds
        vision_hidden_size = image_features.shape[2]
        special_image_token_mask = input_ids == self.config.image_token_id
        #  Fixes RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_features.view(-1, vision_hidden_size)
        # cast to the dtype of the input_embeds to support quantized models
        reshaped_image_hidden_states = reshaped_image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states
        inputs_embeds = new_inputs_embeds

        return inputs_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")

        if video is not None:
            raise ValueError("video input is not supported")

        if audio is not None:
            raise ValueError("Audio input is not supported")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        return inputs


class _OVSmolVLForCasualLM(_OVIdefics3ForCausalLM):
    def merge_vision_text_embeddings(
        self, vision_embeds, inputs_embeds, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-SmolVLM-2/src/transformers/models/smolvlm/modeling_smolvlm.py#L803
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds
        _, patch_size, _ = image_features.shape

        image_mask = input_ids == self.config.image_token_id
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("At least one sample has <image> tokens not divisible by patch_size.")

        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx

        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_features[block_idx[image_mask], local_idx[image_mask], :]

        inputs_embeds = torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)

        return inputs_embeds, attention_mask, position_ids


class _OVPhi4MMForCausalLM(OVModelForVisualCausalLM):
    additional_parts = [
        "vision_projection",
        "audio_embeddings",
        "audio_forward_embeddings",
        "audio_encoder",
        "audio_vision_projection",
        "audio_speech_projection",
    ]

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_embeddings: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        super().__init__(
            language_model,
            text_embeddings,
            vision_embeddings,
            config,
            device,
            dynamic_shapes,
            ov_config,
            model_save_dir,
            quantization_config,
            **kwargs,
        )
        self.sub_GN = torch.tensor(self.config.sub_GN)
        self.glb_GN = torch.tensor(self.config.glb_GN)
        self.audio_config = (
            config.audio_processor["config"] if hasattr(config, "audio_processor") else config.audio_config.to_dict()
        )
        self.chunk_size = self.audio_config.get("chunk_size", -1)
        self.left_chunk = self.audio_config.get("left_chunk", 18)
        self.time_reduction = self.audio_config.get("time_reduction", 8)
        self.image_config = (
            config.img_processor if hasattr(config, "img_processor") else config.vision_config.to_dict()
        )
        self.image_size = self.image_config.get("crop_size", 448)
        self.patch_size = self.image_config.get("patch_size", 14)
        self.num_patches_per_side = self.image_size // self.patch_size
        self._IMAGE_SPECIAL_TOKEN_ID = (
            200010 if "image_token_id" not in self.image_config else self.image_config["image_token_id"]
        )
        self._AUDIO_SPECIAL_TOKEN_ID = (
            200011 if "audio_token_id" not in self.audio_config else self.audio_config["audio_token_id"]
        )
        self._COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility
        self._COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [float("-inf"), -10000]  # For backward compatibility
        self.image_dim_out = self.image_config.get("image_dim_out", self.image_config["hidden_size"])

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L669
    def image_embed(
        self,
        input_ids: torch.LongTensor,
        image_pixel_values: torch.FloatTensor,
        image_attention_mask,
        inputs_embeds,
        image_sizes=None,
        **kwargs,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        positions_tuple = torch.nonzero(input_ids == self._IMAGE_SPECIAL_TOKEN_ID, as_tuple=True)

        if len(positions_tuple[-1]) == 0:
            return None
        batch_size = image_pixel_values.shape[0]

        img_features = self.get_img_features(
            image_pixel_values.flatten(0, 1),
            image_attention_mask=image_attention_mask.flatten(0, 1).to(dtype=bool),
        )

        base_feat_size = int(np.sqrt(img_features.shape[1]))
        img_features = img_features.view(batch_size, -1, base_feat_size**2, self.image_dim_out)
        image_sizes = image_sizes.view(-1, 2)

        output_imgs = []
        for idx in range(batch_size):
            height, width = image_sizes[idx]
            height_ratio = height // self.image_size
            width_ratio = width // self.image_size
            area_ratio = height_ratio * width_ratio

            global_img = img_features[idx, :1]
            global_img = global_img.reshape(1, base_feat_size, base_feat_size, self.image_dim_out).contiguous()
            temporary_extensor = self.sub_GN.repeat(1, base_feat_size, 1, 1)
            global_img = torch.cat([global_img, temporary_extensor], dim=2).reshape(1, -1, self.image_dim_out)

            sub_img = img_features[idx, 1:]
            sub_img = sub_img[:area_ratio]
            sub_img = (
                sub_img.reshape(height_ratio, width_ratio, base_feat_size, base_feat_size, self.image_dim_out)
                .transpose(1, 2)
                .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size, self.image_dim_out)
                .contiguous()
            )

            if image_attention_mask is not None:
                reshaped_image_attention_mask = (
                    image_attention_mask[idx, 1 : area_ratio + 1, 0::2, 0::2]
                    .reshape(height_ratio, width_ratio, base_feat_size, base_feat_size)
                    .transpose(1, 2)
                    .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size)
                )
                useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temporary_extensor = self.sub_GN.repeat(1, useful_height, 1, 1)
            else:
                temporary_extensor = self.sub_GN.repeat(1, height_ratio * base_feat_size, 1, 1)

            sub_img = torch.cat([sub_img, temporary_extensor], dim=2).reshape(1, -1, self.image_dim_out)

            # Merge global and sub
            output_imgs.append(torch.cat([sub_img, self.glb_GN, global_img], dim=1))

        img_set_tensor = []
        for output_img in output_imgs:
            img_feature_proj = torch.from_numpy(self.vision_projection(output_img))
            img_set_tensor.append(img_feature_proj)

        merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)

        image_embeds = inputs_embeds.index_put(indices=positions_tuple, values=merged_img_set_tensor, accumulate=False)

        return image_embeds

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1241
    def audio_embed(
        self,
        input_ids: torch.LongTensor,
        audio_input_embeds: torch.FloatTensor,
        inputs_embeds,
        audio_embed_sizes=None,
        audio_projection_mode="speech",
        **kwargs,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        positions_tuple = torch.nonzero(input_ids == self._AUDIO_SPECIAL_TOKEN_ID, as_tuple=True)

        if len(positions_tuple[-1]) == 0:
            return None

        audio_embeds = self.get_audio_features(audio_input_embeds, audio_projection_mode)

        merged_audio_embeds = torch.cat(
            [audio_embeds[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))], dim=0
        )
        inputs_embeds = inputs_embeds.index_put(indices=positions_tuple, values=merged_audio_embeds, accumulate=False)

        return inputs_embeds

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1165
    def get_audio_features(
        self,
        input_embeds: torch.FloatTensor,
        audio_projection_mode: str = "speech",
    ):
        xs_pad = self.audio_embeddings(input_embeds)
        input_tensor, pos_k, pos_v, hs_mask, masks = self.forward_embeddings(xs_pad)

        unfolded = False
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500  # maximum position for absolute positional encoding
        masks_unfold = None
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                input_tensor_pad = torch.nn.functional.pad(
                    torch.from_numpy(input_tensor), (0, 0, 0, chunk_pad_size), "constant", 0
                )
                input_tensor = input_tensor_pad

            input_tensor = self.unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = masks.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = torch.nn.functional.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = self.unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool()  # unfold op does not support bool tensor
            else:
                masks_unfold = None
        hs_mask = self.calculate_hs_mask(input_tensor, masks_unfold)
        audio_features = self.audio_encoder(input_tensor, hs_mask)
        if unfolded:
            embed_dim = audio_features.shape[-1]
            audio_features = np.reshape(audio_features, (ori_bz, -1, embed_dim))
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                audio_features = audio_features[:, :-chunk_pad_size, :]
        audio_encoder = (
            self.audio_vision_projection if audio_projection_mode == "vision" else self.audio_speech_projection
        )
        audio_set_tensor = audio_encoder(audio_features)

        return torch.from_numpy(audio_set_tensor)

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        """If chunk size is a list, we will randomly select a chunk size."""
        if isinstance(chunk_size, list):
            # Variable chunk size during training
            chunk_size_index = int(torch.randint(low=0, high=len(chunk_size), size=(1,)))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError("Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError("The length of left_chunk must be the same as length of chunk_size.")
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk

        return chunk_size_train_eff, left_chunk_train_eff

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1121
    def forward_embeddings(self, xs_pad, masks=None, chunk_size_nc=None, left_chunk_nc=None):
        """Forwarding the inputs through the top embedding layers

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                input mask
            chunk_size_nc: (optional, default is None) chunk size for non-causal layers
            left_chunk_nc: (optional, default is None) # of left chunks for non-causal layers
        """
        seq_len = int(self.compute_lens_change(xs_pad.shape[1]))
        if seq_len <= 0:
            raise ValueError(
                f"""The squence length after time reduction is invalid: {seq_len}.
                Your input feature is too short. Consider filtering out the very
                short sentence from data loader""",
            )

        batch_size = xs_pad.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.chunk_size, self.left_chunk)

        input_tensor = xs_pad

        input_tensor = self.audio_forward_embeddings(input_tensor)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        else:
            hs_mask = streaming_mask

        if chunk_size_nc is not None:
            enc_streaming_mask_nc = self._streaming_mask(seq_len, batch_size, chunk_size_nc, left_chunk_nc)
            if masks is not None:
                hs_mask_nc = masks & enc_streaming_mask_nc
            else:
                hs_mask_nc = enc_streaming_mask_nc
        else:
            hs_mask_nc = None

        if chunk_size_nc is None:
            return input_tensor, None, None, hs_mask, None
        return input_tensor, None, None, hs_mask, None, hs_mask_nc

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1101
    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(chunk_size, left_chunk)

        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)
        # avoid randomness when run evaluation or decoding

        enc_streaming_mask = (
            self.adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk_train_eff)
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        return enc_streaming_mask

    def compute_lens_change(self, feature_lens):
        """feature_lens: int
        return updated feature lens.

        This used to return a different lambda function for each case that computed
        the right thing.  That does not work within Torchscript.  If you really
        need this to be faster, create nn.Module()-s for all the cases and return
        one of them.  Torchscript does support that.
        """
        nemo_conv_settings = self.audio_config.get("nemo_conv_settings")
        if nemo_conv_settings is None:
            nemo_conv_settings = {"conv_channels": self.audio_config["nemo_conv_channels"]}
        # Handle the special causal case
        subsampling_causal_cond = nemo_conv_settings.get("subsampling", "dw_striding") in [
            "dw_striding",
            "striding",
            "striding_conv1d",
        ]
        is_causal = nemo_conv_settings.get("is_causal", False)
        if is_causal and subsampling_causal_cond:
            lens_change = (
                torch.ceil(feature_lens / self.time_reduction).long()
                if isinstance(feature_lens, torch.Tensor)
                else math.ceil(feature_lens / self.time_reduction)
            )
            feature_lens_remainder = feature_lens % self.time_reduction
            if isinstance(feature_lens, torch.Tensor):
                lens_change[feature_lens_remainder != 1] += 1
            elif feature_lens_remainder != 1:
                lens_change += 1
                return lens_change
        ceil_func = math.ceil if isinstance(feature_lens, int) else torch.ceil
        return ceil_func(feature_lens / self.time_reduction)

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1146
    def calculate_hs_mask(self, xs_pad, mask):
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size, self.chunk_size, self.left_chunk)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = torch.arange(0, max_audio_length).expand(padding_length.size(0), -1) < padding_length.unsqueeze(1)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1034
    @staticmethod
    def unfold_tensor(xs_pad, max_seq_len):
        """
        For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
        this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
        Args:
            xs_pad: N, T, D
        """
        _, _, D = xs_pad.shape
        xs_pad = xs_pad.transpose(-1, -2)  # convert to N, D, T
        # N x D x 1 x T => N x (D x max_seq_len) x T'
        xs_pad = torch.nn.functional.unfold(
            xs_pad[..., None, :],
            kernel_size=(1, max_seq_len),
            stride=(1, max_seq_len),
        )

        new_bsz, _, slen = xs_pad.shape
        # N x D x max_seq_len x T'
        xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
        # N x T' x max_seq_len x D
        xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
        # NT' x max_seq_len x D
        xs_pad = xs_pad.view(-1, max_seq_len, D)
        return xs_pad

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1053
    @staticmethod
    def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
        """
        The function is very important for Transformer Transducer Streaming mode
        Args:
            xs_len (int): sequence length
            chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
            left_window (int): how many left chunks can be seen
            right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
            Returns:
                mask (torch.Tensor): a mask tensor for streaming model
                Torch 1.0.1
                tensor([[1., 1., 0., 0.],
                        [0., 1., 1., 0.],
                        [0., 0., 1., 1.]])
                Torch 1.4.1
                tensor([[True., True., False., False.],
                        [False., True., True., False.],
                        [False., False., True., True.]])
        """
        chunk_start_idx = torch.Tensor(chunk_start_idx).long()  # first idx of each chunk, such as [0,18,36,48].
        start_pad = torch.nn.functional.pad(
            chunk_start_idx, (1, 0)
        )  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
        end_pad = torch.nn.functional.pad(
            chunk_start_idx, (0, 1), value=x_len
        )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
        seq_range = torch.arange(0, x_len).unsqueeze(-1)  # seq_range size: [x_len, 1]
        idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]  # idx size: [x_len]
        end_pad[idx]  # boundary size: [x_len]
        seq_range_expand = (
            torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
        )  # seq_range_expand size [x_len, x_len]
        idx_left = idx - left_window
        idx_left[idx_left < 0] = 0
        boundary_left = start_pad[idx_left]
        mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
        idx_right = idx + right_window
        idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
        boundary_right = end_pad[idx_right]
        mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
        return mask_left & mask_right

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L494-L512
    @staticmethod
    def get_vision_position_ids(pixel_values, patch_attention_mask, patch_size=14, num_patches_per_side=32):
        batch_size = pixel_values.shape[0]
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
        return position_ids

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1561
    def embed_tokens_extend(
        self,
        input_ids: torch.LongTensor,
        input_image_embeds: torch.FloatTensor = None,
        input_audio_embeds: torch.FloatTensor = None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_projection_mode="speech",
        past_key_values=None,
    ):
        if past_key_values is not None:
            return self.language_model.embed_tokens(input_ids)

        new_input_ids = input_ids.clone()
        new_input_ids[
            (input_ids >= self._COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0])
            & (input_ids <= self._COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1])
        ] = self._IMAGE_SPECIAL_TOKEN_ID
        new_input_ids[
            (input_ids >= self._COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0])
            & (input_ids <= self._COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1])
        ] = self._AUDIO_SPECIAL_TOKEN_ID
        input_ids = new_input_ids
        image_position_mask = (input_ids == self._IMAGE_SPECIAL_TOKEN_ID).unsqueeze(-1)
        non_image_position_mask = ~image_position_mask
        hidden_states = torch.from_numpy(self.language_model.embed_tokens(input_ids))
        vision_hidden_states = self.image_embed(
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            image_pixel_values=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
        )
        audio_hidden_states = self.audio_embed(
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            audio_input_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_projection_mode=audio_projection_mode,
        )
        if vision_hidden_states is not None and audio_hidden_states is not None:
            hidden_states = vision_hidden_states * image_position_mask + audio_hidden_states * non_image_position_mask
        elif vision_hidden_states is not None:
            hidden_states = vision_hidden_states
        elif audio_hidden_states is not None:
            hidden_states = audio_hidden_states

        return hidden_states

    def get_multimodal_embeddings(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        input_image_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_attention_mask=None,
        input_audio_embeds: Optional[torch.FloatTensor] = None,
        audio_embed_sizes=None,
        input_mode=None,
        **kwargs,
    ):
        if pixel_values is not None and input_image_embeds is None:
            input_image_embeds = pixel_values
        audio_projection_mode = None
        if input_audio_embeds is not None:
            if isinstance(input_mode, torch.Tensor):
                assert len(input_mode) == 1
                input_mode = input_mode[0].item()
            if input_mode is None:
                input_mode = 1 if input_image_embeds is not None else 2
            input_mode = InputMode(input_mode)

            if input_mode in [InputMode.VISION_SPEECH, InputMode.VISION]:
                audio_projection_mode = "vision"
            elif input_mode == InputMode.SPEECH:
                audio_projection_mode = "speech"
            elif input_mode == InputMode.LANGUAGE:
                audio_projection_mode = "speech"
            else:
                raise ValueError(f"Invalid input_mode: {input_mode}")
        inputs_embeds = self.embed_tokens_extend(
            input_ids=input_ids,
            input_image_embeds=input_image_embeds,
            input_audio_embeds=input_audio_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            audio_embed_sizes=audio_embed_sizes,
            audio_projection_mode=audio_projection_mode,
            past_key_values=kwargs.get("past_key_values"),
        )
        return inputs_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        image_token = getattr(processor.tokenizer, "image_token", "<|image_1|>")
        audio_token = getattr(processor.tokenizer, "audio_token", "<|audio_1|>")
        if audio is not None and audio_token not in text:
            text = audio_token + text
        if image is not None and image_token not in text:
            text = image_token + text
        if processor.tokenizer.chat_template is None:
            if not text.startswith(user_prompt):
                text = user_prompt + text + prompt_suffix + assistant_prompt
        else:
            text = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
        audio_input = {}
        if "audio" in inspect.signature(processor.__call__).parameters:
            sample_rate = None
            if isinstance(audio, tuple):
                audio, sample_rate = audio
            if isinstance(audio, list) and len(audio) == 1 and isinstance(audio[0], tuple):
                audio, sample_rate = audio[0]
            audio_input["audio"] = audio
            if sample_rate is not None:
                audio_input["sampling_rate"] = sample_rate
        else:
            audio_input["audios"] = audio
        inputs = processor(text=text, images=image, **audio_input, return_tensors="pt")
        return inputs

    def get_img_features(self, pixel_values, image_attention_mask):
        patch_position_ids = self.get_vision_position_ids(
            pixel_values, image_attention_mask, self.patch_size, self.num_patches_per_side
        )
        return torch.from_numpy(
            self.vision_embeddings(
                pixel_values=pixel_values,
                patch_attention_mask=image_attention_mask,
                patch_position_ids=patch_position_ids,
            )[0]
        )


class _OVLlama4ForCausalLM(OVModelForVisualCausalLM):
    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        # Llama4 preprocessor creates bf16 tensor for pixel values, it can not be represented as numpy array
        if pixel_values.dtype != torch.float32:
            pixel_values.to(torch.float32)
        return self.vision_embeddings(pixel_values.to(torch.float32)).last_hidden_state

    # Adopted from https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py#L1743-L1759
    def merge_vision_text_embeddings(
        self, vision_embeds, inputs_embeds, input_ids=None, attention_mask=None, position_ids=None, **kwargs
    ):
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds
        original_inputs_embeds_shape = inputs_embeds.shape
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        final_mask = special_image_mask.to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))

        final_mask_1d = final_mask[..., 0].reshape(-1)
        num_tokens_to_fill = final_mask_1d.sum()

        if num_tokens_to_fill != image_features.size(0):
            raise ValueError(
                f"Mismatch: final_mask wants {num_tokens_to_fill} embeddings, "
                f"but multi_modal_projector returned {image_features.size(0)}"
            )

        expanded_mask = final_mask_1d.unsqueeze(-1).expand(-1, inputs_embeds.size(-1))
        inputs_embeds.masked_scatter_(expanded_mask, image_features)

        inputs_embeds = inputs_embeds.view(original_inputs_embeds_shape)

        return inputs_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")

        if video is not None:
            raise ValueError("video input is not supported")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        return inputs


MODEL_TYPE_TO_CLS_MAPPING = {
    "llava": _OVLlavaForCausalLM,
    "llava_next": _OVLlavaNextForCausalLM,
    "llava_next_video": _OVLlavaNextVideoForCausalLM,
    "minicpmv": _OVMiniCPMVForCausalLM,
    "llava-qwen2": _OVNanoLlavaForCausalLM,
    "maira2": _OVMaira2ForCausalLM,
    "phi3_v": _OVPhi3VisionForCausalLM,
    "internvl_chat": _OVInternVLForCausalLM,
    "qwen2_vl": _OVQwen2VLForCausalLM,
    "qwen2_5_vl": _OVQwen2_5_VLForCausalLM,
    "got_ocr2": _OVGotOCR2ForCausalLM,
    "gemma3": _OVGemma3ForCausalLM,
    "idefics3": _OVIdefics3ForCausalLM,
    "smolvlm": _OVSmolVLForCasualLM,
    "phi4mm": _OVPhi4MMForCausalLM,
    "phi4_multimodal": _OVPhi4MMForCausalLM,
    "llama4": _OVLlama4ForCausalLM,
}
