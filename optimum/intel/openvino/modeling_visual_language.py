import os
import logging
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Dict, Optional, Tuple, Union
import warnings
from transformers import PretrainedConfig, GenerationConfig, GenerationMixin, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from .utils import OV_TO_PT_TYPE, _print_compiled_model_properties
import torch
import openvino as ov
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation
import numpy as np
from .modeling_decoder import OVModelForCausalLM, CausalLMOutputWithPast
from .modeling_base import OVBaseModel
from .configuration import OVConfig, OVWeightQuantizationConfig
from ...exporters.openvino import main_export

logger = logging.getLogger(__name__)

core = ov.Core()

class OVModelWithEmbedForCausalLM(OVModelForCausalLM):
    def __init__(self, model: ov.Model, text_embeds_model: ov.Model, config: PretrainedConfig = None, device: str = "CPU", dynamic_shapes: bool = True, ov_config: Dict[str, str] | None = None, model_save_dir: str | Path | TemporaryDirectory | None = None, quantization_config: OVWeightQuantizationConfig | Dict | None = None, **kwargs):
        self.model = model
        self.text_emb_model = text_embeds_model
        self.request = None
        self.text_emb_request = None
        
        super().__init__(model, config, device, dynamic_shapes, ov_config, model_save_dir, quantization_config, **kwargs)

    def compile(self):
        if self.request is None:
            self.request = core.compile_model(self.model, self._device, self.ov_config).create_infer_request()
        self._compile_text_emb()

    def _compile_text_emb(self):
        if self.text_emb_request is None:
            self.text_emb_request = core.compile_model(self.text_emb_model, self._device, self.ov_config)

    def to(self, device: str):
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()

        return self

    def clear_requests(self):
        del self.request
        del self.token_emb_request
        self.request = None
        self.token_emb_request = None

    def embed_tokens(self, input_ids: torch.LongTensor):
        self._compile_token_emb()
        res = self.token_emb_request(input_ids, share_inputs=True)
        return res[0]

    def prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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
                attention_mask = np.array(attention_mask)
            else:
                attention_mask = np.ones((inputs_embeds.shape[0], inputs_embeds.shape[1] + past_len), dtype=int)

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "position_ids" in self.input_names:
            if position_ids is not None:
                position_ids = np.array(position_ids)
            else:
                position_ids = np.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            inputs["position_ids"] = position_ids

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

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
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class OVModelPart:

    def __init__(
        self,
        model: ov.Model,
        parent_model: OVBaseModel,
        ov_config: Optional[Dict[str, str]] = None,
        model_name: str = "encoder",
        model_dir: str = None,
    ):
        self.model = model
        self.parent_model = parent_model
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.input_dtype = {
            inputs.get_any_name(): OV_TO_PT_TYPE[inputs.get_element_type().get_type_name()]
            for inputs in self.model.inputs
        }
        self.ov_config = ov_config or {**self.parent_model.ov_config}
        self.request = None
        self._model_name = model_name
        self.config = self.parent_model.config

    def _compile(self):
        if self.request is None:
            if (
                "CACHE_DIR" not in self.ov_config.keys()
                and not str(self._model_dir).startswith(gettempdir())
                and "GPU" in self._device
            ):
                self.ov_config["CACHE_DIR"] = os.path.join(self._model_dir, self._model_name, "model_cache")

            logger.info(f"Compiling the {self._model_name} to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, self.ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                logger.info(f"{self._device} SUPPORTED_PROPERTIES:")
                _print_compiled_model_properties(self.request)

    @property
    def _device(self) -> str:
        return self.parent_model._device

    @property
    def device(self) -> torch.device:
        return self.parent_model.device


    @property
    def dtype(self) -> Optional[torch.dtype]:
        for dtype in self.input_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        for dtype in self.output_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype.is_floating_point:
                return torch_dtype

        return None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class OVVisionEmbedding(OVModelPart):
    def __init__(self, model:ov.Model, parent_model:OVBaseModel) -> None:
        super(model, parent_model)
        self.output_dtypes = {key.get_any_name(): key.get_element_type().get_type_name() for key in self.model.outputs}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        self.hidden_states_output_names = [key for key in self.output_names if "hidden_states" in key]

    def forward(self, pixel_values, **kwargs):
        result = self.request({"pixel_values": pixel_values})
        pooled_out = result[0]
        last_hidden_state = result[1]
        hidden_states = None
        if self.hidden_states_output_names:
            hidden_states = []
            for out in self.hidden_states_output_names:
                hidden_states.append(result[out])
        return BaseModelOutputWithPooling(pooler_output=pooled_out, last_hidden_state=last_hidden_state, hidden_states=hidden_states)


class OVMultiModalProjector(OVModelPart):
    def forward(self, hidden_state):
        result = self.request(hidden_state)[0]
        return result

MODEL_PARTS_CLS_MAPPING = {
    "multi_modal_projector": OVMultiModalProjector
}

class OVVisionModelForCausalLM(OVBaseModel, GenerationMixin):
    export_feature = "image-text-to-text"
    additional_parts = []

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
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])
        self.lm_model = language_model
        self.text_embdings_model = text_embeddings
        self.vision_embeddings_model = vision_embeddings

        for part in self.additional_parts:
            setattr(self, f"{part}_model", kwargs.get(part))

        enable_compilation = kwargs.get("compile", True)
        self.generation_config = kwargs.get("generation_config", GenerationConfig.from_model_config(config))
        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()
        self.language_model = OVModelWithEmbedForCausalLM(self.lm_model, self.text_embdings_model, config=config, deivce=device, ov_config=ov_config, model_save_dir=model_save_dir, quantization_config=quantization_config, compile=False)
        self.vision_embeddings = OVVisionEmbedding(self.vision_embeddings_model, self)
        for part in self.additional_parts:
            model_part = getattr(self, f"{part}_model", None)
            if model_part is not None:
                model_part = MODEL_PARTS_CLS_MAPPING[part](model_part, self)
            setattr(self, part, model_part)
        
        if enable_compilation:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        try:
            self.auto_model_class.register(AutoConfig, self.__class__)
        except AttributeError:
            pass

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        src_files = [self.lm_model, self.text_embdings_model, self.vision_embeddings_model]
        dst_file_names = ["openvino_language_model.xml", "openvino_text_embeddings_model.xml", "openvino_vision_embeddings.xml"]
        for part in self.additional_parts:
            model = getattr(self, f"{part}_model", None)
            if model is not None:
                src_files.append(model)
                dst_file_names.append(f"openvino_{part}_model.xml")

        for src_file, dst_file_name in zip(src_files, dst_file_names):
            dst_path = os.path.join(save_directory, dst_file_name)
            ov.save_model(src_file, dst_path, compress_to_fp16=False)

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

        language_model_file_name = "openvino_language_model.xml"
        text_embeddings_file_name = "openvino_text_embeddings_model.xml"
        vision_embeddings_file_name = "openvino_vision_embeddings_model.xml"

        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)

        # Load model from a local directory
        if os.path.isdir(model_id):
            language_model = cls.load_model(os.path.join(model_id, language_model_file_name), quantization_config)
            text_embeddings = cls.load_model(os.path.join(model_id, text_embeddings_file_name), quantization_config)
            vision_embeddings = cls.load_model(os.path.join(model_id, vision_embeddings_file_name), quantization_config)

            for part in cls.additional_parts:
                part_file_name = f"openvino_{part}_model.xml"
                part_model = cls.load_model(os.path.join(model_id, part_file_name), quantization_config)
                kwargs[part] = part_model

            model_save_dir = Path(model_id)

        # Load model from hub
        else:
            model_file_names = {"language_model": language_model_file_name, "text_embeddings": text_embeddings_file_name, "vision_embeddings": vision_embeddings_file_name}
            for part in cls.additional_parts:
                model_file_names[part] = part_file_name = f"openvino_{part}_model.xml"

            file_names = model_file_names.copy()
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
            language_model = cls.load_model(file_names["language_model"], quantization_config)
            text_embeddings = cls.load_model(file_names["text_embeddings"], quantization_config)
            vision_embeddings = cls.load_model(file_names["vision_emnbeddings"], quantization_config)
            for part in cls.additional_parts:
                kwargs[part] = cls.load_model(file_names[part], quantization_config)

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

        return cls(
            language_model=language_model,
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            config=config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            **kwargs,
        )

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
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, defaults to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`Optional[str]`, defaults to `None`):
                Deprecated. Please use `token` instead.
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

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
            ov_config = OVConfig(dtype="fp32")

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
        )

        config.save_pretrained(save_dir_path)
        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_cache=use_cache,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )

    
    def reshape(self, batch_size: int, sequence_length: int):
        logger.warning("Static shapes are not supported for causal language model.")
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        apply_moc_transformations(self.lm_model, cf=False)
        compress_model_transformation(self.lm_model)
        apply_moc_transformations(self.text_embdings_model, cf=False)
        compress_model_transformation(self.text_embdings_model)
        apply_moc_transformations(self.vision_embeddings_model, cf=False)
        compress_model_transformation(self.vision_embeddings_model)
        for part in self.additional_parts:
            model = getattr(self, f"{part}_model", None)
            if model is not None:
                apply_moc_transformations(model, cf=False)
                compress_model_transformation(model)
        return self

    def forward(self, input_ids, pixel_values, **kwargs):
        inputs_embeds = self.get_multimodal_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model.forward(input_ids=None, inputs_embeds=inputs_embeds, **kwargs)


    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)


    def get_vision_embeddings(self, pixel_values, **kwargs):
        raise NotImplementedError

    def get_text_embeddings(self, input_ids, **kwargs):
        return self.language_model.embed_tokens(input_ids)

    def merge_vision_text_embeddings(self, vision_embeds, inputs_embeds):
        raise NotImplementedError

    def get_multimodal_embeddings(self, input_ids, pixel_values=None, **kwargs):
        inputs_embeds = self.get_text_embeddings(input_ids, **kwargs)
        if pixel_values is not None:
            vision_embeds = self.get_vision_embeddings(pixel_values, **kwargs)
            inputs_embeds = self.merge_vision_text_embeddings(vision_embeds, inputs_embeds)
        return inputs_embeds
