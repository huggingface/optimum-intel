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
    def __init__(self, model: ov.Model, text_embeds_model: ov.Model,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs
    ):
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
        del self.text_emb_request
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
        self._model_dir = Path(model_dir or parent_model._model_save_dir)

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
    _model_name = "vision_embeddings"
    def __init__(self, model:ov.Model, parent_model:OVBaseModel) -> None:
        super().__init__(model, parent_model, model_name=self._model_name)
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
    _model_name = "multi_modal_projector"
    def forward(self, hidden_state):
        result = self.request(hidden_state)[0]
        return result

MODEL_PARTS_CLS_MAPPING = {
    "multi_modal_projector": OVMultiModalProjector
}


class OVModelForVisualCausalLM(OVBaseModel, GenerationMixin):
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
        self._model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])
        self.lm_model = language_model
        self.text_embdings_model = text_embeddings
        self.vision_embeddings_model = vision_embeddings
        self._supports_cache_class = False
        self.main_input_name = "input_ids"

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

    def compile(self):
        self.language_model.compile()
        self.vision_embeddings._compile()
        for part in self.additional_parts:
            part_model = getattr(self, part, None)
            if part_model is not None:
                part_model._compile()

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

        model_cls = MODEL_TYPE_TO_CLS_MAPPING[config.model_type]

        quantization_config = model_cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)

        # Load model from a local directory
        if os.path.isdir(model_id):
            language_model = model_cls.load_model(os.path.join(model_id, language_model_file_name), quantization_config)
            text_embeddings = model_cls.load_model(os.path.join(model_id, text_embeddings_file_name), quantization_config)
            vision_embeddings = model_cls.load_model(os.path.join(model_id, vision_embeddings_file_name), quantization_config)

            for part in model_cls.additional_parts:
                part_file_name = f"openvino_{part}_model.xml"
                part_model = model_cls.load_model(os.path.join(model_id, part_file_name), quantization_config)
                kwargs[part] = part_model

            model_save_dir = Path(model_id)

        # Load model from hub
        else:
            model_file_names = {"language_model": language_model_file_name, "text_embeddings": text_embeddings_file_name, "vision_embeddings": vision_embeddings_file_name}
            for part in model_cls.additional_parts:
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
            language_model = model_cls.load_model(file_names["language_model"], quantization_config)
            text_embeddings = model_cls.load_model(file_names["text_embeddings"], quantization_config)
            vision_embeddings = model_cls.load_model(file_names["vision_emnbeddings"], quantization_config)
            for part in model_cls.additional_parts:
                kwargs[part] = model_cls.load_model(file_names[part], quantization_config)

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

        return model_cls(
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

    def forward(self, input_ids, pixel_values, past_key_values=None, inputs_embeds=None, image_sizes=None, attention_mask=None, position_ids=None, **kwargs):
        inputs_embeds, attention_mask, position_ids = self.get_multimodal_embeddings(input_ids, pixel_values, image_sizes=image_sizes, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values , **kwargs)
        return self.language_model.forward(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)


    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)


    def get_vision_embeddings(self, pixel_values, **kwargs):
        raise NotImplementedError

    def get_text_embeddings(self, input_ids, **kwargs):
        return self.language_model.embed_tokens(input_ids)

    def merge_vision_text_embeddings(self, vision_embeds, inputs_embeds, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        raise NotImplementedError

    def get_multimodal_embeddings(self, input_ids, pixel_values=None, attention_mask=None, position_ids=None, **kwargs):
        inputs_embeds = self.get_text_embeddings(input_ids, **kwargs)
        if pixel_values is not None:
            vision_embeds = self.get_vision_embeddings(pixel_values, input_ids=input_ids, **kwargs)
            if vision_embeds is not None:
                inputs_embeds, attention_mask, position_ids = self.merge_vision_text_embeddings(vision_embeds, inputs_embeds, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
        return inputs_embeds, attention_mask, position_ids

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
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
                # input_ids based on the past_length.llava
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
                elif self.config.image_token_index in input_ids:
                    input_ids = input_ids[:, input_ids.shape[1] - 1 :]

            position_ids = kwargs.get("position_ids", None)
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch gllavaenerationsubset_siz
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                }
            )
            return model_inputs

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True



class _OVLlavaForCausalLM(OVModelForVisualCausalLM):
    additional_parts = ["multi_modal_projector"]

    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        image_outputs = self.vision_embeddings(pixel_values)
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        image_features = self.multi_modal_projector(selected_image_feature)

        return image_features
    
    def merge_vision_text_embeddings(self, vision_embeds, inputs_embeds, input_ids, attention_mask, position_ids=None, **kwargs):
        image_features = torch.from_numpy(vision_embeds)
        inputs_embeds = torch.from_numpy(inputs_embeds)
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
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        return final_embedding, final_attention_mask, position_ids

    def get_multimodal_embeddings(self, input_ids, pixel_values=None, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        inputs_embeds, attention_mask, position_ids = super().get_multimodal_embeddings(input_ids, pixel_values, attention_mask, position_ids, **kwargs)

        if pixel_values is not None and past_key_values is not None:
            if not self.language_model.stateful:
                first_layer_past_key_value = torch.from_numpy(past_key_values[0][0][:, :, :, 0])
            else:
                first_layer_past_key_value = torch.from_numpy(self.language_model.request.query_state()[0].state.data[:, :, :, 0])

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        return inputs_embeds, attention_mask, position_ids


MODEL_TYPE_TO_CLS_MAPPING = {
    "llava": _OVLlavaForCausalLM,
}
