import os
import logging
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Dict, Optional, Tuple
from openvino.runtime.ie_api import Model
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from optimum.intel.openvino.configuration import OVWeightQuantizationConfig
from .modeling_decoder import OVModelForCausalLM, CausalLMOutputWithPast
from .modeling_base import OVBaseModel
from .utils import OV_TO_PT_TYPE, OV_TO_NP_TYPE, _print_compiled_model_properties
import torch
import openvino as ov
import numpy as np

logger = logging.getLogger(__name__)

core = ov.Core()

class OVModelWithEmbedForCausalLM(OVModelForCausalLM):
    def __init__(self, model: Model, text_embeds_model: Model, config: PretrainedConfig = None, device: str = "CPU", dynamic_shapes: bool = True, ov_config: Dict[str, str] | None = None, model_save_dir: str | Path | TemporaryDirectory | None = None, quantization_config: OVWeightQuantizationConfig | Dict | None = None, **kwargs):
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

class OVModelForVisualCausalLM(OVBaseModel):
    pass


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
    def __init__(self, model:ov.Model, parent_model:OVModelForVisualCausalLM) -> None:
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


class OVVisionModelForCausalLM(OVBaseModel):
    export_feature = "image-text-to-text"

    def __init__(
        self,
        encoder: openvino.runtime.Model,
        decoder: openvino.runtime.Model,
        decoder_with_past: openvino.runtime.Model = None,
        config: PretrainedConfig = None,
        device: str = "CPU",
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        self.config = config
        self.use_cache = decoder_with_past is not None
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])

        if self.is_dynamic:
            encoder = self._reshape(encoder, -1, -1, is_decoder=False)
            decoder = self._reshape(decoder, -1, -1)
            decoder_with_past = self._reshape(decoder_with_past, -1, -1) if self.use_cache else None
        self.encoder_model = encoder
        self.decoder_model = decoder
        self.decoder_with_past_model = decoder_with_past
        if self.can_generate():
            self.generation_config = kwargs.get("generation_config", GenerationConfig.from_model_config(config))
        else:
            self.generation_config = None
        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
        """
        src_files = [self.encoder_model, self.decoder_model]
        dst_file_names = [OV_ENCODER_NAME, OV_DECODER_NAME]
        if self.use_cache:
            src_files.append(self.decoder_with_past_model)
            dst_file_names.append(OV_DECODER_WITH_PAST_NAME)

        for src_file, dst_file_name in zip(src_files, dst_file_names):
            dst_path = os.path.join(save_directory, dst_file_name)
            openvino.save_model(src_file, dst_path, compress_to_fp16=False)

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
        encoder_file_name: Optional[str] = None,
        decoder_file_name: Optional[str] = None,
        decoder_with_past_file_name: Optional[str] = None,
        local_files_only: bool = False,
        use_cache: bool = True,
        from_onnx: bool = False,
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

        default_encoder_file_name = ONNX_ENCODER_NAME if from_onnx else OV_ENCODER_NAME
        default_decoder_file_name = ONNX_DECODER_NAME if from_onnx else OV_DECODER_NAME
        default_decoder_with_past_file_name = ONNX_DECODER_WITH_PAST_NAME if from_onnx else OV_DECODER_WITH_PAST_NAME
        encoder_file_name = encoder_file_name or default_encoder_file_name
        decoder_file_name = decoder_file_name or default_decoder_file_name
        decoder_with_past_file_name = decoder_with_past_file_name or default_decoder_with_past_file_name
        decoder_with_past = None

        quantization_config = cls._prepare_weight_quantization_config(quantization_config, load_in_8bit)

        # Load model from a local directory
        if os.path.isdir(model_id):
            encoder = cls.load_model(os.path.join(model_id, encoder_file_name), quantization_config)
            decoder = cls.load_model(os.path.join(model_id, decoder_file_name), quantization_config)
            if use_cache:
                decoder_with_past = cls.load_model(
                    os.path.join(model_id, decoder_with_past_file_name), quantization_config
                )

            model_save_dir = Path(model_id)

        # Load model from hub
        else:
            model_file_names = {"encoder": encoder_file_name, "decoder": decoder_file_name}
            if use_cache:
                model_file_names["decoder_with_past"] = decoder_with_past_file_name

            # If not ONNX then OpenVINO IR : adds binary files
            if not from_onnx:
                for key in list(model_file_names.keys()):
                    model_file_names[key + "_bin"] = model_file_names[key].replace(".xml", ".bin")
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
            encoder = cls.load_model(file_names["encoder"], quantization_config)
            decoder = cls.load_model(file_names["decoder"], quantization_config)
            if use_cache:
                decoder_with_past = cls.load_model(file_names["decoder_with_past"], quantization_config)

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
            encoder=encoder,
            decoder=decoder,
            decoder_with_past=decoder_with_past,
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
            if use_cache:
                task = task + "-with-past"

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

    def _reshape(self, model: ov.Model, batch_size: int, sequence_length: int, is_decoder=True):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size if not is_decoder else -1
            if inputs.get_any_name().startswith("past_key_values"):
                shapes[inputs][2] = -1
            elif inputs.get_any_name().startswith("cache_position"):
                shapes[inputs][0] = sequence_length
            elif is_decoder and not inputs.get_any_name().startswith("encoder"):
                shapes[inputs][1] = -1
            else:
                shapes[inputs][1] = sequence_length
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            sequence_length (`int`):
                The sequence length.
        """
        logger.warning("Some part of the model's decoder do not support static shapes and will be kept dynamic.")
        self.is_dynamic = True if batch_size == -1 and sequence_length == -1 else False
        self.encoder_model = self._reshape(self.encoder_model, batch_size, sequence_length, is_decoder=False)
        self.decoder_model = self._reshape(self.decoder_model, batch_size, sequence_length)
        if self.use_cache:
            self.decoder_with_past_model = self._reshape(self.decoder_with_past_model, batch_size, sequence_length)

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        apply_moc_transformations(self.encoder_model, cf=False)
        apply_moc_transformations(self.decoder_model, cf=False)
        compress_model_transformation(self.encoder_model)
        compress_model_transformation(self.decoder_model)
        if self.use_cache:
            apply_moc_transformations(self.decoder_with_past_model, cf=False)
            compress_model_transformation(self.decoder_with_past_model)
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError
