#  Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
OpenVINO Paraformer Speech-to-Text Model Implementation
Following the pattern from optimum-intel's modeling_text2speech.py
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openvino
from openvino import CompiledModel, Core
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoConfig, PretrainedConfig
from transformers.utils import ModelOutput

from .utils import OV_DECODER_NAME, OV_ENCODER_NAME, OV_XML_FILE_NAME, OV_TO_PT_TYPE

logger = logging.getLogger(__name__)

core = Core()

# Additional model file name for Paraformer predictor
OV_PREDICTOR_NAME = "openvino_predictor_model.xml"


@dataclass
class ParaformerModelOutput(ModelOutput):
    """
    Output type of ParaformerModel.
    
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Predicted logits for each token.
        token_num (`torch.LongTensor` of shape `(batch_size,)`):
            Number of predicted tokens for each sequence.
        token_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Decoded token IDs (if `decode=True`).
    """
    logits: torch.FloatTensor = None
    token_num: torch.LongTensor = None
    token_ids: torch.LongTensor = None


class OVParaformerModelPart:
    """
    Base class for OpenVINO Paraformer model components.
    Following the OVModelPart pattern from optimum-intel.
    """
    _model_name = "model"
    
    def __init__(
        self,
        model: Union[Model, CompiledModel],
        parent_model: "OVParaformerForSpeechSeq2Seq",
        ov_config: Optional[Dict[str, str]] = None,
        model_name: str = None,
    ):
        self.model = model
        self.parent_model = parent_model
        self._model_name = model_name or self._model_name
        
        self._compile_only = getattr(parent_model, '_compile_only', False)
        self.ov_config = ov_config or getattr(parent_model, 'ov_config', {}).copy()
        
        # Initialize request
        if self._compile_only and isinstance(model, CompiledModel):
            self.request = model.create_infer_request()
        else:
            self.request = None
        
        # Extract input/output metadata
        model_for_meta = model.get_runtime_model() if isinstance(model, CompiledModel) else model
        
        self.input_names = {}
        self.input_dtypes = {}
        for idx, inp in enumerate(model_for_meta.inputs):
            try:
                names = inp.get_names()
                name = next((n for n in names if "/" not in n), list(names)[0] if names else f"input_{idx}")
            except Exception:
                name = f"input_{idx}"
            self.input_names[name] = idx
            self.input_dtypes[name] = inp.get_element_type().get_type_name()
        
        self.output_names = {}
        self.output_dtypes = {}
        for idx, out in enumerate(model_for_meta.outputs):
            try:
                names = out.get_names()
                name = next((n for n in names if "/" not in n), list(names)[0] if names else f"output_{idx}")
            except Exception:
                name = f"output_{idx}"
            self.output_names[name] = idx
            self.output_dtypes[name] = out.get_element_type().get_type_name()
    
    @property
    def _device(self) -> str:
        return self.parent_model._device
    
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")
    
    @property
    def dtype(self) -> Optional[torch.dtype]:
        for dtype in self.input_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype is not None and torch_dtype.is_floating_point:
                return torch_dtype
        for dtype in self.output_dtypes.values():
            torch_dtype = OV_TO_PT_TYPE.get(dtype)
            if torch_dtype is not None and torch_dtype.is_floating_point:
                return torch_dtype
        return None
    
    def compile(self):
        """Compile the model for inference."""
        if self._compile_only and isinstance(self.model, CompiledModel):
            if self.request is None:
                self.request = self.model.create_infer_request()
            return
        
        if self.request is None:
            # Set cache directory for GPU
            model_dir = getattr(self.parent_model, 'model_save_dir', None)
            if (
                model_dir is not None
                and "CACHE_DIR" not in self.ov_config
                and not str(model_dir).startswith(gettempdir())
                and "gpu" in self._device.lower()
            ):
                self.ov_config["CACHE_DIR"] = os.path.join(str(model_dir), self._model_name, "model_cache")
            
            logger.info(f"Compiling {self._model_name} to {self._device}...")
            compiled_model = core.compile_model(self.model, self._device, self.ov_config)
            self.request = compiled_model.create_infer_request()
            logger.info(f"✅ {self._model_name} compiled successfully")
    
    def clear_requests(self):
        """Clear inference request to free resources."""
        if self._compile_only:
            raise ValueError("`clear_requests()` is not supported in `compile_only` mode")
        self.request = None
    
    def _prepare_input(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class OVParaformerEncoder(OVParaformerModelPart):
    """
    Paraformer Encoder component for OpenVINO inference.
    
    Processes input speech features and produces encoder hidden states.
    """
    _model_name = "encoder"
    
    def forward(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            speech: Input speech features [batch, time, features]
            speech_lengths: Valid lengths for each sequence [batch]
        
        Returns:
            encoder_out: Encoded hidden states [batch, time, hidden]
            encoder_out_lens: Output lengths [batch]
        """
        self.compile()
        
        inputs = {
            "speech": self._prepare_input(speech),
            "speech_lengths": self._prepare_input(speech_lengths),
        }
        
        self.request.infer(inputs)
        
        encoder_out = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        encoder_out_lens = torch.from_numpy(self.request.get_output_tensor(1).data.copy())
        
        return encoder_out, encoder_out_lens


class OVParaformerPredictor(OVParaformerModelPart):
    """
    Paraformer CIF Predictor component for OpenVINO inference.
    
    Predicts acoustic embeddings and token counts from encoder output.
    """
    _model_name = "predictor"
    
    def forward(
        self,
        encoder_out: Union[torch.Tensor, np.ndarray],
        encoder_out_lens: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the CIF predictor.
        
        Args:
            encoder_out: Encoder output [batch, time, hidden]
            encoder_out_lens: Encoder output lengths [batch]
        
        Returns:
            acoustic_embeds: Predicted acoustic embeddings [batch, token_num, hidden]
            token_num: Number of predicted tokens [batch]
            alphas: CIF weights [batch, time] (optional)
            peak_index: Peak indices [batch, token_num] (optional)
        """
        self.compile()
        
        # Create attention mask [batch, 1, max_len]
        if isinstance(encoder_out, torch.Tensor):
            batch_size, max_len = encoder_out.shape[0], encoder_out.shape[1]
            arange = torch.arange(max_len, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
            mask = (arange < encoder_out_lens.unsqueeze(1).to(torch.int32)).to(torch.float32)
            mask = mask.unsqueeze(1)
        else:
            batch_size, max_len = encoder_out.shape[0], encoder_out.shape[1]
            arange = np.arange(max_len, dtype=np.int32)[np.newaxis, :].repeat(batch_size, axis=0)
            mask = (arange < encoder_out_lens[:, np.newaxis]).astype(np.float32)
            mask = mask[:, np.newaxis, :]
        
        # Map encoder_out and mask to actual OV input names using discovered input_names
        # to avoid mismatch with TorchScript arg names
        input_names_list = list(self.input_names.keys())
        inputs = {}
        if len(input_names_list) > 0:
            inputs[input_names_list[0]] = self._prepare_input(encoder_out)
        if len(input_names_list) > 1:
            inputs[input_names_list[1]] = self._prepare_input(mask)
        
        self.request.infer(inputs)
        
        acoustic_embeds = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        token_num = torch.from_numpy(self.request.get_output_tensor(1).data.copy())
        
        alphas = None
        peak_index = None
        if len(self.output_names) > 2:
            alphas = torch.from_numpy(self.request.get_output_tensor(2).data.copy())
        if len(self.output_names) > 3:
            peak_index = torch.from_numpy(self.request.get_output_tensor(3).data.copy())
        
        return acoustic_embeds, token_num, alphas, peak_index


class OVParaformerDecoder(OVParaformerModelPart):
    """
    Paraformer Decoder component for OpenVINO inference.
    
    Produces output logits from encoder output and acoustic embeddings.
    """
    _model_name = "decoder"
    
    def forward(
        self,
        encoder_out: Union[torch.Tensor, np.ndarray],
        encoder_out_lens: Union[torch.Tensor, np.ndarray],
        acoustic_embeds: Union[torch.Tensor, np.ndarray],
        token_num: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            encoder_out: Encoder output [batch, time, hidden]
            encoder_out_lens: Encoder output lengths [batch]
            acoustic_embeds: Acoustic embeddings from predictor [batch, token_num, hidden]
            token_num: Number of tokens [batch]
        
        Returns:
            logits: Output logits [batch, token_num, vocab_size]
        """
        self.compile()
        
        inputs = {
            "encoder_out": self._prepare_input(encoder_out),
            "encoder_out_lens": self._prepare_input(encoder_out_lens),
            "acoustic_embeds": self._prepare_input(acoustic_embeds),
            "token_num": self._prepare_input(token_num),
        }
        
        self.request.infer(inputs)
        
        logits = torch.from_numpy(self.request.get_output_tensor(0).data.copy())
        
        return logits


class OVParaformerForSpeechSeq2Seq:
    """
    OpenVINO Paraformer model for automatic speech recognition.
    
    This class provides a unified interface for loading and running inference
    on Paraformer models exported to OpenVINO IR format. It supports both
    single-file models and multi-component (encoder/predictor/decoder) models.
    
    Following the pattern from optimum-intel's OVModelForTextToSpeechSeq2Seq.
    
    Args:
        model_path: Path to the model directory containing OpenVINO IR files
        device: Target device for inference (CPU, GPU, AUTO, etc.)
        ov_config: OpenVINO runtime configuration dictionary
        compile_only: If True, skip model loading and compile directly from files
        
    Example:
        ```python
        model = OVParaformerForSpeechSeq2Seq.from_pretrained(
            "/path/to/paraformer-zh/ov_models",
            device="GPU",
        )
        
        # Run inference
        output = model(speech_features, speech_lengths)
        token_ids = output.token_ids
        ```
    """
    
    auto_model_class = None
    export_feature = "automatic-speech-recognition"
    main_input_name = "speech"
    
    def __init__(
        self,
        model: Optional[Model] = None,
        encoder: Optional[Model] = None,
        predictor: Optional[Model] = None,
        decoder: Optional[Model] = None,
        config: Optional[PretrainedConfig] = None,
        device: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
        compile_only: bool = False,
        compile: bool = True,
        **kwargs,
    ):
        self.config = config
        self.model_save_dir = Path(model_save_dir) if model_save_dir else None
        self._device = device.upper()
        self.ov_config = ov_config.copy() if ov_config else {}
        self._compile_only = compile_only
        self.preprocessors = kwargs.get("preprocessors", [])
        self.generation_config = kwargs.get("generation_config", None)
        
        # Determine if we have a single model or separate components
        self._single_model = model is not None
        
        if self._single_model:
            # Single combined model
            self.model = model
            self._model_component = OVParaformerModelPart(
                model, self, ov_config=self.ov_config, model_name="model"
            )
            self.encoder = None
            self.predictor = None
            self.decoder = None
            
            # Extract I/O metadata from the single model
            self.input_names = self._model_component.input_names.copy()
            self.output_names = self._model_component.output_names.copy()
        else:
            # Separate components
            self.model = None
            self._model_component = None
            self.encoder = OVParaformerEncoder(encoder, self, model_name="encoder") if encoder else None
            self.predictor = OVParaformerPredictor(predictor, self, model_name="predictor") if predictor else None
            self.decoder = OVParaformerDecoder(decoder, self, model_name="decoder") if decoder else None
            
            # Combine I/O names
            self.input_names = {}
            self.output_names = {}
            if self.encoder:
                self.input_names.update(self.encoder.input_names)
            if self.decoder:
                self.output_names.update(self.decoder.output_names)
        
        if compile and not compile_only:
            self.compile()
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        device: str = "CPU",
        ov_config: Optional[Dict[str, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        compile_only: bool = False,
        compile: bool = True,
        **kwargs,
    ) -> "OVParaformerForSpeechSeq2Seq":
        """
        Load a Paraformer model from a local directory or Hugging Face Hub.
        
        Args:
            model_id: Local path or Hugging Face Hub model ID
            device: Target device (CPU, GPU, AUTO)
            ov_config: OpenVINO configuration dictionary
            token: Hugging Face authentication token
            revision: Model revision to use
            force_download: Force re-download from Hub
            cache_dir: Directory to cache downloaded models
            local_files_only: Only use local files, no Hub download
            compile_only: Load as compiled model directly
            compile: Whether to compile models after loading
            
        Returns:
            OVParaformerForSpeechSeq2Seq instance
        """
        model_path = Path(model_id)
        
        # Try to load config
        config = None
        config_paths = [
            model_path / "config.json",
            model_path / "config.yaml",
        ]
        for cfg_path in config_paths:
            if cfg_path.exists():
                try:
                    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    break
                except Exception:
                    pass
        
        # Check for single model file
        single_model_path = model_path / OV_XML_FILE_NAME
        if single_model_path.exists():
            logger.info(f"Loading single Paraformer model from {single_model_path}")
            model = cls._load_model(single_model_path, device if compile_only else None, ov_config)
            return cls(
                model=model,
                config=config,
                device=device,
                ov_config=ov_config,
                model_save_dir=model_path,
                compile_only=compile_only,
                compile=compile,
                **kwargs,
            )
        
        # Check for separate component files
        encoder_path = model_path / OV_ENCODER_NAME
        predictor_path = model_path / OV_PREDICTOR_NAME
        decoder_path = model_path / OV_DECODER_NAME
        
        if encoder_path.exists() and decoder_path.exists():
            logger.info(f"Loading Paraformer components from {model_path}")
            
            encoder = cls._load_model(encoder_path, device if compile_only else None, ov_config)
            decoder = cls._load_model(decoder_path, device if compile_only else None, ov_config)
            predictor = None
            if predictor_path.exists():
                predictor = cls._load_model(predictor_path, device if compile_only else None, ov_config)
            
            return cls(
                encoder=encoder,
                predictor=predictor,
                decoder=decoder,
                config=config,
                device=device,
                ov_config=ov_config,
                model_save_dir=model_path,
                compile_only=compile_only,
                compile=compile,
                **kwargs,
            )
        
        raise FileNotFoundError(
            f"Could not find Paraformer model files in {model_path}. "
            f"Expected either '{OV_XML_FILE_NAME}' or component files like '{OV_ENCODER_NAME}'."
        )
    
    @staticmethod
    def _load_model(
        path: Path,
        device: Optional[str] = None,
        ov_config: Optional[Dict[str, str]] = None,
    ) -> Union[Model, CompiledModel]:
        """Load an OpenVINO model from file."""
        logger.info(f"Loading model from {path}")
        model = core.read_model(path)
        
        if device is not None:
            # Compile directly (compile_only mode)
            return core.compile_model(model, device, ov_config or {})
        
        return model
    
    @property
    def device(self) -> torch.device:
        """Return torch device (always CPU for compatibility)."""
        return torch.device("cpu")
    
    @property
    def dtype(self) -> torch.dtype:
        """Return model dtype."""
        if self._model_component:
            return self._model_component.dtype
        if self.encoder:
            return self.encoder.dtype
        return torch.float32
    
    @property
    def _component_names(self) -> List[str]:
        """Return list of loaded component names."""
        if self._single_model:
            return ["model"]
        names = []
        if self.encoder: names.append("encoder")
        if self.predictor: names.append("predictor")
        if self.decoder: names.append("decoder")
        return names
    
    @property
    def components(self) -> Dict[str, OVParaformerModelPart]:
        """Return dictionary of model components."""
        if self._single_model:
            return {"model": self._model_component}
        comps = {}
        if self.encoder: comps["encoder"] = self.encoder
        if self.predictor: comps["predictor"] = self.predictor
        if self.decoder: comps["decoder"] = self.decoder
        return comps
    
    def to(self, device: str) -> "OVParaformerForSpeechSeq2Seq":
        """
        Move model to specified device.
        
        Args:
            device: Target device (CPU, GPU, AUTO)
            
        Returns:
            self for method chaining
        """
        if self._compile_only:
            raise ValueError("`to()` is not supported in `compile_only` mode")
        
        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()
        
        return self
    
    def compile(self):
        """Compile all model components for inference."""
        for component in self.components.values():
            component.compile()
    
    def clear_requests(self):
        """Clear all inference requests."""
        for component in self.components.values():
            component.clear_requests()
    
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
        decode: bool = True,
        **kwargs,
    ) -> ParaformerModelOutput:
        """
        Run inference on speech input.
        
        Args:
            speech: Input speech features [batch, time, features]
            speech_lengths: Valid lengths for each sequence [batch]
            decode: Whether to decode logits to token IDs
            
        Returns:
            ParaformerModelOutput containing logits, token_num, and optionally token_ids
        """
        return self.forward(speech, speech_lengths, decode=decode, **kwargs)
    
    def forward(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
        decode: bool = True,
        **kwargs,
    ) -> ParaformerModelOutput:
        """
        Forward pass through the model.
        
        Args:
            speech: Input speech features [batch, time, features]
            speech_lengths: Valid lengths for each sequence [batch]
            decode: Whether to decode logits to token IDs
            
        Returns:
            ParaformerModelOutput containing logits, token_num, and optionally token_ids
        """
        if self._single_model:
            return self._forward_single_model(speech, speech_lengths, decode=decode)
        else:
            return self._forward_components(speech, speech_lengths, decode=decode)
    
    def _forward_single_model(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
        decode: bool = True,
    ) -> ParaformerModelOutput:
        """Forward pass for single combined model."""
        self._model_component.compile()
        
        # Find speech input name (might be 'speech' or 'speech.1')
        speech_input_name = None
        for name in self.input_names:
            if 'speech' in name.lower() and 'length' not in name.lower():
                speech_input_name = name
                break
        
        if speech_input_name is None:
            # Fall back to first input
            speech_input_name = list(self.input_names.keys())[0]
        
        # Prepare inputs
        speech_np = speech.cpu().numpy() if isinstance(speech, torch.Tensor) else speech
        lengths_np = speech_lengths.cpu().numpy() if isinstance(speech_lengths, torch.Tensor) else speech_lengths
        
        inputs = {
            speech_input_name: speech_np,
            "speech_lengths": lengths_np,
        }
        
        # Run inference
        self._model_component.request.infer(inputs)
        
        # Get outputs
        logits = torch.from_numpy(self._model_component.request.get_output_tensor(0).data.copy())
        token_num = None
        if len(self.output_names) > 1:
            token_num = torch.from_numpy(self._model_component.request.get_output_tensor(1).data.copy())
        
        # Decode if requested
        token_ids = None
        if decode:
            token_ids = self.decode(logits, token_num)
        
        return ParaformerModelOutput(
            logits=logits,
            token_num=token_num,
            token_ids=token_ids,
        )
    
    def _forward_components(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
        decode: bool = True,
    ) -> ParaformerModelOutput:
        """Forward pass for separate component models."""
        # 1. Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        
        # 2. Predictor (if available)
        if self.predictor is not None:
            acoustic_embeds, token_num, alphas, peak_index = self.predictor(
                encoder_out, encoder_out_lens
            )
        else:
            # Without predictor, pass encoder output directly
            acoustic_embeds = encoder_out
            token_num = encoder_out_lens
        
        # 3. Decoder
        logits = self.decoder(encoder_out, encoder_out_lens, acoustic_embeds, token_num)
        
        # Decode if requested
        token_ids = None
        if decode:
            token_ids = self.decode(logits, token_num)
        
        return ParaformerModelOutput(
            logits=logits,
            token_num=token_num,
            token_ids=token_ids,
        )
    
    def decode(
        self,
        logits: torch.Tensor,
        token_num: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode logits to token IDs using greedy decoding.
        
        Args:
            logits: Output logits [batch, seq_len, vocab_size]
            token_num: Optional token numbers for each batch item [batch]
            
        Returns:
            token_ids: Predicted token IDs [batch, seq_len]
        """
        token_ids = torch.argmax(logits, dim=-1)
        
        # Mask out padding if token_num is provided
        if token_num is not None:
            batch_size = token_ids.shape[0]
            max_len = token_ids.shape[1]
            for i in range(batch_size):
                num = int(token_num[i].item()) if torch.is_tensor(token_num[i]) else int(token_num[i])
                if num < max_len:
                    token_ids[i, num:] = 0
        
        return token_ids
    
    def generate(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate token IDs from speech input.
        
        This is an alias for forward() with decode=True for API compatibility.
        
        Args:
            speech: Input speech features [batch, time, features]
            speech_lengths: Valid lengths for each sequence [batch]
            
        Returns:
            token_ids: Predicted token IDs [batch, seq_len]
            token_num: Number of valid tokens per sequence [batch]
        """
        output = self.forward(speech, speech_lengths, decode=True, **kwargs)
        return output.token_ids, output.token_num
    
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
    ):
        """
        Save model to directory.
        
        Args:
            save_directory: Directory to save model files
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self._single_model:
            model_path = save_path / OV_XML_FILE_NAME
            openvino.save_model(self.model, str(model_path))
            logger.info(f"Saved model to {model_path}")
        else:
            if self.encoder:
                encoder_path = save_path / OV_ENCODER_NAME
                openvino.save_model(self.encoder.model, str(encoder_path))
                logger.info(f"Saved encoder to {encoder_path}")
            if self.predictor:
                predictor_path = save_path / OV_PREDICTOR_NAME
                openvino.save_model(self.predictor.model, str(predictor_path))
                logger.info(f"Saved predictor to {predictor_path}")
            if self.decoder:
                decoder_path = save_path / OV_DECODER_NAME
                openvino.save_model(self.decoder.model, str(decoder_path))
                logger.info(f"Saved decoder to {decoder_path}")
        
        # Save config if available
        if self.config is not None:
            self.config.save_pretrained(save_path)


# Alias for backwards compatibility
OVModelForSpeech2Seq = OVParaformerForSpeechSeq2Seq
load_paraformer_model = OVParaformerForSpeechSeq2Seq.from_pretrained


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paraformer OpenVINO Inference")
    parser.add_argument("--model", required=True, help="Path to OpenVINO model directory")
    parser.add_argument("--device", default="CPU", help="Device (CPU/GPU/AUTO)")
    parser.add_argument("--input", help="Path to input speech .npy file")
    parser.add_argument("--lengths", help="Path to lengths .npy file")
    
    args = parser.parse_args()
    
    # Enable logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    # Load model
    print(f"Loading model from {args.model}")
    model = OVParaformerForSpeechSeq2Seq.from_pretrained(args.model, device=args.device)
    print(f"✅ Model loaded on {args.device}")
    print(f"   Components: {model._component_names}")
    print(f"   Input names: {list(model.input_names.keys())}")
    print(f"   Output names: {list(model.output_names.keys())}")
    
    # Load or create input
    if args.input and args.lengths:
        speech = torch.from_numpy(np.load(args.input))
        speech_lengths = torch.from_numpy(np.load(args.lengths))
        print(f"Loaded input: speech {speech.shape}, lengths {speech_lengths.shape}")
    else:
        # Create dummy input
        speech = torch.randn(1, 100, 560)
        speech_lengths = torch.tensor([100], dtype=torch.int32)
        print("Using dummy input: speech [1, 100, 560]")
    
    # Run inference
    print("\nRunning inference...")
    output = model(speech, speech_lengths)
    
    print(f"\n✅ Inference completed!")
    print(f"   Logits shape: {output.logits.shape}")
    print(f"   Token numbers: {output.token_num}")
    if output.token_ids is not None:
        num = int(output.token_num[0]) if output.token_num is not None else 10
        print(f"   Token IDs (first {num}): {output.token_ids[0, :num].tolist()}")
