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
Paraformer Plugin for OpenVINO Export

This module provides automatic Paraformer model support for optimum-cli export
by hooking into the main_export function.

Usage:
    optimum-cli export openvino --model funasr/paraformer-zh --task automatic-speech-recognition output_dir
    
    # With INT8 quantization:
    optimum-cli export openvino --model funasr/paraformer-zh --task automatic-speech-recognition --weight-format int8 output_dir
"""

import logging
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel

from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx.config import OnnxConfig

logger = logging.getLogger(__name__)


class ParaformerConfig(PretrainedConfig):
    """
    Configuration class for Paraformer ASR models.
    
    This provides a transformers-compatible configuration for FunASR Paraformer models.
    """
    model_type = "paraformer"
    
    def __init__(
        self,
        vocab_size: int = 8404,
        encoder_dim: int = 512,
        attention_heads: int = 4,
        encoder_layers: int = 50,
        decoder_layers: int = 16,
        frontend_conf: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.attention_heads = attention_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.frontend_conf = frontend_conf or {}
    
    @classmethod
    def from_funasr_config(cls, config_path: Union[str, Path]) -> "ParaformerConfig":
        """Load configuration from FunASR config.yaml file."""
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.load(config_path)
            
            return cls(
                vocab_size=config.get("vocab_size", 8404),
                encoder_dim=config.get("encoder_conf", {}).get("output_size", 512),
                attention_heads=config.get("encoder_conf", {}).get("attention_heads", 4),
                encoder_layers=config.get("encoder_conf", {}).get("num_blocks", 50),
                decoder_layers=config.get("decoder_conf", {}).get("num_blocks", 16),
                frontend_conf=dict(config.get("frontend_conf", {})),
            )
        except Exception as e:
            logger.warning(f"Could not load FunASR config: {e}, using defaults")
            return cls()


class ParaformerForASR(PreTrainedModel):
    """
    Transformers-compatible wrapper for Paraformer ASR models.
    
    This class wraps FunASR Paraformer models to make them compatible with
    the optimum-intel export pipeline.
    """
    config_class = ParaformerConfig
    base_model_prefix = "paraformer"
    main_input_name = "speech"
    
    def __init__(self, config: ParaformerConfig, funasr_model=None):
        super().__init__(config)
        self.funasr_model = funasr_model
        self._jit_model = None
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        *model_args,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> "ParaformerForASR":
        """
        Load a Paraformer model from a FunASR model directory or HuggingFace Hub.
        """
        from huggingface_hub import snapshot_download
        
        model_path = Path(model_name_or_path)
        
        # Download from HuggingFace Hub if not a local path
        if not model_path.exists():
            logger.info(f"Downloading Paraformer model from HuggingFace Hub: {model_name_or_path}")
            model_path = Path(snapshot_download(
                repo_id=str(model_name_or_path),
                cache_dir=cache_dir,
                token=kwargs.get("token"),
                revision=kwargs.get("revision", "main"),
            ))
        
        # Load config
        config_yaml_path = model_path / "config.yaml"
        if config_yaml_path.exists():
            config = ParaformerConfig.from_funasr_config(config_yaml_path)
        else:
            config = ParaformerConfig()
        
        # Load the FunASR model
        from optimum.exporters.openvino.modeling_paraformer import build_model
        
        device = kwargs.get("device", "cpu")
        funasr_model, model_kwargs = build_model(model=str(model_path), device=device)
        
        instance = cls(config, funasr_model=funasr_model)
        instance._model_path = model_path
        instance._model_kwargs = model_kwargs
        
        return instance
    
    def get_jit_model(self) -> torch.jit.ScriptModule:
        """Get or create the TorchScript model for export."""
        if self._jit_model is None:
            from optimum.exporters.openvino.modeling_paraformer import export
            
            _, self._jit_model = export(
                self.funasr_model,
                self._model_kwargs,
                type="torchscript",
                quantize=False,
                device=str(self._model_kwargs.get("device", "cpu"))
            )
        return self._jit_model
    
    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        """Forward pass through the model."""
        if self.funasr_model is not None:
            return self.funasr_model(speech, speech_lengths)
        raise ValueError("FunASR model not loaded")


class ParaformerOnnxConfig(OnnxConfig):
    """
    ONNX/OpenVINO export configuration for Paraformer models.
    """
    NORMALIZED_CONFIG_CLASS = ParaformerConfig
    DEFAULT_ONNX_OPSET = 14
    
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "speech": {0: "batch_size", 1: "sequence_length", 2: "feature_dim"},
            "speech_lengths": {0: "batch_size"},
        }
    
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "logits": {0: "batch_size", 1: "sequence_length"},
        }
    
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> Dict[str, Any]:
        """Generate dummy inputs for export."""
        batch_size = 1
        sequence_length = 1000  # ~10 seconds of audio at 16kHz with 10ms frame shift
        feature_dim = 560  # LFR features (80 mel * 7 frames)
        
        return {
            "speech": torch.randn(batch_size, sequence_length, feature_dim),
            "speech_lengths": torch.tensor([sequence_length], dtype=torch.int32),
        }


def _is_paraformer_model(model_name_or_path: str, cache_dir: str = None, **kwargs) -> bool:
    """Check if the model is a Paraformer ASR model."""
    import json
    from pathlib import Path
    
    try:
        from huggingface_hub import HfFileSystem
        
        model_path = Path(model_name_or_path)
        
        # Check if it's a local path
        if model_path.exists():
            if (model_path / "config.yaml").exists() and (model_path / "tokens.json").exists():
                return True
            if (model_path / "am.mvn").exists():
                return True
            return False
        
        # Check HuggingFace Hub
        fs = HfFileSystem(token=kwargs.get("token"))
        try:
            repo_files = fs.ls(f"{model_name_or_path}", detail=False)
            repo_files = [f.split("/")[-1] for f in repo_files]
            
            if "config.yaml" in repo_files and "tokens.json" in repo_files:
                return True
            if "am.mvn" in repo_files:
                return True
        except Exception:
            pass
        
        return False
    except Exception:
        return False


def export_paraformer_to_openvino(
    model_name_or_path: str,
    output: Union[str, Path],
    weight_format: str = "fp16",
    cache_dir: str = None,
    token: Optional[str] = None,
    **kwargs
) -> None:
    """
    Export a Paraformer model to OpenVINO format.
    
    This function handles the complete export pipeline for FunASR Paraformer models.
    """
    import openvino as ov
    import shutil
    from optimum.exporters.openvino.modeling_paraformer import build_model, export
    from huggingface_hub import snapshot_download
    
    model_path = Path(model_name_or_path)
    output_path = Path(output)
    
    # Download from HuggingFace Hub if not a local path
    if not model_path.exists():
        logger.info(f"Downloading Paraformer model from HuggingFace Hub: {model_name_or_path}")
        model_path = Path(snapshot_download(
            repo_id=str(model_name_or_path),
            cache_dir=cache_dir,
            token=token,
        ))
    
    logger.info(f"Loading Paraformer model from {model_path}")
    
    # Build the FunASR model
    device = kwargs.get("device", "cpu")
    model, model_kwargs = build_model(model=str(model_path), device=device)
    
    # Export to TorchScript
    logger.info("Converting to TorchScript...")
    _, jit_model = export(model, model_kwargs, type="torchscript", quantize=False, device=device)
    
    # Convert to OpenVINO
    logger.info("Converting to OpenVINO format...")
    ovm = ov.convert_model(jit_model, input=[([-1, -1, -1], torch.float32), ([-1], torch.int32)])
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    output_model_path = output_path / "openvino_model.xml"
    
    # Apply compression based on weight_format
    compress_to_fp16 = weight_format.lower() in ["fp16", "int8"]
    
    if weight_format.lower() == "int8":
        logger.info("Applying INT8 weight compression...")
        try:
            import nncf
            ovm = nncf.compress_weights(ovm, mode=nncf.CompressWeightsMode.INT8_SYM)
        except ImportError:
            logger.warning("NNCF not available, saving without INT8 compression")
    
    # Save the model
    logger.info(f"Saving model to {output_model_path}")
    ov.save_model(ovm, str(output_model_path), compress_to_fp16=compress_to_fp16)
    
    # Copy auxiliary files
    for aux_file in ["tokens.json", "config.yaml", "configuration.json", "am.mvn", "seg_dict"]:
        src = model_path / aux_file
        if src.exists():
            shutil.copy(src, output_path / aux_file)
    
    logger.info(f"Paraformer model exported successfully to {output_path}")


def _load_paraformer_model(
    model_name_or_path: str,
    subfolder: str = "",
    revision: str = "main",
    cache_dir: str = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Load a Paraformer model for export."""
    return ParaformerForASR.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        token=token,
        revision=revision,
        **kwargs,
    )


def register_paraformer_with_tasks_manager():
    """
    Register Paraformer support with TasksManager.
    
    This function adds the necessary mappings for Paraformer to work
    with the standard optimum-intel export pipeline.
    """
    # Register paraformer library with supported model types
    if "paraformer" not in TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES:
        TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES["paraformer"] = {
            "paraformer": {
                "automatic-speech-recognition": ("ParaformerForASR",),
            }
        }
    
    # Register model loader for paraformer library
    if "paraformer" not in TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP:
        TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP["paraformer"] = {
            "automatic-speech-recognition": _load_paraformer_model,
        }
    
    logger.debug("Registered Paraformer support with TasksManager")


def patch_main_quantize():
    """
    Patch the _main_quantize function to skip Paraformer models
    (since quantization is already handled in main_export).
    """
    try:
        from optimum.exporters.openvino import __main__ as ov_main
        
        original_main_quantize = ov_main._main_quantize
        
        @wraps(original_main_quantize)
        def patched_main_quantize(
            model_name_or_path: str,
            **kwargs
        ):
            # Debug logging
            logger.info(f"patched_main_quantize called for model: {model_name_or_path}")
            
            # Check if this is a Paraformer model
            cache_dir = kwargs.get("cache_dir")
            is_paraformer = _is_paraformer_model(model_name_or_path, cache_dir=cache_dir)
            logger.info(f"Is Paraformer model: {is_paraformer}")
            
            if is_paraformer:
                logger.info("Skipping _main_quantize for Paraformer (already quantized in main_export)")
                # For Paraformer, quantization is already done in main_export, so just return
                return
            
            # Not a Paraformer model, use original quantization
            return original_main_quantize(model_name_or_path, **kwargs)
        
        # Apply the patch
        ov_main._main_quantize = patched_main_quantize
        logger.debug("Patched _main_quantize to skip Paraformer models")
        
    except Exception as e:
        logger.warning(f"Could not patch _main_quantize for Paraformer support: {e}")


def patch_main_export():
    """
    Patch the main_export function to handle Paraformer models automatically.
    
    This allows `optimum-cli export openvino --model funasr/paraformer-zh ...` to work
    without modifying __main__.py directly.
    """
    try:
        from optimum.exporters.openvino import __main__ as ov_main
        
        original_main_export = ov_main.main_export
        
        @wraps(original_main_export)
        def patched_main_export(
            model_name_or_path: str,
            output: Union[str, Path],
            task: str = "auto",
            **kwargs
        ):
            # Check if this is a Paraformer model
            if _is_paraformer_model(model_name_or_path, cache_dir=kwargs.get("cache_dir")):
                logger.info("Detected Paraformer model (FunASR). Using specialized export.")
                
                # Determine weight format from kwargs
                weight_format = kwargs.get("weight_format", "fp16")
                if weight_format is None:
                    weight_format = "fp16"
                
                # Also check ov_config for quantization settings
                ov_config = kwargs.get("ov_config")
                if ov_config is not None:
                    quant_config = getattr(ov_config, "quantization_config", None)
                    if quant_config is not None:
                        if hasattr(quant_config, 'bits') and quant_config.bits == 8:
                            weight_format = "int8"
                        elif hasattr(quant_config, 'weight_dtype') and 'int8' in str(quant_config.weight_dtype).lower():
                            weight_format = "int8"
                
                export_paraformer_to_openvino(
                    model_name_or_path=model_name_or_path,
                    output=output,
                    weight_format=weight_format,
                    cache_dir=kwargs.get("cache_dir"),
                    token=kwargs.get("token"),
                    device=kwargs.get("device", "cpu"),
                )
                return
            
            # Not a Paraformer model, use original export
            return original_main_export(model_name_or_path, output, task, **kwargs)
        
        # Apply the patch
        ov_main.main_export = patched_main_export
        logger.debug("Patched main_export to support Paraformer models")
        
    except Exception as e:
        logger.warning(f"Could not patch main_export for Paraformer support: {e}")


# Auto-register when this module is imported
register_paraformer_with_tasks_manager()
patch_main_export()
patch_main_quantize()

