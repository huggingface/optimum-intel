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
Standalone Paraformer Export Script for OpenVINO

This module provides a standalone export function for Paraformer ASR models
to OpenVINO format, independent of the main optimum-intel export pipeline.

Usage:
    python -m optimum.exporters.openvino.export_paraformer \
        --model /path/to/paraformer/model \
        --output /path/to/output \
        --int8  # optional, for INT8 weight compression

Or programmatically:
    from optimum.exporters.openvino.export_paraformer import export_paraformer
    export_paraformer(model_path, output_path, compress_int8=True)
"""

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def export_paraformer(
    model_name_or_path: Union[str, Path],
    output: Union[str, Path],
    device: str = "cpu",
    compress_int8: bool = False,
    compress_fp16: bool = True,
) -> None:
    """
    Export a Paraformer ASR model to OpenVINO format.
    
    This is a standalone export function that doesn't require modifications
    to the main optimum-intel export pipeline.
    
    Args:
        model_name_or_path: Path to the Paraformer model directory
        output: Output directory for the exported model
        device: Device to use for export (default: "cpu")
        compress_int8: Apply INT8 symmetric weight compression (default: False)
        compress_fp16: Store FP32 constants as FP16 (default: True, recommended for GPU)
    
    Returns:
        None
        
    Example:
        >>> from optimum.exporters.openvino.export_paraformer import export_paraformer
        >>> export_paraformer(
        ...     "/path/to/paraformer/model",
        ...     "/path/to/output",
        ...     compress_int8=True
        ... )
    """
    import openvino as ov
    
    # Import paraformer modeling (lazy import to avoid dependency issues)
    from optimum.exporters.openvino.modeling_paraformer import build_model, export
    
    model_path = str(model_name_or_path)
    output_path = Path(output)
    
    logger.info(f"Exporting Paraformer model from {model_path}")
    logger.info(f"Output directory: {output_path}")
    
    # Build the model
    model, kwargs = build_model(model=model_path, device=device)
    
    # Export to TorchScript
    model_dir, model_jit_scripts = export(model, kwargs, type="torchscript", quantize=False, device=device)
    
    # Convert to OpenVINO
    ovm = ov.convert_model(model_jit_scripts, input=[([-1, -1, -1], torch.float32), ([-1], torch.int32)])
    
    # Create output directory
    target_dir = output_path / "ov_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    output_model_path = target_dir / "openvino_model.xml"
    
    # Apply INT8 weight compression if requested
    if compress_int8:
        try:
            from nncf import compress_weights, CompressWeightsMode
            logger.info("Applying INT8 weight compression (symmetric)...")
            # INT8_SYM: no zero-point bias ops → significantly faster on GPU
            ovm = compress_weights(ovm, mode=CompressWeightsMode.INT8_SYM)
            logger.info("Weight compression complete.")
        except ImportError:
            logger.warning("NNCF not available. Skipping INT8 compression. Install with: pip install nncf")
    
    # Save the model
    if compress_fp16:
        # compress_to_fp16=True: stores remaining FP32 constants as FP16
        # → avoids a second FP32→FP16 conversion pass on GPU at runtime
        ov.save_model(ovm, str(output_model_path), compress_to_fp16=True)
        logger.info(f"Model saved with FP16 compression to {output_model_path}")
    else:
        ov.serialize(ovm, str(output_model_path))
        logger.info(f"Model saved to {output_model_path}")
    
    # Copy model parameter files
    PARAFORMER_PARAM_FILES = ['am.mvn', 'config.yaml', 'configuration.json', 'seg_dict', 'tokens.json']
    
    for file_name in PARAFORMER_PARAM_FILES:
        source_file = os.path.join(model_dir, file_name)
        target_file = target_dir / file_name
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            logger.debug(f"Copied {file_name}")
    
    logger.info(f"Export complete. Model saved to {target_dir}")
    
    return model, kwargs


def main():
    """Command-line interface for Paraformer export."""
    parser = argparse.ArgumentParser(
        description="Export Paraformer ASR model to OpenVINO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python -m optimum.exporters.openvino.export_paraformer \\
      --model /path/to/paraformer \\
      --output /path/to/output

  # Export with INT8 compression
  python -m optimum.exporters.openvino.export_paraformer \\
      --model /path/to/paraformer \\
      --output /path/to/output \\
      --int8
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the Paraformer model directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for the exported model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (default: cpu)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Apply INT8 symmetric weight compression"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 compression for constants"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run export
    export_paraformer(
        model_name_or_path=args.model,
        output=args.output,
        device=args.device,
        compress_int8=args.int8,
        compress_fp16=not args.no_fp16,
    )


if __name__ == "__main__":
    main()
