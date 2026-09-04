"""
Test IR (Intermediate Representation) stability across transformers versions.

This test compares newly exported OpenVINO IRs against reference IRs to detect
regressions when upgrading transformers or other dependencies.
"""

import json

# Import test utilities to get model mappings
import sys
from pathlib import Path
from pathlib import Path as PathlibPath
from typing import Dict, List, Optional

import pytest
from huggingface_hub import snapshot_download
from openvino import Core, Model

from optimum.intel import (
    OVDiffusionPipeline,
    OVFlux2KleinPipeline,
    OVFluxFillPipeline,
    OVFluxPipeline,
    OVLatentConsistencyModelPipeline,
    OVLTX2Pipeline,
    OVLTXPipeline,
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForMultimodalLM,
    OVModelForPix2Struct,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
    OVModelForTextToSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVModelForVision2Seq,
    OVModelForVisualCausalLM,
    OVModelForZeroShotImageClassification,
    OVModelOpenCLIPForZeroShotImageClassification,
    OVSamModel,
    OVSanaPipeline,
    OVStableDiffusion3Pipeline,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLImg2ImgPipeline,
    OVStableDiffusionXLPipeline,
)


# Add tests/openvino to path to import utils_tests
sys.path.insert(0, str(PathlibPath(__file__).parent))
from utils_tests import ARCH_TO_MODEL_CLASS, HUB_MODEL_NAMES, REMOTE_CODE_MODELS


# Map class name strings to actual class objects
CLASS_NAME_TO_CLASS = {
    "OVDiffusionPipeline": OVDiffusionPipeline,
    "OVFlux2KleinPipeline": OVFlux2KleinPipeline,
    "OVFluxFillPipeline": OVFluxFillPipeline,
    "OVFluxPipeline": OVFluxPipeline,
    "OVLatentConsistencyModelPipeline": OVLatentConsistencyModelPipeline,
    "OVLTX2Pipeline": OVLTX2Pipeline,
    "OVLTXPipeline": OVLTXPipeline,
    "OVModelForAudioClassification": OVModelForAudioClassification,
    "OVModelForCausalLM": OVModelForCausalLM,
    "OVModelForFeatureExtraction": OVModelForFeatureExtraction,
    "OVModelForImageClassification": OVModelForImageClassification,
    "OVModelForMaskedLM": OVModelForMaskedLM,
    "OVModelForMultimodalLM": OVModelForMultimodalLM,
    "OVModelForPix2Struct": OVModelForPix2Struct,
    "OVModelForQuestionAnswering": OVModelForQuestionAnswering,
    "OVModelForSeq2SeqLM": OVModelForSeq2SeqLM,
    "OVModelForSequenceClassification": OVModelForSequenceClassification,
    "OVModelForSpeechSeq2Seq": OVModelForSpeechSeq2Seq,
    "OVModelForTextToSpeechSeq2Seq": OVModelForTextToSpeechSeq2Seq,
    "OVModelForTokenClassification": OVModelForTokenClassification,
    "OVModelForVision2Seq": OVModelForVision2Seq,
    "OVModelForVisualCausalLM": OVModelForVisualCausalLM,
    "OVModelForZeroShotImageClassification": OVModelForZeroShotImageClassification,
    "OVModelOpenCLIPForZeroShotImageClassification": OVModelOpenCLIPForZeroShotImageClassification,
    "OVSamModel": OVSamModel,
    "OVSanaPipeline": OVSanaPipeline,
    "OVStableDiffusion3Pipeline": OVStableDiffusion3Pipeline,
    "OVStableDiffusionPipeline": OVStableDiffusionPipeline,
    "OVStableDiffusionXLImg2ImgPipeline": OVStableDiffusionXLImg2ImgPipeline,
    "OVStableDiffusionXLPipeline": OVStableDiffusionXLPipeline,
}


# Additional architecture-to-class mappings from test_export.py, test_decoder.py, and test_quantization.py
# Merged with ARCH_TO_MODEL_CLASS for more complete coverage
ADDITIONAL_ARCH_MAPPINGS = {
    # From test_export.py
    "albert": "OVModelForSequenceClassification",
    "bert": "OVModelForMaskedLM",
    "blenderbot": "OVModelForFeatureExtraction",
    "distilbert": "OVModelForQuestionAnswering",
    "hunyuan_v1_dense": "OVModelForCausalLM",
    "kokoro": "OVModelForTextToSpeechSeq2Seq",
    "mamba": "OVModelForCausalLM",
    "qwen3_next": "OVModelForCausalLM",
    "roberta": "OVModelForTokenClassification",
    "sam": "OVSamModel",
    "smollm3": "OVModelForCausalLM",
    "speecht5": "OVModelForTextToSpeechSeq2Seq",
    "t5": "OVModelForSeq2SeqLM",
    "vit": "OVModelForImageClassification",
    "wav2vec2": "OVModelForAudioClassification",
    "zamba2": "OVModelForCausalLM",
    # From test_decoder.py - CausalLM models
    "arcee": "OVModelForCausalLM",
    "biogpt": "OVModelForCausalLM",
    "bloom": "OVModelForCausalLM",
    "codegen": "OVModelForCausalLM",
    "cohere": "OVModelForCausalLM",
    "falcon": "OVModelForCausalLM",
    "falcon-40b": "OVModelForCausalLM",
    "gemma": "OVModelForCausalLM",
    "gemma2": "OVModelForCausalLM",
    "gemma3_text": "OVModelForCausalLM",
    "glm": "OVModelForCausalLM",
    "glm4": "OVModelForCausalLM",
    "gpt_bigcode": "OVModelForCausalLM",
    "gpt_neo": "OVModelForCausalLM",
    "gpt_neox": "OVModelForCausalLM",
    "gpt_neox_japanese": "OVModelForCausalLM",
    "gpt_oss": "OVModelForCausalLM",
    "gpt_oss_mxfp4": "OVModelForCausalLM",
    "gptj": "OVModelForCausalLM",
    "granite": "OVModelForCausalLM",
    "granitemoe": "OVModelForCausalLM",
    "mistral-nemo": "OVModelForCausalLM",
    "mixtral": "OVModelForCausalLM",
    "mpt": "OVModelForCausalLM",
    "opt": "OVModelForCausalLM",
    "pegasus": "OVModelForCausalLM",
    "persimmon": "OVModelForCausalLM",
    "phi": "OVModelForCausalLM",
    "phi3": "OVModelForCausalLM",
    "qwen2_moe": "OVModelForCausalLM",
    "stablelm": "OVModelForCausalLM",
    "starcoder2": "OVModelForCausalLM",
    "xglm": "OVModelForCausalLM",
    # From test_seq2seq.py - Seq2SeqLM models
    "bigbird_pegasus": "OVModelForSeq2SeqLM",
    "blenderbot-small": "OVModelForSeq2SeqLM",
    "longt5": "OVModelForSeq2SeqLM",
    "m2m_100": "OVModelForSeq2SeqLM",
    "marian": "OVModelForSeq2SeqLM",
    "mbart": "OVModelForSeq2SeqLM",
    "mt5": "OVModelForSeq2SeqLM",
    # Vision models - ImageClassification
    "audio-spectrogram-transformer": "OVModelForAudioClassification",
    "beit": "OVModelForImageClassification",
    "convnext": "OVModelForImageClassification",
    "convnextv2": "OVModelForImageClassification",
    "data2vec-vision": "OVModelForImageClassification",
    "deit": "OVModelForImageClassification",
    "levit": "OVModelForImageClassification",
    "mobilenet_v1": "OVModelForImageClassification",
    "mobilenet_v2": "OVModelForImageClassification",
    "mobilevit": "OVModelForImageClassification",
    "poolformer": "OVModelForImageClassification",
    "resnet": "OVModelForImageClassification",
    "swin": "OVModelForImageClassification",
    # Vision models - Feature Extraction / Object Detection
    "detr": "OVModelForFeatureExtraction",
    "donut-swin": "OVModelForFeatureExtraction",
    "open-clip": "OVModelOpenCLIPForZeroShotImageClassification",
    "segformer": "OVModelForFeatureExtraction",
    # Text models - Masked LM / Feature Extraction / Embeddings
    "bge": "OVModelForFeatureExtraction",
    "camembert": "OVModelForMaskedLM",
    "convbert": "OVModelForSequenceClassification",
    "data2vec-text": "OVModelForFeatureExtraction",
    "deberta": "OVModelForMaskedLM",
    "deberta-v2": "OVModelForMaskedLM",
    "esm": "OVModelForMaskedLM",
    "flaubert": "OVModelForMaskedLM",
    "ibert": "OVModelForMaskedLM",
    "mobilebert": "OVModelForMaskedLM",
    "mpnet": "OVModelForFeatureExtraction",
    "nystromformer": "OVModelForMaskedLM",
    "rembert": "OVModelForMaskedLM",
    "roformer": "OVModelForMaskedLM",
    "sentence-transformers-bert": "OVModelForFeatureExtraction",
    "squeezebert": "OVModelForMaskedLM",
    "st-bert": "OVModelForFeatureExtraction",
    "st-mpnet": "OVModelForFeatureExtraction",
    "xlm": "OVModelForMaskedLM",
    "xlm-roberta": "OVModelForMaskedLM",
    # Audio models
    "data2vec-audio": "OVModelForAudioClassification",
    "hubert": "OVModelForAudioClassification",
    "sew": "OVModelForAudioClassification",
    "sew-d": "OVModelForAudioClassification",
    "speech_to_text": "OVModelForSpeechSeq2Seq",
    "unispeech": "OVModelForAudioClassification",
    "unispeech-sat": "OVModelForAudioClassification",
    "wav2vec2-conformer": "OVModelForAudioClassification",
    "wav2vec2-hf": "OVModelForAudioClassification",
    "wavlm": "OVModelForAudioClassification",
    # Causal LM - Additional models
    "bitnet": "OVModelForCausalLM",
    "cohere2": "OVModelForCausalLM",
    "dbrx": "OVModelForCausalLM",
    "falcon_mamba": "OVModelForCausalLM",
    "gemma3": "OVModelForVisualCausalLM",
    "gemma3n_text": "OVModelForCausalLM",
    "granitemoehybrid": "OVModelForCausalLM",
    "olmo": "OVModelForCausalLM",
    "olmo2": "OVModelForCausalLM",
    "opt125m": "OVModelForCausalLM",
    "phimoe": "OVModelForCausalLM",
    "qwen3_5": "OVModelForVisualCausalLM",
    # Vision-Language / Multimodal models
    "donut": "OVModelForVision2Seq",
    "gemma3n": "OVModelForVisualCausalLM",
    "gemma4": "OVModelForVisualCausalLM",
    "got_ocr2": "OVModelForVisualCausalLM",
    "idefics3": "OVModelForVisualCausalLM",
    "internvl_chat": "OVModelForVisualCausalLM",
    "llava-qwen2": "OVModelForVisualCausalLM",
    "llava_next": "OVModelForVisualCausalLM",
    "llava_next_mistral": "OVModelForVisualCausalLM",
    "llava_next_video": "OVModelForVisualCausalLM",
    "maira2": "OVModelForVisualCausalLM",
    "minicpmo": "OVModelForVisualCausalLM",
    "minicpmv": "OVModelForVisualCausalLM",
    "phi3_v": "OVModelForVisualCausalLM",
    "phi4mm": "OVModelForVisualCausalLM",
    "pix2struct": "OVModelForPix2Struct",
    "qwen2_5_vl": "OVModelForVisualCausalLM",
    "qwen2_vl": "OVModelForVisualCausalLM",
    "qwen3_vl": "OVModelForVisualCausalLM",
    "qwen3_vl_embedding": "OVModelForFeatureExtraction",
    "smolvlm": "OVModelForVisualCausalLM",
    "trocr": "OVModelForVision2Seq",
    "vision-encoder-decoder": "OVModelForVision2Seq",
    # Diffusion pipelines
    "flux": "OVFluxPipeline",
    "flux-fill": "OVFluxFillPipeline",
    "flux.2-klein": "OVFlux2KleinPipeline",
    "latent-consistency": "OVLatentConsistencyModelPipeline",
    "ltx-video": "OVLTXPipeline",
    "ltx2": "OVLTX2Pipeline",
    "sana": "OVSanaPipeline",
    "sana-sprint": "OVSanaPipeline",
    "stable-diffusion-3": "OVStableDiffusion3Pipeline",
    "stable-diffusion-xl": "OVStableDiffusionXLPipeline",
    "stable-diffusion-xl-refiner": "OVStableDiffusionXLImg2ImgPipeline",
    "stable-diffusion-with-custom-variant": "OVStableDiffusionPipeline",
    "stable-diffusion-with-safety-checker": "OVStableDiffusionPipeline",
    "stable-diffusion-with-textual-inversion": "OVStableDiffusionPipeline",
}


# Extra `from_pretrained` arguments needed at export time by some models, keyed by architecture.
# The reference IRs on the `ov` branch are generated with the exact same arguments, so any change
# here must be mirrored by regenerating the affected references.
_EXTRA_EXPORT_KWARGS_BY_ARCH = {
    # SpeechT5 is a text-to-speech model whose export needs an explicit vocoder. For text-to-audio
    # models the remaining `from_pretrained` kwargs are forwarded to `main_export` as `model_kwargs`.
    "speecht5": {"vocoder": "fxmarty/speecht5-hifigan-tiny"},
    # This fixture only ships weights under a non-default variant.
    "stable-diffusion-with-custom-variant": {"variant": "custom"},
}

EXPORT_KWARGS = {
    HUB_MODEL_NAMES[arch]: kwargs for arch, kwargs in _EXTRA_EXPORT_KWARGS_BY_ARCH.items() if arch in HUB_MODEL_NAMES
}


# Generate test parameters: (model_id, model_class) for all models that don't need trust_remote_code
def _generate_test_params():
    """Generate test parameters from utils_tests mappings and test_export mappings."""
    params = []
    for arch, model_id in HUB_MODEL_NAMES.items():
        # Skip models that need trust_remote_code
        if arch in REMOTE_CODE_MODELS:
            continue

        if arch in {  # multimodal models that also need trust_remote_code, but are not in REMOTE_CODE_MODELS
            "internvl_chat",
            "llava-qwen2",
            "maira2",
            "minicpmo",
            "minicpmv",
            "phi3_v",
            "phi4mm",
        }:
            continue

        if arch in {  # seq2seq models
            "bigbird_pegasus",
            "blenderbot-small",
            "longt5",
            "m2m_100",
            "mbart",
            "t5",
        }:
            continue

        if arch in {  # 4.57.6 models
            "data2vec-text",
            "got_ocr2",
            "flaubert",
            "idefics3",
            "zamba2",
            "qwen3_next",
            "xlm",
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_embedding",
            "llava_next_video",
            "marian",
            "mt5",
            "smolvlm",
        }:
            continue

        if arch in {  # max transformers 5.0 required
            "gemma",
            "gemma2",
            "gemma3_text",
            "gemma3n_text",
            "glm",
        }:
            continue

        if arch == "qwen3_5":  # max transformers 5.2.* required
            continue

        if arch in {  # max transformers 5.3.0 required
            "falcon_mamba",
            "granitemoehybrid",
            "mamba",
        }:
            continue

        if arch in {  # max transformers 5.4.0 required
            "lfm2",
            "lfm2_moe",
        }:
            continue

        # Try to get model class from ARCH_TO_MODEL_CLASS first, then ADDITIONAL_ARCH_MAPPINGS
        class_name = None
        if arch in ARCH_TO_MODEL_CLASS:
            class_name = ARCH_TO_MODEL_CLASS[arch]
        elif arch in ADDITIONAL_ARCH_MAPPINGS:
            class_name = ADDITIONAL_ARCH_MAPPINGS[arch]

        # Skip if no mapping found or class not available
        if class_name is None or class_name not in CLASS_NAME_TO_CLASS:
            continue

        model_class = CLASS_NAME_TO_CLASS[class_name]
        params.append((model_id, model_class))
    return params


TEST_MODELS = _generate_test_params()


# OpenVINO model comparison functions
# Adapted from: https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/tests/utils/helpers.py
def _compare_models(model_one: Model, model_two: Model, compare_names: bool = True) -> tuple[bool, str]:
    """Function to compare OpenVINO model (ops names, types and shapes).

    Note that the functions uses get_ordered_ops, so the topological order of ops should be also preserved.

    :param model_one: The first model to compare.
    :param model_two: The second model to compare.
    :param compare_names: Flag to control friendly names checking. Default: True
    :return: tuple which consists of bool value (True if models are equal, otherwise False)
             and string with the message to reuse for debug/testing purposes. The string value
             is empty when models are equal.
    """
    result = True
    msg = ""

    # Check friendly names of models
    if compare_names and model_one.get_friendly_name() != model_two.get_friendly_name():
        result = False
        msg += "Friendly names of models are not equal "
        msg += f"model_one: {model_one.get_friendly_name()}, model_two: {model_two.get_friendly_name()}.\n"

    model_one_ops = model_one.get_ordered_ops()
    model_two_ops = model_two.get_ordered_ops()

    # Check overall number of operators
    if len(model_one_ops) != len(model_two_ops):
        result = False
        msg += "Not equal number of ops "
        msg += f"model_one: {len(model_one_ops)}, model_two: {len(model_two_ops)}.\n"

    # Only compare ops that exist in both models
    for i in range(min(len(model_one_ops), len(model_two_ops))):
        op_one_name = model_one_ops[i].get_friendly_name()  # op from model_one
        op_two_name = model_two_ops[i].get_friendly_name()  # op from model_two
        # Check friendly names
        if compare_names and op_one_name != op_two_name and model_one_ops[i].get_type_name() != "Constant":
            result = False
            msg += "Not equal op names "
            msg += f"model_one: {op_one_name}, "
            msg += f"model_two: {op_two_name}.\n"
        # Check output sizes
        if model_one_ops[i].get_output_size() != model_two_ops[i].get_output_size():
            result = False
            msg += f"Not equal output sizes of {op_one_name} and {op_two_name}.\n"
        # Only compare outputs that exist in both ops
        for idx in range(min(model_one_ops[i].get_output_size(), model_two_ops[i].get_output_size())):
            # Check partial shapes of outputs
            op_one_partial_shape = model_one_ops[i].get_output_partial_shape(idx)
            op_two_partial_shape = model_two_ops[i].get_output_partial_shape(idx)
            if op_one_partial_shape != op_two_partial_shape:
                result = False
                msg += f"Not equal op partial shapes of {op_one_name} and {op_two_name} on {idx} index "
                msg += f"model_one: {op_one_partial_shape}, "
                msg += f"model_two: {op_two_partial_shape}.\n"
            # Check element types of outputs
            op_one_element_type = model_one_ops[i].get_output_element_type(idx)
            op_two_element_type = model_two_ops[i].get_output_element_type(idx)
            if op_one_element_type != op_two_element_type:
                result = False
                msg += f"Not equal output element types of {op_one_name} and {op_two_name} on {idx} index "
                msg += f"model_one: {op_one_element_type}, "
                msg += f"model_two: {op_two_element_type}.\n"

    return result, msg


def compare_models(model_one: Model, model_two: Model, compare_names: bool = True):
    """Function to compare OpenVINO model (ops names, types and shapes).

    :param model_one: The first model to compare.
    :param model_two: The second model to compare.
    :param compare_names: Flag to control friendly names checking. Default: True
    :return: True if models are equal, otherwise raise an error with a report of mismatches.
    """
    result, msg = _compare_models(model_one, model_two, compare_names=compare_names)

    if not result:
        raise RuntimeError(msg)

    return result


def load_reference_metadata(ref_dir: Path) -> Optional[Dict]:
    """Load metadata about reference IR generation."""
    metadata_path = ref_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def find_ir_files(directory: Path) -> List[Path]:
    """
    Find all OpenVINO IR XML files in directory and subdirectories.
    Handles both standard naming (openvino_model.xml) and component naming
    (openvino_language_model.xml, openvino_vision_embeddings_model.xml, etc.)
    Returns list of paths relative to the directory.
    """
    ir_files = []
    # Match both openvino_model.xml and openvino_*_model.xml patterns
    for xml_file in directory.rglob("openvino*.xml"):
        if xml_file.name.startswith("openvino") and xml_file.name.endswith("_model.xml"):
            # Get relative path from directory
            rel_path = xml_file.relative_to(directory)
            ir_files.append(rel_path)
    return sorted(ir_files)


class TestIRStability:
    """Test suite for IR stability across transformers versions."""

    @pytest.mark.parametrize(
        "model_id,model_class",
        TEST_MODELS,
    )
    def test_ir_stability(self, model_id: str, model_class, tmp_path: Path):
        """Test that exported IR matches reference IR."""

        # Download reference IRs from HuggingFace (ov branch)
        try:
            ref_ir_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision="ov",
                    repo_type="model",
                )
            )
        except Exception as e:
            pytest.skip(f"Failed to download reference IRs from {model_id} (ov branch): {e}")

        if not ref_ir_dir.exists():
            pytest.skip(f"Reference IR directory not found for {model_id}")

        # Load reference metadata
        ref_metadata = load_reference_metadata(ref_ir_dir)

        if ref_metadata:
            print(
                f"[IR-DEBUG] {model_id}: reference IR generated with "
                f"transformers={ref_metadata.get('transformers_version', 'unknown')}, "
                f"optimum-intel={ref_metadata.get('optimum_intel_version', 'unknown')}, "
                f"openvino={ref_metadata.get('openvino_version', 'unknown')}, "
                f"date={ref_metadata.get('generated_date', 'unknown')}"
            )
        else:
            print(f"[IR-DEBUG] {model_id}: no reference metadata.json found in {ref_ir_dir}")

        # Find all IR files (handles both single models and multi-component pipelines)
        ref_ir_files = find_ir_files(ref_ir_dir)

        if not ref_ir_files:
            pytest.skip(f"No openvino_model.xml files found in {ref_ir_dir}")

        # Export new IR
        model = model_class.from_pretrained(model_id, export=True, **EXPORT_KWARGS.get(model_id, {}))
        new_ir_dir = tmp_path / "new_ir"
        model.save_pretrained(new_ir_dir)

        # Find IR files in new export
        new_ir_files = find_ir_files(new_ir_dir)

        # Check that same components exist
        ref_components = {str(f.parent) for f in ref_ir_files}
        new_components = {str(f.parent) for f in new_ir_files}

        if ref_components != new_components:
            missing = ref_components - new_components
            extra = new_components - ref_components
            pytest.fail(
                f"Component mismatch for {model_id}:\n"
                f"  Missing components: {missing or 'none'}\n"
                f"  Extra components: {extra or 'none'}"
            )

        # Initialize OpenVINO Core for loading models
        core = Core()

        # Compare each component's IR using OpenVINO's compare_models
        all_differences = {}
        for ref_ir_file in ref_ir_files:
            component_name = str(ref_ir_file.parent) if str(ref_ir_file.parent) != "." else "root"

            ref_ir_path = ref_ir_dir / ref_ir_file
            new_ir_path = new_ir_dir / ref_ir_file

            # Load both models using OpenVINO Core
            ref_model = core.read_model(str(ref_ir_path))
            new_model = core.read_model(str(new_ir_path))

            # Compare models (compare_names=False to ignore auto-generated friendly names)
            try:
                compare_models(ref_model, new_model, compare_names=False)
            except RuntimeError as e:
                # Model comparison failed - store the error message
                all_differences[component_name] = str(e).strip().split("\n")

        # Report all differences
        if all_differences:
            diff_lines = []
            for component, diffs in all_differences.items():
                diff_lines.append(f"\n[{component}]")
                for diff in diffs:
                    diff_lines.append(f"  {diff}")

            diff_msg = "\n".join(diff_lines)

            # Add metadata info to failure message
            metadata_info = ""
            if ref_metadata:
                metadata_info = (
                    f"\nReference IR was generated with:\n"
                    f"  - transformers: {ref_metadata.get('transformers_version', 'unknown')}\n"
                    f"  - optimum-intel: {ref_metadata.get('optimum_intel_version', 'unknown')}\n"
                    f"  - openvino: {ref_metadata.get('openvino_version', 'unknown')}\n"
                    f"  - date: {ref_metadata.get('generated_date', 'unknown')}\n"
                )

            pytest.fail(
                f"IR structure has changed for {model_id}:{diff_msg}\n"
                f"{metadata_info}\n"
                f"This may indicate a regression or improvement in IR generation. "
                f"Review the changes and update reference IRs if the change is intentional."
            )
