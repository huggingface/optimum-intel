"""
Test IR (Intermediate Representation) stability across transformers versions.

This test compares newly exported OpenVINO IRs against reference IRs to detect
regressions when upgrading transformers or other dependencies.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from openvino import Core, Model

from optimum.intel import OVDiffusionPipeline


REFERENCE_IR_DIR = Path(__file__).parent / "reference_irs"


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
        for idx in range(model_one_ops[i].get_output_size()):
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
    Find all openvino_model.xml files in directory and subdirectories.
    Returns list of paths relative to the directory.
    """
    ir_files = []
    for xml_file in directory.rglob("openvino_model.xml"):
        # Get relative path from directory
        rel_path = xml_file.relative_to(directory)
        ir_files.append(rel_path)
    return sorted(ir_files)


class TestIRStability:
    """Test suite for IR stability across transformers versions."""

    @pytest.mark.parametrize(
        "model_id,model_class,ref_dir",
        [
            ("optimum-intel-internal-testing/tiny-random-flux", OVDiffusionPipeline, "tiny-random-flux"),
        ],
    )
    def test_ir_stability(self, model_id: str, model_class, ref_dir: str, tmp_path: Path):
        """Test that exported IR matches reference IR."""

        # Load reference IR directory
        ref_ir_dir = REFERENCE_IR_DIR / ref_dir

        if not ref_ir_dir.exists():
            pytest.skip(f"Reference IR directory not found: {ref_ir_dir}")

        # Load reference metadata
        ref_metadata = load_reference_metadata(ref_ir_dir)

        # Find all IR files (handles both single models and multi-component pipelines)
        ref_ir_files = find_ir_files(ref_ir_dir)

        if not ref_ir_files:
            pytest.skip(f"No openvino_model.xml files found in {ref_ir_dir}")

        # Export new IR
        export_kwargs = {"export": True}
        if model_class == OVDiffusionPipeline:
            export_kwargs["trust_remote_code"] = True

        model = model_class.from_pretrained(model_id, **export_kwargs)
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
