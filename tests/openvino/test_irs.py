"""
Test IR (Intermediate Representation) stability across transformers versions.

This test compares newly exported OpenVINO IRs against reference IRs to detect
regressions when upgrading transformers or other dependencies.
"""

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from optimum.intel import OVModelForSequenceClassification, OVModelForCausalLM, OVDiffusionPipeline


REFERENCE_IR_DIR = Path(__file__).parent / "reference_irs"


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


def parse_ir_xml(xml_path: Path) -> Dict:
    """Parse OpenVINO IR XML and extract key structural information."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    layers = root.find("layers")
    if layers is None:
        raise ValueError(f"No layers found in {xml_path}")

    # Extract operator types
    op_types = []
    layer_info = []

    for layer in layers.findall("layer"):
        layer_id = layer.get("id")
        layer_name = layer.get("name")
        layer_type = layer.get("type")
        layer_version = layer.get("version")

        op_types.append(layer_type)
        layer_info.append({
            "id": layer_id,
            "name": layer_name,
            "type": layer_type,
            "version": layer_version
        })

    # Count operator type frequencies
    op_type_counts = Counter(op_types)

    # Extract edges (connections between layers)
    edges_section = root.find("edges")
    edge_count = len(edges_section.findall("edge")) if edges_section is not None else 0

    return {
        "total_layers": len(layer_info),
        "op_type_counts": dict(op_type_counts),
        "layers": layer_info,
        "edge_count": edge_count,
        "net_version": root.get("version"),
    }


def compare_irs(ref_info: Dict, new_info: Dict) -> Tuple[bool, List[str]]:
    """Compare two IR structures and return differences."""
    differences = []

    # Compare total layer count
    if ref_info["total_layers"] != new_info["total_layers"]:
        differences.append(
            f"Layer count mismatch: reference={ref_info['total_layers']}, "
            f"new={new_info['total_layers']}"
        )

    # Compare operator type distributions
    ref_ops = ref_info["op_type_counts"]
    new_ops = new_info["op_type_counts"]

    all_op_types = set(ref_ops.keys()) | set(new_ops.keys())

    for op_type in sorted(all_op_types):
        ref_count = ref_ops.get(op_type, 0)
        new_count = new_ops.get(op_type, 0)

        if ref_count != new_count:
            differences.append(
                f"Operator '{op_type}' count mismatch: "
                f"reference={ref_count}, new={new_count}"
            )

    # Compare edge count
    if ref_info["edge_count"] != new_info["edge_count"]:
        differences.append(
            f"Edge count mismatch: reference={ref_info['edge_count']}, "
            f"new={new_info['edge_count']}"
        )

    # Compare net version
    if ref_info["net_version"] != new_info["net_version"]:
        differences.append(
            f"IR version mismatch: reference={ref_info['net_version']}, "
            f"new={new_info['net_version']}"
        )

    is_identical = len(differences) == 0
    return is_identical, differences


class TestIRStability:
    """Test suite for IR stability across transformers versions."""

    @pytest.mark.parametrize(
        "model_id,model_class,ref_dir",
        [
            (
                "optimum-intel-internal-testing/tiny-random-flux",
                OVDiffusionPipeline,
                "tiny-random-flux"
            ),
        ]
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

        # Compare each component's IR
        all_differences = {}
        for ref_ir_file in ref_ir_files:
            component_name = str(ref_ir_file.parent) if str(ref_ir_file.parent) != "." else "root"

            ref_ir_path = ref_ir_dir / ref_ir_file
            new_ir_path = new_ir_dir / ref_ir_file

            ref_info = parse_ir_xml(ref_ir_path)
            new_info = parse_ir_xml(new_ir_path)

            is_identical, differences = compare_irs(ref_info, new_info)

            if not is_identical:
                all_differences[component_name] = differences

        # Report all differences
        if all_differences:
            diff_lines = []
            for component, diffs in all_differences.items():
                diff_lines.append(f"\n[{component}]")
                for diff in diffs:
                    diff_lines.append(f"  - {diff}")

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
