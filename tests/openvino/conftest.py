import os

import pytest


def pytest_report_header(config):
    # Must live in conftest: pytest-xdist does not forward worker logs, so this only
    # reaches the terminal from the controller process.
    import openvino as ov
    import transformers

    device = os.getenv("OPENVINO_TEST_DEVICE", "CPU")
    header = f"OpenVINO {ov.__version__} Transformers {transformers.__version__} Device: {device}"
    if device == "NPU":
        try:
            header += f", NPU driver version: {ov.Core().get_property('NPU', 'NPU_DRIVER_VERSION')}"
        except Exception as exception:
            header += f", NPU driver version unavailable: {exception}"
    return header


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Dynamically add the 'gemma4' marker to every parameterized test whose
    name contains 'gemma4' (this also covers 'gemma4_moe')."""
    gemma4_marker = pytest.mark.gemma4
    for item in items:
        if "gemma4" in item.nodeid:
            item.add_marker(gemma4_marker)
