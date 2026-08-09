import shutil

import pytest
from utils_tests import MODEL_NAMES, is_qwen3_omni_available


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Dynamically add the 'gemma4' marker to every parameterized test whose
    name contains 'gemma4' (this also covers 'gemma4_moe')."""
    gemma4_marker = pytest.mark.gemma4
    for item in items:
        if "gemma4" in item.nodeid:
            item.add_marker(gemma4_marker)


@pytest.fixture(scope="session", autouse=True)
def qwen3_omni_model_path(tmp_path_factory: pytest.TempPathFactory) -> None:
    # This autouse fixture generates the dense tiny model, so it must not import the generator (which
    # pulls in transformers' dense qwen3_omni classes) on builds where that architecture is absent —
    # doing so would break the whole session. See is_qwen3_omni_available for why a version check
    # alone is insufficient.
    if not is_qwen3_omni_available():
        MODEL_NAMES["qwen3_omni"] = ""
        yield
        return

    from models.tiny_qwen3_omni import generate as generate_tiny_qwen3_omni

    output_dir = tmp_path_factory.mktemp("tiny-qwen3-omni")
    generate_tiny_qwen3_omni(output_dir)
    MODEL_NAMES["qwen3_omni"] = str(output_dir)
    yield
    shutil.rmtree(output_dir, ignore_errors=True)
