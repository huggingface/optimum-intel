import pytest
import shutil
from utils_tests import MODEL_NAMES

from optimum.intel.utils.import_utils import is_transformers_version


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Dynamically add the 'gemma4' marker to every parameterized test whose
    name contains 'gemma4' (this also covers 'gemma4_moe')."""
    gemma4_marker = pytest.mark.gemma4
    for item in items:
        if "gemma4" in item.nodeid:
            item.add_marker(gemma4_marker)


@pytest.fixture(scope="session", autouse=True)
def qwen3_omni_moe_model_path(tmp_path_factory: pytest.TempPathFactory) -> None:
    if not is_transformers_version(">=", "4.57.0"):
        MODEL_NAMES["qwen3_omni_moe"] = ""
        yield
        return
    # Import only when transformers >= 4.57.0 to avoid ImportError
    from models.tiny_qwen3_omni_moe import generate as generate_tiny_qwen3_omni_moe

    output_dir = tmp_path_factory.mktemp("tiny-qwen3-omni-moe")
    generate_tiny_qwen3_omni_moe(output_dir)
    MODEL_NAMES["qwen3_omni_moe"] = str(output_dir)
    yield
    shutil.rmtree(output_dir, ignore_errors=True)
