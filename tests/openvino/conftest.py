import shutil

import pytest
from models.tiny_qwen3_omni import generate as generate_tiny_qwen3_omni
from utils_tests import MODEL_NAMES

from optimum.intel.utils.import_utils import is_transformers_version


@pytest.fixture(scope="session", autouse=True)
def qwen3_omni_model_path(tmp_path_factory: pytest.TempPathFactory) -> None:
    if not is_transformers_version(">=", "4.57.0.dev0"):
        yield
        return
    output_dir = tmp_path_factory.mktemp("tiny-qwen3-omni")
    generate_tiny_qwen3_omni(output_dir)
    MODEL_NAMES["qwen3_omni"] = str(output_dir)
    yield
    shutil.rmtree(output_dir, ignore_errors=True)
