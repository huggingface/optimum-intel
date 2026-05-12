import os
import sys
import unittest

#This code is for eliminating unnecessary code text from the output
import pytest
@pytest.fixture(autouse=True, scope="session")
def set_tb_style(pytestconfig):
    pytestconfig.option.tbstyle = "line"

# we are adding this , so the parent directory (tests/openvino/) is in the python search path for utils_test.py to be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
import textwrap
import re

from parameterized import parameterized
from utils_tests import MODEL_NAMES, OPENVINO_DEVICE, REMOTE_CODE_MODELS

# Maps architecture name -> list of  transformation needed to be applied , as per expected_transformations.txt
def _load_expected_transformations(path):
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            arch, _, transforms_str = line.partition(":")
            result[arch.strip()] = [
                t.strip() for t in transforms_str.split(",") if t.strip()
            ]
    return result


_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "expected_transformations.txt"
)
ARCH_TO_EXPECTED_TRANSFORMATIONS = _load_expected_transformations(_CONFIG_PATH)


def _capture_stderr_during(model_id, OPENVINO_DEVICE, trust_remote_code):
    #  Runs model loading in a subprocess to reliably capture OpenVINO C++ logs.

    code = textwrap.dedent(f"""
        import os
        os.environ["OV_ENABLE_PROFILE_PASS"] = "1"

        from optimum.intel import OVModelForCausalLM

        OVModelForCausalLM.from_pretrained(
            "{model_id}",
            export=True,
            compile=True,
            device="{OPENVINO_DEVICE}",
            trust_remote_code={trust_remote_code},
        )
    """)

    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE, 
       stderr=subprocess.STDOUT, 
        text=True,
    )

    return result.stdout


# Remove separators and lowercase for fuzzy comparison.
def normalize(name: str) -> str:
    return re.sub(r'[\s_\-]', '', name).lower()


# Extract transformation name — always last token before NUMBER ms +/-
def extract_transform_name(line: str) -> str | None:
    match = re.search(
        r'([A-Za-z][A-Za-z0-9_]*)\s+\d+ms\s*[+-]\s*$',
        line.strip()
    )
    return match.group(1) if match else None


# Algo to identify tranformations present with '+' in the log.
def check_failed_transformations(log: str, words: list[str]) -> dict:
    applied_norm_plus = []
    applied_norm_minus = []
    found_not_applied=[]

    for line in log.splitlines():
        stripped = line.strip()

        if not stripped:
            continue
        name = extract_transform_name(stripped)
        
        if name:
            if stripped.endswith('+'):
                applied_norm_plus.append(normalize(name))
            elif stripped.endswith('-'):
                applied_norm_minus.append(normalize(name))

    remaining = {normalize(w): w for w in words}

    for key in list(remaining.keys()):
        if key in applied_norm_plus:
            del remaining[key]
        elif key in applied_norm_minus:
            found_not_applied.append(remaining[key])
            del remaining[key]

   

    return {
        "not_found": list(remaining.values()),
        "not_applied":found_not_applied
    }



class OVTransformationTest(unittest.TestCase):
    
    @parameterized.expand(
        list(ARCH_TO_EXPECTED_TRANSFORMATIONS.items())
    )
    def test_transformations_applied(
        self,
        model_arch,
        expected_transforms
    ):
        model_id = MODEL_NAMES[model_arch]
        trust_remote_code = model_arch in REMOTE_CODE_MODELS

        log_output = _capture_stderr_during(
            model_id,
            OPENVINO_DEVICE,
            trust_remote_code
        )

        result = check_failed_transformations(
            log_output,
            expected_transforms
        )

        errors=[]
        not_found = ", ".join(result["not_found"])
        not_applied = ", ".join(result["not_applied"])
        if not_applied:
            err = (
               f"These transformations were not 'applied' for '{model_arch}' architecture: "
                + not_applied
                  )
            errors.append(err)
           
        if not_found:
            err=   (
                       f"These transformations were not 'found' in the '{model_arch}'  transformation set: "
                          + not_found
                      )
            errors.append(err)
        
        RED = "\033[91m"
        RESET = "\033[0m"
        if errors:
            # raise AssertionError("\n".join(errors))
            raise AssertionError(f"{RED}" + "\n".join(errors) + f"{RESET}")
    
    
            
   


if __name__ == "__main__":
    unittest.main()
