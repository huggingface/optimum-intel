import os
import sys
import unittest



# we are adding this , so the parent directory (tests/openvino/) is in the python search path for utils_test.py to be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import subprocess
import textwrap
import re
from difflib import get_close_matches

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


# Extract transformation name — always last token before NUMBERms +/-
def extract_transform_name(line: str) -> str | None:
    match = re.search(
        r'([A-Za-z][A-Za-z0-9_]*)\s+\d+ms\s*[+-]\s*$',
        line.strip()
    )
    return match.group(1) if match else None


# Algo to identify tranformations present with '+' in the log.
def check_failed_transformations(log: str, words: list[str]) -> dict:
    applied_raw = []
    applied_norm = []

    for line in log.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if not stripped.endswith('+'):  #  neglect '-' because those transformations are not applied
            continue

        name = extract_transform_name(stripped)
        if name:
            applied_raw.append(name)
            applied_norm.append(normalize(name))

    remaining = {normalize(w): w for w in words}

    for key in list(remaining.keys()):
        if key in applied_norm:
            del remaining[key]

    hints = {}
    for key, original in remaining.items():
        matches = get_close_matches(key, applied_norm, n=2, cutoff=0.8)

        if matches:
            readable = [
                applied_raw[applied_norm.index(m)]
                for m in matches
            ]
            hints[original] = readable

    return {
        "not_found": list(remaining.values()),
        "hints": hints
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

        if result["not_found"]:
            not_found = ", ".join(result["not_found"])
            hints = result["hints"]

            hint_lines = ""

            if hints:
                hint_lines = (
                    "\nPossible matches in log:\n"
                    + "\n".join(
                        f"  '{wrong}' → did you mean '{', '.join(suggestions)}'?"
                        for wrong, suggestions in hints.items()
                    )
                )

            raise AssertionError(
                f"The following transformations were not applied for '{model_arch}' architecture: "
                f"{not_found}{hint_lines}"
            )


if __name__ == "__main__":
    unittest.main()
