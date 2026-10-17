#!/usr/bin/env python3
"""
Regenerate the reference OpenVINO IRs that `tests/openvino/test_irs.py` compares against.

The model list is taken from the test suite itself, so a fixture added to `HUB_MODEL_NAMES` /
`ARCH_TO_MODEL_CLASS` is picked up here without a second list to maintain. Each model is exported
in its own subprocess: an export that segfaults or exhausts memory then costs one model rather
than the whole run.

Usage:
    python tests/scripts/batch_upload_irs.py --missing-only
    python tests/scripts/batch_upload_irs.py --models tiny-random-gpt2 tiny-random-llama
    python tests/scripts/batch_upload_irs.py --models-file list.txt
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


UPLOAD_SCRIPT = Path(__file__).resolve().parent / "upload_reference_irs.py"
TESTS_OPENVINO_DIR = Path(__file__).resolve().parents[1] / "openvino"

# 30 minutes: the largest fixtures are 10-component pipelines, and a reference generated from a
# half-finished export is worse than no reference at all.
PER_MODEL_TIMEOUT = 1800


def load_models_from_test_suite():
    """The (model_id, class_name) pairs `test_irs.py` will compare, in collection order."""
    sys.path.insert(0, str(TESTS_OPENVINO_DIR))
    import test_irs

    return [(model_id, model_class.__name__) for model_id, model_class in test_irs.TEST_MODELS]


def load_models_from_file(path):
    """Read tab-separated `<model_id>`/`<model_class>` rows, ignoring blanks and comments."""
    models = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        model_id, model_class = line.split("\t")
        models.append((model_id, model_class))
    return models


def has_reference(model_id):
    """Whether `model_id` already has an `ov` revision on the Hub."""
    from huggingface_hub import HfApi

    try:
        return any(branch.name == "ov" for branch in HfApi().list_repo_refs(model_id).branches)
    except Exception as error:
        print(f"  ! could not list refs for {model_id} ({type(error).__name__}: {error}), treating as missing")
        return False


def run_upload(model_id, model_class, log_file, extra_args):
    """Export and upload one model in a subprocess. Returns True on success."""
    start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n[{timestamp}] {model_id} ({model_class})")
    log_file.write(f"\n{'=' * 80}\n[{timestamp}] {model_id} ({model_class})\n")
    log_file.flush()

    try:
        result = subprocess.run(
            # `sys.executable`, not a bare "python": the child must run in the same interpreter
            # (and therefore the same openvino/transformers versions) as this script, otherwise a
            # forgotten venv activation silently generates references on the wrong stack.
            [sys.executable, str(UPLOAD_SCRIPT), model_id, model_class, *extra_args],
            capture_output=True,
            text=True,
            timeout=PER_MODEL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  x timeout ({elapsed:.1f}s)")
        log_file.write(f"TIMEOUT after {elapsed:.1f}s\n")
        log_file.flush()
        return False

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ok ({elapsed:.1f}s)")
        log_file.write(f"SUCCESS ({elapsed:.1f}s)\n{result.stdout[-500:]}")
        log_file.flush()
        return True

    print(f"  x failed ({elapsed:.1f}s)")
    log_file.write(f"FAILED ({elapsed:.1f}s)\nSTDOUT:\n{result.stdout[-1000:]}\nSTDERR:\n{result.stderr[-1000:]}\n")
    log_file.flush()
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-file", help="File of tab-separated model id / model class rows")
    parser.add_argument("--models", nargs="+", help="Only handle model ids containing one of these substrings")
    parser.add_argument("--missing-only", action="store_true", help="Skip models that already have an `ov` revision")
    parser.add_argument("--commit-directly", action="store_true", help="Commit to `ov` instead of opening PRs")
    parser.add_argument("--log-dir", type=Path, default=Path.cwd(), help="Where to write the run log")
    args = parser.parse_args()

    models = load_models_from_file(args.models_file) if args.models_file else load_models_from_test_suite()

    if args.models:
        models = [(mid, cls) for mid, cls in models if any(pattern in mid for pattern in args.models)]

    if args.missing_only:
        print(f"Checking which of {len(models)} models already have a reference...")
        models = [(mid, cls) for mid, cls in models if not has_reference(mid)]

    if not models:
        print("Nothing to do: no model matched the given filters.")
        return 0

    extra_args = ["--commit-directly"] if args.commit_directly else []

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"upload_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    print(f"Uploading references for {len(models)} models; log: {log_path}")

    succeeded, failed = [], []
    start = time.time()

    with open(log_path, "w") as log_file:
        log_file.write(f"Batch upload started {datetime.now():%Y-%m-%d %H:%M:%S} for {len(models)} models\n")

        for index, (model_id, model_class) in enumerate(models, 1):
            print(f"\n[{index}/{len(models)}] {len(succeeded)} ok, {len(failed)} failed so far")
            if run_upload(model_id, model_class, log_file, extra_args):
                succeeded.append(model_id)
            else:
                failed.append((model_id, model_class))

        summary = (
            f"\n{'=' * 80}\nDone in {(time.time() - start) / 60:.1f} min: "
            f"{len(succeeded)} succeeded, {len(failed)} failed\n"
        )
        print(summary)
        log_file.write(summary)

        for model_id, model_class in failed:
            print(f"  FAILED {model_id} ({model_class})")
            log_file.write(f"  FAILED {model_id} ({model_class})\n")

    # Non-zero exit so a CI run that could not regenerate every reference is not reported green.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
