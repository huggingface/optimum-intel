#!/usr/bin/env python3
"""
Export a model to OpenVINO IR and upload the result to the `ov` branch of its Hub repo,
where `tests/openvino/test_irs.py` looks for reference IRs.

Only the `.xml` / `.bin` files plus a `metadata.json` (recording the versions used to
generate them) are uploaded; everything else the export produces is stripped out.

Uploads open a pull request by default; pass `--commit-directly` to write to `ov` in place.

Usage:
    python tests/scripts/upload_reference_irs.py <model_id> <model_class>
    python tests/scripts/upload_reference_irs.py <model_id> <model_class> --no-upload --output-dir /tmp/ir
"""

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import openvino
import transformers
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

import optimum.intel


# Extra `from_pretrained` arguments some models need at export time. Must stay in sync with
# `EXPORT_KWARGS` in tests/openvino/test_irs.py: the test exports with these same arguments, so a
# reference generated without them would not match what the test produces.
EXPORT_KWARGS = {
    "optimum-intel-internal-testing/tiny-random-SpeechT5ForTextToSpeech": {"vocoder": "fxmarty/speecht5-hifigan-tiny"},
    "optimum-intel-internal-testing/tiny-stable-diffusion-torch-custom-variant": {"variant": "custom"},
}


def resolve_model_class(class_name):
    """
    The `optimum.intel` class called `class_name`.

    Resolved by name, the way `resolve_model_class` in tests/openvino/test_irs.py does it, so the
    set of exportable classes is whatever the installed optimum-intel provides. Listing them here
    instead would be a second copy of `ARCH_TO_MODEL_CLASS` to keep in step, and importing them
    eagerly would make one name this build does not ship break the script for every model.
    """
    model_class = getattr(optimum.intel, class_name, None)
    if model_class is None:
        raise SystemExit(
            f"optimum.intel has no class named {class_name!r}. Pass the class the suite uses for "
            f"this model (see ARCH_TO_MODEL_CLASS in tests/openvino/utils_tests.py)."
        )
    return model_class


def get_version_info():
    """Versions of everything that can influence the generated IR."""
    return {
        "transformers_version": transformers.__version__,
        "optimum_intel_version": version("optimum-intel") or "unknown",
        "openvino_version": openvino.__version__,
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
    }


def export_model(model_id, model_class, export_dir):
    """Export `model_id` to OpenVINO IR under `export_dir`."""
    extra_kwargs = EXPORT_KWARGS.get(model_id, {})
    print(f"Exporting {model_id} using {model_class.__name__}...")
    if extra_kwargs:
        print(f"  extra export kwargs: {extra_kwargs}")
    model = model_class.from_pretrained(model_id, export=True, **extra_kwargs)
    model.save_pretrained(export_dir)
    print(f"✓ Exported to {export_dir}")
    return model


def find_components(export_dir):
    """
    Paths of the exported IRs, relative to `export_dir` ("model" for a lone root-level model).

    Mirrors `find_ir_files` in tests/openvino/test_irs.py: multi-component models name their IRs
    after the component (`openvino_language_model.xml`, `openvino_vision_encoder.xml`), so globbing
    `openvino_model.xml` alone would report no components at all for every VLM and pipeline.
    """
    components = []
    for xml_path in sorted(export_dir.rglob("openvino*.xml")):
        if xml_path.stem in ("openvino_tokenizer", "openvino_detokenizer"):
            continue
        rel_path = xml_path.relative_to(export_dir)
        components.append("model" if str(rel_path) == "openvino_model.xml" else str(rel_path))
    return sorted(components)


def create_metadata(model_id, model_class_name, export_dir):
    """Write `metadata.json` next to the IRs."""
    metadata = {
        "model_id": model_id,
        "model_class": model_class_name,
        **get_version_info(),
        "components": find_components(export_dir),
    }
    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Created metadata.json")
    return metadata


def clean_export_dir(export_dir):
    """Drop everything except the IRs and the top-level metadata.json."""
    keep_suffixes = {".xml", ".bin"}
    files_removed = 0
    for path in export_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in keep_suffixes:
            continue
        if path.name == "metadata.json" and path.parent == export_dir:
            continue
        path.unlink()
        files_removed += 1
    print(f"✓ Cleaned export directory ({files_removed} unnecessary files removed)")


def upload_to_hub(model_id, export_dir, commit_message=None, create_pr=True):
    """
    Upload the cleaned export directory to the `ov` branch of `model_id`.

    Opens a pull request based on `ov` by default. A reference IR is the thing the test trusts, so
    replacing one is a reviewable decision: pushing straight to `ov` from an automated run would
    let the suite rewrite its own expectations and go green on a genuine regression. Pass
    `create_pr=False` to commit directly.

    Built from explicit operations rather than `upload_folder`, which refuses `create_pr` together
    with a non-default `revision`. That is a client-side restriction: the commit endpoint takes the
    base revision in its path and `create_pr` as a query parameter, so `create_commit` can open a
    pull request against `ov`.
    """
    api = HfApi()

    if commit_message is None:
        version_info = get_version_info()
        commit_message = (
            "Add reference OpenVINO IRs\n\n"
            "Generated with:\n"
            f"- transformers=={version_info['transformers_version']}\n"
            f"- optimum-intel=={version_info['optimum_intel_version']}\n"
            f"- openvino=={version_info['openvino_version']}"
        )

    clean_export_dir(export_dir)

    try:
        api.create_branch(repo_id=model_id, branch="ov", repo_type="model")
        print("✓ Created 'ov' branch")
    except Exception:
        print("  'ov' branch already exists")

    new_files = sorted(p.relative_to(export_dir).as_posix() for p in export_dir.rglob("*") if p.is_file())
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(export_dir / name)) for name in new_files]

    # Stand in for `upload_folder`'s `delete_patterns="*"`, which commits cannot express: a freshly
    # created `ov` branch forks from `main` and carries its source weights, and a model whose
    # component set shrank would otherwise keep orphan IRs that nothing compares against.
    stale = set(api.list_repo_files(repo_id=model_id, revision="ov", repo_type="model"))
    stale -= set(new_files) | {".gitattributes"}
    operations += [CommitOperationDelete(path_in_repo=name) for name in sorted(stale)]

    commit = api.create_commit(
        repo_id=model_id,
        repo_type="model",
        revision="ov",
        operations=operations,
        commit_message=commit_message,
        create_pr=create_pr,
    )

    print(f"  {len(new_files)} files uploaded, {len(stale)} stale files removed")
    if create_pr:
        print(f"✓ Opened PR against {model_id} (base 'ov'): {commit.pr_url}")
    else:
        print(f"✓ Uploaded to {model_id} (revision 'ov'): {commit.commit_url}")


def main():
    parser = argparse.ArgumentParser(description="Export a model to OpenVINO IR and upload it as a test reference.")
    parser.add_argument("model_id", help="Hub model id, e.g. optimum-intel-internal-testing/tiny-random-gpt2")
    parser.add_argument("model_class", help="optimum-intel class to export with, e.g. OVModelForCausalLM")
    parser.add_argument("--commit-message", help="Override the generated commit message")
    parser.add_argument("--no-upload", action="store_true", help="Export only, do not touch the Hub")
    parser.add_argument("--output-dir", type=Path, help="Where to write the export (requires --no-upload)")
    parser.add_argument(
        "--commit-directly",
        action="store_true",
        help="Commit to the `ov` branch instead of opening a pull request",
    )
    args = parser.parse_args()

    if args.output_dir and not args.no_upload:
        parser.error("--output-dir requires --no-upload")

    model_class = resolve_model_class(args.model_class)

    if args.output_dir:
        export_dir = args.output_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        export_model(args.model_id, model_class, export_dir)
        create_metadata(args.model_id, args.model_class, export_dir)
        print(f"\nExport kept at {export_dir} (not uploaded)")
        return

    with TemporaryDirectory() as tmp_dir:
        export_dir = Path(tmp_dir)
        export_model(args.model_id, model_class, export_dir)
        create_metadata(args.model_id, args.model_class, export_dir)

        if args.no_upload:
            print("\nSkipping upload (--no-upload)")
            return

        upload_to_hub(args.model_id, export_dir, args.commit_message, create_pr=not args.commit_directly)


if __name__ == "__main__":
    main()
