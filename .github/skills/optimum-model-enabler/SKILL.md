---
name: optimum-model-enabler
description: "Add and validate support for a Hugging Face model architecture in the Optimum Intel OpenVINO backend, including exporter configuration, patching, repository tests, and documentation."
disable-model-invocation: false
user-invocable: true
argument-hint: "<model_id> <task>"
---

# Optimum Intel Model Enabler

Use this skill to diagnose an OpenVINO export or inference failure and add the
smallest complete model-support implementation.

## Step 1 — Reproduce the failure

Run the requested export before editing anything:

```bash
optimum-cli export openvino \
  --model <model_id> \
  --task <task> \
  <output_dir>
```

Add `--trust-remote-code` only when required. Record the command, installed
package versions, first actionable traceback, and exact failing class or
function. Distinguish model-support defects from environment, dependency, Hub,
and benchmark-tool failures.

When the failure is an environment, dependency, Transformers/Optimum API, or
benchmark-tool mismatch, diagnose that concrete mismatch before changing model
support code. Prefer the smallest targeted dependency or source compatibility
fix, then rerun the exact failing command. Do not repeatedly change package
versions until a traceback disappears, and never patch a similarly named class
when the traceback identifies a different one.

## Step 2 — Analyze the real architecture

Inspect the original model configuration and implementation. Record:

- `model_type`, `architectures`, `auto_map`, task, and modality;
- text, vision, audio, projector, cache, position-ID, and MoE components;
- required processor/tokenizer assets and special tokens;
- forward input/output contracts and custom dummy-input requirements;
- the closest supported Optimum Intel architecture and material differences.

For remote-code models, derive `MIN_TRANSFORMERS_VERSION` and
`MAX_TRANSFORMERS_VERSION` from verified upstream configuration and
compatibility evidence. Never use placeholder bounds such as `0`, `999`, or
`999.9.9`.

Record the exact compatibility fields found in the original configuration,
including `transformers_version` and explicit minimum or maximum fields. When
possible, verify the selected version range at its boundaries. Do not infer a
range solely from a nearby in-library architecture.

Do not map a model to a nearby architecture unless its inputs, outputs, cache,
position IDs, tracing behavior, and runtime behavior are compatible.

## Step 3 — Implement targeted support

Depending on the reproduced failure, update only the required integration
points:

- `optimum/exporters/openvino/model_configs.py` for registration, behaviors,
  inputs/outputs, dummy generators, and version constraints;
- `optimum/exporters/openvino/input_generators.py` for model-specific inputs;
- `optimum/exporters/openvino/model_patcher.py` for tracing-incompatible code;
- `optimum/intel/openvino/` for runtime loading, preprocessing, generation, or
  cache/stateful handling.

Patch the exact class identified by the traceback. Replace data-dependent
Python control flow with traceable tensor operations where necessary without
changing unrelated architectures. After every source edit, rerun the original
reproducer and verify it passes the previous failure point.

For multi-behavior exporters, verify every value in `SUPPORTED_BEHAVIORS`:

- `get_model_for_behavior()` returns the intended real submodel;
- `with_behavior()` returns a non-`None` export configuration;
- its inputs and outputs match that submodel;
- unsupported enum values are excluded instead of being advertised.

Do not consider successful conversion sufficient when a behavior silently
maps to the wrong component or produces a `None` configuration.

## Step 4 — Add repository tests

Source support without tests is incomplete. Update the applicable files:

- `tests/openvino/test_decoder.py` for decoder-only generation;
- `tests/openvino/test_seq2seq.py` for seq2seq and VLM integration;
- `tests/openvino/test_export.py` for model/API export coverage;
- `tests/openvino/test_exporters_cli.py` for CLI export coverage;
- `tests/openvino/test_quantization.py` for compression and quantization;
- `tests/openvino/utils_tests.py` for fixtures and expected IR details.

For a new architecture, invoke the **tiny-model-creator** agent and follow the
Kokoro pattern in `tests/openvino/utils_tests.py`:

1. Add `_create_tiny_<model_type>_model()`.
2. Construct a reduced, architecture-identical random model without loading
   original weights.
3. Save required model, config, tokenizer, processor, and remote-code assets
   into a cached temporary directory.
4. Reuse the completed directory on repeated calls.
5. Map `"<model_type>"` to the helper instead of adding a new Hub model ID.

Do not upload new tiny models to the Hub. Reuse a Hub fixture only when it
predates the support and is already the established repository fixture.

For VLMs, cover all exported submodels and update relevant architecture,
remote-code, video, compression, and preprocessing matrices. Run targeted
tests from every modified test file. A `-k` command selecting zero tests is not
evidence.

Specifically verify:

- every applicable `REMOTE_CODE_MODELS` collection contains the architecture
  when `trust_remote_code=True` is required;
- a later class-level assignment does not overwrite an earlier registration;
- `_ARCHITECTURES_TO_EXPECTED_INT8` contains measured expected node counts for
  every exported VLM submodel;
- CLI export, tokenizer export, save/reload, deterministic generation, and
  HF-vs-OpenVINO comparison are covered where applicable.

Run pytest with collection output or explicit node IDs and confirm that each
command selected at least one test.

Use the same tiny-model artifact for repository tests and the evidence claimed
for that fixture. A WWB score or generation result from a separately generated
tiny model does not validate `_create_tiny_<model_type>_model()`. Run HF-vs-OV
comparison, generation, and save/reload against the exact helper output.

When a repository test exposes a fixture defect, repair the helper rather than
weakening tolerances or skipping the architecture. Typical checks include
effective nested precision, cache invalidation, remote-code registration,
processor assets, supported exporter behaviors, and architectural dimension
invariants.

## Local installation safety

When validation requires installing this checkout, use an editable no-deps
installation:

```bash
python -m pip install -e . --no-deps
```

Record the Transformers version before and after installation and verify that
it did not change. Never use a plain editable installation that may silently
downgrade or replace the model-compatible Transformers version.

## Step 5 — Validate end to end

Run all applicable checks:

1. Export the local tiny model and, when resources allow, the real model.
2. Load the export through the appropriate Optimum Intel OpenVINO API.
3. Execute deterministic `model.generate()` through the requested task.
4. For image-text tasks, use a real image and text prompt together.
5. Compare with the Hugging Face reference using identical preprocessing,
   prompt, image, generation settings, and decoded-token boundary.
6. Run targeted repository tests and formatting or lint checks.

Loading, saving, conversion alone, or a forward-only call does not prove
generative support. If quality differs, compare preprocessing tensors,
submodel outputs, embedding insertion, logits, and generated token IDs to find
the first divergence.

## Step 6 — Documentation and cleanup

Add the supported model type to `docs/source/openvino/models.mdx` using the
existing format.

Inspect `git diff --name-only` before reporting success. Remove scratch files,
debug prints, absolute local paths, and unrelated edits. Do not commit, push,
or open a pull request unless explicitly requested.
