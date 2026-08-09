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

## Technical references

- [model-patching-patterns.md](model-patching-patterns.md) — tracing failures,
  data-dependent control flow, and vectorized MoE reference.
- [inference-validation.md](inference-validation.md) — common accuracy-diagnosis
  guidance for end-to-end validation.

Use `.model_analysis/<model_type>_analysis.md` from the model-analysis agent as
the architecture source of truth. Do not replace real-model evidence with
assumptions from a nearby family or tiny fixture.

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

Before adding new implementation code, search the repository for existing
helpers, base classes, patchers, dummy generators, runtime utilities, and
model-family implementations that already provide the required behavior.
Prefer, in order:

1. reuse an existing implementation unchanged;
2. subclass or configure an existing implementation;
3. extract or minimally generalize shared behavior when multiple architectures
   need it;
4. add model-specific code only when the required behavior is genuinely unique.

Do not copy an existing implementation into a new model-specific helper merely
to change names or small constants. When new code is required, document why the
closest existing implementation cannot be reused safely.

Depending on the reproduced failure, update only the required integration
points:

- `optimum/exporters/openvino/model_configs.py` for registration, behaviors,
  inputs/outputs, dummy generators, and version constraints;
- `optimum/exporters/openvino/input_generators.py` for model-specific inputs;
- `optimum/exporters/openvino/model_patcher.py` for tracing-incompatible code;
- `optimum/intel/openvino/` for runtime loading, preprocessing, generation, or
  cache/stateful handling.

Patch the exact class identified by the traceback. Before implementing a new
patch, check whether an existing patcher or traceable helper already handles
the same operation and can be reused or minimally generalized. Replace
data-dependent Python control flow with traceable tensor operations where
necessary without changing unrelated architectures.
After every source edit, rerun the original
reproducer and verify it passes the previous failure point.

Read [model-patching-patterns.md](model-patching-patterns.md) when tracing,
dynamic routing, or MoE behavior is involved.

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
- `tests/openvino/test_transformations.py` for applicable graph-transformation coverage;
- `tests/openvino/utils_tests.py` for fixtures and expected IR details.

If the model requires a Transformers version that is outside the versions
supported by the main test matrix, also add its focused tests to
`.github/workflows/test_openvino_preview_models.yml`. Install the verified
required Transformers version in a dedicated step and run only that model's
tests, so existing models continue to use their supported versions.

For architectures that use supported graph transformations, add the
corresponding coverage in `tests/openvino/test_transformations.py` by following
the existing tests for the closest supported architecture:

- for models with MoE blocks, verify the applicable transformation such as
  `ConvertTiledMoEBlockToGatherMatmuls`;
- for models with RoPE blocks, verify `RoPEFusion`.

Add these tests only when the architecture actually exercises the corresponding
pattern. Reuse existing transformation-test helpers and patterns instead of
duplicating equivalent test logic.

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

Follow the task-specific repository-test requirements supplied for `<task>`.

If a required test cannot run because of hardware, time, credentials, or an
unavailable fixture, report the exact blocker and leave support incomplete.
Do not substitute an ad-hoc script for required repository coverage.

Use the same tiny-model artifact for repository tests and the evidence claimed
for that fixture. A WWB score or generation result from a separately generated
tiny model does not validate `_create_tiny_<model_type>_model()`. Run HF-vs-OV
comparison, task execution, and save/reload against the exact helper output.

When a repository test exposes a fixture defect, repair the helper rather than
weakening tolerances or skipping the architecture. Typical checks include
effective nested precision, cache invalidation, remote-code registration,
processor assets, supported exporter behaviors, and architectural dimension
invariants.

Follow the task-specific output-validity requirements before using the fixture
for accuracy comparison.

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
3. Execute the requested task through its real end-to-end inference path.
4. Compare with the Hugging Face reference using identical preprocessing,
   inputs, settings, and output boundary.
5. Run targeted repository tests and formatting or lint checks.

Follow the task-specific end-to-end validation instructions supplied for
`<task>`. Use [inference-validation.md](inference-validation.md) for common
accuracy-diagnosis guidance.

Loading, saving, conversion alone, or a forward-only call does not prove
support. If quality differs, compare preprocessing tensors, submodel outputs,
embedding insertion, logits, and task outputs to find the first divergence.

## Step 6 — Documentation and cleanup

Add the supported model type to `docs/source/openvino/models.mdx` using the
existing format.

Inspect the complete diff before reporting success. Search the touched areas
for existing helpers, patchers, configurations, and tests with the same
behavior as newly added code. Replace semantic duplicates with reuse,
subclassing, configuration, or a minimal shared generalization when safe, and
remove now-redundant code. Then remove scratch files, debug prints, absolute
local paths, and unrelated edits. Do not commit, push, or open a pull request
unless explicitly requested.
