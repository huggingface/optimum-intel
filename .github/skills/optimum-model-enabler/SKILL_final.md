---
name: optimum-model-enabler
description: "Add and validate support for a Hugging Face model architecture in the Optimum Intel OpenVINO backend, including exporter configuration, patching, repository tests, and documentation."
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

When a model-compatible Transformers version is inside Optimum Intel's
declared supported dependency range but the local source imports an API
available only in a newer version, treat it as an Optimum Intel compatibility
defect rather than an unrecoverable environment failure. Reuse an existing
version guard or compatibility helper, or add the smallest guarded
implementation, and validate the model using a compatible Transformers version
inside the declared supported dependency range.

Do not add the model to the Preview Models Support workflow solely because the
default CI matrix does not currently exercise that exact compatible version.
Preview-workflow routing is decided separately from Optimum Intel's declared
Transformers dependency range.

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

Treat algorithmic equivalence as duplication even when symbol names, tensor
names, dimensions, configuration field names, or model names differ.

If a new method follows the same algorithmic steps as an existing method and
differs only by:
- input rank or shape;
- which columns of a shape tensor are used;
- model/config attribute names;
- constant values;
- module lookup paths;
- optional inputs;
- minor pre/post-processing;

do NOT introduce a separate model-specific implementation of that algorithm. Prefer, in order:

1. call the existing method directly;
2. parameterize the existing method;
3. extract the common algorithm into a shared helper and keep only a thin
   model-specific adapter;
4. use a narrow subclass override containing only the genuinely different
   operations.

A method is not considered model-specific merely because the surrounding model
architecture is different.

For VLMs, compare the requested behavior with the existing language-model
patchers, MLA/attention helpers, vision rotary/window-index helpers, multimodal
embedding insertion, generation input preparation, and dummy generators before
adding a new implementation. An empty subclass or a copied method body is not a
model integration; use the existing class directly, subclass only the differing
behavior, or extract the common operation into one shared helper.

Do not add an override that only forwards arguments to `super()` or reproduces
the inherited implementation without changing behavior. For every new override,
identify the exact behavioral delta from the inherited implementation. If there
is no behavioral delta, remove the override. If a redundant-looking override is
required for framework dispatch, signature inspection, exporter discovery,
serialization, tracing, or another mechanical requirement, record that reason
and prove it with a targeted test.

### Mandatory reuse audit

Treat reuse analysis as a completion gate, not a suggestion. Perform it once
before implementation and again on the complete diff before reporting success.
Scope the audit to every added or materially changed production/test symbol and
the nearest supported architecture families; do not refactor unrelated legacy
duplication merely because the audit finds it.

For each new class, method, function, fixture helper, configuration block, and
test helper:

1. Search for the exact symbol name and for the behavior it implements. Search
   concepts and tensor operations as well as names; renamed copies are still
   duplicates.
2. Compare it with at least the closest supported architecture and any shared
   base/helper used by that architecture.
3. Classify it as one of: reused unchanged, configured/subclassed, shared helper
   extracted, or genuinely model-specific.
4. For genuinely model-specific code, record the concrete incompatible input,
   output, cache layout, tensor shape, module API, or runtime behavior that makes
   reuse unsafe. A different model name is not sufficient justification.
5. Replace semantic duplicates with direct reuse, a narrow subclass/override,
   or a minimal shared generalization. Preserve existing behavior with targeted
   tests when extracting shared code.

At minimum, separately audit all of the following operations before adding
model-specific implementations:

- language-model / decoder patching;
- KV-cache layout and dummy past-key-value generation;
- attention forward replacements;
- RoPE / rotary-position construction;
- window-index construction and reverse-index restoration;
- cumulative sequence-length construction;
- full-attention and window-attention mask construction;
- multimodal token detection and vision/text embedding insertion;
- cached-decode vision skipping;
- `prepare_inputs_for_generation`;
- processor/chat-template preprocessing;
- dummy tensor generation;
- VLM behavior routing;
- patcher enter/exit restoration logic;
- tiny-model construction and repository-test utilities.

Search by operation as well as symbol name. For example, search for distinctive
operations such as `masked_scatter`, `window_index`, `cu_seqlens`,
`scaled_dot_product_attention`, rotary embedding construction, image-token
masking, and vision-embedding insertion.

If a newly added implementation is described in comments or documentation with
phrases such as "mirrors", "same as", "based on", "matching", "adapted from", or
"similar to" an existing architecture, that is a mandatory signal to perform a
shared-helper/subclassing audit before keeping the new implementation.

Include a mandatory reuse-audit table in the handoff with these columns:

| New/changed symbol | Existing candidate(s) inspected | Shared behavior |
| Decision | Remaining model-specific delta |

Every new production method or class must appear in the table.

For anything classified as genuinely model-specific, describe the precise lines
of behavior that cannot be shared. Statements such as "different architecture",
"different model", "different tensor names", or "different shapes" are
insufficient unless those differences prevent safe parameterization.

Before reporting success, inspect the final diff for newly added functions whose
bodies substantially reproduce an existing repository algorithm. If the common
algorithm can be represented as a shared helper with model-specific parameters
or adapters, perform that refactor before completion.

Do not accept comments saying that code "mirrors" another implementation as
justification for keeping both implementations.

### Reuse completion gate

The implementation is incomplete if any of the following remains:

- an exact copied method;
- a renamed semantic copy;
- a model-specific wrapper that only calls `super()`;
- two implementations of the same tensor algorithm differing only by shape,
  constants, or field names when those differences can be parameters;
- repeated multimodal embedding insertion/scatter logic that can use a shared
  helper;
- repeated rotary/window/mask construction that can use a shared helper;
- duplicated test setup or fixture logic already available through repository
  utilities.

Do not proceed to final validation or report success until these are resolved or
a concrete technical reason for non-reuse has been recorded.

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
After every source edit, rerun the original reproducer and verify it passes
the previous failure point.

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

Before deciding whether version-specific CI coverage belongs in
`.github/workflows/test_openvino_preview_models.yml`, read Optimum Intel's
declared Transformers dependency range from the repository packaging metadata
(for example `setup.py` or `pyproject.toml`).

Compare the model's verified Transformers compatibility interval against the
declared Optimum Intel dependency range.

- If at least one verified model-compatible Transformers version is inside
  Optimum Intel's declared supported dependency range, do NOT add the model to
  `.github/workflows/test_openvino_preview_models.yml` solely because the
  default CI matrix does not currently test that exact version. Validate the
  model with a compatible version inside the supported dependency range and
  keep normal repository tests/model registration additive.

- Use `.github/workflows/test_openvino_preview_models.yml` only when the model
  requires a Transformers version outside Optimum Intel's currently declared
  supported dependency range, or when the repository already explicitly
  classifies that architecture/version-specific coverage as preview-only.

The workflow change must be additive: never rename, replace, or delete an
existing model's validation step to add the requested model.

For architectures that use supported graph transformations, add the
corresponding coverage in `tests/openvino/test_transformations.py` by following
the existing tests for the closest supported architecture:

- for models with MoE blocks, verify the applicable transformation such as
  `ConvertTiledMoEBlockToGatherMatmuls`;
- for models with RoPE blocks, add the architecture to the appropriate
  `ARCH_TO_EXPECTED_TRANSFORMATIONS` version guard and verify `RoPEFusion`.

Treat this transformation coverage as required when the architecture exercises
the corresponding pattern. Do not omit it silently. If an expected
transformation cannot apply, record the exported-graph evidence and exact reason
in the handoff and leave the support incomplete for maintainer review. Reuse
existing transformation-test helpers and patterns instead of duplicating
equivalent test logic.

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

Before INT8 or INT4 real-model export, run a small `nncf.compress_weights`
smoke test on an unrelated OpenVINO MatMul model in the exact export
interpreter. Record the OpenVINO, NNCF, NumPy, Python, and Optimum Intel
versions. If the smoke test fails, repair that isolated validation environment
before changing model code. In particular, when OpenVINO is a nightly `.dev`
build, follow `.github/workflows/test_openvino_nightly.yml`: test it with NNCF
from `https://github.com/openvinotoolkit/nncf.git` in an isolated overlay, then
rerun the smoke test and both quantized exports. Preserve the base environment,
record the resolved NNCF revision, and do not try arbitrary dependency versions.

Loading, saving, conversion alone, or a forward-only call does not prove
support. If quality differs, compare preprocessing tensors, submodel outputs,
embedding insertion, logits, and task outputs to find the first divergence.

## Step 6 — Documentation and cleanup

Add the supported model type to `docs/source/openvino/models.mdx` using the
existing format.

Inspect the complete diff and finish the mandatory reuse audit before reporting
success. Inspect every deletion against the current upstream base. Supported-
model lists, registry tables, repository-test parameter lists, documentation
model lists, and Preview Models workflow steps are additive for a new
architecture: restore any unrelated model, test, import, or workflow line that
was removed or replaced. Keep dependency workarounds out of global imports;
never add a broad `except Exception` fallback merely to make this model's
environment importable. Then remove scratch files, debug prints, absolute local
paths, and unrelated edits. Do not commit, push, or open a pull request unless
explicitly requested.
