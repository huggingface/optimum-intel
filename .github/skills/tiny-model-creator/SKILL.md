---
name: tiny-model-creator
description: "Create a small random Hugging Face model that preserves the original architecture and can serve as a local Optimum Intel repository test fixture."
disable-model-invocation: false
user-invocable: true
argument-hint: "<model_id> <task>"
---

# Tiny Model Creator

Create a tiny random model for the requested architecture without loading the
original model weights. The artifact must preserve the real architecture and
execute the requested generation path.

If repairing a tiny model after a validation or benchmark failure, start from
the exact failing local artifact, creation script, traceback, and generation
reproducer. Repair that artifact and rerun the same command; do not generate a
different model and use its success as evidence for the failed one.

## Step 1 — Inspect the original model

Download or load configuration and lightweight code/processor assets only.
Record:

- `model_type`, `architectures`, `auto_map`, and Transformers metadata;
- nested text, vision, audio, and projector configurations;
- hidden-size, head-count, grouping, rotary, cache, vocabulary, and special
  token invariants;
- tokenizer, processor, chat-template, and remote-code files required to load;
- the correct AutoModel class and generation interface for the task.

Determine and record the Transformers version or version range supported by
the original model, especially for trust-remote-code architectures. Create and
validate the tiny model using a compatible version; do not silently substitute
another model implementation because the active Transformers version lacks the
architecture.

Inspect precision fields under every relevant configuration level. Remote
models may use `dtype`, `torch_dtype`, or both, including separate values in
vision, text, audio, or projector sub-configs.

## Step 2 — Write a reusable constructor

Create `create_tiny_model.py` in the designated working directory. It must:

1. Load the original configuration without loading original weights.
2. Reduce layers, hidden dimensions, intermediate sizes, vocabulary, image
   resolution or patch counts, experts, and similar scale parameters.
3. Preserve divisibility and coupling invariants such as head dimensions,
   grouped-query attention, projector sizes, vision/text bridges, cache
   dimensions, and special-token IDs.
4. Instantiate random weights through the real architecture class.
5. Save every required config, tokenizer, processor, chat template, generation
   config, and remote-code asset.
6. Reuse a completed cached output directory on repeated calls.

Do not reuse a cache merely because `config.json` and a weight file exist.
Before returning it, validate a cache-format/version marker and all critical
configuration invariants, including architecture identity, dimensions,
special tokens, processor assets, and nested precision fields. Rebuild the
cache when the generator logic or required invariants change.

Keep construction logic easy to adapt into
`_create_tiny_<model_type>_model()` in `tests/openvino/utils_tests.py`.

## Step 3 — Verify architecture identity

Compare the original and tiny configurations. Preserve:

- `model_type` and `architectures`;
- task-relevant sub-config types and component roles;
- cache/stateful and position-ID contracts;
- VLM processor classes, placeholder/image tokens, and merge contracts;
- MoE/expert topology, even when expert counts are reduced.

When deliberately forcing a test model to float32, update and verify every
effective precision field used by the remote configuration (`dtype` and/or
`torch_dtype`, including nested sub-configs). Reload the saved model and check
its actual parameter dtypes; editing an ignored config key is not sufficient.

Never rename the model type, substitute a nearby architecture, or remove a
component merely to make export pass.

If the tiny model's `model_type`, `architectures`, or task-relevant component
identity differs from the original model, stop and report the mismatch. A tiny
model that generates successfully through another architecture is not a valid
fixture.

## Step 4 — Validate real generation

Reload the saved directory through its documented Transformers API and run
deterministic `model.generate()` with at least one new token.

- Text generation: tokenize a real prompt and decode generated tokens.
- Image-text generation: process a real image and prompt together, apply the
  correct chat template, and decode only newly generated tokens.
- Other generative tasks: exercise the real task pipeline, not a component
  forward pass.

If generation fails, repair the violated configuration invariant, recreate the
model, and rerun it. Loading, saving, or a forward pass alone is not success.

### Prevent collapsed or invalid tiny-model outputs

Before accepting the fixture, verify that its outputs are numerically valid and
informative enough for HF-vs-OpenVINO comparison:

1. Run a forward pass on representative task inputs and assert that logits and
   relevant intermediate outputs contain no NaN or Inf values.
2. Check that logits have non-zero finite variance and are not uniformly zero.
3. Generate multiple new tokens for at least two distinct prompts or inputs.
   Reject a fixture when all generated continuations collapse to zeros, one
   repeated token, or the same constant sequence across every input.
4. Record the finite-value check, logits variance, and generated token IDs in
   the validation evidence.

Example numerical checks:

```python
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits.float()
assert torch.isfinite(logits).all(), "Tiny model produced NaN/Inf logits"
assert logits.std().item() > 0, "Tiny model logits collapsed to a constant"
```

If outputs collapse, do not weaken the comparison test. Experiment with a less
aggressive reduction, retain more layers/hidden dimensions or vocabulary
structure, and verify all normalization/scaling invariants. When appropriate
for the architecture, reinitialize the output head with greater variance, for
example:

```python
with torch.no_grad():
    model.lm_head.weight.normal_(mean=0.0, std=0.2)
```

Respect tied embeddings and architecture-specific initialization: if the
output head is tied to input embeddings, reinitialize through the model's
actual shared parameter and confirm tying remains intact after saving and
reloading. Re-run finite-logit, diversity, generation, and save/reload checks
after every initialization change.

The final generation evidence must load the exact output directory returned by
the creator, execute the requested task, generate at least one new token, and
include the command and output. Do not validate one directory and return a
different cached or previously generated artifact.

## Rules

- Do not upload the tiny model to Hugging Face.
- Do not edit installed packages or the virtual environment.
- Do not modify system files or system-wide package installations.
- Use a deterministic seed where supported.
- Avoid original weight downloads and large generated artifacts.
- Never commit machine-specific absolute paths.
- Cache repository-test fixtures so test collection does not rebuild them
  unnecessarily.

## Report

Report the output directory, script path, configuration comparison, parameter
count, exact generation command, generated output, and any dependency or
remote-code constraints.
