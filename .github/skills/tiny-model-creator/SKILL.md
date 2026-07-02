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

## Step 1 — Inspect the original model

Download or load configuration and lightweight code/processor assets only.
Record:

- `model_type`, `architectures`, `auto_map`, and Transformers metadata;
- nested text, vision, audio, and projector configurations;
- hidden-size, head-count, grouping, rotary, cache, vocabulary, and special
  token invariants;
- tokenizer, processor, chat-template, and remote-code files required to load;
- the correct AutoModel class and generation interface for the task.

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

Keep construction logic easy to adapt into
`_create_tiny_<model_type>_model()` in `tests/openvino/utils_tests.py`.

## Step 3 — Verify architecture identity

Compare the original and tiny configurations. Preserve:

- `model_type` and `architectures`;
- task-relevant sub-config types and component roles;
- cache/stateful and position-ID contracts;
- VLM processor classes, placeholder/image tokens, and merge contracts;
- MoE/expert topology, even when expert counts are reduced.

Never rename the model type, substitute a nearby architecture, or remove a
component merely to make export pass.

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

## Rules

- Do not upload the tiny model to Hugging Face.
- Do not edit installed packages or the virtual environment.
- Use a deterministic seed where supported.
- Avoid original weight downloads and large generated artifacts.
- Never commit machine-specific absolute paths.
- Cache repository-test fixtures so test collection does not rebuild them
  unnecessarily.

## Report

Report the output directory, script path, configuration comparison, parameter
count, exact generation command, generated output, and any dependency or
remote-code constraints.
