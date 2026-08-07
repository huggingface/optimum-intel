# Inference Validation

Use this reference for common accuracy diagnosis after running the
task-specific end-to-end validation instructions supplied for the requested
task.

Keep validation deterministic and compare the same artifact, processor or
preprocessor, inputs, and settings.

## Accuracy diagnosis

When generated output differs, compare the first divergent boundary:
preprocessing tensors, component outputs, merged embeddings, cache/position
inputs, logits, then task outputs. Check effective nested `dtype`/`torch_dtype`
and internal input casts. A semantic WWB score does not replace component or
repository-test validation.
