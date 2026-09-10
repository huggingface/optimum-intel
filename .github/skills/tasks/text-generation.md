# Text Generation

Use these instructions when the requested task is `text-generation`.

## Model analysis

Use the documented causal-language-model class for the architecture rather than
a multimodal or pipeline class. A typical in-library inspection starts with:

```python
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "<model_id>"
trust_remote_code = False
config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=trust_remote_code,
)
model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=trust_remote_code,
)

for name, module in model.named_modules():
    kind = type(module).__name__
    if any(key in kind for key in ("Attention", "Norm", "MLP", "MoE", "Expert")):
        print(name, kind)
```

Use a different documented class when the model architecture requires it.

## Repository tests

For decoder-only generation support, inspect and update the applicable files:

- `tests/openvino/test_decoder.py` for generation integration;
- `tests/openvino/test_export.py` for model/API export coverage;
- `tests/openvino/test_exporters_cli.py` for CLI export coverage;
- `tests/openvino/test_quantization.py` when compression or quantization changes;
- `tests/openvino/utils_tests.py` for the tiny-model fixture and expected IR details.

Run targeted pytest selections for every modified test file and confirm that
each command selects at least one test.

## Tiny-model validation

Tokenize a real prompt and run deterministic `model.generate()` with multiple
new tokens. Decode the generated continuation.

Before accepting the fixture:

1. Run a forward pass on representative inputs and assert that logits contain
   no NaN or Inf values.
2. Check that logits have non-zero finite variance and are not uniformly zero.
3. Generate multiple new tokens for at least two distinct prompts.
4. Reject a fixture when all continuations collapse to zeros, one repeated
   token, or the same constant sequence across every input.
5. Record the finite-value check, logits variance, and generated token IDs.

Example numerical checks:

```python
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits.float()
assert torch.isfinite(logits).all(), "Tiny model produced NaN/Inf logits"
assert logits.std().item() > 0, "Tiny model logits collapsed to a constant"
```

If outputs collapse, do not weaken the comparison test. Use a less aggressive
reduction, retain more capacity, or repair architecture-specific
initialization while preserving the real architecture and any tied embeddings.

When repairing a below-threshold HF-vs-OpenVINO result, reproduce the exact
benchmark inputs and deterministic generation settings, compare generated
token IDs, and inspect the top-1/top-2 logit margin at the first divergence
against the observed backend numerical error. Regenerate ground truth from the
exact saved fixture and rerun the failing comparison.

## End-to-end validation

```python
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_dir = "output_dir"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device="CPU",
    trust_remote_code=True,
)
inputs = tokenizer("What is the capital of France?", return_tensors="pt")
inputs.pop("token_type_ids", None)
output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Keep generation deterministic and compare Hugging Face and OpenVINO using the
same artifact, tokenizer, prompt, generation settings, and decoded-token
boundary.
