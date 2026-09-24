# Image Text to Text

Use these instructions when the requested task is `image-text-to-text`.

## Model analysis

Inspect the complete multimodal contract, including:

- vision and language components;
- projector, merger, resampler, and position submodels where present;
- processor/tokenizer assets and chat template;
- image placeholder/special tokens and merge contracts;
- task-specific forward inputs and exported component boundaries;
- remote-code requirements and the documented visual causal model class.

Do not force a causal-language-only class onto a multimodal architecture.

## Repository tests

Cover all exported VLM submodels and update relevant architecture, remote-code,
video, compression, and preprocessing matrices. Run targeted tests from every
modified test file. A `-k` command selecting zero tests is not evidence.

Inspect and update every applicable VLM test matrix explicitly:

- `tests/openvino/test_export.py`: register the architecture with
  `OVModelForVisualCausalLM` where full VLM export is supported;
- `tests/openvino/test_seq2seq.py`: update
  `OVModelForVisualCausalLMIntegrationTest.SUPPORTED_ARCHITECTURES`,
  `SUPPORT_VIDEO`, every applicable `REMOTE_CODE_MODELS` collection,
  skip/unsupported sets, and model-specific preprocessing branches;
- `tests/openvino/test_exporters_cli.py`: cover the model/task pair, tokenizer
  export expectations, compression, and 4-bit cases when supported;
- `tests/openvino/test_quantization.py`: update auto-compression coverage and
  expected quantized submodel counts when applicable;
- `tests/openvino/utils_tests.py`: provide the model fixture and expected IR
  details for language, text embedding, vision embedding, projector, merger,
  resampler, and position submodels that the architecture actually exports.

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

At minimum for VLM source changes, run the relevant selections from:

```bash
python -m pytest -v tests/openvino/test_export.py -k <model_type>
python -m pytest -v tests/openvino/test_seq2seq.py -k <model_type>
python -m pytest -v tests/openvino/test_exporters_cli.py -k <model_type>
python -m pytest -v tests/openvino/test_quantization.py -k <model_type>
```

Run the quantization selection when quantization tables or behavior changed.

## Tiny-model validation

Process a real image and text prompt together, apply the architecture's actual
processor/chat-template contract, and run deterministic generation with
multiple new tokens. Decode only newly generated tokens.

Preserve VLM processor classes, placeholder/image tokens, component identities,
and merge contracts in the tiny fixture.

Before accepting the fixture:

1. Verify task-relevant logits and intermediate outputs contain no NaN or Inf.
2. Check that logits have non-zero finite variance and are not uniformly zero.
3. Generate continuations for at least two distinct image/text inputs.
4. Reject a fixture when outputs collapse to zeros, a repeated token, or the
   same constant continuation across every input.
5. Record generated token IDs and the numerical-validity evidence.

When repairing a below-threshold HF-vs-OpenVINO result, reproduce the exact
image, prompt, processor behavior, and deterministic generation settings.
Compare generated token IDs and inspect the first divergent boundary, including
preprocessing tensors, component outputs, merged embeddings, logits, and token
IDs.

## End-to-end validation

```python
from PIL import Image
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

model_dir = "output_dir"
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = OVModelForVisualCausalLM.from_pretrained(
    model_dir,
    device="CPU",
    trust_remote_code=True,
)
image = Image.new("RGB", (224, 224), "white")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(
    processor.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
)
```

If the remote model documents separate tokenizer/image-processor inputs, use
its actual `preprocess_inputs` contract rather than forcing `AutoProcessor`.

Keep generation deterministic and compare Hugging Face and OpenVINO using the
same image, prompt, preprocessing, generation settings, and decoded-token
boundary. A component-only forward pass is not end-to-end validation.
