# Inference Validation Templates

Adapt these templates using the model-analysis report. Keep generation
deterministic and compare the same artifact, processor, inputs, and settings.

## Text generation

```python
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_dir = "output_dir"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = OVModelForCausalLM.from_pretrained(model_dir, device="CPU", trust_remote_code=True)
inputs = tokenizer("What is the capital of France?", return_tensors="pt")
inputs.pop("token_type_ids", None)
output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Image-text generation

```python
from PIL import Image
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

model_dir = "output_dir"
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = OVModelForVisualCausalLM.from_pretrained(model_dir, device="CPU", trust_remote_code=True)
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
print(processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

If the remote model documents separate tokenizer/image-processor inputs, use
its actual `preprocess_inputs` contract rather than forcing `AutoProcessor`.

## Text-to-video

```python
from diffusers.utils import export_to_video
from optimum.intel import OVLTXPipeline

pipe = OVLTXPipeline.from_pretrained("output_dir", device="CPU")
frames = pipe(
    prompt="A koala eating eucalyptus leaves in daylight.",
    width=704,
    height=480,
    num_frames=121,
    num_inference_steps=30,
).frames[0]
export_to_video(frames, "output.mp4", fps=24)
```

## Accuracy diagnosis

When generated output differs, compare the first divergent boundary:
preprocessing tensors, component outputs, merged embeddings, cache/position
inputs, logits, then token IDs. Check effective nested `dtype`/`torch_dtype`
and internal input casts. A semantic WWB score does not replace component or
repository-test validation.
