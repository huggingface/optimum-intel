# Text to Video

Use these instructions when the requested task is `text-to-video`.

## Model analysis

Inspect the documented text-to-video pipeline and every exported component
required by that pipeline. Record the text encoder, denoising/transformer,
decoder/VAE, scheduler-related inputs, expected frame dimensions, and any
task-specific preprocessing or output contract.

Use the documented pipeline class rather than forcing a language-model
`generate()` interface.

## Repository tests

Update the repository tests that cover the actual pipeline/export path and any
changed compression or quantization behavior. Run targeted pytest selections
for every modified test file and confirm that each command selects at least one
test.

## Tiny-model validation

Execute the real task pipeline from a text prompt and verify that it produces
finite, non-empty frame output with the expected shape. Loading, saving,
conversion, or a component-only forward pass is not sufficient.

Use deterministic settings where the pipeline supports them and compare the
same prompt, preprocessing, inference parameters, and output boundary between
the reference and OpenVINO paths.

## End-to-end validation

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

Verify that the pipeline produces actual frames and that the video artifact is
created successfully. Adapt the pipeline class and parameters from the
model-analysis report when the requested architecture uses a different
documented text-to-video contract.
