# Text to Video

Use these instructions when the requested task is `text-to-video`.

## Model analysis

Inspect the documented text-to-video pipeline and every exported component
required by that pipeline. Record the text encoder, denoising/transformer,
decoder/VAE, scheduler-related inputs, expected frame dimensions, and any
task-specific preprocessing or output contract.

Record every modality the pipeline returns. `text-to-video` also covers
audio-video architectures whose output carries an `audio` tensor beside
`frames`, and whose audio VAE and vocoder are separately exported components.
Analyse and validate those components too; a report that stops at the video
branch is incomplete.

Record the pipeline's shape invariants, which are usually enforced by
flooring rather than by an error: `num_frames` is typically
`k * vae_temporal_compression_ratio + 1`, and height and width must divide by
the spatial compression ratio times the patch size. A request that violates
them is silently rounded, so an assertion written against the requested value
will fail for a reason that has nothing to do with the export.

Record which `__call__` arguments switch on extra denoising passes —
classifier-free guidance, spatio-temporal guidance, modality isolation — and
their default values **in the installed version**. These defaults change
between releases, and they decide what a default-settings comparison actually
exercises. A pass the exported graph cannot serve is the most common cause of
a large, otherwise unexplained reference mismatch.

Use the documented pipeline class rather than forcing a language-model
`generate()` interface.

## Repository tests

Update the repository tests that cover the actual pipeline/export path and any
changed compression or quantization behavior. Run targeted pytest selections
for every modified test file and confirm that each command selects at least one
test.

Gate every architecture entry and every version-dependent call argument on the
library version that introduced it. The test matrix runs more than one
diffusers and transformers version, and an argument that is required on the
newer one is a `TypeError` on the older.

Derive per-architecture call arguments inside the shared `generate_inputs`
helper rather than in individual tests, threading the architecture name into
it. One override then reaches every call site, including the shape,
reproducibility and static-shape tests that also invoke the pipeline.

When the exported model cannot serve a feature the reference pipeline enables
by default, assert the explicit error, and turn that feature off on **both**
sides in the comparison tests. Quietly falling back to the plain pass lets the
comparison pass while the outputs differ.

## Tiny-model validation

Execute the real task pipeline from a text prompt and verify that it produces
finite, non-empty frame output with the expected shape. Loading, saving,
conversion, or a component-only forward pass is not sufficient.

Check that the output is also **not constant**. A broken export very often
returns a uniform field, which satisfies a finite/non-empty check. Validate any
audio output the same way, but do not treat quiet audio as a failure on its
own: the audio level follows the prompt.

Size the fixture so the features under test are reachable. A feature selected
by block index — spatio-temporal guidance defaults to an index far beyond a
one-block fixture — is a silent no-op on a model that small, and a packed
per-layer text-encoder output needs more than one hidden layer before the
packing is exercised at all. Likewise keep `num_frames` above a single latent
frame, or the temporal path never runs.

Use deterministic settings where the pipeline supports them and compare the
same prompt, preprocessing, inference parameters, and output boundary between
the reference and OpenVINO paths. Pass a **fresh** generator to each call;
generators are consumed, so reusing one makes the second call diverge for
reasons unrelated to the export.

Build the fixture with the oldest supported transformers version, or normalise
the saved `tokenizer_class` afterwards. Newer versions write a renamed class
that older ones cannot load, which makes the fixture fail only on the older
leg of the matrix.

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

For an audio-video architecture, validate both branches of the output, and
write the artifact with the muxing helper the reference pipeline documents
rather than `export_to_video`, which drops the audio track:

```python
from diffusers.utils import encode_video
from optimum.intel import OVLTX2Pipeline

frame_rate = 24.0
pipe = OVLTX2Pipeline.from_pretrained("output_dir", device="CPU")
video, audio = pipe(
    prompt="A koala eating eucalyptus leaves, birds calling nearby.",
    width=768,
    height=512,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=30,
    output_type="np",
    return_dict=False,
)
encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="output.mp4",
)
```

Verify that the pipeline produces actual frames and that the video artifact is
created successfully. Adapt the pipeline class and parameters from the
model-analysis report when the requested architecture uses a different
documented text-to-video contract.
