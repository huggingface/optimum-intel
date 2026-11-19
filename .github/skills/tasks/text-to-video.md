# Text to Video

Use these instructions when the requested task is `text-to-video`.

## Model analysis

Inspect the documented text-to-video pipeline and every component listed in
the original diffusers `model_index.json`. Record the text encoder,
denoising/transformer, decoder/VAE, scheduler-related inputs, expected frame
dimensions, and any task-specific preprocessing or output contract.

Export **every** non-null component of `model_index.json`, including the ones
text-to-video does not call: both halves of each VAE (encoder and decoder,
video and audio), prompt enhancers and their processors, auxiliary heads.
Image-to-video, video-to-video and prompt enhancement load the same export, and
a component that is missing there cannot be recovered downstream. An entry of
`[null, null]` means the checkpoint does not have that component; skip it
rather than failing. Check that the OpenVINO pipeline object keeps every
exported component as an attribute: a missing one surfaces only later, as a
misleading `no attribute '_ov_model_*'` error from `save_pretrained` or weight
compression.

One export serves every pipeline class of the family (text-, image- and
video-conditioned). A behaviour fixed at trace time instead of exposed as an
input — a flag whose reference default may change between diffusers releases —
must be the one all of those pipelines need. Record such a choice in the IR's
`rt_info` so runtimes can tell export variants apart, and document which
reference setting it corresponds to.

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
will fail for a reason that has nothing to do with the export. OpenVINO CPU
lowers i64 shape arithmetic to i32, so dynamic length computations on long
token sequences can overflow and truncate the output without an error.

Record which tensors the pipeline computes on the host and feeds to the
models. Keep values the graph cannot represent in reduced precision outside
it: rotary angles grow to ~1e4 rad, so pass precomputed cos/sin as inputs
rather than computing them in a graph that may run in fp16/bf16.

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
changed compression or quantization behavior — `tests/openvino/test_diffusion.py`
for the pipelines, `test_quantization.py` for compression, with the expected
per-component int8 node counts in `utils_tests.py`. Run targeted pytest
selections for every modified test file and confirm that each command selects
at least one test.

Diffusion fixtures are Hub repositories under `optimum-intel-internal-testing`,
and their `ov` revision holds the reference IRs `test_irs.py` compares against.
A change to an exported graph means regenerating those reference IRs. CI's model
cache pins fixture revisions: a fixture updated on the Hub reaches CI only after
the cache workflow runs with `force_update`.

When an export bakes in a setting whose reference default differs between the
diffusers versions of the test matrix, pass the baked-in value to the reference
pipeline explicitly, and only when its `__call__` signature accepts it.

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

## Accuracy

Compare each component's outputs with the fp32 PyTorch reference on identical
inputs; the acceptance criterion is a max abs diff below `1e-2`. Locate a large
end-to-end difference at the first component that exceeds it.

Most diffusion checkpoints ship in bf16, which hides precision traps:

- fp32 parameters inside an otherwise bf16 checkpoint (AdaLN/modulation
  tables such as `scale_shift_table`) are rounded by the 16-bit load cast;
  this can dominate the transformer's error. Keep them fp32 at load.
- A text encoder stored as F32 may hold bf16 values; exporting it in fp32
  then costs memory and buys no accuracy. Check the values, not the dtype.
- OpenVINO's 16-bit tracing keeps 16-bit weights only for `nn.Linear`,
  `nn.Embedding` and `Conv1D` (exact class); convolutions and subclasses are
  upcast to fp32. That costs size, not accuracy.

Text-encoder hidden states may differ by O(1) at padded positions while the
conditioning is unaffected. Mask the padding, or judge the text encoder by the
output of the model that consumes it.

## End-to-end validation

Use the model card's recommended settings. Distilled checkpoints rely on a
custom sigma schedule and disabled guidance; overriding `num_inference_steps`
replaces that schedule, so do not copy the step count from the example below.

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

For the full-size model, score the export with WWB (`--model-type
text-to-video`) against ground truth generated by the diffusers pipeline with
the same settings. Inspect the per-prompt scores: video generation is chaotic,
so a few prompts can dominate a low average even when the export is exact.
Check whether the low-scoring prompts are out of the model's distribution
(e.g. sizes it was not trained at), and whether their component-level error is
actually larger than the others', before blaming the export. WWB's runner passes `num_inference_steps`, which replaces a distilled
schedule. Run long jobs under `nohup /usr/bin/time -v` and record the peak RSS;
a full-size bf16 export can need over 100 GB.

## Weight compression

Verify that INT8/INT4 compression leaves modulation and AdaLN parameters in
floating point, and report the output shift against the uncompressed export
alongside the size reduction.
