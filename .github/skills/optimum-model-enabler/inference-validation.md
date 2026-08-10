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

## Precision and device diagnosis

When CPU or Intel GPU inference fails with an element-type mismatch such as
`bf16 != f32`, inspect the failing boundary before changing model code:

1. Query `ov::device::capabilities` for the target device and record the
   compiled model's effective `ov::hint::inference_precision`.
2. Inspect the OpenVINO IR port types and the runtime tensor types on both sides
   of the first failing operation. Distinguish FP16 (`f16`) from BF16 (`bf16`).
3. Reproduce with `ov::hint::execution_mode(ACCURACY)` or an explicit supported
   `ov::hint::inference_precision` as a diagnostic control. Intel GPU commonly
   supports `f32` and `f16` inference hints; never force `bf16` without verified
   device support.

   When validating through an Optimum Intel model class, pass the equivalent
   runtime property through `ov_config` and select the target device explicitly:

   ```python
   from optimum.intel.openvino import OVModelForCausalLM

   model = OVModelForCausalLM.from_pretrained(
       export_dir,
       device="GPU",
       ov_config={"INFERENCE_PRECISION_HINT": "f16"},
   )
   ```

   Adapt the `OVModel...` class to the requested task. `f16` means FP16,
   `bf16` means BF16, and `f32` means FP32; never use the invalid spelling
   `fb16`. First reproduce without a hint. For the FP16 GPU acceptance run, use
   the default configuration or the supported `f16` hint. Retry with `f32` only
   as a diagnostic control to determine whether reduced-precision execution
   causes the failure; do not report that control as FP16 GPU evidence. Use
   `bf16` only when the target device reports BF16 capability. Record whether
   each hint was accepted and whether it changed the failure. The hint controls
   floating-point execution of uncompressed operations; it does not change an
   INT8/INT4 weight-compression request into an FP32/FP16 export.
4. Treat precision properties as hints for internal primitive execution, not as
   automatic conversion of incompatible model inputs, outputs, or merge
   boundaries. If a hint changes the result, still locate and explain the first
   divergent or incompatible boundary.
5. Fix the narrow owner of the mismatch: preserve the intended floating type in
   export, normalize the runtime tensor at the real interface, or add an
   explicit OpenVINO `Convert` only where the consumer contract requires it.
   Do not cast every tensor globally or disable quantization to hide the error.
6. Re-run the same real-model FP16, INT8, and INT4 checks on CPU and Intel GPU,
   and add focused regression coverage for the failing device/precision path.

For INT8 and INT4 models, remember that compressed operations are quantized
while uncompressed operations still use floating-point tensors. Diagnose the
floating residual path separately when a quantized model exposes an
`f16`/`bf16`/`f32` mismatch.
