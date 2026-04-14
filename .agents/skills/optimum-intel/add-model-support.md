# Skill: Add Full New Model Support

**Trigger:** User asks to add complete support for a new model architecture in optimum-intel.
Also triggered when `error_class=unknown_arch_transformers_too_old` or
`requires_optimum_new_arch=true` — meaning the architecture is absent from both
transformers (or needs upgrade) AND optimum-intel.

## Prerequisites

- Run **optimum_bootstrap** skill first.
- Study reference PRs for patterns relevant to your architecture type:
  ```bash
  cd /tmp/optimum-intel
  # Canonical full support example (Afmoe):
  git fetch origin pull/1569/head:pr-1569
  git diff main...pr-1569 --name-status

  # GatedDeltaNet / CausalConv1D (hybrid linear+standard attention, SSM-like):
  git fetch origin pull/1523/head:pr-1523  # Qwen3-Next
  git diff main...pr-1523 --name-status

  # VLM with vision encoder (multimodal):
  git fetch origin pull/1551/head:pr-1551  # Qwen3VL
  git diff main...pr-1551 --name-status
  ```
  Choose the reference PR that best matches your model's architecture type.

## Steps

Execute in order:

### 0. Resolve Transformers Version

Before any export or analysis, ensure the correct transformers version is active.

```bash
# Check if the model_type is recognised by installed transformers
python -c "
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained('$MODEL_ID', trust_remote_code=False)
    print(f'OK: model_type={cfg.model_type}')
except Exception as e:
    print(f'FAIL: {e}')
"
```

If the check fails:
1. Follow the **Transformers Version Handling** escalation ladder in
   `optimum_debug_export.md` (PyPI latest → git-HEAD → source patch via
   `optimum_patch_transformers` skill).
2. Once transformers loads the config successfully, continue to Step 1.
3. Record the working `transformers_install` in the manifest for subsequent agents.

> Do not proceed to architecture analysis until `AutoConfig.from_pretrained`
> succeeds — all downstream analysis depends on being able to inspect the config.

### 1. Model Architecture Analysis

**Do not skip this step.** Understanding the architecture fully before writing
any code prevents misdirected patching efforts.

→ Invoke **`optimum_analyze_architecture`** skill
(`skills/optimum-intel/analyze-architecture.md`) for the systematic analysis.

The skill will produce a structured report covering:
- Model family and attention type (standard / GQA / MoE / SSM / hybrid)
- Cache layout (`DynamicCache` vs custom hybrid or SSM cache)
- Custom ops requiring `ModuleExtension` or special conversion rules
- Whether the model is a VLM (multimodal: vision encoder + language model)
- Closest existing supported model in optimum-intel to use as template

Record the analysis output as `arch_analysis.md` before proceeding.

### 2. Update `optimum/exporters/openvino/model_configs.py`

Add a new config class inheriting from the closest existing one.
→ Use **optimum_create_model_config** skill for details.

Key things to capture from the architecture analysis:
- Hybrid cache inputs (conv states, recurrent states, KV cache) if applicable
- Multimodal submodel splits (vision patch embed / pos embed / merger / language)
- Correct task registration (`text-generation-with-past`, `image-text-to-text`, etc.)

### 2.5. Add Custom OpenVINO Ops (if architecture requires them)

Required when the architecture analysis (Step 1) identifies ops that
`torch.jit.trace` cannot handle via standard patching — e.g.:
- Recurrent cells (GatedDeltaNet, Mamba, RWKV, RetNet)
- CausalConv1D with stateful recurrence
- Any module where the computation must be expressed as an OV Loop op

→ Invoke **`optimum_add_custom_ov_op`** skill
(`skills/optimum-intel/add-custom-ov-op.md`) to add a conversion rule in
`optimum/exporters/openvino/_ov_ops.py` via `ModuleExtension`.

This step must come **before** the patcher (Step 3) because the patcher will
reference the `ModuleExtension` registered here.

Skip this step if the architecture only needs Python-level patching (control
flow removal, MoE vectorisation) without custom OV op registration.

### 3. Update `optimum/exporters/openvino/model_patcher.py` (if needed)

Add patching functions for `torch.jit.trace`-incompatible patterns. Principles:
- Replace Python control flow that depends on runtime tensor values with vectorised torch ops.
- Replace `for` loops over experts (MoE) with batched matrix multiplications.
- For SSM/recurrent cells: use the `ModuleExtension` registered in Step 2.5.
- Ensure all code paths produce the same graph regardless of input data.

### 3.5. Add VLM Inference Pipeline (if model is multimodal)

Required when the architecture analysis identifies a VLM pattern (vision encoder
+ language model with token-type-based or token-ID-based modality dispatch).

Add a new class in `optimum/intel/openvino/modeling_visual_language.py`:
- Subclass `_OVModelForVisualCausalLM` (or closest existing VLM class)
- Handle model-specific multimodal dispatch patterns:
  - Token-ID-based dispatch (e.g., Qwen2VL): `image_token_id` lookup in `input_ids`
  - Token-type-based dispatch (e.g., Qwen3.5): `mm_token_type_ids` from `image_token_id`
    and `video_token_id`
  - `cu_seqlens`-style packing → `attention_mask`-based unpacking if needed
- Override `get_rope_index` / vision embedding interpolation matching the model's API

Also check whether the language model backbone needs a `modeling_decoder.py` update:
- Full-context attention mask requirement (needed for SSM/hybrid models)
- Custom past key value handling

### 4. Create / Update Tests

| Test file | What to add |
|-----------|-------------|
| `tests/openvino/test_decoder.py` | Export and inference validation |
| `tests/openvino/test_export.py` | Export configurations |
| `tests/openvino/test_exporters_cli.py` | CLI tests |
| `tests/openvino/test_quantization.py` | Add to `SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION`, update `_ARCHITECTURES_TO_EXPECTED_INT8` |
| `tests/openvino/utils_tests.py` | Define test model IDs |

For VLM models, also add tests to `tests/openvino/test_modeling_basic.py`
(or the VLM-specific test file) covering the multimodal inference path.

### 5. Update Documentation

Add the new `model_type` (first letter capitalised) to `docs/source/openvino/models.mdx`.

### 6. Verify

```bash
cd /tmp/optimum-intel
pip install -e ".[tests]"
pytest tests/openvino/test_export.py -k "<model_type>" -v
pytest tests/openvino/test_decoder.py -k "<model_type>" -v
```

### 7. Signal tokenizer agent

After successful export + inference validation, emit:
```bash
echo "requires_tokenizer_check=true" >> "$GITHUB_OUTPUT"
echo "model_type=<model_type>" >> "$GITHUB_OUTPUT"
echo "transformers_install=<url_or_version_used>" >> "$GITHUB_OUTPUT"
```

This tells the orchestrator to invoke the tokenizers agent for tokenizer
conversion validation — completing the full enablement pipeline.

## Key Files in optimum-intel

| File | Purpose |
|------|---------|
| `optimum/exporters/openvino/model_configs.py` | Model config classes for OV export |
| `optimum/exporters/openvino/model_patcher.py` | Model patching for trace-safe conversion |
| `optimum/exporters/openvino/_ov_ops.py` | Custom OpenVINO conversion rules (ModuleExtension) |
| `optimum/exporters/openvino/utils.py` | Model type registrations (SSM_MODELS, MULTI_MODAL_TEXT_GENERATION_MODELS, etc.) |
| `optimum/exporters/tasks.py` | Task manager - model type ↔ task registration |
| `optimum/intel/openvino/modeling_decoder.py` | Decoder inference, attention mask handling |
| `optimum/intel/openvino/modeling_visual_language.py` | VLM inference classes |
| `tests/openvino/test_decoder.py` | Decoder model export + inference tests |
| `tests/openvino/test_export.py` | Export configuration tests |
| `tests/openvino/test_exporters_cli.py` | CLI export tests |
| `tests/openvino/test_quantization.py` | Quantisation workflow tests |
| `tests/openvino/utils_tests.py` | Test model IDs and helper constants |
| `docs/source/openvino/models.mdx` | Supported models documentation |

## Conventions

- **Patching safety:** All patches must produce identical torch graphs regardless of input data - no Python `if/for` that depends on tensor values.
- **Vectorise MoE:** Replace per-expert loops with batched `torch.bmm` over stacked weight matrices.
- **Tiny models for CI:** Always < 1 GB (`num_hidden_layers=1`, `hidden_size=64`, `num_attention_heads=2`).

## External References

- **torch.jit.trace docs:** https://pytorch.org/docs/stable/generated/torch.jit.trace.html
- **Optimum Intel docs:** https://huggingface.co/docs/optimum-intel/en/index
- **OpenVINO documentation:** https://docs.openvino.ai/2025/index.html
