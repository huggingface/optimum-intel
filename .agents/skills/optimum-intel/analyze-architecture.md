# Skill: Analyze Model Architecture for OpenVINO Export

**Trigger:** Called as Step 1 of `optimum_add_model_support` before writing any
export code. Also useful standalone when debugging an unknown export failure and
the root cause is unclear.

## Purpose

Systematically reverse-engineer a model's architecture from its `modeling_*.py`
and `configuration_*.py` files to determine:

1. What attention mechanism(s) it uses — and whether standard patching is enough
2. Whether it needs custom OpenVINO ops (`_ov_ops.py` via `ModuleExtension`)
3. What cache layout the KV/state cache uses
4. Whether it is a multimodal (VLM) model and how modality dispatch works
5. Which existing supported model in optimum-intel is the best template

Output: `arch_analysis.md` in the working directory.

---

## Step 1 — Load and Print the Config

```python
from transformers import AutoConfig

MODEL_ID = "<model_id>"
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
print(f"model_type: {cfg.model_type}")
print(f"architectures: {cfg.architectures}")
print(f"hidden_size: {getattr(cfg, 'hidden_size', 'N/A')}")
print(f"num_hidden_layers: {getattr(cfg, 'num_hidden_layers', 'N/A')}")
print(f"num_attention_heads: {getattr(cfg, 'num_attention_heads', 'N/A')}")
print(f"num_key_value_heads: {getattr(cfg, 'num_key_value_heads', 'N/A')}")

# Detect hybrid / SSM / MoE markers in config attributes
import dataclasses, inspect
config_dict = cfg.to_dict()
interesting_keys = [k for k in config_dict if any(
    kw in k.lower() for kw in
    ['layer_type', 'ssm', 'mamba', 'rwkv', 'moe', 'expert', 'conv',
     'recurrent', 'delta', 'vision', 'image', 'video', 'mm_', 'vlm',
     'audio', 'cross_attn', 'sliding', 'hybrid']
)]
for k in interesting_keys:
    print(f"  {k}: {config_dict[k]}")
```

Key indicators to look for:
- `layer_types` / `layer_type` list → hybrid model (some layers are SSM/recurrent)
- `num_experts` / `moe_*` fields → Mixture of Experts
- `vision_config` / `image_token_id` / `video_token_id` → VLM
- `conv_kernel` / `causal_conv*` / `ssm_state_size` → SSM/State Space model
- `sliding_window` → sliding window attention (affects cache)

---

## Step 2 — Inspect the Modeling File

Find and read the primary modeling file from the installed transformers:

```bash
MODEL_TYPE="<model_type>"
python -c "
import importlib, inspect
# Try standard naming convention
for mod_name in [
    f'transformers.models.{MODEL_TYPE}.modeling_{MODEL_TYPE}',
    f'transformers.models.{MODEL_TYPE.replace(\"_\",\"\")}.modeling_{MODEL_TYPE.replace(\"_\",\"\")}',
]:
    try:
        mod = importlib.import_module(mod_name)
        print(inspect.getfile(mod))
        break
    except ImportError:
        pass
" 2>/dev/null
```

Once you have the file path, search for these patterns:

```bash
MODELING_FILE="<path from above>"

# 1. Attention class names
grep -n "class.*Attention" "$MODELING_FILE" | head -20

# 2. Cache usage — what type of cache does forward() accept/return?
grep -n "DynamicCache\|HybridCache\|StaticCache\|SlidingWindowCache\|MambaCache" "$MODELING_FILE" | head -20

# 3. SSM / recurrent patterns
grep -n "GatedDeltaNet\|Mamba\|RWKV\|RecurrentAttention\|CausalConv1d\|ssm_state" "$MODELING_FILE" | head -20

# 4. MoE patterns
grep -n "num_experts\|TopK\|MoE\|MixtureOfExperts\|router\|gate.*expert\|expert.*mlp" "$MODELING_FILE" | head -20

# 5. VLM patterns
grep -n "image_token_id\|video_token_id\|mm_token_type_ids\|cu_seqlens\|vision_model\|get_rope_index" "$MODELING_FILE" | head -30

# 6. forward() signature — what inputs are expected?
grep -n "def forward" "$MODELING_FILE" | head -20
```

---

## Step 3 — Classify the Architecture

Based on Steps 1–2, assign your model to one or more categories:

### A. Standard Transformer (dense attention only)

**Markers:** Only `*Attention` classes, standard `DynamicCache`, no SSM markers.
**Example models:** LLaMA, Qwen3 (pure text), Gemma2 (without recurrence).
**OV export complexity:** Low — model_config + optional patcher for control flow.
**Custom OV ops:** Not needed.

### B. Mixture of Experts (MoE)

**Markers:** `num_experts`, router/gate logic, multiple expert MLP blocks.
**Example models:** Mixtral, DeepSeek, Afmoe, Qwen2-MoE.
**OV export complexity:** Medium — patch the expert dispatch loop.
**Custom OV ops:** Not needed; use `torch.bmm` over stacked weight tensors.
**Patcher strategy:** Replace `for expert in experts` with batched `torch.stack` + `bmm`.

### C. SSM / Recurrent (pure state-space)

**Markers:** `Mamba`/`RWKV`/`GatedDeltaNet` class, SSM state tensors, no standard KV cache.
**Example models:** Mamba, RWKV, Falcon-Mamba.
**OV export complexity:** High — recurrent loop cannot be traced directly.
**Custom OV ops:** Required → `ModuleExtension` + OV Loop op (see `optimum_add_custom_ov_op`).

### D. Hybrid (SSM + standard attention)

**Markers:** `layer_types` list mixing attention and SSM/recurrent layers,
`HybridCache` or separate conv/recurrent state tensors alongside KV cache.
**Example models:** Qwen3.5, Qwen3-Next (GatedDeltaNet), Zamba, Jamba, GraniteMoeHybrid.
**OV export complexity:** High — need both standard attention config AND custom OV ops
for the recurrent layers.
**Custom OV ops:** Required for the recurrent layers → `ModuleExtension`.
**Reference PRs:** PR #1523 (Qwen3-Next CausalConv1D/GatedDeltaNet), PR #1561 (Zamba).

### E. Vision Language Model (VLM)

**Markers:** `vision_config`, `image_token_id`, vision encoder classes, `get_rope_index`
with vision-specific arguments.
**Example models:** LLaVA, Qwen2VL, Qwen3VL, Qwen3.5 (multimodal variant), InternVL.
**OV export complexity:** Medium‒High — model must be split into subcomponents for export;
inference pipeline needs multimodal dispatch.
**Custom OV ops:** Depends on language backbone (if SSM/hybrid → yes).
**Reference PRs:** PR #1551 (Qwen3VL), existing `OVQwen2VLForCausalLM` class.

**VLM subcomponent split pattern:**
```
vision_patch_embed  → handles pixel_values → patch embeddings
vision_pos_embed    → positional embeddings for image patches
vision_merger       → merges visual features into language space  
language_model      → main autoregressive language model
```
Check the upstream VLM config in `model_configs.py` for the exact subcomponent
names and input specification.

---

## Step 4 — Identify Patching Requirements

For each recurrent/SSM module found in Step 2, determine:

```python
# Find the forward signature of the problematic module
import inspect
from transformers.models.<model_type>.modeling_<model_type> import <RecurrentClass>

print(inspect.getsource(<RecurrentClass>.forward))
```

Ask:
1. Does `forward()` contain a Python `for` loop over time steps? → **OV Loop op needed**
2. Does it use dynamic shapes that change per step? → **ModuleExtension needed**
3. Can it be rewritten as a single vectorised torch op? → **Standard patcher sufficient**
4. Does it reference external state that must persist between decode steps? → **Stateful model**

---

## Step 5 — Find the Closest Template in optimum-intel

```python
from optimum.exporters.tasks import TasksManager

# List all supported model_types for OpenVINO export
try:
    supported = TasksManager._SUPPORTED_MODEL_TYPE
    print(sorted(supported.keys()))
except AttributeError:
    # Newer API
    from optimum.exporters.openvino.model_configs import OV_EXPORT_CONFIGS
    print(sorted(OV_EXPORT_CONFIGS.keys()))
```

Comparison heuristic (choose the best match):
| Your architecture | Best template model_type |
|---|---|
| Standard dense | `qwen3`, `llama`, `mistral` |
| MoE | `qwen2_moe`, `mixtral`, check PR #1569 (Afmoe) |
| SSM/recurrent | `mamba`, `falcon_mamba` |
| Hybrid (SSM+attn) | `qwen3_moe` with SSM patch, PR #1523 (Qwen3-Next GatedDeltaNet) |
| VLM (no SSM) | `qwen2_vl`, `llava`, `internvl_chat` |
| VLM + hybrid | PR #3 (Qwen3.5) — study `Qwen3_5OpenVINOConfig` |

```bash
# Inspect the template config class
cd /tmp/optimum-intel
grep -n "class <TemplateModel>OpenVINOConfig" optimum/exporters/openvino/model_configs.py
```

---

## Step 6 — Document the Analysis

Write `arch_analysis.md`:

```markdown
# Architecture Analysis: <model_id>

## Summary
- model_type: <value>
- Architecture category: <A/B/C/D/E from Step 3>
- transformers support: <version or git-HEAD required>

## Attention / Layer types
<list layers with their type>

## Cache layout
<DynamicCache / HybridCache with conv+recurrent+KV / custom>

## Custom ops required
<yes/no — list any recurrent modules needing ModuleExtension>

## VLM details (if applicable)
<subcomponent split, modality dispatch method, key parameters>

## Selected template
<model_type + file reference>

## Work items
1. [ ] Step 0: Transformers version ← (done or todo)
2. [ ] Step 2: model_configs.py — <ClassName>
3. [ ] Step 2.5: _ov_ops.py — <OpName> (if custom ops needed)
4. [ ] Step 3: model_patcher.py — <PatcherName>
5. [ ] Step 3.5: modeling_visual_language.py (if VLM)
6. [ ] Step 3.5: modeling_decoder.py (if full-context mask needed)
7. [ ] Tiny model, tests, docs
```

This document is the contract between analysis and implementation — reference it
in every subsequent experiment log entry.
