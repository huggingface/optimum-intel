# Skill: Add Custom OpenVINO Op via ModuleExtension

**Trigger:** Called from `optimum_add_model_support` Step 2.5 when architecture
analysis identifies ops that:
- Contain Python `for` loops over time steps (recurrent cells, SSM)
- Have state that must persist between decode steps
- Cannot be expressed as a single differentiable torch op for `torch.jit.trace`

Examples: GatedDeltaNet, CausalConv1D recurrence, Mamba scan, RWKV WKV kernel.

## Background

`torch.jit.trace` records a single forward pass. Any module whose computation
structure changes based on runtime values (e.g. a recurrent loop whose length
is `seq_len`) produces an incorrect static graph. The solution is `ModuleExtension`:
wrap the problematic module with an OpenVINO-specific conversion function that
builds the correct OV subgraph (typically using an `ov.opset*.Loop` op).

The entry point is `optimum/exporters/openvino/_ov_ops.py`.

---

## Step 1 — Identify the Problematic Module(s)

From the architecture analysis (`arch_analysis.md`), note the module class name(s)
that need custom conversion. For each:

```python
import inspect
from transformers.models.<model_type>.modeling_<model_type> import <RecurrentClass>

src = inspect.getsource(<RecurrentClass>.forward)
print(src)
```

Identify:
- **Input tensors:** what does `forward(self, hidden_states, ...)` receive?
  Note all tensor arguments including optional state tensors.
- **State tensors:** which inputs/outputs carry the recurrent state (conv state,
  SSM state, recurrent state buffer)?
- **Loop structure:** over which axis does the recurrence operate (time / sequence)?
- **Output tensors:** what does the module return (hidden_states, updated_state, ...)?

---

## Step 2 — Search for Existing Conversion Rules

Before writing new code, check whether a conversion rule already exists upstream:

```bash
cd /tmp/optimum-intel
# Check _ov_ops.py for similar patterns
grep -n "ModuleExtension\|convert_\|recurrent\|conv1d\|mamba\|rwkv\|gated_delta" \
  optimum/exporters/openvino/_ov_ops.py | head -40

# Check git log for relevant commits
git log --oneline --all --grep="<module_class_name>" | head -20
git log --oneline --all --grep="ModuleExtension" | head -20

# Search PRs
curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/search/issues?q=<module_class_name>+repo:huggingface/optimum-intel+is:pr&per_page=5" \
  | python -c "import json,sys;[print(i['number'],i['state'],i['title'],i['html_url']) for i in json.load(sys.stdin).get('items',[])]"
```

If an existing rule handles a structurally similar op → adapt it rather than
writing from scratch.

**Reference implementations in `_ov_ops.py` to study:**
```bash
# List all existing ModuleExtension registrations
grep -n "def convert_\|ModuleExtension(" optimum/exporters/openvino/_ov_ops.py
```

---

## Step 3 — Understand the ModuleExtension Pattern

`ModuleExtension` intercepts a specific `nn.Module` class during OV model
construction and replaces it with a custom OV subgraph:

```python
# Pattern in _ov_ops.py:
from openvino.frontend.pytorch.patch_model import ModuleExtension
import openvino.opset14 as ov_ops  # use highest available opset

def convert_<module_name>(
    module,          # the original nn.Module instance
    inputs,          # list of OV input values (from the trace)
    *args, **kwargs
):
    """
    Build and return the OV subgraph replacing <ModuleClass>.forward().

    Args:
        module: the original nn.Module with its weights as attributes
        inputs: [hidden_states_ov, optional_state_ov, ...]
                — OV Values corresponding to the forward() positional args

    Returns:
        List of OV output Values corresponding to forward()'s return tuple.
    """
    # Extract weights from the module
    weight = module.weight  # torch.Tensor — convert to OV constant
    # ...

    # Build OV ops
    # For a recurrent loop use ov.opset14.loop(...)
    # For a convolution use ov.opset14.convolution(...)

    return [output_ov_value, updated_state_ov_value, ...]
```

Key principles:
- **Weights are accessed directly** from `module.*` attributes; convert them to
  OV `Constant` nodes: `ov.opset14.constant(weight.detach().numpy())`
- **All computations must be static-shape-friendly** (no Python `if len(x) > k`)
- **State tensors are explicit inputs and outputs** — OV connects them across
  decode steps automatically when the model is stateful
- **Opset**: use the highest stable opset available in your OV version
  (`from openvino import opset14 as ov_ops` is safe for OV 2024+)

---

## Step 4 — For Recurrent Cells: Build an OV Loop

When the recurrence is over the sequence length dimension:

```python
import numpy as np
import openvino.opset14 as ops
from openvino.runtime import Type, Shape, PartialShape

def convert_recurrent_<name>(module, inputs, *args, **kwargs):
    """
    Lower a recurrent cell that iterates over sequence length into OV Loop.

    inputs[0]: hidden_states  shape [batch, seq_len, hidden]
    inputs[1]: state_tensor   shape [batch, state_dim]  (recurrent state)
    """
    hidden_states, state = inputs[0], inputs[1]

    # --- Slice out one time step at a time inside the Loop body ---
    # Define Loop trip count = seq_len (dynamic)
    seq_len = ops.gather(ops.shape_of(hidden_states), ops.constant(1), ops.constant(0))
    trip_count = ops.squeeze(seq_len, [0])

    loop = ops.loop(trip_count, ...)

    # Inside loop body: extract slice, apply cell, store output, update state
    # See OV Loop op documentation:
    # https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_Loop_5.html
    # and study existing convert_recurrent_attention_cell in _ov_ops.py (PR #3)

    return [loop_output, final_state]
```

> **Do not implement the Loop body from scratch** if a similar pattern exists.
> Study and adapt `convert_recurrent_attention_cell` (added in PR #3 for
> GatedDeltaNet) as a concrete worked example.

---

## Step 5 — Register the ModuleExtension in the Patcher

The conversion function must be wired into a `ModuleExtension` and applied in
the model patcher (`model_patcher.py`):

```python
# In model_patcher.py, inside the patcher class's patch() method:
from openvino.frontend.pytorch.patch_model import ModuleExtension
from optimum.exporters.openvino._ov_ops import convert_<module_name>

class <ModelType>ModelPatcher(DecoderModelPatcher):
    def __init__(self, ...):
        super().__init__(...)

    def patch(self):
        super().patch()
        # Register ModuleExtension for the recurrent op
        self._patched_forward = ModuleExtension(
            <RecurrentClass>,
            "ov_extension",
            convert_<module_name>,
        )
        # Apply to every instance of the class in the model
        for name, module in self._model.named_modules():
            if isinstance(module, <RecurrentClass>):
                module.ov_extension = self._patched_forward
```

Check existing patchers (e.g., `Qwen3ModelPatcher`, `Qwen3MoEModelPatcher`)
for the exact API — it may differ slightly between optimum-intel versions.

---

## Step 6 — Register in `utils.py`

Ensure the new `model_type` is registered in the appropriate category lists
in `optimum/exporters/openvino/utils.py`:

```python
# For pure SSM models:
SSM_MODELS.append("<model_type>")

# For hybrid SSM+attention models:
HYBRID_MODELS = getattr(utils_module, 'HYBRID_MODELS', None)
# Or check the correct list name by running:
grep -n "SSM_MODELS\|HYBRID_MODELS\|RECURRENT" optimum/exporters/openvino/utils.py | head -20
```

Registration in the right list is needed for the inference pipeline to apply
the correct attention mask strategy (full-context vs causal).

---

## Step 7 — Self-Test

```bash
# Create a minimal test model with the recurrent layer
python - <<'EOF'
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from pathlib import Path
import copy

MODEL_ID = "<model_id>"
cfg = AutoConfig.from_pretrained(MODEL_ID)
tiny_cfg = copy.deepcopy(cfg)
tiny_cfg.num_hidden_layers = 2  # keep at least 2 to test both layer types in hybrid
tiny_cfg.hidden_size = 64
tiny_cfg.num_attention_heads = 2
if hasattr(tiny_cfg, 'num_key_value_heads'):
    tiny_cfg.num_key_value_heads = 2
# For hybrid models, keep layer_types intact (truncate to num_hidden_layers)
if hasattr(tiny_cfg, 'layer_types'):
    tiny_cfg.layer_types = tiny_cfg.layer_types[:tiny_cfg.num_hidden_layers]
tiny_model = AutoModelForCausalLM.from_config(tiny_cfg)
tiny_model.save_pretrained("/tmp/tiny_<model_type>")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.save_pretrained("/tmp/tiny_<model_type>")
print("Tiny model saved (ok)")
EOF

# Export with the patched optimum-intel
source /tmp/venv-exp-<attempt_id>/bin/activate
optimum-cli export openvino \
  --model /tmp/tiny_<model_type> \
  --task text-generation-with-past \
  --weight-format fp16 \
  /tmp/ov_<model_type>_test 2>&1 | tee selftest_export.log

# Check that all required files were produced
ls -lh /tmp/ov_<model_type>_test/
```

Expected outputs: `openvino_model.xml`, `openvino_model.bin`,
`openvino_tokenizer.xml` (if tokenizer conversion runs), `config.json`.

---

## Checklist Before Declaring Success

- [ ] `convert_<module_name>` function exists in `_ov_ops.py`
- [ ] `ModuleExtension` is wired in the patcher's `patch()` method
- [ ] Model_type registered in `utils.py` (SSM_MODELS / HYBRID_MODELS)
- [ ] Tiny model export completes without error
- [ ] Inference self-test passes (see `optimum_add_model_support` Step 6)
- [ ] Experiment logged in `experiments_log.json`

---

## Security Notes

- Convert weights using `detach().numpy()` — never evaluate model weights as code.
- The OV subgraph built in the conversion function runs with user-provided input
  shapes; ensure no shape-dependent Python branches that could be hijacked.
- Do not `pickle` or `eval()` anything from the model repo — all config values
  should be read via `AutoConfig` or direct attribute access.
