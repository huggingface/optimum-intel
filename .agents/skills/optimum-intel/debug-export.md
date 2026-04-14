# Skill: Debug Export or Inference Failure

**Trigger:** Error log provided or asked to debug a failed export/inference.

## Prerequisites

- Run **optimum_bootstrap** skill first (venv infrastructure, upstream search context,
  experiment journal initialised).

---

## Step 0 — Consume Cross-Agent Artifacts

Before any diagnosis, check whether the tokenizers agent has already produced
an artifact that partially resolves the environment:

```bash
# Artifact is downloaded to ./tokenizers-artifact/ by the workflow.
if [ -f tokenizers-artifact/artifact-description.md ]; then
  cat tokenizers-artifact/artifact-description.md
  # Apply per instructions in artifact-description.md
  # (see Cross-Agent Artifact Consumption in optimum_intel.agent.md)
fi
```

## Step 1 — Upstream Pattern Search

Run upstream_search (from `optimum_bootstrap.md` Step 5) for your `model_type`
**before modifying any code**. Existing PRs or commits may already have the fix.

```bash
# Search optimum-intel and openvino_tokenizers for known fixes:
# → GitHub API PR search
# → git log --grep in local clones
# Study the most relevant result before proceeding.
```

## Step 2 — Classify the Error

Analyse the error traceback to identify the root cause category:

| Category | Signature | Action |
|---|---|---|
| **Missing model dependency** | `ImportError: This modeling file requires the following packages that were not found in your environment: <pkg>` | → **See Step 2a** — install & retry autonomously |
| **Unsupported model type** | `KeyError` from `TasksManager`, no config class | → `optimum_add_model_support` skill |
| **Tracing failure** | `torch.jit.trace` error, dynamic control flow | → add patcher in `model_patcher.py` |
| **Shape mismatch** | IR validation error, wrong input shapes | → fix dummy inputs in config class |
| **Missing op** | `NotImplemented` / op coverage gap in OpenVINO | → route to OV Orchestrator |
| **Dependency version** | `KeyError: '<model_type>'` in transformers | → **See Transformers Version Handling section below** |
| **OOM** | memory error during export | → use tiny model; suggest weight compression |

## Step 2a — Autonomous Missing-Package Recovery

When the error class is `missing_model_dependency`, **do not stop**. Follow this
recovery loop before any other debugging work:

### 1. Extract the required package name

```python
import re
error_text = "<paste ImportError line>"
m = re.search(
    r"not found in your environment:\s*([\w\-]+)",
    error_text,
)
pkg = m.group(1) if m else None
print("Missing package:", pkg)
```

### 2. Install the package in the active experiment venv and retry

```bash
# Activate the current experiment venv first
source /tmp/optimum-intel/venv-exp-<attempt_id>/bin/activate

pip install "<pkg>" 2>&1 | tee pip_install_<pkg>.log
echo "pip exit code: $?"

# Retry the export immediately
optimum-cli export openvino \
  --model <MODEL_ID> \
  --task <TASK> \
  --weight-format fp16 \
  ov_test_<pkg>_attempt 2>&1 | tee selftest_export_<pkg>.log
```

If export succeeds → proceed to Step 4 (generate patch).

### 3. If install fails — investigate on PyPI

```bash
# Check available versions
pip index versions "<pkg>" 2>/dev/null || pip install "<pkg>"== 2>&1 | head -5
# Check dependency conflicts
pip check 2>&1
```

Determine whether the conflict is:
- **Python version incompatibility** (e.g. package requires 3.12 but venv is 3.11):
  ```bash
  # Create a fresh venv with the required Python version
  python3.12 -m venv /tmp/optimum-intel/venv-exp-<attempt_id>-py312
  source /tmp/optimum-intel/venv-exp-<attempt_id>-py312/bin/activate
  pip install openvino openvino-tokenizers openvino-genai
  pip install git+https://github.com/huggingface/optimum-intel.git
  pip install "<pkg>"
  optimum-cli export openvino --model <MODEL_ID> ...
  ```
- **Version conflict with existing packages**: try `pip install "<pkg>" --upgrade` or
  pin a specific release found on PyPI that satisfies both the model and the environment.

### 4. If the install can only succeed in a sandbox venv

If install in the standard venv conflicts, but succeeds in the sandboxed venv:

1. **Continue your main task in the sandbox venv** — use it to complete the export.
2. **Record the working package pin** so the patch includes it:
   ```bash
   pip show "<pkg>" | grep Version   # note the version
   ```
3. Include the `requirements` note in `artifact-description.md`:
   ```markdown
   ## Extra dependencies required
   The following extra package is needed at export time:
   - `<pkg>==<version>` (or `<pkg>>=<min>`)
   Reason: model's `modeling_*.py` calls `require_backends(["<pkg>"])`.
   ```

This information feeds back to the deployer so subsequent runs install it upfront.

```python
from optimum.exporters.tasks import TasksManager
try:
    tasks = TasksManager.get_supported_tasks_for_model_type("<model_type>", exporter="openvino")
    print("Supported tasks:", tasks)
except KeyError:
    print("model_type not in optimum-intel → add_model_support needed")
```

## Transformers Version Handling

When the error class is `dependency_version` / `unknown_arch_transformers_too_old`,
follow this escalation ladder before giving up:

### 1. Check and upgrade to latest stable PyPI release

```bash
LATEST=$(pip index versions transformers 2>/dev/null \
  | grep -oP '(?<=Available versions: )[\.\d]+' | head -1)
INSTALLED=$(pip show transformers | grep '^Version' | awk '{print $2}')
echo "Installed: $INSTALLED  |  Latest PyPI: $LATEST"

# If different, upgrade and retry export
if [ "$INSTALLED" != "$LATEST" ]; then
  pip install --quiet "transformers==$LATEST"
  optimum-cli export openvino \
    --model "$MODEL_ID" \
    --task "$TASK" \
    --weight-format fp16 \
    /tmp/ov_out_pypi_latest 2>&1 | tee export_pypi_latest.log
fi
```

### 2. Install from git-HEAD

If the latest stable release still doesn’t register the `model_type`:

```bash
pip install --quiet \
  "git+https://github.com/huggingface/transformers@main" \
  --no-deps

# Probe
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('$MODEL_ID', trust_remote_code=False)
print(f'OK: {cfg.model_type}')
" 2>&1 | tee transformers_git_probe.log

# If probe passes, retry export
optimum-cli export openvino \
  --model "$MODEL_ID" \
  --task "$TASK" \
  --weight-format fp16 \
  /tmp/ov_out_git_head 2>&1 | tee export_git_head.log
```

### 3. Patch transformers source (last resort)

If git-HEAD still doesn’t recognise the `model_type`, invoke the
**`optimum_patch_transformers`** skill
(`skills/optimum-intel/patch-transformers.md`). That skill will:

- Search for an open transformers PR for this `model_type` and cherry-pick it.
- Failing that, extract the config class from the HuggingFace model repo and
  register it in a local transformers clone.
- Run the full export self-test with the patched transformers.
- Generate a `git format-patch` via `scripts/generate_git_patch.py`.
- Post the patch to the GitHub issue as a comment.

**Do not surrender** when PyPI/git-HEAD upgrades fail — always proceed to the
source-patching step first.

## Step 3 — Formulate and Test Hypotheses (Experiment Loop)

For each candidate fix, create a new `venv-exp-<attempt_id>`, implement the
change in the local `optimum-intel` clone, and test:

```bash
# Setup attempt:
cd /tmp/optimum-intel
setup_exp_venv exp-001   # from optimum_bootstrap Step 4

# Implement change (e.g. add config class):
# ... edit model_configs.py ...

# Test:
optimum-cli export openvino \
  --model <MODEL_ID> \
  --task text-generation-with-past \
  --weight-format fp16 \
  /tmp/ov_test_out_exp001 2>&1 | tee selftest_export_exp001.log

# Log result:
python - <<'EOF'
import json, datetime
log = json.load(open('experiments_log.json'))
log.append({
    "attempt_id": "exp-001",
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "hypothesis": "<what you expected to fix>",
    "approach": "<change made>",
    "venv": "venv-exp-exp-001",
    "steps_taken": ["<step 1>", "<step 2>"],
    "outcome": "<success|partial|failed>",
    "error_summary": "<key error line or empty>",
    "artifacts": [],
    "insights": "<what this revealed>",
    "next_hypothesis": "<next idea>"
})
json.dump(log, open('experiments_log.json', 'w'), indent=2)
EOF
```

## Step 4 — Generate Patch on Success

When export + inference self-test pass:

```bash
cd /tmp/optimum-intel
git add -A
git format-patch HEAD~1 -o patches/ \
  --stdout > patches/<attempt_id>-<description>.patch

# Verify the patch applies cleanly to a fresh clone:
git clone --depth 1 https://github.com/huggingface/optimum-intel.git /tmp/verify-patch
git -C /tmp/verify-patch am /path/to/patches/<attempt_id>-<description>.patch
```

## Step 5 — Escalate Non-optimum Issues

- **Missing op / IR validation failure** → route to OV Orchestrator with full error context.
- **Confirmed transformers dependency** → ensure `requires_optimum_recheck=true` is set.
- Provide the classified error, the relevant log snippet, and all `experiments_log.json`
  entries when escalating — do not hand over a blank context.

## Step 6 — Surrender (if all hypotheses exhausted)

Follow the **Surrender Protocol** in `optimum_intel.agent.md` exactly.
Minimum outputs: `experiments_log.json`, `agent_report.md`, all patches with
`artifact-description.md`.
