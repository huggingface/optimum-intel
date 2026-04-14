# Skill: Patch Transformers for Unsupported Model Type

**Trigger:**
- Export or tokenizer conversion fails because `model_type '<type>'` is not
  registered in the installed `transformers` — and the fix is NOT yet available
  in any stable PyPI release *or* in git-HEAD.
- Invoked by `optimum_debug_export` and `openvino_tokenizers_unknown_arch`
  **only after** the simpler upgrade paths (PyPI latest → git-HEAD) have been
  exhausted.

> This skill is part of the **Autonomous Experiment Loop** and must log every
> attempt to `experiments_log.json`.  Follow the Surrender Protocol defined in
> the calling agent if all strategies below are exhausted.

---

## When NOT to use this skill

- If upgrading to the latest stable PyPI release fixes the problem → stop there.
- If installing from `git+https://github.com/huggingface/transformers@main` fixes
  the problem → stop there (generate `transformers_dep.json`, no source patch needed).
- If the root cause is something other than an unregistered `model_type` → escalate.

---

## Step 1 — Confirm Exact Failure

```python
import subprocess, sys

MODEL_ID = "<model_id>"

result = subprocess.run(
    [sys.executable, "-c",
     f"from transformers import AutoConfig; "
     f"cfg = AutoConfig.from_pretrained('{MODEL_ID}', trust_remote_code=False); "
     f"print(cfg.model_type)"],
    capture_output=True, text=True,
)
print("stdout:", result.stdout)
print("stderr:", result.stderr[-2000:])

# Extract model_type from error message
import re
m = re.search(r"model type `?['\"]?(\w+)['\"]?`?", result.stderr) or \
    re.search(r"KeyError: '(\w+)'", result.stderr)
model_type = m.group(1) if m else None
print(f"Unrecognised model_type: {model_type}")
```

Log this as `attempt_id: exp-00-confirm` in `experiments_log.json` before
proceeding.

---

## Step 2 — Search for an Open or Merged Transformers PR

Run *before* writing any code.  A PR may already exist — cherry-picking it is
faster and more correct than writing from scratch.

```bash
MODEL_TYPE="<model_type>"   # e.g. qwen3_5

# Search transformers GitHub
curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/search/issues?q=${MODEL_TYPE}+repo:huggingface/transformers+label:New+model&per_page=5" \
  | python -c "
import json, sys
for i in json.load(sys.stdin).get('items', []):
    print(i['number'], i['state'], i['title'], i['html_url'])
"

# Also try without the label filter in case it was tagged differently
curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/search/issues?q=${MODEL_TYPE}+repo:huggingface/transformers+is:pr&per_page=5" \
  | python -c "
import json, sys
for i in json.load(sys.stdin).get('items', []):
    print(i['number'], i['state'], i['title'], i['html_url'])
"
```

---

## Step 3 — Clone Transformers for Patching

```bash
TRANSFORMERS_PATCH="/tmp/transformers-patch"

if [ ! -d "$TRANSFORMERS_PATCH/.git" ]; then
  git clone --depth 50 https://github.com/huggingface/transformers.git \
    "$TRANSFORMERS_PATCH"
fi
cd "$TRANSFORMERS_PATCH"
git fetch origin
git checkout main && git reset --hard origin/main
```

---

## Step 3a — Apply Existing PR (preferred)

If a PR was found in Step 2:

```bash
PR_NUMBER=<NNN>
cd "$TRANSFORMERS_PATCH"
git fetch origin pull/${PR_NUMBER}/head:pr-${PR_NUMBER} --depth 20

# Inspect what the PR changes
git diff main...pr-${PR_NUMBER} --stat
git diff main...pr-${PR_NUMBER} -- src/transformers/models/auto/configuration_auto.py

# Cherry-pick it onto the tip of main
git cherry-pick pr-${PR_NUMBER} --no-commit   # stage only, review before committing
git diff --staged --stat
```

Verify the relevant `model_type` entry is now present:

```bash
grep -n "<model_type>" \
  src/transformers/models/auto/configuration_auto.py | head -5
```

If the cherry-pick looks clean → commit and continue to **Step 5**.

---

## Step 3b — Extract Config from Model Hub (fallback)

When no upstream PR exists, the HuggingFace model repository itself usually
ships the configuration class that was submitted upstream but not yet merged.

```python
from huggingface_hub import list_repo_files, hf_hub_download
import pathlib, shutil

MODEL_ID = "<model_id>"
PATCH_DIR = pathlib.Path("/tmp/transformers-patch")
MODEL_TYPE = "<model_type>"   # e.g. "qwen3_5"

# Find config Python files uploaded with the model
all_files = list(list_repo_files(MODEL_ID))
cfg_files  = [f for f in all_files if f.startswith("configuration_") and f.endswith(".py")]
print("Config files in model repo:", cfg_files)

for fname in cfg_files:
    local = hf_hub_download(repo_id=MODEL_ID, filename=fname)
    # Derive the canonical transformers module name, e.g. qwen3_5
    module_name = fname.replace("configuration_", "").replace(".py", "")
    dest_dir = PATCH_DIR / "src" / "transformers" / "models" / module_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Copy the config file into the transformers source tree
    dest = dest_dir / f"configuration_{module_name}.py"
    shutil.copy(local, dest)
    print(f"Installed: {dest}")

    # Create a minimal __init__.py if needed
    init_file = dest_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
```

Now open `src/transformers/models/auto/configuration_auto.py` and add the
mapping entry.  Identify the `CONFIG_MAPPING_NAMES = OrderedDict(` block and
insert the new pair in alphabetical order:

```python
# Example for model_type "qwen3_5" → config class name "Qwen3_5Config"
# Find the correct alphabetical position in CONFIG_MAPPING_NAMES and add:
#   ("qwen3_5", "Qwen3_5Config"),

import re, pathlib

auto_cfg = pathlib.Path(
    "/tmp/transformers-patch/src/transformers/models/auto/configuration_auto.py"
)
text = auto_cfg.read_text()

# Find the line just before the alphabetically-next entry
# and insert our new tuple.  Adjust anchor string as needed:
INSERT_AFTER = '("qwen3", '   # or the preceding entry
new_tuple = f'    ("<model_type>", "<ConfigClassName>"),\n'
assert INSERT_AFTER in text, f"Anchor not found: {INSERT_AFTER!r}"
text = text.replace(INSERT_AFTER, new_tuple + "    " + INSERT_AFTER, 1)
auto_cfg.write_text(text)
print("Updated configuration_auto.py")
```

> Replace `<model_type>` and `<ConfigClassName>` with the actual values
> extracted from the downloaded config file's `model_type` attribute and class
> name (look for `class <ClassName>(PretrainedConfig):` in the file).

---

## Step 4 — Install Patched Transformers and Test

```bash
ATTEMPT_ID="<attempt_id>"  # e.g. "exp-003-transformers-patch"
VENV="/tmp/venv-${ATTEMPT_ID}"
python -m venv "$VENV"
source "$VENV/bin/activate"

pip install -q --upgrade pip
# Install patched transformers from local clone
pip install -q -e /tmp/transformers-patch
pip install -q openvino openvino-tokenizers optimum[openvino]

# Probe: does AutoConfig now recognise the model?
python - <<'EOF'
from transformers import AutoConfig
import sys
try:
    cfg = AutoConfig.from_pretrained("<model_id>", trust_remote_code=False)
    print(f"AutoConfig OK: {cfg.model_type}")
except Exception as e:
    print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
EOF
```

If the probe fails → the config extraction in Step 3b was incomplete.  Debug
the import chain, add missing `__all__` / module `__init__.py` entries, and
re-test.

If the probe passes → run the full export / tokenizer conversion:

```bash
# For optimum-intel export:
optimum-cli export openvino \
  --model "<model_id>" \
  --task text-generation-with-past \
  --weight-format fp16 \
  /tmp/ov_out_${ATTEMPT_ID} 2>&1 | tee export_${ATTEMPT_ID}.log

echo "Export exit code: $?"

# For tokenizer conversion:
python - <<'EOF'
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer
from openvino import Core

tokenizer = AutoTokenizer.from_pretrained("<model_id>")
ov_tok, ov_detok = convert_tokenizer(tokenizer, with_detokenizer=True)

ie = Core()
ct  = ie.compile_model(ov_tok)
cdt = ie.compile_model(ov_detok)

for text in ["Hello, world!", "Тест", "你好"]:
    ids  = tokenizer(text, return_tensors="np").input_ids
    oids = ct({"string_input": [text]})["input_ids"]
    assert (ids == oids).all(), f"Mismatch for: {text!r}"
print("Tokenizer round-trip: OK")
EOF
```

Log the outcome (success or failure with key error lines) to
`experiments_log.json`.

---

## Step 5 — Generate the Transformers Patch

On success, generate a `git format-patch` from the local transformers clone:

```bash
cd /tmp/transformers-patch
git add -A
git commit -m "feat(<model_type>): add <model_type> config support

Auto-generated patch to register <ModelType>Config in the transformers
auto-mapping. Required to export model <model_id> with optimum-intel.

Tested with:
  transformers: $(python -c 'import transformers; print(transformers.__version__)')
  optimum: $(pip show optimum | grep Version | awk '{print $2}')
"

python scripts/generate_git_patch.py \
  --repo /tmp/transformers-patch \
  --base HEAD~1 \
  --output patches/${ATTEMPT_ID}-transformers-<model_type>.patch

echo "Patch written to patches/${ATTEMPT_ID}-transformers-<model_type>.patch"
cat patches/${ATTEMPT_ID}-transformers-<model_type>.patch | head -40
```

Verify the patch applies cleanly to a fresh checkout:

```bash
git clone --depth 1 https://github.com/huggingface/transformers.git \
  /tmp/verify-transformers-patch
git -C /tmp/verify-transformers-patch am \
  /tmp/transformers-patch/patches/${ATTEMPT_ID}-transformers-<model_type>.patch
echo "Patch applies cleanly: $?"
```

---

## Step 6 — Post Patch to GitHub Issue

Embed the patch in a GitHub comment so it can be retrieved and applied by any
downstream consumer (deployer, human reviewer, or orchestrator re-run):

```bash
PATCH_FILE="patches/${ATTEMPT_ID}-transformers-<model_type>.patch"
PATCH_CONTENT=$(cat "$PATCH_FILE")

python scripts/post_issue_comment.py \
  --repo   "$GITHUB_REPOSITORY" \
  --issue  "$ISSUE_NUMBER" \
  --title  "Transformers Source Patch — <model_type>" \
  --body   "## Transformers Source Patch

The exported model requires a transformers version that does not yet
recognise \`<model_type>\`. The patch below adds the missing config
registration to transformers main.

**How to apply:**
\`\`\`bash
git clone --depth 1 https://github.com/huggingface/transformers.git /tmp/t
git -C /tmp/t am <patch_file>
pip install -e /tmp/t
\`\`\`

<details><summary>Patch contents</summary>

\`\`\`diff
${PATCH_CONTENT}
\`\`\`

</details>

**Tested with:** transformers \`$(pip show transformers | grep ^Version)\`, model \`<model_id>\`
**Round-trip validation:** passed" \
  --token  "$GITHUB_TOKEN"
```

Also write a `transformers_source_patch.json` alongside the tokenizers artifact:

```json
{
  "model_type": "<model_type>",
  "patch_file": "patches/<attempt_id>-transformers-<model_type>.patch",
  "transformers_base_commit": "<git rev-parse HEAD before patch>",
  "status": "source_patched",
  "reason": "model_type_not_in_any_release"
}
```

---

## Step 7 — Signal to Orchestrator

```bash
echo "transformers_source_patched=true"       >> "$GITHUB_OUTPUT"
echo "transformers_patch_file=${PATCH_FILE}"   >> "$GITHUB_OUTPUT"
echo "requires_optimum_recheck=true"           >> "$GITHUB_OUTPUT"
echo "model_type=<model_type>"                 >> "$GITHUB_OUTPUT"
```

The orchestrator should pick up `transformers_source_patched=true` and pass
the patch file into any downstream agent that needs to build with transformers.

---

## Surrender Conditions

Trigger the calling agent's Surrender Protocol if:

- The model's HF repo ships **no** `configuration_*.py` and no transformers PR was found.
- Cherry-picking the found PR causes merge conflicts too complex to resolve automatically.
- The config class requires a new model architecture in `modeling_*.py` that cannot be
  auto-imported → this is a deeper no-code fix; escalate to OV Orchestrator.
- After patching, `AutoConfig` loads but the full export fails with a new error class
  (not `unknown_arch`/`KeyError`) → this is a different issue, re-classify.
