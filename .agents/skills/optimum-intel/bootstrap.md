# Skill: Optimum-Intel Bootstrap

**Trigger:** Called automatically before every optimum-intel task.

## Purpose

Lay the stable foundation for all optimum-intel experiments:
- Local, up-to-date `optimum-intel` clone with SKILL.md conventions loaded
- Isolated per-attempt venv infrastructure
- Package allowlist enforced with `pip-audit` back-stop
- Upstream search context ready (read existing fixes before writing new code)
- Experiment journal initialised

---

## Step 1 — Clone or update optimum-intel

```bash
OPTIMUM_DIR="${OPTIMUM_DIR:-/tmp/optimum-intel}"
if [ ! -d "$OPTIMUM_DIR/.git" ]; then
  git clone --depth 50 https://github.com/huggingface/optimum-intel.git "$OPTIMUM_DIR"
fi
cd "$OPTIMUM_DIR" && git checkout main && git pull --ff-only
```

## Step 2 — Load SKILL.md conventions

```bash
SKILL_FILE=""
for path in "skills/adding-new-model-support/SKILL.md" "skills/SKILL.md"; do
  [ -f "$OPTIMUM_DIR/$path" ] && SKILL_FILE="$OPTIMUM_DIR/$path" && break
done
if [ -z "$SKILL_FILE" ]; then
  cd "$OPTIMUM_DIR"
  git fetch origin pull/1616/head:pr-1616 --depth 1 2>/dev/null || true
  git checkout pr-1616 2>/dev/null || true
  SKILL_FILE_SRC="$OPTIMUM_DIR/skills/adding-new-model-support/SKILL.md"
  if [ -f "$SKILL_FILE_SRC" ]; then
    cp "$SKILL_FILE_SRC" /tmp/optimum_skill.md
    SKILL_FILE=/tmp/optimum_skill.md
  fi
  git checkout main
fi
# Read SKILL_FILE fully before writing any code.
```

## Step 3 — Package Allowlist

All packages **not** in the list below require the `pip-audit gate` before install.
`--trust-remote-code` is **never** acceptable — treat any requirement for it as an
immediate blocker and trigger the Surrender Protocol.

```
# ALLOWED — install freely
Core:         numpy scipy packaging wheel setuptools
ML:           torch torchvision torchaudio
HuggingFace:  transformers tokenizers huggingface-hub datasets accelerate
              safetensors peft sentencepiece tiktoken
OpenVINO:     openvino openvino-tokenizers openvino-genai nncf
Optimum:      optimum optimum-intel
Testing:      pytest
Utilities:    requests tqdm filelock psutil pip-audit

# ALLOWED git+https install origins (must start with one of these prefixes)
https://github.com/huggingface/
https://github.com/openvinotoolkit/
https://github.com/intel/

# ALWAYS FORBIDDEN
--trust-remote-code               # never, under any circumstances
Unknown git forks                 # any git+https not matching the prefixes above
Any install requiring user confirmation → treat as FAIL, trigger Surrender Protocol
```

**pip-audit gate** (run for any package outside the allowlist):

```bash
pip download --no-deps --dest /tmp/pip_audit_check "<package-spec>"
pip-audit --path /tmp/pip_audit_check --format json > /tmp/audit_result.json
python - <<'EOF'
import json, sys
data = json.load(open('/tmp/audit_result.json'))
vulns = [v for d in data.get('dependencies', []) for v in d.get('vulns', [])]
if vulns:
    print('SECURITY: vulnerabilities found:', json.dumps(vulns, indent=2))
    sys.exit(1)
print('pip-audit: clean')
EOF
# Install only if exit code is 0.
```

## Step 4 — Isolated Experiment Venv

Each attempt gets its own venv. State bleed between attempts masks root causes.

```bash
# Call at the start of each new attempt:
setup_exp_venv() {
  local attempt_id="$1"
  local venv_path="/tmp/venv-exp-${attempt_id}"
  python -m venv "$venv_path"
  source "$venv_path/bin/activate"
  pip install -q --upgrade pip
  pip install openvino openvino-tokenizers openvino-genai
  pip install -e "$OPTIMUM_DIR"   # editable local clone
  echo "Active venv: $venv_path"
}
```

Naming: `venv-exp-<attempt_id>` — short slugs, e.g. `exp-001`, `exp-add-config`, `exp-patcher-moe`.

## Step 5 — Upstream Search Context

Shallow-clone reference repos for **read-only** local pattern search.
Never commit or push to these clones.

```bash
TOKENIZERS_DIR="${TOKENIZERS_DIR:-/tmp/openvino-tokenizers-ref}"
if [ ! -d "$TOKENIZERS_DIR/.git" ]; then
  git clone --depth 200 https://github.com/openvinotoolkit/openvino_tokenizers.git "$TOKENIZERS_DIR"
fi
cd "$TOKENIZERS_DIR" && git pull --ff-only || true
cd "$OPTIMUM_DIR"
```

### upstream_search — procedure

Run **before writing any new code**. Existing fixes save hours — find them first.

**A. GitHub API — search PRs/issues by keyword** (requires `GITHUB_TOKEN`):

```bash
MODEL_TYPE="qwen3"  # replace with actual model_type

# optimum-intel
curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/search/issues?q=${MODEL_TYPE}+repo:huggingface/optimum-intel&per_page=5" \
  | python -c "
import json, sys
for i in json.load(sys.stdin).get('items', []):
    print(i['number'], i['state'], i['title'])
    print('  ', i['html_url'])
"

# openvino_tokenizers
curl -sf -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/search/issues?q=${MODEL_TYPE}+repo:openvinotoolkit/openvino_tokenizers&per_page=5" \
  | python -c "
import json, sys
for i in json.load(sys.stdin).get('items', []):
    print(i['number'], i['state'], i['title'])
    print('  ', i['html_url'])
"
```

**B. Local git history — commit message grep**:

```bash
cd "$OPTIMUM_DIR"
git log --oneline --all --grep="$MODEL_TYPE" | head -20
git log --oneline --all --grep="model_patcher" | head -10
# Inspect a specific commit:
# git show <hash> -- optimum/exporters/openvino/model_configs.py

cd "$TOKENIZERS_DIR"
git log --oneline --all --grep="$MODEL_TYPE" | head -20
```

**C. Fetch and study a relevant upstream PR**:

```bash
cd "$OPTIMUM_DIR"
# git fetch origin pull/<NNN>/head:pr-<NNN> --depth 5
# git diff main...pr-<NNN> -- optimum/exporters/openvino/model_configs.py
# git diff main...pr-<NNN> -- optimum/exporters/openvino/model_patcher.py
```

**D. Find the closest analogous model config**:

```bash
cd "$OPTIMUM_DIR"
grep -n "class.*OpenVINOConfig" optimum/exporters/openvino/model_configs.py | head -40
# Identify the most structurally similar model type and read that class fully.
```

## Step 6 — Experiment Journal

Maintain `experiments_log.json` in the working directory. Append one record per attempt.
This file is the single source of truth for the Surrender Report.

```json
[
  {
    "attempt_id": "exp-001",
    "timestamp": "ISO8601",
    "hypothesis": "Single sentence: what you expect this attempt to prove or fix",
    "approach": "Concrete change or install made",
    "venv": "venv-exp-exp-001",
    "steps_taken": ["description of step 1", "description of step 2"],
    "outcome": "success | partial | failed",
    "error_summary": "Key error line if failed, empty if success",
    "artifacts": ["patches/exp-001-add-config.patch"],
    "insights": "What this attempt revealed about the root cause",
    "next_hypothesis": "What to try next (or 'surrender' if all ideas exhausted)"
  }
]
```

Initialise once at bootstrap start:
```bash
[ -f experiments_log.json ] || echo '[]' > experiments_log.json
```

## External References

- **optimum-intel:** https://github.com/huggingface/optimum-intel
- **openvino_tokenizers:** https://github.com/openvinotoolkit/openvino_tokenizers
- **SKILL.md PR:** https://github.com/huggingface/optimum-intel/pull/1616
- **Reference PR (Afmoe):** https://github.com/huggingface/optimum-intel/pull/1569
