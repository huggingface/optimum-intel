---
name: adding-new-model-support
description: "Generate patches to add support for new models from HuggingFace transformers and diffusers libraries to optimum-intel. Enables model export to OpenVINO IR format and inference through optimum-intel API with OpenVINO backend. Includes model config creation, modeling file patches, stateful model support, and integration testing."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "model family: llama, qwen, phi, gemma, mistral, stable-diffusion, flux, ltx-video; task: text-generation, image-generation, video-generation"
---

## When This Skill Applies

Use this skill when:
- Adding support for a **new model architecture** from transformers or diffusers
- Debugging issues with **model export to OpenVINO IR**

This skill helps generate patches to add support for new models from HuggingFace **transformers** and **diffusers** libraries to **optimum-intel**, enabling:
- Export to OpenVINO IR format
- Inference through optimum-intel API with OpenVINO backend
- Quantization and optimization support
- Stateful model support for improved generation performance

## Quick Start

### Test if a Model is Already Supported

```bash
# Try exporting the model
optimum-cli export openvino --model <model-id> output_dir

# If export fails with "Model type <model-type> is not supported", you need to add support
```

## Typical Workflow for Adding Support for a New Model:

1. **Model Architecture Analysis**
2. **Update `optimum/exporters/openvino/model_configs.py`** to add new model config class
3. **Update `optimum/exporters/openvino/model_patcher.py`** to add new model patching class if needed
4. **Create tests** to validate export and inference
5. **Update documentation** to include the new model

For more details about executing each step, refer to the sections below.

### Model Architecture Analysis

Identify the model family (e.g., LLaMA, Qwen, Phi, Stable Diffusion, FLUX) and determine block types used in the architecture (e.g., attention, feed-forward, MoE, linear attention)

```python
# 1. Check pipeline components
from transformers import AutoModelForCausalLM
import inspect

pipe = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-350M", torch_dtype=torch.bfloat16)
print("Components:", [k for k in dir(pipe) if not k.startswith('_') and hasattr(getattr(pipe, k), 'named_modules')])

# 2. Analyze transformer structure
for name, module in pipe.named_modules():
    class_name = type(module).__name__
    if 'Norm' in class_name or 'Attention' in class_name or 'GELU' in class_name:
        has_weight = hasattr(module, 'weight') and module.weight is not None
        print(f"{name}: {class_name} (has_weight={has_weight})")
```

Retrieve `model_type` from the model's config to determine if it matches an existing supported type or if a new config class is needed.

### Update documentation

After adding support for a new model, update the documentation in `docs/source/openvino/models.mdx` to include the corresponding `model_type`, ensuring it reflects the newly supported model. The `model_type` should be written with its first letter capitalized.

## Model Patching Patterns

The original implementation of a model from Transformers or Diffusers may contain code patterns that are not compatible with torch.jit.trace. This typically happens because torch.jit.trace records operations based on a specific example_input, and different inputs can produce different torch.Graph representations. As a result, the traced graph may not generalize correctly.
To ensure stable tracing, such dynamic or data-dependent code blocks must be rewritten using Torch-native primitives. In particular, Python control flow (e.g., for loops or conditional branches) that depends on runtime tensor values—rather than static configuration parameters—can lead to different graphs for different inputs.
In these cases, the recommended approach is to replace Python-level control flow with equivalent vectorized operations or other torch primitives. This guarantees consistent graph generation across inputs and makes the model traceable.
Below are examples demonstrating how to patch such patterns in different scenarios.

### Mixture of Experts (MoE)

For example, in the corresponding `modelling_afmoe.py` file for `afmoe` model, we have the following code block that implements
Mixture of Experts (MoE) logic with dynamic control flow. The original version contains a for-loop that iterates over experts and applies them to selected tokens, which can lead to different graphs based on the input data:

```python
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Get routing decisions
        top_scores, selected_experts = self.router(hidden_states, self.expert_bias)

        # Process through shared experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
        else:
            shared_output = torch.zeros_like(hidden_states_flat)

        # Reorder tokens by expert for efficient processing
        token_indices_sorted = torch.argsort(selected_experts.view(-1), stable=True)
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_to_expert = selected_experts.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.config.num_experts_per_tok

        # Gather input tokens
        token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(
            -1, hidden_dim
        )
        routed_input = torch.gather(
            hidden_states_flat, dim=0, index=token_indices_expanded
        )

        routed_output = torch.zeros_like(routed_input)
        for expert_id in range(self.config.num_experts):
            mask = token_to_expert == expert_id
            if mask.any():
                expert_input = routed_input[mask]
                expert_out = self.experts[expert_id](expert_input)
                routed_output[mask] = expert_out
          
        routed_output = (
            routed_output.to(torch.float32) * top_scores_sorted.unsqueeze(-1)
        ).to(hidden_states.dtype)

        # Scatter back to original positions
        output = shared_output.scatter_add(
            dim=0, index=token_indices_expanded, src=routed_output
        )

        return output.view(batch_size, seq_len, hidden_dim)
```

The original code contains a conditional branch inside a Python for-loop. For certain example inputs, this branch may be skipped during tracing, resulting in an incorrect or incomplete final graph. Additionally, the non-vectorized implementation produces a very large OpenVINO graph with excessive nodes, which is expensive for graph transformations and significantly increases model conversion time. So here is the patch that provides a vectorized form of MoE:

```python
def afmoe_moe_forward_patched(self, hidden_states):
    num_experts = self.config.num_experts
    batch_size, seq_len, hidden_dim = hidden_states.shape
    routing_weights, selected_experts = self.router(hidden_states, self.expert_bias)
    new_routing_weights = torch.zeros(batch_size * seq_len, self.config.num_experts, dtype=routing_weights.dtype)
    new_routing_weights.scatter_(dim=1, index=selected_experts, src=routing_weights)
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Process through shared experts
    if self.shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
    else:
        shared_output = torch.zeros_like(hidden_states)

    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_dim)
    act_fn = self.experts[0].act_fn

    # compute experts outputs in a vectorized form
    gate = torch.bmm(hidden_states, self.gate_projs)
    up = torch.bmm(hidden_states, self.up_projs)
    gate_up = act_fn(gate) * up
    next_states = torch.bmm(gate_up, self.down_projs)
    next_states = next_states.view(num_experts, batch_size, -1, hidden_dim)
    next_states = next_states * new_routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    next_states = next_states.sum(dim=0)

    shared_output = shared_output.view(batch_size, -1, hidden_dim)
    output = shared_output + next_states
    return output.view(batch_size, seq_len, hidden_dim)
```

## See Also

### Learning From Reference Pull Requests
When adding support for a new model to optimum-intel, study the following merged pull requests as canonical implementation examples:

- [Afmoe model](https://github.com/huggingface/optimum-intel/pull/1569/) - Adding support for Afmoe model with Mixture of Experts (MoE) logic
- [TODO](todo)

These PRs demonstrate:
- Proper integration into the export pipeline
- Correct configuration wiring
- Model registration patterns
- Test structure
- Documentation updates
Edge case handling

The agent should analyze structure and patterns — not blindly copy code.

Reference PRs must be analyzed using structured git diffs, not GitHub HTML rendering.

Fetch and Inspect PR Locally:

```bash
git clone https://github.com/huggingface/optimum-intel.git
cd optimum-intel
git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git checkout pr-<PR_NUMBER>
```

Inspect changes:

```bash
git diff main...HEAD --name-status
git diff main...HEAD
```

### Reference
- [docs/openvino/](../docs/openvino/) - Documentation on OpenVINO export and inference

### External Resources
- [optimum-intel](https://huggingface.co/docs/optimum/intel/overview) - HuggingFace Optimum Intel documentation
- [OpenVINO](https://docs.openvino.ai/2025/index.html) - OpenVINO documentation
- [torch.jit.trace](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html) - PyTorch JIT tracing documentation

## Project Structure

```
/skills/cuda-kernels/
└── SKILL.md           # This file
```
