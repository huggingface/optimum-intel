# Model Patching Patterns

Use this reference only after the traceback or analysis proves that original
model code is incompatible with tracing/export.

## Data-dependent control flow

`torch.jit.trace` records the path taken by example inputs. Python conditions,
loops, `.item()`, `.tolist()`, or tensor-dependent indexing can therefore
produce a graph that works only for the dummy input. Replace runtime-dependent
Python logic with tensor operations and validate with shapes different from the
export dummy input.

## Vectorized Mixture of Experts

A loop such as this is unsafe when the trace input does not visit every expert:

```python
for expert_id in range(self.config.num_experts):
    mask = selected_experts == expert_id
    if mask.any():
        output[mask] = self.experts[expert_id](hidden_states[mask])
```

When the architecture exposes stacked expert weights, use a vectorized form:

```python
def moe_forward_patched(self, hidden_states):
    batch, sequence, hidden = hidden_states.shape
    scores, selected = self.router(hidden_states)

    routing = torch.zeros(
        batch * sequence,
        self.config.num_experts,
        dtype=scores.dtype,
        device=scores.device,
    )
    routing.scatter_(1, selected.reshape(-1, selected.shape[-1]), scores.reshape(-1, scores.shape[-1]))

    flat = hidden_states.reshape(-1, hidden)
    repeated = flat.unsqueeze(0).expand(self.config.num_experts, -1, -1)
    gate = torch.bmm(repeated, self.gate_projs)
    up = torch.bmm(repeated, self.up_projs)
    expert_output = torch.bmm(self.act_fn(gate) * up, self.down_projs)
    expert_output = expert_output * routing.transpose(0, 1).unsqueeze(-1)
    return expert_output.sum(0).reshape(batch, sequence, hidden)
```

Adapt parameter layout and routing semantics to the real model. Do not apply
this template when experts are not represented by compatible stacked tensors.

## Validation

- Compare original and patched PyTorch outputs before conversion.
- Export using more than one representative shape when possible.
- Compare OpenVINO component outputs against the unpatched reference.
- Ensure all experts/branches remain represented in the graph.
- Do not accept a patch solely because conversion completes.
