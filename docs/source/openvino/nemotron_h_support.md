# NemotronH (NVIDIA Nemotron Hybrid Mamba-2 + MoE) OpenVINO Export Support

## Overview

NemotronH is NVIDIA's hybrid language model combining:
- **Selective State Space Models (Mamba-2)** for efficient sequence modeling with sub-quadratic complexity
- **Mixture of Experts (MoE)** for sparse computation and improved efficiency
- **Hybrid Architecture** with alternating Mamba and attention layers with MoE blocks

The model achieves competitive performance with improved inference efficiency compared to dense transformer-only models.

## Model Characteristics

- **Architecture Type**: Hybrid (Mamba-2 + Attention + MoE)
- **Sequence Length Efficiency**: Sub-quadratic complexity through Mamba-2
- **Sparsity**: MoE layers provide sparse activation paths
- **Supported Tasks**: 
  - `text-generation` - Standard next-token prediction
  - `text-generation-with-past` - Cached key-value generation for inference

## Configuration Parameters

### Mamba-2 SSM Parameters
```python
ssm_state_size: int          # State space model hidden size (e.g., 64)
expand: int                   # Expansion factor for SSM (typically 2)
conv_kernel: int              # Convolution kernel size (typically 4)
head_dim: int                 # Head dimension for SSM (e.g., 64)
```

### Mixture of Experts Parameters
```python
num_experts: int              # Total number of experts (e.g., 64)
num_experts_per_tok: int      # Experts selected per token (e.g., 8)
```

### Hybrid Architecture Parameters
```python
layers_block_type: List[str]  # Layer sequence (e.g., ["mamba", "attention", "moe"])
hybrid_override_pattern: str  # Pattern override (e.g., "ME*M" for repeated cycles)
```

## OpenVINO Export

### Basic Export
```bash
optimum-cli export openvino \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --task text-generation-with-past \
  --output_dir ./nemotron_h_openvino \
  --trust_remote_code
```

### Export with Optimization
```bash
optimum-cli export openvino \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --task text-generation-with-past \
  --output_dir ./nemotron_h_openvino \
  --int8 \
  --trust_remote_code
```

### Programmatic Export
```python
from optimum.intel import OVModelForCausalLM

model = OVModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    export=True,
    device="CPU",
    trust_remote_code=True,
    quantization_config={
        "bits": 8,
        "sym": True,
        "group_size": 128,
    }
)
```

## Inference with OpenVINO

### Basic Inference
```python
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM

model_dir = "./nemotron_h_openvino"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = OVModelForCausalLM.from_pretrained(model_dir, device="CPU")

# Generate text
input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
outputs = model.generate(input_ids, max_length=100)
text = tokenizer.decode(outputs[0])
print(text)
```

### Batch Generation with Past Cache
```python
import torch
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM

model_dir = "./nemotron_h_openvino"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = OVModelForCausalLM.from_pretrained(model_dir, device="CPU")

prompt = "What is the capital of France?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate with cached past key-values (faster for subsequent tokens)
outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    use_cache=True,  # Enable KV cache for efficiency
    num_beams=1,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Architecture Details

### Hybrid Layer Composition
The NemotronH model alternates between three layer types:

1. **Mamba-2 Layers** - Sub-quadratic sequence modeling
   - Uses selective SSM mechanism
   - Parameters: `ssm_state_size`, `expand`, `conv_kernel`, `head_dim`
   
2. **Attention Layers** - Standard multi-head attention
   - Parameters: `num_attention_heads`, `num_key_value_heads`
   
3. **MoE Layers** - Sparse mixture of experts
   - Parameters: `num_experts`, `num_experts_per_tok`

### Layer Pattern Example
```
Input
  ↓
[Mamba Layer 1]          ← Sub-quadratic SSM
  ↓
[Attention Layer 1]      ← Dense attention
  ↓
[MoE Layer 1]            ← Sparse experts
  ↓
[Mamba Layer 2]
  ↓
... (repeat) ...
  ↓
[Lm Head]
  ↓
Logits
```

## Performance Characteristics

### Inference Speed
- **Sub-quadratic complexity** through Mamba-2 layers
- **Sparse activation** through MoE layers
- **KV cache support** for faster generation with past

### Memory Requirements
- Reduced KV cache due to Mamba layers (no attention in those layers)
- Efficient expert selection in MoE layers (only `num_experts_per_tok` active)
- Supports quantization (INT8, etc.) for further memory reduction

### Latency Benefits
- Smaller per-token latency through sparse computation
- Efficient context handling with Mamba SSM
- Faster batch inference with optimized operator kernels

## Testing

NemotronH export and inference functionality is tested through:

1. **Model Type Registration** - `test_nemotron_h.py::TestNemotronHRegistration`
   - Verifies model type is registered in TasksManager
   - Tests inheritance chain (OVConfig → GraniteMoeHybridOpenVINOConfig → NemotronHOpenVINOConfig)

2. **Configuration Tests** - `test_nemotron_h.py::TestNemotronHConfig`
   - NemotronHOpenVINOConfig initialization
   - Default parameter handling

3. **Normalized Config Tests** - `test_nemotron_h.py::TestNemotronHNormalizedConfig`
   - Layer type mapping for hybrid architecture
   - MoE parameter handling
   - Mamba SSM parameter handling

4. **Hybrid Architecture Tests** - `test_nemotron_h.py::TestNemotronHHybridArchitecture`
   - Hybrid override pattern support
   - Alternating layer validation

5. **Task Support Tests** - `test_nemotron_h.py::TestNemotronHTaskSupport`
   - text-generation task
   - text-generation-with-past task

6. **Integration Tests** - `test_exporters_cli.py`
   - End-to-end export and tokenizer conversion
   - Decoder functionality

## Troubleshooting

### Issue: "trust_remote_code" Error
**Solution**: Set `trust_remote_code=True` when loading the model, as NemotronH uses custom code:
```python
model = OVModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)
```

### Issue: Out of Memory During Export
**Solution**: Export with quantization to reduce memory footprint:
```bash
optimum-cli export openvino \
  --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --int8 \
  --output_dir ./nemotron_h_openvino
```

### Issue: Slow Generation
**Solution**: Enable KV cache and batch generation:
```python
outputs = model.generate(
    input_ids,
    use_cache=True,           # Enable KV cache
    max_new_tokens=100,
    num_beams=1,
    repetition_penalty=1.2
)
```

## References

- [NVIDIA Nemotron Models](https://huggingface.co/nvidia)
- [Selective Structured State-Spaces (Mamba)](https://arxiv.org/abs/2312.00752)
- [Optimum OpenVINO Documentation](https://huggingface.co/docs/optimum/main/en/intel/optimization_ov)
- [OpenVINO Documentation](https://docs.openvino.ai/)

## Implementation Details

### NemotronHOpenVINOConfig
Located in `optimum/exporters/openvino/model_configs.py`
- Inherits from `GraniteMoeHybridOpenVINOConfig`
- Registers with TasksManager for both text-generation tasks
- Provides ONNX conversion configuration
- Handles dummy input generation for model export

### NemotronHNormalizedTextConfig
Located in `optimum/exporters/openvino/model_configs.py`
- Normalizes config attributes from HuggingFace format to OpenVINO format
- Maps layer block types for hybrid architecture
- Handles MoE and SSM parameter normalization

### NemotronHModelPatcher
Located in `optimum/intel/openvino/io_binding.py`
- Patches NemotronH models for OpenVINO inference
- Handles KV cache management
- Optimizes operator execution
