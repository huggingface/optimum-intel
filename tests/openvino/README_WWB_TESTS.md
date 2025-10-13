# WWB (Who What Benchmark) Tests for Optimum vs GenAI

This directory contains tests for WWB (Who What Benchmark) integration that specifically test the `optimum_vs_genai` evaluation mode.

## Test Files

### `test_wwb_optimum_vs_genai.py`
The main integration test that:
- Takes a model path as input
- Runs WWB optimum_vs_genai comparison
- Verifies that scores from WWB were evaluated
- Uses optimum.intel for model preparation

### `test_wwb_standalone.py`
A standalone test that focuses purely on WWB CLI functionality without complex optimum.intel dependencies:
- Tests WWB command-line interface
- Tests model path validation
- Tests output structure verification
- Can run with HuggingFace model IDs directly

### `test_wwb.py`
A comprehensive test suite with additional features and edge cases.

## Requirements

To run these tests, you need:

1. **WWB (Who What Benchmark)** - Available in the environment PATH
2. **openvino-genai** - For GenAI inference backend
3. **optimum-intel** - For Optimum inference backend (test_wwb_optimum_vs_genai.py only)
4. **pandas** - For results parsing
5. **transformers** - For model handling

## Usage

### Running all WWB tests:
```bash
cd /path/to/optimum-intel
python -m pytest tests/openvino/test_wwb*.py -v
```

### Running specific tests:
```bash
# Test basic functionality without external dependencies
python -m pytest tests/openvino/test_wwb_standalone.py::WWBStandaloneTest::test_score_evaluation_output_structure -v

# Test WWB help command (requires WWB to be available)
python -m pytest tests/openvino/test_wwb_standalone.py::WWBStandaloneTest::test_wwb_help_command -v

# Test full optimum vs genai evaluation (requires all dependencies)
python -m pytest tests/openvino/test_wwb_standalone.py::WWBStandaloneTest::test_optimum_vs_genai_with_hf_model -v
```

### Running with model path input:
The tests are designed to accept model paths as input. You can modify the test model by changing the `model_path` variable in the test methods, or by setting environment variables:

```bash
export TEST_MODEL_PATH="/path/to/your/openvino/model"
python -m pytest tests/openvino/test_wwb_standalone.py -v
```

## Test Behavior

### What the tests do:

1. **Model Path Input**: Tests accept a model path (OpenVINO format or HuggingFace model ID)

2. **Two-Stage Evaluation**:
   - Stage 1: Generate ground truth data using optimum backend
   - Stage 2: Run evaluation using genai backend and compare against ground truth

3. **Score Validation**: Verify that WWB produced evaluation scores by checking:
   - `metrics.csv` exists and contains numerical data
   - `metrics_per_question.csv` exists and contains per-sample results
   - Output directory structure is correct
   - All files contain valid data (not empty, not all NaN)

### Expected Outputs:

When tests pass, you should see:
- ✓ Ground truth data generated successfully
- ✓ Target model evaluation completed
- ✓ Found N evaluated samples
- ✓ Available metrics: [list of metric names]
- ✓ Numerical metrics: [list of numerical metric names]

## Error Handling

Tests will be skipped if dependencies are missing:
- "WWB is not available in the environment" - Install WWB or add to PATH
- "openvino-genai is not available in the environment" - Install openvino-genai package
- "optimum.intel is not available in the environment" - Install optimum-intel package

Tests will fail if:
- Model path is invalid or inaccessible
- WWB commands return non-zero exit codes
- Output files are not generated or contain invalid data
- Evaluation scores are not properly computed

## Customization

To test with your own models:

1. **OpenVINO models**: Provide the path to a directory containing the OpenVINO model files
2. **HuggingFace models**: Provide the model ID (e.g., "microsoft/DialoGPT-small")

Example:
```python
# In test file, modify:
model_path = "/path/to/your/openvino/model"
# or
model_path = "your-org/your-model-name"
```

## Integration with PnP Validation

These tests are designed to integrate with the existing PnP validation framework that uses WWBAccuracyCLI. The test patterns match the expected workflow:

1. Model path input
2. WWB optimum_vs_genai execution
3. Score evaluation verification
4. Results validation

This ensures compatibility with the broader validation pipeline while providing standalone test capabilities.