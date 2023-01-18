class NeuralCoderAdaptor:
    """API design adaption for Neural Coder"""

    default_quant_dynamic = f"""\
def eval_func(model):
    EVAL_FUNC_LINES
from neural_compressor.config import PostTrainingQuantConfig
from optimum.intel.neural_compressor import INCQuantizer
quantization_config = PostTrainingQuantConfig(approach="dynamic")
quantizer = INCQuantizer.from_pretrained(MODEL_NAME, eval_fn=eval_func)
quantizer.quantize(quantization_config=quantization_config, save_directory="quantized_model", save_onnx_model=False)
MODEL_NAME = quantizer._quantized_model
MODEL_NAME.eval()
"""

    default_quant_static = f"""\
def eval_func(model):
    EVAL_FUNC_LINES
from neural_compressor.config import PostTrainingQuantConfig
from optimum.intel.neural_compressor import INCQuantizer
quantization_config = PostTrainingQuantConfig(approach="static")
quantizer = INCQuantizer.from_pretrained(MODEL_NAME, eval_fn=eval_func)
quantizer.quantize(quantization_config=quantization_config, calibration_dataset=eval_dataset, save_directory="quantized_model", save_onnx_model=False)
MODEL_NAME = quantizer._quantized_model
MODEL_NAME.eval()
"""
