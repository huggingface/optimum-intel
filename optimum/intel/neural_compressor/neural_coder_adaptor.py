class NeuralCoderAdaptor():
    """API design adaption for Neural Coder"""
    
    default_quant_dynamic = \
f"""\
def eval_func(model):
    EVAL_FUNC_LINES
from optimum.intel.neural_compressor import IncOptimizer
from optimum.intel.neural_compressor import IncQuantizationConfig
from optimum.intel.neural_compressor import IncQuantizer
quantization_config = OPTIMUM_QUANT_CONFIG
quantizer = IncQuantizer(quantization_config, eval_func=eval_func)
optimizer = IncOptimizer(MODEL_NAME, quantizer=quantizer)
MODEL_NAME = optimizer.fit()
MODEL_NAME.eval()
"""

    default_quant_static = \
f"""\
def eval_func(model):
    EVAL_FUNC_LINES
from optimum.intel.neural_compressor import IncOptimizer
from optimum.intel.neural_compressor import IncQuantizationConfig
from optimum.intel.neural_compressor import IncQuantizer
quantization_config = OPTIMUM_QUANT_CONFIG
calib_dataloader = DATALOADER_NAME
quantizer = IncQuantizer(quantization_config, eval_func=eval_func, calib_dataloader=calib_dataloader)
optimizer = IncOptimizer(MODEL_NAME, quantizer=quantizer)
MODEL_NAME = optimizer.fit()
MODEL_NAME.eval()
"""
