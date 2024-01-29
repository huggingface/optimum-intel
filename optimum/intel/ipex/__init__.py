from optimum.intel.ipex.modeling_base import (
    IPEXModel,
    IPEXModelForCausalLM,
    IPEXModelForMaskedLM,
    IPEXModelForQuestionAnswering,
    IPEXModelForSequenceClassification,
    IPEXModelForTokenClassification,
)

from .inference import inference_mode
