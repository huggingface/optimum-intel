from optimum.intel.ipex.modeling_base import (
    IPEXModelForCausalLM,
    IPEXModelForMaskedLM,
    IPEXModelForQuestionAnswering,
    IPEXModelForSequenceClassification,
    IPEXModelForTokenClassification,
    IPEXModel,
)

from .inference import inference_mode
