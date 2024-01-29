from optimum.intel.ipex.modeling_base import (
    IPEXModelForCausalLM,
    IPEXModelForMaskedLM,
    IPEXModelForQuestionAnswering,
    IPEXModelForSequenceClassification,
    IPEXModelForTokenClassification,
)

from .inference import inference_mode
