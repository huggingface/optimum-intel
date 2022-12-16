from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class OVTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to OpenVINO/NNCF-enabled training flow
    """
    nncf_compression_config: str = field(default=None,
        metadata={"help": "NNCF configuration .json file for compression-enabled training"}
    )
    teacher_model_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    distillation_weight: float = field(
        default=0.5, metadata={"help": "weightage of distillation loss, value between 0 to 1"}
    )
    distillation_temperature: float = field(
        default=2.0, metadata={"help": "temperature of distillation."}
    )