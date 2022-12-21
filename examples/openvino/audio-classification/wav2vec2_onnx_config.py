from typing import Any, Mapping, Optional, Union

from transformers.onnx import OnnxConfig
from transformers.onnx.utils import compute_effective_axis_dimension
from transformers.utils import TensorType


class Wav2Vec2OnnxConfig(OnnxConfig):
    """
    Class for ONNX exportable model describing metadata on how to export the Wav2Vec2 model for
    audio classification through the ONNX format.
    """

    @property
    def inputs(self):
        dynamic_axis = {0: "batch", 1: "sequence"}
        return dict([
            ("input_values", dynamic_axis),
        ])

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin", "ImageProcessingMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        sampling_rate: int = 22050,
        time_duration: float = 5.0,
        frequency: int = 220,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ) -> Mapping[str, Any]:
        batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
        dummy_input = self._generate_dummy_audio(batch_size, sampling_rate, time_duration, frequency)
        return dict(preprocessor(dummy_input, return_tensors=framework, sampling_rate=preprocessor.sampling_rate))
