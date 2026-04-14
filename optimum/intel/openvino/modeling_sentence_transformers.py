import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from optimum.intel.utils.import_utils import _sentence_transformers_version, is_sentence_transformers_version


logger = logging.getLogger(__name__)

_MIN_SENTENCE_TRANSFORMERS_VERSION = "5.0"


class OVSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path: Optional[str] = None, **kwargs):
        logger.warning(
            "`OVSentenceTransformer` is deprecated and will be removed in optimum-intel v2.0. Please us sentence_transformers.SentenceTransformer instead."
        )

        if is_sentence_transformers_version("<", _MIN_SENTENCE_TRANSFORMERS_VERSION):
            raise ImportError(
                f"The minimum required version of sentence-transformers is {_MIN_SENTENCE_TRANSFORMERS_VERSION}, "
                f"got: {_sentence_transformers_version}"
            )

        kwargs["backend"] = "openvino"
        super().__init__(model_name_or_path, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "OVSentenceTransformer":
        model_kwargs = kwargs.pop("model_kwargs", {})
        for key in ("export", "ov_config", "device", "file_name"):
            if key in kwargs:
                model_kwargs[key] = kwargs.pop(key)
        return cls(model_name_or_path, model_kwargs=model_kwargs, **kwargs)
