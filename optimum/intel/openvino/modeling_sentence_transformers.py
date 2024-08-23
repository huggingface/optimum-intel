import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PretrainedConfig
from transformers.file_utils import add_start_docstrings

from ...exporters.openvino import main_export
from .configuration import OVConfig, OVWeightQuantizationConfig
from .modeling import MODEL_START_DOCSTRING, OVModel


@add_start_docstrings(
    """
    OpenVINO Model for feature extraction tasks for Sentence Transformers.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForSentenceTransformer(OVModel):
    export_feature = "feature-extraction"

    def __init__(self, model=None, config=None, orig_model_id_or_path=None, **kwargs):
        super().__init__(model, config, **kwargs)

        self.encode = MethodType(SentenceTransformer.encode, self)
        self._text_length = MethodType(SentenceTransformer._text_length, self)
        self.default_prompt_name = None
        self.truncate_dim = None
        self.orig_model_id_or_path = orig_model_id_or_path
        self.tokenizer_args = {
            "token": None,
            "trust_remote_code": False,
            "revision": None,
            "local_files_only": False,
            "model_max_length": 384,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.orig_model_id_or_path,
            **self.tokenizer_args,
        )

    def _save_pretrained(self, save_directory: Union[str, Path]):
        super()._save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        self.compile()
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")

        np_inputs = isinstance(input_ids, np.ndarray)
        if not np_inputs:
            input_ids = np.array(input_ids)
            attention_mask = np.array(attention_mask)
            token_type_ids = np.array(token_type_ids) if token_type_ids is not None else token_type_ids

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add the token_type_ids when needed
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids if token_type_ids is not None else np.zeros_like(input_ids)

        outputs = self._inference(inputs)
        return {
            "token_embeddings": torch.from_numpy(outputs["token_embeddings"]).to(self.device),
            "sentence_embedding": torch.from_numpy(outputs["sentence_embedding"]).to(self.device),
        }

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        **kwargs,
    ):
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting
        # would end-up removing the directory containing the underlying OpenVINO model
        cls._model_save_dir_tempdirectory_instance = save_dir

        # If load_in_8bit and quantization_config not specified then ov_config is set to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="fp32")

        # OVModelForFeatureExtraction works with Transformers type of models, thus even sentence-transformers models are loaded as such.
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task or cls.export_feature,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            ov_config=ov_config,
            library_name="sentence_transformers",
        )

        config.save_pretrained(save_dir_path)
        model = cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            orig_model_id_or_path=model_id,
            **kwargs,
        )

        model.model_id = model_id

        return model

    def tokenize(
        self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]], padding: Union[str, bool] = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.tokenizer_args["model_max_length"],
            )
        )
        return output
