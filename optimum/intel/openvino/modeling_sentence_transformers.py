from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PretrainedConfig
from transformers.file_utils import add_start_docstrings

from .modeling import MODEL_START_DOCSTRING, OVModel


@add_start_docstrings(
    """
    OpenVINO Model for feature extraction tasks for Sentence Transformers.
    """,
    MODEL_START_DOCSTRING,
)
class OVSentenceTransformer(OVModel):
    export_feature = "feature-extraction"
    _library_name = "sentence_transformers"

    def __init__(self, model=None, config=None, tokenizer=None, **kwargs):
        super().__init__(model, config, **kwargs)

        self.encode = MethodType(SentenceTransformer.encode, self)
        self._text_length = MethodType(SentenceTransformer._text_length, self)
        self.default_prompt_name = None
        self.truncate_dim = None
        self.tokenizer = tokenizer

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
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        local_files_only: bool = False,
        **kwargs,
    ):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", None)

        tokenizer_args = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        if tokenizer_kwargs:
            kwargs["tokenizer_args"].update(tokenizer_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_args)

        return super()._from_pretrained(
            model_id=model_id,
            config=config,
            token=token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            file_name=file_name,
            subfolder=subfolder,
            from_onnx=from_onnx,
            local_files_only=local_files_only,
            tokenizer=tokenizer,
            **kwargs,
        )

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
            )
        )
        return output
