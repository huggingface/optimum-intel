from types import MethodType
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers.file_utils import add_start_docstrings
from transformers import AutoTokenizer

from .modeling import MODEL_START_DOCSTRING, OVModel


@add_start_docstrings(
    """
    OpenVINO Model for feature extraction tasks for Sentence Transformers.
    """,
    MODEL_START_DOCSTRING,
)
class OVModelForSTFeatureExtraction(OVModel):
    export_feature = "feature-extraction"

    def __init__(self, model=None, config=None, model_id=None, **kwargs):
        super().__init__(model, config, **kwargs)

        self.encode = MethodType(SentenceTransformer.encode, self)
        self._text_length = MethodType(SentenceTransformer._text_length, self)
        self.default_prompt_name = None
        self.truncate_dim = None
        self.model_id = model_id

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

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]], padding: Union[str, bool] = True) -> Dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        tokenizer_args = {"token": None, "trust_remote_code": False, "revision": None, "local_files_only": False, "model_max_length": 384}
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            **tokenizer_args,
        )

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
            tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=tokenizer_args["model_max_length"],
            )
        )
        return output
