import json
import os
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import EntryNotFoundError
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig
from transformers.file_utils import add_start_docstrings

from .configuration import OVQuantizationConfigBase
from .modeling import MODEL_START_DOCSTRING, OVModel

from optimum.intel.utils.import_utils import is_sentence_transformers_version

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
        if is_sentence_transformers_version(">=", "5.4.0"):
            self.supports = MethodType(SentenceTransformer.supports, self)
            self._input_length = SentenceTransformer._input_length
            self._resolve_prompt = MethodType(SentenceTransformer._resolve_prompt, self)
            self.is_singular_input = MethodType(SentenceTransformer.is_singular_input, self)
            self.modalities = ['text', 'image', 'video', 'message']
            self.default_prompt_name = kwargs.get("default_prompt_name", None)
            self.prompts = kwargs.get("prompts", {}) or {}
            self.processor = kwargs.get("processor", None)
        else:
            self._text_length = MethodType(SentenceTransformer._text_length, self)
        self.default_prompt_name = None
        self.truncate_dim = None
        self.tokenizer = tokenizer

    def _save_pretrained(self, save_directory: Union[str, Path]):
        super()._save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def _can_flatten_inputs(self):
        return False

    def forward(self, inputs: Dict[str, torch.Tensor]):
        self.compile()
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")

        np_inputs = isinstance(input_ids, np.ndarray)
        if not np_inputs:
            input_ids = input_ids.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()
            token_type_ids = token_type_ids.cpu().numpy() if token_type_ids is not None else token_type_ids

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
        load_in_8bit: bool = False,
        quantization_config: Union[OVQuantizationConfigBase, Dict] = None,
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

        if is_sentence_transformers_version(">=", "5.4.0"):
            processor = None
            try:
                processor = AutoProcessor.from_pretrained(model_id, **tokenizer_args)
            except (OSError, ValueError, KeyError, EnvironmentError):
                processor = None

            # Load sentence-transformers prompts/default_prompt_name from config_sentence_transformers.json,
            # so that SentenceTransformer.encode's prompt resolution behaves the same as the reference model.
            st_prompts: Dict[str, str] = {}
            st_default_prompt_name: Optional[str] = None
            st_config_path: Optional[str] = None
            try:
                if os.path.isdir(model_id):
                    candidate = os.path.join(model_id, subfolder, "config_sentence_transformers.json") if subfolder else os.path.join(model_id, "config_sentence_transformers.json")
                    if os.path.isfile(candidate):
                        st_config_path = candidate
                else:
                    st_config_path = hf_hub_download(
                        repo_id=str(model_id),
                        filename="config_sentence_transformers.json",
                        subfolder=subfolder or None,
                        revision=revision,
                        cache_dir=cache_dir,
                        token=token,
                        local_files_only=local_files_only,
                        force_download=force_download,
                    )
            except (EntryNotFoundError, OSError, ValueError):
                st_config_path = None

            if st_config_path is not None:
                try:
                    with open(st_config_path, "r", encoding="utf-8") as f:
                        st_cfg = json.load(f)
                    st_prompts = st_cfg.get("prompts", {}) or {}
                    st_default_prompt_name = st_cfg.get("default_prompt_name", None)
                except (OSError, json.JSONDecodeError):
                    pass

        model = super()._from_pretrained(
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
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config,
            **kwargs,
        )

        if is_sentence_transformers_version(">=", "5.4.0"):
            model.prompts = st_prompts
            model.default_prompt_name = st_default_prompt_name
            model.processor = processor

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
            )
        )
        return output

    def get_model_kwargs(self):
        return []

    def _preprocess_quantization_config(
        self,
        quantization_config: OVQuantizationConfigBase,
        model_name_or_path: str,
    ) -> OVQuantizationConfigBase:
        if quantization_config.tokenizer is None:
            quantization_config = quantization_config.clone()
            quantization_config.tokenizer = model_name_or_path
        return quantization_config

    def preprocess(
        self,
        inputs,
        prompt,
        **kwargs,
    ):
        """
        Preprocesses the inputs for the model.

        Mirrors :meth:`sentence_transformers.base.modules.transformer.Transformer.preprocess`
        for the text/message modalities so that tokenization matches the reference
        SentenceTransformer model when a chat template is used (e.g. Qwen3-VL-Embedding).
        """
        from sentence_transformers.base.modality import format_modality, infer_batch_modality

        if not inputs:
            return {}

        # Infer modality (used both for validation and to decide preprocessing path).
        modality = None
        try:
            modality = infer_batch_modality(inputs, supported_modalities=self.modalities)
        except (ValueError, TypeError):
            pass

        if modality is not None and not self.supports(modality):
            supported = ", ".join(format_modality(m) for m in self.modalities)
            message = (
                f"Modality '{format_modality(modality)}' is not supported by this {type(self).__name__} model. "
                f"Supported modalities: {supported}"
            )
            if isinstance(modality, tuple) and all(part in self.modalities for part in modality):
                message += (
                    f"\nThis model supports {' and '.join(modality)} individually, "
                    "but not in the same input. Please process each modality separately."
                )
            raise ValueError(message)

        # If the model has a chat template, route inputs through apply_chat_template so the output
        # matches the reference SentenceTransformer (which uses the processor when available).
        tokenizer = self.tokenizer
        processor = getattr(self, "processor", None)
        chat_template_owner = None
        if processor is not None and getattr(processor, "chat_template", None) is not None:
            chat_template_owner = processor
        elif tokenizer is not None and getattr(tokenizer, "chat_template", None) is not None:
            chat_template_owner = tokenizer

        if chat_template_owner is not None and "message" in self.modalities:
            messages_batch = self._build_messages_batch(inputs, modality, prompt)
            preprocessed = chat_template_owner.apply_chat_template(
                messages_batch,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
            )
            preprocessed = dict(preprocessed)
            preprocessed["modality"] = "message"
        else:
            # Fallback: plain tokenization (e.g. for text-only models without a chat template).
            if prompt and modality == "text":
                inputs = [
                    (prompt + inp[0],) + tuple(inp[1:]) if isinstance(inp, tuple) else prompt + inp
                    for inp in inputs
                ]
            preprocessed = self.tokenize(inputs, **kwargs)
            preprocessed["modality"] = modality

        print("inputs_ids {}".format(preprocessed["input_ids"]))
        return preprocessed

    @staticmethod
    def _build_messages_batch(
        inputs: List[Any],
        modality: Any,
        prompt: Optional[str],
    ) -> List[List[Dict[str, Any]]]:
        """Convert SentenceTransformer-style inputs into a list of chat-template message lists.

        Each text input becomes a ``user`` message with structured content; if ``prompt`` is
        provided it is prepended as a ``system`` message (matching
        ``InputFormatter.prepend_prompt_to_messages``).
        """

        def _content_for_item(item: Any) -> List[Dict[str, Any]]:
            if isinstance(item, str):
                return [{"type": "text", "text": item}]
            if isinstance(item, dict):
                content: List[Dict[str, Any]] = []
                for key, value in item.items():
                    if key == "text":
                        content.append({"type": "text", "text": value})
                    elif key in ("image", "image_url"):
                        content.append({"type": "image", "image": value})
                    elif key == "video":
                        content.append({"type": "video", "video": value})
                    else:
                        content.append({"type": key, key: value})
                return content
            # Tuples/lists (e.g. text pairs) - flatten into separate text parts.
            if isinstance(item, (tuple, list)):
                return [{"type": "text", "text": str(v)} for v in item]
            return [{"type": "text", "text": str(item)}]

        messages_batch: List[List[Dict[str, Any]]] = []
        for inp in inputs:
            user_message = {"role": "user", "content": _content_for_item(inp)}
            sample_messages: List[Dict[str, Any]] = []
            if prompt:
                sample_messages.append(
                    {"role": "system", "content": [{"type": "text", "text": prompt}]}
                )
            sample_messages.append(user_message)
            messages_batch.append(sample_messages)
        return messages_batch