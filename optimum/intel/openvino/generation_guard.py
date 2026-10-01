#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Content moderation of OpenVINO text generation.

Guard models come in two families, both reachable through the same [`OVGuard`] front end:

* *token classifiers*, such as `Qwen/Qwen3Guard-Stream-0.6B`, which replace the language modeling
  head with classification heads and score every token of a conversation as it is generated;
* *generative guards*, such as `meta-llama/Llama-Guard-3-1B` or `Qwen/Qwen3Guard-Gen-0.6B`, which
  are ordinary language models prompted to describe the risk of a conversation in plain text.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.generation.stopping_criteria import StoppingCriteria

from optimum.exporters.openvino.utils import get_multi_head_token_classification_spec


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .modeling import OVModelForTokenClassification
    from .modeling_base import OVBaseModel


logger = logging.getLogger(__name__)


PROMPT_SCOPE = "prompt"
RESPONSE_SCOPE = "response"

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

_ROLE_BY_SCOPE = {PROMPT_SCOPE: USER_ROLE, RESPONSE_SCOPE: ASSISTANT_ROLE}

#: Label reported when a generative guard returns text that does not follow its own output format.
UNPARSABLE_LABEL = "Unparsable"

#: A conversation, as accepted by `transformers` chat templates.
Conversation = List[Dict[str, Any]]


# --------------------------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OVGuardVerdict:
    """A risk assessment produced by a guard model.

    Depending on the guard model, a verdict covers a single token, a single message or a whole
    conversation, but its meaning is always the same: the risk of the content moderated so far.

    Args:
        label (`str`):
            Risk label chosen by the guard model, for example `"Safe"`, `"unsafe"`, or
            `"Unparsable"` when a generative guard did not follow its own output format.
        flagged (`bool`):
            Whether `label` is one of the blocking labels and therefore a policy violation.
        scope (`str`):
            `"prompt"` for the input rail, `"response"` for the output rail.
        categories (`Tuple[str, ...]`, defaults to `()`):
            Risk categories reported alongside `label`, using the taxonomy of the guard model.
        score (`Optional[float]`, defaults to `None`):
            Probability of `label`, when the guard model exposes one.
        category_scores (`Tuple[float, ...]`, defaults to `()`):
            Probabilities of `categories`, positionally aligned with them when available.
        batch_index (`int`, defaults to 0):
            Index of the moderated sequence in the batch.
        token_index (`Optional[int]`, defaults to `None`):
            Position of the moderated token in its sequence, for token-level guard models only.
        text (`Optional[str]`, defaults to `None`):
            Raw output of a generative guard model.
    """

    label: str
    flagged: bool
    scope: str
    categories: Tuple[str, ...] = ()
    score: Optional[float] = None
    category_scores: Tuple[float, ...] = ()
    batch_index: int = 0
    token_index: Optional[int] = None
    text: Optional[str] = None

    @property
    def role(self) -> str:
        """Conversation role the moderated content belongs to."""
        return _ROLE_BY_SCOPE[self.scope]

    def __str__(self) -> str:
        details = [f"scope={self.scope}", f"label={self.label!r}"]
        if self.score is not None:
            details.append(f"p={self.score:.2f}")
        if self.categories:
            details.append(f"categories={list(self.categories)}")
        if self.token_index is not None:
            details.append(f"token={self.token_index}")
        return f"{'flagged' if self.flagged else 'cleared'} ({', '.join(details)})"


@dataclass
class OVGuardReport:
    """Verdicts accumulated by a guard model, in the order in which they were produced.

    Args:
        verdicts (`List[OVGuardVerdict]`):
            Every verdict, for both the input and the output rail.
    """

    verdicts: List[OVGuardVerdict] = field(default_factory=list)

    @property
    def prompt(self) -> List[OVGuardVerdict]:
        """Verdicts of the input rail."""
        return [verdict for verdict in self.verdicts if verdict.scope == PROMPT_SCOPE]

    @property
    def response(self) -> List[OVGuardVerdict]:
        """Verdicts of the output rail."""
        return [verdict for verdict in self.verdicts if verdict.scope == RESPONSE_SCOPE]

    @property
    def flagged(self) -> bool:
        """Whether any verdict is a policy violation."""
        return self.first_violation is not None

    @property
    def first_violation(self) -> Optional[OVGuardVerdict]:
        """The earliest flagged verdict, or `None` when the content was cleared."""
        return next((verdict for verdict in self.verdicts if verdict.flagged), None)

    def __str__(self) -> str:
        violation = self.first_violation
        summary = str(violation) if violation is not None else "cleared"
        return f"OVGuardReport({len(self.verdicts)} verdicts, {summary})"


class OVGuardViolationError(RuntimeError):
    """Raised when a guard model flags content and `OVGuardConfig.on_violation` is `"raise"`."""

    def __init__(self, report: OVGuardReport):
        self.report = report
        super().__init__(f"Content flagged by the guard model: {report.first_violation}.")


# --------------------------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------------------------- #


@dataclass
class OVGuardConfig:
    """Controls how a guard model moderates a generation loop.

    Args:
        chunk_size (`Optional[int]`, defaults to `None`):
            Number of newly generated tokens buffered before they are sent to the guard model in a
            single pass. Larger values lower the moderation overhead per generated token but delay
            detection by up to `chunk_size - 1` tokens. `None` selects the default of the guard
            model: every token for a token classifier, and the completed response only for a
            generative guard, for which re-reading a partial response is a full extra inference.
        prompt_mode (`str`, defaults to `"async"`):
            How the prompt is moderated. `"async"` runs the guard model on a background thread,
            concurrently with the first forward pass of the guarded model, so that time to first
            token is not penalised; this pairs well with a guard model placed on another device.
            `"sync"` moderates the prompt before generation starts. `"off"` skips the input rail.
        blocking_labels (`Optional[Sequence[str]]`, defaults to `None`):
            Risk labels that count as a violation, matched case-insensitively. `None` selects the
            defaults of the guard model, usually its unsafe label only.
        on_violation (`str`, defaults to `"stop"`):
            `"stop"` ends generation at the flagged chunk, `"raise"` raises `OVGuardViolationError`,
            `"continue"` only records the verdict.
        emit_before_check (`bool`, defaults to `True`):
            Whether tokens are forwarded to `streamer` before the guard model has cleared them.
            Set to `False` to hold a chunk back until it is moderated, which prevents flagged text
            from ever reaching the streamer at the cost of `chunk_size` tokens of latency.
        on_verdict (`Optional[Callable[[OVGuardVerdict], None]]`, defaults to `None`):
            Called for every verdict as soon as it is produced.
    """

    chunk_size: Optional[int] = None
    prompt_mode: str = "async"
    blocking_labels: Optional[Sequence[str]] = None
    on_violation: str = "stop"
    emit_before_check: bool = True
    on_verdict: Optional[Callable[[OVGuardVerdict], None]] = None

    def __post_init__(self):
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError(f"`chunk_size` must be a positive integer or None, but got {self.chunk_size}.")
        if self.prompt_mode not in {"async", "sync", "off"}:
            raise ValueError(f"`prompt_mode` must be one of 'async', 'sync', 'off', but got {self.prompt_mode!r}.")
        if self.on_violation not in {"stop", "raise", "continue"}:
            raise ValueError(
                f"`on_violation` must be one of 'stop', 'raise', 'continue', but got {self.on_violation!r}."
            )


# --------------------------------------------------------------------------------------------- #
# Token classifier guards
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenClassifierGuardSpec:
    """Maps the classification heads of a guard architecture to the rail they moderate.

    Args:
        risk_head_by_scope (`Dict[str, str]`):
            Head producing the risk-level logits, per rail.
        category_head_by_scope (`Dict[str, str]`):
            Head producing the risk-category logits, per rail.
        default_blocking_labels (`Tuple[str, ...]`):
            Risk labels considered a violation when `OVGuardConfig.blocking_labels` is not set.
    """

    risk_head_by_scope: Dict[str, str]
    category_head_by_scope: Dict[str, str]
    default_blocking_labels: Tuple[str, ...] = ("Unsafe",)


TOKEN_CLASSIFIER_GUARD_ARCHITECTURES: Dict[str, TokenClassifierGuardSpec] = {
    "Qwen3ForGuardModel": TokenClassifierGuardSpec(
        risk_head_by_scope={PROMPT_SCOPE: "query_risk_level_logits", RESPONSE_SCOPE: "risk_level_logits"},
        category_head_by_scope={PROMPT_SCOPE: "query_category_logits", RESPONSE_SCOPE: "category_logits"},
    ),
}


def get_token_classifier_guard_spec(config: Optional[PretrainedConfig]) -> Optional[TokenClassifierGuardSpec]:
    """Returns the guard spec of `config`, or `None` if it is not a token classifier guard model."""
    for architecture in getattr(config, "architectures", None) or []:
        spec = TOKEN_CLASSIFIER_GUARD_ARCHITECTURES.get(architecture)
        if spec is not None:
            return spec
    return None


# --------------------------------------------------------------------------------------------- #
# Generative guards
# --------------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerativeGuardSpec:
    """Describes the prompt format and the output grammar of a generative guard model.

    A generative guard is prompted through its own chat template, which embeds the moderation task,
    the safety policy and the expected answer format. This spec captures the little that cannot be
    read from the template: how to parse the answer back, and the quirks of the template itself.

    Args:
        name (`str`):
            Identifier of the guard family, used in error messages.
        template_marker (`str`):
            Substring of the chat template that identifies the family. It is the very instruction
            that fixes the output grammar, which makes it a reliable fingerprint.
        label_pattern (`str`):
            Regular expression matched against the generated text, with a `label` group.
        labels (`Tuple[str, ...]`):
            Every risk label the guard model can emit.
        default_blocking_labels (`Tuple[str, ...]`):
            Labels considered a violation when `OVGuardConfig.blocking_labels` is not set.
        category_pattern (`Optional[str]`, defaults to `None`):
            Regular expression with a `categories` group, matched against the generated text.
        category_separator (`str`, defaults to `","`):
            Separator between categories inside the `categories` group.
        empty_categories (`Tuple[str, ...]`, defaults to `("none",)`):
            Category names, compared case-insensitively, that stand for "no category".
        content_as_parts (`bool`, defaults to `False`):
            Whether the chat template expects message contents as a list of `{"type", "text"}`
            parts rather than as a plain string.
        categories_template_key (`Optional[str]`, defaults to `None`):
            Chat template argument through which a custom taxonomy can be injected, when the
            template supports one.
        max_new_tokens (`int`, defaults to 32):
            Upper bound on the length of the moderation answer.
    """

    name: str
    template_marker: str
    label_pattern: str
    labels: Tuple[str, ...]
    default_blocking_labels: Tuple[str, ...]
    category_pattern: Optional[str] = None
    category_separator: str = ","
    empty_categories: Tuple[str, ...] = ("none",)
    content_as_parts: bool = False
    categories_template_key: Optional[str] = None
    max_new_tokens: int = 32

    def parse(self, text: str) -> Tuple[str, Tuple[str, ...]]:
        """Extracts the risk label and the risk categories from the answer of the guard model."""
        match = re.search(self.label_pattern, text, re.MULTILINE | re.IGNORECASE)
        if match is None:
            logger.warning(
                f"The {self.name} guard model answered {text!r}, which does not match its expected output format. "
                "The content is reported as flagged."
            )
            return UNPARSABLE_LABEL, ()

        categories: Tuple[str, ...] = ()
        if self.category_pattern is not None:
            category_match = re.search(self.category_pattern, text, re.MULTILINE | re.IGNORECASE)
            if category_match is not None:
                empty = {name.casefold() for name in self.empty_categories}
                categories = tuple(
                    category
                    for category in (
                        part.strip() for part in category_match["categories"].split(self.category_separator)
                    )
                    if category and category.casefold() not in empty
                )
        return match["label"].strip(), categories


GENERATIVE_GUARD_SPECS: Tuple[GenerativeGuardSpec, ...] = (
    GenerativeGuardSpec(
        name="Llama Guard",
        template_marker="First line must read 'safe' or 'unsafe'",
        label_pattern=r"^\s*(?P<label>safe|unsafe)\s*$",
        labels=("safe", "unsafe"),
        default_blocking_labels=("unsafe",),
        category_pattern=r"^\s*(?P<categories>S\d+(?:\s*,\s*S\d+)*)\s*$",
        content_as_parts=True,
        categories_template_key="categories",
    ),
    GenerativeGuardSpec(
        name="Qwen3Guard-Gen",
        template_marker="'Safety: Safe', 'Safety: Unsafe'",
        label_pattern=r"^\s*Safety:\s*(?P<label>\w+)",
        labels=("Safe", "Unsafe", "Controversial"),
        default_blocking_labels=("Unsafe",),
        category_pattern=r"^\s*Categories:\s*(?P<categories>.+)$",
    ),
)


def get_generative_guard_spec(tokenizer: "PreTrainedTokenizerBase") -> Optional[GenerativeGuardSpec]:
    """Returns the generative guard spec matching the chat template of `tokenizer`, if any."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return None
    return next((spec for spec in GENERATIVE_GUARD_SPECS if spec.template_marker in template), None)


# --------------------------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------------------------- #


class OVGuardBackend(ABC):
    """Moderation strategy behind an [`OVGuard`], one per guard model family."""

    #: Whether the backend classifies the token ids of the guarded model as they are generated.
    token_level: bool = False

    def __init__(self, model: "OVBaseModel", tokenizer: Optional["PreTrainedTokenizerBase"] = None):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def incremental(self) -> bool:
        """Whether the backend can be fed only the tokens added since the previous call."""
        return False

    @property
    @abstractmethod
    def labels(self) -> Tuple[str, ...]:
        """Every risk label the guard model can emit."""

    @property
    @abstractmethod
    def default_blocking_labels(self) -> Tuple[str, ...]:
        """Risk labels considered a violation unless the caller says otherwise."""

    @abstractmethod
    def moderate_conversations(
        self, conversations: Sequence[Conversation], scope: str, blocking_labels: Sequence[str]
    ) -> List[OVGuardVerdict]:
        """Moderates one conversation per batch entry, returning one verdict each."""

    def reset(self) -> None:
        """Forgets the conversation moderated so far."""

    def validate_target(self, target_config: PretrainedConfig) -> None:
        """Checks that this guard model can moderate the generations of `target_config`."""

    def moderate_token_ids(
        self,
        token_ids: torch.Tensor,
        scope: str,
        blocking_labels: Sequence[str],
        attention_mask: Optional[torch.Tensor] = None,
        token_offset: int = 0,
    ) -> List[List[OVGuardVerdict]]:
        """Moderates raw token ids of the guarded model, returning per-token verdicts."""
        raise NotImplementedError(
            f"{type(self).__name__} moderates text and cannot classify the token ids of another model."
        )

    def _require_tokenizer(self) -> "PreTrainedTokenizerBase":
        if self.tokenizer is None:
            raise ValueError(
                "Moderating a conversation requires the tokenizer of the guard model. Load the guard with "
                "`OVGuard.from_pretrained`, or pass `tokenizer=` to `OVGuard.from_model`."
            )
        return self.tokenizer


class OVTokenClassifierGuard(OVGuardBackend):
    """Backend for guard models that score every token through dedicated classification heads.

    The model keeps the conversation in its KV cache when it was exported with the
    `token-classification-with-past` task, in which case only the new tokens have to be submitted.
    """

    token_level = True

    def __init__(self, model: "OVModelForTokenClassification", tokenizer=None):
        spec = get_token_classifier_guard_spec(model.config)
        if spec is None:
            raise ValueError(
                f"{type(model).__name__} was loaded from an architecture that does not support moderation. "
                f"Supported token classifier guards are {sorted(TOKEN_CLASSIFIER_GUARD_ARCHITECTURES)}."
            )
        super().__init__(model, tokenizer)
        self.spec = spec
        self.label_maps = _resolve_label_maps(model.config)

    @property
    def incremental(self) -> bool:
        """Whether the model can be fed only the tokens added since the previous call."""
        return bool(self.model.stateful)

    @property
    def labels(self) -> Tuple[str, ...]:
        labels = {label for head in self.spec.risk_head_by_scope.values() for label in self.label_maps[head].values()}
        return tuple(sorted(labels))

    @property
    def default_blocking_labels(self) -> Tuple[str, ...]:
        return self.spec.default_blocking_labels

    def reset(self) -> None:
        self.model.reset_stream()

    def validate_target(self, target_config: PretrainedConfig) -> None:
        # The model consumes the token ids of the guarded model directly and cannot re-tokenize them.
        target_vocab_size = getattr(target_config.get_text_config(), "vocab_size", None)
        guard_vocab_size = getattr(self.model.config.get_text_config(), "vocab_size", None)
        if target_vocab_size is not None and guard_vocab_size is not None and target_vocab_size != guard_vocab_size:
            raise ValueError(
                "A token classifier guard and the guarded model must share a tokenizer, but their vocabulary sizes "
                f"differ ({guard_vocab_size} vs {target_vocab_size})."
            )

    def moderate_conversations(self, conversations, scope, blocking_labels):
        tokenizer = self._require_tokenizer()
        verdicts = []
        for batch_index, conversation in enumerate(conversations):
            token_ids = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=scope == PROMPT_SCOPE, return_tensors="pt"
            )
            self.reset()
            # A verdict covers the whole prefix seen so far, so a conversation is summarised by the
            # verdict of its last token; earlier positions describe truncated conversations.
            token_verdicts = self.moderate_token_ids(token_ids, scope, blocking_labels)[0]
            verdicts.append(_with_batch_index(token_verdicts[-1], batch_index))
        return verdicts

    def moderate_token_ids(self, token_ids, scope, blocking_labels, attention_mask=None, token_offset=0):
        logits = self.model(to_long_tensor(token_ids), attention_mask=attention_mask).logits
        risk_head = self.spec.risk_head_by_scope[scope]
        category_head = self.spec.category_head_by_scope[scope]
        risk_probabilities, risk_indices = torch.softmax(logits[risk_head].float(), dim=-1).max(dim=-1)
        category_probabilities, category_indices = torch.softmax(logits[category_head].float(), dim=-1).max(dim=-1)

        risk_labels = self.label_maps[risk_head]
        category_labels = self.label_maps[category_head]
        blocking = {label.casefold() for label in blocking_labels}

        verdicts = []
        for batch_index in range(risk_indices.shape[0]):
            sequence_verdicts = []
            for position in range(risk_indices.shape[1]):
                label = risk_labels[int(risk_indices[batch_index, position])]
                sequence_verdicts.append(
                    OVGuardVerdict(
                        label=label,
                        flagged=label.casefold() in blocking,
                        scope=scope,
                        categories=(category_labels[int(category_indices[batch_index, position])],),
                        score=float(risk_probabilities[batch_index, position]),
                        category_scores=(float(category_probabilities[batch_index, position]),),
                        batch_index=batch_index,
                        token_index=token_offset + position,
                    )
                )
            verdicts.append(sequence_verdicts)
        return verdicts


class OVGenerativeGuard(OVGuardBackend):
    """Backend for guard models that describe the risk of a conversation as generated text."""

    def __init__(
        self,
        model,
        tokenizer: "PreTrainedTokenizerBase",
        spec: Optional[GenerativeGuardSpec] = None,
        categories: Optional[Dict[str, str]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, tokenizer)
        if tokenizer is None:
            raise ValueError(
                "A generative guard is prompted through its own chat template and cannot be built without a "
                "tokenizer, pass one through `tokenizer=`."
            )
        spec = spec if spec is not None else get_generative_guard_spec(tokenizer)
        if spec is None:
            raise ValueError(
                "The chat template of this model does not match any known generative guard. Supported families are "
                f"{[known.name for known in GENERATIVE_GUARD_SPECS]}. Pass `guard_spec=GenerativeGuardSpec(...)` to "
                "describe another one."
            )
        self.spec = spec
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if categories is not None:
            if spec.categories_template_key is None:
                raise ValueError(
                    f"The chat template of {spec.name} hardcodes its taxonomy and cannot be given custom "
                    "`categories`."
                )
            self.chat_template_kwargs[spec.categories_template_key] = categories

    @property
    def labels(self) -> Tuple[str, ...]:
        return self.spec.labels

    @property
    def default_blocking_labels(self) -> Tuple[str, ...]:
        return self.spec.default_blocking_labels

    def moderate_conversations(self, conversations, scope, blocking_labels):
        # Each conversation gets its own guard prompt, which the chat template builds around it, so
        # they are moderated one at a time rather than as a padded batch.
        return [
            self._moderate_conversation(conversation, scope, blocking_labels, batch_index)
            for batch_index, conversation in enumerate(conversations)
        ]

    def _moderate_conversation(self, conversation, scope, blocking_labels, batch_index):
        tokenizer = self._require_tokenizer()
        inputs = tokenizer.apply_chat_template(
            self._as_template_messages(conversation),
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.spec.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        text = tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)

        label, categories = self.spec.parse(text)
        blocking = {blocking_label.casefold() for blocking_label in blocking_labels}
        return OVGuardVerdict(
            label=label,
            # An answer that could not be parsed is treated as a violation: a guard that cannot be
            # understood must not silently clear content.
            flagged=label == UNPARSABLE_LABEL or label.casefold() in blocking,
            scope=scope,
            categories=categories,
            batch_index=batch_index,
            text=text,
        )

    def _as_template_messages(self, conversation: Conversation) -> List[Dict[str, Any]]:
        if not self.spec.content_as_parts:
            return list(conversation)
        return [
            {**message, "content": [{"type": "text", "text": message["content"]}]}
            if isinstance(message.get("content"), str)
            else dict(message)
            for message in conversation
        ]


# --------------------------------------------------------------------------------------------- #
# Front end
# --------------------------------------------------------------------------------------------- #


class OVGuard:
    """Moderates conversations, and optionally a running generation, with a guard model.

    The guard model is loaded independently of the model it guards, so it can be placed on another
    device, for example a large language model on `"GPU"` guarded from `"CPU"`.

    Example:

    ```python
    >>> from optimum.intel import OVGuard, OVModelForCausalLM
    >>> from transformers import AutoTokenizer

    >>> guard = OVGuard.from_pretrained("Qwen/Qwen3Guard-Gen-0.6B", export=True, device="CPU")
    >>> guard.moderate_prompt("How do I pick a lock?").flagged
    True

    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>> model = OVModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", export=True, device="CPU")
    >>> inputs = tokenizer("Tell me about the sea.", return_tensors="pt")
    >>> outputs = model.generate(**inputs, max_new_tokens=16, guard=guard, tokenizer=tokenizer)
    ```
    """

    def __init__(self, backend: OVGuardBackend):
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, os.PathLike],
        task: Optional[str] = None,
        device: Optional[str] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        categories: Optional[Dict[str, str]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        guard_spec: Optional[GenerativeGuardSpec] = None,
        **kwargs,
    ) -> "OVGuard":
        """Loads a guard model, exporting it to the OpenVINO format if needed.

        Args:
            model_id (`str` or `os.PathLike`):
                Model repository or local directory of the guard model.
            task (`str`, *optional*):
                Export task. Defaults to `"token-classification-with-past"` for token classifier
                guards and to the default causal language modeling task for generative guards.
            device (`str`, *optional*):
                OpenVINO device the guard model runs on, independently of the guarded model.
            tokenizer (`PreTrainedTokenizerBase`, *optional*):
                Tokenizer of the guard model. Loaded from `model_id` when not given.
            categories (`Dict[str, str]`, *optional*):
                Custom taxonomy, as `{key: description}`, for generative guards whose chat template
                accepts one, such as Llama Guard.
            chat_template_kwargs (`Dict[str, Any]`, *optional*):
                Extra arguments forwarded to the chat template of a generative guard.
            guard_spec (`GenerativeGuardSpec`, *optional*):
                Description of a generative guard family that is not in
                [`GENERATIVE_GUARD_SPECS`].
            kwargs:
                Forwarded to the `from_pretrained` method of the underlying OpenVINO model.
        """
        from .modeling import OVModelForTokenClassification
        from .modeling_decoder import OVModelForCausalLM

        if device is not None:
            kwargs["device"] = device
        trust_remote_code = kwargs.get("trust_remote_code", False)
        config = kwargs.get("config") or AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tokenizer is None:
            tokenizer = _load_tokenizer(model_id, trust_remote_code)

        if get_token_classifier_guard_spec(config) is not None:
            model = OVModelForTokenClassification.from_pretrained(
                model_id, task=task or "token-classification-with-past", **kwargs
            )
            return cls(OVTokenClassifierGuard(model, tokenizer))

        if task is not None:
            kwargs["task"] = task
        model = OVModelForCausalLM.from_pretrained(model_id, **kwargs)
        return cls(OVGenerativeGuard(model, tokenizer, guard_spec, categories, chat_template_kwargs))

    @classmethod
    def from_model(
        cls,
        model: "OVBaseModel",
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> "OVGuard":
        """Wraps an already loaded OpenVINO guard model.

        Args:
            model (`OVBaseModel`):
                An `OVModelForTokenClassification` holding a token classifier guard, or an
                `OVModelForCausalLM` holding a generative guard.
            tokenizer (`PreTrainedTokenizerBase`, *optional*):
                Tokenizer of the guard model. Loaded from the model directory when not given.
            kwargs:
                Passed on to [`OVGenerativeGuard`] for generative guards.
        """
        if tokenizer is None:
            tokenizer = _load_tokenizer(model.config._name_or_path, trust_remote_code=False)
        if get_token_classifier_guard_spec(model.config) is not None:
            return cls(OVTokenClassifierGuard(model, tokenizer))
        return cls(OVGenerativeGuard(model, tokenizer, **kwargs))

    @property
    def model(self) -> "OVBaseModel":
        """The OpenVINO model behind the guard."""
        return self.backend.model

    @property
    def tokenizer(self) -> Optional["PreTrainedTokenizerBase"]:
        return self.backend.tokenizer

    @property
    def device(self) -> str:
        """OpenVINO device the guard model runs on."""
        return self.backend.model._device

    @property
    def labels(self) -> Tuple[str, ...]:
        """Every risk label the guard model can emit."""
        return self.backend.labels

    @property
    def token_level(self) -> bool:
        """Whether the guard model can moderate a generation token by token."""
        return self.backend.token_level

    def resolve_blocking_labels(self, blocking_labels: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
        """Validates `blocking_labels` against the labels of the guard model, or returns the defaults."""
        if blocking_labels is None:
            return self.backend.default_blocking_labels
        known = {label.casefold() for label in self.labels}
        unknown = sorted(label for label in blocking_labels if label.casefold() not in known)
        if unknown:
            raise ValueError(f"Unknown blocking labels {unknown}. The guard model labels are {list(self.labels)}.")
        return tuple(blocking_labels)

    def reset(self) -> None:
        """Forgets the conversation moderated so far."""
        self.backend.reset()

    def moderate(
        self,
        conversation: Union[str, Conversation],
        scope: Optional[str] = None,
        blocking_labels: Optional[Sequence[str]] = None,
    ) -> OVGuardReport:
        """Moderates a conversation.

        Args:
            conversation (`str` or `Sequence[Dict[str, Any]]`):
                A user message, or a chat-template conversation. The role of its last message
                selects the rail: a trailing assistant message moderates the response, anything
                else moderates the prompt.
            scope (`str`, *optional*):
                `"prompt"` or `"response"`, overriding the rail inferred from `conversation`.
            blocking_labels (`Sequence[str]`, *optional*):
                Risk labels that count as a violation. Defaults to those of the guard model.

        Returns:
            `OVGuardReport`: One verdict per conversation.
        """
        conversation = (
            [{"role": USER_ROLE, "content": conversation}] if isinstance(conversation, str) else conversation
        )
        if scope is None:
            scope = RESPONSE_SCOPE if conversation and conversation[-1]["role"] == ASSISTANT_ROLE else PROMPT_SCOPE
        elif scope not in _ROLE_BY_SCOPE:
            raise ValueError(f"`scope` must be one of {list(_ROLE_BY_SCOPE)}, but got {scope!r}.")
        verdicts = self.backend.moderate_conversations(
            [conversation], scope, self.resolve_blocking_labels(blocking_labels)
        )
        return OVGuardReport(verdicts)

    def moderate_prompt(self, prompt: Union[str, Conversation], **kwargs) -> OVGuardReport:
        """Input rail: moderates what is about to be sent to a model."""
        return self.moderate(prompt, scope=PROMPT_SCOPE, **kwargs)

    def moderate_response(self, prompt: Union[str, Conversation], response: str, **kwargs) -> OVGuardReport:
        """Output rail: moderates what a model answered to `prompt`."""
        conversation = [{"role": USER_ROLE, "content": prompt}] if isinstance(prompt, str) else list(prompt)
        conversation.append({"role": ASSISTANT_ROLE, "content": response})
        return self.moderate(conversation, scope=RESPONSE_SCOPE, **kwargs)

    def moderate_token_ids(
        self,
        token_ids,
        scope: str = RESPONSE_SCOPE,
        blocking_labels: Optional[Sequence[str]] = None,
        attention_mask=None,
        token_offset: int = 0,
    ) -> List[List[OVGuardVerdict]]:
        """Moderates raw token ids, returning a verdict per token and per sequence in the batch.

        Only token classifier guards support this. Consecutive calls continue the same conversation
        when the guard model has a KV cache, so a generation can be moderated as it unfolds; use
        [`~OVGuard.reset`] to start over.
        """
        return self.backend.moderate_token_ids(
            token_ids, scope, self.resolve_blocking_labels(blocking_labels), attention_mask, token_offset
        )

    def open_session(
        self,
        config: Optional[OVGuardConfig] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        incremental: bool = True,
    ) -> "OVGuardSession":
        """Starts a stateful moderation session over one generation.

        Args:
            config (`OVGuardConfig`, *optional*):
                Moderation settings. Defaults to `OVGuardConfig()`.
            tokenizer (`PreTrainedTokenizerBase`, *optional*):
                Tokenizer of the *guarded* model, needed by generative guards to read back the
                tokens it produces.
            incremental (`bool`, defaults to `True`):
                Whether the guarded sequences only ever grow by appending, which lets a token
                classifier guard reuse its KV cache. Beam search has to set this to `False`.
        """
        return OVGuardSession(self, config, tokenizer, incremental)


# --------------------------------------------------------------------------------------------- #
# Integration with `generate`
# --------------------------------------------------------------------------------------------- #


class OVGuardSession:
    """Drives a guard model over one generation.

    The session moderates the prompt, then the generated tokens, either token by token for a token
    classifier guard or by re-reading the response so far for a generative guard. Instances are
    single use and not reentrant: a guard model holds the conversation it is moderating, so one
    guard cannot back two concurrent sessions.
    """

    def __init__(
        self,
        guard: OVGuard,
        config: Optional[OVGuardConfig] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        incremental: bool = True,
    ):
        self.guard = guard
        self.config = config or OVGuardConfig()
        self.tokenizer = tokenizer
        self.blocking_labels = guard.resolve_blocking_labels(self.config.blocking_labels)
        self.report = OVGuardReport()

        backend = guard.backend
        self._token_level = backend.token_level
        # An incremental scan reuses the KV cache of the guard model, which is only valid while the
        # guarded sequences grow by appending. Beam search reorders them, so it re-scans instead.
        self._incremental = self._token_level and incremental and backend.incremental
        # A generative guard re-reads the whole response, so it only checks once at the end unless
        # the caller explicitly asks for intermediate checks.
        self._chunk_size = self.config.chunk_size or (1 if self._token_level else 0)
        if not self._token_level and tokenizer is None:
            raise ValueError(
                "Guarding a generation with a generative guard model requires the tokenizer of the guarded model, "
                "pass it through the `tokenizer` argument of `generate`."
            )

        self._prompt_ids: Optional[torch.Tensor] = None
        self._prompt_attention_mask: Optional[torch.Tensor] = None
        self._prompt_length = 0
        self._consumed = 0
        self._cleared_response_tokens = 0
        self._prompt_texts: List[str] = []
        self._prompt_future: Optional[Future] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._violation: Optional[OVGuardVerdict] = None

    @property
    def cleared_tokens(self) -> int:
        """Number of generated tokens moderated so far without a violation."""
        return self._cleared_response_tokens

    @property
    def prompt_ids(self) -> Optional[torch.Tensor]:
        return self._prompt_ids

    def start(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> None:
        """Resets the guard model and starts moderating the prompt."""
        self.guard.reset()
        self.report = OVGuardReport()
        self._violation = None
        self._prompt_ids = input_ids
        self._prompt_attention_mask = attention_mask
        self._prompt_length = input_ids.shape[-1]
        self._consumed = input_ids.shape[-1]
        self._cleared_response_tokens = 0
        if not self._token_level:
            self._prompt_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        if self.config.prompt_mode == "off":
            if self._incremental:
                # The prompt still has to go through the guard model to prime its KV cache, but its
                # verdicts are discarded.
                self._moderate_tokens(input_ids, PROMPT_SCOPE, token_offset=0, total_length=self._prompt_length)
            return

        if self.config.prompt_mode == "sync":
            self._collect(self._moderate_prompt())
            return

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ov-guard")
        self._prompt_future = self._executor.submit(self._moderate_prompt)

    def update(self, input_ids: torch.Tensor) -> bool:
        """Moderates the tokens generated since the previous call. Returns whether to stop."""
        self._await_prompt()
        if self._violation is not None:
            return self.config.on_violation == "stop"
        pending = input_ids.shape[-1] - self._consumed
        if self._chunk_size == 0 or pending < self._chunk_size:
            return False
        return self.flush(input_ids)

    def flush(self, input_ids: torch.Tensor) -> bool:
        """Moderates every token generated so far that has not been moderated yet."""
        self._await_prompt()
        if input_ids.shape[-1] <= self._consumed:
            return self._violation is not None and self.config.on_violation == "stop"

        verdicts = self._moderate_response(input_ids)
        self._consumed = input_ids.shape[-1]
        if self._violation is None:
            self._cleared_response_tokens = self._consumed - self._prompt_length
        self._collect(verdicts)
        return self._violation is not None and self.config.on_violation == "stop"

    def close(self) -> None:
        """Releases the background worker used by `prompt_mode="async"`."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            self._prompt_future = None

    def _await_prompt(self) -> None:
        if self._prompt_future is None:
            return
        verdicts = self._prompt_future.result()
        self._prompt_future = None
        self.close()
        self._collect(verdicts)

    def _moderate_prompt(self) -> List[OVGuardVerdict]:
        if not self._token_level:
            conversations = [[{"role": USER_ROLE, "content": text}] for text in self._prompt_texts]
            return self.guard.backend.moderate_conversations(conversations, PROMPT_SCOPE, self.blocking_labels)
        # A verdict covers the whole prefix seen so far, so the prompt is summarised by the verdict
        # of its last token; earlier positions describe truncated prompts and are noisy.
        verdicts = self._moderate_tokens(
            self._prompt_ids, PROMPT_SCOPE, token_offset=0, total_length=self._prompt_length
        )
        return [sequence[-1] for sequence in verdicts]

    def _moderate_response(self, input_ids: torch.Tensor) -> List[OVGuardVerdict]:
        if not self._token_level:
            responses = self.tokenizer.batch_decode(input_ids[:, self._prompt_length :], skip_special_tokens=True)
            conversations = [
                [{"role": USER_ROLE, "content": prompt}, {"role": ASSISTANT_ROLE, "content": response}]
                for prompt, response in zip(self._broadcast_prompts(len(responses)), responses)
            ]
            return self.guard.backend.moderate_conversations(conversations, RESPONSE_SCOPE, self.blocking_labels)

        token_offset = self._consumed - self._prompt_length
        if self._incremental:
            chunk = input_ids[:, self._consumed :]
        else:
            # A full re-scan replays the whole sequence, so any KV cache left from the previous
            # chunk has to be dropped first.
            self.guard.reset()
            chunk = input_ids
        verdicts = self._moderate_tokens(
            chunk, RESPONSE_SCOPE, token_offset=token_offset, total_length=input_ids.shape[-1]
        )
        if not self._incremental:
            # A full re-scan returns verdicts for the prompt as well, keep only the new tokens.
            verdicts = [sequence[self._consumed :] for sequence in verdicts]
        return [verdict for sequence in verdicts for verdict in sequence]

    def _moderate_tokens(
        self, token_ids: torch.Tensor, scope: str, token_offset: int, total_length: int
    ) -> List[List[OVGuardVerdict]]:
        return self.guard.backend.moderate_token_ids(
            token_ids,
            scope,
            self.blocking_labels,
            attention_mask=self._attention_mask(total_length, token_ids.shape[0]),
            token_offset=token_offset,
        )

    def _attention_mask(self, total_length: int, batch_size: int) -> Optional[torch.Tensor]:
        """Builds the mask covering the prompt plus every token generated so far.

        Generated tokens are always attended to, so only the prompt part can contain padding.
        """
        mask = self._prompt_attention_mask
        if mask is None:
            return None
        generated = total_length - self._prompt_length
        if generated > 0:
            ones = torch.ones((mask.shape[0], generated), dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, ones], dim=-1)
        if mask.shape[0] != batch_size:
            # Beam search expands each prompt into `num_beams` running sequences.
            mask = mask.repeat_interleave(batch_size // mask.shape[0], dim=0)
        return mask

    def _broadcast_prompts(self, batch_size: int) -> List[str]:
        # Beam search expands each prompt into `num_beams` running sequences.
        if len(self._prompt_texts) == batch_size:
            return self._prompt_texts
        repeats = batch_size // len(self._prompt_texts)
        return [prompt for prompt in self._prompt_texts for _ in range(repeats)]

    def _collect(self, verdicts: Sequence[OVGuardVerdict]) -> None:
        """Records verdicts and enforces `on_violation="raise"` as soon as one of them is flagged."""
        for verdict in verdicts:
            self.report.verdicts.append(verdict)
            if self.config.on_verdict is not None:
                self.config.on_verdict(verdict)
            if verdict.flagged and self._violation is None:
                self._violation = verdict
        if self._violation is None:
            return
        # Only the tokens preceding the first violation are clean and may be released.
        token_index = self._violation.token_index
        self._cleared_response_tokens = (
            token_index if self._violation.scope == RESPONSE_SCOPE and token_index is not None else 0
        )
        if self.config.on_violation == "raise":
            raise OVGuardViolationError(self.report)


class OVGuardStoppingCriteria(StoppingCriteria):
    """Feeds generated tokens to an [`OVGuardSession`] and stops generation on a violation."""

    def __init__(self, session: OVGuardSession):
        self.session = session

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        should_stop = self.session.update(input_ids)
        return torch.full((input_ids.shape[0],), should_stop, dtype=torch.bool, device=input_ids.device)


class OVGuardedStreamer:
    """Wraps a streamer so that tokens reach it only once the guard model has cleared them.

    Used when `OVGuardConfig.emit_before_check` is `False`. Generation appends a token to the
    streamer before the stopping criteria moderates it, so tokens are buffered here and released
    with a delay of one moderated chunk, and flagged tokens are never forwarded.
    """

    def __init__(self, streamer, session: OVGuardSession):
        self.streamer = streamer
        self.session = session
        self._pending: List[torch.Tensor] = []
        self._released = 0

    def put(self, value: torch.Tensor) -> None:
        if value.ndim > 1:
            # Generation forwards the whole prompt in a single call before the decoding loop, and
            # the wrapped streamer needs it to know the prompt is over.
            self.streamer.put(value)
            return
        self._pending.append(value)
        self._release()

    def end(self) -> None:
        # The last chunk is still unmoderated at this point, the session has the full sequence.
        self.session.flush(self._sequence())
        self._release()
        self._pending.clear()
        self.streamer.end()

    def _sequence(self) -> torch.Tensor:
        prompt_ids = self.session.prompt_ids
        if not self._pending:
            return prompt_ids
        generated = torch.stack([value.reshape(-1) for value in self._pending], dim=-1)
        return torch.cat([prompt_ids, generated.to(prompt_ids.device)], dim=-1)

    def _release(self) -> None:
        while self._released < min(self.session.cleared_tokens, len(self._pending)):
            self.streamer.put(self._pending[self._released])
            self._released += 1


# --------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------- #


def _resolve_label_maps(config: PretrainedConfig) -> Dict[str, Dict[int, str]]:
    """Reads the `{index: label}` map of every classification head from the guard model config."""
    head_spec = get_multi_head_token_classification_spec(config)
    if head_spec is None:
        raise ValueError(
            "The guard model config does not declare the multi-head token classification layout its architecture "
            f"is registered for. Known layouts are for {sorted(TOKEN_CLASSIFIER_GUARD_ARCHITECTURES)}."
        )
    label_maps = {}
    for head in head_spec.heads:
        raw_map = getattr(config, head.label_map_attr, None)
        if raw_map is None:
            raise ValueError(
                f"The guard model config is missing the `{head.label_map_attr}` label map required by the "
                f"`{head.name}` head."
            )
        label_maps[head.name] = {int(index): label for index, label in raw_map.items()}
    return label_maps


def _load_tokenizer(model_id, trust_remote_code: bool) -> Optional["PreTrainedTokenizerBase"]:
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as error:  # noqa: BLE001 - a missing tokenizer is only fatal for some backends
        logger.debug(f"No tokenizer could be loaded from {model_id}: {error}")
        return None


def _with_batch_index(verdict: OVGuardVerdict, batch_index: int) -> OVGuardVerdict:
    return verdict if verdict.batch_index == batch_index else replace(verdict, batch_index=batch_index)


def to_long_tensor(token_ids) -> torch.Tensor:
    """Normalizes user-provided token ids to a 2D `torch.LongTensor`."""
    if isinstance(token_ids, np.ndarray):
        token_ids = torch.from_numpy(token_ids)
    elif not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)
    token_ids = token_ids.to(torch.long)
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    if token_ids.ndim != 2:
        raise ValueError(f"Expected token ids of rank 1 or 2, but got a tensor of rank {token_ids.ndim}.")
    return token_ids
