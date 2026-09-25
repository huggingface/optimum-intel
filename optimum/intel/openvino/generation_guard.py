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
#  See the License for the specific language governing permissions ando 
#  limitations under the License.

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import PretrainedConfig
from transformers.generation.stopping_criteria import StoppingCriteria

from optimum.exporters.openvino.utils import get_multi_head_token_classification_spec


if TYPE_CHECKING:
    from .modeling import OVModelForTokenClassification


logger = logging.getLogger(__name__)


USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


@dataclass(frozen=True)
class GuardArchitectureSpec:
    """Maps the classification heads of a guard architecture to conversation roles.

    Args:
        risk_head_by_role (`Dict[str, str]`):
            Head producing the risk-level logits, per conversation role.
        category_head_by_role (`Dict[str, str]`):
            Head producing the risk-category logits, per conversation role.
        default_blocking_labels (`Tuple[str, ...]`):
            Risk levels considered a violation when `OVGuardConfig.blocking_labels` is not set.
    """

    risk_head_by_role: Dict[str, str]
    category_head_by_role: Dict[str, str]
    default_blocking_labels: Tuple[str, ...] = ("Unsafe",)

    @property
    def roles(self) -> Tuple[str, ...]:
        return tuple(self.risk_head_by_role)


GUARD_ARCHITECTURES: Dict[str, GuardArchitectureSpec] = {
    "Qwen3ForGuardModel": GuardArchitectureSpec(
        risk_head_by_role={USER_ROLE: "query_risk_level_logits", ASSISTANT_ROLE: "risk_level_logits"},
        category_head_by_role={USER_ROLE: "query_category_logits", ASSISTANT_ROLE: "category_logits"},
    ),
}


def get_guard_architecture_spec(config: Optional[PretrainedConfig]) -> Optional[GuardArchitectureSpec]:
    """Returns the guard spec of `config`, or `None` if the architecture is not a guard model."""
    for architecture in getattr(config, "architectures", None) or []:
        spec = GUARD_ARCHITECTURES.get(architecture)
        if spec is not None:
            return spec
    return None


class OVGuardViolationError(RuntimeError):
    """Raised when the guard model flags content and `OVGuardConfig.on_violation` is `"raise"`."""

    def __init__(self, report: "OVGuardReport"):
        self.report = report
        violation = report.first_violation
        super().__init__(
            f"Content flagged by the guard model: role={violation.role}, "
            f"risk_level={violation.risk_level!r} (p={violation.risk_probability:.2f}), "
            f"category={violation.category!r} (p={violation.category_probability:.2f})."
        )


@dataclass(frozen=True)
class OVGuardVerdict:
    """Risk assessment produced by a guard model for a single token.

    Args:
        role (`str`):
            Conversation role the token belongs to, either `"user"` or `"assistant"`.
        batch_index (`int`):
            Index of the sequence in the batch the token belongs to.
        token_index (`int`):
            Position of the token in its sequence.
        risk_level (`str`), risk_probability (`float`):
            Most likely risk level and its probability.
        category (`str`), category_probability (`float`):
            Most likely risk category and its probability.
        flagged (`bool`):
            Whether `risk_level` is one of the blocking labels.
    """

    role: str
    batch_index: int
    token_index: int
    risk_level: str
    risk_probability: float
    category: str
    category_probability: float
    flagged: bool


@dataclass
class OVGuardReport:
    """Verdicts accumulated by a guard model over one `generate` call.

    Args:
        prompt (`List[OVGuardVerdict]`):
            Verdicts for the prompt, empty when prompt moderation is disabled.
        response (`List[OVGuardVerdict]`):
            Verdicts for the generated tokens, in generation order.
    """

    prompt: List[OVGuardVerdict] = field(default_factory=list)
    response: List[OVGuardVerdict] = field(default_factory=list)

    @property
    def verdicts(self) -> List[OVGuardVerdict]:
        return [*self.prompt, *self.response]

    @property
    def flagged(self) -> bool:
        return self.first_violation is not None

    @property
    def first_violation(self) -> Optional[OVGuardVerdict]:
        return next((verdict for verdict in self.verdicts if verdict.flagged), None)


@dataclass
class OVGuardConfig:
    """Controls how a guard model moderates a generation loop.

    Args:
        chunk_size (`int`, defaults to 1):
            Number of newly generated tokens buffered before they are sent to the guard model in a
            single forward pass. Larger values lower the guard overhead per generated token but
            delay detection by up to `chunk_size - 1` tokens.
        prompt_mode (`str`, defaults to `"async"`):
            How the prompt is moderated. `"async"` runs the guard model on a background thread,
            concurrently with the first forward pass of the guarded model, so that time to first
            token is not penalised. `"sync"` moderates the prompt before generation starts.
            `"off"` skips prompt moderation.
        blocking_labels (`Optional[Sequence[str]]`, defaults to `None`):
            Risk levels that count as a violation. When `None`, the defaults declared for the guard
            architecture are used and validated against the label maps of the guard model config.
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

    chunk_size: int = 1
    prompt_mode: str = "async"
    blocking_labels: Optional[Sequence[str]] = None
    on_violation: str = "stop"
    emit_before_check: bool = True
    on_verdict: Optional[Callable[[OVGuardVerdict], None]] = None

    def __post_init__(self):
        if self.chunk_size < 1:
            raise ValueError(f"`chunk_size` must be a positive integer, but got {self.chunk_size}.")
        if self.prompt_mode not in {"async", "sync", "off"}:
            raise ValueError(f"`prompt_mode` must be one of 'async', 'sync', 'off', but got {self.prompt_mode!r}.")
        if self.on_violation not in {"stop", "raise", "continue"}:
            raise ValueError(
                f"`on_violation` must be one of 'stop', 'raise', 'continue', but got {self.on_violation!r}."
            )


def resolve_label_maps(config: PretrainedConfig, spec: GuardArchitectureSpec) -> Dict[str, Dict[int, str]]:
    """Reads the `{index: label}` map of every guard head from the guard model config."""
    head_spec = get_multi_head_token_classification_spec(config)
    if head_spec is None:
        raise ValueError(
            "The guard model config does not declare multi-head token classification. Supported guard "
            f"architectures are {sorted(GUARD_ARCHITECTURES)}."
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


def resolve_blocking_labels(
    config: PretrainedConfig, spec: GuardArchitectureSpec, blocking_labels: Optional[Sequence[str]]
) -> Tuple[str, ...]:
    """Validates the requested blocking labels against the risk-level labels of the guard model."""
    label_maps = resolve_label_maps(config, spec)
    known_labels = set()
    for head_name in spec.risk_head_by_role.values():
        known_labels.update(label_maps[head_name].values())

    labels = tuple(blocking_labels) if blocking_labels is not None else spec.default_blocking_labels
    unknown = sorted(set(labels) - known_labels)
    if unknown:
        raise ValueError(f"Unknown blocking labels {unknown}. The guard model risk levels are {sorted(known_labels)}.")
    return labels


def decode_guard_logits(
    logits: Dict[str, torch.Tensor],
    role: str,
    spec: GuardArchitectureSpec,
    label_maps: Dict[str, Dict[int, str]],
    blocking_labels: Sequence[str],
    token_offset: int = 0,
) -> List[List[OVGuardVerdict]]:
    """Turns the raw head logits of a guard model into per-token verdicts.

    Returns a list of per-token verdicts for every sequence in the batch.
    """
    if role not in spec.risk_head_by_role:
        raise ValueError(f"`role` must be one of {list(spec.roles)}, but got {role!r}.")

    risk_head = spec.risk_head_by_role[role]
    category_head = spec.category_head_by_role[role]
    risk_probs = torch.softmax(logits[risk_head].float(), dim=-1)
    category_probs = torch.softmax(logits[category_head].float(), dim=-1)
    risk_probability, risk_index = risk_probs.max(dim=-1)
    category_probability, category_index = category_probs.max(dim=-1)

    risk_labels = label_maps[risk_head]
    category_labels = label_maps[category_head]
    blocking = set(blocking_labels)

    verdicts = []
    for batch_index in range(risk_index.shape[0]):
        sequence_verdicts = []
        for position in range(risk_index.shape[1]):
            risk_level = risk_labels[int(risk_index[batch_index, position])]
            sequence_verdicts.append(
                OVGuardVerdict(
                    role=role,
                    batch_index=batch_index,
                    token_index=token_offset + position,
                    risk_level=risk_level,
                    risk_probability=float(risk_probability[batch_index, position]),
                    category=category_labels[int(category_index[batch_index, position])],
                    category_probability=float(category_probability[batch_index, position]),
                    flagged=risk_level in blocking,
                )
            )
        verdicts.append(sequence_verdicts)
    return verdicts


class OVGuardSession:
    """Drives a guard model over one `generate` call.

    The session moderates the prompt, then the generated tokens in chunks of
    `OVGuardConfig.chunk_size`. When the guard model has a KV cache it is fed only the new tokens;
    otherwise the whole sequence is re-scanned on every chunk.
    """

    def __init__(
        self,
        guard_model: "OVModelForTokenClassification",
        generation_config: Optional[OVGuardConfig] = None,
        incremental: bool = True,
    ):
        spec = get_guard_architecture_spec(guard_model.config)
        if spec is None:
            raise ValueError(
                f"{guard_model.__class__.__name__} was loaded from an architecture that is not a supported guard "
                f"model. Supported guard architectures are {sorted(GUARD_ARCHITECTURES)}."
            )

        self.guard_model = guard_model
        self.config = generation_config or OVGuardConfig()
        self.spec = spec
        self.label_maps = resolve_label_maps(guard_model.config, spec)
        self.blocking_labels = resolve_blocking_labels(guard_model.config, spec, self.config.blocking_labels)
        # An incremental scan reuses the guard KV cache, which is only valid while the guarded
        # sequences grow by appending. Beam search reorders them, so it falls back to a full re-scan.
        self.incremental = incremental and guard_model.stateful
        self.report = OVGuardReport()

        self._prompt_length = 0
        self._consumed = 0
        self._cleared_response_tokens = 0
        self._prompt_ids: Optional[torch.Tensor] = None
        self._prompt_attention_mask: Optional[torch.Tensor] = None
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
        """Resets the guard state and starts moderating the prompt.

        A stateful guard model holds the conversation in its KV cache, so a single guard model
        instance cannot back two concurrent sessions.
        """
        self.guard_model.reset_stream()
        self.report = OVGuardReport()
        self._violation = None
        self._prompt_ids = input_ids
        self._prompt_attention_mask = attention_mask
        self._prompt_length = input_ids.shape[-1]
        self._consumed = input_ids.shape[-1]
        self._cleared_response_tokens = 0

        if self.config.prompt_mode == "off":
            if self.incremental:
                # The prompt still has to go through the guard model to prime its KV cache, but its
                # verdicts are discarded.
                self._moderate(input_ids, role=USER_ROLE, token_offset=0, total_length=self._prompt_length)
            return

        if self.config.prompt_mode == "sync":
            self._collect(self._moderate_prompt(input_ids), self.report.prompt)
            return

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ov-guard")
        self._prompt_future = self._executor.submit(self._moderate_prompt, input_ids)

    def _await_prompt(self) -> None:
        if self._prompt_future is None:
            return
        verdicts = self._prompt_future.result()
        self._prompt_future = None
        self._executor.shutdown(wait=False)
        self._executor = None
        self._collect(verdicts, self.report.prompt)

    def update(self, input_ids: torch.Tensor) -> bool:
        """Moderates the tokens generated since the previous call. Returns whether to stop."""
        self._await_prompt()
        if self._violation is not None:
            return self.config.on_violation == "stop"

        pending = input_ids.shape[-1] - self._consumed
        if pending <= 0 or pending < self.config.chunk_size:
            return False
        return self.flush(input_ids)

    def flush(self, input_ids: torch.Tensor) -> bool:
        """Moderates every token generated so far that has not been moderated yet."""
        self._await_prompt()
        if input_ids.shape[-1] <= self._consumed:
            return self._violation is not None and self.config.on_violation == "stop"

        token_offset = self._consumed - self._prompt_length
        if self.incremental:
            chunk = input_ids[:, self._consumed :]
        else:
            # A full re-scan replays the whole sequence, so any KV cache left from the previous
            # chunk has to be dropped first.
            self.guard_model.reset_stream()
            chunk = input_ids
        verdicts = self._moderate(
            chunk, role=ASSISTANT_ROLE, token_offset=token_offset, total_length=input_ids.shape[-1]
        )
        if not self.incremental:
            # A full re-scan returns verdicts for the prompt as well, keep only the new tokens.
            verdicts = [sequence[self._consumed :] for sequence in verdicts]
        self._consumed = input_ids.shape[-1]
        self._collect(verdicts, self.report.response)

        if self._violation is None:
            self._cleared_response_tokens = self._consumed - self._prompt_length
        else:
            # Tokens preceding the first violation are still clean and may be released.
            self._cleared_response_tokens = (
                self._violation.token_index if self._violation.role == ASSISTANT_ROLE else 0
            )
            if self.config.on_violation == "raise":
                raise OVGuardViolationError(self.report)
        return self._violation is not None and self.config.on_violation == "stop"

    def _moderate_prompt(self, input_ids: torch.Tensor) -> List[List[OVGuardVerdict]]:
        # A guard verdict covers the whole prefix seen so far, so a prompt is summarised by the
        # verdict of its last token. Intermediate positions describe truncated prompts and are noisy.
        verdicts = self._moderate(input_ids, role=USER_ROLE, token_offset=0, total_length=self._prompt_length)
        return [sequence[-1:] for sequence in verdicts]

    def _moderate(
        self, input_ids: torch.Tensor, role: str, token_offset: int, total_length: int
    ) -> List[List[OVGuardVerdict]]:
        return self.guard_model.moderate(
            input_ids,
            attention_mask=self._attention_mask(total_length, input_ids.shape[0]),
            role=role,
            blocking_labels=self.blocking_labels,
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

    def _collect(self, verdicts: List[List[OVGuardVerdict]], destination: List[OVGuardVerdict]) -> None:
        for sequence_verdicts in verdicts:
            for verdict in sequence_verdicts:
                destination.append(verdict)
                if self.config.on_verdict is not None:
                    self.config.on_verdict(verdict)
                if verdict.flagged and self._violation is None:
                    self._violation = verdict

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            self._prompt_future = None


class OVGuardStoppingCriteria(StoppingCriteria):
    """Feeds generated tokens to an `OVGuardSession` and stops generation on a violation."""

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


def check_guard_compatibility(target_config: PretrainedConfig, guard_config: PretrainedConfig) -> None:
    """Rejects a guard model whose tokenizer cannot be the one of the guarded model.

    A streaming guard model consumes the token ids produced by the guarded model directly, so both
    have to share a tokenizer.
    """
    target_vocab_size = getattr(target_config.get_text_config(), "vocab_size", None)
    guard_vocab_size = getattr(guard_config.get_text_config(), "vocab_size", None)
    if target_vocab_size is not None and guard_vocab_size is not None and target_vocab_size != guard_vocab_size:
        raise ValueError(
            "The guard model and the guarded model must share a tokenizer, but their vocabulary sizes differ "
            f"({guard_vocab_size} vs {target_vocab_size}). Streaming guard models classify the token ids of the "
            "guarded model directly and cannot re-tokenize them."
        )


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
