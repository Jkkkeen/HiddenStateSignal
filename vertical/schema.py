from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


PayloadKind = Literal["raw", "pooled"]


@dataclass(frozen=True)
class HiddenPayload:
    kind: PayloadKind
    values: np.ndarray
    representation: str | None = None
    endpoints: np.ndarray | None = None
    progress: np.ndarray | None = None
    response_start: int | None = None
    response_stop: int | None = None
    layer_kind: np.ndarray | None = None

    def validate(self, metadata: RecordMetadata, *, strict_finite: bool = True) -> None:
        values = np.asarray(self.values)
        if self.kind not in ("raw", "pooled"):
            raise ValueError(f"unknown payload kind: {self.kind}")
        if values.ndim != 3:
            raise ValueError("payload values must have shape (token_or_endpoint, layer, dim)")
        if values.shape[1] != metadata.hidden_state_count:
            raise ValueError(
                "payload hidden_state_count does not match metadata: "
                f"{values.shape[1]} != {metadata.hidden_state_count}"
            )
        if values.shape[2] != metadata.hidden_dimension:
            raise ValueError(
                "payload hidden_dimension does not match metadata: "
                f"{values.shape[2]} != {metadata.hidden_dimension}"
            )
        if strict_finite and not np.isfinite(values).all():
            raise ValueError("payload contains non-finite values")
        if self.layer_kind is not None and len(self.layer_kind) != values.shape[1]:
            raise ValueError("layer_kind must align with hidden-state positions")

        if self.kind == "raw":
            if self.representation is not None:
                raise ValueError("raw payload must not declare a pooled representation")
            has_start = self.response_start is not None
            has_stop = self.response_stop is not None
            if has_start != has_stop:
                raise ValueError("response_start and response_stop must be supplied together")
            if has_start:
                start = int(self.response_start or 0)
                stop = int(self.response_stop or 0)
                if not 0 <= start < stop <= values.shape[0]:
                    raise ValueError("invalid response slice")
                if stop - start != metadata.response_token_count:
                    raise ValueError("response slice does not match response_token_count")
            elif values.shape[0] != metadata.response_token_count:
                raise ValueError(
                    "response-only raw payload length does not match response_token_count"
                )
            return

        if not self.representation:
            raise ValueError("pooled payload requires a representation")
        endpoints = None if self.endpoints is None else np.asarray(self.endpoints)
        progress = None if self.progress is None else np.asarray(self.progress)
        if endpoints is None and progress is None:
            raise ValueError("pooled payload requires endpoints or progress")
        if endpoints is not None:
            if endpoints.ndim != 1 or len(endpoints) != values.shape[0]:
                raise ValueError("endpoints must align with pooled trajectory rows")
            if np.any(endpoints < 1) or np.any(endpoints > metadata.response_token_count):
                raise ValueError("pooled endpoint falls outside response tokens")
        if progress is not None:
            if progress.ndim != 1 or len(progress) != values.shape[0]:
                raise ValueError("progress must align with pooled trajectory rows")
            if not np.isfinite(progress).all() or np.any(progress <= 0) or np.any(progress > 1):
                raise ValueError("progress must be finite and fall in (0, 1]")


@dataclass(frozen=True)
class RecordMetadata:
    record_id: str
    model_family: str
    model_name: str
    condition: str
    question_id: str
    rollout_id: str
    is_correct: bool | None
    response_token_count: int
    num_decoder_layers: int
    hidden_state_count: int
    hidden_dimension: int
    source_path: str
    checkpoint: str | None = None
    global_step: int | None = None
    training_progress: float | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "record_id": self.record_id,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "condition": self.condition,
            "question_id": self.question_id,
            "rollout_id": self.rollout_id,
            "source_path": self.source_path,
        }
        for field_name, value in required.items():
            if not str(value).strip():
                raise ValueError(f"{field_name} must be nonempty")
        for field_name, value in (
            ("response_token_count", self.response_token_count),
            ("num_decoder_layers", self.num_decoder_layers),
            ("hidden_state_count", self.hidden_state_count),
            ("hidden_dimension", self.hidden_dimension),
        ):
            if int(value) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.hidden_state_count < self.num_decoder_layers + 1:
            raise ValueError("hidden_state_count must include embedding plus decoder layers")
        if self.global_step is not None and self.global_step < 0:
            raise ValueError("global_step must be nonnegative")
        if self.training_progress is not None and not 0 <= self.training_progress <= 1:
            raise ValueError("training_progress must fall in [0, 1]")


@dataclass(frozen=True)
class VerticalRecord:
    metadata: RecordMetadata
    payloads: Mapping[str, HiddenPayload]

    def validate(self, *, strict_finite: bool = True) -> None:
        self.metadata.validate()
        if not self.payloads:
            raise ValueError("record must contain at least one hidden payload")
        for name, payload in self.payloads.items():
            if not str(name).strip():
                raise ValueError("payload names must be nonempty")
            payload.validate(self.metadata, strict_finite=strict_finite)
            if payload.kind == "pooled" and payload.representation != name:
                raise ValueError("pooled payload mapping key must match representation")

