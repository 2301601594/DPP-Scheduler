"""Public immutable decision-round contracts for the modular DPP scheduler.

All structures in one scheduling round must carry the same ``snapshot_hash``.
The hash is a deterministic SHA-256 over the versioned snapshot payload.  No
module outside ``vllm_adapter`` is allowed to depend on vLLM internal types.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TypeVar

SCHEMA_VERSION = 2

T = TypeVar("T")


def _canonical_value(value: Any) -> Any:
    """Convert dataclasses and sets into stable JSON-friendly structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _canonical_value(getattr(value, f.name))
            for f in fields(value)
            if f.name != "snapshot_hash"
        }
    if isinstance(value, dict):
        return {str(key): _canonical_value(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical_value(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    """Return canonical JSON for a dataclass or mapping."""
    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":")
    )


def compute_snapshot_hash(
    frame_id: int,
    timestamp: float,
    waiting_prefill_requests: tuple[PrefillRequest, ...],
    active_decode_requests: tuple[DecodeRequest, ...],
    active_ttft_obligations: tuple[Obligation, ...],
    active_tbt_obligations: tuple[Obligation, ...],
    recovery_requests: tuple[str, ...],
    free_kv_blocks: int,
    kv_block_size: int,
    token_budget: int,
    sequence_budget: int,
    total_kv_blocks: int,
    *,
    provenance: str,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    payload = {
        "schema_version": schema_version,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "waiting_prefill_requests": waiting_prefill_requests,
        "active_decode_requests": active_decode_requests,
        "active_ttft_obligations": active_ttft_obligations,
        "active_tbt_obligations": active_tbt_obligations,
        "recovery_requests": recovery_requests,
        "free_kv_blocks": free_kv_blocks,
        "kv_block_size": kv_block_size,
        "token_budget": token_budget,
        "sequence_budget": sequence_budget,
        "total_kv_blocks": total_kv_blocks,
        "provenance": provenance,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PrefillRequest:
    """A request waiting for (more) prefill processing."""

    request_id: str
    arrival_time: float
    token_count: int
    prefilled_tokens: int
    ttft_deadline: float | None = None
    hard_ttft_protected: bool = False
    # True means that vLLM already admitted the request into its running set.
    # It therefore consumes a sequence slot even if this plan does not select
    # another prompt chunk for it.
    is_running: bool = False
    # Partial-prefill ordering uses the original index as a stable tie key.
    ordinal: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.token_count - self.prefilled_tokens)

    @property
    def is_partial(self) -> bool:
        return self.prefilled_tokens > 0


@dataclass(frozen=True)
class DecodeRequest:
    """An active decode request that can emit at most one token per iteration."""

    request_id: str
    arrival_time: float
    kv_context_length: int
    tbt_deadline: float | None = None
    recovery_due: bool = False
    recovery_first_miss_time: float | None = None
    mandatory: bool = False
    ordinal: int = 0


@dataclass(frozen=True)
class Obligation:
    """A TTFT or TBT obligation that must be settled exactly once."""

    obligation_id: str
    request_id: str
    kind: str  # "TTFT" or "TBT"
    deadline: float
    created_at: float
    settled: bool = False


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable image of the scheduler-relevant vLLM state."""

    frame_id: int
    timestamp: float
    snapshot_hash: str
    waiting_prefill_requests: tuple[PrefillRequest, ...]
    active_decode_requests: tuple[DecodeRequest, ...]
    active_ttft_obligations: tuple[Obligation, ...]
    active_tbt_obligations: tuple[Obligation, ...]
    recovery_requests: tuple[str, ...]
    free_kv_blocks: int
    kv_block_size: int
    token_budget: int
    sequence_budget: int
    total_kv_blocks: int
    provenance: str = "test"
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        frame_id: int,
        timestamp: float,
        waiting_prefill_requests: tuple[PrefillRequest, ...],
        active_decode_requests: tuple[DecodeRequest, ...],
        active_ttft_obligations: tuple[Obligation, ...],
        active_tbt_obligations: tuple[Obligation, ...],
        recovery_requests: tuple[str, ...],
        free_kv_blocks: int,
        kv_block_size: int,
        token_budget: int,
        sequence_budget: int,
        total_kv_blocks: int,
        *,
        provenance: str = "test",
        schema_version: int = SCHEMA_VERSION,
    ) -> StateSnapshot:
        snapshot_hash = compute_snapshot_hash(
            frame_id,
            timestamp,
            waiting_prefill_requests,
            active_decode_requests,
            active_ttft_obligations,
            active_tbt_obligations,
            recovery_requests,
            free_kv_blocks,
            kv_block_size,
            token_budget,
            sequence_budget,
            total_kv_blocks,
            provenance=provenance,
            schema_version=schema_version,
        )
        return cls(
            frame_id=frame_id,
            timestamp=timestamp,
            snapshot_hash=snapshot_hash,
            waiting_prefill_requests=waiting_prefill_requests,
            active_decode_requests=active_decode_requests,
            active_ttft_obligations=active_ttft_obligations,
            active_tbt_obligations=active_tbt_obligations,
            recovery_requests=recovery_requests,
            free_kv_blocks=free_kv_blocks,
            kv_block_size=kv_block_size,
            token_budget=token_budget,
            sequence_budget=sequence_budget,
            total_kv_blocks=total_kv_blocks,
            provenance=provenance,
            schema_version=schema_version,
        )


@dataclass(frozen=True)
class BatchPlan:
    """A complete, atomic set of prefill and decode work for one iteration."""

    plan_id: str
    snapshot_hash: str
    template_id: str
    prefill_items: tuple[tuple[str, int], ...]
    decode_items: tuple[str, ...]
    total_prefill_tokens: int
    total_decode_tokens: int
    total_sequences: int
    projected_kv_blocks: int
    mandatory_request_ids: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def validate_snapshot(self, snapshot: StateSnapshot) -> None:
        validate_snapshot_hash(self.snapshot_hash, snapshot.snapshot_hash)


@dataclass(frozen=True)
class Prediction:
    """Predictor output for a BatchPlan."""

    plan_id: str
    snapshot_hash: str
    expected_duration: float | None
    conservative_duration: float | None
    in_support: bool
    predictor_version: str | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class ControlState:
    """DPP control state, containing only the three declared control variables."""

    snapshot_hash: str
    prefill_backlog: int
    ttft_debt: float
    tbt_debt: float
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class SafeSetResult:
    """Result container reserved for G4; G2 may pass all candidates through."""

    snapshot_hash: str
    safe_candidates: tuple[BatchPlan, ...]
    rejected: tuple[tuple[str, tuple[str, ...]], ...] = ()
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class ExecutionObservation:
    """Actual execution result returned by the Adapter."""

    frame_id: int
    snapshot_hash: str
    executed_plan_id: str | None
    executed_prefill_items: tuple[tuple[str, int], ...]
    executed_decode_items: tuple[str, ...]
    started_at: float
    finished_at: float | None = None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION

    def matches(self, plan: BatchPlan) -> bool:
        return (
            self.executed_plan_id == plan.plan_id
            and self.executed_prefill_items == plan.prefill_items
            and self.executed_decode_items == plan.decode_items
        )


@dataclass(frozen=True)
class Decision:
    """The decision produced by the Controller for one round."""

    frame_id: int
    snapshot_hash: str
    selected_plan: BatchPlan | None
    reason: str
    schema_version: int = SCHEMA_VERSION


def validate_snapshot_hash(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(
            f"snapshot_hash mismatch: plan/decision has {actual}, "
            f"snapshot has {expected}"
        )


def contracts_equal(first: T, second: T) -> bool:
    return canonical_json(first) == canonical_json(second)
