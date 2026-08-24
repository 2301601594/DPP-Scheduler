"""Deterministic physical feasibility and conservative SLO-risk filtering."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterable

from dpp_scheduler.candidate_generator import (
    project_kv_blocks,
    project_sequence_count,
)
from dpp_scheduler.contracts import (
    BatchPlan,
    Prediction,
    SafeCandidate,
    SafeSetResult,
    StateSnapshot,
)
from dpp_scheduler.settings import SafeSetSettings


TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
SEQUENCE_BUDGET_EXCEEDED = "SEQUENCE_BUDGET_EXCEEDED"
CURRENT_KV_EXCEEDED = "CURRENT_KV_EXCEEDED"
ROLLING_KV_EXCEEDED = "ROLLING_KV_EXCEEDED"
PREDICTOR_OUT_OF_SUPPORT = "PREDICTOR_OUT_OF_SUPPORT"
PREDICTION_INVALID = "PREDICTION_INVALID"
CONSEQUENCE_MISSING = "CONSEQUENCE_MISSING"
SLO_RISK_WHEN_ZERO_AVAILABLE = "SLO_RISK_WHEN_ZERO_VIOLATION_AVAILABLE"
SLO_RISK_OUTSIDE_TOP_K = "SLO_RISK_OUTSIDE_TOP_K"


def _index_inputs(
    snapshot: StateSnapshot,
    plans: Iterable[BatchPlan],
    predictions: Iterable[Prediction],
) -> tuple[tuple[BatchPlan, ...], dict[str, Prediction]]:
    plan_tuple = tuple(plans)
    plan_ids: set[str] = set()
    for plan in plan_tuple:
        plan.validate_snapshot(snapshot)
        if plan.plan_id in plan_ids:
            raise ValueError("duplicate candidate plan_id")
        plan_ids.add(plan.plan_id)
    by_plan: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("prediction snapshot_hash mismatch")
        if prediction.plan_id not in plan_ids:
            raise ValueError("prediction references an unknown plan_id")
        if prediction.plan_id in by_plan:
            raise ValueError("duplicate prediction plan_id")
        by_plan[prediction.plan_id] = prediction
    if set(by_plan) != plan_ids:
        raise ValueError("predictions must cover candidates exactly once")
    return plan_tuple, by_plan


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("kv_block_size must be positive")
    return (value + divisor - 1) // divisor


def rolling_kv_reserve_blocks(
    snapshot: StateSnapshot, plan: BatchPlan, horizon: int
) -> int:
    """Project bounded Decode growth without using eventual output length."""
    if horizon < 0:
        raise ValueError("Rolling KV horizon must be non-negative")
    selected_decode = set(plan.decode_items)
    contexts: list[int] = [
        request.kv_context_length + (request.request_id in selected_decode)
        for request in snapshot.active_decode_requests
    ]

    # A prompt completed by this plan may become a Decode request next round.
    # Reserving its bounded growth is conservative and remains length-blind.
    scheduled_prefill = dict(plan.prefill_items)
    for request in snapshot.waiting_prefill_requests:
        scheduled = scheduled_prefill.get(request.request_id, 0)
        if request.remaining_tokens > 0 and scheduled >= request.remaining_tokens:
            contexts.append(request.token_count)

    block_size = snapshot.kv_block_size
    return sum(
        _ceil_div(context + horizon, block_size)
        - _ceil_div(context, block_size)
        for context in contexts
    )


def _safe_candidate(
    snapshot: StateSnapshot, plan: BatchPlan, prediction: Prediction
) -> SafeCandidate:
    count = prediction.predicted_violation_count
    lateness = prediction.predicted_total_lateness_seconds
    if count is None or lateness is None:
        raise ValueError("prediction has no attached consequence")
    return SafeCandidate(
        snapshot_hash=snapshot.snapshot_hash,
        plan=plan,
        prediction=prediction,
        predicted_violation_count=count,
        predicted_total_lateness_seconds=lateness,
        conservative_deadline_margin_seconds=(
            prediction.conservative_deadline_margin_seconds
        ),
    )


def hard_rejection_reasons(
    snapshot: StateSnapshot,
    plan: BatchPlan,
    prediction: Prediction,
    settings: SafeSetSettings,
) -> tuple[str, ...]:
    """Validate one plan and return only physical/Predictor hard failures."""
    plan.validate_snapshot(snapshot)
    if prediction.snapshot_hash != snapshot.snapshot_hash:
        raise ValueError("prediction snapshot_hash mismatch")
    if prediction.plan_id != plan.plan_id:
        raise ValueError("prediction/plan mismatch")

    prefill_ids = [request_id for request_id, _ in plan.prefill_items]
    decode_ids = list(plan.decode_items)
    if len(prefill_ids) != len(set(prefill_ids)):
        raise ValueError("duplicate Prefill request in BatchPlan")
    if len(decode_ids) != len(set(decode_ids)):
        raise ValueError("duplicate Decode request in BatchPlan")
    if set(prefill_ids).intersection(decode_ids):
        raise ValueError("request cannot be Prefill and Decode in one BatchPlan")
    computed_prefill = sum(tokens for _, tokens in plan.prefill_items)
    computed_decode = len(plan.decode_items)
    if computed_prefill != plan.total_prefill_tokens:
        raise ValueError("BatchPlan Prefill token total mismatch")
    if computed_decode != plan.total_decode_tokens:
        raise ValueError("BatchPlan Decode token total mismatch")
    if computed_prefill < 0 or any(tokens <= 0 for _, tokens in plan.prefill_items):
        raise ValueError("BatchPlan Prefill token count must be positive")
    computed_sequences = project_sequence_count(snapshot, plan.prefill_items)
    if computed_sequences != plan.total_sequences:
        raise ValueError("BatchPlan sequence projection mismatch")
    computed_kv = project_kv_blocks(snapshot, plan.prefill_items, plan.decode_items)
    if computed_kv != plan.projected_kv_blocks:
        raise ValueError("BatchPlan KV projection mismatch")

    reasons: list[str] = []
    if computed_prefill + computed_decode > snapshot.token_budget:
        reasons.append(TOKEN_BUDGET_EXCEEDED)
    if computed_sequences > snapshot.sequence_budget:
        reasons.append(SEQUENCE_BUDGET_EXCEEDED)
    if computed_kv > snapshot.total_kv_blocks:
        reasons.append(CURRENT_KV_EXCEEDED)
    rolling = rolling_kv_reserve_blocks(
        snapshot, plan, settings.rolling_kv_horizon_iterations
    )
    if computed_kv + settings.reserve_blocks_r0 + rolling > snapshot.total_kv_blocks:
        reasons.append(ROLLING_KV_EXCEEDED)
    if not prediction.in_support:
        reasons.append(PREDICTOR_OUT_OF_SUPPORT)
    expected = prediction.expected_duration
    conservative = prediction.conservative_duration
    if (
        expected is None
        or conservative is None
        or not math.isfinite(expected)
        or not math.isfinite(conservative)
        or expected <= 0
        or conservative <= 0
        or conservative < expected
    ):
        reasons.append(PREDICTION_INVALID)
    return tuple(reasons)


class SafeSet(ABC):
    @abstractmethod
    def filter(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> SafeSetResult:
        raise NotImplementedError

    @abstractmethod
    def hard_rejection_reasons(
        self,
        snapshot: StateSnapshot,
        plan: BatchPlan,
        prediction: Prediction,
    ) -> tuple[str, ...]:
        raise NotImplementedError


class PassThroughSafeSet(SafeSet):
    """G2 temporary safe set: admits every well-formed candidate."""

    def filter(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> SafeSetResult:
        plan_tuple, by_plan = _index_inputs(snapshot, plans, predictions)
        return SafeSetResult(
            snapshot_hash=snapshot.snapshot_hash,
            safe_candidates=tuple(
                SafeCandidate(
                    snapshot_hash=snapshot.snapshot_hash,
                    plan=plan,
                    prediction=by_plan[plan.plan_id],
                    predicted_violation_count=(
                        by_plan[plan.plan_id].predicted_violation_count or 0
                    ),
                    predicted_total_lateness_seconds=(
                        by_plan[plan.plan_id].predicted_total_lateness_seconds or 0.0
                    ),
                    conservative_deadline_margin_seconds=(
                        by_plan[plan.plan_id].conservative_deadline_margin_seconds
                    ),
                )
                for plan in plan_tuple
            ),
        )

    def hard_rejection_reasons(
        self,
        snapshot: StateSnapshot,
        plan: BatchPlan,
        prediction: Prediction,
    ) -> tuple[str, ...]:
        plan.validate_snapshot(snapshot)
        if prediction.snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("prediction snapshot_hash mismatch")
        if prediction.plan_id != plan.plan_id:
            raise ValueError("prediction/plan mismatch")
        return ()


class ResourceAndRiskSafeSet(SafeSet):
    """G4 Safe-Set: hard physical/Predictor filters with risk metadata."""

    def __init__(self, settings: SafeSetSettings) -> None:
        self.settings = settings

    def hard_rejection_reasons(
        self,
        snapshot: StateSnapshot,
        plan: BatchPlan,
        prediction: Prediction,
    ) -> tuple[str, ...]:
        return hard_rejection_reasons(snapshot, plan, prediction, self.settings)

    def filter(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> SafeSetResult:
        plan_tuple, by_plan = _index_inputs(snapshot, plans, predictions)
        feasible: list[SafeCandidate] = []
        rejected: list[tuple[str, tuple[str, ...]]] = []

        for plan in plan_tuple:
            prediction = by_plan[plan.plan_id]
            reasons = list(self.hard_rejection_reasons(snapshot, plan, prediction))
            if (
                prediction.predicted_violation_count is None
                or prediction.predicted_total_lateness_seconds is None
                or prediction.predicted_violation_count < 0
                or not math.isfinite(prediction.predicted_total_lateness_seconds)
                or prediction.predicted_total_lateness_seconds < 0
            ):
                reasons.append(CONSEQUENCE_MISSING)

            if reasons:
                rejected.append((plan.plan_id, tuple(reasons)))
                continue
            feasible.append(_safe_candidate(snapshot, plan, prediction))

        # Risk remains auditable metadata on SafeCandidate.  It must not
        # shrink the DPP action space: every physically/Predictor-feasible
        # plan reaches the Selector, even when its predicted risk is nonzero.
        admitted = tuple(feasible)

        return SafeSetResult(
            snapshot_hash=snapshot.snapshot_hash,
            safe_candidates=admitted,
            rejected=tuple(rejected),
        )
