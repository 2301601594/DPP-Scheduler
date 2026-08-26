"""Deterministic Controller-owned Fallback construction and admission."""

from __future__ import annotations

from dpp_scheduler.candidate_generator import (
    highest_bindable_prefill,
    project_kv_blocks,
    project_sequence_count,
)
from dpp_scheduler.contracts import (
    BatchPlan,
    FallbackResult,
    Prediction,
    StateSnapshot,
)
from dpp_scheduler.predictor import DurationPredictor
from dpp_scheduler.safe_set import (
    CURRENT_KV_EXCEEDED,
    PREDICTION_INVALID,
    PREDICTOR_OUT_OF_SUPPORT,
    ROLLING_KV_EXCEEDED,
    SEQUENCE_BUDGET_EXCEEDED,
    SafeSet,
)
from dpp_scheduler.settings import FallbackSettings


FALLBACK_DECODE_ONLY = "FALLBACK_DECODE_ONLY"
FALLBACK_MINIMUM_PREFILL = "FALLBACK_MINIMUM_PREFILL"
IDLE_EMPTY_QUEUE = "IDLE_EMPTY_QUEUE"
IDLE_NO_DECODE_TOKEN_BUDGET = "IDLE_NO_DECODE_TOKEN_BUDGET"
IDLE_NO_BINDABLE_PREFILL = "IDLE_NO_BINDABLE_PREFILL"
IDLE_PREFILL_CHUNK_BELOW_MINIMUM = "IDLE_PREFILL_CHUNK_BELOW_MINIMUM"
IDLE_FALLBACK_REJECTED = "IDLE_FALLBACK_REJECTED"
PREEMPTION_REQUIRED_MANDATORY_DECODE = "PREEMPTION_REQUIRED_MANDATORY_DECODE"
PREEMPTION_REQUIRED_AFTER_FALLBACK_REJECTION = (
    "PREEMPTION_REQUIRED_AFTER_FALLBACK_REJECTION"
)


LIVENESS_ESCAPE_DECODE = "LIVENESS_ESCAPE_DECODE"
LIVENESS_ESCAPE_PREFILL = "LIVENESS_ESCAPE_PREFILL"
PREEMPTION_REQUIRED_NATIVE_PROGRESS = "PREEMPTION_REQUIRED_NATIVE_PROGRESS"


class NullFallback:
    """Explicit no-op used by G2-only Controller tests."""

    def build(self, snapshot: StateSnapshot) -> FallbackResult:
        return FallbackResult(
            snapshot_hash=snapshot.snapshot_hash,
            plan=None,
            prediction=None,
            reason=IDLE_EMPTY_QUEUE,
        )


class DeterministicFallback:
    """Build the single frozen Decode or Prefill fallback action."""

    def __init__(self, settings: FallbackSettings) -> None:
        self.settings = settings

    def build(self, snapshot: StateSnapshot) -> FallbackResult:
        if snapshot.active_decode_requests:
            return self._build_decode(snapshot)
        if snapshot.waiting_prefill_requests:
            return self._build_prefill(snapshot)
        return FallbackResult(
            snapshot_hash=snapshot.snapshot_hash,
            plan=None,
            prediction=None,
            reason=IDLE_EMPTY_QUEUE,
        )

    def _build_decode(self, snapshot: StateSnapshot) -> FallbackResult:
        # Fallback owns an EDF Decode-only policy independently of the normal
        # V3 Candidate Generator's all-Decode stable arrival order.
        ordered = tuple(
            sorted(
                snapshot.active_decode_requests,
                key=lambda request: (
                    request.tbt_deadline is None,
                    (
                        request.tbt_deadline
                        if request.tbt_deadline is not None
                        else float("inf")
                    ),
                    request.arrival_time,
                    request.ordinal,
                    request.request_id,
                ),
            )
        )
        limit = min(len(ordered), snapshot.token_budget)
        if limit <= 0:
            return FallbackResult(
                snapshot_hash=snapshot.snapshot_hash,
                plan=None,
                prediction=None,
                reason=IDLE_NO_DECODE_TOKEN_BUDGET,
            )
        selected = ordered[:limit]
        selected_ids = {request.request_id for request in selected}
        mandatory = {request.request_id for request in ordered if request.mandatory}
        if not mandatory.issubset(selected_ids):
            return FallbackResult(
                snapshot_hash=snapshot.snapshot_hash,
                plan=None,
                prediction=None,
                reason=PREEMPTION_REQUIRED_MANDATORY_DECODE,
            )
        decode_items = tuple(request.request_id for request in selected)
        prefill_items: tuple[tuple[str, int], ...] = ()
        plan = BatchPlan(
            plan_id="fallback-decode-edf",
            snapshot_hash=snapshot.snapshot_hash,
            template_id="FALLBACK:DECODE_ONLY:EDF",
            prefill_items=prefill_items,
            decode_items=decode_items,
            total_prefill_tokens=0,
            total_decode_tokens=len(decode_items),
            total_sequences=project_sequence_count(snapshot, prefill_items),
            projected_kv_blocks=project_kv_blocks(
                snapshot, prefill_items, decode_items
            ),
            mandatory_request_ids=tuple(
                request.request_id for request in selected if request.mandatory
            ),
        )
        return FallbackResult(
            snapshot_hash=snapshot.snapshot_hash,
            plan=plan,
            prediction=None,
            reason=FALLBACK_DECODE_ONLY,
        )

    def _build_prefill(self, snapshot: StateSnapshot) -> FallbackResult:
        request = highest_bindable_prefill(snapshot)
        if request is None:
            return FallbackResult(
                snapshot_hash=snapshot.snapshot_hash,
                plan=None,
                prediction=None,
                reason=IDLE_NO_BINDABLE_PREFILL,
            )
        minimum = self.settings.minimum_prefill_chunk_tokens
        if request.remaining_tokens <= minimum:
            scheduled = request.remaining_tokens
        elif snapshot.token_budget >= minimum:
            scheduled = minimum
        else:
            return FallbackResult(
                snapshot_hash=snapshot.snapshot_hash,
                plan=None,
                prediction=None,
                reason=IDLE_PREFILL_CHUNK_BELOW_MINIMUM,
            )
        if scheduled <= 0 or scheduled > snapshot.token_budget:
            return FallbackResult(
                snapshot_hash=snapshot.snapshot_hash,
                plan=None,
                prediction=None,
                reason=IDLE_PREFILL_CHUNK_BELOW_MINIMUM,
            )
        prefill_items = ((request.request_id, scheduled),)
        decode_items: tuple[str, ...] = ()
        plan = BatchPlan(
            plan_id="fallback-prefill-minimum",
            snapshot_hash=snapshot.snapshot_hash,
            template_id=f"FALLBACK:PREFILL_MINIMUM:{minimum}",
            prefill_items=prefill_items,
            decode_items=decode_items,
            total_prefill_tokens=scheduled,
            total_decode_tokens=0,
            total_sequences=project_sequence_count(snapshot, prefill_items),
            projected_kv_blocks=project_kv_blocks(
                snapshot, prefill_items, decode_items
            ),
            mandatory_request_ids=(),
        )
        return FallbackResult(
            snapshot_hash=snapshot.snapshot_hash,
            plan=plan,
            prediction=None,
            reason=FALLBACK_MINIMUM_PREFILL,
        )


def build_liveness_escape(
    snapshot: StateSnapshot,
    resolved: FallbackResult,
) -> FallbackResult:
    """Admit only a physically safe fallback rejected by the Predictor.

    Normal fallback resolution has already applied every hard check. The
    liveness escape may bypass Predictor support/validity only; resource and
    Rolling-KV failures remain explicit native-preemption requests.
    """
    if resolved.snapshot_hash != snapshot.snapshot_hash:
        raise ValueError("Fallback snapshot_hash mismatch")
    workload_nonempty = bool(
        snapshot.active_decode_requests or snapshot.waiting_prefill_requests
    )
    if resolved.plan is None:
        if not workload_nonempty:
            return resolved
        return FallbackResult(
            snapshot_hash=snapshot.snapshot_hash,
            plan=None,
            prediction=None,
            reason=PREEMPTION_REQUIRED_NATIVE_PROGRESS,
            rejection_reasons=resolved.rejection_reasons or (resolved.reason,),
        )
    predictor_only = {PREDICTOR_OUT_OF_SUPPORT, PREDICTION_INVALID}
    if not resolved.rejection_reasons or not set(
        resolved.rejection_reasons
    ).issubset(predictor_only):
        return FallbackResult(
            snapshot_hash=snapshot.snapshot_hash,
            plan=None,
            prediction=resolved.prediction,
            reason=PREEMPTION_REQUIRED_NATIVE_PROGRESS,
            rejection_reasons=resolved.rejection_reasons,
        )
    reason = (
        LIVENESS_ESCAPE_DECODE
        if resolved.plan.total_decode_tokens > 0
        else LIVENESS_ESCAPE_PREFILL
    )
    return FallbackResult(
        snapshot_hash=snapshot.snapshot_hash,
        plan=resolved.plan,
        prediction=resolved.prediction,
        reason=reason,
        rejection_reasons=resolved.rejection_reasons,
    )


def resolve_fallback(
    snapshot: StateSnapshot,
    fallback: DeterministicFallback | NullFallback,
    predictor: DurationPredictor,
    safe_set: SafeSet,
) -> FallbackResult:
    """Predict and hard-check Fallback without invoking DPP risk/scoring."""
    built = fallback.build(snapshot)
    if built.snapshot_hash != snapshot.snapshot_hash:
        raise ValueError("Fallback snapshot_hash mismatch")
    if built.plan is None:
        return built
    predictions = predictor.predict(snapshot, (built.plan,))
    if len(predictions) != 1:
        raise ValueError("Predictor must return exactly one Fallback prediction")
    prediction: Prediction = predictions[0]
    reasons = safe_set.hard_rejection_reasons(snapshot, built.plan, prediction)
    if not reasons:
        return FallbackResult(
            snapshot_hash=snapshot.snapshot_hash,
            plan=built.plan,
            prediction=prediction,
            reason=built.reason,
        )
    resource_reasons = {
        SEQUENCE_BUDGET_EXCEEDED,
        CURRENT_KV_EXCEEDED,
        ROLLING_KV_EXCEEDED,
    }
    reason = (
        PREEMPTION_REQUIRED_AFTER_FALLBACK_REJECTION
        if resource_reasons.intersection(reasons)
        else IDLE_FALLBACK_REJECTED
    )
    return FallbackResult(
        snapshot_hash=snapshot.snapshot_hash,
        plan=built.plan,
        prediction=prediction,
        reason=reason,
        rejection_reasons=reasons,
    )
