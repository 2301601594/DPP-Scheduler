"""Request-level service-deficit DPP v2 scoring and selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dpp_scheduler.contracts import (
    BatchPlan,
    ControlState,
    Decision,
    SafeCandidate,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.settings import DPPSettings


@dataclass(frozen=True)
class DPPScore:
    plan_id: str
    prefill_drift: float
    decode_drift: float
    total_drift: float
    effective_duration: float
    score: float
    prefill_budget: int
    current_prefill_count: int
    current_decode_count: int
    prefill_reference_concurrency: int
    decode_reference_concurrency: int


def _finite_positive(label: str, value: float | None, maximum: float) -> float:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or value > maximum
    ):
        raise ValueError(f"{label} must be finite and in (0, {maximum}]")
    return float(value)


def effective_duration(candidate: SafeCandidate, maximum: float) -> float:
    prediction = candidate.prediction
    if prediction.in_support and prediction.prediction_mode == "INTERPOLATION":
        return _finite_positive(
            "expected_duration", prediction.expected_duration, maximum
        )
    if (
        not prediction.in_support
        and prediction.prediction_mode == "CONSTRAINED_EXTRAPOLATION"
    ):
        return _finite_positive(
            "conservative_duration", prediction.conservative_duration, maximum
        )
    raise ValueError("prediction support flag/mode mismatch")


class DPPSelector:
    """Maximize negative fixed-reference Lyapunov drift per wall-clock second."""

    def __init__(self, settings: DPPSettings) -> None:
        self.settings = settings

    def score_candidate(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        candidate: SafeCandidate,
    ) -> DPPScore:
        validate_snapshot_hash(control.snapshot_hash, snapshot.snapshot_hash)
        validate_snapshot_hash(candidate.snapshot_hash, snapshot.snapshot_hash)
        candidate.plan.validate_snapshot(snapshot)
        prediction = candidate.prediction
        validate_snapshot_hash(prediction.snapshot_hash, snapshot.snapshot_hash)
        if prediction.plan_id != candidate.plan.plan_id:
            raise ValueError("SafeCandidate prediction/plan mismatch")

        tau = effective_duration(candidate, self.settings.maximum_numeric)
        ttft = control.ttft_debt_map()
        tbt = control.tbt_debt_map()
        prefill_by_id = {
            request.request_id: request
            for request in snapshot.waiting_prefill_requests
        }
        decode_by_id = {
            request.request_id: request for request in snapshot.active_decode_requests
        }
        if set(ttft) != set(prefill_by_id):
            raise ValueError("TTFT service-debt keys must equal live Prefill requests")
        if not set(tbt).issubset(decode_by_id):
            raise ValueError("TBT service-debt key is not an active Decode request")
        prefill_service = dict(candidate.plan.prefill_items)
        decode_service = set(candidate.plan.decode_items)

        prefill_changes: list[float] = []
        for request_id in sorted(ttft):
            request = prefill_by_id[request_id]
            if request.ttft_slo_seconds <= 0 or request.token_count <= 0:
                raise ValueError("Prefill SLO/prompt length must be positive")
            current = float(ttft[request_id])
            if not math.isfinite(current) or current < 0:
                raise ValueError("TTFT service debt must be finite and non-negative")
            service = prefill_service.get(request_id, 0) / request.token_count
            predicted = max(
                0.0,
                current + tau / request.ttft_slo_seconds - service,
            )
            prefill_changes.append(predicted * predicted - current * current)

        decode_changes: list[float] = []
        for request_id in sorted(tbt):
            request = decode_by_id[request_id]
            if request.tbt_slo_seconds <= 0:
                raise ValueError("Decode TBT SLO must be positive")
            current = float(tbt[request_id])
            if not math.isfinite(current) or current < 0:
                raise ValueError("TBT service debt must be finite and non-negative")
            service = 1.0 if request_id in decode_service else 0.0
            predicted = max(
                0.0,
                current + tau / request.tbt_slo_seconds - service,
            )
            decode_changes.append(predicted * predicted - current * current)

        prefill_drift = math.fsum(prefill_changes) / (
            2.0 * self.settings.prefill_reference_concurrency
        )
        decode_drift = math.fsum(decode_changes) / (
            2.0 * self.settings.decode_reference_concurrency
        )
        total_drift = prefill_drift + decode_drift
        score = -total_drift / tau
        if not all(
            math.isfinite(value)
            for value in (prefill_drift, decode_drift, total_drift, score)
        ):
            raise ValueError("DPP v2 score is non-finite")
        return DPPScore(
            plan_id=candidate.plan.plan_id,
            prefill_drift=prefill_drift,
            decode_drift=decode_drift,
            total_drift=total_drift,
            effective_duration=tau,
            score=score,
            prefill_budget=candidate.plan.total_prefill_tokens,
            current_prefill_count=len(prefill_by_id),
            current_decode_count=len(decode_by_id),
            prefill_reference_concurrency=(
                self.settings.prefill_reference_concurrency
            ),
            decode_reference_concurrency=self.settings.decode_reference_concurrency,
        )

    def _score_candidates(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
    ) -> tuple[tuple[SafeCandidate, DPPScore], ...]:
        validate_snapshot_hash(control.snapshot_hash, snapshot.snapshot_hash)
        return tuple(
            (candidate, self.score_candidate(snapshot, control, candidate))
            for candidate in safe_candidates
        )

    def _decision_from_scored(
        self,
        snapshot: StateSnapshot,
        scored: tuple[tuple[SafeCandidate, DPPScore], ...],
    ) -> Decision:
        if not scored:
            return Decision(
                frame_id=snapshot.frame_id,
                snapshot_hash=snapshot.snapshot_hash,
                selected_plan=None,
                reason="NO_SAFE_DECISION",
            )
        selected_candidate, selected_score = scored[0]
        for candidate, score in scored[1:]:
            tied = math.isclose(
                score.score,
                selected_score.score,
                rel_tol=self.settings.score_rel_tol,
                abs_tol=self.settings.score_abs_tol,
            )
            if score.score > selected_score.score and not tied:
                selected_candidate, selected_score = candidate, score
                continue
            if tied and (
                score.effective_duration,
                score.prefill_budget,
                score.plan_id,
            ) < (
                selected_score.effective_duration,
                selected_score.prefill_budget,
                selected_score.plan_id,
            ):
                selected_candidate, selected_score = candidate, score
        return Decision(
            frame_id=snapshot.frame_id,
            snapshot_hash=snapshot.snapshot_hash,
            selected_plan=selected_candidate.plan,
            reason="DPP_V2_MAX_DRIFT_RATE",
        )

    def select(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
    ) -> Decision:
        return self._decision_from_scored(
            snapshot, self._score_candidates(snapshot, control, safe_candidates)
        )

    def select_with_audit(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
    ) -> tuple[Decision, tuple[DPPScore, ...]]:
        scored = self._score_candidates(snapshot, control, safe_candidates)
        return self._decision_from_scored(snapshot, scored), tuple(
            score for _, score in scored
        )


class TemporarySelector:
    """Compatibility selector for isolated Adapter tests."""

    def select(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate | BatchPlan, ...],
    ) -> Decision:
        validate_snapshot_hash(control.snapshot_hash, snapshot.snapshot_hash)
        if not safe_candidates:
            return Decision(
                frame_id=snapshot.frame_id,
                snapshot_hash=snapshot.snapshot_hash,
                selected_plan=None,
                reason="NO_SAFE_DECISION",
            )
        plans = tuple(
            item.plan if isinstance(item, SafeCandidate) else item
            for item in safe_candidates
        )
        selected = min(plans, key=lambda plan: plan.plan_id)
        return Decision(
            frame_id=snapshot.frame_id,
            snapshot_hash=snapshot.snapshot_hash,
            selected_plan=selected,
            reason="TEMPORARY_SMALLEST_PLAN_ID",
        )
