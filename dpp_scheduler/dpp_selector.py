"""Two-stage ZERO-relative TBT-constrained Prefill service-rate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dpp_scheduler.consequence_estimator import _misses, obligation_completes
from dpp_scheduler.contracts import (
    BatchPlan,
    ControlState,
    Decision,
    Obligation,
    SafeCandidate,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.settings import DPPSettings


TWO_STAGE_ALGORITHM = "two_stage_zero_relative_tbt_prefill_service_rate_v2b"
TIE_BREAK_ORDER = (
    "prefill_service_rate_desc",
    "effective_duration_asc",
    "prefill_service_tokens_desc",
    "prefill_budget_asc",
    "plan_id_asc",
)


@dataclass(frozen=True)
class TBTRequestSlack:
    request_id: str
    deadline: float
    slack_seconds: float


@dataclass(frozen=True)
class TBTCandidateResult:
    plan_id: str
    effective_duration: float
    risk_duration_seconds: float
    violation_count: int
    zero_violation_count: int
    delta_violation_count: int
    delta_lateness_seconds: float
    passed: bool


@dataclass(frozen=True)
class TBTStageResult:
    status: str
    delta_seconds: float
    request_slacks: tuple[TBTRequestSlack, ...]
    min_slack_seconds: float | None
    duration_limit_seconds: float | None
    maximum_incremental_tbt_violations: int
    candidates: tuple[TBTCandidateResult, ...]
    eligible_plan_ids: tuple[str, ...]
    reference_plan_id: str | None = None
    reference_template_id: str | None = None
    reference_risk_duration_seconds: float | None = None
    reference_violation_count: int | None = None
    zero_reference_resolution: str | None = None
    fallback_plan_id: str | None = None


@dataclass(frozen=True)
class DPPScore:
    """Stage-2 Prefill service-rate result retained under the audit name."""

    plan_id: str
    effective_duration: float
    prefill_service_tokens: int
    prefill_service_rate: float
    score: float
    prefill_budget: int
    current_prefill_count: int
    current_prefill_backlog_tokens: int
    current_decode_count: int
    decode_coverage_complete: bool
    rank: int = 0


@dataclass(frozen=True)
class SelectorAudit:
    algorithm: str
    stage1: TBTStageResult
    stage2_scores: tuple[DPPScore, ...]
    winner_tie_plan_ids: tuple[str, ...]
    selected_plan_id: str | None
    decision_reason: str
    tie_break_order: tuple[str, ...] = TIE_BREAK_ORDER


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


def stage1_risk_duration(candidate: SafeCandidate, maximum: float) -> float:
    """Stage-1 risk duration: always the conservative duration."""
    return _finite_positive(
        "conservative_duration", candidate.prediction.conservative_duration, maximum
    )


class DPPSelector:
    """Filter by ZERO-relative incremental TBT violations, then maximize
    Prefill service rate."""

    def __init__(self, settings: DPPSettings) -> None:
        self.settings = settings

    def _validate_candidate(
        self, snapshot: StateSnapshot, candidate: SafeCandidate
    ) -> None:
        validate_snapshot_hash(candidate.snapshot_hash, snapshot.snapshot_hash)
        candidate.plan.validate_snapshot(snapshot)
        prediction = candidate.prediction
        validate_snapshot_hash(prediction.snapshot_hash, snapshot.snapshot_hash)
        if prediction.plan_id != candidate.plan.plan_id:
            raise ValueError("SafeCandidate prediction/plan mismatch")

    def _tbt_request_slacks(
        self, snapshot: StateSnapshot
    ) -> tuple[TBTRequestSlack, ...]:
        if (
            isinstance(snapshot.timestamp, bool)
            or not isinstance(snapshot.timestamp, (int, float))
            or not math.isfinite(snapshot.timestamp)
        ):
            raise ValueError("snapshot timestamp must be finite")
        decode_by_id = {
            request.request_id: request for request in snapshot.active_decode_requests
        }
        if len(decode_by_id) != len(snapshot.active_decode_requests):
            raise ValueError("active Decode request IDs must be unique")

        obligations: dict[str, float] = {}
        for obligation in snapshot.active_tbt_obligations:
            if obligation.kind != "TBT":
                raise ValueError("active_tbt_obligations contains a non-TBT item")
            if obligation.settled:
                raise ValueError("active TBT obligation must not be settled")
            if obligation.request_id not in decode_by_id:
                raise ValueError("TBT obligation request is not an active Decode")
            if obligation.request_id in obligations:
                raise ValueError("active Decode has duplicate TBT obligations")
            if (
                isinstance(obligation.deadline, bool)
                or not isinstance(obligation.deadline, (int, float))
                or not math.isfinite(obligation.deadline)
            ):
                raise ValueError("TBT deadline must be finite")
            obligations[obligation.request_id] = float(obligation.deadline)

        for request_id, request in decode_by_id.items():
            request_deadline = request.tbt_deadline
            obligation_deadline = obligations.get(request_id)
            if request_deadline is None:
                if obligation_deadline is not None:
                    raise ValueError("Decode TBT deadline/obligation mismatch")
                continue
            if (
                isinstance(request_deadline, bool)
                or not isinstance(request_deadline, (int, float))
                or not math.isfinite(request_deadline)
            ):
                raise ValueError("Decode TBT deadline must be finite or None")
            if obligation_deadline is None or float(request_deadline) != obligation_deadline:
                raise ValueError("Decode TBT deadline/obligation mismatch")

        return tuple(
            TBTRequestSlack(
                request_id=request_id,
                deadline=deadline,
                slack_seconds=deadline - snapshot.timestamp,
            )
            for request_id, deadline in sorted(obligations.items())
        )

    def _zero_reference(
        self,
        snapshot: StateSnapshot,
        candidates_by_id: dict[str, SafeCandidate],
    ) -> tuple[SafeCandidate, str]:
        """Resolve the ZERO Decode-only baseline for delta risk.

        The baseline must plan zero Prefill tokens and cover exactly the full
        active Decode set. Deterministic preference: ZERO template, then STOCK
        identity (canonical dedup preserves STOCK over a materially identical
        ZERO), then any zero-service full-decode candidate.
        """
        active_decode_ids = {
            request.request_id for request in snapshot.active_decode_requests
        }
        ordered = sorted(
            candidates_by_id.values(), key=lambda candidate: candidate.plan.plan_id
        )

        def full_decode_zero(candidate: SafeCandidate) -> bool:
            return (
                candidate.plan.total_prefill_tokens == 0
                and set(candidate.plan.decode_items) == active_decode_ids
            )

        for prefix, resolution in (
            ("ZERO", "ZERO_TEMPLATE"),
            ("STOCK", "STOCK_IDENTITY"),
        ):
            for candidate in ordered:
                template = candidate.plan.template_id
                if not (
                    template == prefix or template.startswith(f"{prefix}:")
                ):
                    continue
                if full_decode_zero(candidate):
                    return candidate, resolution
        for candidate in ordered:
            if full_decode_zero(candidate):
                return candidate, "ZERO_SERVICE_MATCH"
        raise RuntimeError(
            "ZERO_REFERENCE_MISSING: no zero-Prefill full-Decode candidate "
            "exists while active TBT obligations require a risk baseline"
        )

    @staticmethod
    def _candidate_risk(
        snapshot: StateSnapshot,
        candidate: SafeCandidate,
        risk_duration: float,
        obligations: dict[str, Obligation],
        request_slacks: tuple[TBTRequestSlack, ...],
    ) -> tuple[dict[str, bool], dict[str, float]]:
        """Per-obligation TBT miss and predicted lateness under risk duration."""
        misses: dict[str, bool] = {}
        lateness: dict[str, float] = {}
        for item in request_slacks:
            obligation = obligations[item.request_id]
            misses[item.request_id] = _misses(
                snapshot=snapshot,
                plan=candidate.plan,
                obligation=obligation,
                duration=risk_duration,
            )
            lateness[item.request_id] = max(
                0.0, snapshot.timestamp + risk_duration - item.deadline
            )
        return misses, lateness

    def _tbt_stage(
        self,
        snapshot: StateSnapshot,
        safe_candidates: tuple[SafeCandidate, ...],
    ) -> tuple[TBTStageResult, tuple[SafeCandidate, ...], dict[str, float]]:
        request_slacks = self._tbt_request_slacks(snapshot)
        durations: dict[str, float] = {}
        risk_durations: dict[str, float] = {}
        candidates_by_id: dict[str, SafeCandidate] = {}
        for candidate in safe_candidates:
            self._validate_candidate(snapshot, candidate)
            plan_id = candidate.plan.plan_id
            if plan_id in candidates_by_id:
                raise ValueError("SafeCandidate plan IDs must be unique")
            candidates_by_id[plan_id] = candidate
            durations[plan_id] = effective_duration(
                candidate, self.settings.maximum_numeric
            )
            risk_durations[plan_id] = stage1_risk_duration(
                candidate, self.settings.maximum_numeric
            )

        limit = self.settings.maximum_incremental_tbt_violations
        min_slack = (
            min(item.slack_seconds for item in request_slacks)
            if request_slacks
            else None
        )

        if not safe_candidates:
            result = TBTStageResult(
                status="NO_SAFE_CANDIDATES",
                delta_seconds=self.settings.tbt_delta_seconds,
                request_slacks=request_slacks,
                min_slack_seconds=min_slack,
                duration_limit_seconds=None,
                maximum_incremental_tbt_violations=limit,
                candidates=(),
                eligible_plan_ids=(),
            )
            return result, (), durations

        plan_ids = tuple(candidate.plan.plan_id for candidate in safe_candidates)

        if not request_slacks:
            result = TBTStageResult(
                status="NO_ACTIVE_TBT_OBLIGATION",
                delta_seconds=self.settings.tbt_delta_seconds,
                request_slacks=(),
                min_slack_seconds=None,
                duration_limit_seconds=None,
                maximum_incremental_tbt_violations=limit,
                candidates=tuple(
                    TBTCandidateResult(
                        plan_id=plan_id,
                        effective_duration=durations[plan_id],
                        risk_duration_seconds=risk_durations[plan_id],
                        violation_count=0,
                        zero_violation_count=0,
                        delta_violation_count=0,
                        delta_lateness_seconds=0.0,
                        passed=True,
                    )
                    for plan_id in plan_ids
                ),
                eligible_plan_ids=plan_ids,
                zero_reference_resolution="NOT_NEEDED",
            )
            return result, safe_candidates, durations

        zero_ref, resolution = self._zero_reference(snapshot, candidates_by_id)
        obligations = {
            obligation.request_id: obligation
            for obligation in snapshot.active_tbt_obligations
        }
        zero_risk = risk_durations[zero_ref.plan.plan_id]
        zero_misses, zero_lateness = self._candidate_risk(
            snapshot, zero_ref, zero_risk, obligations, request_slacks
        )

        passed_ids: list[str] = []
        candidate_results: list[TBTCandidateResult] = []
        for candidate in safe_candidates:
            risk_duration = risk_durations[candidate.plan.plan_id]
            misses, lateness = self._candidate_risk(
                snapshot, candidate, risk_duration, obligations, request_slacks
            )
            delta_n = sum(
                1
                for item in request_slacks
                if misses[item.request_id] and not zero_misses[item.request_id]
            )
            delta_l = sum(
                max(
                    0.0,
                    lateness[item.request_id] - zero_lateness[item.request_id],
                )
                for item in request_slacks
            )
            passed = delta_n <= limit
            if passed:
                passed_ids.append(candidate.plan.plan_id)
            candidate_results.append(
                TBTCandidateResult(
                    plan_id=candidate.plan.plan_id,
                    effective_duration=durations[candidate.plan.plan_id],
                    risk_duration_seconds=risk_duration,
                    violation_count=sum(misses.values()),
                    zero_violation_count=sum(zero_misses.values()),
                    delta_violation_count=delta_n,
                    delta_lateness_seconds=delta_l,
                    passed=passed,
                )
            )

        result = TBTStageResult(
            status="DELTA_N_ADMITTED",
            delta_seconds=self.settings.tbt_delta_seconds,
            request_slacks=request_slacks,
            min_slack_seconds=min_slack,
            duration_limit_seconds=None,
            maximum_incremental_tbt_violations=limit,
            candidates=tuple(candidate_results),
            eligible_plan_ids=tuple(passed_ids),
            reference_plan_id=zero_ref.plan.plan_id,
            reference_template_id=zero_ref.plan.template_id,
            reference_risk_duration_seconds=zero_risk,
            reference_violation_count=sum(zero_misses.values()),
            zero_reference_resolution=resolution,
        )
        return (
            result,
            tuple(candidates_by_id[plan_id] for plan_id in passed_ids),
            durations,
        )

    def _score_one(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        candidate: SafeCandidate,
        tau: float,
        *,
        capture_request_details: bool,
    ) -> DPPScore:
        del control, capture_request_details
        prefill_by_id = {
            request.request_id: request
            for request in snapshot.waiting_prefill_requests
        }
        if len(prefill_by_id) != len(snapshot.waiting_prefill_requests):
            raise ValueError("live Prefill request IDs must be unique")
        prefill_items = candidate.plan.prefill_items
        if len(prefill_items) != len({item[0] for item in prefill_items}):
            raise ValueError("BatchPlan contains duplicate Prefill service")
        if not {item[0] for item in prefill_items}.issubset(prefill_by_id):
            raise ValueError("BatchPlan Prefill service targets a non-live request")
        for request_id, tokens in prefill_items:
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise ValueError("BatchPlan Prefill service must be a positive integer")
            if tokens > prefill_by_id[request_id].remaining_tokens:
                raise ValueError("BatchPlan Prefill service exceeds remaining tokens")

        service_tokens = candidate.plan.total_prefill_tokens
        if isinstance(service_tokens, bool) or not isinstance(service_tokens, int):
            raise ValueError("Prefill service tokens must be an integer")
        if service_tokens < 0:
            raise ValueError("Prefill service tokens must be non-negative")
        if service_tokens != sum(tokens for _, tokens in prefill_items):
            raise ValueError("BatchPlan total Prefill tokens mismatch")
        service_rate = service_tokens / tau
        if not math.isfinite(service_rate):
            raise ValueError("Prefill service rate must be finite")

        backlog_tokens = 0
        for request in snapshot.waiting_prefill_requests:
            remaining = request.remaining_tokens
            if isinstance(remaining, bool) or not isinstance(remaining, int):
                raise ValueError("Prefill remaining tokens must be an integer")
            if remaining <= 0:
                raise ValueError("live Prefill request must have remaining tokens")
            backlog_tokens += remaining

        active_decode_ids = {
            request.request_id for request in snapshot.active_decode_requests
        }
        planned_decode_ids = set(candidate.plan.decode_items)
        return DPPScore(
            plan_id=candidate.plan.plan_id,
            effective_duration=tau,
            prefill_service_tokens=service_tokens,
            prefill_service_rate=service_rate,
            score=service_rate,
            prefill_budget=service_tokens,
            current_prefill_count=len(snapshot.waiting_prefill_requests),
            current_prefill_backlog_tokens=backlog_tokens,
            current_decode_count=len(snapshot.active_decode_requests),
            decode_coverage_complete=planned_decode_ids == active_decode_ids,
        )

    @staticmethod
    def _tie_key(score: DPPScore) -> tuple[object, ...]:
        return (
            score.effective_duration,
            -score.prefill_service_tokens,
            score.prefill_budget,
            score.plan_id,
        )

    def _rank_scores(
        self, scores: tuple[DPPScore, ...]
    ) -> tuple[tuple[DPPScore, ...], tuple[str, ...]]:
        remaining = sorted(
            scores,
            key=lambda score: (-score.score, score.plan_id),
        )
        ranked: list[DPPScore] = []
        winner_tie: tuple[str, ...] = ()
        while remaining:
            leader = remaining[0]
            group = [
                score
                for score in remaining
                if (
                    (score.score == 0.0) == (leader.score == 0.0)
                    and math.isclose(
                        score.score,
                        leader.score,
                        rel_tol=self.settings.score_rel_tol,
                        abs_tol=self.settings.score_abs_tol,
                    )
                )
            ]
            group_ids = {score.plan_id for score in group}
            remaining = [
                score for score in remaining if score.plan_id not in group_ids
            ]
            ordered_group = sorted(group, key=self._tie_key)
            if not ranked:
                winner_tie = tuple(score.plan_id for score in ordered_group)
            for score in ordered_group:
                ranked.append(replace(score, rank=len(ranked) + 1))
        return tuple(ranked), winner_tie

    def _run(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
        *,
        capture_request_details: bool,
    ) -> tuple[Decision, SelectorAudit]:
        validate_snapshot_hash(control.snapshot_hash, snapshot.snapshot_hash)
        stage1, eligible, durations = self._tbt_stage(snapshot, safe_candidates)
        if not eligible:
            decision = Decision(
                frame_id=snapshot.frame_id,
                snapshot_hash=snapshot.snapshot_hash,
                selected_plan=None,
                reason="NO_SAFE_DECISION",
            )
            return decision, SelectorAudit(
                algorithm=TWO_STAGE_ALGORITHM,
                stage1=stage1,
                stage2_scores=(),
                winner_tie_plan_ids=(),
                selected_plan_id=None,
                decision_reason=decision.reason,
            )

        scored = tuple(
            self._score_one(
                snapshot,
                control,
                candidate,
                durations[candidate.plan.plan_id],
                capture_request_details=capture_request_details,
            )
            for candidate in eligible
        )
        ranked, winner_tie = self._rank_scores(scored)
        winner = ranked[0]
        if any(score.prefill_service_tokens > 0 for score in ranked) and (
            winner.prefill_service_tokens == 0
        ):
            raise RuntimeError(
                "ZERO candidate selected while an eligible candidate performs "
                "actual Prefill work"
            )
        candidate_by_id = {
            candidate.plan.plan_id: candidate for candidate in eligible
        }
        reason = "TWO_STAGE_TBT_PREFILL_SERVICE_RATE"
        decision = Decision(
            frame_id=snapshot.frame_id,
            snapshot_hash=snapshot.snapshot_hash,
            selected_plan=candidate_by_id[winner.plan_id].plan,
            reason=reason,
        )
        return decision, SelectorAudit(
            algorithm=TWO_STAGE_ALGORITHM,
            stage1=stage1,
            stage2_scores=ranked,
            winner_tie_plan_ids=winner_tie,
            selected_plan_id=winner.plan_id,
            decision_reason=reason,
        )

    def score_candidate(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        candidate: SafeCandidate,
        *,
        capture_request_details: bool = False,
    ) -> DPPScore:
        validate_snapshot_hash(control.snapshot_hash, snapshot.snapshot_hash)
        self._validate_candidate(snapshot, candidate)
        return self._score_one(
            snapshot,
            control,
            candidate,
            effective_duration(candidate, self.settings.maximum_numeric),
            capture_request_details=capture_request_details,
        )

    def select(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
    ) -> Decision:
        return self._run(
            snapshot,
            control,
            safe_candidates,
            capture_request_details=False,
        )[0]

    def select_with_audit(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
        *,
        capture_request_details: bool = False,
    ) -> tuple[Decision, SelectorAudit]:
        return self._run(
            snapshot,
            control,
            safe_candidates,
            capture_request_details=capture_request_details,
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
