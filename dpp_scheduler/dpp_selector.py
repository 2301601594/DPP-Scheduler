"""Two-stage TBT-constrained TTFT-DPP selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dpp_scheduler.contracts import (
    BatchPlan,
    ControlState,
    Decision,
    SafeCandidate,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.settings import DPPSettings


TWO_STAGE_ALGORITHM = "two_stage_tbt_ttft_v1"
TIE_BREAK_ORDER = (
    "ttft_score_desc",
    "completed_prefill_count_desc",
    "prefill_progress_desc",
    "effective_duration_asc",
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
    duration_limit: float | None
    passed: bool
    selected_by_fallback: bool = False
    rejection_reason: str | None = None


@dataclass(frozen=True)
class TBTStageResult:
    status: str
    delta_seconds: float
    request_slacks: tuple[TBTRequestSlack, ...]
    min_slack_seconds: float | None
    duration_limit_seconds: float | None
    candidates: tuple[TBTCandidateResult, ...]
    eligible_plan_ids: tuple[str, ...]
    fallback_plan_id: str | None = None


@dataclass(frozen=True)
class TTFTRequestResult:
    request_id: str
    current_debt: float
    prompt_tokens: int
    remaining_tokens: int
    scheduled_tokens: int
    completion_this_frame: bool
    ttft_slo_seconds: float
    duration_increment: float
    normalized_service: float
    predicted_next_debt: float
    current_debt_squared: float
    next_debt_squared: float
    drift_contribution: float


@dataclass(frozen=True)
class DPPScore:
    """TTFT-only candidate result retained under the historical audit name."""

    plan_id: str
    prefill_drift: float
    effective_duration: float
    score: float
    prefill_budget: int
    current_prefill_count: int
    current_decode_count: int
    prefill_reference_concurrency: int
    prefill_progress: float
    completed_prefill_count: int
    request_results: tuple[TTFTRequestResult, ...] = ()
    rank: int = 0

    @property
    def normalized_ttft_drift(self) -> float:
        return self.prefill_drift


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


class DPPSelector:
    """Filter by live TBT time, then maximize TTFT drift reduction rate."""

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

    def _tbt_stage(
        self,
        snapshot: StateSnapshot,
        safe_candidates: tuple[SafeCandidate, ...],
    ) -> tuple[TBTStageResult, tuple[SafeCandidate, ...], dict[str, float]]:
        request_slacks = self._tbt_request_slacks(snapshot)
        durations: dict[str, float] = {}
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

        if not safe_candidates:
            result = TBTStageResult(
                status="NO_SAFE_CANDIDATES",
                delta_seconds=self.settings.tbt_delta_seconds,
                request_slacks=request_slacks,
                min_slack_seconds=(
                    min(item.slack_seconds for item in request_slacks)
                    if request_slacks
                    else None
                ),
                duration_limit_seconds=None,
                candidates=(),
                eligible_plan_ids=(),
            )
            return result, (), durations

        if not request_slacks:
            plan_ids = tuple(candidate.plan.plan_id for candidate in safe_candidates)
            result = TBTStageResult(
                status="NO_ACTIVE_TBT_OBLIGATION",
                delta_seconds=self.settings.tbt_delta_seconds,
                request_slacks=(),
                min_slack_seconds=None,
                duration_limit_seconds=None,
                candidates=tuple(
                    TBTCandidateResult(
                        plan_id=plan_id,
                        effective_duration=durations[plan_id],
                        duration_limit=None,
                        passed=True,
                    )
                    for plan_id in plan_ids
                ),
                eligible_plan_ids=plan_ids,
            )
            return result, safe_candidates, durations

        min_slack = min(item.slack_seconds for item in request_slacks)
        duration_limit = min_slack + self.settings.tbt_delta_seconds
        passed_ids = tuple(
            candidate.plan.plan_id
            for candidate in safe_candidates
            if durations[candidate.plan.plan_id] <= duration_limit
        )
        if passed_ids:
            passed_set = set(passed_ids)
            result = TBTStageResult(
                status="WITHIN_SLACK",
                delta_seconds=self.settings.tbt_delta_seconds,
                request_slacks=request_slacks,
                min_slack_seconds=min_slack,
                duration_limit_seconds=duration_limit,
                candidates=tuple(
                    TBTCandidateResult(
                        plan_id=candidate.plan.plan_id,
                        effective_duration=durations[candidate.plan.plan_id],
                        duration_limit=duration_limit,
                        passed=candidate.plan.plan_id in passed_set,
                        rejection_reason=(
                            None
                            if candidate.plan.plan_id in passed_set
                            else "EFFECTIVE_DURATION_EXCEEDS_TBT_LIMIT"
                        ),
                    )
                    for candidate in safe_candidates
                ),
                eligible_plan_ids=passed_ids,
            )
            return (
                result,
                tuple(candidates_by_id[plan_id] for plan_id in passed_ids),
                durations,
            )

        fallback = min(
            safe_candidates,
            key=lambda candidate: (
                durations[candidate.plan.plan_id],
                candidate.plan.total_prefill_tokens,
                candidate.plan.plan_id,
            ),
        )
        fallback_id = fallback.plan.plan_id
        result = TBTStageResult(
            status="NO_CANDIDATE_WITHIN_SLACK",
            delta_seconds=self.settings.tbt_delta_seconds,
            request_slacks=request_slacks,
            min_slack_seconds=min_slack,
            duration_limit_seconds=duration_limit,
            candidates=tuple(
                TBTCandidateResult(
                    plan_id=candidate.plan.plan_id,
                    effective_duration=durations[candidate.plan.plan_id],
                    duration_limit=duration_limit,
                    passed=False,
                    selected_by_fallback=candidate.plan.plan_id == fallback_id,
                    rejection_reason="EFFECTIVE_DURATION_EXCEEDS_TBT_LIMIT",
                )
                for candidate in safe_candidates
            ),
            eligible_plan_ids=(fallback_id,),
            fallback_plan_id=fallback_id,
        )
        return result, (fallback,), durations

    def _score_one(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        candidate: SafeCandidate,
        tau: float,
        *,
        capture_request_details: bool,
    ) -> DPPScore:
        ttft = control.ttft_debt_map()
        prefill_by_id = {
            request.request_id: request
            for request in snapshot.waiting_prefill_requests
        }
        if len(prefill_by_id) != len(snapshot.waiting_prefill_requests):
            raise ValueError("live Prefill request IDs must be unique")
        if set(ttft) != set(prefill_by_id):
            raise ValueError("TTFT service-debt keys must equal live Prefill requests")

        if len(candidate.plan.prefill_items) != len(
            {request_id for request_id, _ in candidate.plan.prefill_items}
        ):
            raise ValueError("BatchPlan contains duplicate Prefill service")
        prefill_service = dict(candidate.plan.prefill_items)
        if not set(prefill_service).issubset(prefill_by_id):
            raise ValueError("BatchPlan Prefill service targets a non-live request")

        contributions: list[float] = []
        details: list[TTFTRequestResult] = []
        completed = 0
        progress_terms: list[float] = []
        for request_id in sorted(ttft):
            request = prefill_by_id[request_id]
            if (
                isinstance(request.ttft_slo_seconds, bool)
                or not isinstance(request.ttft_slo_seconds, (int, float))
                or not math.isfinite(request.ttft_slo_seconds)
                or request.ttft_slo_seconds <= 0
                or isinstance(request.token_count, bool)
                or not isinstance(request.token_count, int)
                or request.token_count <= 0
            ):
                raise ValueError("Prefill SLO/prompt length must be positive")
            if (
                isinstance(request.prefilled_tokens, bool)
                or not isinstance(request.prefilled_tokens, int)
                or request.prefilled_tokens < 0
                or request.remaining_tokens <= 0
            ):
                raise ValueError("live Prefill request must have remaining tokens")
            raw_current = ttft[request_id]
            if (
                isinstance(raw_current, bool)
                or not isinstance(raw_current, (int, float))
                or not math.isfinite(raw_current)
                or raw_current < 0
            ):
                raise ValueError("TTFT service debt must be finite and non-negative")
            current = float(raw_current)
            service = prefill_service.get(request_id, 0)
            if isinstance(service, bool) or not isinstance(service, int) or service < 0:
                raise ValueError("Prefill service must be a non-negative integer")
            normalized_service = service / request.token_count
            duration_increment = tau / request.ttft_slo_seconds
            completion = service >= request.remaining_tokens
            predicted = (
                0.0
                if completion
                else max(0.0, current + duration_increment - normalized_service)
            )
            if completion:
                completed += 1
            current_squared = current * current
            next_squared = predicted * predicted
            contribution = next_squared - current_squared
            contributions.append(contribution)
            progress_terms.append(normalized_service)
            if capture_request_details:
                details.append(
                    TTFTRequestResult(
                        request_id=request_id,
                        current_debt=current,
                        prompt_tokens=request.token_count,
                        remaining_tokens=request.remaining_tokens,
                        scheduled_tokens=service,
                        completion_this_frame=completion,
                        ttft_slo_seconds=request.ttft_slo_seconds,
                        duration_increment=duration_increment,
                        normalized_service=normalized_service,
                        predicted_next_debt=predicted,
                        current_debt_squared=current_squared,
                        next_debt_squared=next_squared,
                        drift_contribution=contribution,
                    )
                )

        prefill_drift = math.fsum(contributions) / (
            2.0 * self.settings.prefill_reference_concurrency
        )
        score = -prefill_drift / tau
        prefill_progress = math.fsum(progress_terms)
        if not all(
            math.isfinite(value)
            for value in (prefill_drift, score, prefill_progress)
        ):
            raise ValueError("two-stage TTFT score is non-finite")
        return DPPScore(
            plan_id=candidate.plan.plan_id,
            prefill_drift=prefill_drift,
            effective_duration=tau,
            score=score,
            prefill_budget=candidate.plan.total_prefill_tokens,
            current_prefill_count=len(prefill_by_id),
            current_decode_count=len(snapshot.active_decode_requests),
            prefill_reference_concurrency=self.settings.prefill_reference_concurrency,
            prefill_progress=prefill_progress,
            completed_prefill_count=completed,
            request_results=tuple(details),
        )

    @staticmethod
    def _tie_key(score: DPPScore) -> tuple[object, ...]:
        return (
            -score.completed_prefill_count,
            -score.prefill_progress,
            score.effective_duration,
            score.prefill_budget,
            score.plan_id,
        )

    def _rank_scores(
        self, scores: tuple[DPPScore, ...]
    ) -> tuple[tuple[DPPScore, ...], tuple[str, ...]]:
        remaining = sorted(scores, key=lambda score: (-score.score, score.plan_id))
        ranked: list[DPPScore] = []
        winner_tie: tuple[str, ...] = ()
        while remaining:
            leader = remaining[0]
            group = [
                score
                for score in remaining
                if math.isclose(
                    score.score,
                    leader.score,
                    rel_tol=self.settings.score_rel_tol,
                    abs_tol=self.settings.score_abs_tol,
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
        candidate_by_id = {
            candidate.plan.plan_id: candidate for candidate in eligible
        }
        reason = (
            "TBT_NO_CANDIDATE_MIN_DURATION"
            if stage1.status == "NO_CANDIDATE_WITHIN_SLACK"
            else "TWO_STAGE_TBT_TTFT"
        )
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
