"""Deterministic normalized DPP scoring and selection."""

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
    """Auditable terms for one SafeCandidate's normalized score."""

    plan_id: str
    prefill_term: float
    ttft_term: float
    tbt_term: float
    utility_term: float
    numerator: float
    expected_duration: float
    score: float
    predicted_misses: int
    conservative_deadline_margin_seconds: float | None


def _require_nonnegative(
    label: str, value: int | float | None, maximum: float
) -> float:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > maximum
    ):
        raise ValueError(f"{label} must be finite and in [0, {maximum}]")
    return float(value)


class DPPSelector:
    """Select the maximum authoritative DPP score with frozen tie-breaks."""

    def __init__(self, settings: DPPSettings) -> None:
        self.settings = settings

    def score_candidate(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        candidate: SafeCandidate,
    ) -> DPPScore:
        maximum = self.settings.maximum_numeric
        validate_snapshot_hash(control.snapshot_hash, snapshot.snapshot_hash)
        validate_snapshot_hash(candidate.snapshot_hash, snapshot.snapshot_hash)
        candidate.plan.validate_snapshot(snapshot)
        prediction = candidate.prediction
        validate_snapshot_hash(prediction.snapshot_hash, snapshot.snapshot_hash)
        if prediction.plan_id != candidate.plan.plan_id:
            raise ValueError("SafeCandidate prediction/plan mismatch")

        expected = _require_nonnegative(
            "expected_duration", prediction.expected_duration, maximum
        )
        if expected == 0.0:
            raise ValueError("expected_duration must be positive")
        prefill_backlog = _require_nonnegative(
            "prefill_backlog", control.prefill_backlog, maximum
        )
        ttft_debt = _require_nonnegative("ttft_debt", control.ttft_debt, maximum)
        tbt_debt = _require_nonnegative("tbt_debt", control.tbt_debt, maximum)
        prefill_service = _require_nonnegative(
            "total_prefill_tokens", candidate.plan.total_prefill_tokens, maximum
        )
        ttft_success = _require_nonnegative(
            "ttft_success", prediction.ttft_success, maximum
        )
        ttft_miss = _require_nonnegative("ttft_miss", prediction.ttft_miss, maximum)
        tbt_success = _require_nonnegative(
            "tbt_success", prediction.tbt_success, maximum
        )
        tbt_miss = _require_nonnegative("tbt_miss", prediction.tbt_miss, maximum)
        utility = _require_nonnegative(
            "service_utility", prediction.service_utility, maximum
        )

        if ttft_success + ttft_miss > len(snapshot.active_ttft_obligations):
            raise ValueError("TTFT consequences exceed active obligations")
        if tbt_success + tbt_miss > len(snapshot.active_tbt_obligations):
            raise ValueError("TBT consequences exceed active obligations")
        if utility != ttft_success + tbt_success:
            raise ValueError("service_utility must equal TTFT plus TBT successes")

        misses = int(ttft_miss + tbt_miss)
        if misses != ttft_miss + tbt_miss:
            raise ValueError("predicted misses must be integral")
        margin = candidate.conservative_deadline_margin_seconds
        if margin is not None and (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(margin)
            or abs(margin) > maximum
        ):
            raise ValueError("conservative deadline margin must be finite")

        token_scale = float(self.settings.token_normalization)
        obligation_scale = float(self.settings.obligation_normalization)
        prefill_term = (
            prefill_backlog / token_scale
        ) * (prefill_service / token_scale)
        ttft_term = (ttft_debt / obligation_scale) * (
            self.settings.epsilon_ttft * (ttft_success / obligation_scale)
            - (1.0 - self.settings.epsilon_ttft)
            * (ttft_miss / obligation_scale)
        )
        tbt_term = (tbt_debt / obligation_scale) * (
            self.settings.epsilon_tbt * (tbt_success / obligation_scale)
            - (1.0 - self.settings.epsilon_tbt)
            * (tbt_miss / obligation_scale)
        )
        utility_term = self.settings.weight_v * (utility / obligation_scale)
        numerator = math.fsum((prefill_term, ttft_term, tbt_term, utility_term))
        score = numerator / expected
        for label, value in (
            ("prefill_term", prefill_term),
            ("ttft_term", ttft_term),
            ("tbt_term", tbt_term),
            ("utility_term", utility_term),
            ("numerator", numerator),
            ("score", score),
        ):
            if not math.isfinite(value) or abs(value) > maximum:
                raise ValueError(f"{label} is outside the frozen numeric range")

        return DPPScore(
            plan_id=candidate.plan.plan_id,
            prefill_term=prefill_term,
            ttft_term=ttft_term,
            tbt_term=tbt_term,
            utility_term=utility_term,
            numerator=numerator,
            expected_duration=expected,
            score=score,
            predicted_misses=misses,
            conservative_deadline_margin_seconds=margin,
        )

    def select(
        self,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
    ) -> Decision:
        validate_snapshot_hash(control.snapshot_hash, snapshot.snapshot_hash)
        if not safe_candidates:
            return Decision(
                frame_id=snapshot.frame_id,
                snapshot_hash=snapshot.snapshot_hash,
                selected_plan=None,
                reason="NO_SAFE_DECISION",
            )

        scored = tuple(
            (candidate, self.score_candidate(snapshot, control, candidate))
            for candidate in safe_candidates
        )
        selected, _ = min(
            scored,
            key=lambda item: (
                -item[1].score,
                item[1].predicted_misses,
                -(
                    item[1].conservative_deadline_margin_seconds
                    if item[1].conservative_deadline_margin_seconds is not None
                    else math.inf
                ),
                item[0].plan.plan_id,
            ),
        )
        return Decision(
            frame_id=snapshot.frame_id,
            snapshot_hash=snapshot.snapshot_hash,
            selected_plan=selected.plan,
            reason="DPP_MAX_SCORE",
        )


class TemporarySelector:
    """Compatibility selector retained for isolated G2 tests."""

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

        plans: list[BatchPlan] = []
        for candidate in safe_candidates:
            if isinstance(candidate, SafeCandidate):
                validate_snapshot_hash(
                    candidate.snapshot_hash, snapshot.snapshot_hash
                )
                validate_snapshot_hash(
                    candidate.prediction.snapshot_hash, snapshot.snapshot_hash
                )
                if candidate.prediction.plan_id != candidate.plan.plan_id:
                    raise ValueError("SafeCandidate prediction/plan mismatch")
                plan = candidate.plan
            else:
                plan = candidate
            validate_snapshot_hash(plan.snapshot_hash, snapshot.snapshot_hash)
            plans.append(plan)

        non_idle = tuple(
            plan
            for plan in plans
            if plan.total_prefill_tokens + plan.total_decode_tokens > 0
        )
        pool = non_idle or tuple(plans)
        selected = min(
            pool,
            key=lambda plan: (
                plan.plan_id,
                plan.template_id,
                plan.prefill_items,
                plan.decode_items,
            ),
        )
        return Decision(
            frame_id=snapshot.frame_id,
            snapshot_hash=snapshot.snapshot_hash,
            selected_plan=selected,
            reason=(
                "TEMPORARY_SMALLEST_NON_IDLE_PLAN_ID"
                if non_idle
                else "TEMPORARY_SMALLEST_PLAN_ID"
            ),
        )
