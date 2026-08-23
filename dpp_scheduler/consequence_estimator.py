"""Deterministic TTFT/TBT consequence projection for complete BatchPlans."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

from dpp_scheduler.contracts import (
    BatchPlan,
    Obligation,
    Prediction,
    StateSnapshot,
)


def _indexed_inputs(
    snapshot: StateSnapshot,
    plans: Iterable[BatchPlan],
    predictions: Iterable[Prediction],
) -> tuple[tuple[BatchPlan, ...], dict[str, Prediction]]:
    plan_tuple = tuple(plans)
    prediction_tuple = tuple(predictions)
    plan_ids: set[str] = set()
    for plan in plan_tuple:
        plan.validate_snapshot(snapshot)
        if plan.plan_id in plan_ids:
            raise ValueError("duplicate candidate plan_id")
        plan_ids.add(plan.plan_id)

    by_plan: dict[str, Prediction] = {}
    for prediction in prediction_tuple:
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


def _active_obligations(snapshot: StateSnapshot) -> tuple[Obligation, ...]:
    obligations = snapshot.active_ttft_obligations + snapshot.active_tbt_obligations
    seen: set[str] = set()
    for obligation in obligations:
        if obligation.obligation_id in seen:
            raise ValueError("duplicate obligation_id")
        seen.add(obligation.obligation_id)
        if obligation.settled:
            raise ValueError("active obligation is already settled")
        if obligation.kind not in {"TTFT", "TBT"}:
            raise ValueError(f"unknown obligation kind: {obligation.kind}")
        if not math.isfinite(obligation.deadline):
            raise ValueError("obligation deadline must be finite")
    if any(item.kind != "TTFT" for item in snapshot.active_ttft_obligations):
        raise ValueError("non-TTFT obligation in active_ttft_obligations")
    if any(item.kind != "TBT" for item in snapshot.active_tbt_obligations):
        raise ValueError("non-TBT obligation in active_tbt_obligations")
    return obligations


def obligation_completes(
    snapshot: StateSnapshot, plan: BatchPlan, obligation: Obligation
) -> bool:
    """Project whether the exact plan services an obligation this iteration.

    A selected Decode request emits at most one token in the locked v1 design.
    For TTFT, only a Prefill item that consumes the request's complete remaining
    prompt is completion-capable. This is a scheduling projection; actual
    settlement remains owned by the Adapter/Observer and client return event.
    """
    if obligation.kind == "TBT":
        return obligation.request_id in plan.decode_items
    prefill = {
        request.request_id: request for request in snapshot.waiting_prefill_requests
    }
    request = prefill.get(obligation.request_id)
    if request is None:
        return False
    scheduled = dict(plan.prefill_items).get(obligation.request_id, 0)
    return request.remaining_tokens > 0 and scheduled >= request.remaining_tokens


def _misses(
    *, snapshot: StateSnapshot, plan: BatchPlan, obligation: Obligation, duration: float
) -> bool:
    end = snapshot.timestamp + duration
    if obligation_completes(snapshot, plan, obligation):
        return end > obligation.deadline
    return end >= obligation.deadline


class ConsequenceEstimator:
    """Attach expected service counts and conservative SLO risk to predictions."""

    def attach(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> tuple[Prediction, ...]:
        plan_tuple, by_plan = _indexed_inputs(snapshot, plans, predictions)
        obligations = _active_obligations(snapshot)
        result: list[Prediction] = []
        for plan in plan_tuple:
            prediction = by_plan[plan.plan_id]
            expected = prediction.expected_duration
            conservative = prediction.conservative_duration
            valid = (
                prediction.in_support
                and expected is not None
                and conservative is not None
                and math.isfinite(expected)
                and math.isfinite(conservative)
                and expected > 0
                and conservative > 0
                and conservative >= expected
            )
            if not valid:
                result.append(prediction)
                continue

            expected_counts = {
                "TTFT": {"success": 0, "miss": 0},
                "TBT": {"success": 0, "miss": 0},
            }
            conservative_violations = 0
            conservative_lateness = 0.0
            conservative_margins: list[float] = []
            conservative_end = snapshot.timestamp + conservative

            for obligation in obligations:
                completes = obligation_completes(snapshot, plan, obligation)
                expected_miss = _misses(
                    snapshot=snapshot,
                    plan=plan,
                    obligation=obligation,
                    duration=expected,
                )
                if expected_miss:
                    expected_counts[obligation.kind]["miss"] += 1
                elif completes:
                    expected_counts[obligation.kind]["success"] += 1

                conservative_margins.append(obligation.deadline - conservative_end)
                if _misses(
                    snapshot=snapshot,
                    plan=plan,
                    obligation=obligation,
                    duration=conservative,
                ):
                    conservative_violations += 1
                    conservative_lateness += max(
                        0.0, conservative_end - obligation.deadline
                    )

            result.append(
                replace(
                    prediction,
                    ttft_success=expected_counts["TTFT"]["success"],
                    ttft_miss=expected_counts["TTFT"]["miss"],
                    tbt_success=expected_counts["TBT"]["success"],
                    tbt_miss=expected_counts["TBT"]["miss"],
                    predicted_violation_count=conservative_violations,
                    predicted_total_lateness_seconds=conservative_lateness,
                    conservative_deadline_margin_seconds=(
                        min(conservative_margins)
                        if conservative_margins
                        else None
                    ),
                    # Immediate service utility is obligation-level and is not
                    # terminal request-level Goodput.
                    service_utility=float(
                        expected_counts["TTFT"]["success"]
                        + expected_counts["TBT"]["success"]
                    ),
                )
            )
        return tuple(result)


class NullConsequenceEstimator:
    """Compatibility stub for the temporary G2 pass-through path."""

    def attach(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> tuple[Prediction, ...]:
        del snapshot, plans
        return tuple(predictions)
