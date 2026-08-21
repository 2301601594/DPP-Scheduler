"""Deterministic temporary selector for G2.

This selector intentionally does not implement DPP scoring.  It is a stable,
auditable placeholder that lets the exact BatchPlan path be tested before the
real DPP Selector is added in G5/G6.

For a usable smoke test it prefers non-idle plans: an empty BatchPlan would
otherwise be selected by the smallest plan_id and would permanently leave
waiting work unscheduled.
"""

from __future__ import annotations

from dpp_scheduler.contracts import BatchPlan, Decision, StateSnapshot, validate_snapshot_hash


class TemporarySelector:
    """Select a deterministic non-idle plan from the candidate set."""

    def __init__(self) -> None:
        pass

    def select(
        self, snapshot: StateSnapshot, safe_candidates: tuple[BatchPlan, ...]
    ) -> Decision:
        if not safe_candidates:
            return Decision(
                frame_id=snapshot.frame_id,
                snapshot_hash=snapshot.snapshot_hash,
                selected_plan=None,
                reason="NO_SAFE_DECISION",
            )

        for plan in safe_candidates:
            validate_snapshot_hash(plan.snapshot_hash, snapshot.snapshot_hash)

        def plan_key(plan: BatchPlan) -> tuple:
            return (
                plan.plan_id,
                plan.template_id,
                plan.prefill_items,
                plan.decode_items,
            )

        non_idle = [
            plan for plan in safe_candidates
            if plan.total_prefill_tokens + plan.total_decode_tokens > 0
        ]
        pool = non_idle if non_idle else list(safe_candidates)
        selected = min(pool, key=plan_key)
        reason = (
            "TEMPORARY_SMALLEST_NON_IDLE_PLAN_ID"
            if non_idle
            else "TEMPORARY_SMALLEST_PLAN_ID"
        )
        return Decision(
            frame_id=snapshot.frame_id,
            snapshot_hash=snapshot.snapshot_hash,
            selected_plan=selected,
            reason=reason,
        )
