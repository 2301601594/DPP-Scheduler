"""Deterministic temporary selector for G2.

This selector intentionally does not implement DPP scoring.  It is a stable,
auditable placeholder that lets the exact BatchPlan path be tested before the
real DPP Selector is added in G5/G6.
"""

from __future__ import annotations

from dpp_scheduler.contracts import BatchPlan, Decision, StateSnapshot, validate_snapshot_hash


class TemporarySelector:
    """Select the smallest plan_id from a deterministic candidate set."""

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

        selected = min(
            safe_candidates,
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
            reason="TEMPORARY_SMALLEST_PLAN_ID",
        )
