"""Deterministic temporary selector used until the G5 DPP score is frozen."""

from __future__ import annotations

from dpp_scheduler.contracts import (
    BatchPlan,
    Decision,
    StateSnapshot,
    validate_snapshot_hash,
)


class TemporarySelector:
    """Select a deterministic non-idle plan from the safe candidate set."""

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

        non_idle = tuple(
            plan
            for plan in safe_candidates
            if plan.total_prefill_tokens + plan.total_decode_tokens > 0
        )
        pool = non_idle or safe_candidates
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
