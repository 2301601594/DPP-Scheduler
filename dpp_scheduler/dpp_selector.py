"""Deterministic temporary selector used until the G5 DPP score is frozen."""

from __future__ import annotations

from dpp_scheduler.contracts import (
    BatchPlan,
    Decision,
    SafeCandidate,
    StateSnapshot,
    validate_snapshot_hash,
)


class TemporarySelector:
    """Select a deterministic non-idle plan from the safe candidate set."""

    def select(
        self,
        snapshot: StateSnapshot,
        safe_candidates: tuple[SafeCandidate | BatchPlan, ...],
    ) -> Decision:
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
                # Controller-owned Fallback remains independent of DPP scoring.
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
