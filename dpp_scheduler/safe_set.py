"""Safe-Set interface placeholder.

The complete Safe-Set with hard resource filters, Rolling KV, SLO risk ranking,
and Fallback is G4.  G2 uses a deterministic pass-through so the Controller can
exercise the whole exact-plan path without inventing unsafe filters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from dpp_scheduler.contracts import (
    BatchPlan,
    Prediction,
    SafeSetResult,
    StateSnapshot,
)


class SafeSet(ABC):
    @abstractmethod
    def filter(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> SafeSetResult:
        raise NotImplementedError


class PassThroughSafeSet(SafeSet):
    """G2 temporary safe set: admits every well-formed candidate."""

    def filter(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> SafeSetResult:
        plan_tuple = tuple(plans)
        prediction_tuple = tuple(predictions)
        for plan in plan_tuple:
            if plan.snapshot_hash != snapshot.snapshot_hash:
                raise ValueError("candidate snapshot_hash mismatch")
        plan_ids = {plan.plan_id for plan in plan_tuple}
        if len(plan_ids) != len(plan_tuple):
            raise ValueError("duplicate candidate plan_id")
        prediction_ids: set[str] = set()
        for prediction in prediction_tuple:
            if prediction.snapshot_hash != snapshot.snapshot_hash:
                raise ValueError("prediction snapshot_hash mismatch")
            if prediction.plan_id not in plan_ids:
                raise ValueError("prediction references an unknown plan_id")
            if prediction.plan_id in prediction_ids:
                raise ValueError("duplicate prediction plan_id")
            prediction_ids.add(prediction.plan_id)
        if prediction_ids != plan_ids:
            raise ValueError("predictions must cover candidates exactly once")
        return SafeSetResult(
            snapshot_hash=snapshot.snapshot_hash,
            safe_candidates=plan_tuple,
        )
