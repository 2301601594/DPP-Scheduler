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
        for plan in plan_tuple:
            if plan.snapshot_hash != snapshot.snapshot_hash:
                raise ValueError("candidate snapshot_hash mismatch")
        return SafeSetResult(
            snapshot_hash=snapshot.snapshot_hash,
            candidates=plan_tuple,
        )
