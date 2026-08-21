"""Consequence estimator placeholder.

The real estimator attaches obligation-level TTFT/TBT consequences to plans and
is required for Safe-Set/DPP.  G2 leaves the interface as a deterministic stub.
"""

from __future__ import annotations

from typing import Iterable

from dpp_scheduler.contracts import BatchPlan, Prediction, StateSnapshot


class NullConsequenceEstimator:
    def attach(
        self,
        snapshot: StateSnapshot,
        plans: Iterable[BatchPlan],
        predictions: Iterable[Prediction],
    ) -> tuple[Prediction, ...]:
        return tuple(predictions)
