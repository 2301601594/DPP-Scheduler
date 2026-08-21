"""Predictor interface and a G2 non-predicting placeholder.

The real shallow Random-Forest predictor is G3.  G2 only needs the public
contract so that the Controller can be written against the final interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from dpp_scheduler.contracts import BatchPlan, Prediction, StateSnapshot


class DurationPredictor(ABC):
    @abstractmethod
    def predict(
        self, snapshot: StateSnapshot, plans: Iterable[BatchPlan]
    ) -> tuple[Prediction, ...]:
        raise NotImplementedError


class NullDurationPredictor(DurationPredictor):
    """G2 placeholder: every prediction is explicitly out-of-support."""

    def predict(
        self, snapshot: StateSnapshot, plans: Iterable[BatchPlan]
    ) -> tuple[Prediction, ...]:
        return tuple(
            Prediction(
                plan_id=plan.plan_id,
                snapshot_hash=snapshot.snapshot_hash,
                expected_duration=None,
                conservative_duration=None,
                in_support=False,
                predictor_version="null-g2",
            )
            for plan in plans
        )
