"""vLLM Adapter boundary for the modular DPP scheduler.

Only this module may import vLLM internal types.  The G2 implementation
provides the exact-plan adapter contract plus a callback-based adapter used by
unit tests and controllers; the locked-vLLM-specific construction is isolated
here so it can be completed against the verified commit without touching other
modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dpp_scheduler.contracts import (
    BatchPlan,
    ExecutionObservation,
    StateSnapshot,
    validate_snapshot_hash,
)


class ExactPlanAdapter(ABC):
    """Adapter contract: make a snapshot and atomically execute a BatchPlan."""

    @abstractmethod
    def make_snapshot(self) -> StateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def execute_exact_plan(self, plan: BatchPlan) -> ExecutionObservation:
        raise NotImplementedError


@dataclass(frozen=True)
class CallbackVllmAdapter(ExactPlanAdapter):
    """Adapter useful for tests and for wrapping a future vLLM binding.

    The callables are intentionally dependency-free; vLLM-specific translation
    belongs in a specialized subclass.
    """

    snapshot_factory: Callable[[], StateSnapshot]
    executor: Callable[[BatchPlan], ExecutionObservation]

    def make_snapshot(self) -> StateSnapshot:
        return self.snapshot_factory()

    def execute_exact_plan(self, plan: BatchPlan) -> ExecutionObservation:
        observation = self.executor(plan)
        validate_snapshot_hash(plan.snapshot_hash, observation.snapshot_hash)
        if not observation.matches(plan):
            raise RuntimeError(
                f"selected plan {plan.plan_id} does not match executed plan "
                f"{observation.executed_plan_id}"
            )
        return observation


class VllmAdapter(ExactPlanAdapter):
    """Placeholder for the commit-specific real vLLM binding.

    A G2-complete repository can use `CallbackVllmAdapter` with a fake executor
    until the G0 remote smoke has frozen the exact vLLM scheduler interfaces.
    The constructor accepts a vLLM scheduler object but deliberately does not
    import vLLM at module import time.
    """

    def __init__(self, scheduler: Any = None) -> None:
        self._scheduler = scheduler

    def make_snapshot(self) -> StateSnapshot:
        if self._scheduler is None:
            raise NotImplementedError(
                "VllmAdapter.make_snapshot requires a vLLM scheduler; "
                "use CallbackVllmAdapter for G2 unit tests or bind the real "
                "scheduler after G0."
            )
        # The actual translation is intentionally isolated here.  It will be
        # implemented against the verified vLLM commit after the G0 smoke.
        raise NotImplementedError(
            "VllmAdapter.make_snapshot is not yet wired to the locked vLLM "
            "Scheduler internals; G0 must first freeze SchedulerConfig and "
            "iteration event semantics."
        )

    def execute_exact_plan(self, plan: BatchPlan) -> ExecutionObservation:
        raise NotImplementedError(
            "VllmAdapter.execute_exact_plan is not yet wired to the locked "
            "vLLM Scheduler internals."
        )
