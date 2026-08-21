"""vLLM Scheduler subclass that uses the modular G2 BatchPlan path.

This module is the integration point for ``--scheduler-cls``.  It must be
imported by vLLM, so it intentionally imports vLLM Scheduler internals at the
top level.  All pure scheduling logic remains in the other ``dpp_scheduler``
modules.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler

from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.selector import TemporarySelector
from dpp_scheduler.vllm_adapter import VllmAdapter

logger = init_logger(__name__)


class ModularDPPScheduler(Scheduler):
    """A G2 custom scheduler: deterministic BatchPlan exact execution.

    This is a development scaffold.  It uses the provisional values in
    ``SchedulerSettings`` and does not yet implement Safe-Set, Predictor, DPP
    scoring, or Fallback.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dpp_adapter = VllmAdapter(self)
        self._dpp_generator = CandidateGenerator()
        self._dpp_selector = TemporarySelector()

    def schedule(self, throttle_prefills: bool = False):
        snapshot = self._dpp_adapter.make_snapshot()
        plans = self._dpp_generator.generate(snapshot)
        decision = self._dpp_selector.select(snapshot, plans)
        logger.info(
            "ModularDPPScheduler frame=%s plans=%d selected=%s prefill=%d decode=%d",
            snapshot.frame_id,
            len(plans),
            decision.selected_plan.plan_id if decision.selected_plan is not None else "NONE",
            (
                decision.selected_plan.total_prefill_tokens
                if decision.selected_plan is not None else 0
            ),
            (
                decision.selected_plan.total_decode_tokens
                if decision.selected_plan is not None else 0
            ),
        )
        if decision.selected_plan is None:
            # G4 will replace this with an audited Fallback/Preemption path.
            return super().schedule(throttle_prefills=throttle_prefills)
        return self._dpp_adapter.build_scheduler_output(decision.selected_plan)
