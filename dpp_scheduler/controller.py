"""Controller for the modular exact-BatchPlan scheduler.

G2 version: snapshot -> candidates -> temporary selector -> exact adapter ->
observation/decision log.  It intentionally leaves DPP scoring and real
feedback updates to later gates.
"""

from __future__ import annotations

from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.contracts import (
    Decision,
    ExecutionObservation,
    validate_snapshot_hash,
)
from dpp_scheduler.observer import InMemoryObserver
from dpp_scheduler.predictor import DurationPredictor, NullDurationPredictor
from dpp_scheduler.safe_set import PassThroughSafeSet, SafeSet
from dpp_scheduler.selector import TemporarySelector
from dpp_scheduler.state_store import InMemoryStateStore
from dpp_scheduler.vllm_adapter import ExactPlanAdapter


class Controller:
    """One scheduling round through the public modular interfaces."""

    def __init__(
        self,
        adapter: ExactPlanAdapter,
        *,
        generator: CandidateGenerator | None = None,
        predictor: DurationPredictor | None = None,
        safe_set: SafeSet | None = None,
        selector: TemporarySelector | None = None,
        observer: InMemoryObserver | None = None,
        state_store: InMemoryStateStore | None = None,
    ) -> None:
        self.adapter = adapter
        self.generator = generator or CandidateGenerator()
        self.predictor = predictor or NullDurationPredictor()
        self.safe_set = safe_set or PassThroughSafeSet()
        self.selector = selector or TemporarySelector()
        self.observer = observer or InMemoryObserver()
        self.state_store = state_store or InMemoryStateStore()

    def schedule_once(self) -> Decision:
        snapshot = self.adapter.make_snapshot()
        plans = self.generator.generate(snapshot)
        predictions = self.predictor.predict(snapshot, plans)
        safe_result = self.safe_set.filter(snapshot, plans, predictions)
        decision = self.selector.select(snapshot, safe_result.candidates)

        observation: ExecutionObservation | None = None
        if decision.selected_plan is not None:
            validate_snapshot_hash(
                decision.selected_plan.snapshot_hash, snapshot.snapshot_hash
            )
            observation = self.adapter.execute_exact_plan(decision.selected_plan)
            if not observation.matches(decision.selected_plan):
                raise RuntimeError(
                    "Controller: executed plan does not match selected BatchPlan"
                )

        self.observer.record(snapshot, decision, observation)
        return decision
