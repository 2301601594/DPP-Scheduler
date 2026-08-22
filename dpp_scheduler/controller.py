"""Controller for the modular exact-BatchPlan scheduler.

G2 version: snapshot -> candidates -> temporary selector -> exact adapter ->
observation/decision log.  It intentionally leaves DPP scoring and real
feedback updates to later gates.
"""

from __future__ import annotations

from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.consequence_estimator import NullConsequenceEstimator
from dpp_scheduler.contracts import (
    Decision,
    ExecutionObservation,
    validate_snapshot_hash,
)
from dpp_scheduler.observer import InMemoryObserver
from dpp_scheduler.fallback import NullFallback
from dpp_scheduler.predictor import DurationPredictor, NullDurationPredictor
from dpp_scheduler.safe_set import PassThroughSafeSet, SafeSet
from dpp_scheduler.dpp_selector import TemporarySelector
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
        consequence_estimator: NullConsequenceEstimator | None = None,
        safe_set: SafeSet | None = None,
        selector: TemporarySelector | None = None,
        fallback: NullFallback | None = None,
        observer: InMemoryObserver | None = None,
        state_store: InMemoryStateStore | None = None,
    ) -> None:
        self.adapter = adapter
        self.generator = generator or CandidateGenerator()
        self.predictor = predictor or NullDurationPredictor()
        self.consequence_estimator = (
            consequence_estimator or NullConsequenceEstimator()
        )
        self.safe_set = safe_set or PassThroughSafeSet()
        self.selector = selector or TemporarySelector()
        self.fallback = fallback or NullFallback()
        self.observer = observer or InMemoryObserver()
        self.state_store = state_store or InMemoryStateStore()

    def schedule_once(self) -> Decision:
        snapshot = self.adapter.make_snapshot()
        plans = self.generator.generate(snapshot)
        predictions = self.predictor.predict(snapshot, plans)
        predictions = self.consequence_estimator.attach(
            snapshot, plans, predictions
        )
        safe_result = self.safe_set.filter(snapshot, plans, predictions)
        validate_snapshot_hash(safe_result.snapshot_hash, snapshot.snapshot_hash)
        safe_candidates = safe_result.safe_candidates
        if not safe_candidates:
            fallback_plan = self.fallback.build(snapshot)
            if fallback_plan is not None:
                fallback_plan.validate_snapshot(snapshot)
                safe_candidates = (fallback_plan,)
        decision = self.selector.select(snapshot, safe_candidates)
        validate_snapshot_hash(decision.snapshot_hash, snapshot.snapshot_hash)

        observation: ExecutionObservation | None = None
        if decision.selected_plan is not None:
            validate_snapshot_hash(
                decision.selected_plan.snapshot_hash, snapshot.snapshot_hash
            )
            observation = self.adapter.execute_exact_plan(decision.selected_plan)
            validate_snapshot_hash(observation.snapshot_hash, snapshot.snapshot_hash)
            if observation.frame_id != snapshot.frame_id:
                raise RuntimeError("Controller: observation frame_id mismatch")
            if not observation.matches(decision.selected_plan):
                raise RuntimeError(
                    "Controller: executed plan does not match selected BatchPlan"
                )

        self.observer.record(snapshot, decision, observation)
        return decision
