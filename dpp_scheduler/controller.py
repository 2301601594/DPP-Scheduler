"""Controller for the modular exact-BatchPlan Scheduler."""

from __future__ import annotations

from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.consequence_estimator import (
    ConsequenceEstimator,
    NullConsequenceEstimator,
)
from dpp_scheduler.contracts import (
    Decision,
    ExecutionObservation,
    validate_snapshot_hash,
)
from dpp_scheduler.observer import InMemoryObserver
from dpp_scheduler.fallback import (
    DeterministicFallback,
    NullFallback,
    build_liveness_escape,
    resolve_fallback,
)
from dpp_scheduler.predictor import DurationPredictor, NullDurationPredictor
from dpp_scheduler.safe_set import PassThroughSafeSet, SafeSet
from dpp_scheduler.dpp_selector import DPPSelector, TemporarySelector
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
        consequence_estimator: (
            ConsequenceEstimator | NullConsequenceEstimator | None
        ) = None,
        safe_set: SafeSet | None = None,
        selector: DPPSelector | TemporarySelector | None = None,
        fallback: DeterministicFallback | NullFallback | None = None,
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

    def _expire_before_snapshot(self) -> tuple[object, ...]:
        hook = getattr(
            self.adapter, "expire_obligations_before_snapshot", None
        )
        if hook is None:
            return ()
        updates = hook()
        return tuple(updates) if updates is not None else ()

    def schedule_once(self) -> Decision:
        expiry_updates = self._expire_before_snapshot()
        snapshot = self.adapter.make_snapshot()
        control = self.state_store.bind_snapshot(snapshot)
        # Obligation expiry remains metrics/Goodput state. It does not update
        # request-level v2 service deficits.
        del expiry_updates
        plans = self.generator.generate(snapshot)
        predictions = self.predictor.predict(snapshot, plans)
        safe_result = self.safe_set.filter(snapshot, plans, predictions)
        validate_snapshot_hash(safe_result.snapshot_hash, snapshot.snapshot_hash)
        safe_candidates = safe_result.safe_candidates
        if not safe_candidates:
            fallback_result = resolve_fallback(
                snapshot, self.fallback, self.predictor, self.safe_set
            )
            if fallback_result.rejection_reasons:
                fallback_result = build_liveness_escape(
                    snapshot, fallback_result
                )
            fallback_plan = fallback_result.plan
            decision = Decision(
                frame_id=snapshot.frame_id,
                snapshot_hash=snapshot.snapshot_hash,
                selected_plan=fallback_plan,
                reason=fallback_result.reason,
            )
        else:
            decision = self.selector.select(snapshot, control, safe_candidates)
            if decision.selected_plan is None:
                fallback_result = resolve_fallback(
                    snapshot, self.fallback, self.predictor, self.safe_set
                )
                escape = build_liveness_escape(snapshot, fallback_result)
                decision = Decision(
                    frame_id=snapshot.frame_id,
                    snapshot_hash=snapshot.snapshot_hash,
                    selected_plan=escape.plan,
                    reason=escape.reason,
                )
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
            if observation.error is None:
                if observation.finished_at is None:
                    raise RuntimeError("Controller: actual iteration duration is missing")
                self.state_store.update_from_actual(
                    previous_snapshot=snapshot,
                    actual_duration_seconds=(
                        observation.finished_at - observation.started_at
                    ),
                    executed_prefill_items=observation.executed_prefill_items,
                    executed_decode_items=observation.executed_decode_items,
                )

        self.observer.record(snapshot, decision, observation)
        return decision
