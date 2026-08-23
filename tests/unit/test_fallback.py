from __future__ import annotations

import inspect
import unittest

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    load_active_runtime,
    load_fallback_settings,
    load_frozen_candidate_settings,
)
from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.consequence_estimator import ConsequenceEstimator
from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    ExecutionObservation,
    Prediction,
    PrefillRequest,
    StateSnapshot,
)
from dpp_scheduler.controller import Controller
from dpp_scheduler.fallback import (
    FALLBACK_DECODE_ONLY,
    FALLBACK_MINIMUM_PREFILL,
    IDLE_FALLBACK_REJECTED,
    PREEMPTION_REQUIRED_AFTER_FALLBACK_REJECTION,
    PREEMPTION_REQUIRED_MANDATORY_DECODE,
    DeterministicFallback,
    resolve_fallback,
)
from dpp_scheduler.predictor import DurationPredictor
from dpp_scheduler.safe_set import ResourceAndRiskSafeSet
from dpp_scheduler.settings import FallbackSettings, SafeSetSettings
from dpp_scheduler.vllm_adapter import CallbackVllmAdapter, get_modular_scheduler_class


def snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    token_budget: int = 2048,
    sequence_budget: int = 64,
    free_kv_blocks: int = 100,
    total_kv_blocks: int = 100,
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=7,
        timestamp=100.0,
        waiting_prefill_requests=prefill,
        active_decode_requests=decode,
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=free_kv_blocks,
        kv_block_size=16,
        token_budget=token_budget,
        sequence_budget=sequence_budget,
        total_kv_blocks=total_kv_blocks,
        provenance="fallback-test",
    )


class StaticPredictor(DurationPredictor):
    def __init__(self, *, in_support: bool = True) -> None:
        self.in_support = in_support

    def predict(self, state, plans):
        return tuple(
            Prediction(
                plan_id=plan.plan_id,
                snapshot_hash=state.snapshot_hash,
                expected_duration=0.1 if self.in_support else None,
                conservative_duration=0.2 if self.in_support else None,
                in_support=self.in_support,
                predictor_version="fallback-test",
            )
            for plan in plans
        )


def safe_set(*, reserve: int = 0) -> ResourceAndRiskSafeSet:
    return ResourceAndRiskSafeSet(
        SafeSetSettings(
            rolling_kv_horizon_iterations=1,
            reserve_blocks_r0=reserve,
            top_k_when_all_risky=3,
        )
    )


class FallbackConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fallback = DeterministicFallback(FallbackSettings(6))

    def test_decode_fallback_stops_prefill_and_uses_edf(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 32, 0),),
            decode=(
                DecodeRequest("late", 0.0, 32, tbt_deadline=102.0),
                DecodeRequest("early", 1.0, 32, tbt_deadline=101.0),
            ),
        )
        result = self.fallback.build(state)
        self.assertEqual(result.reason, FALLBACK_DECODE_ONLY)
        assert result.plan is not None
        self.assertEqual(result.plan.prefill_items, ())
        self.assertEqual(result.plan.decode_items, ("early", "late"))

    def test_prefill_fallback_uses_minimum_and_allows_short_completion(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 32, 0),))
        result = self.fallback.build(state)
        assert result.plan is not None
        self.assertEqual(result.reason, FALLBACK_MINIMUM_PREFILL)
        self.assertEqual(result.plan.prefill_items, (("p", 6),))

        finishing = snapshot(prefill=(PrefillRequest("short", 0.0, 4, 0),))
        completed = self.fallback.build(finishing)
        assert completed.plan is not None
        self.assertEqual(completed.plan.prefill_items, (("short", 4),))

    def test_mandatory_decode_is_never_silently_dropped(self) -> None:
        state = snapshot(
            decode=(
                DecodeRequest("edf", 0.0, 32, tbt_deadline=101.0),
                DecodeRequest("mandatory", 1.0, 32, mandatory=True),
            ),
            token_budget=1,
        )
        result = self.fallback.build(state)
        self.assertIsNone(result.plan)
        self.assertEqual(result.reason, PREEMPTION_REQUIRED_MANDATORY_DECODE)

    def test_hard_rejection_selects_preemption_or_idle_explicitly(self) -> None:
        state = snapshot(
            decode=(DecodeRequest("d", 0.0, 16),),
            free_kv_blocks=0,
            total_kv_blocks=10,
        )
        preemption = resolve_fallback(
            state, self.fallback, StaticPredictor(), safe_set()
        )
        self.assertEqual(
            preemption.reason, PREEMPTION_REQUIRED_AFTER_FALLBACK_REJECTION
        )
        self.assertTrue(preemption.rejection_reasons)

        supported_state = snapshot(prefill=(PrefillRequest("p", 0.0, 32, 0),))
        idle = resolve_fallback(
            supported_state,
            self.fallback,
            StaticPredictor(in_support=False),
            safe_set(),
        )
        self.assertEqual(idle.reason, IDLE_FALLBACK_REJECTED)


class FallbackIntegrationTests(unittest.TestCase):
    def test_controller_executes_fallback_without_calling_selector(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 32, 0),))
        executed: list[BatchPlan] = []

        def execute(plan: BatchPlan) -> ExecutionObservation:
            executed.append(plan)
            return ExecutionObservation(
                frame_id=state.frame_id,
                snapshot_hash=state.snapshot_hash,
                executed_plan_id=plan.plan_id,
                executed_prefill_items=plan.prefill_items,
                executed_decode_items=plan.decode_items,
                started_at=state.timestamp,
                finished_at=state.timestamp + 0.1,
            )

        class EmptyGenerator(CandidateGenerator):
            def generate(self, state):
                return ()

        class SelectorMustNotRun:
            def select(self, state, candidates):
                raise AssertionError("Fallback must not enter DPP Selector")

        controller = Controller(
            CallbackVllmAdapter(lambda: state, execute),
            generator=EmptyGenerator(),
            predictor=StaticPredictor(),
            consequence_estimator=ConsequenceEstimator(),
            safe_set=safe_set(),
            selector=SelectorMustNotRun(),
            fallback=DeterministicFallback(FallbackSettings(6)),
        )
        decision = controller.schedule_once()
        self.assertEqual(decision.reason, FALLBACK_MINIMUM_PREFILL)
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0].plan_id, "fallback-prefill-minimum")

    def test_active_config_loads_integration_freeze_and_fallback(self) -> None:
        runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
        candidate = load_frozen_candidate_settings(runtime)
        fallback = load_fallback_settings(runtime)
        self.assertEqual(candidate.settings.critical_horizon_seconds, 0.22)
        self.assertEqual(candidate.settings.prefill_knee_tokens, 768)
        self.assertEqual(fallback.minimum_prefill_chunk_tokens, 6)

    def test_live_scheduler_factory_wires_fallback(self) -> None:
        source = inspect.getsource(get_modular_scheduler_class)
        self.assertIn("resolve_fallback", source)
        self.assertIn("_dpp_fallback", source)
        self.assertIn("skip_predictor_update", source)
        self.assertLess(
            source.index('if pending["skip_predictor_update"]'),
            source.index('actual_duration = pending["actual_duration_seconds"]'),
        )


if __name__ == "__main__":
    unittest.main()
