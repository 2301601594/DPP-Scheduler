from __future__ import annotations

import inspect
import math
import unittest
from dataclasses import replace

from dpp_scheduler.candidate_generator import (
    CandidateGenerator,
    project_kv_blocks,
    project_sequence_count,
)
from dpp_scheduler.consequence_estimator import ConsequenceEstimator
from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    ExecutionObservation,
    Obligation,
    Prediction,
    PrefillRequest,
    StateSnapshot,
)
from dpp_scheduler.controller import Controller
from dpp_scheduler.predictor import DurationPredictor
from dpp_scheduler.safe_set import (
    PREDICTOR_OUT_OF_SUPPORT,
    ROLLING_KV_EXCEEDED,
    ResourceAndRiskSafeSet,
    rolling_kv_reserve_blocks,
)
from dpp_scheduler.settings import SafeSetSettings, SchedulerDiagnosticsSettings
from dpp_scheduler.vllm_adapter import CallbackVllmAdapter, get_modular_scheduler_class


def make_snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    ttft: tuple[Obligation, ...] = (),
    tbt: tuple[Obligation, ...] = (),
    free_kv_blocks: int = 100,
    total_kv_blocks: int = 100,
    timestamp: float = 100.0,
    token_budget: int = 2048,
    sequence_budget: int = 64,
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=1,
        timestamp=timestamp,
        waiting_prefill_requests=prefill,
        active_decode_requests=decode,
        active_ttft_obligations=ttft,
        active_tbt_obligations=tbt,
        recovery_requests=(),
        free_kv_blocks=free_kv_blocks,
        kv_block_size=16,
        token_budget=token_budget,
        sequence_budget=sequence_budget,
        total_kv_blocks=total_kv_blocks,
        provenance="safe-set-test",
    )


def make_plan(
    snapshot: StateSnapshot,
    plan_id: str,
    *,
    prefill_items: tuple[tuple[str, int], ...] = (),
    decode_items: tuple[str, ...] = (),
) -> BatchPlan:
    return BatchPlan(
        plan_id=plan_id,
        snapshot_hash=snapshot.snapshot_hash,
        template_id="test",
        prefill_items=prefill_items,
        decode_items=decode_items,
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=len(decode_items),
        total_sequences=project_sequence_count(snapshot, prefill_items),
        projected_kv_blocks=project_kv_blocks(
            snapshot, prefill_items, decode_items
        ),
        mandatory_request_ids=(),
    )


def make_prediction(
    snapshot: StateSnapshot,
    plan: BatchPlan,
    *,
    expected: float = 0.1,
    conservative: float = 0.2,
    in_support: bool = True,
) -> Prediction:
    return Prediction(
        plan_id=plan.plan_id,
        snapshot_hash=snapshot.snapshot_hash,
        expected_duration=expected,
        conservative_duration=conservative,
        in_support=in_support,
        predictor_version="test",
    )


def settings(*, horizon: int = 1, reserve: int = 0, top_k: int = 2) -> SafeSetSettings:
    return SafeSetSettings(
        rolling_kv_horizon_iterations=horizon,
        reserve_blocks_r0=reserve,
        top_k_when_all_risky=top_k,
    )


class ConsequenceEstimatorTests(unittest.TestCase):
    def test_completion_uses_strict_deadline_boundary(self) -> None:
        snapshot = make_snapshot(
            prefill=(PrefillRequest("p", 0.0, 10, 0),),
            decode=(DecodeRequest("d", 0.0, 16),),
            ttft=(Obligation("f", "p", "TTFT", 100.2, 0.0),),
            tbt=(Obligation("b", "d", "TBT", 100.2, 0.0),),
        )
        completed = make_plan(
            snapshot, "completed", prefill_items=(("p", 10),), decode_items=("d",)
        )
        deferred = make_plan(snapshot, "deferred")
        predictions = ConsequenceEstimator().attach(
            snapshot,
            (completed, deferred),
            (
                make_prediction(snapshot, completed, expected=0.2, conservative=0.2),
                make_prediction(snapshot, deferred, expected=0.2, conservative=0.2),
            ),
        )
        self.assertEqual(predictions[0].ttft_success, 1)
        self.assertEqual(predictions[0].tbt_success, 1)
        self.assertEqual(predictions[0].predicted_violation_count, 0)
        self.assertEqual(predictions[1].ttft_miss, 1)
        self.assertEqual(predictions[1].tbt_miss, 1)
        self.assertEqual(predictions[1].predicted_violation_count, 2)
        self.assertEqual(predictions[1].predicted_total_lateness_seconds, 0.0)

    def test_invalid_prediction_remains_unattached_for_safe_rejection(self) -> None:
        snapshot = make_snapshot(decode=(DecodeRequest("d", 0.0, 16),))
        plan = make_plan(snapshot, "plan", decode_items=("d",))
        prediction = make_prediction(
            snapshot, plan, expected=math.nan, conservative=0.2
        )
        attached = ConsequenceEstimator().attach(snapshot, (plan,), (prediction,))[0]
        self.assertIsNone(attached.predicted_violation_count)


class SafeSetTests(unittest.TestCase):
    @staticmethod
    def attach(
        snapshot: StateSnapshot, plans: tuple[BatchPlan, ...]
    ) -> tuple[Prediction, ...]:
        return ConsequenceEstimator().attach(
            snapshot,
            plans,
            tuple(make_prediction(snapshot, plan) for plan in plans),
        )

    def test_all_feasible_candidates_retain_risk_metadata(self) -> None:
        snapshot = make_snapshot(decode=(DecodeRequest("d", 0.0, 16),))
        safe_plan = make_plan(snapshot, "safe", decode_items=("d",))
        risky_plan = make_plan(snapshot, "risky")
        predictions = (
            replace(
                make_prediction(snapshot, safe_plan),
                predicted_violation_count=0,
                predicted_total_lateness_seconds=0.0,
            ),
            replace(
                make_prediction(snapshot, risky_plan),
                predicted_violation_count=1,
                predicted_total_lateness_seconds=0.1,
            ),
        )
        result = ResourceAndRiskSafeSet(settings()).filter(
            snapshot, (safe_plan, risky_plan), predictions
        )
        self.assertEqual(
            tuple(item.plan.plan_id for item in result.safe_candidates),
            ("safe", "risky"),
        )
        self.assertEqual(result.rejected, ())
        self.assertEqual(result.safe_candidates[1].predicted_violation_count, 1)
        self.assertAlmostEqual(
            result.safe_candidates[1].predicted_total_lateness_seconds, 0.1
        )

    def test_all_risk_candidates_ignore_legacy_top_k(self) -> None:
        snapshot = make_snapshot(decode=(DecodeRequest("d", 0.0, 16),))
        plans = tuple(make_plan(snapshot, name) for name in ("c", "a", "b"))
        risks = ((1, 0.2), (1, 0.1), (2, 0.0))
        predictions = tuple(
            replace(
                make_prediction(snapshot, plan),
                predicted_violation_count=count,
                predicted_total_lateness_seconds=lateness,
            )
            for plan, (count, lateness) in zip(plans, risks)
        )
        result = ResourceAndRiskSafeSet(settings(top_k=2)).filter(
            snapshot, plans, predictions
        )
        self.assertEqual(
            tuple(item.plan.plan_id for item in result.safe_candidates),
            ("c", "a", "b"),
        )
        self.assertEqual(result.rejected, ())
        self.assertEqual(
            [item.predicted_violation_count for item in result.safe_candidates],
            [1, 1, 2],
        )

    def test_rolling_kv_boundary_and_reserve_rejection(self) -> None:
        snapshot = make_snapshot(
            decode=(DecodeRequest("d", 0.0, 16),),
            free_kv_blocks=1,
            total_kv_blocks=10,
        )
        plan = make_plan(snapshot, "plan")
        prediction = replace(
            make_prediction(snapshot, plan),
            predicted_violation_count=0,
            predicted_total_lateness_seconds=0.0,
        )
        self.assertEqual(rolling_kv_reserve_blocks(snapshot, plan, 1), 1)
        admitted = ResourceAndRiskSafeSet(settings(horizon=1, reserve=0)).filter(
            snapshot, (plan,), (prediction,)
        )
        self.assertEqual(len(admitted.safe_candidates), 1)
        rejected = ResourceAndRiskSafeSet(settings(horizon=1, reserve=1)).filter(
            snapshot, (plan,), (prediction,)
        )
        self.assertEqual(rejected.safe_candidates, ())
        self.assertIn(ROLLING_KV_EXCEEDED, rejected.rejected[0][1])

    def test_token_sequence_and_current_kv_boundaries(self) -> None:
        token_snapshot = make_snapshot(
            decode=(DecodeRequest("d", 0.0, 15),), token_budget=1
        )
        token_plan = make_plan(token_snapshot, "token", decode_items=("d",))
        admitted = ResourceAndRiskSafeSet(settings(horizon=0)).filter(
            token_snapshot, (token_plan,), self.attach(token_snapshot, (token_plan,))
        )
        self.assertEqual(len(admitted.safe_candidates), 1)
        over_token = make_snapshot(
            decode=(DecodeRequest("d", 0.0, 15),), token_budget=0
        )
        over_token_plan = make_plan(over_token, "over-token", decode_items=("d",))
        token_result = ResourceAndRiskSafeSet(settings(horizon=0)).filter(
            over_token,
            (over_token_plan,),
            self.attach(over_token, (over_token_plan,)),
        )
        self.assertIn("TOKEN_BUDGET_EXCEEDED", token_result.rejected[0][1])

        sequence_snapshot = make_snapshot(
            prefill=(PrefillRequest("p", 0.0, 10, 0),),
            decode=(DecodeRequest("d", 0.0, 15),),
            sequence_budget=1,
        )
        sequence_plan = make_plan(
            sequence_snapshot, "sequence", prefill_items=(("p", 1),)
        )
        sequence_result = ResourceAndRiskSafeSet(settings(horizon=0)).filter(
            sequence_snapshot,
            (sequence_plan,),
            self.attach(sequence_snapshot, (sequence_plan,)),
        )
        self.assertIn("SEQUENCE_BUDGET_EXCEEDED", sequence_result.rejected[0][1])

        kv_snapshot = make_snapshot(
            decode=(DecodeRequest("d", 0.0, 16),),
            free_kv_blocks=0,
            total_kv_blocks=10,
        )
        kv_equal = make_plan(kv_snapshot, "kv-equal")
        equal_result = ResourceAndRiskSafeSet(settings(horizon=0)).filter(
            kv_snapshot, (kv_equal,), self.attach(kv_snapshot, (kv_equal,))
        )
        self.assertEqual(len(equal_result.safe_candidates), 1)
        kv_over = make_plan(kv_snapshot, "kv-over", decode_items=("d",))
        over_result = ResourceAndRiskSafeSet(settings(horizon=0)).filter(
            kv_snapshot, (kv_over,), self.attach(kv_snapshot, (kv_over,))
        )
        self.assertIn("CURRENT_KV_EXCEEDED", over_result.rejected[0][1])

    def test_filter_is_side_effect_free_and_hash_mismatch_fails_closed(self) -> None:
        snapshot = make_snapshot(decode=(DecodeRequest("d", 0.0, 16),))
        plan = make_plan(snapshot, "plan", decode_items=("d",))
        before = (snapshot.snapshot_hash, snapshot.free_kv_blocks)
        ResourceAndRiskSafeSet(settings()).filter(
            snapshot, (plan,), self.attach(snapshot, (plan,))
        )
        self.assertEqual(before, (snapshot.snapshot_hash, snapshot.free_kv_blocks))
        wrong = make_snapshot(timestamp=101.0)
        with self.assertRaisesRegex(ValueError, "snapshot_hash mismatch"):
            ResourceAndRiskSafeSet(settings()).filter(
                wrong, (plan,), (make_prediction(snapshot, plan),)
            )

    def test_out_of_support_is_audited(self) -> None:
        snapshot = make_snapshot(decode=(DecodeRequest("d", 0.0, 16),))
        plan = make_plan(snapshot, "plan", decode_items=("d",))
        prediction = make_prediction(snapshot, plan, in_support=False)
        result = ResourceAndRiskSafeSet(settings()).filter(
            snapshot, (plan,), (prediction,)
        )
        self.assertEqual(result.safe_candidates, ())
        self.assertIn(PREDICTOR_OUT_OF_SUPPORT, result.rejected[0][1])

    def test_scheduler_diagnostics_settings_are_bounded_and_provisional(self) -> None:
        diagnostics = SchedulerDiagnosticsSettings.from_mapping(
            {
                "parameter_status": "provisional_for_scheduler_integration",
                "bounded_records": 1024,
                "zero_progress_watchdog_iterations": 8,
                "fail_fast_development": False,
                "performance_logging_default": False,
                "performance_logging_enable_env": "DPP_DIAGNOSTIC_ITERATION_LOG",
            }
        )
        self.assertEqual(diagnostics.bounded_records, 1024)
        self.assertEqual(diagnostics.zero_progress_watchdog_iterations, 8)
        self.assertFalse(diagnostics.fail_fast_development)
        self.assertFalse(diagnostics.performance_logging_default)

    def test_settings_reject_unfrozen_nulls(self) -> None:
        with self.assertRaisesRegex(ValueError, "not frozen"):
            SafeSetSettings.from_mapping(
                {
                    "rolling_kv_horizon_iterations": None,
                    "reserve_blocks_r0": None,
                    "top_k_when_all_risky": None,
                    "full_output_length_reservation": "forbidden",
                    "slo_risk_is_hard_filter": False,
                    "all_risk_order": [
                        "predicted_violation_count",
                        "predicted_total_lateness",
                        "stable_plan_key",
                    ],
                }
            )


class _StaticPredictor(DurationPredictor):
    def predict(self, snapshot, plans):
        return tuple(make_prediction(snapshot, plan) for plan in plans)


class SafeSetIntegrationSmokeTests(unittest.TestCase):
    def test_controller_runs_candidate_predictor_consequence_safe_set_path(self) -> None:
        snapshot = make_snapshot(prefill=(PrefillRequest("p", 0.0, 10, 0),))
        executed: list[BatchPlan] = []

        def execute(plan: BatchPlan) -> ExecutionObservation:
            executed.append(plan)
            return ExecutionObservation(
                frame_id=snapshot.frame_id,
                snapshot_hash=snapshot.snapshot_hash,
                executed_plan_id=plan.plan_id,
                executed_prefill_items=plan.prefill_items,
                executed_decode_items=plan.decode_items,
                started_at=snapshot.timestamp,
                finished_at=snapshot.timestamp + 0.1,
            )

        controller = Controller(
            CallbackVllmAdapter(lambda: snapshot, execute),
            generator=CandidateGenerator(),
            predictor=_StaticPredictor(),
            consequence_estimator=ConsequenceEstimator(),
            safe_set=ResourceAndRiskSafeSet(settings()),
        )
        decision = controller.schedule_once()
        self.assertIsNotNone(decision.selected_plan)
        self.assertEqual(executed, [decision.selected_plan])

    def test_live_modular_scheduler_factory_wires_safe_set(self) -> None:
        source = inspect.getsource(get_modular_scheduler_class)
        self.assertIn("_dpp_consequence_estimator.attach", source)
        self.assertIn("_dpp_safe_set.filter", source)
        self.assertIn("safe_result.safe_candidates", source)


if __name__ == "__main__":
    unittest.main()
