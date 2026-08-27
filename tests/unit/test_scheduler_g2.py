from __future__ import annotations

import dataclasses
import inspect
import unittest

from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.contracts import (
    ControlState,
    DecodeRequest,
    ExecutionObservation,
    PrefillRequest,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.controller import Controller
from dpp_scheduler.predictor import NullDurationPredictor
from dpp_scheduler.safe_set import PassThroughSafeSet
from dpp_scheduler.settings import SchedulerSettings
from dpp_scheduler.vllm_adapter import CallbackVllmAdapter


def snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    token_budget: int = 100,
    sequence_budget: int = 64,
    timestamp: float = 10.0,
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=1,
        timestamp=timestamp,
        waiting_prefill_requests=prefill,
        active_decode_requests=decode,
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=100,
        kv_block_size=16,
        token_budget=token_budget,
        sequence_budget=sequence_budget,
        total_kv_blocks=100,
    )


class ContractTests(unittest.TestCase):
    def test_snapshot_hash_is_deterministic_and_content_bound(self) -> None:
        first = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        second = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        changed = snapshot(prefill=(PrefillRequest("p", 0.0, 101, 0),))
        self.assertEqual(first.snapshot_hash, second.snapshot_hash)
        self.assertNotEqual(first.snapshot_hash, changed.snapshot_hash)
        with self.assertRaises(ValueError):
            validate_snapshot_hash(first.snapshot_hash, changed.snapshot_hash)

    def test_public_contracts_are_frozen(self) -> None:
        self.assertTrue(dataclasses.is_dataclass(ControlState))
        self.assertTrue(ControlState.__dataclass_params__.frozen)


class CandidateGeneratorV3Tests(unittest.TestCase):
    def test_all_candidates_include_all_active_decode(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d2", 1.0, 20), DecodeRequest("d1", 0.0, 20)),
        )
        generator = CandidateGenerator(SchedulerSettings.provisional())
        plans = generator.generate(state)
        self.assertGreaterEqual(len(plans), 2)
        self.assertLessEqual(len(plans), 12)
        fractions = [p for p in plans if p.template_id.startswith("P")]
        self.assertTrue(all(plan.decode_items == ("d1", "d2") for plan in fractions))

    def test_multiplier_neighborhood_includes_zero(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 8, 0),), token_budget=8
        )
        generator = CandidateGenerator(SchedulerSettings.provisional())
        plans = generator.generate(state)
        # ZERO plus the deduplicated 5×3 neighborhood of the single-request
        # prefill. With one prefill request and a 100-token token budget
        # (decode_count=0), the floor(0.5*8)=4 .. floor(1.5*8)=12 budgets
        # collapse to canonical plans depending on chunk sizes.
        self.assertTrue(any(plan.total_prefill_tokens == 0 for plan in plans))
        self.assertTrue(all(plan.total_prefill_tokens <= 8 for plan in plans))

    def test_running_prefill_precedes_waiting_fcfs_without_slo_priority(self) -> None:
        state = snapshot(
            prefill=(
                PrefillRequest("waiting", 0.0, 100, 0),
                PrefillRequest("running", 5.0, 100, 10, is_running=True),
            )
        )
        generator = CandidateGenerator(SchedulerSettings.provisional())
        plans = generator.generate(state)
        nonzero = [plan for plan in plans if plan.total_prefill_tokens > 0]
        self.assertTrue(nonzero, "expected at least one non-zero Mixed plan")
        running_first = [
            plan for plan in nonzero if plan.prefill_items[0][0] == "running"
        ]
        self.assertTrue(
            running_first,
            "expected at least one non-zero plan to serve the running "
            "request before the waiting one",
        )

    def test_generator_has_no_predictor_or_safe_set_dependency(self) -> None:
        source = inspect.getsource(
            __import__("dpp_scheduler.candidate_generator", fromlist=["x"])
        )
        self.assertNotIn("DurationPredictor", source)
        self.assertNotIn("SafeSet", source)

    def test_no_decode_path_emits_fraction_and_stock_candidates(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(),
        )
        generator = CandidateGenerator(SchedulerSettings.provisional())
        plans = generator.generate(state)
        self.assertGreaterEqual(len(plans), 1)
        self.assertTrue(any(plan.template_id.startswith("P") for plan in plans))
        self.assertTrue(any(plan.template_id == "STOCK" for plan in plans))


class ControllerExactPlanTests(unittest.TestCase):
    def test_controller_executes_exact_selected_plan(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 10, 0),))
        executed = []

        def execute(plan):
            executed.append(plan)
            return ExecutionObservation(
                frame_id=state.frame_id,
                snapshot_hash=state.snapshot_hash,
                executed_plan_id=plan.plan_id,
                executed_prefill_items=plan.prefill_items,
                executed_decode_items=plan.decode_items,
                started_at=10.0,
                finished_at=10.1,
            )

        controller = Controller(
            CallbackVllmAdapter(lambda: state, execute),
            predictor=NullDurationPredictor(),
            safe_set=PassThroughSafeSet(),
        )
        decision = controller.schedule_once()
        self.assertEqual(executed, [decision.selected_plan])


class SettingsTests(unittest.TestCase):
    def test_mapping_freezes_fixed_fraction_candidate_constants(self) -> None:
        value = {
            "prefill_budget_fractions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "maximum_seed_candidates": 12,
            "minimum_prefill_chunk_tokens": 6,
            "parameters_frozen": True,
        }
        settings = SchedulerSettings.from_mapping(value)
        self.assertEqual(settings.maximum_seed_candidates, 12)
        self.assertEqual(settings.prefill_budget_fractions[0], 0.1)
        self.assertEqual(settings.prefill_budget_fractions[-1], 1.0)

    def test_mapping_rejects_obsolete_multiplier_fields(self) -> None:
        with self.assertRaises(ValueError):
            SchedulerSettings.from_mapping(
                {
                    "prefill_budget_multipliers": [0.5, 1.0],
                    "maximum_seed_candidates": 12,
                    "minimum_prefill_chunk_tokens": 6,
                    "parameters_frozen": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
