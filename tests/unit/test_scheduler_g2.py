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
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=1,
        timestamp=10.0,
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


class CandidateGeneratorV2Tests(unittest.TestCase):
    def test_all_candidates_include_all_active_decode(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d2", 1.0, 20), DecodeRequest("d1", 0.0, 20)),
        )
        plans = CandidateGenerator().generate(state)
        self.assertGreaterEqual(len(plans), 5)
        self.assertLessEqual(len(plans), 6)
        self.assertTrue(all(plan.decode_items == ("d1", "d2") for plan in plans))
        self.assertEqual(
            [plan.total_prefill_tokens for plan in plans], [0, 24, 49, 73, 98]
        )

    def test_finish_and_fraction_candidates_deduplicate(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 8, 0),), token_budget=8
        )
        plans = CandidateGenerator().generate(state)
        self.assertEqual(
            [plan.total_prefill_tokens for plan in plans], [0, 2, 4, 6, 8]
        )
        self.assertEqual(len({(p.prefill_items, p.decode_items) for p in plans}), 5)

    def test_running_prefill_precedes_waiting_fcfs_without_slo_priority(self) -> None:
        state = snapshot(
            prefill=(
                PrefillRequest("waiting", 0.0, 100, 0),
                PrefillRequest("running", 5.0, 100, 10, is_running=True),
            )
        )
        plans = CandidateGenerator().generate(state)
        nonzero = next(plan for plan in plans if plan.total_prefill_tokens > 0)
        self.assertEqual(nonzero.prefill_items[0][0], "running")

    def test_generator_has_no_predictor_or_safe_set_dependency(self) -> None:
        source = inspect.getsource(__import__("dpp_scheduler.candidate_generator", fromlist=["x"]))
        self.assertNotIn("DurationPredictor", source)
        self.assertNotIn("SafeSet", source)


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
    def test_mapping_freezes_v2_candidate_constants(self) -> None:
        value = {
            "prefill_budget_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
            "include_finish_boundary": True,
            "maximum_seed_candidates": 6,
            "minimum_prefill_chunk_tokens": 6,
            "parameters_frozen": True,
        }
        settings = SchedulerSettings.from_mapping(value)
        self.assertEqual(settings.maximum_seed_candidates, 6)
        self.assertEqual(settings.prefill_budget_fractions[1], 0.25)


if __name__ == "__main__":
    unittest.main()
