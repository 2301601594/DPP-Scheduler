from __future__ import annotations

import dataclasses
import inspect
import unittest

from dpp_scheduler.budget_resolver import (
    RESOLUTION_INVERTED_OK,
    RESOLUTION_NO_DECODE_USE_MAX,
    BudgetResolution,
    BudgetResolver,
)
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


class _StaticBudgetResolver(BudgetResolver):
    """Resolver stub for v3 Generator layout tests.

    Returns ``base_prefill_budget`` from the constructor argument, allowing
    tests to drive the Generator without depending on a Predictor artifact.
    """

    def __init__(self, base: int, status: str = RESOLUTION_INVERTED_OK) -> None:
        self._base = base
        self._status = status

    def resolve(self, snapshot: StateSnapshot) -> BudgetResolution:  # noqa: ARG002
        return BudgetResolution(
            base_prefill_budget=int(self._base),
            target_duration_seconds=0.250,
            resolution_status=self._status,
        )


class CandidateGeneratorV3Tests(unittest.TestCase):
    def test_all_candidates_include_all_active_decode(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d2", 1.0, 20), DecodeRequest("d1", 0.0, 20)),
        )
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=_StaticBudgetResolver(base=512),
        )
        plans = generator.generate(state)
        # Expect ZERO + 5 multipliers × 3 policies (when every policy produces
        # the same canonical plan under only one Prefill request, dedup
        # collapses to ZERO + ≤15 distinct plans).
        self.assertGreaterEqual(len(plans), 2)
        self.assertLessEqual(len(plans), 16)
        self.assertTrue(all(plan.decode_items == ("d1", "d2") for plan in plans))

    def test_multiplier_neighborhood_includes_zero(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 8, 0),), token_budget=8
        )
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=_StaticBudgetResolver(base=8),
        )
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
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=_StaticBudgetResolver(base=512),
        )
        plans = generator.generate(state)
        # With a single (resource-clamped) budget, COMPLETION_AWARE and
        # CONTINUATION both produce the same canonical plan; the dedup keeps
        # the first one encountered, which by iteration order is the
        # COMPLETION_AWARE plan. The key invariant is that some non-zero
        # plan places the running request first, regardless of which priority
        # policy wins the dedup race.
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

    def test_p_zero_only_zero(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
        )
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=_StaticBudgetResolver(base=0),
        )
        plans = generator.generate(state)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].template_id, "ALL_DECODE:ZERO")
        self.assertEqual(plans[0].total_prefill_tokens, 0)

    def test_no_decode_path_emits_neighborhood(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(),
        )
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=_StaticBudgetResolver(
                base=600, status=RESOLUTION_NO_DECODE_USE_MAX
            ),
        )
        plans = generator.generate(state)
        self.assertGreaterEqual(len(plans), 1)
        # At least one plan should carry a SLACK_BUDGET multiplier prefix
        # and one should be ZERO.
        self.assertTrue(
            any(plan.template_id.startswith("ALL_DECODE:SLACK_BUDGET:") for plan in plans)
        )
        self.assertTrue(any(plan.template_id == "ALL_DECODE:ZERO" for plan in plans))


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
    def test_mapping_freezes_v3_candidate_constants(self) -> None:
        value = {
            "prefill_budget_multipliers": [0.50, 0.75, 1.00, 1.25, 1.50],
            "maximum_seed_candidates": 16,
            "minimum_prefill_chunk_tokens": 6,
            "predictor_inversion_safety_margin_seconds": 0.020,
            "predictor_inversion_budget_grid": [0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048],
            "parameters_frozen": True,
        }
        settings = SchedulerSettings.from_mapping(value)
        self.assertEqual(settings.maximum_seed_candidates, 16)
        self.assertEqual(settings.prefill_budget_multipliers[0], 0.50)
        self.assertEqual(settings.prefill_budget_multipliers[-1], 1.50)
        self.assertEqual(settings.predictor_inversion_safety_margin_seconds, 0.020)
        self.assertEqual(
            settings.predictor_inversion_budget_grid[0], 0
        )
        self.assertEqual(
            settings.predictor_inversion_budget_grid[-1], 2048
        )

    def test_mapping_rejects_v2_field_names(self) -> None:
        with self.assertRaises(ValueError):
            SchedulerSettings.from_mapping(
                {
                    "prefill_budget_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "maximum_seed_candidates": 6,
                    "minimum_prefill_chunk_tokens": 1,
                    "parameters_frozen": True,
                }
            )
        with self.assertRaises(ValueError):
            SchedulerSettings.from_mapping(
                {
                    "include_finish_boundary": True,
                    "prefill_budget_multipliers": [0.50, 0.75, 1.00, 1.25, 1.50],
                    "maximum_seed_candidates": 16,
                    "minimum_prefill_chunk_tokens": 6,
                    "parameters_frozen": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
