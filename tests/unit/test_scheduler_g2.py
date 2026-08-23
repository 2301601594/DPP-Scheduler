from __future__ import annotations

import ast
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import dpp_scheduler.candidate_generator as candidate_generator_module
from dpp_scheduler.candidate_generator import (
    CandidateGenerator,
    _prefill_breakpoints,
    _rank_prefill,
)
from dpp_scheduler.contracts import (
    BatchPlan,
    ControlState,
    DecodeRequest,
    ExecutionObservation,
    Obligation,
    PrefillRequest,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.controller import Controller
from dpp_scheduler.dpp_selector import TemporarySelector
from dpp_scheduler.settings import SchedulerSettings
from dpp_scheduler.vllm_adapter import CallbackVllmAdapter, VllmAdapter


def make_snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    recovery: tuple[str, ...] = (),
    free_kv_blocks: int = 1000,
    total_kv_blocks: int = 1000,
    kv_block_size: int = 16,
    token_budget: int = 2048,
    sequence_budget: int = 64,
    frame_id: int = 1,
    timestamp: float = 100.0,
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=frame_id,
        timestamp=timestamp,
        waiting_prefill_requests=prefill,
        active_decode_requests=decode,
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=recovery,
        free_kv_blocks=free_kv_blocks,
        kv_block_size=kv_block_size,
        token_budget=token_budget,
        sequence_budget=sequence_budget,
        total_kv_blocks=total_kv_blocks,
        provenance="test",
    )


def make_settings(
    *,
    critical_horizon_seconds: float | None = 0.25,
    prefill_knee_tokens: int | None = 512,
    minimum_prefill_chunk_tokens: int = 1,
) -> SchedulerSettings:
    return SchedulerSettings(
        critical_horizon_seconds=critical_horizon_seconds,
        prefill_knee_tokens=prefill_knee_tokens,
        minimum_prefill_chunk_tokens=minimum_prefill_chunk_tokens,
    )


class ContractTests(unittest.TestCase):
    def test_snapshot_hash_is_deterministic(self) -> None:
        first = make_snapshot()
        second = make_snapshot()
        self.assertEqual(first.snapshot_hash, second.snapshot_hash)

    def test_snapshot_hash_changes_with_content(self) -> None:
        base = make_snapshot()
        changed = make_snapshot(token_budget=4096)
        self.assertNotEqual(base.snapshot_hash, changed.snapshot_hash)

    def test_validate_snapshot_hash_rejects_mismatch(self) -> None:
        snapshot = make_snapshot()
        other = make_snapshot(frame_id=2)
        with self.assertRaises(ValueError):
            validate_snapshot_hash(snapshot.snapshot_hash, other.snapshot_hash)

    def test_public_contracts_are_frozen(self) -> None:
        snapshot = make_snapshot()
        with self.assertRaises(FrozenInstanceError):
            snapshot.token_budget = 1234  # type: ignore[misc]


class CandidateGeneratorTests(unittest.TestCase):
    def test_generator_has_no_predictor_or_safe_set_dependency(self) -> None:
        source = Path(candidate_generator_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        self.assertTrue(
            imported_modules.isdisjoint(
                {
                    "dpp_scheduler.predictor",
                    "dpp_scheduler.safe_set",
                    "dpp_scheduler.dpp_selector",
                }
            )
        )

    def test_empty_state_is_deterministic(self) -> None:
        snap = make_snapshot()
        gen = CandidateGenerator()
        first = gen.generate(snap)
        second = gen.generate(snap)
        self.assertEqual(first, ())
        self.assertEqual(second, ())

    def test_prefill_only_produces_plans(self) -> None:
        prefill = (
            PrefillRequest("p1", 0.0, 100, 0, ttft_deadline=10.0, hard_ttft_protected=True, ordinal=0),
            PrefillRequest("p2", 1.0, 50, 0, ordinal=1),
        )
        snap = make_snapshot(prefill=prefill, token_budget=512, sequence_budget=8)
        plans = CandidateGenerator(
            make_settings(prefill_knee_tokens=128)
        ).generate(snap)
        self.assertLessEqual(len(plans), 12)
        self.assertTrue(any(p.total_prefill_tokens > 0 for p in plans))
        for plan in plans:
            self.assertLessEqual(plan.total_prefill_tokens + plan.total_decode_tokens, 512)

    def test_decode_only_produces_mandatory_critical_all(self) -> None:
        decode = (
            DecodeRequest(
                "d1", 0.0, 100, tbt_deadline=100.5, mandatory=True, ordinal=0
            ),
            DecodeRequest("d2", 1.0, 50, tbt_deadline=101.0, ordinal=1),
            DecodeRequest("d3", 2.0, 60, tbt_deadline=100.2, ordinal=2),
        )
        snap = make_snapshot(decode=decode)
        plans = CandidateGenerator(
            make_settings(critical_horizon_seconds=0.25)
        ).generate(snap)
        decode_itemsets = {plan.decode_items for plan in plans}
        self.assertIn(("d1",), decode_itemsets)  # MANDATORY
        self.assertIn(("d3", "d1"), decode_itemsets)  # CRITICAL plus mandatory
        self.assertIn(("d3", "d1", "d2"), decode_itemsets)  # ALL in ranked order
        for plan in plans:
            self.assertLessEqual(plan.total_decode_tokens, 3)
            self.assertTrue(all(item in {d.request_id for d in decode} for item in plan.decode_items))

    def test_token_budget_limits_plan(self) -> None:
        decode = tuple(
            DecodeRequest(f"d{i}", float(i), 10, ordinal=i) for i in range(10)
        )
        snap = make_snapshot(decode=decode, token_budget=3, sequence_budget=3)
        plans = CandidateGenerator().generate(snap)
        for plan in plans:
            self.assertLessEqual(plan.total_decode_tokens, 3)
            self.assertLessEqual(plan.total_sequences, 3)

    def test_generation_has_no_kv_side_effects_on_snapshot(self) -> None:
        snap = make_snapshot(
            prefill=(PrefillRequest("p1", 0.0, 100, 0, ordinal=0),),
            free_kv_blocks=900,
            total_kv_blocks=1000,
        )
        before = snap.snapshot_hash
        CandidateGenerator().generate(snap)
        self.assertEqual(snap.snapshot_hash, before)
        self.assertEqual(snap.free_kv_blocks, 900)

    def test_running_partial_prefill_is_counted_as_active_sequence(self) -> None:
        prefill = (
            PrefillRequest("running", 0.0, 100, 32, is_running=True),
            PrefillRequest("new", 1.0, 100, 0),
        )
        snap = make_snapshot(prefill=prefill, sequence_budget=1)
        plans = CandidateGenerator().generate(snap)
        self.assertTrue(plans)
        self.assertTrue(
            all("new" not in {request_id for request_id, _ in plan.prefill_items}
                for plan in plans)
        )
        self.assertTrue(all(plan.total_sequences == 1 for plan in plans))

    def test_full_sequence_budget_still_allows_running_partial_prefill(self) -> None:
        prefill = (
            PrefillRequest(
                "blocked-new", 0.0, 100, 0,
                hard_ttft_protected=True,
            ),
            PrefillRequest("running", 1.0, 100, 32, is_running=True),
        )
        plans = CandidateGenerator().generate(
            make_snapshot(prefill=prefill, sequence_budget=1)
        )
        self.assertTrue(
            any(
                "running" in {request_id for request_id, _ in plan.prefill_items}
                for plan in plans
            )
        )
        self.assertTrue(
            all(
                "blocked-new"
                not in {request_id for request_id, _ in plan.prefill_items}
                for plan in plans
            )
        )

    def test_critical_uses_snapshot_horizon_and_ignores_missing_deadline(self) -> None:
        decode = (
            DecodeRequest("mandatory", 0.0, 10, mandatory=True),
            DecodeRequest("no-deadline", 1.0, 10),
            DecodeRequest("boundary", 2.0, 10, tbt_deadline=100.25),
            DecodeRequest("outside", 3.0, 10, tbt_deadline=100.251),
        )
        plans = CandidateGenerator(make_settings()).generate(
            make_snapshot(decode=decode)
        )
        decode_itemsets = {plan.decode_items for plan in plans}
        self.assertIn(("boundary", "mandatory"), decode_itemsets)
        critical = next(
            itemset
            for itemset in decode_itemsets
            if "boundary" in itemset and "outside" not in itemset
        )
        self.assertNotIn("no-deadline", critical)

    def test_only_oldest_due_recovery_is_mandatory(self) -> None:
        decode = (
            DecodeRequest(
                "older", 0.0, 10, recovery_due=True,
                recovery_first_miss_time=2.0,
            ),
            DecodeRequest(
                "newer", 1.0, 10, recovery_due=True,
                recovery_first_miss_time=3.0,
            ),
        )
        snap = make_snapshot(decode=decode, recovery=("older", "newer"))
        decode_itemsets = {
            plan.decode_items for plan in CandidateGenerator().generate(snap)
        }
        self.assertIn(("older",), decode_itemsets)
        self.assertNotIn(("newer",), decode_itemsets)

    def test_prefill_breakpoints_finish_knee_and_bindable_max(self) -> None:
        decode = tuple(
            DecodeRequest(f"d{i}", float(i), 10, mandatory=True, ordinal=i)
            for i in range(8)
        )
        prefill = (
            PrefillRequest("first", 0.0, 300, 0, ordinal=0),
            PrefillRequest("second", 1.0, 3000, 0, ordinal=1),
        )
        plans = CandidateGenerator(make_settings(prefill_knee_tokens=512)).generate(
            make_snapshot(prefill=prefill, decode=decode)
        )
        totals = {plan.total_prefill_tokens for plan in plans}
        self.assertEqual(totals, {0, 300, 512, 2040})

    def test_equal_prefill_breakpoints_are_canonically_deduplicated(self) -> None:
        prefill = (PrefillRequest("p", 0.0, 512, 0),)
        plans = CandidateGenerator(make_settings(prefill_knee_tokens=512)).generate(
            make_snapshot(prefill=prefill, token_budget=512)
        )
        actions = {(plan.prefill_items, plan.decode_items) for plan in plans}
        self.assertEqual(len(plans), len(actions))
        self.assertEqual({plan.total_prefill_tokens for plan in plans}, {0, 512})

    def test_finish_is_omitted_when_request_cannot_complete(self) -> None:
        prefill = (PrefillRequest("p", 0.0, 300, 0),)
        snapshot = make_snapshot(prefill=prefill, token_budget=30)
        breakpoints = _prefill_breakpoints(
            snapshot,
            0,
            _rank_prefill(snapshot),
            make_settings(prefill_knee_tokens=None),
        )
        self.assertNotIn("FINISH", {source for source, _ in breakpoints})
        plans = CandidateGenerator(make_settings(prefill_knee_tokens=None)).generate(
            snapshot
        )
        self.assertTrue(any(plan.total_prefill_tokens == 30 for plan in plans))

    def test_minimum_chunk_rejects_small_partial_but_allows_small_finish(self) -> None:
        settings = make_settings(
            prefill_knee_tokens=None,
            minimum_prefill_chunk_tokens=64,
        )
        partial_plans = CandidateGenerator(settings).generate(
            make_snapshot(
                prefill=(PrefillRequest("partial", 0.0, 300, 0),),
                token_budget=30,
            )
        )
        self.assertTrue(partial_plans)
        self.assertTrue(
            all(plan.total_prefill_tokens == 0 for plan in partial_plans)
        )

        finish_plans = CandidateGenerator(settings).generate(
            make_snapshot(
                prefill=(PrefillRequest("finish", 0.0, 30, 0),),
                token_budget=30,
            )
        )
        self.assertTrue(any(plan.total_prefill_tokens == 30 for plan in finish_plans))

    def test_mandatory_decode_is_not_silently_trimmed(self) -> None:
        decode = tuple(
            DecodeRequest(f"d{i}", float(i), 10, mandatory=True, ordinal=i)
            for i in range(3)
        )
        plans = CandidateGenerator(make_settings()).generate(
            make_snapshot(decode=decode, token_budget=2, sequence_budget=2)
        )
        self.assertEqual(plans, ())

    def test_generator_never_exceeds_twelve_seed_actions(self) -> None:
        prefill = (
            PrefillRequest("p1", 0.0, 300, 0, ordinal=0),
            PrefillRequest("p2", 1.0, 3000, 0, ordinal=1),
        )
        decode = (
            DecodeRequest("mandatory", 0.0, 10, mandatory=True),
            DecodeRequest("critical", 1.0, 10, tbt_deadline=100.1),
            DecodeRequest("all", 2.0, 10),
        )
        plans = CandidateGenerator(make_settings()).generate(
            make_snapshot(prefill=prefill, decode=decode)
        )
        self.assertLessEqual(len(plans), 12)

    def test_provisional_settings_do_not_invent_horizon_or_knee(self) -> None:
        settings = SchedulerSettings.provisional()
        self.assertIsNone(settings.critical_horizon_seconds)
        self.assertIsNone(settings.prefill_knee_tokens)
        self.assertFalse(settings.frozen)


class SelectorTests(unittest.TestCase):
    @staticmethod
    def control(snapshot: StateSnapshot) -> ControlState:
        return ControlState(
            snapshot_hash=snapshot.snapshot_hash,
            prefill_backlog=sum(
                request.remaining_tokens
                for request in snapshot.waiting_prefill_requests
            ),
            ttft_debt=0.0,
            tbt_debt=0.0,
        )

    def test_selector_prefers_non_idle_plan(self) -> None:
        prefill = (PrefillRequest("p1", 0.0, 10, 0, ordinal=0),)
        snap = make_snapshot(prefill=prefill)
        plans = CandidateGenerator().generate(snap)
        decision = TemporarySelector().select(snap, self.control(snap), plans)
        self.assertIsNotNone(decision.selected_plan)
        self.assertGreater(
            decision.selected_plan.total_prefill_tokens
            + decision.selected_plan.total_decode_tokens,
            0,
        )

    def test_selector_is_deterministic(self) -> None:
        prefill = (PrefillRequest("p1", 0.0, 10, 0, ordinal=0),)
        snap = make_snapshot(prefill=prefill)
        plans = CandidateGenerator().generate(snap)
        selector = TemporarySelector()
        first = selector.select(snap, self.control(snap), plans)
        second = selector.select(snap, self.control(snap), plans)
        self.assertEqual(first.selected_plan, second.selected_plan)

    def test_selector_empty_returns_no_safe_decision(self) -> None:
        snap = make_snapshot()
        decision = TemporarySelector().select(snap, self.control(snap), ())
        self.assertIsNone(decision.selected_plan)
        self.assertEqual(decision.reason, "NO_SAFE_DECISION")


class ControllerTests(unittest.TestCase):
    def test_controller_executes_exact_selected_plan(self) -> None:
        prefill = (PrefillRequest("p1", 0.0, 10, 0, ordinal=0),)
        snap = make_snapshot(prefill=prefill)
        selected_plan: list[BatchPlan] = []

        def snapshot_factory() -> StateSnapshot:
            return snap

        def execute(plan: BatchPlan) -> ExecutionObservation:
            selected_plan.append(plan)
            return ExecutionObservation(
                frame_id=snap.frame_id,
                snapshot_hash=snap.snapshot_hash,
                executed_plan_id=plan.plan_id,
                executed_prefill_items=plan.prefill_items,
                executed_decode_items=plan.decode_items,
                started_at=snap.timestamp,
                finished_at=snap.timestamp + 0.001,
            )

        adapter = CallbackVllmAdapter(snapshot_factory, execute)
        controller = Controller(adapter)
        decision = controller.schedule_once()
        self.assertIsNotNone(decision.selected_plan)
        self.assertEqual(len(selected_plan), 1)
        self.assertEqual(selected_plan[0].plan_id, decision.selected_plan.plan_id)
        self.assertEqual(selected_plan[0].prefill_items, decision.selected_plan.prefill_items)
        self.assertEqual(selected_plan[0].decode_items, decision.selected_plan.decode_items)

    def test_controller_records_decision_for_executed_plan(self) -> None:
        snap = make_snapshot()
        adapter = CallbackVllmAdapter(
            lambda: snap,
            lambda plan: ExecutionObservation(
                frame_id=snap.frame_id,
                snapshot_hash=snap.snapshot_hash,
                executed_plan_id=plan.plan_id,
                executed_prefill_items=plan.prefill_items,
                executed_decode_items=plan.decode_items,
                started_at=snap.timestamp,
                finished_at=snap.timestamp + 0.001,
            ),
        )
        controller = Controller(adapter)
        decision = controller.schedule_once()
        self.assertIsNone(decision.selected_plan)
        self.assertEqual(len(controller.observer.records), 1)


class AdapterSnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_running_partial_prefill_and_never_reads_max_tokens(self) -> None:
        class Request:
            request_id = "partial"
            arrival_time = 1.0
            num_computed_tokens = 32
            num_prompt_tokens = 128
            status = "RUNNING"

            @property
            def max_tokens(self):
                raise AssertionError("Scheduler must not read the client length guard")

        class BlockPool:
            num_gpu_blocks = 100

            @staticmethod
            def get_num_free_blocks() -> int:
                return 90

        class KVManager:
            block_pool = BlockPool()

        class Config:
            enable_prefix_caching = False
            async_scheduling = False

        class Scheduler:
            requests = {"partial": Request()}
            running = [requests["partial"]]
            waiting = ()
            kv_cache_manager = KVManager()
            block_size = 16
            max_num_scheduled_tokens = 2048
            max_num_running_reqs = 64
            cache_config = Config()
            scheduler_config = Config()
            num_spec_tokens = 0
            connector = None

        snapshot = VllmAdapter(Scheduler()).make_snapshot()
        self.assertEqual(snapshot.active_decode_requests, ())
        self.assertEqual(len(snapshot.waiting_prefill_requests), 1)
        self.assertTrue(snapshot.waiting_prefill_requests[0].is_running)
        self.assertEqual(snapshot.waiting_prefill_requests[0].prefilled_tokens, 32)


if __name__ == "__main__":
    unittest.main()
