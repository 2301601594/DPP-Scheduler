from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    ExecutionObservation,
    Obligation,
    PrefillRequest,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.controller import Controller
from dpp_scheduler.selector import TemporarySelector
from dpp_scheduler.settings import SchedulerSettings
from dpp_scheduler.vllm_adapter import CallbackVllmAdapter


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
    def test_empty_state_is_deterministic(self) -> None:
        snap = make_snapshot()
        gen = CandidateGenerator()
        first = gen.generate(snap)
        second = gen.generate(snap)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 12)
        for plan in first:
            self.assertEqual(plan.snapshot_hash, snap.snapshot_hash)

    def test_prefill_only_produces_plans(self) -> None:
        prefill = (
            PrefillRequest("p1", 0.0, 100, 0, ttft_deadline=10.0, hard_ttft_protected=True, ordinal=0),
            PrefillRequest("p2", 1.0, 50, 0, ordinal=1),
        )
        snap = make_snapshot(prefill=prefill, token_budget=512, sequence_budget=8)
        plans = CandidateGenerator(SchedulerSettings(
            prefill_caps_small=32,
            prefill_caps_medium=128,
            prefill_caps_large=512,
            urgent_limit_u=4,
            recovery_age_threshold=30.0,
        )).generate(snap)
        self.assertLessEqual(len(plans), 12)
        self.assertTrue(any(p.total_prefill_tokens > 0 for p in plans))
        for plan in plans:
            self.assertLessEqual(plan.total_prefill_tokens + plan.total_decode_tokens, 512)

    def test_decode_only_produces_mandatory_urgent_all(self) -> None:
        decode = (
            DecodeRequest("d1", 0.0, 100, tbt_deadline=5.0, mandatory=True, ordinal=0),
            DecodeRequest("d2", 1.0, 50, tbt_deadline=10.0, ordinal=1),
            DecodeRequest("d3", 2.0, 60, tbt_deadline=3.0, ordinal=2),
        )
        snap = make_snapshot(decode=decode)
        plans = CandidateGenerator().generate(snap)
        decode_itemsets = {plan.decode_items for plan in plans}
        self.assertIn(("d1",), decode_itemsets)  # MANDATORY
        self.assertIn(("d1", "d3", "d2"), decode_itemsets)  # URGENT adds earliest deadlines
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


class SelectorTests(unittest.TestCase):
    def test_selector_prefers_non_idle_plan(self) -> None:
        prefill = (PrefillRequest("p1", 0.0, 10, 0, ordinal=0),)
        snap = make_snapshot(prefill=prefill)
        plans = CandidateGenerator().generate(snap)
        decision = TemporarySelector().select(snap, plans)
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
        first = selector.select(snap, plans)
        second = selector.select(snap, plans)
        self.assertEqual(first.selected_plan, second.selected_plan)

    def test_selector_empty_returns_no_safe_decision(self) -> None:
        snap = make_snapshot()
        decision = TemporarySelector().select(snap, ())
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
        self.assertIsNotNone(decision.selected_plan)
        self.assertEqual(len(controller.observer.records), 1)


if __name__ == "__main__":
    unittest.main()
