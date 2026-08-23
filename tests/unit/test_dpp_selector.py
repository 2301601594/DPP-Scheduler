from __future__ import annotations

import inspect
import math
import unittest
from dataclasses import replace

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    load_active_runtime,
    load_dpp_settings,
)
from dpp_scheduler.contracts import (
    BatchPlan,
    ControlState,
    DecodeRequest,
    Obligation,
    Prediction,
    PrefillRequest,
    SafeCandidate,
    StateSnapshot,
)
from dpp_scheduler.dpp_selector import DPPSelector
from dpp_scheduler.settings import DPPSettings
from dpp_scheduler.state_store import (
    DuplicateLedgerEvent,
    InMemoryStateStore,
    LedgerUpdate,
)
from dpp_scheduler.vllm_adapter import get_modular_scheduler_class


def settings() -> DPPSettings:
    return DPPSettings(
        0.05,
        0.05,
        1.0,
        2048,
        64,
        float.fromhex("0x1.fffffffffffffp+1023"),
    )


def snapshot() -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=1,
        timestamp=100.0,
        waiting_prefill_requests=(PrefillRequest("p", 0.0, 1024, 0),),
        active_decode_requests=(DecodeRequest("d", 0.0, 128),),
        active_ttft_obligations=(Obligation("f", "p", "TTFT", 101.0, 0.0),),
        active_tbt_obligations=(Obligation("b", "d", "TBT", 101.0, 0.0),),
        recovery_requests=(),
        free_kv_blocks=1000,
        kv_block_size=16,
        token_budget=2048,
        sequence_budget=64,
        total_kv_blocks=1000,
        provenance="dpp-test",
    )


def candidate(
    state: StateSnapshot,
    plan_id: str,
    *,
    prefill_tokens: int = 0,
    ttft_success: int = 0,
    ttft_miss: int = 0,
    tbt_success: int = 0,
    tbt_miss: int = 0,
    margin: float | None = 0.0,
) -> SafeCandidate:
    plan = BatchPlan(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        template_id="test",
        prefill_items=(("p", prefill_tokens),) if prefill_tokens else (),
        decode_items=("d",) if tbt_success else (),
        total_prefill_tokens=prefill_tokens,
        total_decode_tokens=1 if tbt_success else 0,
        total_sequences=2,
        projected_kv_blocks=10,
        mandatory_request_ids=(),
    )
    prediction = Prediction(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        expected_duration=0.5,
        conservative_duration=0.6,
        in_support=True,
        predictor_version="test",
        ttft_success=ttft_success,
        ttft_miss=ttft_miss,
        tbt_success=tbt_success,
        tbt_miss=tbt_miss,
        predicted_violation_count=ttft_miss + tbt_miss,
        predicted_total_lateness_seconds=0.0,
        conservative_deadline_margin_seconds=margin,
        service_utility=float(ttft_success + tbt_success),
    )
    return SafeCandidate(
        snapshot_hash=state.snapshot_hash,
        plan=plan,
        prediction=prediction,
        predicted_violation_count=ttft_miss + tbt_miss,
        predicted_total_lateness_seconds=0.0,
        conservative_deadline_margin_seconds=margin,
    )


class DPPSelectorTests(unittest.TestCase):
    def test_score_uses_frozen_normalization_and_all_terms(self) -> None:
        state = snapshot()
        control = ControlState(state.snapshot_hash, 2048, 64.0, 64.0)
        item = candidate(
            state,
            "plan",
            prefill_tokens=1024,
            ttft_success=1,
            tbt_miss=1,
        )
        score = DPPSelector(settings()).score_candidate(state, control, item)
        self.assertAlmostEqual(score.prefill_term, 0.5)
        self.assertAlmostEqual(score.ttft_term, 0.05 / 64)
        self.assertAlmostEqual(score.tbt_term, -0.95 / 64)
        self.assertAlmostEqual(score.utility_term, 1 / 64)
        self.assertAlmostEqual(score.score, 1.003125)

    def test_tie_breaks_by_misses_margin_then_plan_id(self) -> None:
        state = snapshot()
        control = ControlState(state.snapshot_hash, 0, 0.0, 0.0)
        selector = DPPSelector(settings())
        a = candidate(state, "a", margin=1.0)
        b = candidate(state, "b", margin=1.0)
        c = candidate(state, "c", margin=2.0)
        missed = candidate(state, "missed", ttft_miss=1, margin=10.0)
        self.assertEqual(
            selector.select(state, control, (missed, a, b, c)).selected_plan,
            c.plan,
        )
        self.assertEqual(
            selector.select(state, control, (b, a)).selected_plan,
            a.plan,
        )

    def test_invalid_duration_fails_closed(self) -> None:
        state = snapshot()
        control = ControlState(state.snapshot_hash, 0, 0.0, 0.0)
        item = candidate(state, "bad")
        bad_prediction = replace(item.prediction, expected_duration=math.nan)
        bad = replace(item, prediction=bad_prediction)
        with self.assertRaises(ValueError):
            DPPSelector(settings()).select(state, control, (bad,))

    def test_actual_feedback_updates_once_and_stays_nonnegative(self) -> None:
        state = ControlState("hash", 100, 1.0, 2.0)
        store = InMemoryStateStore(current=state, settings=settings())
        updates = (
            LedgerUpdate("e1", "p", ttft_success=1),
            LedgerUpdate("e2", "d", tbt_miss=1),
        )
        result = store.update_from_actual(
            snapshot_hash="hash",
            actual_prefill_tokens=20,
            ledger_updates=updates,
        )
        self.assertEqual(result.prefill_backlog, 80)
        self.assertAlmostEqual(result.ttft_debt, 0.95)
        self.assertAlmostEqual(result.tbt_debt, 2.95)
        with self.assertRaises(DuplicateLedgerEvent):
            store.update_from_actual(
                snapshot_hash="hash",
                actual_prefill_tokens=0,
                ledger_updates=updates,
            )

    def test_active_config_and_live_factory_use_dpp(self) -> None:
        runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
        frozen = load_dpp_settings(runtime)
        self.assertEqual((frozen.epsilon_ttft, frozen.epsilon_tbt), (0.05, 0.05))
        self.assertEqual(frozen.weight_v, 1.0)
        source = inspect.getsource(get_modular_scheduler_class)
        self.assertIn("DPPSelector(dpp_settings)", source)
        self.assertIn("_dpp_state_store.update_from_actual", source)


if __name__ == "__main__":
    unittest.main()
