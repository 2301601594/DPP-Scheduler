from __future__ import annotations

import inspect
import math
import unittest

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    load_active_runtime,
    load_dpp_settings,
)
from dpp_scheduler.candidate_generator import project_kv_blocks, project_sequence_count
from dpp_scheduler.contracts import (
    BatchPlan,
    ControlState,
    DecodeRequest,
    Prediction,
    PrefillRequest,
    SafeCandidate,
    StateSnapshot,
)
from dpp_scheduler.dpp_selector import DPPScore, DPPSelector
from dpp_scheduler.settings import DPPSettings
from dpp_scheduler.state_store import DuplicateLedgerEvent, InMemoryStateStore
from dpp_scheduler.vllm_adapter import get_modular_scheduler_class


def settings(*, prefill_ref: int = 1, decode_ref: int = 1) -> DPPSettings:
    return DPPSettings(prefill_ref, decode_ref, float.fromhex("0x1.fffffffffffffp+1023"))


def snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    frame: int = 1,
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=frame,
        timestamp=float(frame),
        waiting_prefill_requests=prefill,
        active_decode_requests=decode,
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=100,
        kv_block_size=16,
        token_budget=2048,
        sequence_budget=64,
        total_kv_blocks=100,
    )


def candidate(
    state: StateSnapshot,
    plan_id: str,
    *,
    prefill_tokens: int = 0,
    duration: float = 0.1,
    extrapolated: bool = False,
) -> SafeCandidate:
    prefill_items = (
        ((state.waiting_prefill_requests[0].request_id, prefill_tokens),)
        if prefill_tokens
        else ()
    )
    decode_items = tuple(item.request_id for item in state.active_decode_requests)
    plan = BatchPlan(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        template_id="test",
        prefill_items=prefill_items,
        decode_items=decode_items,
        total_prefill_tokens=prefill_tokens,
        total_decode_tokens=len(decode_items),
        total_sequences=project_sequence_count(state, prefill_items),
        projected_kv_blocks=project_kv_blocks(state, prefill_items, decode_items),
        mandatory_request_ids=(),
    )
    prediction = Prediction(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        expected_duration=duration,
        conservative_duration=duration + 0.2,
        in_support=not extrapolated,
        ood_distance=1.0 if extrapolated else 0.0,
        prediction_mode=(
            "CONSTRAINED_EXTRAPOLATION" if extrapolated else "INTERPOLATION"
        ),
        predictor_version="test",
    )
    return SafeCandidate(
        snapshot_hash=state.snapshot_hash,
        plan=plan,
        prediction=prediction,
        predicted_violation_count=0,
        predicted_total_lateness_seconds=0.0,
        conservative_deadline_margin_seconds=None,
    )


class DPPSelectorV2Tests(unittest.TestCase):
    def test_prefill_pressure_can_choose_larger_prefill(self) -> None:
        state = snapshot(
            prefill=(
                PrefillRequest(
                    "p", 0.0, 100, 0, ttft_slo_seconds=2.0
                ),
            )
        )
        control = ControlState(state.snapshot_hash, (("p", 5.0),), ())
        small = candidate(state, "small", duration=0.1)
        large = candidate(state, "large", prefill_tokens=100, duration=0.5)
        decision = DPPSelector(settings()).select(state, control, (small, large))
        self.assertEqual(decision.selected_plan, large.plan)

    def test_decode_pressure_prefers_shorter_iteration(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d", 0.0, 100, tbt_slo_seconds=0.25),),
        )
        control = ControlState(state.snapshot_hash, (("p", 5.0),), (("d", 5.0),))
        short = candidate(state, "short", duration=0.1)
        long = candidate(state, "long", prefill_tokens=100, duration=0.5)
        decision = DPPSelector(settings()).select(state, control, (long, short))
        self.assertEqual(decision.selected_plan, short.plan)

    def test_fixed_reference_preserves_decode_overload_signal(self) -> None:
        one = snapshot(
            decode=(DecodeRequest("d0", 0.0, 100),),
        )
        two = snapshot(
            decode=(DecodeRequest("d0", 0.0, 100), DecodeRequest("d1", 0.0, 100)),
            frame=2,
        )
        selector = DPPSelector(settings(decode_ref=1))
        one_score = selector.score_candidate(
            one,
            ControlState(one.snapshot_hash, (), (("d0", 1.0),)),
            candidate(one, "one", duration=0.5),
        )
        two_score = selector.score_candidate(
            two,
            ControlState(two.snapshot_hash, (), (("d0", 1.0), ("d1", 1.0))),
            candidate(two, "two", duration=0.5),
        )
        self.assertAlmostEqual(two_score.decode_drift, 2 * one_score.decode_drift)

    def test_extrapolation_uses_conservative_effective_duration(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        item = candidate(state, "ood", extrapolated=True, duration=0.1)
        score = DPPSelector(settings()).score_candidate(
            state, ControlState(state.snapshot_hash, (("p", 0.0),), ()), item
        )
        self.assertAlmostEqual(score.effective_duration, 0.3)

    def test_isclose_tie_breaks_duration_budget_then_id(self) -> None:
        state = snapshot()
        control = ControlState(state.snapshot_hash)
        selector = DPPSelector(settings())
        slower = candidate(state, "a", duration=0.2)
        faster = candidate(state, "z", duration=0.1)
        self.assertEqual(
            selector.select(state, control, (slower, faster)).selected_plan,
            faster.plan,
        )

    def test_invalid_duration_fails_closed(self) -> None:
        state = snapshot()
        bad = candidate(state, "bad")
        bad = SafeCandidate(
            **{
                **bad.__dict__,
                "prediction": Prediction(
                    **{**bad.prediction.__dict__, "expected_duration": math.nan}
                ),
            }
        )
        with self.assertRaises(ValueError):
            DPPSelector(settings()).select(
                state, ControlState(state.snapshot_hash), (bad,)
            )


def multi_candidate(
    state: StateSnapshot,
    plan_id: str,
    prefill_items: tuple[tuple[str, int], ...],
    *,
    duration: float = 0.1,
) -> SafeCandidate:
    decode_items = tuple(item.request_id for item in state.active_decode_requests)
    plan = BatchPlan(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        template_id="test",
        prefill_items=prefill_items,
        decode_items=decode_items,
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=len(decode_items),
        total_sequences=project_sequence_count(state, prefill_items),
        projected_kv_blocks=project_kv_blocks(state, prefill_items, decode_items),
        mandatory_request_ids=(),
    )
    prediction = Prediction(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        expected_duration=duration,
        conservative_duration=duration + 0.2,
        in_support=True,
        ood_distance=0.0,
        prediction_mode="INTERPOLATION",
        predictor_version="test",
    )
    return SafeCandidate(
        snapshot_hash=state.snapshot_hash,
        plan=plan,
        prediction=prediction,
        predicted_violation_count=0,
        predicted_total_lateness_seconds=0.0,
        conservative_deadline_margin_seconds=None,
    )


def manual_score(
    plan_id: str,
    *,
    score: float,
    prefill_progress: float,
    duration: float = 0.1,
    prefill_budget: int = 0,
) -> DPPScore:
    return DPPScore(
        plan_id=plan_id,
        prefill_drift=0.0,
        decode_drift=0.0,
        total_drift=-score * duration,
        effective_duration=duration,
        score=score,
        prefill_budget=prefill_budget,
        current_prefill_count=0,
        current_decode_count=0,
        prefill_reference_concurrency=1,
        decode_reference_concurrency=1,
        prefill_progress=prefill_progress,
    )


class ProgressTieBreakTests(unittest.TestCase):
    """T1 progress-first tie-break semantics (plan Phase B section 11)."""

    def test_case1_identical_score_larger_progress_wins(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0, ttft_slo_seconds=2.0),)
        )
        control = ControlState(state.snapshot_hash, (("p", 0.0),), ())
        p25 = candidate(state, "p25", prefill_tokens=50, duration=0.1)
        max_plan = candidate(state, "max", prefill_tokens=100, duration=0.1)
        decision = DPPSelector(settings()).select(state, control, (p25, max_plan))
        self.assertEqual(decision.selected_plan, max_plan.plan)

    def test_case2_isclose_range_score_uses_progress(self) -> None:
        state = snapshot()
        selector = DPPSelector(settings())
        cand_a = candidate(state, "a")
        cand_b = candidate(state, "b")
        scored = (
            (cand_a, manual_score("a", score=0.0, prefill_progress=0.20)),
            (cand_b, manual_score("b", score=5e-13, prefill_progress=0.50)),
        )
        self.assertTrue(math.isclose(0.0, 5e-13, rel_tol=1e-9, abs_tol=1e-12))
        decision = selector._decision_from_scored(state, scored)
        self.assertEqual(decision.selected_plan, cand_b.plan)

    def test_case3_clear_score_winner_not_overridden(self) -> None:
        state = snapshot()
        selector = DPPSelector(settings())
        cand_a = candidate(state, "a")
        cand_b = candidate(state, "b")
        scored = (
            (cand_a, manual_score("a", score=0.02, prefill_progress=0.10)),
            (cand_b, manual_score("b", score=0.01, prefill_progress=0.90)),
        )
        self.assertFalse(math.isclose(0.02, 0.01, rel_tol=1e-9, abs_tol=1e-12))
        decision = selector._decision_from_scored(state, scored)
        self.assertEqual(decision.selected_plan, cand_a.plan)

    def test_case4_equal_score_and_progress_uses_duration(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0, ttft_slo_seconds=2.0),)
        )
        control = ControlState(state.snapshot_hash, (("p", 0.0),), ())
        slow = candidate(state, "slow", prefill_tokens=50, duration=0.22)
        fast = candidate(state, "fast", prefill_tokens=50, duration=0.18)
        decision = DPPSelector(settings()).select(state, control, (slow, fast))
        self.assertEqual(decision.selected_plan, fast.plan)

    def test_case5_budget_then_plan_id_remain_final_fallbacks(self) -> None:
        state = snapshot(
            prefill=(
                PrefillRequest("r1", 0.0, 100, 0, ttft_slo_seconds=2.0),
                PrefillRequest("r2", 0.0, 200, 0, ttft_slo_seconds=2.0),
            )
        )
        control = ControlState(
            state.snapshot_hash, (("r1", 0.0), ("r2", 0.0)), ()
        )
        smaller_budget = multi_candidate(state, "plan-a", (("r1", 50),))
        larger_budget = multi_candidate(state, "plan-b", (("r2", 100),))
        decision = DPPSelector(settings()).select(
            state, control, (larger_budget, smaller_budget)
        )
        self.assertEqual(decision.selected_plan, smaller_budget.plan)

        same_budget_a = multi_candidate(state, "plan-a", (("r1", 50),))
        same_budget_b = multi_candidate(state, "plan-b", (("r1", 50),))
        decision = DPPSelector(settings()).select(
            state, control, (same_budget_b, same_budget_a)
        )
        self.assertEqual(decision.selected_plan, same_budget_a.plan)

    def test_case6_decode_only_keeps_previous_behavior(self) -> None:
        state = snapshot(decode=(DecodeRequest("d", 0.0, 100, tbt_slo_seconds=0.25),))
        control = ControlState(state.snapshot_hash, (), (("d", 0.0),))
        slow = candidate(state, "slow", duration=0.2)
        fast = candidate(state, "fast", duration=0.1)
        decision = DPPSelector(settings()).select(state, control, (slow, fast))
        self.assertEqual(decision.selected_plan, fast.plan)

    def test_case7_clear_winner_survives_large_progress_gap(self) -> None:
        state = snapshot()
        selector = DPPSelector(settings())
        p25 = candidate(state, "p25")
        max_plan = candidate(state, "max")
        scored = (
            (p25, manual_score("p25", score=0.01, prefill_progress=0.2)),
            (max_plan, manual_score("max", score=-0.05, prefill_progress=1.0)),
        )
        decision = selector._decision_from_scored(state, scored)
        self.assertEqual(decision.selected_plan, p25.plan)


class RequestDebtStateTests(unittest.TestCase):
    def test_prefill_initialization_and_completion_removal(self) -> None:
        state = snapshot(
            prefill=(
                PrefillRequest(
                    "p", 0.0, 100, 20, ttft_slo_seconds=2.0
                ),
            ),
            frame=2,
        )
        store = InMemoryStateStore(settings=settings())
        bound = store.bind_snapshot(state)
        self.assertAlmostEqual(dict(bound.ttft_service_debts)["p"], 0.8)
        updated = store.update_from_actual(
            previous_snapshot=state,
            actual_duration_seconds=0.2,
            executed_prefill_items=(("p", 80),),
            executed_decode_items=(),
        )
        self.assertEqual(updated.ttft_service_debts, ())
        with self.assertRaises(DuplicateLedgerEvent):
            store.update_from_actual(
                previous_snapshot=state,
                actual_duration_seconds=0.2,
                executed_prefill_items=(("p", 80),),
                executed_decode_items=(),
            )

    def test_first_token_initializes_then_second_token_services_tbt(self) -> None:
        first = snapshot(decode=(DecodeRequest("d", 0.0, 100),), frame=1)
        store = InMemoryStateStore(settings=settings())
        store.bind_snapshot(first)
        after_first = store.update_from_actual(
            previous_snapshot=first,
            actual_duration_seconds=0.2,
            executed_prefill_items=(),
            executed_decode_items=(),
            initialized_tbt_request_ids=("d",),
        )
        self.assertEqual(after_first.tbt_service_debts, (("d", 0.0),))

        second = snapshot(decode=(DecodeRequest("d", 0.0, 101),), frame=2)
        store.bind_snapshot(second)
        after_second = store.update_from_actual(
            previous_snapshot=second,
            actual_duration_seconds=0.5,
            executed_prefill_items=(),
            executed_decode_items=("d",),
        )
        self.assertAlmostEqual(dict(after_second.tbt_service_debts)["d"], 1.0)


class V2ConfigTests(unittest.TestCase):
    def test_active_config_is_explicitly_provisional_and_live_factory_gates(self) -> None:
        runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
        dpp = load_dpp_settings(runtime)
        self.assertFalse(dpp.live_v2_ready)
        source = inspect.getsource(get_modular_scheduler_class)
        self.assertIn("live v2 is disabled", source)
        self.assertIn("DPPSelector(dpp_settings)", source)


if __name__ == "__main__":
    unittest.main()
