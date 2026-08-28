from __future__ import annotations

import math
import unittest

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
from dpp_scheduler.dpp_selector import DPPScore, DPPSelector
from dpp_scheduler.settings import DPPSettings
from dpp_scheduler.state_store import DuplicateLedgerEvent, InMemoryStateStore
from dpp_scheduler.vllm_adapter import (
    DPP_SELECTION_MODE_ENV,
    DPP_TTFT_DRIFT_WEIGHT_ENV,
    resolve_dpp_runtime_overrides,
)


def settings(*, prefill_ref: int = 1, delta: float = 0.020) -> DPPSettings:
    return DPPSettings(
        prefill_ref,
        1,
        float.fromhex("0x1.fffffffffffffp+1023"),
        tbt_delta_seconds=delta,
    )


def snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    obligations: tuple[Obligation, ...] = (),
    frame: int = 1,
    timestamp: float = 10.0,
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=frame,
        timestamp=timestamp,
        waiting_prefill_requests=prefill,
        active_decode_requests=decode,
        active_ttft_obligations=(),
        active_tbt_obligations=obligations,
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
    prefill_items: tuple[tuple[str, int], ...] = (),
    duration: float = 0.1,
    extrapolated: bool = False,
    template_id: str = "test",
) -> SafeCandidate:
    plan = BatchPlan(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        template_id=template_id,
        prefill_items=prefill_items,
        decode_items=tuple(item.request_id for item in state.active_decode_requests),
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=len(state.active_decode_requests),
        total_sequences=len(state.active_decode_requests) + len(prefill_items),
        projected_kv_blocks=0,
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


def deadline_snapshot(*, slack: float) -> StateSnapshot:
    deadline = 10.0 + slack
    return snapshot(
        decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=deadline),),
        obligations=(Obligation("tbt:d", "d", "TBT", deadline, 9.0),),
    )


def manual_score(
    plan_id: str,
    *,
    score: float = 0.0,
    completed: int = 0,
    progress: float = 0.0,
    duration: float = 0.1,
    budget: int = 0,
) -> DPPScore:
    return DPPScore(
        plan_id=plan_id,
        prefill_drift=-score,
        effective_duration=duration,
        score=score,
        prefill_budget=budget,
        current_prefill_count=0,
        current_decode_count=0,
        prefill_reference_concurrency=1,
        prefill_progress=progress,
        completed_prefill_count=completed,
    )


class TwoStageSelectorTests(unittest.TestCase):
    def test_no_safe_candidate_preserves_controller_fallback_contract(self) -> None:
        state = snapshot()
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, ControlState(state.snapshot_hash), ()
        )
        self.assertIsNone(decision.selected_plan)
        self.assertEqual(decision.reason, "NO_SAFE_DECISION")
        self.assertEqual(audit.stage1.status, "NO_SAFE_CANDIDATES")

    def test_decode_without_live_obligation_does_not_create_deadline(self) -> None:
        state = snapshot(decode=(DecodeRequest("d", 0.0, 100),))
        slow = candidate(state, "slow", duration=0.2)
        fast = candidate(state, "fast", duration=0.1)
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, ControlState(state.snapshot_hash), (slow, fast)
        )
        self.assertEqual(audit.stage1.status, "NO_ACTIVE_TBT_OBLIGATION")
        self.assertEqual(decision.selected_plan, fast.plan)

    def test_tbt_stage_filters_by_inclusive_duration_limit(self) -> None:
        state = deadline_snapshot(slack=0.15)
        boundary = candidate(state, "boundary", duration=0.17)
        long = candidate(state, "long", duration=0.171)
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, ControlState(state.snapshot_hash), (long, boundary)
        )
        self.assertEqual(audit.stage1.status, "WITHIN_SLACK")
        self.assertEqual(audit.stage1.eligible_plan_ids, ("boundary",))
        self.assertEqual(decision.selected_plan, boundary.plan)
        self.assertEqual(decision.reason, "TWO_STAGE_TBT_TTFT")

    def test_negative_slack_falls_back_to_shortest_safe_candidate(self) -> None:
        state = deadline_snapshot(slack=-0.05)
        same_fast_large = candidate(state, "b", duration=0.1)
        same_fast_small = candidate(state, "a", duration=0.1)
        slow = candidate(state, "slow", duration=0.2)
        decision, audit = DPPSelector(settings()).select_with_audit(
            state,
            ControlState(state.snapshot_hash),
            (slow, same_fast_large, same_fast_small),
        )
        self.assertEqual(audit.stage1.status, "NO_CANDIDATE_WITHIN_SLACK")
        self.assertEqual(audit.stage1.fallback_plan_id, "a")
        self.assertEqual(decision.selected_plan, same_fast_small.plan)
        self.assertEqual(decision.reason, "TBT_NO_CANDIDATE_MIN_DURATION")

    def test_deadline_and_obligation_must_match_exactly(self) -> None:
        state = snapshot(
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.2),),
            obligations=(Obligation("o", "d", "TBT", 10.3, 9.0),),
        )
        with self.assertRaisesRegex(ValueError, "deadline/obligation mismatch"):
            DPPSelector(settings()).select(
                state,
                ControlState(state.snapshot_hash),
                (candidate(state, "p"),),
            )

    def test_duplicate_or_settled_tbt_obligation_fails_closed(self) -> None:
        state = snapshot(
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.2),),
            obligations=(Obligation("o", "d", "TBT", 10.2, 9.0, settled=True),),
        )
        with self.assertRaisesRegex(ValueError, "must not be settled"):
            DPPSelector(settings()).select(
                state, ControlState(state.snapshot_hash), (candidate(state, "p"),)
            )

    def test_extrapolation_uses_conservative_duration_in_both_stages(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        item = candidate(state, "ood", extrapolated=True, duration=0.1)
        score = DPPSelector(settings()).score_candidate(
            state, ControlState(state.snapshot_hash, (("p", 0.0),)), item
        )
        self.assertAlmostEqual(score.effective_duration, 0.3)

    def test_completed_prefill_resets_predicted_debt(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 40),))
        item = candidate(state, "finish", prefill_items=(("p", 60),), duration=0.2)
        score = DPPSelector(settings()).score_candidate(
            state,
            ControlState(state.snapshot_hash, (("p", 2.0),)),
            item,
            capture_request_details=True,
        )
        detail = score.request_results[0]
        self.assertTrue(detail.completion_this_frame)
        self.assertEqual(detail.predicted_next_debt, 0.0)
        self.assertEqual(detail.drift_contribution, -4.0)

    def test_unfinished_prefill_uses_request_slo_and_prompt_normalization(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 200, 40, ttft_slo_seconds=2.0),)
        )
        item = candidate(state, "partial", prefill_items=(("p", 20),), duration=0.2)
        score = DPPSelector(settings()).score_candidate(
            state,
            ControlState(state.snapshot_hash, (("p", 0.5),)),
            item,
            capture_request_details=True,
        )
        self.assertAlmostEqual(score.request_results[0].predicted_next_debt, 0.5)

    def test_score_is_negative_absolute_prefill_drift(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        item = candidate(state, "partial", prefill_items=(("p", 10),), duration=0.2)
        score = DPPSelector(settings()).score_candidate(
            state,
            ControlState(state.snapshot_hash, (("p", 0.5),)),
            item,
            capture_request_details=True,
        )
        self.assertAlmostEqual(score.score, -score.prefill_drift)
        self.assertAlmostEqual(
            score.ttft_score_rate_old,
            -score.prefill_drift / score.effective_duration,
        )

    def test_duration_still_increases_next_debt(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        control = ControlState(state.snapshot_hash, (("p", 0.2),))
        short = DPPSelector(settings()).score_candidate(
            state,
            control,
            candidate(state, "short", prefill_items=(("p", 5),), duration=0.1),
            capture_request_details=True,
        )
        long = DPPSelector(settings()).score_candidate(
            state,
            control,
            candidate(state, "long", prefill_items=(("p", 5),), duration=0.3),
            capture_request_details=True,
        )
        self.assertLess(
            short.request_results[0].predicted_next_debt,
            long.request_results[0].predicted_next_debt,
        )

    def test_short_zero_vs_longer_nonzero_exposes_counterfactual_direction(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        control = ControlState(state.snapshot_hash, (("p", 0.0),))
        selector = DPPSelector(settings())
        zero_candidate = candidate(
            state, "zero", duration=0.1, template_id="ZERO"
        )
        nonzero_candidate = candidate(
            state,
            "nonzero",
            prefill_items=(("p", 4),),
            duration=0.2,
            template_id="P10",
        )
        zero = selector.score_candidate(
            state,
            control,
            zero_candidate,
        )
        nonzero = selector.score_candidate(
            state,
            control,
            nonzero_candidate,
        )
        self.assertGreater(nonzero.ttft_score_rate_old, zero.ttft_score_rate_old)
        self.assertGreater(
            zero.ttft_score_absolute_new,
            nonzero.ttft_score_absolute_new,
        )
        self.assertEqual(
            selector.select(
                state, control, (zero_candidate, nonzero_candidate)
            ).selected_plan,
            zero_candidate.plan,
        )

    def test_prefill_pressure_can_choose_completion(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        control = ControlState(state.snapshot_hash, (("p", 5.0),))
        zero = candidate(state, "zero", duration=0.1)
        finish = candidate(
            state, "finish", prefill_items=(("p", 100),), duration=0.5
        )
        self.assertEqual(
            DPPSelector(settings()).select(state, control, (zero, finish)).selected_plan,
            finish.plan,
        )

    def test_rank_groups_isclose_then_applies_complete_tie_break(self) -> None:
        selector = DPPSelector(settings())
        scores = (
            manual_score("duration", score=4e-13, duration=0.05),
            manual_score("progress", score=0.0, progress=0.8),
            manual_score("complete", score=2e-13, completed=1),
        )
        ranked, tie = selector._rank_scores(scores)
        self.assertEqual([score.plan_id for score in ranked], ["complete", "progress", "duration"])
        self.assertEqual(tie, ("complete", "progress", "duration"))

    def test_tie_break_finishes_with_budget_then_plan_id(self) -> None:
        selector = DPPSelector(settings())
        scores = (
            manual_score("b", budget=1),
            manual_score("z", budget=0),
            manual_score("a", budget=1),
        )
        ranked, _ = selector._rank_scores(scores)
        self.assertEqual([score.plan_id for score in ranked], ["z", "a", "b"])

    def test_invalid_duration_and_debt_fail_closed(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 10, 0),))
        bad = candidate(state, "bad", duration=math.nan)
        with self.assertRaises(ValueError):
            DPPSelector(settings()).select(
                state, ControlState(state.snapshot_hash, (("p", 0.0),)), (bad,)
            )
        with self.assertRaises(ValueError):
            DPPSelector(settings()).select(
                state,
                ControlState(state.snapshot_hash, (("p", math.inf),)),
                (candidate(state, "ok"),),
            )


class RuntimeOverrideTests(unittest.TestCase):
    def test_forced_stock_remains_development_only(self) -> None:
        resolved, mode = resolve_dpp_runtime_overrides(
            settings(),
            execution_scope="development_nonformal",
            environment={DPP_SELECTION_MODE_ENV: "forced_stock_plan"},
        )
        self.assertIs(resolved.__class__, DPPSettings)
        self.assertEqual(mode, "forced_stock_plan")
        with self.assertRaises(ValueError):
            resolve_dpp_runtime_overrides(
                settings(),
                execution_scope="formal",
                environment={DPP_SELECTION_MODE_ENV: "forced_stock_plan"},
            )

    def test_retired_weight_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "obsolete"):
            resolve_dpp_runtime_overrides(
                settings(),
                execution_scope="development_nonformal",
                environment={DPP_TTFT_DRIFT_WEIGHT_ENV: "1"},
            )


class RequestDebtStateTests(unittest.TestCase):
    def test_prefill_completion_remains_actual_only(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 20),), frame=2)
        store = InMemoryStateStore(settings=settings())
        store.bind_snapshot(state)
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


class TwoStageConfigTests(unittest.TestCase):
    def test_active_config_loads_two_stage_contract(self) -> None:
        runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
        dpp = load_dpp_settings(runtime)
        self.assertTrue(dpp.live_v2_ready)
        self.assertEqual(dpp.algorithm, "two_stage_tbt_ttft_absolute_v1")
        self.assertEqual(dpp.tbt_delta_seconds, 0.020)
        self.assertFalse(dpp.diagnosis_enabled_default)
        self.assertEqual(dpp.diagnosis_schema_version, 2)


if __name__ == "__main__":
    unittest.main()
