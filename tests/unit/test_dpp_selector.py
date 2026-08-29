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
    DPP_STAGE1_MAX_DELTA_N_ENV,
    DPP_TTFT_DRIFT_WEIGHT_ENV,
    resolve_dpp_runtime_overrides,
)


def settings(
    *,
    prefill_ref: int = 1,
    delta: float = 0.020,
    stage1_max_delta_n: int = 0,
) -> DPPSettings:
    return DPPSettings(
        prefill_ref,
        1,
        float.fromhex("0x1.fffffffffffffp+1023"),
        tbt_delta_seconds=delta,
        maximum_incremental_tbt_violations=stage1_max_delta_n,
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
    decode_items: tuple[str, ...] | None = None,
) -> SafeCandidate:
    planned_decode = (
        tuple(item.request_id for item in state.active_decode_requests)
        if decode_items is None
        else decode_items
    )
    plan = BatchPlan(
        plan_id=plan_id,
        snapshot_hash=state.snapshot_hash,
        template_id=template_id,
        prefill_items=prefill_items,
        decode_items=planned_decode,
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=len(planned_decode),
        total_sequences=len(planned_decode) + len(prefill_items),
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
    duration: float = 0.1,
    tokens: int = 0,
    budget: int = 0,
) -> DPPScore:
    return DPPScore(
        plan_id=plan_id,
        effective_duration=duration,
        prefill_service_tokens=tokens,
        prefill_service_rate=score,
        score=score,
        prefill_budget=budget,
        current_prefill_count=0,
        current_prefill_backlog_tokens=0,
        current_decode_count=0,
        decode_coverage_complete=True,
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

    def test_same_violation_set_delta_n_zero_admits_candidate(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.4),),
            obligations=(Obligation("tbt:d", "d", "TBT", 10.4, 9.0),),
        )
        zero = candidate(state, "zero", duration=0.1, template_id="ZERO")
        mixed = candidate(
            state,
            "mixed",
            prefill_items=(("p", 4),),
            duration=0.1,
            template_id="P10",
        )
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, ControlState(state.snapshot_hash, (("p", 0.0),)), (zero, mixed)
        )
        self.assertEqual(audit.stage1.status, "DELTA_N_ADMITTED")
        self.assertEqual(
            audit.stage1.eligible_plan_ids, ("zero", "mixed")
        )
        for result in audit.stage1.candidates:
            self.assertEqual(result.delta_violation_count, 0)
            self.assertTrue(result.passed)
        self.assertEqual(decision.selected_plan, mixed.plan)
        self.assertEqual(decision.reason, "TWO_STAGE_TBT_PREFILL_SERVICE_RATE")

    def test_one_new_miss_rejected_at_zero_limit_admitted_at_one(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.4),),
            obligations=(Obligation("tbt:d", "d", "TBT", 10.4, 9.0),),
        )
        zero = candidate(state, "zero", duration=0.1, template_id="ZERO")
        mixed = candidate(
            state,
            "mixed",
            prefill_items=(("p", 4),),
            duration=0.3,
            template_id="P10",
        )
        control = ControlState(state.snapshot_hash, (("p", 0.0),))
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, control, (zero, mixed)
        )
        self.assertEqual(audit.stage1.status, "DELTA_N_ADMITTED")
        self.assertEqual(audit.stage1.eligible_plan_ids, ("zero",))
        mixed_result = next(
            item for item in audit.stage1.candidates if item.plan_id == "mixed"
        )
        self.assertEqual(mixed_result.delta_violation_count, 1)
        self.assertFalse(mixed_result.passed)
        self.assertEqual(decision.selected_plan, zero.plan)

        decision_n1, audit_n1 = DPPSelector(
            settings(stage1_max_delta_n=1)
        ).select_with_audit(state, control, (zero, mixed))
        self.assertEqual(audit_n1.stage1.eligible_plan_ids, ("zero", "mixed"))
        self.assertEqual(decision_n1.selected_plan, mixed.plan)

    def test_zero_already_missing_same_request_delta_n_zero_admits(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.2),),
            obligations=(Obligation("tbt:d", "d", "TBT", 10.2, 9.0),),
        )
        zero = candidate(state, "zero", duration=0.1, template_id="ZERO")
        mixed = candidate(
            state,
            "mixed",
            prefill_items=(("p", 4),),
            duration=0.3,
            template_id="P10",
        )
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, ControlState(state.snapshot_hash, (("p", 0.0),)), (zero, mixed)
        )
        self.assertEqual(audit.stage1.status, "DELTA_N_ADMITTED")
        self.assertEqual(audit.stage1.reference_plan_id, "zero")
        self.assertGreater(audit.stage1.reference_violation_count, 0)
        mixed_result = next(
            item for item in audit.stage1.candidates if item.plan_id == "mixed"
        )
        self.assertEqual(mixed_result.delta_violation_count, 0)
        self.assertGreater(mixed_result.delta_lateness_seconds, 0.0)
        self.assertTrue(mixed_result.passed)
        self.assertEqual(decision.selected_plan, mixed.plan)

    def test_boundary_served_strict_greater_unserved_greater_equal(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.5),),
            obligations=(Obligation("tbt:d", "d", "TBT", 10.5, 9.0),),
        )
        control = ControlState(state.snapshot_hash, (("p", 0.0),))
        zero = candidate(state, "zero", duration=0.1, template_id="ZERO")
        served_boundary = candidate(
            state,
            "served",
            prefill_items=(("p", 2),),
            duration=0.3,
            template_id="P10",
        )
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, control, (zero, served_boundary)
        )
        served_result = next(
            item for item in audit.stage1.candidates if item.plan_id == "served"
        )
        self.assertEqual(served_result.delta_violation_count, 0)
        self.assertTrue(served_result.passed)
        self.assertEqual(decision.selected_plan, served_boundary.plan)

        unserved = candidate(
            state,
            "unserved",
            duration=0.3,
            template_id="P10",
            decode_items=(),
        )
        decision_u, audit_u = DPPSelector(settings()).select_with_audit(
            state, control, (zero, unserved)
        )
        unserved_result = next(
            item for item in audit_u.stage1.candidates if item.plan_id == "unserved"
        )
        self.assertEqual(unserved_result.delta_violation_count, 1)
        self.assertFalse(unserved_result.passed)
        self.assertEqual(decision_u.selected_plan, zero.plan)

    def test_no_tbt_obligation_admits_all_with_not_needed_reference(self) -> None:
        state = snapshot(decode=(DecodeRequest("d", 0.0, 100),))
        slow = candidate(state, "slow", duration=0.2)
        fast = candidate(state, "fast", duration=0.1)
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, ControlState(state.snapshot_hash), (slow, fast)
        )
        self.assertEqual(audit.stage1.status, "NO_ACTIVE_TBT_OBLIGATION")
        self.assertEqual(audit.stage1.zero_reference_resolution, "NOT_NEEDED")
        self.assertEqual(audit.stage1.eligible_plan_ids, ("slow", "fast"))
        self.assertEqual(decision.selected_plan, fast.plan)

    def test_zero_reference_missing_fails_fast(self) -> None:
        state = snapshot(
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.4),),
            obligations=(Obligation("tbt:d", "d", "TBT", 10.4, 9.0),),
        )
        partial = candidate(
            state, "partial", template_id="P10", decode_items=()
        )
        with self.assertRaisesRegex(RuntimeError, "ZERO_REFERENCE_MISSING"):
            DPPSelector(settings()).select(
                state, ControlState(state.snapshot_hash), (partial,)
            )

    def test_stock_identity_serves_as_zero_reference(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.4),),
            obligations=(Obligation("tbt:d", "d", "TBT", 10.4, 9.0),),
        )
        stock = candidate(state, "stock", duration=0.1, template_id="STOCK")
        mixed = candidate(
            state,
            "mixed",
            prefill_items=(("p", 4),),
            duration=0.1,
            template_id="P10",
        )
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, ControlState(state.snapshot_hash, (("p", 0.0),)), (stock, mixed)
        )
        self.assertEqual(audit.stage1.zero_reference_resolution, "STOCK_IDENTITY")
        self.assertEqual(audit.stage1.reference_plan_id, "stock")
        self.assertEqual(decision.selected_plan, mixed.plan)

    def test_two_new_misses_require_limit_two(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(
                DecodeRequest("a", 0.0, 100, tbt_deadline=10.4),
                DecodeRequest("b", 0.0, 100, tbt_deadline=10.45),
            ),
            obligations=(
                Obligation("tbt:a", "a", "TBT", 10.4, 9.0),
                Obligation("tbt:b", "b", "TBT", 10.45, 9.0),
            ),
        )
        control = ControlState(state.snapshot_hash, (("p", 0.0),))
        zero = candidate(state, "zero", duration=0.1, template_id="ZERO")
        mixed = candidate(
            state,
            "mixed",
            prefill_items=(("p", 4),),
            duration=0.35,
            template_id="P10",
        )
        _, audit = DPPSelector(settings(stage1_max_delta_n=1)).select_with_audit(
            state, control, (zero, mixed)
        )
        mixed_result = next(
            item for item in audit.stage1.candidates if item.plan_id == "mixed"
        )
        self.assertEqual(mixed_result.delta_violation_count, 2)
        self.assertFalse(mixed_result.passed)
        decision, audit2 = DPPSelector(
            settings(stage1_max_delta_n=2)
        ).select_with_audit(state, control, (zero, mixed))
        self.assertEqual(audit2.stage1.eligible_plan_ids, ("zero", "mixed"))
        self.assertEqual(decision.selected_plan, mixed.plan)

    def test_stage1_max_delta_n_defaults_to_zero(self) -> None:
        self.assertEqual(settings().maximum_incremental_tbt_violations, 0)

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

    def test_score_is_actual_prefill_tokens_per_effective_duration(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 40),))
        item = candidate(state, "finish", prefill_items=(("p", 60),), duration=0.2)
        score = DPPSelector(settings()).score_candidate(
            state,
            ControlState(state.snapshot_hash, (("p", 2.0),)),
            item,
            capture_request_details=True,
        )
        self.assertEqual(score.prefill_service_tokens, 60)
        self.assertAlmostEqual(score.prefill_service_rate, 300.0)
        self.assertEqual(score.score, score.prefill_service_rate)
        self.assertEqual(score.current_prefill_backlog_tokens, 60)

    def test_stage2_does_not_read_ttft_debt_or_request_slo(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 200, 40, ttft_slo_seconds=2.0),)
        )
        item = candidate(state, "partial", prefill_items=(("p", 20),), duration=0.2)
        empty_debt = DPPSelector(settings()).score_candidate(
            state,
            ControlState(state.snapshot_hash),
            item,
        )
        arbitrary_debt = DPPSelector(settings()).score_candidate(
            state,
            ControlState(state.snapshot_hash, (("unrelated", math.inf),)),
            item,
        )
        self.assertEqual(empty_debt, arbitrary_debt)

    def test_duration_normalizes_prefill_service_rate(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 100, 0),))
        control = ControlState(state.snapshot_hash)
        short = DPPSelector(settings()).score_candidate(
            state,
            control,
            candidate(state, "short", prefill_items=(("p", 5),), duration=0.1),
        )
        long = DPPSelector(settings()).score_candidate(
            state,
            control,
            candidate(state, "long", prefill_items=(("p", 5),), duration=0.3),
        )
        self.assertGreater(short.score, long.score)

    def test_zero_cannot_win_when_eligible_candidate_has_actual_prefill(self) -> None:
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
        self.assertGreater(nonzero.score, zero.score)
        self.assertEqual(
            selector.select(
                state, control, (zero_candidate, nonzero_candidate)
            ).selected_plan,
            nonzero_candidate.plan,
        )

    def test_positive_service_does_not_isclose_tie_with_zero(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 1, 0),))
        zero = candidate(state, "zero", duration=0.1, template_id="ZERO")
        tiny = candidate(
            state,
            "tiny",
            prefill_items=(("p", 1),),
            duration=1.0e15,
            template_id="P10",
        )
        decision = DPPSelector(settings()).select(
            state, ControlState(state.snapshot_hash), (zero, tiny)
        )
        self.assertEqual(decision.selected_plan, tiny.plan)

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

    def test_rank_groups_isclose_then_applies_service_tie_break(self) -> None:
        selector = DPPSelector(settings())
        scores = (
            manual_score("long", score=1.0, duration=0.2, tokens=20),
            manual_score("short-low", score=1.0 + 4e-10, duration=0.1, tokens=10),
            manual_score("short-high", score=1.0 + 2e-10, duration=0.1, tokens=11),
        )
        ranked, tie = selector._rank_scores(scores)
        self.assertEqual(
            [score.plan_id for score in ranked],
            ["short-high", "short-low", "long"],
        )
        self.assertEqual(tie, ("short-high", "short-low", "long"))

    def test_tie_break_finishes_with_budget_then_plan_id(self) -> None:
        selector = DPPSelector(settings())
        scores = (
            manual_score("b", budget=1),
            manual_score("z", budget=0),
            manual_score("a", budget=1),
        )
        ranked, _ = selector._rank_scores(scores)
        self.assertEqual([score.plan_id for score in ranked], ["z", "a", "b"])

    def test_invalid_duration_fails_closed_but_debt_is_not_a_score_input(self) -> None:
        state = snapshot(prefill=(PrefillRequest("p", 0.0, 10, 0),))
        bad = candidate(state, "bad", duration=math.nan)
        with self.assertRaises(ValueError):
            DPPSelector(settings()).select(
                state, ControlState(state.snapshot_hash, (("p", 0.0),)), (bad,)
            )
        decision = DPPSelector(settings()).select(
            state,
            ControlState(state.snapshot_hash, (("p", math.inf),)),
            (candidate(state, "ok"),),
        )
        self.assertEqual(decision.selected_plan.plan_id, "ok")


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

    def test_stage1_delta_n_override_accepted_in_development(self) -> None:
        resolved, mode = resolve_dpp_runtime_overrides(
            settings(),
            execution_scope="development_nonformal",
            environment={DPP_STAGE1_MAX_DELTA_N_ENV: "4"},
        )
        self.assertEqual(resolved.maximum_incremental_tbt_violations, 4)
        self.assertEqual(mode, "normal")

    def test_stage1_delta_n_override_rejected_in_formal_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "development_nonformal"):
            resolve_dpp_runtime_overrides(
                settings(),
                execution_scope="formal",
                environment={DPP_STAGE1_MAX_DELTA_N_ENV: "4"},
            )

    def test_stage1_delta_n_override_rejects_invalid_values(self) -> None:
        for invalid in ("x", "-1", "1.5", "true"):
            with self.assertRaises(ValueError):
                resolve_dpp_runtime_overrides(
                    settings(),
                    execution_scope="development_nonformal",
                    environment={DPP_STAGE1_MAX_DELTA_N_ENV: invalid},
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
        self.assertEqual(
            dpp.algorithm, "two_stage_zero_relative_tbt_prefill_service_rate_v2b"
        )
        self.assertEqual(dpp.tbt_delta_seconds, 0.020)
        self.assertEqual(dpp.stage1_mode, "zero_relative_incremental_violation")
        self.assertEqual(dpp.stage1_duration_source, "conservative_duration")
        self.assertEqual(dpp.maximum_incremental_tbt_violations, 0)
        self.assertFalse(dpp.diagnosis_enabled_default)
        self.assertEqual(dpp.diagnosis_schema_version, 4)


if __name__ == "__main__":
    unittest.main()
