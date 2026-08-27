from __future__ import annotations

import inspect
import unittest

from dpp_scheduler.budget_resolver import (
    RESOLUTION_INVERTED_OK,
    RESOLUTION_INVERTED_OOD,
    RESOLUTION_NO_DECODE_NO_BACKLOG,
    RESOLUTION_NO_DECODE_USE_MAX,
    RESOLUTION_NO_FEASIBLE_BUDGET,
    RESOLUTION_PREDICTOR_INVALID,
    DEFAULT_INVERSION_BUDGET_GRID,
    BudgetResolution,
    NullBudgetResolver,
    RidgeBudgetResolver,
    derive_executable_inversion_grid,
)
from dpp_scheduler.candidate_generator import (
    BUDGET_FRACTIONS,
    FRACTION_LABELS,
    CandidateGenerator,
    build_stock_plan,
    derive_candidate_budgets,
    rank_prefill_requests,
)
from dpp_scheduler.contracts import (
    DecodeRequest,
    Obligation,
    PrefillRequest,
    Prediction,
    StateSnapshot,
)
from dpp_scheduler.predictor import DurationPredictor
from dpp_scheduler.settings import SchedulerSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    token_budget: int = 4096,
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
        free_kv_blocks=1000,
        kv_block_size=16,
        token_budget=token_budget,
        sequence_budget=sequence_budget,
        total_kv_blocks=1000,
    )


class _FixedPredictor(DurationPredictor):
    """Predictor stub that returns a fixed expected_duration for every plan."""

    def __init__(self, expected_duration: float, in_support: bool = True) -> None:
        self._expected = float(expected_duration)
        self._in_support = bool(in_support)

    def predict(self, snapshot, plans):
        return tuple(
            Prediction(
                plan_id=plan.plan_id,
                snapshot_hash=snapshot.snapshot_hash,
                expected_duration=self._expected,
                conservative_duration=self._expected + 0.02,
                in_support=self._in_support,
            )
            for plan in plans
        )


class _BudgetKeyedPredictor(DurationPredictor):
    """Predictor stub that maps the shadow plan's ``template_id`` → duration.

    The shadow plans built by :class:`RidgeBudgetResolver` carry
    ``template_id = "SHADOW:INVERSION:requested_{budget}"``. This stub parses
    that suffix and looks up the corresponding duration in the supplied table,
    so the resolver's grid sweep can observe distinct predictions per budget
    even when no actual Prefill backlog is present.
    """

    def __init__(
        self,
        duration_by_budget: dict[int, float],
        in_support: bool = True,
    ) -> None:
        self._table = dict(duration_by_budget)
        self._in_support = bool(in_support)

    @staticmethod
    def _budget_from_template(template_id: str) -> int | None:
        prefix = "SHADOW:INVERSION:requested_"
        if not template_id.startswith(prefix):
            return None
        suffix = template_id[len(prefix):]
        try:
            return int(suffix)
        except ValueError:
            return None

    def predict(self, snapshot, plans):
        results = []
        for plan in plans:
            budget = self._budget_from_template(plan.template_id)
            duration = self._table.get(budget) if budget is not None else None
            results.append(
                Prediction(
                    plan_id=plan.plan_id,
                    snapshot_hash=snapshot.snapshot_hash,
                    expected_duration=duration,
                    conservative_duration=(
                        None if duration is None else duration + 0.02
                    ),
                    in_support=self._in_support,
                )
            )
        return tuple(results)


# ---------------------------------------------------------------------------
# Fixed-fraction and Stock-plan behavior
# ---------------------------------------------------------------------------


class FixedFractionCandidateTests(unittest.TestCase):
    def test_fraction_grid_uses_pmax(self) -> None:
        budgets = derive_candidate_budgets(
            token_budget=1001,
            decode_count=1,
            total_prefill_backlog=2000,
        )
        self.assertEqual([label for label, _, _ in budgets], list(FRACTION_LABELS))
        self.assertEqual([value for _, value, _ in budgets], list(range(100, 1001, 100)))
        self.assertEqual([value for _, _, value in budgets], list(BUDGET_FRACTIONS))

    def test_running_first_then_waiting_fcfs(self) -> None:
        state = _snapshot(
            prefill=(
                PrefillRequest("newer", 2.0, 100, 0, ordinal=2),
                PrefillRequest("running", 9.0, 100, 20, is_running=True, ordinal=0),
                PrefillRequest("older", 1.0, 100, 0, ordinal=1),
            )
        )
        self.assertEqual(
            [item.request_id for item in rank_prefill_requests(state)],
            ["running", "older", "newer"],
        )

    def test_generator_has_at_most_twelve_canonical_candidates(self) -> None:
        state = _snapshot(
            prefill=(PrefillRequest("p", 0.0, 4000, 0),),
            decode=(DecodeRequest("d", 0.0, 20),),
            token_budget=2048,
        )
        plans = CandidateGenerator().generate(state)
        self.assertLessEqual(len(plans), 12)
        self.assertEqual(len(plans), len({(p.prefill_items, p.decode_items) for p in plans}))
        self.assertEqual(plans[0].template_id, "STOCK")
        self.assertIn("ZERO", {plan.template_id for plan in plans})

    def test_stock_plan_follows_running_interleaving_then_waiting(self) -> None:
        state = _snapshot(
            prefill=(
                PrefillRequest("rp", 0.0, 50, 10, is_running=True, ordinal=0),
                PrefillRequest("wp", 2.0, 100, 0, ordinal=2),
            ),
            decode=(DecodeRequest("d", 1.0, 32, ordinal=1),),
            token_budget=80,
        )
        plan = build_stock_plan(state)
        self.assertEqual(plan.template_id, "STOCK")
        self.assertEqual(plan.prefill_items, (("rp", 40), ("wp", 39)))
        self.assertEqual(plan.decode_items, ("d",))
        self.assertEqual(plan.total_prefill_tokens + plan.total_decode_tokens, 80)

    def test_fraction_candidates_clamp_to_available_kv(self) -> None:
        state = StateSnapshot.create(
            frame_id=1,
            timestamp=10.0,
            waiting_prefill_requests=(PrefillRequest("p", 0.0, 100, 0),),
            active_decode_requests=(),
            active_ttft_obligations=(),
            active_tbt_obligations=(),
            recovery_requests=(),
            free_kv_blocks=1,
            kv_block_size=16,
            token_budget=100,
            sequence_budget=64,
            total_kv_blocks=1,
        )
        plans = CandidateGenerator().generate(state)
        fractions = [plan for plan in plans if plan.template_id.startswith("P")]
        self.assertTrue(fractions)
        self.assertTrue(all(plan.total_prefill_tokens <= 16 for plan in fractions))
        self.assertTrue(all(plan.projected_kv_blocks <= 1 for plan in fractions))

    def test_stock_plan_obeys_sequence_capacity(self) -> None:
        state = _snapshot(
            prefill=(
                PrefillRequest("first", 0.0, 10, 0, ordinal=0),
                PrefillRequest("second", 1.0, 10, 0, ordinal=1),
            ),
            sequence_budget=1,
        )
        self.assertEqual(build_stock_plan(state).prefill_items, (("first", 10),))

    def test_no_predictor_dependency(self) -> None:
        source = inspect.getsource(
            __import__("dpp_scheduler.candidate_generator", fromlist=["x"])
        )
        self.assertNotIn("DurationPredictor", source)
        self.assertNotIn("BudgetResolver", source)

    def test_empty_snapshot_returns_no_candidates(self) -> None:
        self.assertEqual(CandidateGenerator().generate(_snapshot()), ())


# ---------------------------------------------------------------------------
# BudgetResolver inversion behavior
# ---------------------------------------------------------------------------


class RidgeBudgetResolverTests(unittest.TestCase):
    def test_inverts_to_largest_feasible_budget(self) -> None:
        # Grid budgets × predicted durations: 0..128 fit in target, 256+ exceed.
        predictor = _BudgetKeyedPredictor(
            duration_by_budget={
                0: 0.100,
                64: 0.150,
                128: 0.200,
                256: 0.500,
                384: 0.600,
                512: 0.700,
                768: 0.900,
                1024: 1.100,
                1536: 1.300,
                2048: 1.500,
            }
        )
        # Add Prefill backlog so the resolver does not degenerate to P=0
        # due to resource clamp; this test is about inversion, not clamping.
        prefill = (PrefillRequest("p", 0.0, 4096, 0),)
        decode = (
            DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),
        )
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=predictor,
            settings=SchedulerSettings.provisional(),
            safety_margin_seconds=0.020,
        )
        resolution = resolver.resolve(state)
        # s_k^min = 0.5s; T_target = 0.48s. 128 (0.200) is feasible; 256 (0.500)
        # exceeds. So P = 128.
        self.assertEqual(resolution.base_prefill_budget, 128)
        self.assertEqual(resolution.resolution_status, RESOLUTION_INVERTED_OK)
        self.assertEqual(resolution.target_duration_seconds, 0.480)
        self.assertIn(128, resolution.feasible_grid_budgets)
        self.assertNotIn(256, resolution.feasible_grid_budgets)

    def test_no_decode_uses_max_resource(self) -> None:
        # No Decode requests → resource-cap path. With backlog=400, token_budget=1024
        # and decode_count=0, P = min(400, 1024) = 400.
        prefill = (PrefillRequest("p", 0.0, 400, 0),)
        state = _snapshot(prefill=prefill, decode=(), token_budget=1024)
        resolver = RidgeBudgetResolver(
            predictor=_FixedPredictor(0.20),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 400)
        self.assertEqual(resolution.resolution_status, RESOLUTION_NO_DECODE_USE_MAX)
        self.assertIsNone(resolution.target_duration_seconds)

    def test_no_decode_no_backlog(self) -> None:
        state = _snapshot(prefill=(), decode=())
        resolver = RidgeBudgetResolver(
            predictor=_FixedPredictor(0.20),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 0)
        self.assertEqual(resolution.resolution_status, RESOLUTION_NO_DECODE_NO_BACKLOG)

    def test_no_feasible_budget(self) -> None:
        # Every grid budget produces a duration > target_duration.
        predictor = _BudgetKeyedPredictor(
            duration_by_budget={
                0: 0.500,
                64: 0.600,
                128: 0.700,
                256: 0.800,
                384: 0.900,
                512: 1.000,
                768: 1.200,
                1024: 1.400,
                1536: 1.600,
                2048: 2.000,
            }
        )
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.3),)
        state = _snapshot(decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=predictor,
            settings=SchedulerSettings.provisional(),
            safety_margin_seconds=0.020,
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 0)
        self.assertEqual(resolution.resolution_status, RESOLUTION_NO_FEASIBLE_BUDGET)

    def test_ood_marks_inverted_ood(self) -> None:
        # Same Predictor table as the inversion test, but every prediction is
        # flagged as out-of-support. Resolution status should flip to OOD.
        predictor = _BudgetKeyedPredictor(
            duration_by_budget={
                0: 0.100,
                64: 0.150,
                128: 0.200,
                256: 0.500,
                384: 0.600,
                512: 0.700,
                768: 0.900,
                1024: 1.100,
                1536: 1.300,
                2048: 1.500,
            },
            in_support=False,
        )
        prefill = (PrefillRequest("p", 0.0, 4096, 0),)
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=predictor,
            settings=SchedulerSettings.provisional(),
            safety_margin_seconds=0.020,
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 128)
        self.assertEqual(resolution.resolution_status, RESOLUTION_INVERTED_OOD)
        self.assertEqual(resolution.predictor_in_support_ratio, 0.0)

    def test_safety_margin_changes_p(self) -> None:
        # Same fixed Predictor table — safety_margin controls how much of the
        # TBT slack the resolver may consume. Tighter margin → larger P.
        predictor = _BudgetKeyedPredictor(
            duration_by_budget={
                0: 0.100,
                64: 0.150,
                128: 0.200,
                256: 0.500,
                384: 0.600,
                512: 0.700,
                768: 0.900,
                1024: 1.100,
                1536: 1.300,
                2048: 1.500,
            }
        )
        prefill = (PrefillRequest("p", 0.0, 4096, 0),)
        # First case: tight safety margin (0.05) on slack=0.4 → target=0.35.
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.4),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        resolver_tight = RidgeBudgetResolver(
            predictor=predictor,
            settings=SchedulerSettings.provisional(),
            safety_margin_seconds=0.050,
        )
        # 0.10/0.15/0.20 ≤ 0.35 ✓ → 128 feasible; 0.50 > 0.35 ✗.
        self.assertEqual(resolver_tight.resolve(state).base_prefill_budget, 128)

        # Second case: wider deadline (slack=0.6), default margin → target=0.58.
        decode2 = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.6),)
        state2 = _snapshot(prefill=prefill, decode=decode2, timestamp=10.0)
        resolver_default = RidgeBudgetResolver(
            predictor=predictor,
            settings=SchedulerSettings.provisional(),
            safety_margin_seconds=0.020,
        )
        # 0.10/0.15/0.20/0.50/0.55? Actually we only have integer-budget
        # durations. 0.500 ≤ 0.580 → 256 feasible; 0.60 ≤ 0.580 ✗ → 384 not.
        # So P = 256.
        self.assertEqual(resolver_default.resolve(state2).base_prefill_budget, 256)

    def test_null_resolver_returns_zero(self) -> None:
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(decode=decode, timestamp=10.0)
        resolution = NullBudgetResolver().resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 0)
        self.assertEqual(resolution.resolution_status, RESOLUTION_NO_FEASIBLE_BUDGET)

    def test_predictor_returns_mismatched_predictions(self) -> None:
        class _WrongShapePredictor(DurationPredictor):
            def predict(self, snapshot, plans):  # noqa: ARG002
                return ()  # Wrong length on purpose.

        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        # A small backlog so the resolver actually attempts the Predictor
        # sweep; with backlog=0 the resolver short-circuits to
        # NO_FEASIBLE_BUDGET before ever calling the Predictor.
        prefill = (PrefillRequest("p", 0.0, 64, 0),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=_WrongShapePredictor(),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 0)
        self.assertEqual(resolution.resolution_status, RESOLUTION_PREDICTOR_INVALID)

    def test_zero_max_executable_short_circuits(self) -> None:
        # With backlog = 0 the resolver must short-circuit to
        # NO_FEASIBLE_BUDGET before touching the Predictor, even when a
        # live Decode TBT deadline would otherwise drive the inversion path.
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=_FixedPredictor(0.10),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 0)
        self.assertEqual(resolution.max_executable_prefill_budget, 0)
        self.assertEqual(
            resolution.resolution_status, RESOLUTION_NO_FEASIBLE_BUDGET
        )
        self.assertEqual(resolution.executable_grid_budgets, (0,))

    # --- §9 BudgetResolver actual-budget and grid-clamp semantics ------

    def test_backlog_smaller_than_grid_clamps_inversion_grid(self) -> None:
        # §9.1: backlog = 500; grid 768/1024/1536/2048 must collapse to 500
        # rather than remain as distinct test points.
        prefill = (PrefillRequest("p", 0.0, 500, 0),)
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=_FixedPredictor(0.10),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.max_executable_prefill_budget, 500)
        # Clamp + add P_max + dedup + sort yields exactly {0, 64, 128,
        # 256, 384, 500}.
        self.assertEqual(
            resolution.executable_grid_budgets,
            (0, 64, 128, 256, 384, 500),
        )
        # High grid values never appear as separate test points.
        for collapsed in (768, 1024, 1536, 2048):
            self.assertNotIn(collapsed, resolution.executable_grid_budgets)
        # The configured grid is still preserved for diagnostic introspection.
        self.assertEqual(
            resolution.configured_grid_budgets, DEFAULT_INVERSION_BUDGET_GRID
        )
        self.assertEqual(resolution.requested_grid_budgets, DEFAULT_INVERSION_BUDGET_GRID)

    def test_p_collapses_to_actual_when_predicted_at_max(self) -> None:
        # §9.2: shadow fill caps the work at 512 even when the configured
        # grid asks for 1024/1536/2048; P must therefore be 512, not 2048.
        prefill = (PrefillRequest("p", 0.0, 512, 0),)
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        # Feasible at every budget the predictor knows about; P_max becomes
        # the largest *actual* feasible budget, not the largest grid value.
        predictor = _BudgetKeyedPredictor(
            duration_by_budget={
                0: 0.10,
                64: 0.10,
                128: 0.10,
                256: 0.10,
                384: 0.10,
                512: 0.10,
                768: 0.10,
                1024: 0.10,
                1536: 0.10,
                2048: 0.10,
            }
        )
        resolver = RidgeBudgetResolver(
            predictor=predictor,
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        # Backlog is 512 so max_executable_prefill = 512.
        self.assertEqual(resolution.max_executable_prefill_budget, 512)
        self.assertEqual(resolution.base_prefill_budget, 512)
        # 512 must appear as the canonical actual budget; 2048 (the largest
        # configured value) must not.
        self.assertEqual(
            resolution.actual_shadow_prefill_budgets[-1], 512
        )
        self.assertIn(512, resolution.feasible_actual_budgets)
        for blocked in (768, 1024, 1536, 2048):
            self.assertNotIn(blocked, resolution.actual_shadow_prefill_budgets)
            self.assertNotIn(blocked, resolution.feasible_actual_budgets)

    def test_token_capacity_clamps_max_executable(self) -> None:
        # §9.3: backlog = 2000 but token_budget - decode_count = 700; P_max
        # must therefore be 700 and no plan may exceed it.
        prefill = (PrefillRequest("p", 0.0, 2000, 0),)
        decode = tuple(
            DecodeRequest(f"d{i}", 0.0, 20, tbt_deadline=10.5) for i in range(5)
        )
        state = _snapshot(
            prefill=prefill, decode=decode, token_budget=705, timestamp=10.0
        )
        resolver = RidgeBudgetResolver(
            predictor=_FixedPredictor(0.10),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.max_executable_prefill_budget, 700)
        self.assertLessEqual(resolution.base_prefill_budget, 700)
        # 700 appears in the executable grid; values larger than the cap do
        # not appear as distinct entries.
        self.assertIn(700, resolution.executable_grid_budgets)
        for oversized in (768, 1024, 1536, 2048):
            self.assertNotIn(oversized, resolution.executable_grid_budgets)
        # The feasible-actual set and the chosen P are both <= the cap.
        for value in resolution.feasible_actual_budgets:
            self.assertLessEqual(value, 700)
        for value in resolution.actual_shadow_prefill_budgets:
            self.assertLessEqual(value, 700)

    def test_p_max_added_when_not_in_configured_grid(self) -> None:
        # §9.4: P_max = 450 is not present in the configured grid, yet the
        # executable grid must contain 450 so the Predictor sweep evaluates
        # the resource boundary directly.
        prefill = (PrefillRequest("p", 0.0, 450, 0),)
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=_FixedPredictor(0.10),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.max_executable_prefill_budget, 450)
        self.assertIn(450, resolution.executable_grid_budgets)
        # 450 is the last entry (ceiling) — verifies it was appended after
        # the configured grid.
        self.assertEqual(resolution.executable_grid_budgets[-1], 450)
        # And the helper alone produces the same result.
        self.assertEqual(
            derive_executable_inversion_grid(DEFAULT_INVERSION_BUDGET_GRID, 450),
            (0, 64, 128, 256, 384, 450),
        )

    def test_shadow_fill_actual_used_for_selection(self) -> None:
        # §9.5: requested=512 collapses to actual=384 due to sequence /
        # minimum-chunk constraints; feasibility and P must reflect 384,
        # never 512.
        settings = SchedulerSettings(minimum_prefill_chunk_tokens=400)
        # Running A (384 remaining) + waiting B (600). With min_chunk=400
        # the partial fill of B at requested=512 (= 128 tokens) is skipped,
        # so actual_prefill_tokens for the requested=512 plan is exactly 384.
        prefill = (
            PrefillRequest("A", 0.0, 384, 0, is_running=True),
            PrefillRequest("B", 1.0, 600, 0),
        )
        decode = (DecodeRequest("d", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        # Low grid points are feasible; mid grid points are infeasible.
        predictor = _BudgetKeyedPredictor(
            duration_by_budget={
                0: 0.10,
                64: 0.10,
                128: 0.10,
                256: 0.10,
                384: 0.10,
                512: 1.0,
                768: 1.0,
                1024: 1.0,
                1536: 1.0,
                2048: 1.0,
            }
        )
        resolver = RidgeBudgetResolver(
            predictor=predictor,
            settings=settings,
        )
        resolution = resolver.resolve(state)
        # backlog=984, decode_count=1, token_budget=4096 → P_max = 984.
        self.assertEqual(resolution.max_executable_prefill_budget, 984)
        # 384 appears as an actual budget because requested=512 collapses
        # to actual=384 under the minimum-chunk constraint.
        self.assertIn(384, resolution.actual_shadow_prefill_budgets)
        # 512 is NOT an actual budget (it was never realized by the fill).
        self.assertNotIn(512, resolution.actual_shadow_prefill_budgets)
        # The feasible-actual set uses 384 as the largest feasible actual;
        # chosen P is therefore 384, NOT 512.
        self.assertEqual(resolution.base_prefill_budget, 384)
        self.assertIn(384, resolution.feasible_actual_budgets)
        self.assertNotIn(512, resolution.feasible_actual_budgets)

    def test_live_candidate_generator_is_decoupled_from_resolver(self) -> None:
        # Preserve the resolver implementation for diagnostics, but prove that
        # live Candidate generation no longer accepts or calls it.
        prefill = tuple(
            PrefillRequest(
                request_id=f"p{i}", arrival_time=float(i),
                token_count=400, prefilled_tokens=0, ordinal=i,
            )
            for i in range(5)
        )
        decode = (DecodeRequest("d1", 0.0, 20, tbt_deadline=10.5),)
        state = _snapshot(prefill=prefill, decode=decode, timestamp=10.0)
        with self.assertRaises(TypeError):
            CandidateGenerator(
                SchedulerSettings.provisional(),
                budget_resolver=RidgeBudgetResolver(
                    predictor=_FixedPredictor(0.10),
                    settings=SchedulerSettings.provisional(),
                ),
            )
        plans = CandidateGenerator(SchedulerSettings.provisional()).generate(state)
        self.assertIn("STOCK", {plan.template_id for plan in plans})
        self.assertTrue(any(plan.template_id.startswith("P") for plan in plans))


class DeriveExecutableInversionGridTests(unittest.TestCase):
    """Direct coverage of :func:`derive_executable_inversion_grid`."""

    def test_clamp_high_values_to_ceiling(self) -> None:
        # grid 512/768/1024/1536/2048 collapses to 500 when ceiling = 500.
        grid = (0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048)
        self.assertEqual(
            derive_executable_inversion_grid(grid, 500),
            (0, 64, 128, 256, 384, 500),
        )

    def test_appends_ceiling_when_not_in_configured(self) -> None:
        # ceiling 450 is not in the default grid; it must be appended.
        self.assertEqual(
            derive_executable_inversion_grid(
                DEFAULT_INVERSION_BUDGET_GRID, 450
            ),
            (0, 64, 128, 256, 384, 450),
        )

    def test_no_clamp_when_ceiling_above_grid_max(self) -> None:
        # ceiling = 5000 leaves the default grid untouched and adds 5000.
        self.assertEqual(
            derive_executable_inversion_grid(
                DEFAULT_INVERSION_BUDGET_GRID, 5000
            ),
            (0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 5000),
        )

    def test_zero_or_negative_ceiling_collapses_to_zero(self) -> None:
        # A non-positive ceiling means no Prefill work is executable; the
        # helper returns the canonical zero budget so the resolver still
        # has a single grid point to report.
        self.assertEqual(
            derive_executable_inversion_grid(DEFAULT_INVERSION_BUDGET_GRID, 0),
            (0,),
        )
        self.assertEqual(
            derive_executable_inversion_grid(DEFAULT_INVERSION_BUDGET_GRID, -10),
            (0,),
        )

    def test_dedups_collapsing_values(self) -> None:
        # Two grid points that collapse to the same ceiling must dedup.
        self.assertEqual(
            derive_executable_inversion_grid((10, 10, 20, 30), 20),
            (10, 20),
        )


if __name__ == "__main__":
    unittest.main()
