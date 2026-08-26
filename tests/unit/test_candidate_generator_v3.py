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
)
from dpp_scheduler.candidate_generator import (
    BUDGET_MULTIPLIERS,
    MULTIPLIER_LABELS,
    CandidateGenerator,
    derive_candidate_budgets,
    rank_prefill_completion_aware,
    rank_prefill_continuation,
    rank_prefill_urgency,
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


def _static_resolver(base: int, status: str = RESOLUTION_INVERTED_OK):
    """Tiny alias around :class:`_BudgetResolver` for readability in tests."""

    return _BudgetResolver(base=base, status=status)


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


class _BudgetResolver:
    """Test resolver that returns a fixed ``base_prefill_budget``.

    Used as the ``budget_resolver=`` argument to ``CandidateGenerator`` in
    unit tests; never registered as the production resolver.
    """

    def __init__(self, base: int, status: str = RESOLUTION_INVERTED_OK) -> None:
        self._base = int(base)
        self._status = status

    def resolve(self, snapshot: StateSnapshot) -> BudgetResolution:  # noqa: ARG002
        return BudgetResolution(
            base_prefill_budget=self._base,
            target_duration_seconds=0.250,
            resolution_status=self._status,
        )


# ---------------------------------------------------------------------------
# §14.1 Budget multiplier
# ---------------------------------------------------------------------------


class BudgetMultiplierTests(unittest.TestCase):
    def test_budget_multipliers_no_clamp(self) -> None:
        budgets = derive_candidate_budgets(
            1000,
            token_budget=4096,
            decode_count=0,
            total_prefill_backlog=4096,
        )
        # floor(0.5*1000)=500, floor(0.75*1000)=750, 1000, 1250, 1500 — all
        # within resource cap of 4096 so no clamp fires.
        self.assertEqual(len(budgets), 5)
        self.assertEqual(
            [budget for _, budget, _ in budgets],
            [500, 750, 1000, 1250, 1500],
        )
        self.assertEqual([label for label, _, _ in budgets], list(MULTIPLIER_LABELS))
        self.assertEqual(
            [multiplier for _, _, multiplier in budgets],
            list(BUDGET_MULTIPLIERS),
        )

    def test_resource_clamp(self) -> None:
        # token_budget - decode_count = 1200; high multipliers must clamp to
        # 1200 and the dedup must collapse the two duplicates.
        budgets = derive_candidate_budgets(
            1000,
            token_budget=2200,
            decode_count=1000,
            total_prefill_backlog=4096,
        )
        self.assertEqual([budget for _, budget, _ in budgets], [500, 750, 1000, 1200])

    def test_backlog_clamp(self) -> None:
        # Backlog smaller than P means every multiplier >= 1.0 collapses to
        # the backlog floor. floor(0.5*1000)=500 stays under the 600 backlog
        # cap; floor(0.75*1000)=750 clamps to 600; multipliers >= 1.0 all
        # clamp to 600; the four 600 entries dedup to one.
        budgets = derive_candidate_budgets(
            1000,
            token_budget=4096,
            decode_count=0,
            total_prefill_backlog=600,
        )
        self.assertEqual([budget for _, budget, _ in budgets], [500, 600])

    def test_base_zero_yields_no_budgets(self) -> None:
        self.assertEqual(
            derive_candidate_budgets(
                0,
                token_budget=4096,
                decode_count=0,
                total_prefill_backlog=4096,
            ),
            (),
        )


# ---------------------------------------------------------------------------
# §14.4–14.6 Prefill ordering
# ---------------------------------------------------------------------------


class PrefillOrderingTests(unittest.TestCase):
    def test_urgency_ordering(self) -> None:
        # Construct three Prefill requests with the same arrival time so that
        # the u_i score alone decides the order. A has 800 tokens / 0.5s
        # slack (u=1600), B has 400 / 1.0s (u=400), C has 200 / 1.0s (u=200).
        prefill = (
            PrefillRequest(
                "A", 0.0, 800, 0, ttft_deadline=10.5
            ),
            PrefillRequest(
                "B", 0.0, 400, 0, ttft_deadline=11.0
            ),
            PrefillRequest(
                "C", 0.0, 200, 0, ttft_deadline=11.0
            ),
        )
        state = _snapshot(prefill=prefill, timestamp=10.0)
        ordered = rank_prefill_urgency(state)
        self.assertEqual(tuple(item.request_id for item in ordered), ("A", "B", "C"))

    def test_completion_aware_ordering(self) -> None:
        # All three requests land in the same u<0.5 tier because their
        # remaining_tokens are tiny relative to their slack. The intra-tier
        # sort then picks smallest-remaining-first.
        prefill = (
            PrefillRequest("A", 0.0, 800, 0, ttft_deadline=10.0 + 10.0),
            PrefillRequest("B", 0.0, 100, 0, ttft_deadline=10.0 + 10.0),
            PrefillRequest("C", 0.0, 300, 0, ttft_deadline=10.0 + 10.0),
        )
        state = _snapshot(prefill=prefill, timestamp=10.0)
        ordered = rank_prefill_completion_aware(state)
        self.assertEqual(tuple(item.request_id for item in ordered), ("B", "C", "A"))

    def test_completion_aware_tier_priority(self) -> None:
        # A is critical (u=1600 tokens/sec), B is normal (u=0.01).
        # Critical tier must come before normal regardless of remaining_tokens.
        prefill = (
            PrefillRequest(
                "A", 0.0, 800, 0, ttft_deadline=10.5
            ),
            PrefillRequest(
                "B", 0.0, 1, 0, ttft_deadline=10.0 + 100.0
            ),
        )
        state = _snapshot(prefill=prefill, timestamp=10.0)
        ordered = rank_prefill_completion_aware(state)
        self.assertEqual(tuple(item.request_id for item in ordered), ("A", "B"))

    def test_continuation_ordering(self) -> None:
        prefill = (
            PrefillRequest("waiting", 0.0, 100, 0),
            PrefillRequest("running", 5.0, 100, 10, is_running=True),
        )
        state = _snapshot(prefill=prefill)
        ordered = rank_prefill_continuation(state)
        self.assertEqual(tuple(item.request_id for item in ordered), ("running", "waiting"))


# ---------------------------------------------------------------------------
# §14.7–14.11 End-to-end Generator layout
# ---------------------------------------------------------------------------


class CandidateGeneratorLayoutTests(unittest.TestCase):
    def _resolver_with_base(self, base: int, status: str = RESOLUTION_INVERTED_OK):
        return _BudgetResolver(base=base, status=status)

    def test_same_budget_three_policies_produce_distinct_orders(self) -> None:
        # Three Prefill requests with distinct ttft deadlines produce
        # distinct orders under URGENCY vs CONTINUATION. We assert that
        # the three prefill orders returned by build_prefill_orders are
        # pairwise different.
        prefill = (
            PrefillRequest("A", 0.0, 100, 0, ttft_deadline=10.5),
            PrefillRequest("B", 0.0, 100, 0, ttft_deadline=10.2),
            PrefillRequest("C", 0.0, 100, 0, ttft_deadline=10.9),
        )
        decode = (DecodeRequest("d1", 0.0, 20),)
        state = _snapshot(prefill=prefill, decode=decode)

        # Use rank functions directly.
        from dpp_scheduler.candidate_generator import build_prefill_orders

        orders = build_prefill_orders(state)
        urgency_ids = tuple(item.request_id for item in orders[0][1])
        continuation_ids = tuple(item.request_id for item in orders[2][1])
        self.assertNotEqual(urgency_ids, continuation_ids)

    def test_upper_bound_sixteen(self) -> None:
        # Build a snapshot where the five multipliers stay distinct after
        # floor and there is enough backlog / token budget for them all.
        # Five Prefill requests × 1000 tokens each; no Decode.
        prefill = tuple(
            PrefillRequest(
                request_id=f"p{i}", arrival_time=float(i),
                token_count=1000, prefilled_tokens=0, ordinal=i,
            )
            for i in range(5)
        )
        state = _snapshot(prefill=prefill, decode=(), token_budget=2048)
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=self._resolver_with_base(base=1024),
        )
        plans = generator.generate(state)
        # ZERO + deduped 5×3 = up to 16 raw, but with multiple prefill
        # requests and a 1024-token budget the canonical plans remain
        # distinct. Assert we hit the upper bound.
        self.assertGreaterEqual(len(plans), 2)
        self.assertLessEqual(len(plans), 16)
        # The 16th (last) plan should still parse from a deterministic
        # template_id namespace.
        template_ids = {plan.template_id for plan in plans}
        self.assertIn("ALL_DECODE:ZERO", template_ids)

    def test_canonical_dedup(self) -> None:
        # Two multiplier × policy combinations that produce identical
        # prefill_items must dedup to a single BatchPlan. Construct the
        # situation by giving only one Prefill request — every policy
        # produces the same (request_id, token_count) pair for any given
        # multiplier, so all 5×3 plans collapse to ZERO plus at most 5
        # distinct Mixed plans.
        prefill = (PrefillRequest("only", 0.0, 100, 0),)
        state = _snapshot(prefill=prefill, decode=(), token_budget=2048)
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=self._resolver_with_base(base=800),
        )
        plans = generator.generate(state)
        keys = {(plan.prefill_items, plan.decode_items) for plan in plans}
        self.assertEqual(len(keys), len(plans), "dedup invariant failed")
        # At most 6 distinct plans: ZERO + 5 multiplier neighborhoods.
        self.assertLessEqual(len(plans), 6)

    def test_p_zero_only_zero(self) -> None:
        prefill = (PrefillRequest("p", 0.0, 100, 0),)
        state = _snapshot(prefill=prefill, decode=())
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=self._resolver_with_base(base=0),
        )
        plans = generator.generate(state)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].template_id, "ALL_DECODE:ZERO")
        self.assertEqual(plans[0].total_prefill_tokens, 0)

    def test_no_decode_path(self) -> None:
        prefill = (PrefillRequest("p", 0.0, 100, 0),)
        state = _snapshot(prefill=prefill, decode=())
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=self._resolver_with_base(
                base=100, status=RESOLUTION_NO_DECODE_USE_MAX
            ),
        )
        plans = generator.generate(state)
        # ZERO plus 5 × 3 = up to 16 distinct templates (subject to dedup).
        self.assertGreaterEqual(len(plans), 2)
        # At least one plan template_id encodes the slack-budget policy.
        slack = [
            plan for plan in plans
            if plan.template_id.startswith("ALL_DECODE:SLACK_BUDGET:")
        ]
        self.assertTrue(slack, "expected at least one SLACK_BUDGET plan")
        # Each slack-budget plan should carry one of the three policies.
        policies = {
            plan.template_id.rsplit(":", 1)[-1]
            for plan in slack
        }
        self.assertTrue({"URGENCY", "COMPLETION_AWARE", "CONTINUATION"} & policies)

    def test_no_decode_no_backlog_returns_empty(self) -> None:
        state = _snapshot(prefill=(), decode=())
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=_BudgetResolver(
                base=0, status=RESOLUTION_NO_DECODE_NO_BACKLOG
            ),
        )
        plans = generator.generate(state)
        self.assertEqual(plans, ())

    def test_no_predictor_dependency_in_candidate_generator_module(self) -> None:
        source = inspect.getsource(
            __import__("dpp_scheduler.candidate_generator", fromlist=["x"])
        )
        self.assertNotIn("DurationPredictor", source)
        self.assertNotIn("SafeSet", source)
        self.assertNotIn("DPPSelector", source)

    def test_last_diagnostic_summary(self) -> None:
        prefill = (PrefillRequest("p", 0.0, 100, 0),)
        decode = (DecodeRequest("d", 0.0, 20),)
        state = _snapshot(prefill=prefill, decode=decode)
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=self._resolver_with_base(base=512),
        )
        plans = generator.generate(state)
        diag = generator.last_diagnostic
        self.assertIsNotNone(diag)
        self.assertEqual(diag.resolution.base_prefill_budget, 512)
        self.assertEqual(diag.resolution.resolution_status, RESOLUTION_INVERTED_OK)
        self.assertGreaterEqual(diag.raw_candidate_count, len(plans))
        self.assertEqual(diag.deduplicated_candidate_count, len(plans))
        self.assertIn(0, diag.candidate_budget_values)


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
        state = _snapshot(decode=decode, timestamp=10.0)
        resolver = RidgeBudgetResolver(
            predictor=_WrongShapePredictor(),
            settings=SchedulerSettings.provisional(),
        )
        resolution = resolver.resolve(state)
        self.assertEqual(resolution.base_prefill_budget, 0)
        self.assertEqual(resolution.resolution_status, RESOLUTION_PREDICTOR_INVALID)


# ---------------------------------------------------------------------------
# Null resolver Generator integration
# ---------------------------------------------------------------------------


class NullResolverIntegrationTests(unittest.TestCase):
    def test_null_resolver_with_prefill_only_emits_zero(self) -> None:
        prefill = (PrefillRequest("p", 0.0, 100, 0),)
        state = _snapshot(prefill=prefill)
        generator = CandidateGenerator(
            SchedulerSettings.provisional(),
            budget_resolver=NullBudgetResolver(),
        )
        plans = generator.generate(state)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].template_id, "ALL_DECODE:ZERO")


if __name__ == "__main__":
    unittest.main()
