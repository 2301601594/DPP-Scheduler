from __future__ import annotations

import unittest

from dpp_scheduler.contracts import DecodeRequest, PrefillRequest, StateSnapshot
from dpp_scheduler.targeted_profile import (
    ISOLATED_KNEE_TARGET_COUNT,
    ISOLATED_PARTIAL_SETUP_TOKENS,
    TargetRecipe,
    build_isolated_setup_plan,
    build_isolated_target_plan,
    build_target_recipes,
    isolated_client_request_ids,
    isolated_prefill_allocations,
    resolve_isolated_request_ids,
)


def _snapshot(
    prefills: tuple[PrefillRequest, ...],
    decodes: tuple[DecodeRequest, ...],
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=1,
        timestamp=1.0,
        waiting_prefill_requests=prefills,
        active_decode_requests=decodes,
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=30000,
        kv_block_size=16,
        token_budget=2048,
        sequence_budget=64,
        total_kv_blocks=30149,
        provenance="test",
    )


class IsolatedRecipeTests(unittest.TestCase):
    def test_completion_transport_ids_map_to_engine_ids(self) -> None:
        recipe = TargetRecipe(
            "cell", "mixed", 256, 1, 1, "fresh", "balanced", "measured"
        )
        client_prefills, client_decodes = isolated_client_request_ids(recipe)
        engine_prefills, engine_decodes = resolve_isolated_request_ids(
            {
                f"cmpl-{client_prefills[0]}-0-0123abcd",
                f"cmpl-{client_decodes[0]}-0-89abcdef",
            },
            recipe,
        )
        self.assertEqual(
            engine_prefills,
            tuple(f"cmpl-{request_id}-0-0123abcd" for request_id in client_prefills),
        )
        self.assertEqual(
            engine_decodes,
            tuple(f"cmpl-{request_id}-0-89abcdef" for request_id in client_decodes),
        )

    def test_request_id_binding_rejects_unknown_and_incomplete_sets(self) -> None:
        recipe = TargetRecipe(
            "cell", "mixed", 256, 1, 1, "fresh", "balanced", "measured"
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected isolated request ID"):
            resolve_isolated_request_ids({"cmpl-other-0-0123abcd"}, recipe)
        with self.assertRaisesRegex(RuntimeError, "admission is incomplete"):
            resolve_isolated_request_ids(
                {"cmpl-cell-p00-0-0123abcd"}, recipe
            )

    def test_full_matrix_is_exact_and_sequence_feasible(self) -> None:
        recipes = build_target_recipes(3001, mode="isolated_knee")
        self.assertEqual(len(recipes), ISOLATED_KNEE_TARGET_COUNT)
        self.assertEqual(len(recipes), 2160)
        self.assertEqual(sum(r.decode_request_cap == 0 for r in recipes), 480)
        self.assertEqual(sum(r.decode_request_cap > 0 for r in recipes), 1680)
        self.assertEqual(
            sum(r.prefill_request_cap + r.decode_request_cap for r in recipes),
            53040,
        )
        self.assertEqual(
            {r.decode_request_cap for r in recipes}, {0, 8, 16, 32, 48}
        )
        self.assertLessEqual(
            max(r.decode_request_cap + r.prefill_request_cap for r in recipes),
            64,
        )
        self.assertLessEqual(
            max(r.prefill_token_cap + r.decode_request_cap for r in recipes),
            2048,
        )

    def test_balanced_and_skewed_allocations_are_exact(self) -> None:
        balanced = TargetRecipe(
            "b", "prefill_only", 256, 4, 0, "fresh", "balanced", "none"
        )
        skewed = TargetRecipe(
            "s", "prefill_only", 256, 4, 0, "fresh", "skewed", "none"
        )
        self.assertEqual(isolated_prefill_allocations(balanced), (64, 64, 64, 64))
        allocations = isolated_prefill_allocations(skewed)
        self.assertEqual(sum(allocations), 256)
        self.assertEqual(len(allocations), 4)
        self.assertGreater(allocations[0], sum(allocations[1:]))

    def test_setup_prepares_only_partial_prefill_and_decode_inputs(self) -> None:
        recipe = TargetRecipe(
            "cell", "mixed", 256, 4, 8, "partial", "balanced", "measured"
        )
        client_prefills, client_decodes = isolated_client_request_ids(recipe)
        prefill_ids, decode_ids = resolve_isolated_request_ids(
            {
                *(f"cmpl-{item}-0-{index:08x}" for index, item in enumerate(client_prefills)),
                *(f"cmpl-{item}-0-{index + 16:08x}" for index, item in enumerate(client_decodes)),
            },
            recipe,
        )
        prefills = tuple(
            PrefillRequest(
                request_id=request_id,
                arrival_time=0.0,
                token_count=512,
                prefilled_tokens=0,
                is_running=False,
                ordinal=index,
            )
            for index, request_id in enumerate(prefill_ids + decode_ids)
        )
        plan = build_isolated_setup_plan(_snapshot(prefills, ()), recipe)
        assert plan is not None
        setup = dict(plan.prefill_items)
        self.assertTrue(
            all(setup[request_id] == ISOLATED_PARTIAL_SETUP_TOKENS for request_id in prefill_ids)
        )
        self.assertTrue(all(request_id in setup for request_id in decode_ids))
        self.assertEqual(plan.total_decode_tokens, 0)

    def test_target_plan_requires_exact_state_and_shape(self) -> None:
        recipe = TargetRecipe(
            "cell", "mixed", 384, 4, 8, "partial", "skewed", "measured"
        )
        client_prefills, client_decodes = isolated_client_request_ids(recipe)
        prefill_ids, decode_ids = resolve_isolated_request_ids(
            {
                *(f"cmpl-{item}-0-{index:08x}" for index, item in enumerate(client_prefills)),
                *(f"cmpl-{item}-0-{index + 16:08x}" for index, item in enumerate(client_decodes)),
            },
            recipe,
        )
        prefills = tuple(
            PrefillRequest(
                request_id=request_id,
                arrival_time=0.0,
                token_count=1024,
                prefilled_tokens=ISOLATED_PARTIAL_SETUP_TOKENS,
                is_running=True,
                ordinal=index,
            )
            for index, request_id in enumerate(prefill_ids)
        )
        decodes = tuple(
            DecodeRequest(
                request_id=request_id,
                arrival_time=0.0,
                kv_context_length=256,
                ordinal=index,
            )
            for index, request_id in enumerate(decode_ids)
        )
        plan, realized = build_isolated_target_plan(
            _snapshot(prefills, decodes), recipe
        )
        self.assertEqual(plan.total_prefill_tokens, 384)
        self.assertEqual(plan.total_decode_tokens, 8)
        self.assertEqual(realized["partial_prefill_requests"], 4)
        broken = prefills[:-1]
        with self.assertRaisesRegex(RuntimeError, "admission is incomplete"):
            build_isolated_target_plan(_snapshot(broken, decodes), recipe)


if __name__ == "__main__":
    unittest.main()
