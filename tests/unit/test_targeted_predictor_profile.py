from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from benchmarks.run_targeted_predictor_profile import _prepare_controlled_rows
from benchmarks.run_isolated_candidate_profile import (
    _settle_isolated_request_tasks,
)
from benchmarks.targeted_predictor_profile import (
    validate_reused_stock_trace,
    validate_target_rows,
)
from dpp_scheduler.contracts import DecodeRequest, PrefillRequest, StateSnapshot
from dpp_scheduler.targeted_profile import (
    ISOLATED_PARTIAL_SETUP_TOKENS,
    TARGET_CAMPAIGN_MATRIX,
    TIME_TO_BUDGET_BUDGETS,
    TargetRecipe,
    build_isolated_setup_plan,
    build_isolated_target_plan,
    build_target_plan,
    build_target_recipes,
    isolated_prefill_allocations,
    isolated_prefill_backlog_allocations,
)


def make_snapshot(
    *,
    prefill: tuple[PrefillRequest, ...] = (),
    decode: tuple[DecodeRequest, ...] = (),
    token_budget: int = 2048,
    sequence_budget: int = 64,
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=1,
        timestamp=100.0,
        waiting_prefill_requests=prefill,
        active_decode_requests=decode,
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=9000,
        kv_block_size=16,
        token_budget=token_budget,
        sequence_budget=sequence_budget,
        total_kv_blocks=10000,
        provenance="target-test",
    )


class TargetRecipeTests(unittest.TestCase):
    def test_isolated_runner_cancels_client_after_audited_cleanup(self) -> None:
        async def exercise() -> tuple[list[dict[str, object]], bool]:
            async def pending_request() -> dict[str, object]:
                await asyncio.Future()
                raise AssertionError("unreachable")

            task = asyncio.create_task(pending_request())
            await asyncio.sleep(0)
            results = await _settle_isolated_request_tasks(
                [task],
                [{"request_id": "request-0"}],
                event={"event": "batch_complete"},
            )
            return results, task.cancelled()

        results, cancelled = asyncio.run(exercise())
        self.assertTrue(cancelled)
        self.assertEqual(
            results,
            [
                {
                    "request_id": "request-0",
                    "completed": False,
                    "error": None,
                    "isolated_client_outcome": (
                        "cancelled_after_scheduler_cleanup"
                    ),
                    "isolated_terminal_event": "batch_complete",
                }
            ],
        )

    def test_knee_trace_reuse_allows_only_stale_whole_config_hash(self) -> None:
        runtime = SimpleNamespace(
            model_revision="model-revision",
            tokenizer_revision="tokenizer-revision",
            client_safety_ceiling_tokens=2048,
            ignore_eos=False,
            temperature=0.0,
            top_p=1.0,
            seed_source="per_request_trace",
            config_sha256="current-config",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "qps_0.2_seed_1001.jsonl"
            trace.write_text("{}\n", encoding="utf-8")
            manifest = {
                "kind": "qwen3_14b_poisson_length_blind_traces",
                "model_revision": runtime.model_revision,
                "tokenizer_revision": runtime.tokenizer_revision,
                "client_safety_ceiling_tokens": 2048,
                "client_safety_ceiling_role": (
                    "termination_guard_only_never_scheduler_input"
                ),
                "predetermined_output_length": False,
                "ignore_eos": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed_source": "per_request_trace",
                "num_requests_per_trace": 500,
                "config_sha256": "old-config",
                "files": [
                    {
                        "file": trace.name,
                        "requested_qps": 0.2,
                        "seed": 1001,
                        "num_requests": 500,
                        "sha256": sha256(trace.read_bytes()).hexdigest(),
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = validate_reused_stock_trace(
                trace,
                manifest_path,
                runtime,
                source_qps=0.2,
                source_seed=1001,
                expected_request_count=500,
            )
            self.assertTrue(result["config_hash_mismatch_allowed"])
            manifest["model_revision"] = "wrong-model"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_revision mismatch"):
                validate_reused_stock_trace(
                    trace,
                    manifest_path,
                    runtime,
                    source_qps=0.2,
                    source_seed=1001,
                    expected_request_count=500,
                )

    def test_formal_matrix_has_two_runs_and_expected_target_counts(self) -> None:
        self.assertEqual(len(TARGET_CAMPAIGN_MATRIX), 2)
        recipes = build_target_recipes(2001)
        self.assertEqual(len(recipes), 456)
        self.assertEqual(
            sum(recipe.batch_kind == "prefill_only" for recipe in recipes), 240
        )
        self.assertEqual(sum(recipe.batch_kind == "mixed" for recipe in recipes), 216)
        self.assertEqual(recipes, build_target_recipes(2001))
        self.assertNotEqual(recipes, build_target_recipes(2002))

    def test_prefill_only_plan_excludes_decode_and_hits_token_cap(self) -> None:
        snapshot = make_snapshot(
            prefill=tuple(
                PrefillRequest(f"p{i}", float(i), 512, 0, ordinal=i)
                for i in range(4)
            ),
            decode=(DecodeRequest("d0", 0.0, 300),),
        )
        recipe = TargetRecipe(
            "p-test", "prefill_only", 256, 4, 0, "fresh", "balanced", "none"
        )
        built = build_target_plan(snapshot, recipe)
        self.assertIsNotNone(built)
        assert built is not None
        plan, realized = built
        self.assertEqual(plan.decode_items, ())
        self.assertEqual(plan.total_prefill_tokens, 256)
        self.assertEqual(len(plan.prefill_items), 4)
        self.assertEqual(realized["batch_kind"], "prefill_only")

    def test_mixed_plan_uses_real_decode_context_and_one_token_each(self) -> None:
        snapshot = make_snapshot(
            prefill=(PrefillRequest("p0", 0.0, 1024, 128, is_running=True),),
            decode=tuple(
                DecodeRequest(f"d{i}", float(i), 100 + i * 100, ordinal=i)
                for i in range(3)
            ),
        )
        recipe = TargetRecipe(
            "m-test", "mixed", 64, 1, 2, "partial", "balanced", "long"
        )
        built = build_target_plan(snapshot, recipe)
        self.assertIsNotNone(built)
        assert built is not None
        plan, realized = built
        self.assertEqual(plan.prefill_items, (("p0", 64),))
        self.assertEqual(plan.decode_items, ("d2", "d1"))
        self.assertEqual(plan.total_decode_tokens, 2)
        self.assertEqual(realized["partial_prefill_requests"], 1)

    def test_ood_recipes_cover_all_batch_kinds_deterministically(self) -> None:
        recipes = build_target_recipes(7001, mode="ood")
        self.assertEqual(len(recipes), 128)
        self.assertEqual(
            {recipe.batch_kind for recipe in recipes},
            {"prefill_only", "decode_only", "mixed"},
        )
        self.assertTrue(all(recipe.strict_shape for recipe in recipes))
        self.assertEqual(recipes, build_target_recipes(7001, mode="ood"))
        self.assertNotEqual(recipes, build_target_recipes(8001, mode="ood"))

    def test_time_to_budget_matrix_keeps_backlog_when_budget_is_zero(self) -> None:
        recipes = build_target_recipes(6001, mode="time_to_budget_validation")
        self.assertEqual(len(recipes), 118)
        self.assertEqual(
            {recipe.prefill_token_cap for recipe in recipes},
            set(TIME_TO_BUDGET_BUDGETS),
        )
        zero = next(recipe for recipe in recipes if recipe.prefill_token_cap == 0)
        self.assertGreater(zero.prefill_request_cap, 0)
        self.assertGreater(zero.decode_request_cap, 0)
        self.assertEqual(
            isolated_prefill_allocations(zero),
            (0,) * zero.prefill_request_cap,
        )
        self.assertEqual(sum(isolated_prefill_backlog_allocations(zero)), 1024)

        prefills = tuple(
            PrefillRequest(
                f"cmpl-{zero.recipe_id}-p{index:02d}-0",
                float(index),
                1024,
                0,
                ordinal=index,
            )
            for index in range(zero.prefill_request_cap)
        )
        decodes = tuple(
            DecodeRequest(
                f"cmpl-{zero.recipe_id}-d{index:02d}-0",
                float(index),
                256,
                ordinal=index,
            )
            for index in range(zero.decode_request_cap)
        )
        plan, realized = build_isolated_target_plan(
            make_snapshot(prefill=prefills, decode=decodes), zero
        )
        self.assertEqual(plan.prefill_items, ())
        self.assertEqual(len(plan.decode_items), zero.decode_request_cap)
        self.assertEqual(realized["batch_kind"], "decode_only")

    def test_time_to_budget_smoke_reproduces_multi_request_admission(self) -> None:
        recipes = build_target_recipes(
            6001,
            mode="time_to_budget_validation_smoke",
        )
        self.assertEqual(len(recipes), 1)
        recipe = recipes[0]
        self.assertEqual(recipe.prefill_request_cap, 4)
        self.assertEqual(recipe.prefill_token_cap, 64)
        self.assertEqual(recipe.prefill_backlog_token_cap, 1024)
        self.assertTrue(recipe.strict_shape)

    def test_isolated_setup_plan_dedups_decode_candidates_across_passes(self) -> None:
        """Regression for the ``ttb-r0-f03-b0000`` EngineCore crash.

        A decode candidate whose prompt exceeds the first-pass ``share`` must
        appear in ``prefill_items`` exactly once with the cumulative grant,
        not once per pass. The Adapter validator rejects duplicate Prefill
        IDs and vLLM would also over-allocate KV for the same request.
        """
        snapshot = make_snapshot(
            prefill=(
                PrefillRequest(
                    "cmpl-ttb-r0-f03-b0000-p00-0-aaaaaaaa",
                    0.0,
                    1073,
                    0,
                    ordinal=0,
                ),
            )
            + tuple(
                PrefillRequest(
                    (
                        "cmpl-ttb-r0-f03-b0000-d00-0-bbbbbbbb",
                        "cmpl-ttb-r0-f03-b0000-d01-0-cccccccc",
                        "cmpl-ttb-r0-f03-b0000-d02-0-dddddddd",
                        "cmpl-ttb-r0-f03-b0000-d03-0-eeeeeeee",
                        "cmpl-ttb-r0-f03-b0000-d04-0-ffffffff",
                    )[index],
                    0.0,
                    prompt_tokens,
                    0,
                    ordinal=index + 1,
                )
                for index, prompt_tokens in enumerate((230, 230, 229, 539, 228))
            ),
        )
        recipe = TargetRecipe(
            "ttb-r0-f03-b0000",
            "decode_only",
            0,
            1,
            5,
            "partial",
            "balanced",
            "measured",
            0,
            True,
            1024,
            3,
        )
        plan = build_isolated_setup_plan(snapshot, recipe)
        self.assertIsNotNone(plan)
        assert plan is not None

        items = plan.prefill_items
        request_ids = [request_id for request_id, _ in items]
        self.assertEqual(
            len(set(request_ids)),
            len(request_ids),
            f"duplicate prefill request id in setup plan: {items!r}",
        )

        grants = dict(items)
        self.assertEqual(
            grants["cmpl-ttb-r0-f03-b0000-p00-0-aaaaaaaa"],
            ISOLATED_PARTIAL_SETUP_TOKENS,
        )
        self.assertEqual(
            grants["cmpl-ttb-r0-f03-b0000-d03-0-eeeeeeee"],
            539,
            "d03 (539-token prompt) must be fully prefilled in one entry",
        )
        self.assertEqual(
            plan.total_prefill_tokens,
            ISOLATED_PARTIAL_SETUP_TOKENS + 230 + 230 + 229 + 539 + 228,
        )

    def test_decode_only_target_uses_only_running_decode(self) -> None:
        snapshot = make_snapshot(
            prefill=(PrefillRequest("p0", 0.0, 128, 0),),
            decode=tuple(
                DecodeRequest(f"d{i}", float(i), 100 + i, ordinal=i)
                for i in range(3)
            ),
        )
        recipe = TargetRecipe(
            "d-test",
            "decode_only",
            0,
            0,
            2,
            "none",
            "balanced",
            "short",
            strict_shape=True,
        )
        built = build_target_plan(snapshot, recipe)
        self.assertIsNotNone(built)
        assert built is not None
        plan, realized = built
        self.assertEqual(plan.prefill_items, ())
        self.assertEqual(plan.decode_items, ("d0", "d1"))
        self.assertEqual(realized["batch_kind"], "decode_only")

    def test_full_sequence_budget_rejects_new_prefill(self) -> None:
        snapshot = make_snapshot(
            prefill=(PrefillRequest("new", 0.0, 128, 0),),
            decode=tuple(
                DecodeRequest(f"d{i}", float(i), 100, ordinal=i)
                for i in range(64)
            ),
        )
        recipe = TargetRecipe(
            "p-blocked", "prefill_only", 16, 1, 0, "fresh", "balanced", "none"
        )
        self.assertIsNone(build_target_plan(snapshot, recipe))

    def test_controlled_rows_replace_arrivals_and_request_ids_only(self) -> None:
        rows = [
            {
                "request_id": f"old-{index}",
                "arrival_time_s": float(index + 10),
                "prompt": f"prompt-{index}",
            }
            for index in range(3)
        ]
        controlled = _prepare_controlled_rows(
            rows,
            request_count=2,
            recipe_seed=9001,
            dispatch_interval_seconds=0.002,
        )
        self.assertEqual(
            [row["request_id"] for row in controlled],
            ["target_s9001_0000", "target_s9001_0001"],
        )
        self.assertEqual(
            [row["arrival_time_s"] for row in controlled], [0.0, 0.002]
        )
        self.assertEqual(controlled[1]["prompt"], "prompt-1")

    def test_target_validator_requires_every_smoke_recipe_in_order(self) -> None:
        recipes = build_target_recipes(9001, mode="smoke")
        rows = []
        for index, recipe in enumerate(recipes):
            selected = [
                {
                    "request_id": f"p-{index}",
                    "phase": "prefill",
                    "current_context_tokens": 0,
                    "scheduled_tokens": recipe.prefill_token_cap,
                }
            ]
            if recipe.batch_kind == "mixed":
                selected.append(
                    {
                        "request_id": f"d-{index}",
                        "phase": "decode",
                        "current_context_tokens": 100,
                        "scheduled_tokens": 1,
                    }
                )
            rows.append(
                {
                    "schema_version": 1,
                    "run_id": "smoke-run",
                    "iteration_index": index,
                    "plan_id": f"target-{index}",
                    "snapshot_hash": f"{index:064x}",
                    "sample_role": "target",
                    "recipe_id": recipe.recipe_id,
                    "recipe_seed": 9001,
                    "recipe_mode": "smoke",
                    "requested_shape": recipe.as_dict(),
                    "realized_shape": {
                        "batch_kind": recipe.batch_kind,
                        "prefill_requests": 1,
                        "prefill_tokens": recipe.prefill_token_cap,
                        "fresh_prefill_requests": 1,
                        "partial_prefill_requests": 0,
                        "decode_requests": int(recipe.batch_kind == "mixed"),
                    },
                    "selected_requests": selected,
                    "actual_duration_seconds": 0.1,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iteration_profile.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            validation = validate_target_rows(
                path,
                expected_run_id="smoke-run",
                recipe_seed=9001,
                recipe_mode="smoke",
            )
        self.assertEqual(validation["target_count"], 4)
        self.assertEqual(
            validation["target_batch_kind_counts"],
            {"mixed": 2, "prefill_only": 2},
        )


if __name__ == "__main__":
    unittest.main()
