from __future__ import annotations

import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from benchmarks.run_targeted_predictor_profile import _prepare_controlled_rows
from benchmarks.targeted_predictor_profile import (
    validate_reused_stock_trace,
    validate_target_rows,
)
from dpp_scheduler.contracts import DecodeRequest, PrefillRequest, StateSnapshot
from dpp_scheduler.targeted_profile import (
    TARGET_CAMPAIGN_MATRIX,
    TargetRecipe,
    build_target_plan,
    build_target_recipes,
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
