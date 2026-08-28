from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.build_online_ridge_predictor import evaluate_window, select_window
from benchmarks.predictor_online_evaluation import (
    _effectiveness_result,
    load_evaluation_rows,
)
from dpp_scheduler.predictor import ONLINE_PREDICTOR_VERSION
from dpp_scheduler.targeted_profile import build_target_recipes
from dpp_scheduler.vllm_adapter import (
    VLLM_ALIGNED_ITERATION_TIMING,
    VLLM_OFFICIAL_ITERATION_TIMING,
    _build_iteration_timing_bridge,
    _classify_isolated_scheduler_update,
    build_shadow_plan,
)
from tests.unit.test_online_predictor import _snapshot


class OnlinePredictorEvaluationTests(unittest.TestCase):
    def test_isolated_update_accepts_only_registered_empty_admission_wait(self) -> None:
        output = SimpleNamespace(
            total_num_scheduled_tokens=0,
            num_scheduled_tokens={},
        )
        self.assertEqual(
            _classify_isolated_scheduler_update(None, output, output),
            "admission_wait",
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _classify_isolated_scheduler_update(None, None, output)

    def test_isolated_update_rejects_nonempty_admission_wait(self) -> None:
        output = SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"request": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "not empty"):
            _classify_isolated_scheduler_update(None, output, output)

    def test_isolated_update_matches_exact_pending_plan(self) -> None:
        output = SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"request": 1},
        )
        pending = {"scheduler_output": output}
        self.assertEqual(
            _classify_isolated_scheduler_update(pending, None, output),
            "planned",
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _classify_isolated_scheduler_update(
                pending,
                None,
                SimpleNamespace(
                    total_num_scheduled_tokens=1,
                    num_scheduled_tokens={"request": 1},
                ),
            )

    def test_iteration_bridge_prefers_exact_official_duration(self) -> None:
        callbacks = []

        @contextmanager
        def official_capture(_core, _scheduler_output):
            details = SimpleNamespace(iteration_index=7, elapsed_ms=0.0)
            yield details
            details.elapsed_ms = 12.5

        scheduler = SimpleNamespace(
            _dpp_record_iteration_duration=lambda **kwargs: callbacks.append(kwargs)
        )
        core = SimpleNamespace(scheduler=scheduler)
        output = SimpleNamespace(total_num_scheduled_tokens=1)
        bridge = _build_iteration_timing_bridge(official_capture)
        with bridge(core, output):
            pass
        self.assertEqual(len(callbacks), 1)
        self.assertIs(callbacks[0]["scheduler_output"], output)
        self.assertEqual(callbacks[0]["iteration_index"], 7)
        self.assertEqual(callbacks[0]["duration_seconds"], 0.0125)
        self.assertEqual(
            callbacks[0]["timing_source"], VLLM_OFFICIAL_ITERATION_TIMING
        )

    def test_iteration_bridge_uses_same_boundary_without_detail_logging(self) -> None:
        callbacks = []

        @contextmanager
        def aligned_capture(_core, _scheduler_output):
            yield None

        scheduler = SimpleNamespace(
            _dpp_record_iteration_duration=lambda **kwargs: callbacks.append(kwargs)
        )
        core = SimpleNamespace(scheduler=scheduler)
        output = SimpleNamespace(total_num_scheduled_tokens=1)
        with _build_iteration_timing_bridge(aligned_capture)(core, output):
            pass
        self.assertEqual(callbacks[0]["iteration_index"], None)
        self.assertGreater(callbacks[0]["duration_seconds"], 0.0)
        self.assertEqual(
            callbacks[0]["timing_source"], VLLM_ALIGNED_ITERATION_TIMING
        )

    def test_iteration_bridge_skips_zero_token_timing_without_opt_in(self) -> None:
        callbacks = []

        @contextmanager
        def aligned_capture(_core, _scheduler_output):
            yield None

        scheduler = SimpleNamespace(
            _dpp_record_iteration_duration=lambda **kwargs: callbacks.append(kwargs)
        )
        core = SimpleNamespace(scheduler=scheduler)
        output = SimpleNamespace(total_num_scheduled_tokens=0)
        with _build_iteration_timing_bridge(aligned_capture)(core, output):
            pass
        self.assertEqual(callbacks, [])

    def test_iteration_bridge_captures_opted_in_zero_token_cleanup(self) -> None:
        callbacks = []

        @contextmanager
        def aligned_capture(_core, _scheduler_output):
            yield None

        scheduler = SimpleNamespace(
            _dpp_capture_zero_token_iteration_timing=True,
            _dpp_record_iteration_duration=lambda **kwargs: callbacks.append(kwargs),
        )
        core = SimpleNamespace(scheduler=scheduler)
        output = SimpleNamespace(total_num_scheduled_tokens=0)
        with _build_iteration_timing_bridge(aligned_capture)(core, output):
            pass
        self.assertEqual(len(callbacks), 1)
        self.assertIs(callbacks[0]["scheduler_output"], output)
        self.assertIsNone(callbacks[0]["iteration_index"])
        self.assertGreater(callbacks[0]["duration_seconds"], 0.0)
        self.assertEqual(
            callbacks[0]["timing_source"], VLLM_ALIGNED_ITERATION_TIMING
        )

    def test_diagnostic_aggregate_bucket_maps_candidate_and_zero_plan_templates(
        self,
    ) -> None:
        from dpp_scheduler.vllm_adapter import get_modular_scheduler_class

        scheduler_cls = get_modular_scheduler_class()
        cases = {
            "ZERO": "ZERO",
            "ZERO:IDLE_EMPTY_QUEUE": "ZERO",
            "ZERO:NO_SAFE_DECISION": "ZERO",
            "P10": "P10",
            "P50": "P50",
            "P100": "P100",
            "STOCK": "STOCK",
            "ALL_DECODE:ZERO": "OTHER",
            "FALLBACK_DECODE_ONLY": "OTHER",
        }
        for template_id, expected in cases.items():
            self.assertEqual(
                scheduler_cls._dpp_aggregate_bucket(template_id), expected
            )

    def test_timing_incompatibility_suppresses_effectiveness_conclusion(self) -> None:
        result = _effectiveness_result(
            {
                "mae_improved": True,
                "absolute_bias_not_worse": True,
                "conservative_coverage_at_least_0p95": True,
            },
            timing_compatible=False,
        )
        self.assertTrue(result["candidate_effective_without_timing_guard"])
        self.assertFalse(result["conclusion_available"])
        self.assertIsNone(result["effective"])
        self.assertEqual(result["unavailable_reason"], "timing_incompatible")

    def test_window_replay_uses_only_prior_rows_and_selection_rule(self) -> None:
        rows = [
            {"run_id": "run", "iteration_index": index} for index in range(40)
        ]
        actual = np.asarray([0.11] * 40)
        base = np.asarray([0.10] * 40)
        residuals = actual - base
        result = evaluate_window(
            rows=rows,
            actual=actual,
            base_prediction=base,
            residuals=residuals,
            window_size=32,
        )
        self.assertEqual(result["evaluated_rows"], 8)
        self.assertAlmostEqual(result["expected_mae_seconds"], 0.0)
        selection = select_window(
            [
                {"window_size": 32, "expected_mae_seconds": 0.02, "conservative_coverage": 0.96},
                {"window_size": 64, "expected_mae_seconds": 0.01, "conservative_coverage": 0.96},
                {"window_size": 128, "expected_mae_seconds": 0.001, "conservative_coverage": 0.94},
            ]
        )
        self.assertEqual(selection["selected"]["window_size"], 64)

    def test_stock_output_is_bound_to_snapshot_without_reselection(self) -> None:
        snapshot = _snapshot()
        output = SimpleNamespace(
            total_num_scheduled_tokens=5,
            num_scheduled_tokens={"p1": 4, "d1": 1},
        )
        plan = build_shadow_plan(snapshot=snapshot, scheduler_output=output)
        assert plan is not None
        self.assertEqual(plan.prefill_items, (("p1", 4),))
        self.assertEqual(plan.decode_items, ("d1",))
        self.assertEqual(plan.total_prefill_tokens, 4)
        self.assertEqual(plan.total_decode_tokens, 1)

    def test_telemetry_rejects_forbidden_output_length_state(self) -> None:
        recipes = build_target_recipes(4000, mode="smoke")
        rows = []
        for index, recipe in enumerate(recipes):
            row = {
                "schema_version": 2,
                "run_id": "run",
                "iteration_index": index,
                "frame_id": index + 1,
                "snapshot_hash": f"{index + 1:064x}",
                "plan_id": f"target-{index}",
                "sample_role": "target",
                "batch_kind": recipe.batch_kind,
                "in_support": True,
                "base_duration_seconds": 0.1,
                "expected_duration_seconds": 0.1,
                "conservative_duration_seconds": 0.12,
                "actual_duration_seconds": 0.11,
                "timing_source": VLLM_OFFICIAL_ITERATION_TIMING,
                "timing_boundary": (
                    "after_execute_model_submission_through_model_result_and_sampling"
                ),
                "residual_seconds": 0.01,
                "calibration_source": "offline_oof_cold_start",
                "calibration_samples_before": 0,
                "calibration_samples_after": 1,
                "calibration_updated": True,
                "predictor_cpu_seconds": 0.0001,
                "rejection_reason": None,
                "predictor_version": ONLINE_PREDICTOR_VERSION,
                "selected_requests": [
                    {
                        "request_id": f"request-{index}",
                        "phase": "prefill",
                        "current_context_tokens": 0,
                        "scheduled_tokens": 1,
                    },
                    *(
                        [
                            {
                                "request_id": f"decode-{index}",
                                "phase": "decode",
                                "current_context_tokens": 10,
                                "scheduled_tokens": 1,
                            }
                        ]
                        if recipe.batch_kind == "mixed"
                        else []
                    ),
                ],
                "recipe_seed": 4000,
                "recipe_mode": "smoke",
                "recipe_id": recipe.recipe_id,
                "requested_shape": recipe.as_dict(),
                "realized_shape": {"batch_kind": recipe.batch_kind},
            }
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            loaded = load_evaluation_rows(
                path, expected_run_id="run", recipe_seed=4000, recipe_mode="smoke"
            )
            self.assertEqual(len(loaded), 4)
            rows[0]["remaining_output_tokens"] = 10
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_evaluation_rows(
                    path,
                    expected_run_id="run",
                    recipe_seed=4000,
                    recipe_mode="smoke",
                )


if __name__ == "__main__":
    unittest.main()
