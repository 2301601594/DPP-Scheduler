from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from benchmarks.freeze_candidate_parameters import (
    KNEE_CAPS,
    analyze_horizon,
    analyze_knee,
    is_horizon_development_source,
)
from benchmarks.qwen3_runtime import (
    ActiveConfigError,
    candidate_runtime_signature,
    load_active_runtime,
    load_frozen_candidate_settings,
    sha256_file,
)
from benchmarks.run_candidate_knee_profile_campaign import (
    _initial_state,
    _remaining_attempts,
)
from dpp_scheduler.settings import SchedulerSettings
from dpp_scheduler.targeted_profile import (
    KNEE_CAMPAIGN_MATRIX,
    build_target_recipes,
)


class CandidateFreezeStatisticsTests(unittest.TestCase):
    def _horizon_rows(self) -> list[dict[str, object]]:
        rows = []
        for count in (1, 4, 12, 24, 48):
            for context_group, context in enumerate((100, 1000, 2000, 4000)):
                for index in range(200):
                    rows.append(
                        {
                            "run_id": f"run-{index % 2}",
                            "seed": 1001 + 2 * (index % 2),
                            "decode_count": count,
                            "cumulative_context_tokens": context + index % 7,
                            "duration_seconds": 0.100001 + context_group * 0.001,
                        }
                    )
        return rows

    def test_horizon_buckets_bootstrap_and_rounding_are_deterministic(self) -> None:
        first = analyze_horizon(self._horizon_rows(), bootstrap_replicates=20)
        second = analyze_horizon(self._horizon_rows(), bootstrap_replicates=20)
        self.assertEqual(first, second)
        self.assertTrue(first["eligible"])
        self.assertEqual(len(first["buckets"]), 20)
        self.assertEqual(first["critical_horizon_seconds"], 0.104)
        self.assertTrue(all(item["row_count"] >= 200 for item in first["buckets"]))

    def test_horizon_sparse_bucket_fails_closed(self) -> None:
        report = analyze_horizon(self._horizon_rows()[:-1], bootstrap_replicates=5)
        self.assertFalse(report["eligible"])
        self.assertIsNone(report["critical_horizon_seconds"])
        self.assertTrue(report["failures"])

    def test_horizon_excludes_even_seed_held_out_rows(self) -> None:
        row = {"source_kind": "stock", "batch_kind": "decode_only"}
        self.assertTrue(is_horizon_development_source(row, {"seed": 1001}))
        self.assertFalse(is_horizon_development_source(row, {"seed": 1002}))
        self.assertFalse(
            is_horizon_development_source(
                {"source_kind": "targeted", "batch_kind": "decode_only"},
                {"seed": 1001},
            )
        )

    @staticmethod
    def _knee_rows(*, exact_per_cell: int = 5) -> list[dict[str, object]]:
        ratios = dict(zip(KNEE_CAPS, (0.70, 0.85, 0.90, 0.96, 1.0, 0.99, 0.98, 0.97)))
        rows = []
        for request_cap in (1, 4, 8):
            for state in ("fresh", "partial"):
                for distribution in ("balanced", "skewed"):
                    for cap in KNEE_CAPS:
                        for repeat in range(5):
                            realized = cap if repeat < exact_per_cell else cap - 1
                            efficiency = ratios[cap] * 1000.0
                            rows.append(
                                {
                                    "actual_duration_seconds": realized / efficiency,
                                    "requested_shape": {
                                        "batch_kind": "prefill_only",
                                        "prefill_token_cap": cap,
                                        "prefill_request_cap": request_cap,
                                        "prefill_state": state,
                                        "prefill_distribution": distribution,
                                    },
                                    "realized_shape": {
                                        "batch_kind": "prefill_only",
                                        "prefill_tokens": realized,
                                        "prefill_requests": request_cap,
                                        "fresh_prefill_requests": (
                                            request_cap if state == "fresh" else 0
                                        ),
                                        "partial_prefill_requests": (
                                            request_cap if state == "partial" else 0
                                        ),
                                    },
                                }
                            )
        return rows

    def test_knee_matrix_and_selection(self) -> None:
        recipes = build_target_recipes(3001, mode="knee")
        self.assertEqual(len(recipes), 480)
        self.assertEqual(len(KNEE_CAMPAIGN_MATRIX), 1)
        self.assertEqual({recipe.prefill_token_cap for recipe in recipes}, set(KNEE_CAPS))
        report = analyze_knee(self._knee_rows(), bootstrap_replicates=20)
        self.assertTrue(report["eligible"])
        self.assertEqual(report["knee_tokens"], 768)
        self.assertEqual(report["shape_count"], 12)
        self.assertEqual(len(report["cells"]), 96)

    def test_knee_requires_four_exact_realizations(self) -> None:
        eligible = analyze_knee(
            self._knee_rows(exact_per_cell=4), bootstrap_replicates=5
        )
        self.assertTrue(eligible["eligible"])
        report = analyze_knee(
            self._knee_rows(exact_per_cell=3), bootstrap_replicates=5
        )
        self.assertFalse(report["eligible"])
        self.assertIsNone(report["knee_tokens"])


class CandidateFreezeConfigurationTests(unittest.TestCase):
    @staticmethod
    def _frozen_runtime(root: Path):
        base = load_active_runtime("configs/dgx_spark_experiment.yaml")
        processed = root / "results" / "processed"
        artifact = processed / "candidate_parameter_freeze_v2"
        artifact.mkdir(parents=True)
        config = yaml.safe_load(base.config_path.read_text(encoding="utf-8"))
        runtime = replace(
            base,
            config_path=root / "config.yaml",
            workspace=root,
            processed_results=processed,
        )
        config["candidate_generator"].update(
            {
                "critical_horizon_seconds": 0.211,
                "parameters_frozen": True,
                "freeze_kind": "measured",
                "freeze_manifest_path": str(artifact / "manifest.json"),
            }
        )
        config["candidate_generator"]["prefill_breakpoints"].update(
            {"knee_tokens": 768, "knee_status": "frozen"}
        )
        runtime.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        _, signature = candidate_runtime_signature(runtime)
        manifest_path = artifact / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact_id": "candidate_parameter_freeze_v2",
                    "status": "frozen",
                    "parameters_frozen": True,
                    "critical_horizon_seconds": 0.211,
                    "prefill_knee_tokens": 768,
                    "runtime_signature_sha256": signature,
                }
            ),
            encoding="utf-8",
        )
        config["candidate_generator"].update(
            {
                "freeze_manifest_sha256": sha256_file(manifest_path),
                "runtime_signature_sha256": signature,
            }
        )
        runtime.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return runtime, config

    def test_settings_mapping_rejects_frozen_nulls(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be null"):
            SchedulerSettings.from_mapping(
                {
                    "critical_horizon_seconds": None,
                    "prefill_breakpoints": {"knee_tokens": None},
                    "parameters_frozen": True,
                }
            )

    def test_settings_mapping_rejects_unfrozen_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizon must remain null"):
            SchedulerSettings.from_mapping(
                {
                    "critical_horizon_seconds": 0.211,
                    "prefill_breakpoints": {
                        "knee_tokens": 768,
                        "knee_status": "provisional",
                    },
                    "parameters_frozen": False,
                }
            )

    def test_settings_mapping_loads_provisional_knee(self) -> None:
        settings = SchedulerSettings.from_mapping(
            {
                "critical_horizon_seconds": None,
                "prefill_breakpoints": {
                    "knee_tokens": 768,
                    "knee_status": "provisional",
                },
                "parameters_frozen": False,
                "maximum_seed_candidates": 12,
            }
        )
        self.assertFalse(settings.frozen)
        self.assertEqual(settings.prefill_knee_tokens, 768)

    def test_settings_mapping_loads_measured_values(self) -> None:
        settings = SchedulerSettings.from_mapping(
            {
                "critical_horizon_seconds": 0.211,
                "prefill_breakpoints": {"knee_tokens": 768},
                "parameters_frozen": True,
                "maximum_seed_candidates": 12,
            }
        )
        self.assertTrue(settings.frozen)
        self.assertEqual(settings.prefill_knee_tokens, 768)

    def test_loader_rejects_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, config = self._frozen_runtime(Path(directory))
            config["candidate_generator"]["freeze_manifest_sha256"] = "0" * 64
            runtime.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ActiveConfigError, "hash mismatch"):
                load_frozen_candidate_settings(runtime)

    def test_loader_accepts_matching_frozen_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self._frozen_runtime(Path(directory))
            frozen = load_frozen_candidate_settings(runtime)
            self.assertEqual(frozen.settings.prefill_knee_tokens, 768)

    def test_loader_rejects_runtime_signature_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime, config = self._frozen_runtime(Path(directory))
            config["candidate_generator"]["runtime_signature_sha256"] = "f" * 64
            runtime.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ActiveConfigError, "runtime signature mismatch"):
                load_frozen_candidate_settings(runtime)

    def test_checkpoint_keeps_global_two_attempt_bound(self) -> None:
        state = _initial_state()
        item = state["runs"][0]
        self.assertEqual(_remaining_attempts(item), 2)
        item["attempts"].extend([{}, {}])
        self.assertEqual(_remaining_attempts(item), 0)


if __name__ == "__main__":
    unittest.main()
