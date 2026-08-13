from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmarks.benchmark_client import build_arrival_times
from benchmarks.dppbench.aggregate import (
    _aggregate_seed_rows,
    aggregate_g1,
    aggregate_g2,
    g2_brackets,
    select_best_fixed,
    select_measured_capacities,
    serial_tpot_drift,
    write_stock_reference,
)
from benchmarks.dppbench.config import (
    ConfigError,
    _validate,
    config_hash,
    slo_thresholds_fingerprint,
)
from benchmarks.dppbench.io import sha256_file, write_jsonl
from benchmarks.dppbench.matrix import (
    RunSpec,
    g1_initial_specs,
    g1_low_load_specs,
    g2_coarse_specs,
    g2_coarse_specs_for_condition,
    g2_measurement_requests,
    g2_seeds,
    g3_specs,
)
from benchmarks.dppbench.metrics import (
    confidence_interval_95,
    percentile,
    request_derived_metrics,
    slo_attainment,
)
from benchmarks.dppbench.traces import (
    _iter_json_array,
    _normalized_poisson_times,
    verify_manifest,
)
from benchmarks.dppbench.runner import _validity
from benchmarks.dppbench.results import record_matches_config, run_record_usable
from benchmarks.run_g1_g2 import dry_run_summary, execution_preflight
from benchmarks.run_matrix import (
    g2_bracket_complete,
    g2_extension_rates,
    g2_fine_rates,
    run_specs,
)


class MetricTests(unittest.TestCase):
    def test_percentile_and_request_metrics(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        metrics = request_derived_metrics(
            {"actual_output_tokens": 4, "ttft_ms": 10, "itls_ms": [2, 3, 4]}
        )
        self.assertEqual(metrics["tpot_ms"], 3)
        self.assertEqual(metrics["max_tbt_ms"], 4)
        self.assertEqual(metrics["e2e_ms"], 19)

    def test_slo_counts_failed_requests_as_miss(self) -> None:
        requests = [
            {
                "success": True,
                "workload_class": "balanced",
                "actual_output_tokens": 2,
                "ttft_ms": 5,
                "itls_ms": [4],
            },
            {"success": False, "workload_class": "balanced"},
        ]
        value = slo_attainment(
            requests, {"balanced": {"ttft_ms": 10, "tpot_ms": 10}}
        )
        self.assertEqual(value["joint_attainment"], 0.5)

    def test_ci_is_computed_over_seed_values(self) -> None:
        mean, low, high = confidence_interval_95([1, 2, 3])
        self.assertEqual(mean, 2)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_serial_tpot_drift_uses_first_and_last_deciles(self) -> None:
        requests = []
        for index in range(100):
            tpot = 20.0 if index >= 90 else 10.0
            requests.append(
                {
                    "success": True,
                    "actual_output_tokens": 2,
                    "ttft_ms": 1.0,
                    "itls_ms": [tpot],
                }
            )
        drift = serial_tpot_drift({"result": {"requests": requests}})
        self.assertEqual(drift["first_mean_tpot_ms"], 10.0)
        self.assertEqual(drift["last_mean_tpot_ms"], 20.0)
        self.assertGreater(drift["relative_drift"], 0.10)

    def test_slo_fingerprint_changes_only_with_calibration_inputs(self) -> None:
        slo = {
            "config_sha256": "config-a",
            "thresholds": {
                "medium": {"balanced": {"ttft_ms": 10, "tpot_ms": 5}}
            },
            "lambda_cap_rps": {},
        }
        first = slo_thresholds_fingerprint(slo)
        slo["lambda_cap_rps"] = {"balanced": 2.0}
        self.assertEqual(first, slo_thresholds_fingerprint(slo))
        slo["thresholds"]["medium"]["balanced"]["ttft_ms"] = 11
        self.assertNotEqual(first, slo_thresholds_fingerprint(slo))


class TraceTests(unittest.TestCase):
    def test_streaming_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            path.write_text('[{"a": 1}, {"b": [2, 3]}]', encoding="utf-8")
            self.assertEqual(list(_iter_json_array(path)), [{"a": 1}, {"b": [2, 3]}])

    def test_arrivals_are_deterministic_and_normalized(self) -> None:
        first = build_arrival_times(50, 2.0, 0.5, 7)
        second = build_arrival_times(50, 2.0, 0.5, 7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[-1], 24.5)
        phase = _normalized_poisson_times(10, 90, 3)
        self.assertEqual(phase[0], 0)
        self.assertAlmostEqual(phase[-1], 90)

    def test_jsonl_sha_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            rows = [{"b": 2, "a": 1}]
            write_jsonl(first, rows)
            write_jsonl(second, rows)
            self.assertEqual(sha256_file(first), sha256_file(second))


class MatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "statistics": {
                "seeds": [1, 2, 3],
                "measurement_requests": 300,
                "serial_measurement_requests": 100,
                "saturation_measurement_requests": 300,
                "low_load_measurement_requests": 100,
            },
            "g1": {
                "scenarios": ["decode_heavy", "balanced", "prefill_heavy"],
                "slo_source": "serial",
            },
            "g2": {
                "slo_tier": "medium",
                "seeds": [1],
                "measurement_requests": 100,
                "conditions": [
                    {
                        "scenario": scenario,
                        "arrival": "poisson",
                        "burstiness": 1.0,
                    }
                    for scenario in (
                        "decode_heavy",
                        "balanced",
                        "prefill_heavy",
                    )
                ],
            },
            "paths": {
                "workspace": "/tmp/dpp-test",
                "traces": "traces",
                "raw_results": "results/raw",
                "processed_results": "results/processed",
            },
            "arrivals": {
                "g3_load_factors": [0.5, 0.9, 1.1],
                "coarse_saturation_factors": [
                    0.1,
                    0.2,
                    0.35,
                    0.5,
                    0.65,
                    0.8,
                    0.95,
                    1.1,
                ],
                "fine_points": 5,
                "max_bracket_extensions": 3,
            },
            "validity": {
                "max_run_attempts": 2,
                "low_load_initial_saturation_fraction": 0.10,
            },
            "budgets": {
                "fixed": [256, 512, 1024, 4096, 8192],
                "equivalence_budget": 2048,
            },
        }

    def test_frozen_matrix_sizes(self) -> None:
        self.assertEqual(len(g1_initial_specs(self.config)), 18)
        coarse = g2_coarse_specs(
            self.config,
            {
                "decode_heavy": 1.0,
                "balanced": 1.0,
                "prefill_heavy": 1.0,
            },
            "slo-fingerprint",
        )
        self.assertEqual(len(coarse), 24)
        self.assertEqual(g2_seeds(self.config), (1,))
        self.assertEqual(g2_measurement_requests(self.config), 100)
        self.assertTrue(all(spec.request_limit == 100 for spec in coarse))
        decode = g2_coarse_specs_for_condition(
            self.config,
            {
                "decode_heavy": 1.0,
                "balanced": 1.0,
                "prefill_heavy": 1.0,
            },
            "slo-fingerprint",
            "decode_heavy",
            "poisson",
        )
        self.assertEqual(
            [spec.load_factor for spec in decode],
            [1.1, 0.95, 0.8, 0.65, 0.5, 0.35, 0.2, 0.1],
        )
        self.assertTrue(
            all(spec.slo_fingerprint == "slo-fingerprint" for spec in coarse)
        )
        capacities = {
            name: 1.0
            for name in (
                "decode_heavy",
                "balanced",
                "prefill_heavy",
                "long_prefill",
                "heterogeneous",
                "sharegpt",
            )
        }
        self.assertEqual(len(g3_specs(self.config, capacities)), 399)

    def test_low_load_can_target_only_failed_scenarios(self) -> None:
        specs = g1_low_load_specs(
            self.config,
            {
                "decode_heavy": 1.0,
                "balanced": 2.0,
                "prefill_heavy": 3.0,
            },
            attempt=1,
            scenarios=["balanced"],
        )
        self.assertEqual(len(specs), 3)
        self.assertEqual({spec.scenario for spec in specs}, {"balanced"})

    def test_dry_run_counts_current_minimal_pipeline(self) -> None:
        summary = dry_run_summary(self.config)
        self.assertEqual(summary["g1"]["nominal_runs"], 18)
        self.assertEqual(summary["g1"]["nominal_measurement_requests"], 3600)
        self.assertFalse(summary["g1"]["low_load_scheduled"])
        self.assertEqual(summary["g2"]["nominal_runs"], 39)
        self.assertEqual(summary["g2"]["nominal_measurement_requests"], 3900)
        self.assertEqual(summary["g2"]["maximum_adaptive_runs"], 48)
        self.assertEqual(
            summary["g2"]["maximum_adaptive_measurement_requests"], 4800
        )
        self.assertEqual(
            summary["g2"]["minimum_runs_if_second_coarse_point_passes"], 21
        )
        self.assertEqual(summary["g2"]["replication_status"], "exploratory_single_seed")

    def test_dry_run_respects_low_load_start_attempts(self) -> None:
        self.config["g1"]["slo_source"] = "low_load"
        self.config["g1"]["low_load_start_attempt"] = {
            scenario: 1 for scenario in self.config["g1"]["scenarios"]
        }
        summary = dry_run_summary(self.config)
        self.assertEqual(summary["g1"]["maximum_adaptive_runs"], 45)
        self.assertEqual(
            summary["g1"]["maximum_adaptive_measurement_requests"], 6300
        )
        self.assertEqual(summary["g1"]["low_load_requests_per_seed"], 100)

    def test_g2_adaptive_rates_are_measured_points(self) -> None:
        self.assertTrue(
            g2_bracket_complete({"pass_rate": 1.0, "fail_rate": 2.0})
        )
        self.assertFalse(
            g2_bracket_complete({"pass_rate": 1.0, "fail_rate": None})
        )
        self.assertEqual(
            g2_extension_rates(
                {
                    "min_rate": 1.0,
                    "max_rate": 4.0,
                    "pass_rate": None,
                    "fail_rate": 2.0,
                },
                1.25,
            ),
            [0.5],
        )
        self.assertEqual(g2_fine_rates(1.0, 2.0, 5), [1.0, 1.25, 1.5, 1.75, 2.0])
        capacities = select_measured_capacities(
            [
                {
                    "scenario": "balanced",
                    "arrival": "poisson",
                    "request_rate_rps": 1.25,
                    "capacity_pass": True,
                },
                {
                    "scenario": "balanced",
                    "arrival": "poisson",
                    "request_rate_rps": 1.5,
                    "capacity_pass": False,
                },
            ]
        )
        self.assertEqual(capacities["balanced:poisson"], 1.25)

    def test_g2_bracket_ignores_hard_invalid_run(self) -> None:
        slo = {
            "config_sha256": "config",
            "thresholds": {
                "medium": {
                    "balanced": {"ttft_ms": 10.0, "tpot_ms": 10.0}
                }
            },
        }
        fingerprint = slo_thresholds_fingerprint(slo)
        invalid = {
            "metadata": {
                "status": "complete",
                "validity": {"valid": False},
                "run_spec": {
                    "scenario": "balanced",
                    "mode": "capacity_poisson",
                    "seed": 1,
                    "request_rate_rps": 2.0,
                    "slo_fingerprint": fingerprint,
                },
            },
            "result": {
                "requests": [
                    {
                        "success": True,
                        "workload_class": "balanced",
                        "actual_output_tokens": 2,
                        "ttft_ms": 20.0,
                        "itls_ms": [2.0],
                    }
                ]
            },
        }
        config = {
            "statistics": {"seeds": [1], "joint_attainment_target": 0.90},
            "g2": {
                "slo_tier": "medium",
                "seeds": [1],
                "conditions": [
                    {
                        "scenario": "balanced",
                        "arrival": "poisson",
                        "burstiness": 1.0,
                    }
                ],
            },
        }
        with (
            patch("benchmarks.dppbench.aggregate._load_slo", return_value=slo),
            patch(
                "benchmarks.dppbench.aggregate._stage_records",
                return_value=[invalid],
            ),
        ):
            bracket = g2_brackets(config)["balanced:poisson"]
        self.assertIsNone(bracket["min_rate"])
        self.assertIsNone(bracket["pass_rate"])
        self.assertIsNone(bracket["fail_rate"])

    def test_best_fixed_near_tie_uses_smaller_budget(self) -> None:
        winner, _ = select_best_fixed({256: 0.991, 512: 1.0, 1024: 0.8}, 0.01)
        self.assertEqual(winner, 256)

    def test_seed_rows_are_not_pooled_before_ci(self) -> None:
        rows = []
        for seed, value in ((1, 1.0), (2, 2.0), (3, 3.0)):
            rows.append(
                {
                    "scenario": "balanced",
                    "arrival": "poisson",
                    "policy": "fixed_b256",
                    "budget": 256,
                    "load_factor": 0.9,
                    "seed": seed,
                    "valid": True,
                    "goodput_rps": value,
                }
            )
        aggregate = _aggregate_seed_rows(rows)[0]
        self.assertEqual(aggregate["goodput_rps_mean"], 2.0)
        self.assertEqual(aggregate["seeds"], 3)


class ValidationTests(unittest.TestCase):
    def test_config_rejects_prefix_caching(self) -> None:
        config = {
            "schema_version": 1,
            "status": "frozen",
            "paths": {},
            "environment": {},
            "model": {
                "max_num_seqs": 64,
                "max_model_len": 8192,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
            },
            "statistics": {
                "measurement_requests": 1,
                "serial_measurement_requests": 1,
                "saturation_measurement_requests": 1,
                "low_load_measurement_requests": 1,
                "seeds": [1],
            },
            "g1": {"scenarios": ["balanced"]},
            "g2": {
                "reference_policy": "stock_auto",
                "slo_tier": "medium",
                "conditions": [
                    {
                        "scenario": "balanced",
                        "arrival": "poisson",
                        "burstiness": 1.0,
                    }
                ],
            },
            "workloads": {
                name: {"input_tokens": 4, "output_tokens": 4}
                for name in ("decode_heavy", "balanced", "prefill_heavy", "long_prefill")
            },
            "datasets": {},
            "arrivals": {},
            "budgets": {"fixed": [256]},
            "validity": {
                "max_run_attempts": 2,
                "serial_tpot_drift_warning_relative": 0.10,
            },
        }
        with self.assertRaises(ConfigError):
            _validate(config)

    def test_invalid_run_detection(self) -> None:
        config = {
            "validity": {
                "failed_requests_max": 0,
                "output_length_mismatch_max": 0,
                "offered_rate_relative_error_max": 0.05,
            }
        }
        result = {
            "scheduled_offered_rate_rps": 1.0,
            "actual_offered_rate_rps": 0.8,
            "summary": {
                "failed": 0,
                "output_length_mismatches": 0,
                "input_length_mismatches": 0,
                "multi_token_chunks": 0,
            },
        }
        validity = _validity(config, result, "", True, {})
        self.assertFalse(validity["valid"])
        self.assertFalse(validity["checks"]["offered_rate"])

    def test_g1_saturation_stream_coalescing_is_throughput_warning(self) -> None:
        config = {
            "validity": {
                "failed_requests_max": 0,
                "output_length_mismatch_max": 0,
                "offered_rate_relative_error_max": 0.05,
            }
        }
        result = {
            "scheduled_offered_rate_rps": None,
            "actual_offered_rate_rps": 100.0,
            "summary": {
                "failed": 0,
                "output_length_mismatches": 0,
                "input_length_mismatches": 0,
                "multi_token_chunks": 1,
            },
        }
        saturation = RunSpec(
            stage="g1",
            scenario="decode_heavy",
            mode="saturation",
            policy="stock_auto",
            seed=1,
            trace_path="unused.jsonl",
        )
        validity = _validity(config, result, "", True, {}, saturation)
        self.assertTrue(validity["valid"])
        self.assertTrue(validity["metric_validity"]["throughput"])
        self.assertFalse(validity["metric_validity"]["token_timing_exact"])
        self.assertEqual(
            validity["warnings"][0]["code"],
            "g1_saturation_multi_token_stream_chunks",
        )

        serial = RunSpec(
            stage="g1",
            scenario="decode_heavy",
            mode="serial",
            policy="stock_auto",
            seed=1,
            trace_path="unused.jsonl",
        )
        self.assertFalse(_validity(config, result, "", True, {}, serial)["valid"])

    def test_legacy_g1_saturation_stream_warning_is_usable(self) -> None:
        record = {
            "metadata": {
                "status": "complete",
                "client_exit_code": 0,
                "run_spec": {"stage": "g1", "mode": "saturation"},
                "validity": {
                    "valid": False,
                    "checks": {
                        "failed_requests": True,
                        "output_lengths": True,
                        "single_token_stream_chunks": False,
                    },
                },
            },
            "result": {"summary": {"multi_token_chunks": 1}},
        }
        self.assertTrue(run_record_usable(record))


class PipelineTests(unittest.TestCase):
    def test_legacy_hash_reuses_only_unchanged_g1_baselines(self) -> None:
        legacy = "a" * 64
        config = {
            "statistics": {
                "serial_measurement_requests": 100,
                "saturation_measurement_requests": 300,
                "low_load_measurement_requests": 100,
            },
            "g1": {"scenarios": ["balanced"]},
            "compatibility": {"g1_baseline_config_sha256": [legacy]},
        }

        def record(mode: str, limit: int) -> dict:
            return {
                "metadata": {
                    "config_sha256": legacy,
                    "run_spec": {
                        "stage": "g1",
                        "scenario": "balanced",
                        "mode": mode,
                        "policy": "stock_auto",
                        "request_limit": limit,
                    },
                }
            }

        self.assertTrue(record_matches_config(config, "g1", record("serial", 100)))
        self.assertTrue(
            record_matches_config(config, "g1", record("saturation", 300))
        )
        self.assertFalse(
            record_matches_config(config, "g1", record("low_load", 300))
        )
        self.assertFalse(record_matches_config(config, "g1", record("serial", 99)))

    def test_trace_manifest_can_use_explicit_compatible_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "traces"
            traces.mkdir()
            legacy = "b" * 64
            (traces / "manifest.json").write_text(
                json.dumps(
                    {
                        "config_sha256": legacy,
                        "traces": {},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "paths": {"workspace": str(root), "traces": "traces"},
                "compatibility": {"trace_manifest_config_sha256": [legacy]},
            }
            self.assertEqual(verify_manifest(config), [])

    def test_preflight_refuses_existing_health_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            vllm = root / "vllm"
            model = root / "model"
            python.touch()
            vllm.touch()
            model.mkdir()
            config = {
                "paths": {
                    "python": str(python),
                    "vllm_cli": str(vllm),
                    "model_snapshot": str(model),
                },
                "model": {"host": "127.0.0.1", "port": 8000},
            }
            with (
                patch("benchmarks.run_g1_g2.verify_manifest", return_value=[]),
                patch("benchmarks.run_g1_g2.server_is_healthy", return_value=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "already serves"):
                    execution_preflight(config)

    def test_invalid_run_is_retried_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid"
            valid = root / "valid"
            invalid.mkdir()
            valid.mkdir()
            (invalid / "metadata.json").write_text(
                json.dumps(
                    {
                        "validity": {
                            "valid": False,
                            "checks": {"output_lengths": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (valid / "metadata.json").write_text(
                json.dumps(
                    {"validity": {"valid": True, "checks": {"output_lengths": True}}}
                ),
                encoding="utf-8",
            )
            spec = RunSpec(
                stage="test",
                scenario="balanced",
                mode="serial",
                policy="stock_auto",
                seed=1,
                trace_path="unused.jsonl",
                request_limit=1,
            )
            config = {"validity": {"max_run_attempts": 2}}
            with patch(
                "benchmarks.run_matrix.execute_run", side_effect=[invalid, valid]
            ) as execute:
                run_specs(config, "test", [spec], resume=False)
            self.assertEqual(execute.call_count, 2)

    def test_resume_does_not_reset_exhausted_retry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "results/raw"
            raw.mkdir(parents=True)
            config = {
                "paths": {
                    "workspace": str(root),
                    "raw_results": "results/raw",
                },
                "validity": {"max_run_attempts": 2},
            }
            spec = RunSpec(
                stage="test",
                scenario="balanced",
                mode="serial",
                policy="stock_auto",
                seed=1,
                trace_path="unused.jsonl",
                request_limit=1,
            )
            for attempt in (1, 2):
                run_dir = raw / f"attempt-{attempt}"
                run_dir.mkdir()
                (run_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "run_id": run_dir.name,
                            "run_key": spec.run_key,
                            "status": "complete",
                            "config_sha256": config_hash(config),
                            "run_spec": {"stage": "test", "mode": "serial"},
                            "validity": {
                                "valid": False,
                                "checks": {"output_lengths": False},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            with patch("benchmarks.run_matrix.execute_run") as execute:
                with self.assertRaisesRegex(RuntimeError, "already exhausted"):
                    run_specs(config, "test", [spec], resume=True)
            execute.assert_not_called()

    def test_incomplete_g1_does_not_freeze_slo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            pending = {
                "schema_version": 1,
                "status": "pending_g1",
                "thresholds": {},
            }
            slo_path = root / "configs/slo.yaml"
            slo_path.write_text(yaml.safe_dump(pending), encoding="utf-8")
            config = {
                "paths": {
                    "workspace": str(root),
                    "raw_results": "results/raw",
                    "processed_results": "results/processed",
                },
                "statistics": {
                    "seeds": [1, 2, 3],
                    "serial_measurement_requests": 100,
                    "saturation_measurement_requests": 300,
                    "low_load_measurement_requests": 100,
                },
                "g1": {"scenarios": ["balanced"]},
                "g2": {"slo_tier": "medium"},
                "validity": {
                    "seed_variation_target": 0.05,
                    "serial_tpot_drift_warning_relative": 0.10,
                    "low_load_ttft_increase_max": 0.25,
                },
            }
            derived = aggregate_g1(config, allow_missing_low_load=True)
            self.assertFalse(derived["gate_passed"])
            self.assertEqual(
                yaml.safe_load(slo_path.read_text(encoding="utf-8"))["status"],
                "pending_g1",
            )

    def test_serial_source_freezes_slo_without_low_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "results/raw"
            configs = root / "configs"
            raw.mkdir(parents=True)
            configs.mkdir()
            (configs / "slo.yaml").write_text(
                yaml.safe_dump({"status": "pending_g1", "thresholds": {}}),
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "workspace": str(root),
                    "raw_results": "results/raw",
                    "processed_results": "results/processed",
                },
                "statistics": {
                    "seeds": [1, 2, 3],
                    "serial_measurement_requests": 100,
                    "saturation_measurement_requests": 300,
                    "low_load_measurement_requests": 100,
                },
                "g1": {"scenarios": ["balanced"], "slo_source": "serial"},
                "g2": {"slo_tier": "medium", "seeds": [1]},
                "validity": {
                    "seed_variation_target": 0.05,
                    "serial_tpot_drift_warning_relative": 0.10,
                    "low_load_ttft_increase_max": 0.25,
                },
            }
            config_sha = config_hash(config)
            serial_requests = [
                {
                    "success": True,
                    "actual_output_tokens": 2,
                    "ttft_ms": 10.0,
                    "itls_ms": [4.0],
                }
                for _ in range(100)
            ]
            for mode, limit, throughput in (
                ("serial", 100, 0.1),
                ("saturation", 300, 2.0),
            ):
                for seed in (1, 2, 3):
                    run = raw / f"{mode}-{seed}"
                    run.mkdir()
                    metadata = {
                        "run_id": run.name,
                        "run_key": run.name,
                        "status": "complete",
                        "client_exit_code": 0,
                        "config_sha256": config_sha,
                        "validity": {"valid": True, "checks": {}},
                        "run_spec": {
                            "stage": "g1",
                            "scenario": "balanced",
                            "mode": mode,
                            "policy": "stock_auto",
                            "seed": seed,
                            "attempt": 0,
                            "request_limit": limit,
                        },
                    }
                    result = {
                        "summary": {
                            "p90_ttft_ms": 10.0,
                            "p90_tpot_ms": 4.0,
                            "request_throughput_rps": throughput,
                            "multi_token_chunks": 0,
                        },
                        "requests": serial_requests if mode == "serial" else [],
                    }
                    (run / "metadata.json").write_text(
                        json.dumps(metadata), encoding="utf-8"
                    )
                    (run / "client_result.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
            derived = aggregate_g1(config, allow_missing_low_load=True)
            slo = yaml.safe_load(
                (configs / "slo.yaml").read_text(encoding="utf-8")
            )
            self.assertTrue(derived["gate_passed"])
            self.assertFalse(derived["low_load_gate_required"])
            self.assertEqual(derived["low_load"], {})
            self.assertEqual(slo["calibration_sample"]["mode"], "serial")
            self.assertEqual(
                slo["thresholds"]["medium"]["balanced"],
                {"ttft_ms": 40.0, "tpot_ms": 8.0},
            )

    def test_stock_reference_contains_slo_qps_goodput_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            g1_dir = root / "results/processed/g1"
            g1_dir.mkdir(parents=True)
            warning = {"code": "serial_tpot_drift", "scenario": "balanced"}
            (g1_dir / "derived.json").write_text(
                json.dumps({"quality_warnings": [warning]}), encoding="utf-8"
            )
            config = {
                "paths": {
                    "workspace": str(root),
                    "processed_results": "results/processed",
                },
                "g2": {
                    "slo_tier": "medium",
                    "seeds": [1],
                    "conditions": [
                        {
                            "scenario": "balanced",
                            "arrival": "poisson",
                            "burstiness": 1.0,
                        }
                    ],
                },
            }
            slo = {
                "thresholds_fingerprint": "fingerprint",
                "thresholds": {
                    "medium": {
                        "balanced": {"ttft_ms": 100.0, "tpot_ms": 20.0}
                    }
                },
            }
            (root / "configs/slo.yaml").write_text(
                yaml.safe_dump(slo), encoding="utf-8"
            )
            point = {
                "scenario": "balanced",
                "arrival": "poisson",
                "request_rate_rps": 2.0,
                "achieved_offered_rate_rps_mean": 2.0,
                "achieved_offered_rate_rps_ci95_low": 1.9,
                "achieved_offered_rate_rps_ci95_high": 2.1,
                "completed_throughput_rps_mean": 1.99,
                "completed_throughput_rps_ci95_low": 1.9,
                "completed_throughput_rps_ci95_high": 2.0,
                "goodput_rps_mean": 1.8,
                "goodput_rps_ci95_low": 1.7,
                "goodput_rps_ci95_high": 1.9,
                "joint_attainment_mean": 0.91,
                "joint_attainment_ci95_low": 0.90,
                "joint_attainment_ci95_high": 0.92,
                "seeds": 3,
            }
            payload = write_stock_reference(
                config,
                {
                    "gate_passed": True,
                    "lambda_cap_by_arrival": {"balanced:poisson": 2.0},
                    "points": [point],
                },
            )
            row = payload["rows"][0]
            self.assertEqual(row["scheduler"], "stock_auto")
            self.assertEqual(row["ttft_slo_ms"], 100.0)
            self.assertEqual(row["lambda_cap_target_rps"], 2.0)
            self.assertEqual(row["goodput_rps_mean"], 1.8)
            self.assertEqual(row["quality_status"], "complete_with_warnings")
            self.assertTrue(
                (root / "results/processed/g1_g2/stock_reference.csv").exists()
            )

    def test_g2_aggregation_builds_capacity_and_seed_ci_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "results/raw"
            g1_dir = root / "results/processed/g1"
            configs = root / "configs"
            raw.mkdir(parents=True)
            g1_dir.mkdir(parents=True)
            configs.mkdir()
            config = {
                "paths": {
                    "workspace": str(root),
                    "raw_results": "results/raw",
                    "processed_results": "results/processed",
                    "artifacts": "artifacts",
                },
                "statistics": {
                    "seeds": [1, 2, 3],
                    "joint_attainment_target": 0.90,
                },
                "g2": {
                    "slo_tier": "medium",
                    "seeds": [1],
                    "conditions": [
                        {
                            "scenario": "balanced",
                            "arrival": "poisson",
                            "burstiness": 1.0,
                        }
                    ],
                },
            }
            config_sha = config_hash(config)
            slo = {
                "status": "frozen_g1",
                "config_sha256": config_sha,
                "thresholds": {
                    "medium": {
                        "balanced": {"ttft_ms": 10.0, "tpot_ms": 10.0}
                    }
                },
                "lambda_cap_rps": {},
                "lambda_cap_by_arrival": {},
            }
            fingerprint = slo_thresholds_fingerprint(slo)
            slo["thresholds_fingerprint"] = fingerprint
            (configs / "slo.yaml").write_text(
                yaml.safe_dump(slo), encoding="utf-8"
            )
            (g1_dir / "derived.json").write_text(
                json.dumps({"quality_warnings": []}), encoding="utf-8"
            )
            requests = [
                {
                    "success": True,
                    "workload_class": "balanced",
                    "actual_output_tokens": 2,
                    "ttft_ms": 5.0,
                    "itls_ms": [2.0],
                }
                for _ in range(2)
            ]
            for seed in (1,):
                run = raw / f"run-{seed}"
                run.mkdir()
                metadata = {
                    "run_id": f"run-{seed}",
                    "run_key": f"key-{seed}",
                    "status": "complete",
                    "config_sha256": config_sha,
                    "validity": {"valid": True},
                    "run_spec": {
                        "stage": "g2",
                        "scenario": "balanced",
                        "mode": "capacity_poisson",
                        "seed": seed,
                        "request_rate_rps": 1.0,
                        "slo_fingerprint": fingerprint,
                    },
                }
                result = {
                    "actual_offered_rate_rps": 1.0,
                    "summary": {
                        "duration_s": 1.0,
                        "request_throughput_rps": 2.0,
                    },
                    "requests": requests,
                }
                (run / "metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                (run / "client_result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
            derived = aggregate_g2(config)
            self.assertTrue(derived["gate_passed"])
            self.assertEqual(derived["lambda_cap_rps"]["balanced"], 1.0)
            self.assertEqual(derived["points"][0]["goodput_rps_mean"], 2.0)
            self.assertEqual(
                derived["replication_status"], "exploratory_single_seed"
            )
            self.assertTrue(math.isnan(derived["points"][0]["goodput_rps_ci95_low"]))
            self.assertEqual(derived["quality_warnings"][0]["code"], "g2_single_seed")
            self.assertTrue(
                (root / "results/processed/g2/per_seed.csv").exists()
            )
            self.assertTrue((root / "artifacts/g2_goodput.svg").exists())


if __name__ == "__main__":
    unittest.main()
