from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.generate_qwen3_poisson_traces import resolve_trace_pairs
from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    MODULAR_DPP_SCHEDULER_CLASS,
    REPOSITORY_ROOT,
    load_active_runtime,
)
from benchmarks.run_scheduler_comparison_campaign import (
    _initial_state,
    _load_state,
    worker,
)
from benchmarks.scheduler_comparison import (
    ComparisonRun,
    load_comparison_settings,
    validate_pair,
    validate_run_directory,
)


class SchedulerComparisonCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
        cls.settings = load_comparison_settings(cls.runtime)

    def test_fixed_matrix_has_six_n300_single_seed_runs(self) -> None:
        self.assertEqual(self.settings.request_count, 300)
        self.assertEqual(self.settings.seed, 1001)
        self.assertFalse(self.settings.formal_benchmark_eligible)
        self.assertEqual(
            [(item.policy, item.qps, item.seed) for item in self.settings.matrix],
            [
                ("stock", 0.2, 1001),
                ("dpp", 0.2, 1001),
                ("dpp", 0.25, 1001),
                ("stock", 0.25, 1001),
                ("stock", 0.3, 1001),
                ("dpp", 0.3, 1001),
            ],
        )
        self.assertEqual(
            [(item.policy, item.qps) for item in self.settings.smoke_matrix],
            [("stock", 0.2), ("dpp", 0.2)],
        )

    def test_trace_generator_accepts_one_seed_at_three_qps_values(self) -> None:
        pairs = resolve_trace_pairs(
            qps_values=None,
            seeds=None,
            explicit_pairs=((0.2, 1001), (0.25, 1001), (0.3, 1001)),
        )
        self.assertEqual(
            pairs, ((0.2, 1001), (0.25, 1001), (0.3, 1001))
        )

    def test_checkpoint_rejects_changed_main_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = _initial_state(self.settings)
            (root / "campaign_checkpoint.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            self.assertEqual(
                len(_load_state(root, self.settings)["runs"]), 6
            )
            state["runs"][0]["qps"] = 9.0
            (root / "campaign_checkpoint.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "main matrix mismatch"):
                _load_state(root, self.settings)

    @staticmethod
    def _write_synthetic_run(
        root: Path, *, policy: str, smoke: bool, trace_sha: str = "a" * 64
    ) -> None:
        root.mkdir(parents=True)
        request_count = 1 if smoke else 300
        stock_command = ["vllm", "serve", "model"]
        command = (
            stock_command
            if policy == "stock"
            else stock_command + ["--scheduler-cls", MODULAR_DPP_SCHEDULER_CLASS]
        )
        row = {
            "request_id": "r1",
            "prompt_id": "p1",
            "planned_arrival_s": 0.0,
            "generation_seed": 1,
            "client_safety_ceiling_tokens": 2048,
            "input_tokens": 128,
            "completed": True,
            "error": None,
            "http_status": 200,
            "finish_reason": "stop",
            "actual_output_tokens": 1,
            "observed_stream_tokens": 1,
            "token_timing_exact": True,
            "ttft_ms": 10.0,
            "itls_ms": [],
        }
        rows = [dict(row, request_id=f"r{index}") for index in range(request_count)]
        summary = {
            "num_requests": request_count,
            "completed": request_count,
            "failed": 0,
            "input_token_mismatches": 0,
            "stream_token_count_mismatches": 0,
            "finish_reason_counts": {"stop": request_count},
        }
        resolved = {
            "config_sha256": "c" * 64,
            "trace": "/trace.jsonl",
            "trace_sha256": trace_sha,
            "trace_manifest": "/manifest.json",
            "trace_manifest_sha256": "b" * 64,
            "request_count": request_count,
            "source_request_count": 300,
            "diagnostic_prefix": smoke,
            "planned_arrival_span_s": 0.0 if smoke else 1000.0,
            "scheduler_policy": policy,
            "comparison_scope": "development_nonformal",
            "client_safety_ceiling_tokens": 2048,
            "scheduler_receives_safety_ceiling": False,
            "required_env": {"VLLM_USE_V1": "1"},
            "dpp_diagnostic_iteration_log": False,
            "server_command": command,
        }
        manifest = {
            "schema_version": 2,
            "run_id": root.name,
            "scheduler_policy": policy,
            "comparison_eligible": not smoke,
            "formal_comparison_eligible": False,
            "status": "complete",
            "resolved": resolved,
            "summary": summary,
            "git": {"root": {"commit": "1"}, "vllm": {"commit": "2"}},
        }
        (root / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (root / "per_request.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
        )
        (root / "startup.log").write_text("clean shutdown\n", encoding="utf-8")

    def test_run_and_pair_validation_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock = root / "smoke_stock"
            dpp = root / "smoke_dpp"
            self._write_synthetic_run(stock, policy="stock", smoke=True)
            self._write_synthetic_run(dpp, policy="dpp", smoke=True)
            validate_run_directory(
                stock,
                expected_run=ComparisonRun("stock", 0.2, 1001),
                expected_request_count=1,
                smoke=True,
            )
            self.assertTrue(validate_pair(stock, dpp, smoke=True)["valid"])
            manifest = json.loads((dpp / "run_manifest.json").read_text())
            manifest["resolved"]["trace_sha256"] = "d" * 64
            (dpp / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "resolved mismatch"):
                validate_pair(stock, dpp, smoke=True)

    def test_smoke_failure_blocks_all_main_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / self.settings.campaign_id
            with (
                patch(
                    "benchmarks.run_scheduler_comparison_campaign.require_frozen_for_execution"
                ),
                patch(
                    "benchmarks.run_scheduler_comparison_campaign._campaign_root",
                    return_value=root,
                ),
                patch(
                    "benchmarks.run_scheduler_comparison_campaign._ensure_traces",
                    return_value="traces_attempt_01",
                ),
                patch(
                    "benchmarks.run_scheduler_comparison_campaign._run_with_attempt_bound",
                    return_value=False,
                ) as run_mock,
                patch(
                    "benchmarks.run_scheduler_comparison_campaign._git_state",
                    return_value={"commit": "test", "dirty": False, "status": []},
                ),
            ):
                self.assertEqual(
                    worker(
                        self.runtime,
                        self.settings,
                        resume=False,
                        smoke_only=True,
                    ),
                    1,
                )
            self.assertEqual(run_mock.call_count, 1)
            state = _load_state(root, self.settings)
            self.assertEqual(state["status"], "smoke_failed")
            self.assertTrue(all(item["status"] == "pending" for item in state["runs"]))

    def test_shell_launcher_uses_detached_tmux_and_preflight(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/scheduler_comparison_campaign.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("tmux new-session -d", source)
        self.assertIn("resource_preflight", source)
        self.assertIn("--resource-approved", source)
        self.assertIn("smoke gate", source.lower())


if __name__ == "__main__":
    unittest.main()
