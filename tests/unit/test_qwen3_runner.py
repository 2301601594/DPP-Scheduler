from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    MODULAR_DPP_SCHEDULER_CLASS,
    REPOSITORY_ROOT,
    build_dpp_server_command,
    build_stock_server_command,
    load_active_runtime,
)
from benchmarks.generate_qwen3_poisson_traces import generation_seed
from benchmarks.run_stock_natural_eos import (
    TRACE_FORBIDDEN_FIELDS,
    _token_count,
    build_request_payload,
    load_trace,
    resolve_execution_scope,
    verify_trace_manifest,
)
from dpp_scheduler import contracts


class Qwen3RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)

    def test_server_command_is_derived_from_active_config(self) -> None:
        command = build_stock_server_command(self.runtime, port=8010)
        joined = " ".join(command)
        self.assertIn("--gpu-memory-utilization 0.84", joined)
        self.assertIn("--max-num-batched-tokens 2048", joined)
        self.assertIn("--max-num-seqs 64", joined)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertEqual(self.runtime.shutdown_timeout_seconds, 10)
        self.assertIn("--shutdown-timeout 10", joined)
        self.assertNotIn("--scheduler-cls", command)
        self.assertNotIn("max_tokens", joined)
        self.assertEqual(self.runtime.pool_size, 3000)
        self.assertEqual(self.runtime.pool_seed, 1001)
        self.assertEqual(self.runtime.request_pool.name, "request_pool.jsonl")
        required_env = dict(self.runtime.required_env)
        self.assertEqual(
            required_env["CPATH"],
            "/home/dongj/LLM/.uv-python/"
            "cpython-3.12.3-linux-aarch64-gnu/include/python3.12",
        )

    def test_dpp_server_command_changes_only_scheduler_class(self) -> None:
        stock = build_stock_server_command(self.runtime, port=8010)
        dpp = build_dpp_server_command(self.runtime, port=8010)
        self.assertEqual(dpp[: len(stock)], stock)
        self.assertEqual(
            dpp[len(stock) :], ["--scheduler-cls", MODULAR_DPP_SCHEDULER_CLASS]
        )

    def test_dpp_development_trace_cannot_request_formal_artifacts(self) -> None:
        self.assertEqual(
            resolve_execution_scope(
                policy="dpp",
                comparison_scope="development_nonformal",
                diagnostic_prefix=False,
            ),
            "development_nonformal",
        )

    def test_only_complete_frozen_trace_dpp_run_is_formal(self) -> None:
        self.assertEqual(
            resolve_execution_scope(
                policy="dpp",
                comparison_scope="active_frozen_trace",
                diagnostic_prefix=False,
            ),
            "formal",
        )
        self.assertEqual(
            resolve_execution_scope(
                policy="dpp",
                comparison_scope="active_frozen_trace",
                diagnostic_prefix=True,
            ),
            "development_nonformal",
        )
        self.assertEqual(
            resolve_execution_scope(
                policy="stock",
                comparison_scope="active_frozen_trace",
                diagnostic_prefix=False,
            ),
            "stock",
        )

    def test_engine_core_setup_links_scheduler_and_config_loader(self) -> None:
        source = (REPOSITORY_ROOT / "scripts/setup_g2_scheduler.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${REPO_ROOT}/dpp_scheduler"', source)
        self.assertIn('"${REPO_ROOT}/benchmarks/qwen3_runtime.py"', source)
        self.assertNotIn('PYTHONPATH=', source)

    def test_dgx_environment_verifies_pinned_python_headers(self) -> None:
        source = (REPOSITORY_ROOT / "scripts/setup_dgx_vllm_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("verify_python_headers", source)
        self.assertIn(
            "d822ddffe62de3eafa3dd7415a0042f2a5790a779551281df64fdc1aa30c966d",
            source,
        )
        self.assertIn('python install "${PYTHON_HEADER_VERSION}"', source)

    def test_request_guard_is_api_only_and_seed_is_honored(self) -> None:
        row = {
            "prompt": "hello",
            "client_safety_ceiling_tokens": self.runtime.client_safety_ceiling_tokens,
            "temperature": self.runtime.temperature,
            "top_p": self.runtime.top_p,
            "generation_seed": 123,
            "ignore_eos": False,
        }
        payload = build_request_payload(self.runtime, row)
        self.assertEqual(payload["max_tokens"], self.runtime.client_safety_ceiling_tokens)
        self.assertEqual(payload["seed"], 123)
        public_fields = {
            field.name
            for value in vars(contracts).values()
            if isinstance(value, type) and dataclasses.is_dataclass(value)
            for field in dataclasses.fields(value)
        }
        self.assertTrue(TRACE_FORBIDDEN_FIELDS.isdisjoint(public_fields))
        self.assertNotIn("client_safety_ceiling_tokens", public_fields)
        self.assertNotIn("max_tokens", public_fields)

    def test_trace_rejects_predetermined_output_state(self) -> None:
        row = {
            "request_id": "r1",
            "prompt_id": "p1",
            "prompt": "hello",
            "input_tokens": 1,
            "arrival_time_s": 0,
            "generation_seed": 1,
            "remaining_output_tokens": 9,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predetermined output state"):
                load_trace(path, self.runtime)

    def test_trace_rejects_deprecated_guard_field(self) -> None:
        row = {
            "request_id": "r1",
            "prompt_id": "p1",
            "prompt": "hello",
            "input_tokens": 1,
            "arrival_time_s": 0,
            "generation_seed": 1,
            "max_tokens_safety": self.runtime.client_safety_ceiling_tokens,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deprecated max_tokens_safety"):
                load_trace(path, self.runtime)

    def test_terminal_chunk_without_token_is_not_counted(self) -> None:
        self.assertEqual(
            _token_count({"text": "", "finish_reason": "stop"}), (0, True)
        )
        self.assertEqual(_token_count({"token_ids": [7], "text": "x"}), (1, True))

    def test_frozen_trace_provenance_survives_scheduler_config_changes(self) -> None:
        trace = self.runtime.active_traces / "qps_0.2_seed_1001_cap2048.jsonl"
        manifest = self.runtime.active_traces / "manifest_cap2048_lowqps.json"
        payload = verify_trace_manifest(trace, manifest, self.runtime)
        self.assertNotEqual(payload["config_sha256"], self.runtime.config_sha256)

    def test_generation_seed_is_valid_signed_int64(self) -> None:
        seed = generation_seed(1001, 999)
        self.assertGreaterEqual(seed, 0)
        self.assertLessEqual(seed, (1 << 63) - 1)


if __name__ == "__main__":
    unittest.main()
