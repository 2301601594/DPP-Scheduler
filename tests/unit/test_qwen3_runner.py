from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    build_stock_server_command,
    load_active_runtime,
)
from benchmarks.generate_qwen3_poisson_traces import generation_seed
from benchmarks.run_stock_natural_eos import (
    TRACE_FORBIDDEN_FIELDS,
    _token_count,
    build_request_payload,
    load_trace,
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
        self.assertNotIn("--scheduler-cls", command)
        self.assertNotIn("max_tokens", joined)
        self.assertEqual(self.runtime.pool_size, 3000)
        self.assertEqual(self.runtime.pool_seed, 1001)
        self.assertEqual(self.runtime.request_pool.name, "request_pool.jsonl")

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

    def test_generation_seed_is_valid_signed_int64(self) -> None:
        seed = generation_seed(1001, 999)
        self.assertGreaterEqual(seed, 0)
        self.assertLessEqual(seed, (1 << 63) - 1)


if __name__ == "__main__":
    unittest.main()
