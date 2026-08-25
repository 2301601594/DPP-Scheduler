from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.generate_qwen3_poisson_traces import resolve_trace_pairs
from benchmarks.predictor_profile import (
    CAMPAIGN_ID,
    CAMPAIGN_MATRIX,
    FORMAL_REQUEST_COUNT,
    merge_iteration_profiles,
    validate_run_directory,
)
from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    build_stock_profile_server_command,
    build_targeted_profile_server_command,
    load_active_runtime,
)
from benchmarks.run_predictor_profile_campaign import (
    _initial_state,
    _load_state,
    _save_state,
    worker,
)
from dpp_scheduler.vllm_adapter import (
    build_stock_profile_record,
    capture_stock_profile_state,
)


class _CachedRequests:
    def __init__(self, prefill_ids: set[str]) -> None:
        self.prefill_ids = prefill_ids

    def is_context_phase(self, request_id: str) -> bool:
        return request_id in self.prefill_ids


class PredictorProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)

    def test_fixed_matrix_has_twelve_explicit_runs(self) -> None:
        self.assertEqual(len(CAMPAIGN_MATRIX), 12)
        self.assertEqual(FORMAL_REQUEST_COUNT, 500)
        self.assertEqual(
            [(item.qps, item.seed) for item in CAMPAIGN_MATRIX],
            [
                (0.2, 1001),
                (0.2, 1002),
                (0.25, 1003),
                (0.25, 1004),
                (0.3, 1005),
                (0.3, 1006),
                (0.5, 1007),
                (0.5, 1008),
                (1.0, 1009),
                (1.0, 1010),
                (2.0, 1011),
                (2.0, 1012),
            ],
        )
        self.assertEqual(len({item.seed for item in CAMPAIGN_MATRIX}), 12)

    def test_explicit_pairs_do_not_form_cross_product(self) -> None:
        pairs = resolve_trace_pairs(
            qps_values=None,
            seeds=None,
            explicit_pairs=((0.2, 1001), (0.25, 1002)),
        )
        self.assertEqual(pairs, ((0.2, 1001), (0.25, 1002)))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_trace_pairs(
                qps_values=(0.2,),
                seeds=(1001,),
                explicit_pairs=((0.2, 1001),),
            )

    def test_profile_server_is_stock_plus_instrumentation(self) -> None:
        stock = build_stock_profile_server_command(self.runtime, port=8010)
        self.assertIn("--enable-logging-iteration-details", stock)
        scheduler_index = stock.index("--scheduler-cls")
        self.assertEqual(
            stock[scheduler_index + 1],
            "dpp_scheduler.stock_profile_scheduler.StockProfilingScheduler",
        )
        self.assertEqual(stock.count("--scheduler-cls"), 1)

    def test_target_profile_server_uses_exact_plan_scheduler(self) -> None:
        command = build_targeted_profile_server_command(self.runtime, port=8010)
        self.assertIn("--enable-logging-iteration-details", command)
        scheduler_index = command.index("--scheduler-cls")
        self.assertEqual(
            command[scheduler_index + 1],
            "dpp_scheduler.targeted_profile_scheduler.TargetedProfilingScheduler",
        )
        self.assertEqual(command.count("--scheduler-cls"), 1)

    def test_stock_record_contains_only_current_batch_information(self) -> None:
        output = SimpleNamespace(
            total_num_scheduled_tokens=5,
            scheduled_new_reqs=[SimpleNamespace(req_id="prefill-new")],
            scheduled_cached_reqs=_CachedRequests({"prefill-old"}),
            num_scheduled_tokens={
                "prefill-new": 2,
                "prefill-old": 2,
                "decode": 1,
            },
        )
        record = build_stock_profile_record(
            run_id="run-1",
            iteration_index=0,
            captured_state={
                "snapshot_hash": "a" * 64,
                "snapshot_concurrency_semantics": "dpp_stage_queues_v2",
                "snapshot_prefill_count": 2,
                "snapshot_decode_count": 1,
                "snapshot_running_count": 2,
                "snapshot_waiting_count": 1,
                "snapshot_running_prefill_count": 1,
                "snapshot_running_decode_count": 1,
                "snapshot_waiting_prefill_count": 1,
                "snapshot_waiting_decode_count": 0,
                "snapshot_preempted_count": 0,
                "snapshot_other_waiting_count": 0,
                "snapshot_requests_with_preemptions_count": 0,
                "snapshot_total_preemptions": 0,
                "snapshot_running_request_ids": ("decode", "prefill-old"),
                "snapshot_waiting_request_ids": ("prefill-new",),
                "snapshot_running_prefill_request_ids": ("prefill-old",),
                "snapshot_running_decode_request_ids": ("decode",),
                "snapshot_waiting_prefill_request_ids": ("prefill-new",),
                "snapshot_waiting_decode_request_ids": (),
                "snapshot_preempted_request_ids": (),
                "snapshot_other_waiting_request_ids": (),
                "current_context_tokens": {
                    "prefill-new": 0,
                    "prefill-old": 100,
                    "decode": 500,
                },
            },
            scheduler_output=output,
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(
            [item["phase"] for item in record["selected_requests"]],
            ["prefill", "prefill", "decode"],
        )
        rendered = json.dumps(record)
        self.assertNotIn("remaining_output_tokens", rendered)
        self.assertNotIn("expected_output_tokens", rendered)
        self.assertNotIn("max_tokens", rendered)

    def test_stock_concurrency_matches_dpp_stage_queues(self) -> None:
        def request(
            request_id: str,
            *,
            status: str,
            computed: int,
            prompt: int = 100,
            current: int | None = None,
            preemptions: int = 0,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                request_id=request_id,
                status=status,
                num_computed_tokens=computed,
                num_prompt_tokens=prompt,
                num_tokens=prompt if current is None else current,
                num_preemptions=preemptions,
            )

        running_prefill = request("rp", status="RUNNING", computed=50)
        running_decode = request("rd", status="RUNNING", computed=100, current=101)
        waiting_prefill = request("wp", status="WAITING", computed=0)
        waiting_decode = request("wd", status="WAITING", computed=100, current=101)
        preempted_decode = request(
            "pd", status="PREEMPTED", computed=0, current=101, preemptions=1
        )
        scheduler = SimpleNamespace(
            requests={
                item.request_id: item
                for item in (
                    running_prefill,
                    running_decode,
                    waiting_prefill,
                    waiting_decode,
                    preempted_decode,
                )
            },
            running=[running_prefill, running_decode],
            waiting=[waiting_prefill, waiting_decode, preempted_decode],
            kv_cache_manager=SimpleNamespace(
                block_pool=SimpleNamespace(get_num_free_blocks=lambda: 10)
            ),
            max_num_scheduled_tokens=2048,
            max_num_running_reqs=64,
        )
        captured = capture_stock_profile_state(scheduler)
        self.assertEqual(captured["snapshot_prefill_count"], 2)
        self.assertEqual(captured["snapshot_decode_count"], 1)
        self.assertEqual(captured["snapshot_waiting_decode_count"], 1)
        self.assertEqual(captured["snapshot_preempted_count"], 1)
        self.assertNotIn(
            "pd", captured["snapshot_running_decode_request_ids"]
        )

    def test_iteration_batches_and_durations_merge_one_to_one(self) -> None:
        batch = {
            "schema_version": 1,
            "run_id": "run-1",
            "iteration_index": 0,
            "plan_id": "stock-abc",
            "snapshot_hash": "a" * 64,
            "selected_requests": [
                {
                    "request_id": "r1",
                    "phase": "decode",
                    "current_context_tokens": 10,
                    "scheduled_tokens": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batches = root / "scheduled.jsonl"
            startup = root / "startup.log"
            output = root / "iteration_profile.jsonl"
            batches.write_text(json.dumps(batch) + "\n", encoding="utf-8")
            startup.write_text(
                "Engine 000: Iteration(0): 0 context requests, 0 context tokens, "
                "1 generation requests, 1 generation tokens, "
                "iteration elapsed time: 12.50 ms, GPU KV cache usage: 1.0%\n",
                encoding="utf-8",
            )
            validation = merge_iteration_profiles(
                scheduled_batches_path=batches,
                startup_log_path=startup,
                output_path=output,
                expected_run_id="run-1",
            )
            self.assertTrue(validation["valid"])
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["actual_duration_seconds"], 0.0125)

            second_output = root / "mismatch.jsonl"
            startup.write_text(
                startup.read_text(encoding="utf-8").replace("Iteration(0)", "Iteration(1)"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "iteration mismatch"):
                merge_iteration_profiles(
                    scheduled_batches_path=batches,
                    startup_log_path=startup,
                    output_path=second_output,
                    expected_run_id="run-1",
                )

    def test_checkpoint_matrix_is_rejected_if_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = _initial_state()
            _save_state(root, state)
            self.assertEqual(_load_state(root)["campaign_id"], CAMPAIGN_ID)
            state["runs"][0]["seed"] = 999
            _save_state(root, state)
            with self.assertRaisesRegex(ValueError, "matrix mismatch"):
                _load_state(root)

    def test_worker_retries_once_and_resume_skips_valid_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / CAMPAIGN_ID
            runtime = SimpleNamespace(workspace=REPOSITORY_ROOT)
            calls: dict[str, int] = {}

            def fake_attempt(_runtime, _root, _trace_dir, item, state_item, **_kwargs):
                calls[item.key] = calls.get(item.key, 0) + 1
                state_item["attempts"].append({"attempt": calls[item.key]})
                if calls[item.key] == 1:
                    state_item["status"] = "failed"
                    return False
                state_item["status"] = "valid"
                state_item["valid_attempt"] = f"{item.key}_attempt_02"
                return True

            patches = (
                patch(
                    "benchmarks.run_predictor_profile_campaign.require_frozen_for_execution"
                ),
                patch(
                    "benchmarks.run_predictor_profile_campaign._campaign_root",
                    return_value=root,
                ),
                patch(
                    "benchmarks.run_predictor_profile_campaign._git_state",
                    return_value={"commit": "test", "dirty": False, "status": []},
                ),
                patch(
                    "benchmarks.run_predictor_profile_campaign._ensure_traces",
                    return_value="traces_attempt_01",
                ),
                patch(
                    "benchmarks.run_predictor_profile_campaign._attempt_run",
                    side_effect=fake_attempt,
                ),
                patch(
                    "benchmarks.run_predictor_profile_campaign.validate_campaign",
                    return_value={"valid": True},
                ),
                patch("benchmarks.run_predictor_profile_campaign._append_log"),
            )
            for context in patches:
                context.start()
            try:
                self.assertEqual(worker(runtime, resume=False), 0)
                self.assertTrue(all(value == 2 for value in calls.values()))
                calls.clear()
                self.assertEqual(worker(runtime, resume=True), 0)
                self.assertEqual(calls, {})
            finally:
                for context in reversed(patches):
                    context.stop()


if __name__ == "__main__":
    unittest.main()
