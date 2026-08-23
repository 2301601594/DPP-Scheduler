from __future__ import annotations

import inspect
import unittest

from dpp_scheduler.state_store import DuplicateLedgerEvent, ObligationLedger
from dpp_scheduler.vllm_adapter import VllmAdapter, get_modular_scheduler_class


class ObligationLedgerTests(unittest.TestCase):
    def test_actual_tokens_settle_once_and_eos_creates_no_next_obligation(self) -> None:
        ledger = ObligationLedger(
            ttft_slo_seconds=2.0,
            tbt_slo_seconds=0.25,
            recovery_age_threshold_seconds=0.5,
        )
        ledger.register_request("r", 10.0)

        first = ledger.observe_output(
            event_id="e1",
            request_id="r",
            returned_at=11.0,
            token_count=1,
            terminal_reason=None,
        )
        self.assertEqual((first.ttft_success, first.ttft_miss), (1, 0))
        view = ledger.request_view("r", 11.8)
        self.assertTrue(view.recovery)
        self.assertTrue(view.recovery_due)

        second = ledger.observe_output(
            event_id="e2",
            request_id="r",
            returned_at=11.8,
            token_count=1,
            terminal_reason=None,
        )
        self.assertEqual((second.tbt_success, second.tbt_miss), (0, 1))

        terminal = ledger.observe_output(
            event_id="e3",
            request_id="r",
            returned_at=11.9,
            token_count=1,
            terminal_reason="stop",
        )
        self.assertEqual((terminal.tbt_success, terminal.tbt_miss), (1, 0))
        self.assertFalse(ledger.has_request("r"))
        self.assertEqual(ledger.terminal_counts, {"stop": 1})
        with self.assertRaises(DuplicateLedgerEvent):
            ledger.observe_output(
                event_id="e3",
                request_id="r",
                returned_at=11.9,
                token_count=1,
                terminal_reason="stop",
            )

    def test_live_snapshot_contains_deadlines_and_recovery(self) -> None:
        class Request:
            def __init__(
                self,
                request_id: str,
                status: str,
                computed: int,
                prompt: int,
            ) -> None:
                self.request_id = request_id
                self.status = status
                self.num_computed_tokens = computed
                self.num_prompt_tokens = prompt
                self.arrival_time = 100.0

        class BlockPool:
            num_gpu_blocks = 100

            @staticmethod
            def get_num_free_blocks() -> int:
                return 90

        class KVManager:
            block_pool = BlockPool()

        class Config:
            enable_prefix_caching = False
            async_scheduling = False

        class Scheduler:
            def __init__(self) -> None:
                prefill = Request("p", "WAITING", 0, 64)
                decode = Request("d", "RUNNING", 64, 64)
                self.requests = {"p": prefill, "d": decode}
                self.running = [decode]
                self.waiting = (prefill,)
                self.kv_cache_manager = KVManager()
                self.block_size = 16
                self.max_num_scheduled_tokens = 2048
                self.max_num_running_reqs = 64
                self.cache_config = Config()
                self.scheduler_config = Config()
                self.num_spec_tokens = 0
                self.connector = None

        ledger = ObligationLedger(2.0, 0.25, 0.2)
        ledger.register_request("p", 100.0)
        ledger.register_request("d", 100.0)
        ledger.observe_output(
            event_id="d-first",
            request_id="d",
            returned_at=101.0,
            token_count=1,
            terminal_reason=None,
        )
        snapshot = VllmAdapter(
            Scheduler(), obligation_ledger=ledger, clock=lambda: 101.5
        ).make_snapshot()

        self.assertEqual(len(snapshot.active_ttft_obligations), 1)
        self.assertEqual(len(snapshot.active_tbt_obligations), 1)
        self.assertEqual(snapshot.recovery_requests, ("d",))
        self.assertEqual(snapshot.waiting_prefill_requests[0].ttft_deadline, 102.0)
        self.assertEqual(snapshot.active_decode_requests[0].tbt_deadline, 101.25)
        self.assertTrue(snapshot.active_decode_requests[0].mandatory)

    def test_live_factory_consumes_engine_output_events(self) -> None:
        source = inspect.getsource(get_modular_scheduler_class)
        self.assertIn("_dpp_obligation_ledger.observe_output", source)
        self.assertIn("output.new_token_ids", source)
        self.assertIn("output.finish_reason", source)


if __name__ == "__main__":
    unittest.main()
