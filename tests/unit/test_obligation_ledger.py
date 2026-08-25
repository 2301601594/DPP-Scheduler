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
        self.assertTrue(first.initializes_tbt_service)
        self.assertEqual(first.tbt_service_tokens, 0)

        expired = ledger.expire_deadlines(11.8)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].event_id, "expiry:r:TBT:1")
        self.assertEqual((expired[0].tbt_success, expired[0].tbt_miss), (0, 1))
        self.assertEqual(ledger.expire_deadlines(11.8), ())
        self.assertEqual(ledger.active_obligations({"r"}), ((), ()))

        view = ledger.request_view("r", 11.8)
        self.assertTrue(view.recovery)
        self.assertTrue(view.recovery_due)
        self.assertFalse(view.goodput_eligible)
        self.assertIsNone(view.tbt_deadline)

        late = ledger.observe_output(
            event_id="e2",
            request_id="r",
            returned_at=11.8,
            token_count=1,
            terminal_reason=None,
        )
        self.assertEqual((late.tbt_success, late.tbt_miss), (0, 0))
        self.assertFalse(late.initializes_tbt_service)
        self.assertEqual(late.tbt_service_tokens, 1)
        self.assertFalse(ledger.request_view("r", 11.8).goodput_eligible)
        self.assertTrue(ledger.request_view("r", 11.8).recovery)

        terminal = ledger.observe_output(
            event_id="e3",
            request_id="r",
            returned_at=11.9,
            token_count=1,
            terminal_reason="stop",
        )
        self.assertEqual((terminal.tbt_success, terminal.tbt_miss), (0, 0))
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

    def test_ttft_deadline_expiry_is_once_and_late_first_token_is_not_success(self) -> None:
        ledger = ObligationLedger(2.0, 0.25, 0.2)
        ledger.register_request("r", 10.0)

        expired = ledger.expire_deadlines(12.0)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].event_id, "expiry:r:TTFT:0")
        self.assertEqual((expired[0].ttft_success, expired[0].ttft_miss), (0, 1))
        self.assertEqual(ledger.expire_deadlines(12.0), ())
        self.assertEqual(ledger.active_obligations({"r"}), ((), ()))

        view = ledger.request_view("r", 12.0)
        self.assertFalse(view.goodput_eligible)
        self.assertIsNone(view.ttft_deadline)
        self.assertFalse(view.recovery)

        late = ledger.observe_output(
            event_id="late-first",
            request_id="r",
            returned_at=12.1,
            token_count=1,
            terminal_reason=None,
        )
        self.assertEqual(
            (late.ttft_success, late.ttft_miss, late.tbt_success, late.tbt_miss),
            (0, 0, 0, 0),
        )
        self.assertFalse(ledger.request_view("r", 12.1).goodput_eligible)
        self.assertEqual(ledger.active_obligations({"r"}), ((), ()))

    def test_tbt_deadline_expiry_is_once_and_recovery_survives_late_token(self) -> None:
        ledger = ObligationLedger(2.0, 0.25, 0.2)
        ledger.register_request("r", 10.0)
        first = ledger.observe_output(
            event_id="first",
            request_id="r",
            returned_at=10.0,
            token_count=1,
            terminal_reason=None,
        )
        self.assertEqual((first.ttft_success, first.ttft_miss), (1, 0))

        expired = ledger.expire_deadlines(10.25)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].event_id, "expiry:r:TBT:1")
        self.assertEqual((expired[0].tbt_success, expired[0].tbt_miss), (0, 1))
        self.assertEqual(ledger.expire_deadlines(10.25), ())
        self.assertEqual(ledger.active_obligations({"r"}), ((), ()))

        view = ledger.request_view("r", 10.45)
        self.assertTrue(view.recovery)
        self.assertTrue(view.recovery_due)
        self.assertEqual(view.recovery_first_miss_time, 10.25)
        self.assertFalse(view.goodput_eligible)
        self.assertIsNone(view.tbt_deadline)

        late = ledger.observe_output(
            event_id="late-tbt",
            request_id="r",
            returned_at=10.5,
            token_count=1,
            terminal_reason=None,
        )
        self.assertEqual((late.tbt_success, late.tbt_miss), (0, 0))
        late_view = ledger.request_view("r", 10.5)
        self.assertTrue(late_view.recovery)
        self.assertEqual(late_view.recovery_first_miss_time, 10.25)
        self.assertFalse(late_view.goodput_eligible)
        self.assertEqual(ledger.active_obligations({"r"}), ((), ()))

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
        expired = ledger.expire_deadlines(101.8)
        self.assertEqual(len(expired), 1)
        self.assertEqual((expired[0].tbt_success, expired[0].tbt_miss), (0, 1))

        snapshot = VllmAdapter(
            Scheduler(),
            obligation_ledger=ledger,
            critical_horizon_seconds=0.22,
            clock=lambda: 101.8,
        ).make_snapshot()

        self.assertEqual(len(snapshot.active_ttft_obligations), 1)
        self.assertEqual(len(snapshot.active_tbt_obligations), 0)
        self.assertEqual(snapshot.recovery_requests, ("d",))
        self.assertEqual(snapshot.waiting_prefill_requests[0].ttft_deadline, 102.0)
        self.assertIsNone(snapshot.active_decode_requests[0].tbt_deadline)
        self.assertTrue(snapshot.active_decode_requests[0].mandatory)
        self.assertTrue(
            snapshot.waiting_prefill_requests[0].hard_ttft_protected
        )
        self.assertEqual(snapshot.waiting_prefill_requests[0].ttft_slo_seconds, 2.0)
        self.assertEqual(snapshot.active_decode_requests[0].tbt_slo_seconds, 0.25)

    def test_live_factory_consumes_engine_output_events(self) -> None:
        source = inspect.getsource(get_modular_scheduler_class)
        self.assertIn("_dpp_obligation_ledger.observe_output", source)
        self.assertIn("output.new_token_ids", source)
        self.assertIn("output.finish_reason", source)


if __name__ == "__main__":
    unittest.main()
