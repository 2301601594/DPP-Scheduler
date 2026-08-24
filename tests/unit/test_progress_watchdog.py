from __future__ import annotations

import unittest

from dpp_scheduler.observer import (
    ProgressWatchdog,
    ZeroProgressWatchdogError,
)

class ProgressWatchdogTests(unittest.TestCase):
    def test_bounded_diagnostic_and_fail_fast_dump(self) -> None:
        watchdog = ProgressWatchdog(
            max_records=1, zero_progress_limit=2, fail_fast=False
        )
        first = watchdog.record_iteration(
            workload_nonempty=True,
            scheduled_tokens=0,
            prefill_tokens=0,
            decode_tokens=0,
            diagnostic={"candidate_count": 0, "safe_set_rejections": ("OOD",)},
        )
        self.assertFalse(first["watchdog_triggered"])
        second = watchdog.record_iteration(
            workload_nonempty=True,
            scheduled_tokens=0,
            prefill_tokens=0,
            decode_tokens=0,
            diagnostic={"candidate_count": 0, "safe_set_rejections": ("OOD",)},
        )
        self.assertTrue(second["watchdog_triggered"])
        self.assertEqual(second["consecutive_zero_progress"], 2)
        self.assertEqual(len(watchdog.records), 1)

        fail_fast = ProgressWatchdog(
            max_records=1, zero_progress_limit=1, fail_fast=True
        )
        with self.assertRaises(ZeroProgressWatchdogError) as raised:
            fail_fast.record_iteration(
                workload_nonempty=True,
                scheduled_tokens=0,
                prefill_tokens=0,
                decode_tokens=0,
                diagnostic={
                    "candidate_count": 0,
                    "selected_plan": "NONE",
                    "safe_set_rejections": ("OOD",),
                    "scheduler_cpu_seconds": 0.001,
                },
            )
        self.assertTrue(raised.exception.diagnostic["watchdog_triggered"])

