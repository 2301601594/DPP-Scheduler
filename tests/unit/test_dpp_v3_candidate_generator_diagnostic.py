from __future__ import annotations

import unittest

from scripts.dpp_v3_candidate_generator_diagnostic import build_schedule


class SyntheticScheduleTests(unittest.TestCase):
    def test_retains_multiple_waiting_prefill_requests(self) -> None:
        trace = [
            {"arrival_time_s": 0.0, "input_tokens": 100},
            {"arrival_time_s": 1.0, "input_tokens": 200},
        ]
        schedule = build_schedule(
            trace,
            assumed_output_tokens=256,
            ttft_slo_seconds=2.0,
            tbt_slo_seconds=0.25,
            synthetic_prefill_hold_seconds=2.0,
        )
        _, waiting, decode = schedule[1]
        self.assertEqual([item.request_id for item in waiting], ["r0000", "r0001"])
        self.assertEqual(decode, [])

    def test_decode_deadline_is_for_the_next_token(self) -> None:
        trace = [
            {"arrival_time_s": 0.0, "input_tokens": 100},
            {"arrival_time_s": 3.0, "input_tokens": 200},
        ]
        schedule = build_schedule(
            trace,
            assumed_output_tokens=256,
            ttft_slo_seconds=2.0,
            tbt_slo_seconds=0.25,
            synthetic_prefill_hold_seconds=2.0,
        )
        timestamp, _, decode = schedule[1]
        self.assertEqual(len(decode), 1)
        self.assertEqual(decode[0].tokens_decoded, 4)
        self.assertAlmostEqual(decode[0].tbt_deadline, timestamp + 0.25)


if __name__ == "__main__":
    unittest.main()
