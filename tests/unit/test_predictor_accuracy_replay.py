from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.replay_predictor_accuracy import (
    SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
    analyze_replay,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnosis_row(
    frame: int,
    *,
    expected: float,
    conservative: float,
    actual_prefill: int = 0,
    actual_decode: int = 1,
    in_support: bool = True,
) -> dict:
    plan_id = "plan-STOCK"
    return {
        "schema_version": SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
        "frame_id": frame,
        "decision": {
            "executed_plan_id": plan_id,
            "selector_selected_plan_id": plan_id,
            "controller_selected_plan_id": plan_id,
            "selector_reason": "TEST",
            "controller_reason": "TEST",
        },
        "stage1": {"status": "WITHIN_SLACK"},
        "candidates": [
            {
                "plan_id": plan_id,
                "template_id": "STOCK",
                "plan": {
                    "total_prefill_tokens": actual_prefill,
                    "total_decode_tokens": actual_decode,
                },
                "duration": {
                    "expected": expected,
                    "conservative": conservative,
                    "effective": expected,
                    "in_support": in_support,
                    "prediction_mode": (
                        "INTERPOLATION"
                        if in_support
                        else "CONSTRAINED_EXTRAPOLATION"
                    ),
                },
            }
        ],
    }


class PredictorAccuracyReplayTests(unittest.TestCase):
    def _fixture(self, root: Path, *, omit_second_feedback: bool = False):
        diagnosis = root / "selector_diagnosis.jsonl"
        rows = [
            _diagnosis_row(1, expected=0.10, conservative=0.13),
            _diagnosis_row(2, expected=0.11, conservative=0.15),
            {
                "schema_version": SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
                "frame_id": 3,
                "decision": {
                    "executed_plan_id": "zero-frame-3",
                    "selector_reason": "NO_SAFE_DECISION",
                    "controller_reason": "IDLE_EMPTY_QUEUE",
                },
                "stage1": {"status": "NO_SAFE_CANDIDATES"},
                "candidates": [],
            },
        ]
        diagnosis.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        startup = root / "startup.log"
        feedback = [
            "ModularDPPScheduler feedback frame=1 scheduled_tokens=1 "
            "actual_duration_seconds=0.120000000 "
            "timing_source=vllm_aligned_monotonic iteration_index=None "
            "actual_prefill=0 actual_decode=1\n",
        ]
        if not omit_second_feedback:
            feedback.append(
                "ModularDPPScheduler feedback frame=2 scheduled_tokens=1 "
                "actual_duration_seconds=0.140000000 "
                "timing_source=vllm_aligned_monotonic iteration_index=None "
                "actual_prefill=0 actual_decode=1\n"
            )
        feedback.append(
            "ModularDPPScheduler feedback frame=3 scheduled_tokens=0 "
            "actual_duration_seconds=0.000002500 "
            "timing_source=vllm_aligned_monotonic iteration_index=None "
            "actual_prefill=0 actual_decode=0\n"
        )
        startup.write_text("".join(feedback), encoding="utf-8")
        manifest = root / "run_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "run_id": "run",
                    "selector_diagnosis_valid": True,
                    "selector_diagnosis_sha256": _sha(diagnosis),
                    "startup_log_sha256": _sha(startup),
                    "selector_diagnosis_replay": {
                        "frames_replayed": 3,
                        "stage1_mismatch": 0,
                        "winner_mismatch": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return diagnosis, startup, manifest

    def test_replay_joins_actual_only_frames_and_computes_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            diagnosis, startup, manifest = self._fixture(Path(temporary))
            rows, summary = analyze_replay(
                diagnosis_path=diagnosis,
                startup_log_path=startup,
                run_manifest_path=manifest,
                minimum_samples=1,
                window_size=2,
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["calibration_source_replayed"], "offline_oof_cold_start")
        self.assertEqual(rows[1]["calibration_source_replayed"], "online_window")
        self.assertAlmostEqual(summary["overall"]["expected"]["mae_seconds"], 0.025)
        self.assertEqual(summary["overall"]["conservative"]["coverage_rate"], 1.0)
        self.assertEqual(summary["alignment"]["omitted_nonexecuting_frames"][0]["frame_id"], 3)
        self.assertEqual(summary["alignment"]["actual_feedback_frames"], 3)

    def test_replay_rejects_missing_actual_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            diagnosis, startup, manifest = self._fixture(
                Path(temporary), omit_second_feedback=True
            )
            with self.assertRaisesRegex(ValueError, "missing actual feedback"):
                analyze_replay(
                    diagnosis_path=diagnosis,
                    startup_log_path=startup,
                    run_manifest_path=manifest,
                    minimum_samples=1,
                    window_size=2,
                )

    def test_replay_rejects_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            diagnosis, startup, manifest = self._fixture(Path(temporary))
            diagnosis.write_text(diagnosis.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                analyze_replay(
                    diagnosis_path=diagnosis,
                    startup_log_path=startup,
                    run_manifest_path=manifest,
                    minimum_samples=1,
                    window_size=2,
                )


if __name__ == "__main__":
    unittest.main()
