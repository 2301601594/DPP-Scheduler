from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.build_predictor_dataset import (
    DATASET_ID,
    build_dataset,
    extract_features,
)
from benchmarks.build_predictor_features import (
    FEATURE_DATASET_ID,
    FEATURE_NAMES,
    build_feature_dataset,
    validate_feature_dataset,
)
from benchmarks.predictor_profile import CampaignRun
from dpp_scheduler.targeted_profile import TargetCampaignRun


def _row(
    run_id: str,
    index: int,
    selected: list[dict[str, object]],
    *,
    role: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "iteration_index": index,
        "plan_id": f"plan-{run_id}-{index}",
        "snapshot_hash": f"{index + 1:064x}",
        "actual_duration_seconds": 0.1 + index / 100,
        "selected_requests": selected,
    }
    if role is not None:
        row["sample_role"] = role
    return row


def _request(
    request_id: str, phase: str, context: int, scheduled: int
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "phase": phase,
        "current_context_tokens": context,
        "scheduled_tokens": scheduled,
    }


def _campaigns(root: Path) -> tuple[Path, Path]:
    stock = root / "stock"
    target = root / "target"
    stock_run = stock / "runs" / "stock-valid"
    target_run = target / "runs" / "target-valid"
    stock_run.mkdir(parents=True)
    target_run.mkdir(parents=True)
    common_git = {
        "root": {"commit": "root", "dirty": True},
        "vllm": {"commit": "vllm", "dirty": False},
    }
    for run in (stock_run, target_run):
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "resolved": {"config_sha256": "config"},
                    "git": common_git,
                }
            ),
            encoding="utf-8",
        )
    for campaign in (stock, target):
        (campaign / "campaign_manifest.json").write_text("{}\n", encoding="utf-8")
        (campaign / "campaign_checkpoint.json").write_text("{}\n", encoding="utf-8")

    stock_rows = [
        _row("stock-valid", 0, [_request("d", "decode", 20, 1)]),
        _row("stock-valid", 1, [_request("p", "prefill", 0, 8)]),
    ]
    target_rows = [
        _row("target-valid", 10, [_request("setup", "decode", 5, 1)], role="setup"),
        _row(
            "target-valid",
            11,
            [
                _request("m-p", "prefill", 4, 16),
                _request("m-d", "decode", 30, 1),
            ],
            role="target",
        ),
        _row(
            "target-valid",
            12,
            [_request("target-p", "prefill", 10, 32)],
            role="target",
        ),
        _row("target-valid", 13, [_request("drain", "decode", 6, 1)], role="drain"),
    ]
    for path, rows in (
        (stock_run / "iteration_profile.jsonl", stock_rows),
        (target_run / "iteration_profile.jsonl", target_rows),
    ):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return stock, target


class PredictorFeatureTests(unittest.TestCase):
    def test_extract_features_matches_definitions(self) -> None:
        selected = [
            _request("p1", "prefill", 10, 20),
            _request("p2", "prefill", 5, 4),
            _request("d1", "decode", 30, 1),
            _request("d2", "decode", 7, 1),
        ]
        self.assertEqual(
            extract_features(selected),
            {
                "x_1": 20 * (10 + 20) + 4 * (5 + 4),
                "x_2": 20 * 20 + 4 * 4,
                "x_3": 10 + 5 + 30 + 7,
                "x_4": 2,
                "x_5": 30 + 7,
                "x_6": 20 + 4,
                "x_7": 20,
                "x_8": 2,
            },
        )

    def test_builds_and_validates_feature_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock, target = _campaigns(root)
            input_root = root / DATASET_ID
            with (
                patch(
                    "benchmarks.build_predictor_dataset._valid_stock_runs",
                    return_value=[(CampaignRun(0.2, 1001), "stock-valid")],
                ),
                patch(
                    "benchmarks.build_predictor_dataset._valid_target_runs",
                    return_value=[
                        (TargetCampaignRun(0.2, 1001, 2001), "target-valid")
                    ],
                ),
            ):
                build_dataset(
                    stock_root=stock,
                    target_root=target,
                    output_root=input_root,
                    max_tokens=2048,
                    max_sequences=64,
                )
            output_root = root / FEATURE_DATASET_ID
            build_feature_dataset(input_root=input_root, output_root=output_root)
            validation = validate_feature_dataset(output_root)
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["total_rows"], 4)
            with gzip.open(
                output_root / "mixed.jsonl.gz", "rt", encoding="utf-8"
            ) as stream:
                mixed = json.loads(stream.readline())
            self.assertEqual(mixed["x_1"], 320)
            self.assertEqual(mixed["x_2"], 256)
            self.assertEqual(mixed["x_3"], 34)
            self.assertEqual(mixed["x_4"], 1)
            self.assertEqual(mixed["x_5"], 30)
            self.assertEqual(mixed["x_6"], 16)
            self.assertEqual(mixed["x_7"], 16)
            self.assertEqual(mixed["x_8"], 1)
            self.assertNotIn("selected_requests", mixed)
            self.assertEqual(set(FEATURE_NAMES), set(mixed) & set(FEATURE_NAMES))
            with self.assertRaisesRegex(FileExistsError, "append-only"):
                build_feature_dataset(input_root=input_root, output_root=output_root)


if __name__ == "__main__":
    unittest.main()
