from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.build_predictor_dataset import (
    BATCH_KINDS,
    DATASET_ID,
    build_dataset,
    validate_dataset,
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


class PredictorDatasetTests(unittest.TestCase):
    def _campaigns(self, root: Path) -> tuple[Path, Path]:
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

    def test_builds_three_lossless_strata_and_excludes_control_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock, target = self._campaigns(root)
            output = root / DATASET_ID
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
                manifest = build_dataset(
                    stock_root=stock,
                    target_root=target,
                    output_root=output,
                    max_tokens=2048,
                    max_sequences=64,
                    build_config_sha256="build-config",
                )
            self.assertEqual(
                manifest["counts"]["by_batch_kind"],
                {"decode_only": 1, "mixed": 1, "prefill_only": 2},
            )
            self.assertEqual(
                manifest["selection"]["excluded_target_roles"],
                {"drain": 1, "setup": 1},
            )
            self.assertTrue(validate_dataset(output)["valid"])
            for kind in BATCH_KINDS:
                with gzip.open(output / f"{kind}.jsonl.gz", "rt", encoding="utf-8") as stream:
                    rendered = stream.read()
                self.assertNotIn("sample_role", rendered)
                self.assertNotIn("requested_shape", rendered)
                self.assertNotIn("remaining_output_tokens", rendered)

            with self.assertRaisesRegex(FileExistsError, "append-only"):
                build_dataset(
                    stock_root=stock,
                    target_root=target,
                    output_root=output,
                    max_tokens=2048,
                    max_sequences=64,
                )

    def test_rejects_forbidden_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock, target = self._campaigns(root)
            path = stock / "runs" / "stock-valid" / "iteration_profile.jsonl"
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            row["remaining_output_tokens"] = 10
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
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
                self.assertRaisesRegex(ValueError, "forbidden fields"),
            ):
                build_dataset(
                    stock_root=stock,
                    target_root=target,
                    output_root=root / DATASET_ID,
                    max_tokens=2048,
                    max_sequences=64,
                )


if __name__ == "__main__":
    unittest.main()
