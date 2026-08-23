from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.build_predictor_dataset import BATCH_KINDS, sha256_file
from benchmarks.build_predictor_features import FEATURE_NAMES
from benchmarks.split_predictor_ridge_data import (
    ACTIVE_FEATURES,
    RIDGE_DATASET_ID,
    build_ridge_splits,
    validate_ridge_splits,
)


def _feature_row(kind: str, run_id: str, index: int, base: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_campaign_id": "campaign",
        "source_kind": "targeted" if "recipe" in run_id else "stock",
        "batch_kind": kind,
        "run_id": run_id,
        "iteration_index": index,
        "plan_id": f"plan-{kind}-{index}",
        "snapshot_hash": f"{index + 1:064x}",
        "actual_duration_seconds": 0.1 + index / 100,
        **{name: base * feature_index for feature_index, name in enumerate(FEATURE_NAMES, 1)},
    }


class PredictorRidgeSplitTests(unittest.TestCase):
    def _feature_dataset(self, root: Path) -> Path:
        source = root / "features"
        source.mkdir()
        files = {}
        for kind_index, kind in enumerate(BATCH_KINDS):
            rows = [
                _feature_row(kind, "qps_0p2_seed_1001_attempt_01", kind_index * 10, 1),
                _feature_row(kind, "qps_0p2_seed_1001_attempt_01", kind_index * 10 + 1, 3),
                _feature_row(kind, "qps_0p2_seed_1002_attempt_01", kind_index * 10 + 2, 100),
            ]
            path = source / f"{kind}.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            files[kind] = {
                "file": path.name,
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        (source / "dataset_manifest.json").write_text(
            json.dumps({"dataset_id": "predictor_feature_dataset_v1", "files": files}),
            encoding="utf-8",
        )
        return source

    def test_run_split_and_train_only_standardization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._feature_dataset(root)
            output = root / RIDGE_DATASET_ID
            with patch(
                "benchmarks.split_predictor_ridge_data.validate_feature_dataset"
            ):
                manifest = build_ridge_splits(input_root=source, output_root=output)
            validation = validate_ridge_splits(output)
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["train_runs"], 1)
            self.assertEqual(validation["test_runs"], 1)
            for kind in BATCH_KINDS:
                self.assertEqual(manifest["counts"][kind], {"train": 2, "test": 1})
                self.assertEqual(
                    manifest["scenario_features"][kind]["active"],
                    list(ACTIVE_FEATURES[kind]),
                )
                first = ACTIVE_FEATURES[kind][0]
                self.assertEqual(
                    manifest["standardization"]["by_batch_kind"][kind][first],
                    {
                        "mean": 2.0 * FEATURE_NAMES.index(first) + 2.0,
                        "scale": float(FEATURE_NAMES.index(first) + 1),
                    },
                )
                with gzip.open(
                    output / f"{kind}_test.jsonl.gz", "rt", encoding="utf-8"
                ) as stream:
                    test_row = json.loads(stream.readline())
                self.assertEqual(test_row["standardized_features"][first], 98.0)

            with self.assertRaisesRegex(FileExistsError, "append-only"):
                build_ridge_splits(input_root=source, output_root=output)


if __name__ == "__main__":
    unittest.main()
