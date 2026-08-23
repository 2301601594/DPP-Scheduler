#!/usr/bin/env python3
"""Create leakage-safe, standardized train/test data for Ridge Predictors."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from benchmarks.build_predictor_dataset import (
    BATCH_KINDS,
    _close_gzip_text,
    _json,
    _open_gzip_text,
    sha256_file,
)
from benchmarks.build_predictor_features import (
    FEATURE_DATASET_ID,
    FEATURE_NAMES,
    validate_feature_dataset,
)


RIDGE_DATASET_ID = "predictor_ridge_splits_v1"
RIDGE_DATASET_SCHEMA_VERSION = 1
SPLITS = ("train", "test")
ACTIVE_FEATURES = {
    "decode_only": ("x_4", "x_5"),
    "mixed": FEATURE_NAMES,
    "prefill_only": ("x_1", "x_2", "x_3", "x_6", "x_7", "x_8"),
}
ROW_FIELDS = frozenset(
    {
        "schema_version",
        "source_campaign_id",
        "source_kind",
        "batch_kind",
        "split",
        "run_id",
        "iteration_index",
        "plan_id",
        "snapshot_hash",
        "actual_duration_seconds",
        "features",
        "standardized_features",
    }
)
STOCK_SEED_PATTERN = re.compile(r"_seed_(?P<seed>\d+)_attempt_\d+$")
TARGET_RECIPE_PATTERN = re.compile(r"_recipe_(?P<seed>\d+)_attempt_\d+$")


class _Moments:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.count = 0
        self.mean = {name: 0.0 for name in names}
        self.m2 = {name: 0.0 for name in names}

    def add(self, row: dict[str, Any]) -> None:
        self.count += 1
        for name in self.names:
            value = float(row[name])
            delta = value - self.mean[name]
            self.mean[name] += delta / self.count
            self.m2[name] += delta * (value - self.mean[name])

    def standardization(self) -> dict[str, dict[str, float]]:
        if self.count < 2:
            raise ValueError("at least two training rows are required")
        result: dict[str, dict[str, float]] = {}
        for name in self.names:
            scale = math.sqrt(self.m2[name] / self.count)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"training feature has zero/invalid variance: {name}")
            result[name] = {"mean": self.mean[name], "scale": scale}
        return result


def _split_for_row(row: dict[str, Any]) -> str:
    run_id = str(row["run_id"])
    source_kind = row.get("source_kind")
    pattern = STOCK_SEED_PATTERN if source_kind == "stock" else TARGET_RECIPE_PATTERN
    if source_kind not in {"stock", "targeted"}:
        raise ValueError(f"unknown source_kind: {source_kind!r}")
    match = pattern.search(run_id)
    if match is None:
        raise ValueError(f"cannot derive split seed from run_id: {run_id}")
    seed = int(match.group("seed"))
    return "train" if seed % 2 else "test"


def _iter_feature_rows(root: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    manifest = _json(root / "dataset_manifest.json")
    if manifest.get("dataset_id") != FEATURE_DATASET_ID:
        raise ValueError("feature dataset identity mismatch")
    for kind in BATCH_KINDS:
        record = manifest["files"].get(kind)
        if not isinstance(record, dict):
            raise ValueError(f"feature manifest missing {kind}")
        with gzip.open(root / str(record["file"]), "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("batch_kind") != kind:
                    raise ValueError(f"invalid feature row: {kind}:{line_number}")
                yield kind, row


def _output_row(
    source: dict[str, Any],
    *,
    kind: str,
    split: str,
    standardization: dict[str, dict[str, float]],
) -> dict[str, Any]:
    names = ACTIVE_FEATURES[kind]
    features = {name: float(source[name]) for name in names}
    standardized = {
        name: (features[name] - standardization[name]["mean"])
        / standardization[name]["scale"]
        for name in names
    }
    return {
        "schema_version": RIDGE_DATASET_SCHEMA_VERSION,
        "source_campaign_id": str(source["source_campaign_id"]),
        "source_kind": str(source["source_kind"]),
        "batch_kind": kind,
        "split": split,
        "run_id": str(source["run_id"]),
        "iteration_index": int(source["iteration_index"]),
        "plan_id": str(source["plan_id"]),
        "snapshot_hash": str(source["snapshot_hash"]),
        "actual_duration_seconds": float(source["actual_duration_seconds"]),
        "features": features,
        "standardized_features": standardized,
    }


def build_ridge_splits(*, input_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"append-only Ridge dataset already exists: {output_root}")
    validate_feature_dataset(input_root)
    source_manifest = _json(input_root / "dataset_manifest.json")

    moments = {kind: _Moments(ACTIVE_FEATURES[kind]) for kind in BATCH_KINDS}
    counts: Counter[tuple[str, str]] = Counter()
    origins: Counter[tuple[str, str, str]] = Counter()
    run_splits: dict[str, str] = {}
    for kind, row in _iter_feature_rows(input_root):
        split = _split_for_row(row)
        run_id = str(row["run_id"])
        previous = run_splits.setdefault(run_id, split)
        if previous != split:
            raise ValueError(f"run assigned to multiple splits: {run_id}")
        counts[(kind, split)] += 1
        origins[(kind, split, str(row["source_kind"]))] += 1
        if split == "train":
            moments[kind].add(row)

    standardization = {
        kind: moments[kind].standardization() for kind in BATCH_KINDS
    }
    for kind in BATCH_KINDS:
        if not all(counts[(kind, split)] > 0 for split in SPLITS):
            raise ValueError(f"both train and test rows are required for {kind}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    handles = {}
    identities: set[tuple[str, int]] = set()
    try:
        for kind in BATCH_KINDS:
            for split in SPLITS:
                handles[(kind, split)] = _open_gzip_text(
                    temporary / f"{kind}_{split}.jsonl.gz"
                )

        written: Counter[tuple[str, str]] = Counter()
        for kind, source in _iter_feature_rows(input_root):
            split = _split_for_row(source)
            row = _output_row(
                source,
                kind=kind,
                split=split,
                standardization=standardization[kind],
            )
            identity = (row["run_id"], row["iteration_index"])
            if identity in identities:
                raise ValueError("duplicate identity in Ridge dataset")
            identities.add(identity)
            handles[(kind, split)][2].write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            written[(kind, split)] += 1

        for key in tuple(handles):
            _close_gzip_text(handles.pop(key))

        files: dict[str, dict[str, Any]] = {}
        for kind in BATCH_KINDS:
            for split in SPLITS:
                key = f"{kind}_{split}"
                path = temporary / f"{key}.jsonl.gz"
                files[key] = {
                    "file": path.name,
                    "batch_kind": kind,
                    "split": split,
                    "rows": written[(kind, split)],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "by_source_kind": {
                        source: origins[(kind, split, source)]
                        for source in ("stock", "targeted")
                        if origins[(kind, split, source)]
                    },
                }

        manifest = {
            "schema_version": RIDGE_DATASET_SCHEMA_VERSION,
            "dataset_id": RIDGE_DATASET_ID,
            "status": "complete",
            "source_dataset_id": FEATURE_DATASET_ID,
            "source_dataset_manifest_sha256": sha256_file(
                input_root / "dataset_manifest.json"
            ),
            "split_policy": {
                "unit": "run_id",
                "train": "odd Stock seed and odd targeted recipe seed",
                "test": "even Stock seed and even targeted recipe seed",
                "test_use": "final evaluation only",
                "run_assignments": dict(sorted(run_splits.items())),
            },
            "scenario_features": {
                kind: {
                    "active": list(ACTIVE_FEATURES[kind]),
                    "dropped": [
                        name for name in FEATURE_NAMES if name not in ACTIVE_FEATURES[kind]
                    ],
                }
                for kind in BATCH_KINDS
            },
            "standardization": {
                "fit_split": "train",
                "formula": "z=(x-mean)/scale",
                "variance_denominator": "n",
                "by_batch_kind": standardization,
            },
            "counts": {
                kind: {split: counts[(kind, split)] for split in SPLITS}
                for kind in BATCH_KINDS
            },
            "files": files,
            "source_files": {
                kind: {
                    "file": source_manifest["files"][kind]["file"],
                    "sha256": source_manifest["files"][kind]["sha256"],
                }
                for kind in BATCH_KINDS
            },
        }
        (temporary / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_root)
        return manifest
    except Exception:
        for value in handles.values():
            try:
                _close_gzip_text(value)
            except Exception:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_ridge_splits(path: Path) -> dict[str, Any]:
    manifest = _json(path / "dataset_manifest.json")
    if (
        manifest.get("schema_version") != RIDGE_DATASET_SCHEMA_VERSION
        or manifest.get("dataset_id") != RIDGE_DATASET_ID
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Ridge dataset manifest identity/status mismatch")
    identities: set[tuple[str, int]] = set()
    runs: dict[str, set[str]] = {split: set() for split in SPLITS}
    observed: Counter[tuple[str, str]] = Counter()
    train_moments = {
        kind: _Moments(ACTIVE_FEATURES[kind]) for kind in BATCH_KINDS
    }
    for kind in BATCH_KINDS:
        names = ACTIVE_FEATURES[kind]
        scaler = manifest["standardization"]["by_batch_kind"][kind]
        for split in SPLITS:
            record = manifest["files"][f"{kind}_{split}"]
            file_path = path / str(record["file"])
            if file_path.parent != path or sha256_file(file_path) != record["sha256"]:
                raise ValueError("Ridge dataset file path/hash mismatch")
            with gzip.open(file_path, "rt", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    row = json.loads(line)
                    if not isinstance(row, dict) or set(row) != ROW_FIELDS:
                        raise ValueError(f"Ridge row schema mismatch: {file_path}:{line_number}")
                    if row.get("batch_kind") != kind or row.get("split") != split:
                        raise ValueError("Ridge row partition mismatch")
                    if _split_for_row(row) != split:
                        raise ValueError("Ridge row violates the frozen split policy")
                    if tuple(row["features"]) != names:
                        raise ValueError("Ridge raw feature schema/order mismatch")
                    if tuple(row["standardized_features"]) != names:
                        raise ValueError("Ridge standardized feature schema/order mismatch")
                    for name in names:
                        value = float(row["features"][name])
                        z_value = float(row["standardized_features"][name])
                        expected = (value - float(scaler[name]["mean"])) / float(
                            scaler[name]["scale"]
                        )
                        if not math.isfinite(z_value) or not math.isclose(
                            z_value, expected, rel_tol=1e-12, abs_tol=1e-12
                        ):
                            raise ValueError("Ridge standardization mismatch")
                    identity = (str(row["run_id"]), int(row["iteration_index"]))
                    if identity in identities:
                        raise ValueError("duplicate Ridge row identity")
                    identities.add(identity)
                    runs[split].add(str(row["run_id"]))
                    observed[(kind, split)] += 1
                    if split == "train":
                        train_moments[kind].add(row["features"])
            if observed[(kind, split)] != int(record["rows"]):
                raise ValueError("Ridge split row count mismatch")
    if runs["train"].intersection(runs["test"]):
        raise ValueError("run leakage between Ridge train and test splits")
    expected = manifest["counts"]
    for kind in BATCH_KINDS:
        recomputed = train_moments[kind].standardization()
        recorded = manifest["standardization"]["by_batch_kind"][kind]
        for name in ACTIVE_FEATURES[kind]:
            for statistic in ("mean", "scale"):
                if not math.isclose(
                    recomputed[name][statistic],
                    float(recorded[name][statistic]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError("standardization was not fitted from train only")
        for split in SPLITS:
            if observed[(kind, split)] != int(expected[kind][split]):
                raise ValueError("Ridge aggregate count mismatch")
    return {
        "schema_version": RIDGE_DATASET_SCHEMA_VERSION,
        "dataset_id": RIDGE_DATASET_ID,
        "valid": True,
        "counts": expected,
        "train_runs": len(runs["train"]),
        "test_runs": len(runs["test"]),
    }


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "status"))
    parser.add_argument(
        "--input",
        default=str(repository / "results" / "dataset" / FEATURE_DATASET_ID),
    )
    parser.add_argument(
        "--output",
        default=str(repository / "results" / "dataset" / RIDGE_DATASET_ID),
    )
    args = parser.parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    if args.command == "status":
        if not output_root.exists():
            print(json.dumps({"dataset_id": RIDGE_DATASET_ID, "status": "absent"}))
            return 0
        print(json.dumps(_json(output_root / "dataset_manifest.json"), indent=2))
        return 0
    if args.command == "build":
        manifest = build_ridge_splits(input_root=input_root, output_root=output_root)
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(json.dumps(validate_ridge_splits(output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
