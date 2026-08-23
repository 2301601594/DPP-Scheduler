#!/usr/bin/env python3
"""Build the validated Predictor feature dataset from the iteration dataset.

The iteration dataset contains one row per executed BatchPlan with per-request
phase, current context, and scheduled token counts.  This module derives the
candidate aggregate features used by offline Predictor model selection.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO

from benchmarks.build_predictor_dataset import (
    BATCH_KINDS,
    DATASET_ID as SOURCE_DATASET_ID,
    FEATURE_NAMES,
    _close_gzip_text,
    _json,
    _open_gzip_text,
    extract_features,
    sha256_file,
    validate_dataset,
)


FEATURE_DATASET_ID = "predictor_feature_dataset_v1"
FEATURE_SCHEMA_VERSION = 1
FEATURE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "source_campaign_id",
        "source_kind",
        "batch_kind",
        "run_id",
        "iteration_index",
        "plan_id",
        "snapshot_hash",
        "actual_duration_seconds",
        *FEATURE_NAMES,
    }
)
FEATURE_DEFINITIONS = (
    {
        "name": "x_1",
        "definition": "sum_{i in P} q_i * (u_i + q_i)",
        "rationale": "Prefill attention compute",
    },
    {
        "name": "x_2",
        "definition": "sum_{i in P} q_i^2",
        "rationale": "Prefill chunk self-attention intensity",
    },
    {
        "name": "x_3",
        "definition": "sum_{i in P union D} u_i",
        "rationale": "Current batch total KV/context size",
    },
    {
        "name": "x_4",
        "definition": "|D|",
        "rationale": "Decode batch size",
    },
    {
        "name": "x_5",
        "definition": "sum_{i in D} u_i",
        "rationale": "Decode attention/KV read amount",
    },
    {
        "name": "x_6",
        "definition": "sum_{i in P} q_i",
        "rationale": "This iteration Prefill token total",
    },
    {
        "name": "x_7",
        "definition": "max_{i in P} q_i",
        "rationale": "Largest single-request Prefill chunk",
    },
    {
        "name": "x_8",
        "definition": "|P|",
        "rationale": "Prefill request count",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _feature_row(source: dict[str, Any]) -> dict[str, Any]:
    features = extract_features(source["selected_requests"])
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source_campaign_id": str(source["source_campaign_id"]),
        "source_kind": str(source["source_kind"]),
        "batch_kind": str(source["batch_kind"]),
        "run_id": str(source["run_id"]),
        "iteration_index": int(source["iteration_index"]),
        "plan_id": str(source["plan_id"]),
        "snapshot_hash": str(source["snapshot_hash"]),
        "actual_duration_seconds": float(source["actual_duration_seconds"]),
        **features,
    }


def _iter_source_rows(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    manifest = _json(path / "dataset_manifest.json")
    if manifest.get("dataset_id") != SOURCE_DATASET_ID:
        raise ValueError(f"source dataset identity mismatch: {path}")
    for kind in BATCH_KINDS:
        record = manifest["files"].get(kind)
        if not isinstance(record, dict):
            raise ValueError(f"source manifest missing {kind}")
        file_path = path / str(record["file"])
        with gzip.open(file_path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"source row is not an object: {file_path}:{line_number}"
                    )
                yield kind, row


def build_feature_dataset(
    *,
    input_root: Path,
    output_root: Path,
    build_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one append-only feature dataset directory from an iteration dataset."""
    if output_root.exists():
        raise FileExistsError(f"append-only feature dataset already exists: {output_root}")
    validate_dataset(input_root)
    source_manifest = _json(input_root / "dataset_manifest.json")
    if build_config_sha256 is None:
        build_config_sha256 = source_manifest.get("source_config_sha256")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    handles: dict[str, tuple[BinaryIO, gzip.GzipFile, TextIO]] = {}
    counts: Counter[str] = Counter()
    for kind in BATCH_KINDS:
        counts[kind] = 0
    origins: Counter[str] = Counter()
    selected_records: Counter[str] = Counter()
    durations: dict[str, list[float]] = {kind: [] for kind in BATCH_KINDS}
    identities: set[tuple[str, int]] = set()
    snapshots: set[str] = set()
    try:
        for kind in BATCH_KINDS:
            handles[kind] = _open_gzip_text(temporary / f"{kind}.jsonl.gz")

        for kind, source in _iter_source_rows(input_root):
            if source.get("batch_kind") != kind:
                raise ValueError("source row stored in the wrong batch-kind file")
            row = _feature_row(source)
            identity = (row["run_id"], row["iteration_index"])
            snapshot = row["snapshot_hash"]
            if identity in identities or snapshot in snapshots:
                raise ValueError("duplicate feature iteration identity or snapshot hash")
            identities.add(identity)
            snapshots.add(snapshot)
            handles[kind][2].write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            counts[kind] += 1
            origins[row["source_kind"]] += 1
            selected_records[kind] += len(source["selected_requests"])
            durations[kind].append(row["actual_duration_seconds"])

        for kind in BATCH_KINDS:
            _close_gzip_text(handles.pop(kind))

        files: dict[str, dict[str, Any]] = {}
        for kind in BATCH_KINDS:
            path = temporary / f"{kind}.jsonl.gz"
            files[kind] = {
                "file": path.name,
                "rows": counts[kind],
                "selected_request_records": selected_records[kind],
                "duration_min_seconds": min(durations[kind]) if durations[kind] else None,
                "duration_max_seconds": max(durations[kind]) if durations[kind] else None,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_id": FEATURE_DATASET_ID,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "build_config_sha256": build_config_sha256,
            "source_dataset_id": SOURCE_DATASET_ID,
            "source_dataset_manifest_sha256": sha256_file(
                input_root / "dataset_manifest.json"
            ),
            "source_dataset_files": {
                kind: {
                    "file": str(source_manifest["files"][kind]["file"]),
                    "sha256": str(source_manifest["files"][kind]["sha256"]),
                }
                for kind in BATCH_KINDS
            },
            "feature_schema": {
                "label": "actual_duration_seconds",
                "features": FEATURE_DEFINITIONS,
                "row_fields": sorted(FEATURE_ROW_FIELDS),
                "feature_names": list(FEATURE_NAMES),
            },
            "counts": {
                "total": sum(counts.values()),
                "by_batch_kind": dict(sorted(counts.items())),
                "by_source_kind": dict(sorted(origins.items())),
            },
            "files": files,
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


def validate_feature_dataset(path: Path) -> dict[str, Any]:
    manifest = _json(path / "dataset_manifest.json")
    if (
        manifest.get("schema_version") != FEATURE_SCHEMA_VERSION
        or manifest.get("dataset_id") != FEATURE_DATASET_ID
        or manifest.get("status") != "complete"
    ):
        raise ValueError("feature dataset manifest identity/status mismatch")
    counts: Counter[str] = Counter()
    for kind in BATCH_KINDS:
        counts[kind] = 0
    identities: set[tuple[str, int]] = set()
    snapshots: set[str] = set()
    for kind in BATCH_KINDS:
        record = manifest["files"].get(kind)
        if not isinstance(record, dict):
            raise ValueError(f"feature dataset manifest is missing {kind}")
        file_path = path / str(record["file"])
        if file_path.parent != path or file_path.suffixes[-2:] != [".jsonl", ".gz"]:
            raise ValueError("feature dataset file path is invalid")
        if sha256_file(file_path) != record.get("sha256"):
            raise ValueError(f"feature dataset file hash mismatch: {kind}")
        with gzip.open(file_path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != FEATURE_ROW_FIELDS:
                    raise ValueError(
                        f"feature dataset row schema mismatch: {kind}:{line_number}"
                    )
                if row.get("schema_version") != FEATURE_SCHEMA_VERSION:
                    raise ValueError("feature dataset row version mismatch")
                if row.get("batch_kind") != kind:
                    raise ValueError("feature dataset row stored in wrong batch-kind file")
                if row.get("source_kind") not in {"stock", "targeted"}:
                    raise ValueError("feature dataset source kind is invalid")
                duration = float(row["actual_duration_seconds"])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("feature dataset duration must be finite and positive")
                for name in FEATURE_NAMES:
                    value = row[name]
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(
                            f"feature {name} must be a finite number: {kind}:{line_number}"
                        )
                identity = (str(row["run_id"]), int(row["iteration_index"]))
                snapshot = str(row["snapshot_hash"])
                if identity in identities or snapshot in snapshots:
                    raise ValueError("feature dataset contains duplicate identity")
                identities.add(identity)
                snapshots.add(snapshot)
                counts[kind] += 1
        if counts[kind] != int(record.get("rows", -1)):
            raise ValueError(f"feature dataset row count mismatch: {kind}")
    expected = manifest.get("counts", {}).get("by_batch_kind")
    if dict(sorted(counts.items())) != expected:
        raise ValueError("feature dataset aggregate counts mismatch")
    if sum(counts.values()) != int(manifest.get("counts", {}).get("total", -1)):
        raise ValueError("feature dataset total row count mismatch")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_id": FEATURE_DATASET_ID,
        "valid": True,
        "total_rows": sum(counts.values()),
        "batch_kind_counts": dict(sorted(counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "status"))
    parser.add_argument(
        "--input",
        default=str(
            Path(__file__).resolve().parents[1]
            / "results"
            / "processed"
            / "qwen3_14b_dgx_spark"
            / SOURCE_DATASET_ID
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "results"
            / "dataset"
            / FEATURE_DATASET_ID
        ),
    )
    parser.add_argument("--build-config-sha256", default=None)
    args = parser.parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    if args.command == "status":
        if not output_root.exists():
            print(json.dumps({"dataset_id": FEATURE_DATASET_ID, "status": "absent"}))
            return 0
        print(json.dumps(_json(output_root / "dataset_manifest.json"), indent=2))
        return 0
    if args.command == "build":
        manifest = build_feature_dataset(
            input_root=input_root,
            output_root=output_root,
            build_config_sha256=args.build_config_sha256,
        )
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    validation = validate_feature_dataset(output_root)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
