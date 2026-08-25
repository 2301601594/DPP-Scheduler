"""Freeze v2 reference concurrency from normal Stock Snapshot profiles.

The input rows must be schema-v2 Stock natural-workload profiling records.
Selected-request counts are deliberately ignored: the reference is derived
only from the complete pre-decision Snapshot counts captured by the scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
PROFILE_SCHEMA_VERSION = 2
STATISTIC = "p50_positive_frames"
FORBIDDEN_SOURCE_MARKERS = (
    "targeted",
    "artificial",
    "dpp_candidate_diag",
    "scheduler_comparison",
    "formal_benchmark",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _higher_percentile(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("positive-frame percentile requires at least one value")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * (len(ordered) - 1))]


def _stats(values: list[int]) -> dict[str, int | float]:
    positive = [value for value in values if value > 0]
    if not positive:
        raise ValueError("profiling has no positive concurrency frames")
    return {
        "p50": _higher_percentile(positive, 0.50),
        "p75": _higher_percentile(positive, 0.75),
        "p90": _higher_percentile(positive, 0.90),
        "mean": statistics.fmean(positive),
        "max": max(positive),
        "sample_count": len(values),
        "positive_sample_count": len(positive),
    }


def _load_manifest(profile_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = profile_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest is absent for {profile_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError(f"run is not complete: {manifest_path}")
    if manifest.get("kind") != "qwen3_14b_stock_predictor_profile":
        raise ValueError(f"source is not normal Qwen3-14B Stock profiling: {profile_path}")
    resolved = manifest.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError(f"run manifest has no resolved configuration: {manifest_path}")
    command = resolved.get("server_command")
    if not isinstance(command, list) or not any("Qwen3-14B" in str(item) for item in command):
        raise ValueError(f"source command does not identify Qwen3-14B: {manifest_path}")
    return manifest_path, manifest


def _load_counts(path: Path) -> tuple[list[int], list[int], int]:
    prefill: list[int] = []
    decode: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("schema_version") != PROFILE_SCHEMA_VERSION
                or row.get("profile_kind") != "stock_natural_workload"
            ):
                raise ValueError(
                    f"{path}:{line_number} schema mismatch: expected a schema-v2 "
                    "Stock natural frame"
                )
            counts: list[int] = []
            for field in ("snapshot_prefill_count", "snapshot_decode_count"):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{path}:{line_number} has invalid {field}")
                counts.append(value)
            prefill.append(counts[0])
            decode.append(counts[1])
    if not prefill:
        raise ValueError(f"profiling source has no frames: {path}")
    return prefill, decode, len(prefill)


def build_artifact(paths: Iterable[Path], *, repository: Path) -> dict[str, Any]:
    resolved_paths = tuple(sorted({path.resolve() for path in paths}))
    if not resolved_paths:
        raise ValueError("at least one profiling source is required")

    all_prefill: list[int] = []
    all_decode: list[int] = []
    sources: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for path in resolved_paths:
        if not path.is_file() or path.name != "iteration_profile.jsonl":
            raise ValueError(f"source must be an iteration_profile.jsonl file: {path}")
        lowered = str(path).lower()
        marker = next((item for item in FORBIDDEN_SOURCE_MARKERS if item in lowered), None)
        if marker is not None:
            raise ValueError(f"forbidden profiling source marker {marker!r}: {path}")
        manifest_path, manifest = _load_manifest(path)
        run_id = str(manifest.get("run_id", ""))
        if not run_id or run_id in run_ids:
            raise ValueError(f"missing or duplicate run_id: {path}")
        run_ids.add(run_id)
        prefill, decode, frame_count = _load_counts(path)
        all_prefill.extend(prefill)
        all_decode.extend(decode)
        try:
            display_path = str(path.relative_to(repository))
            display_manifest = str(manifest_path.relative_to(repository))
        except ValueError:
            display_path = str(path)
            display_manifest = str(manifest_path)
        sources.append(
            {
                "run_id": run_id,
                "profile_path": display_path,
                "profile_sha256": _sha256(path),
                "run_manifest_path": display_manifest,
                "run_manifest_sha256": _sha256(manifest_path),
                "frame_count": frame_count,
                "qps": manifest.get("qps"),
                "seed": manifest.get("seed"),
            }
        )

    prefill_stats = _stats(all_prefill)
    decode_stats = _stats(all_decode)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "qwen3_14b_dgx_spark_reference_concurrency_v2",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen3-14B",
        "platform": "NVIDIA DGX Spark",
        "source_kind": "normal_stock_natural_workload_profiling_only",
        "statistic": STATISTIC,
        "percentile_method": "higher",
        "prefill_reference_concurrency": max(1, int(prefill_stats["p50"])),
        "decode_reference_concurrency": max(1, int(decode_stats["p50"])),
        "prefill": prefill_stats,
        "decode": decode_stats,
        "total_frames": len(all_prefill),
        "positive_prefill_frames": prefill_stats["positive_sample_count"],
        "positive_decode_frames": decode_stats["positive_sample_count"],
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(args.source, repository=args.repository.resolve())
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
