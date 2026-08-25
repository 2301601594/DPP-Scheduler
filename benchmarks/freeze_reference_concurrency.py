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

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    candidate_runtime_signature,
    load_active_runtime,
)
from benchmarks.predictor_profile import STOCK_CONCURRENCY_SEMANTICS


SCHEMA_VERSION = 1
PROFILE_SCHEMA_VERSION = 2
STATISTIC = "p50_positive_frames"
FORBIDDEN_SOURCE_MARKERS = (
    "targeted",
    "artificial",
    "dpp_candidate_diag",
    "scheduler_comparison",
    "formal_benchmark",
    "smoke",
)
ALLOWED_QPS = frozenset({0.20, 0.25, 0.30})
DEVELOPMENT_QPS = 0.25
DEVELOPMENT_SEED = 1003
DEVELOPMENT_REQUEST_COUNT = 300
ARTIFACT_SCOPES = ("formal", "development_nonformal")


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


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    qps = float(manifest.get("qps", -1.0))
    if qps not in ALLOWED_QPS or float(resolved.get("qps", -1.0)) != qps:
        raise ValueError(f"reference source QPS is outside development region: {manifest_path}")
    if int(resolved.get("seed", -1)) != int(manifest.get("seed", -2)):
        raise ValueError(f"reference source seed identity mismatch: {manifest_path}")
    runtime_payload = resolved.get("runtime_consistency")
    runtime_hash = resolved.get("runtime_consistency_sha256")
    if not isinstance(runtime_payload, dict) or _canonical_sha256(
        runtime_payload
    ) != runtime_hash:
        raise ValueError(f"reference source runtime identity is invalid: {manifest_path}")
    file_hashes = manifest.get("file_sha256")
    if not isinstance(file_hashes, dict) or file_hashes.get(
        profile_path.name
    ) != _sha256(profile_path):
        raise ValueError(f"reference profile hash is not bound by manifest: {profile_path}")
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
                or row.get("snapshot_concurrency_semantics")
                != STOCK_CONCURRENCY_SEMANTICS
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
            audit_fields = (
                "snapshot_running_count",
                "snapshot_waiting_count",
                "snapshot_running_prefill_count",
                "snapshot_running_decode_count",
                "snapshot_waiting_prefill_count",
                "snapshot_waiting_decode_count",
                "snapshot_preempted_count",
                "snapshot_other_waiting_count",
                "snapshot_requests_with_preemptions_count",
                "snapshot_total_preemptions",
            )
            audit: dict[str, int] = {}
            for field in audit_fields:
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{path}:{line_number} has invalid {field}")
                audit[field] = value
            if counts[0] != (
                audit["snapshot_running_prefill_count"]
                + audit["snapshot_waiting_prefill_count"]
            ):
                raise ValueError(f"{path}:{line_number} Prefill audit mismatch")
            if counts[1] != audit["snapshot_running_decode_count"]:
                raise ValueError(f"{path}:{line_number} Decode audit mismatch")
            if audit["snapshot_running_count"] != (
                audit["snapshot_running_prefill_count"]
                + audit["snapshot_running_decode_count"]
            ):
                raise ValueError(f"{path}:{line_number} running audit mismatch")
            if audit["snapshot_waiting_count"] != (
                audit["snapshot_waiting_prefill_count"]
                + audit["snapshot_waiting_decode_count"]
                + audit["snapshot_preempted_count"]
                + audit["snapshot_other_waiting_count"]
            ):
                raise ValueError(f"{path}:{line_number} waiting audit mismatch")
            prefill.append(counts[0])
            decode.append(counts[1])
    if not prefill:
        raise ValueError(f"profiling source has no frames: {path}")
    return prefill, decode, len(prefill)


def build_artifact(
    paths: Iterable[Path],
    *,
    repository: Path,
    scope: str = "formal",
    expected_runtime_signature_sha256: str | None = None,
) -> dict[str, Any]:
    if scope not in ARTIFACT_SCOPES:
        raise ValueError(f"unknown reference artifact scope: {scope}")
    resolved_paths = tuple(sorted({path.resolve() for path in paths}))
    if not resolved_paths:
        raise ValueError("at least one profiling source is required")

    all_prefill: list[int] = []
    all_decode: list[int] = []
    sources: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    run_seeds: set[int] = set()
    qps_values: set[float] = set()
    runtime_signature: str | None = None
    runtime_payload: dict[str, Any] | None = None
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
        seed = int(manifest["seed"])
        if seed in run_seeds:
            raise ValueError(f"reference runs must use independent seeds: {seed}")
        run_seeds.add(seed)
        qps = float(manifest["qps"])
        qps_values.add(qps)
        resolved = manifest["resolved"]
        source_signature = str(resolved["runtime_consistency_sha256"])
        source_payload = resolved["runtime_consistency"]
        if runtime_signature is None:
            runtime_signature = source_signature
            runtime_payload = source_payload
        elif runtime_signature != source_signature or runtime_payload != source_payload:
            raise ValueError("reference sources use inconsistent runtimes")
        if (
            expected_runtime_signature_sha256 is not None
            and source_signature != expected_runtime_signature_sha256
        ):
            raise ValueError("reference source runtime differs from active runtime")
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
                "config_path": resolved.get("config"),
                "config_sha256": resolved.get("config_sha256"),
                "runtime_consistency_sha256": source_signature,
            }
        )

    if scope == "formal":
        if qps_values != ALLOWED_QPS:
            raise ValueError(
                "reference sources must cover exactly QPS 0.20, 0.25, and 0.30"
            )
        if any(source_manifest["resolved"].get("diagnostic_prefix") is not False
               for _, source_manifest in (_load_manifest(path) for path in resolved_paths)):
            raise ValueError("formal reference sources must be complete normal runs")
    else:
        if len(sources) != 1:
            raise ValueError("development reference requires exactly one source")
        manifest = _load_manifest(resolved_paths[0])[1]
        resolved = manifest["resolved"]
        if (
            qps_values != {DEVELOPMENT_QPS}
            or int(manifest["seed"]) != DEVELOPMENT_SEED
            or resolved.get("diagnostic_prefix") is not True
            or int(resolved.get("request_count", -1)) != DEVELOPMENT_REQUEST_COUNT
            or resolved.get("comparison_scope")
            != "stock_profile_development_reference_n300"
        ):
            raise ValueError(
                "development reference must be one normal Stock n=300 prefix "
                "at QPS=0.25, seed=1003"
            )
    assert runtime_signature is not None and runtime_payload is not None

    prefill_stats = _stats(all_prefill)
    decode_stats = _stats(all_decode)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": "qwen3_14b_dgx_spark_reference_concurrency_v2",
        "status": "frozen" if scope == "formal" else "frozen_development",
        "scope": scope,
        "formal_benchmark_eligible": scope == "formal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen3-14B",
        "platform": "NVIDIA DGX Spark",
        "source_kind": "normal_stock_natural_workload_profiling_only",
        "development_operating_region_qps": sorted(qps_values),
        "runtime_consistency": runtime_payload,
        "runtime_consistency_sha256": runtime_signature,
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
    parser.add_argument("--scope", choices=ARTIFACT_SCOPES, default="formal")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = load_active_runtime(args.config)
    _, runtime_signature = candidate_runtime_signature(runtime)
    artifact = build_artifact(
        args.source,
        repository=args.repository.resolve(),
        scope=args.scope,
        expected_runtime_signature_sha256=runtime_signature,
    )
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
