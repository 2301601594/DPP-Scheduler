#!/usr/bin/env python3
"""Build the validated, lossless Predictor iteration dataset by batch kind."""

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

from benchmarks.predictor_profile import (
    CAMPAIGN_ID as STOCK_CAMPAIGN_ID,
    CAMPAIGN_MATRIX,
    FORMAL_REQUEST_COUNT,
    FORBIDDEN_PROFILE_FIELDS,
    CampaignRun,
    validate_run_directory,
)
from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    load_active_runtime,
    resolve_under,
    sha256_file,
)
from benchmarks.targeted_predictor_profile import validate_target_run_directory
from dpp_scheduler.targeted_profile import (
    TARGET_CAMPAIGN_ID,
    TARGET_CAMPAIGN_MATRIX,
    TARGET_REQUEST_COUNT,
    TargetCampaignRun,
)


DATASET_ID = "predictor_iteration_dataset_v1"
DATASET_SCHEMA_VERSION = 1
BATCH_KINDS = ("decode_only", "mixed", "prefill_only")
ROW_FIELDS = frozenset(
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
        "selected_requests",
    }
)
REQUEST_FIELDS = frozenset(
    {"request_id", "phase", "current_context_tokens", "scheduled_tokens"}
)
FEATURE_NAMES = ("x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8")


def extract_features(selected_requests: list[dict[str, Any]]) -> dict[str, int]:
    """Derive the candidate Predictor features from a selected request list.

    For Prefill requests ``u_i`` is ``current_context_tokens`` and ``q_i`` is
    ``scheduled_tokens``.  For Decode requests only ``u_i`` is used.
    """
    prefills = [item for item in selected_requests if item["phase"] == "prefill"]
    decodes = [item for item in selected_requests if item["phase"] == "decode"]
    prefill_uq = [
        (int(item["current_context_tokens"]), int(item["scheduled_tokens"]))
        for item in prefills
    ]
    decode_u = [int(item["current_context_tokens"]) for item in decodes]
    return {
        "x_1": sum(q * (u + q) for u, q in prefill_uq),
        "x_2": sum(q * q for _, q in prefill_uq),
        "x_3": sum(u for u, _ in prefill_uq) + sum(decode_u),
        "x_4": len(decodes),
        "x_5": sum(decode_u),
        "x_6": sum(q for _, q in prefill_uq),
        "x_7": max((q for _, q in prefill_uq), default=0),
        "x_8": len(prefills),
    }


# Convenience alias matching the usual feature-engineering wording.
derive_features = extract_features


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PROFILE_FIELDS:
                found.add(key)
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def _batch_kind(selected: list[dict[str, Any]]) -> str:
    has_prefill = any(item.get("phase") == "prefill" for item in selected)
    has_decode = any(item.get("phase") == "decode" for item in selected)
    if has_prefill and has_decode:
        return "mixed"
    if has_prefill:
        return "prefill_only"
    if has_decode:
        return "decode_only"
    raise ValueError("iteration row contains no Prefill or Decode request")


def _validate_source_row(
    row: dict[str, Any],
    *,
    expected_run_id: str,
    max_tokens: int,
    max_sequences: int,
) -> str:
    forbidden = _forbidden_keys(row)
    if forbidden:
        raise ValueError(f"source row has forbidden fields: {sorted(forbidden)}")
    if row.get("run_id") != expected_run_id:
        raise ValueError("source row run_id mismatch")
    duration = float(row.get("actual_duration_seconds", 0.0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("source row duration must be finite and positive")
    selected = row.get("selected_requests")
    if not isinstance(selected, list) or not selected:
        raise ValueError("source row selected_requests must be non-empty")
    if len(selected) > max_sequences:
        raise ValueError("source row exceeds sequence budget")
    seen_requests: set[str] = set()
    total_tokens = 0
    for request in selected:
        if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
            raise ValueError("selected request schema mismatch")
        request_id = str(request["request_id"])
        if not request_id or request_id in seen_requests:
            raise ValueError("selected request ID is empty or duplicated")
        seen_requests.add(request_id)
        phase = request["phase"]
        context = int(request["current_context_tokens"])
        scheduled = int(request["scheduled_tokens"])
        if phase not in {"prefill", "decode"} or context < 0 or scheduled <= 0:
            raise ValueError("selected request values are invalid")
        if phase == "decode" and scheduled != 1:
            raise ValueError("Decode requests must schedule exactly one token")
        total_tokens += scheduled
    if total_tokens > max_tokens:
        raise ValueError("source row exceeds token budget")
    return _batch_kind(selected)


def _dataset_row(
    source: dict[str, Any], *, source_campaign_id: str, source_kind: str
) -> dict[str, Any]:
    selected = [
        {
            "request_id": str(item["request_id"]),
            "phase": str(item["phase"]),
            "current_context_tokens": int(item["current_context_tokens"]),
            "scheduled_tokens": int(item["scheduled_tokens"]),
        }
        for item in source["selected_requests"]
    ]
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_campaign_id": source_campaign_id,
        "source_kind": source_kind,
        "batch_kind": _batch_kind(selected),
        "run_id": str(source["run_id"]),
        "iteration_index": int(source["iteration_index"]),
        "plan_id": str(source["plan_id"]),
        "snapshot_hash": str(source["snapshot_hash"]),
        "actual_duration_seconds": float(source["actual_duration_seconds"]),
        "selected_requests": selected,
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            yield value


def _valid_stock_runs(root: Path) -> list[tuple[CampaignRun, str]]:
    checkpoint = _json(root / "campaign_checkpoint.json")
    if checkpoint.get("campaign_id") != STOCK_CAMPAIGN_ID:
        raise ValueError("Stock campaign identity mismatch")
    if checkpoint.get("status") != "complete":
        raise ValueError("Stock campaign is not complete")
    entries = checkpoint.get("runs")
    if not isinstance(entries, list) or len(entries) != len(CAMPAIGN_MATRIX):
        raise ValueError("Stock checkpoint matrix length mismatch")
    runs: list[tuple[CampaignRun, str]] = []
    for expected, entry in zip(CAMPAIGN_MATRIX, entries):
        if (
            entry.get("key") != expected.key
            or float(entry.get("qps", -1)) != expected.qps
            or int(entry.get("seed", -1)) != expected.seed
            or entry.get("status") != "valid"
        ):
            raise ValueError(f"Stock checkpoint entry mismatch: {expected.key}")
        run_id = str(entry.get("valid_attempt", ""))
        if not run_id:
            raise ValueError(f"Stock run has no valid attempt: {expected.key}")
        validate_run_directory(
            root / "runs" / run_id,
            expected_run=expected,
            expected_request_count=FORMAL_REQUEST_COUNT,
        )
        runs.append((expected, run_id))
    return runs


def _valid_target_runs(root: Path) -> list[tuple[TargetCampaignRun, str]]:
    checkpoint = _json(root / "campaign_checkpoint.json")
    if checkpoint.get("campaign_id") != TARGET_CAMPAIGN_ID:
        raise ValueError("target campaign identity mismatch")
    if checkpoint.get("status") != "complete":
        raise ValueError("target campaign is not complete")
    entries = checkpoint.get("runs")
    if not isinstance(entries, list) or len(entries) != len(TARGET_CAMPAIGN_MATRIX):
        raise ValueError("target checkpoint matrix length mismatch")
    runs: list[tuple[TargetCampaignRun, str]] = []
    for expected, entry in zip(TARGET_CAMPAIGN_MATRIX, entries):
        if (
            entry.get("key") != expected.key
            or float(entry.get("source_qps", -1)) != expected.source_qps
            or int(entry.get("source_seed", -1)) != expected.source_seed
            or int(entry.get("recipe_seed", -1)) != expected.recipe_seed
            or entry.get("status") != "valid"
        ):
            raise ValueError(f"target checkpoint entry mismatch: {expected.key}")
        run_id = str(entry.get("valid_attempt", ""))
        if not run_id:
            raise ValueError(f"target run has no valid attempt: {expected.key}")
        validate_target_run_directory(
            root / "runs" / run_id,
            expected_run=expected,
            expected_request_count=TARGET_REQUEST_COUNT,
        )
        runs.append((expected, run_id))
    return runs


def _open_gzip_text(path: Path) -> tuple[BinaryIO, gzip.GzipFile, TextIO]:
    raw = path.open("xb")
    compressed = gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
    return raw, compressed, text


def _close_gzip_text(handles: tuple[BinaryIO, gzip.GzipFile, TextIO]) -> None:
    raw, compressed, text = handles
    text.flush()
    text.detach()
    compressed.close()
    raw.close()


def _source_record(
    campaign_root: Path,
    run_id: str,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    run_root = campaign_root / "runs" / run_id
    manifest = _json(run_root / "run_manifest.json")
    return {
        **metadata,
        "run_id": run_id,
        "config_sha256": manifest["resolved"]["config_sha256"],
        "root_git_commit": manifest["git"]["root"]["commit"],
        "root_git_dirty": manifest["git"]["root"]["dirty"],
        "vllm_git_commit": manifest["git"]["vllm"]["commit"],
        "vllm_git_dirty": manifest["git"]["vllm"]["dirty"],
        "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
        "iteration_profile_sha256": sha256_file(
            run_root / "iteration_profile.jsonl"
        ),
    }


def build_dataset(
    *,
    stock_root: Path,
    target_root: Path,
    output_root: Path,
    max_tokens: int,
    max_sequences: int,
    build_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one append-only dataset directory from validated campaign attempts."""
    if output_root.exists():
        raise FileExistsError(f"append-only dataset already exists: {output_root}")
    stock_runs = _valid_stock_runs(stock_root)
    target_runs = _valid_target_runs(target_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    handles: dict[str, tuple[BinaryIO, gzip.GzipFile, TextIO]] = {}
    counts: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    selected_records: Counter[str] = Counter()
    durations: dict[str, list[float]] = {kind: [] for kind in BATCH_KINDS}
    identities: set[tuple[str, int]] = set()
    snapshots: set[str] = set()
    excluded_target_roles: Counter[str] = Counter()
    source_runs: list[dict[str, Any]] = []
    try:
        for kind in BATCH_KINDS:
            handles[kind] = _open_gzip_text(temporary / f"{kind}.jsonl.gz")

        def emit(source: dict[str, Any], campaign_id: str, source_kind: str) -> None:
            kind = _validate_source_row(
                source,
                expected_run_id=str(source["run_id"]),
                max_tokens=max_tokens,
                max_sequences=max_sequences,
            )
            row = _dataset_row(
                source, source_campaign_id=campaign_id, source_kind=source_kind
            )
            identity = (row["run_id"], row["iteration_index"])
            snapshot = row["snapshot_hash"]
            if identity in identities or snapshot in snapshots:
                raise ValueError("duplicate iteration identity or snapshot hash")
            identities.add(identity)
            snapshots.add(snapshot)
            handles[kind][2].write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            counts[kind] += 1
            origins[source_kind] += 1
            selected_records[kind] += len(row["selected_requests"])
            durations[kind].append(row["actual_duration_seconds"])

        for run, run_id in stock_runs:
            path = stock_root / "runs" / run_id / "iteration_profile.jsonl"
            source_runs.append(
                _source_record(
                    stock_root,
                    run_id,
                    metadata={
                        "source_campaign_id": STOCK_CAMPAIGN_ID,
                        "source_kind": "stock",
                        "qps": run.qps,
                        "seed": run.seed,
                    },
                )
            )
            for source in _iter_jsonl(path):
                emit(source, STOCK_CAMPAIGN_ID, "stock")

        for run, run_id in target_runs:
            path = target_root / "runs" / run_id / "iteration_profile.jsonl"
            source_runs.append(
                _source_record(
                    target_root,
                    run_id,
                    metadata={
                        "source_campaign_id": TARGET_CAMPAIGN_ID,
                        "source_kind": "targeted",
                        "source_qps": run.source_qps,
                        "source_seed": run.source_seed,
                        "recipe_seed": run.recipe_seed,
                    },
                )
            )
            for source in _iter_jsonl(path):
                role = str(source.get("sample_role", ""))
                if role != "target":
                    excluded_target_roles[role or "missing"] += 1
                    continue
                emit(source, TARGET_CAMPAIGN_ID, "targeted")

        for kind in BATCH_KINDS:
            _close_gzip_text(handles.pop(kind))

        files: dict[str, dict[str, Any]] = {}
        for kind in BATCH_KINDS:
            path = temporary / f"{kind}.jsonl.gz"
            files[kind] = {
                "file": path.name,
                "rows": counts[kind],
                "selected_request_records": selected_records[kind],
                "duration_min_seconds": min(durations[kind]),
                "duration_max_seconds": max(durations[kind]),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        source_config_hashes = sorted(
            {str(record["config_sha256"]) for record in source_runs}
        )
        vllm_commits = sorted(
            {str(record["vllm_git_commit"]) for record in source_runs}
        )
        if len(source_config_hashes) != 1 or len(vllm_commits) != 1:
            raise ValueError("source runs do not share one config and vLLM commit")
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "build_config_sha256": build_config_sha256,
            "source_config_sha256": source_config_hashes[0],
            "source_vllm_commit": vllm_commits[0],
            "selection": {
                "stock": "checkpoint valid_attempt; all executed iterations",
                "targeted": "checkpoint valid_attempt; sample_role=target only",
                "excluded_target_roles": dict(sorted(excluded_target_roles.items())),
            },
            "row_schema": {
                "fields": sorted(ROW_FIELDS),
                "selected_request_fields": sorted(REQUEST_FIELDS),
                "label": "actual_duration_seconds",
                "feature_policy": (
                    "derive features later only from pre-execution selected_requests; "
                    "identity/source fields are audit-only"
                ),
                "forbidden_fields": sorted(FORBIDDEN_PROFILE_FIELDS),
            },
            "counts": {
                "total": sum(counts.values()),
                "by_batch_kind": dict(sorted(counts.items())),
                "by_source_kind": dict(sorted(origins.items())),
            },
            "files": files,
            "sources": {
                "stock_campaign_manifest_sha256": sha256_file(
                    stock_root / "campaign_manifest.json"
                ),
                "stock_checkpoint_sha256": sha256_file(
                    stock_root / "campaign_checkpoint.json"
                ),
                "target_campaign_manifest_sha256": sha256_file(
                    target_root / "campaign_manifest.json"
                ),
                "target_checkpoint_sha256": sha256_file(
                    target_root / "campaign_checkpoint.json"
                ),
                "runs": source_runs,
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


def validate_dataset(path: Path) -> dict[str, Any]:
    manifest = _json(path / "dataset_manifest.json")
    if (
        manifest.get("schema_version") != DATASET_SCHEMA_VERSION
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("status") != "complete"
    ):
        raise ValueError("dataset manifest identity/status mismatch")
    counts: Counter[str] = Counter()
    for kind in BATCH_KINDS:
        counts[kind] = 0
    selected_records: Counter[str] = Counter()
    identities: set[tuple[str, int]] = set()
    snapshots: set[str] = set()
    for kind in BATCH_KINDS:
        record = manifest["files"].get(kind)
        if not isinstance(record, dict):
            raise ValueError(f"dataset manifest is missing {kind}")
        file_path = path / str(record["file"])
        if file_path.parent != path or file_path.suffixes[-2:] != [".jsonl", ".gz"]:
            raise ValueError("dataset file path is invalid")
        if sha256_file(file_path) != record.get("sha256"):
            raise ValueError(f"dataset file hash mismatch: {kind}")
        with gzip.open(file_path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != ROW_FIELDS:
                    raise ValueError(f"dataset row schema mismatch: {kind}:{line_number}")
                if row.get("schema_version") != DATASET_SCHEMA_VERSION:
                    raise ValueError("dataset row version mismatch")
                if row.get("batch_kind") != kind:
                    raise ValueError("dataset row stored in the wrong batch-kind file")
                if row.get("source_kind") not in {"stock", "targeted"}:
                    raise ValueError("dataset source kind is invalid")
                observed_kind = _validate_source_row(
                    row,
                    expected_run_id=str(row["run_id"]),
                    max_tokens=2**31 - 1,
                    max_sequences=2**31 - 1,
                )
                if observed_kind != kind:
                    raise ValueError("dataset row batch kind mismatch")
                identity = (str(row["run_id"]), int(row["iteration_index"]))
                snapshot = str(row["snapshot_hash"])
                if identity in identities or snapshot in snapshots:
                    raise ValueError("dataset contains duplicate iteration identity")
                identities.add(identity)
                snapshots.add(snapshot)
                counts[kind] += 1
                selected_records[kind] += len(row["selected_requests"])
        if counts[kind] != int(record.get("rows", -1)):
            raise ValueError(f"dataset row count mismatch: {kind}")
        if selected_records[kind] != int(record.get("selected_request_records", -1)):
            raise ValueError(f"selected request count mismatch: {kind}")
    expected = manifest.get("counts", {}).get("by_batch_kind")
    if dict(sorted(counts.items())) != expected:
        raise ValueError("dataset aggregate counts mismatch")
    if sum(counts.values()) != int(manifest.get("counts", {}).get("total", -1)):
        raise ValueError("dataset total row count mismatch")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "valid": True,
        "total_rows": sum(counts.values()),
        "batch_kind_counts": dict(sorted(counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "status"))
    parser.add_argument("--config", default=str(ACTIVE_CONFIG_RELATIVE))
    args = parser.parse_args()
    runtime = load_active_runtime(args.config)
    processed_root = resolve_under(
        runtime.processed_results, DATASET_ID, label="Predictor dataset"
    )
    if args.command == "status":
        if not processed_root.exists():
            print(json.dumps({"dataset_id": DATASET_ID, "status": "absent"}))
            return 0
        print(json.dumps(_json(processed_root / "dataset_manifest.json"), indent=2))
        return 0
    if args.command == "build":
        manifest = build_dataset(
            stock_root=resolve_under(
                runtime.raw_results, STOCK_CAMPAIGN_ID, label="Stock profile campaign"
            ),
            target_root=resolve_under(
                runtime.raw_results, TARGET_CAMPAIGN_ID, label="target profile campaign"
            ),
            output_root=processed_root,
            max_tokens=runtime.max_num_batched_tokens,
            max_sequences=runtime.max_num_seqs,
            build_config_sha256=runtime.config_sha256,
        )
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    validation = validate_dataset(processed_root)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
