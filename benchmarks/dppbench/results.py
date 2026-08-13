from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmarks.dppbench.config import (
    compatible_config_hashes,
    config_hash,
    workspace_path,
)


_COMPLETE_STATUSES = {"complete", "complete_with_warnings"}


def has_saturation_stream_warning(record: dict[str, Any]) -> bool:
    """Return whether only G1 saturation token-stream exactness failed.

    Saturation runs establish completed-request throughput. A coalesced SSE
    delta makes token-level latency for that run inexact, but it does not
    invalidate throughput when every other run-validity check passed.
    """
    metadata = record["metadata"]
    spec = metadata.get("run_spec", {})
    validity = metadata.get("validity", {})
    checks = validity.get("checks", {})
    return (
        metadata.get("status") in _COMPLETE_STATUSES
        and record.get("result") is not None
        and metadata.get("client_exit_code") in (0, 2)
        and spec.get("stage") == "g1"
        and spec.get("mode") == "saturation"
        and checks.get("single_token_stream_chunks") is False
        and all(
            passed
            for name, passed in checks.items()
            if name != "single_token_stream_chunks"
        )
    )


def run_record_usable(record: dict[str, Any]) -> bool:
    metadata = record["metadata"]
    if metadata.get("status") not in _COMPLETE_STATUSES:
        return False
    if record.get("result") is None:
        return False
    return (
        metadata.get("validity", {}).get("valid") is True
        or has_saturation_stream_warning(record)
    )


def record_matches_config(
    config: dict[str, Any], stage: str, record: dict[str, Any]
) -> bool:
    """Match the current config or an explicitly compatible G1 baseline.

    Compatibility is deliberately narrow: only unchanged Stock-Auto serial
    and saturation runs with the current request counts may be imported. Old
    low-load runs can therefore never calibrate the amended SLO.
    """
    metadata = record["metadata"]
    if metadata.get("config_sha256") == config_hash(config):
        return True
    if stage != "g1" or metadata.get("config_sha256") not in compatible_config_hashes(
        config, "g1_baseline_config_sha256"
    ):
        return False
    spec = metadata.get("run_spec", {})
    mode = spec.get("mode")
    expected_limit = {
        "serial": int(config["statistics"]["serial_measurement_requests"]),
        "saturation": int(
            config["statistics"]["saturation_measurement_requests"]
        ),
    }.get(mode)
    return (
        spec.get("stage") == "g1"
        and spec.get("policy") == "stock_auto"
        and spec.get("scenario") in config["g1"]["scenarios"]
        and expected_limit is not None
        and int(spec.get("request_limit", -1)) == expected_limit
    )


def iter_run_records(
    config: dict[str, Any], stage: str | None = None
) -> Iterator[dict[str, Any]]:
    root = workspace_path(config, "raw_results")
    if not root.exists():
        return
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if stage is not None and metadata.get("run_spec", {}).get("stage") != stage:
            continue
        result_path = metadata_path.parent / "client_result.json"
        result = None
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        yield {
            "run_dir": metadata_path.parent,
            "metadata": metadata,
            "result": result,
        }


def completed_run_keys(config: dict[str, Any], stage: str) -> set[str]:
    return {
        record["metadata"]["run_key"]
        for record in iter_run_records(config, stage)
        if run_record_usable(record) and record_matches_config(config, stage, record)
    }


def run_attempt_counts(config: dict[str, Any], stage: str) -> dict[str, int]:
    """Count persisted attempts so --resume cannot reset the retry budget."""
    counts: dict[str, int] = {}
    expected_hash = config_hash(config)
    for record in iter_run_records(config, stage):
        metadata = record["metadata"]
        if metadata.get("config_sha256") != expected_hash:
            continue
        run_key = metadata.get("run_key")
        if run_key:
            counts[run_key] = counts.get(run_key, 0) + 1
    return counts


def load_processed(config: dict[str, Any], stage: str) -> dict[str, Any] | None:
    path = workspace_path(config, "processed_results") / stage / "derived.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("config_sha256") != config_hash(config):
        return None
    return value
