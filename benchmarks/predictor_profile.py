"""Shared contracts and validators for Predictor profiling collection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
CAMPAIGN_ID = "predictor_profile_stock_n500_v1"
TMUX_SESSION = "predictor-profile-stock-n500-v1"
FORMAL_REQUEST_COUNT = 500
REQUEST_TIMEOUT_SECONDS = 5400.0
RUN_TIMEOUT_SECONDS = 7200.0
CAMPAIGN_TIMEOUT_SECONDS = 30 * 60 * 60
MAX_ATTEMPTS_PER_INVOCATION = 2

FORBIDDEN_PROFILE_FIELDS = frozenset(
    {
        "output_tokens",
        "expected_output_tokens",
        "remaining_output_tokens",
        "eventual_eos_position",
        "target_output_tokens",
        "client_safety_ceiling_tokens",
        "max_tokens",
    }
)

ITERATION_LOG_PATTERN = re.compile(
    r"Iteration\((?P<index>\d+)\):.*?iteration elapsed time:\s*"
    r"(?P<elapsed>[0-9]+(?:\.[0-9]+)?)\s*ms"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CampaignRun:
    qps: float
    seed: int

    @property
    def key(self) -> str:
        return f"qps_{qps_tag(self.qps)}_seed_{self.seed}"

    @property
    def trace_filename(self) -> str:
        return f"qps_{qps_text(self.qps)}_seed_{self.seed}.jsonl"


CAMPAIGN_MATRIX = (
    CampaignRun(0.2, 1001),
    CampaignRun(0.2, 1002),
    CampaignRun(0.25, 1003),
    CampaignRun(0.25, 1004),
    CampaignRun(0.3, 1005),
    CampaignRun(0.3, 1006),
    CampaignRun(0.5, 1007),
    CampaignRun(0.5, 1008),
    CampaignRun(1.0, 1009),
    CampaignRun(1.0, 1010),
    CampaignRun(2.0, 1011),
    CampaignRun(2.0, 1012),
)


def qps_text(qps: float) -> str:
    return str(float(qps))


def qps_tag(qps: float) -> str:
    return qps_text(qps).replace(".", "p")


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


def load_scheduled_batches(
    path: Path,
    *,
    expected_run_id: str,
    allowed_plan_prefixes: tuple[str, ...] = ("stock-",),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = _forbidden_keys(row)
            if forbidden:
                raise ValueError(
                    f"scheduled batch line {line_number} has forbidden fields: "
                    f"{sorted(forbidden)}"
                )
            if row.get("schema_version") != PROFILE_SCHEMA_VERSION:
                raise ValueError(f"scheduled batch line {line_number} schema mismatch")
            if row.get("run_id") != expected_run_id:
                raise ValueError(f"scheduled batch line {line_number} run_id mismatch")
            index = int(row.get("iteration_index", -1))
            if index < 0 or index in seen_indices:
                raise ValueError(
                    f"scheduled batch line {line_number} has invalid/duplicate index"
                )
            seen_indices.add(index)
            if not SHA256_PATTERN.fullmatch(str(row.get("snapshot_hash", ""))):
                raise ValueError(f"scheduled batch line {line_number} snapshot hash invalid")
            if not str(row.get("plan_id", "")).startswith(allowed_plan_prefixes):
                raise ValueError(f"scheduled batch line {line_number} plan_id invalid")
            selected = row.get("selected_requests")
            if not isinstance(selected, list) or not selected:
                raise ValueError(f"scheduled batch line {line_number} has no requests")
            request_ids: set[str] = set()
            for request in selected:
                request_id = str(request.get("request_id", ""))
                if not request_id or request_id in request_ids:
                    raise ValueError(
                        f"scheduled batch line {line_number} request IDs invalid"
                    )
                request_ids.add(request_id)
                if request.get("phase") not in {"prefill", "decode"}:
                    raise ValueError(f"scheduled batch line {line_number} phase invalid")
                context = int(request.get("current_context_tokens", -1))
                scheduled = int(request.get("scheduled_tokens", 0))
                if context < 0 or scheduled <= 0:
                    raise ValueError(
                        f"scheduled batch line {line_number} token counts invalid"
                    )
            rows.append(row)
    if not rows:
        raise ValueError("scheduled batch profile is empty")
    rows.sort(key=lambda row: int(row["iteration_index"]))
    expected_indices = list(range(len(rows)))
    if [int(row["iteration_index"]) for row in rows] != expected_indices:
        raise ValueError("scheduled batch iteration indices are not contiguous from zero")
    return rows


def parse_iteration_durations(path: Path) -> dict[int, float]:
    durations: dict[int, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = ITERATION_LOG_PATTERN.search(line)
            if match is None:
                continue
            index = int(match.group("index"))
            elapsed_seconds = float(match.group("elapsed")) / 1000.0
            if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
                raise ValueError(f"iteration {index} duration is not positive")
            if index in durations:
                raise ValueError(f"duplicate duration for iteration {index}")
            durations[index] = elapsed_seconds
    if not durations:
        raise ValueError("startup log contains no iteration durations")
    return durations


def merge_iteration_profiles(
    *,
    scheduled_batches_path: Path,
    startup_log_path: Path,
    output_path: Path,
    expected_run_id: str,
    allowed_plan_prefixes: tuple[str, ...] = ("stock-",),
) -> dict[str, Any]:
    batches = load_scheduled_batches(
        scheduled_batches_path,
        expected_run_id=expected_run_id,
        allowed_plan_prefixes=allowed_plan_prefixes,
    )
    durations = parse_iteration_durations(startup_log_path)
    batch_indices = {int(row["iteration_index"]) for row in batches}
    duration_indices = set(durations)
    if batch_indices != duration_indices:
        raise ValueError(
            "scheduled batch/duration iteration mismatch: "
            f"missing_durations={sorted(batch_indices - duration_indices)}, "
            f"missing_batches={sorted(duration_indices - batch_indices)}"
        )

    with output_path.open("x", encoding="utf-8") as stream:
        for batch in batches:
            index = int(batch["iteration_index"])
            row = dict(batch)
            row["actual_duration_seconds"] = durations[index]
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "valid": True,
        "run_id": expected_run_id,
        "iteration_count": len(batches),
        "selected_request_records": sum(
            len(row["selected_requests"]) for row in batches
        ),
        "duration_min_seconds": min(durations.values()),
        "duration_max_seconds": max(durations.values()),
    }


def validate_run_directory(
    path: Path,
    *,
    expected_run: CampaignRun,
    expected_request_count: int,
) -> dict[str, Any]:
    with (path / "run_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (path / "summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with (path / "profile_validation.json").open("r", encoding="utf-8") as stream:
        profile = json.load(stream)

    if manifest.get("status") != "complete":
        raise ValueError(f"run manifest is not complete: {path}")
    if float(manifest.get("qps")) != expected_run.qps:
        raise ValueError(f"run qps mismatch: {path}")
    if int(manifest.get("seed")) != expected_run.seed:
        raise ValueError(f"run seed mismatch: {path}")
    if summary.get("num_requests") != expected_request_count:
        raise ValueError(f"run request count mismatch: {path}")
    if summary.get("completed") != expected_request_count or summary.get("failed") != 0:
        raise ValueError(f"run has failed requests: {path}")
    if profile.get("valid") is not True or int(profile.get("iteration_count", 0)) <= 0:
        raise ValueError(f"run iteration profile is invalid: {path}")

    per_request_count = 0
    with (path / "per_request.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                per_request_count += 1
    if per_request_count != expected_request_count:
        raise ValueError(f"per-request row count mismatch: {path}")

    profile_rows = load_scheduled_batches(
        path / "iteration_profile.jsonl",
        expected_run_id=str(manifest["run_id"]),
    )
    for row in profile_rows:
        duration = float(row.get("actual_duration_seconds", 0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"profile duration invalid: {path}")
    if len(profile_rows) != int(profile["iteration_count"]):
        raise ValueError(f"profile row count mismatch: {path}")
    return {
        "valid": True,
        "run_id": manifest["run_id"],
        "request_count": per_request_count,
        "iteration_count": len(profile_rows),
    }
