"""Contracts and fail-closed validation for the six-run development comparison."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmarks.qwen3_runtime import (
    MODULAR_DPP_SCHEDULER_CLASS,
    ActiveRuntime,
)
from benchmarks.run_stock_natural_eos import load_trace, verify_trace_manifest


SCHEMA_VERSION = 1
POLICIES = ("stock", "dpp")
FORBIDDEN_LOG_MARKERS = (
    "Traceback (most recent call last)",
    "EngineDeadError",
    "Python.h: No such file or directory",
)


def qps_text(qps: float) -> str:
    return str(float(qps))


def qps_tag(qps: float) -> str:
    return qps_text(qps).replace(".", "p")


@dataclass(frozen=True)
class ComparisonRun:
    policy: str
    qps: float
    seed: int

    @property
    def key(self) -> str:
        return f"{self.policy}_qps_{qps_tag(self.qps)}_seed_{self.seed}"

    @property
    def pair_key(self) -> str:
        return f"qps_{qps_tag(self.qps)}_seed_{self.seed}"

    @property
    def trace_filename(self) -> str:
        return f"qps_{qps_text(self.qps)}_seed_{self.seed}.jsonl"


@dataclass(frozen=True)
class ComparisonSettings:
    campaign_id: str
    tmux_session: str
    formal_benchmark_eligible: bool
    policies: tuple[str, ...]
    qps_values: tuple[float, ...]
    seed: int
    request_count: int
    smoke_request_count: int
    smoke_qps: float
    run_order: tuple[tuple[float, tuple[str, ...]], ...]
    request_timeout_seconds: float
    run_timeout_seconds: float
    campaign_timeout_seconds: float
    max_attempts_per_invocation: int

    @property
    def matrix(self) -> tuple[ComparisonRun, ...]:
        return tuple(
            ComparisonRun(policy=policy, qps=qps, seed=self.seed)
            for qps, policies in self.run_order
            for policy in policies
        )

    @property
    def smoke_matrix(self) -> tuple[ComparisonRun, ...]:
        return tuple(
            ComparisonRun(policy=policy, qps=self.smoke_qps, seed=self.seed)
            for policy in self.policies
        )


def load_comparison_settings(runtime: ActiveRuntime) -> ComparisonSettings:
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    experiments = config.get("experiments") if isinstance(config, dict) else None
    raw = experiments.get("development_comparison") if isinstance(experiments, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("experiments.development_comparison is missing")

    policies = tuple(str(value) for value in raw.get("policies", ()))
    qps_values = tuple(
        float(value) for value in raw.get("offered_loads_requests_per_second", ())
    )
    run_order_raw = raw.get("run_order")
    if not isinstance(run_order_raw, list):
        raise ValueError("development comparison run_order must be a list")
    run_order: list[tuple[float, tuple[str, ...]]] = []
    for item in run_order_raw:
        if not isinstance(item, dict):
            raise ValueError("development comparison run_order entry must be a mapping")
        run_order.append(
            (
                float(item.get("qps")),
                tuple(str(value) for value in item.get("policies", ())),
            )
        )

    settings = ComparisonSettings(
        campaign_id=str(raw.get("campaign_id", "")),
        tmux_session=str(raw.get("tmux_session", "")),
        formal_benchmark_eligible=bool(raw.get("formal_benchmark_eligible")),
        policies=policies,
        qps_values=qps_values,
        seed=int(raw.get("trace_seed", -1)),
        request_count=int(raw.get("request_count_per_run", 0)),
        smoke_request_count=int(raw.get("smoke_request_count", 0)),
        smoke_qps=float(raw.get("smoke_qps", math.nan)),
        run_order=tuple(run_order),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", math.nan)),
        run_timeout_seconds=float(raw.get("run_timeout_seconds", math.nan)),
        campaign_timeout_seconds=float(raw.get("campaign_timeout_seconds", math.nan)),
        max_attempts_per_invocation=int(raw.get("max_attempts_per_invocation", 0)),
    )
    if raw.get("status") != "approved_for_nonformal_engineering_comparison":
        raise ValueError("development comparison status is not approved")
    if settings.formal_benchmark_eligible:
        raise ValueError("development comparison must not be formal-benchmark eligible")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", settings.campaign_id):
        raise ValueError("development comparison campaign_id is invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", settings.tmux_session):
        raise ValueError("development comparison tmux_session is invalid")
    if settings.policies != POLICIES:
        raise ValueError(f"development comparison policies must be {POLICIES!r}")
    if len(settings.qps_values) != 3 or len(set(settings.qps_values)) != 3:
        raise ValueError("development comparison requires three unique QPS values")
    if any(not math.isfinite(value) or value <= 0 for value in settings.qps_values):
        raise ValueError("development comparison QPS values must be finite and positive")
    if settings.seed < 0:
        raise ValueError("development comparison seed must be nonnegative")
    if settings.request_count != 300:
        raise ValueError("development comparison must contain exactly 300 requests per run")
    if not 1 <= settings.smoke_request_count < settings.request_count:
        raise ValueError("development comparison smoke request count is invalid")
    if settings.smoke_qps not in settings.qps_values:
        raise ValueError("development comparison smoke QPS is not in the matrix")
    if tuple(qps for qps, _ in settings.run_order) != settings.qps_values:
        raise ValueError("development comparison run order QPS values do not match")
    if any(tuple(sorted(order)) != tuple(sorted(POLICIES)) for _, order in settings.run_order):
        raise ValueError("each development comparison pair must contain Stock and DPP once")
    if len(settings.matrix) != 6:
        raise ValueError("development comparison matrix must contain exactly six runs")
    timeouts = (
        settings.request_timeout_seconds,
        settings.run_timeout_seconds,
        settings.campaign_timeout_seconds,
    )
    if any(not math.isfinite(value) or value <= 0 for value in timeouts):
        raise ValueError("development comparison timeouts must be finite and positive")
    if settings.run_timeout_seconds <= settings.request_timeout_seconds:
        raise ValueError("run timeout must exceed the per-request timeout")
    if settings.campaign_timeout_seconds <= settings.run_timeout_seconds:
        raise ValueError("campaign timeout must exceed a single-run timeout")
    if not 1 <= settings.max_attempts_per_invocation <= 2:
        raise ValueError("max attempts per invocation must be in [1, 2]")
    return settings


def validate_trace_directory(
    runtime: ActiveRuntime, trace_dir: Path, settings: ComparisonSettings
) -> dict[str, Any]:
    manifest_path = trace_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("kind") != "qwen3_14b_poisson_length_blind_traces":
        raise ValueError("comparison trace kind mismatch")
    if manifest.get("pairing") != "explicit":
        raise ValueError("comparison traces must use explicit QPS/seed pairs")
    if int(manifest.get("num_requests_per_trace", 0)) != settings.request_count:
        raise ValueError("comparison trace request count mismatch")
    if manifest.get("predetermined_output_length") is not False:
        raise ValueError("comparison trace exposes predetermined output length")

    expected = {
        (ComparisonRun("stock", qps, settings.seed).trace_filename, qps, settings.seed)
        for qps in settings.qps_values
    }
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("comparison trace manifest files are missing")
    observed = {
        (str(item.get("file")), float(item.get("requested_qps")), int(item.get("seed")))
        for item in entries
    }
    if observed != expected or len(entries) != len(expected):
        raise ValueError("comparison trace matrix mismatch")

    validations: list[dict[str, Any]] = []
    for qps in settings.qps_values:
        trace_name = ComparisonRun("stock", qps, settings.seed).trace_filename
        trace_path = trace_dir / trace_name
        rows = load_trace(trace_path, runtime)
        if len(rows) != settings.request_count:
            raise ValueError(f"comparison trace row count mismatch: {trace_name}")
        verify_trace_manifest(trace_path, manifest_path, runtime)
        validations.append(
            {
                "trace": trace_name,
                "qps": qps,
                "seed": settings.seed,
                "request_count": len(rows),
                "arrival_span_s": float(rows[-1]["arrival_time_s"]),
            }
        )
    return {"valid": True, "traces": validations}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def validate_run_directory(
    run_root: Path,
    *,
    expected_run: ComparisonRun,
    expected_request_count: int,
    smoke: bool,
    expected_source_request_count: int = 300,
) -> dict[str, Any]:
    manifest = _read_json(run_root / "run_manifest.json")
    if manifest.get("schema_version") != 2:
        raise ValueError(f"run manifest schema mismatch: {run_root}")
    if manifest.get("run_id") != run_root.name:
        raise ValueError(f"run ID mismatch: {run_root}")
    if manifest.get("scheduler_policy") != expected_run.policy:
        raise ValueError(f"run policy mismatch: {run_root}")
    if manifest.get("status") != "complete":
        raise ValueError(f"run is not complete: {run_root}")
    if bool(manifest.get("comparison_eligible")) == smoke:
        raise ValueError(f"run comparison eligibility mismatch: {run_root}")
    if manifest.get("formal_comparison_eligible") is not False:
        raise ValueError(f"development run became formal eligible: {run_root}")

    resolved = manifest.get("resolved")
    summary = manifest.get("summary")
    if not isinstance(resolved, dict) or not isinstance(summary, dict):
        raise ValueError(f"run resolved/summary payload is missing: {run_root}")
    if resolved.get("comparison_scope") != "development_nonformal":
        raise ValueError(f"run comparison scope mismatch: {run_root}")
    if resolved.get("scheduler_policy") != expected_run.policy:
        raise ValueError(f"resolved policy mismatch: {run_root}")
    if int(resolved.get("request_count", 0)) != expected_request_count:
        raise ValueError(f"resolved request count mismatch: {run_root}")
    if int(resolved.get("source_request_count", 0)) != expected_source_request_count:
        raise ValueError(f"resolved source request count mismatch: {run_root}")
    if bool(resolved.get("diagnostic_prefix")) != smoke:
        raise ValueError(f"resolved diagnostic-prefix mismatch: {run_root}")
    if bool(resolved.get("dpp_diagnostic_iteration_log")):
        raise ValueError(f"comparison run enabled detailed iteration logging: {run_root}")

    if int(summary.get("num_requests", 0)) != expected_request_count:
        raise ValueError(f"summary request count mismatch: {run_root}")
    if int(summary.get("completed", -1)) != expected_request_count:
        raise ValueError(f"run has incomplete requests: {run_root}")
    if int(summary.get("failed", -1)) != 0:
        raise ValueError(f"run has failed requests: {run_root}")
    if int(summary.get("input_token_mismatches", -1)) != 0:
        raise ValueError(f"run has input-token mismatches: {run_root}")
    if int(summary.get("stream_token_count_mismatches", -1)) != 0:
        raise ValueError(f"run has stream-token mismatches: {run_root}")

    rows = _read_jsonl(run_root / "per_request.jsonl")
    if len(rows) != expected_request_count:
        raise ValueError(f"per-request row count mismatch: {run_root}")
    for row in rows:
        if row.get("completed") is not True or row.get("error") is not None:
            raise ValueError(f"per-request failure in {run_root}")
        if int(row.get("http_status", 0)) != 200:
            raise ValueError(f"non-200 response in {run_root}")
        if row.get("finish_reason") not in {"stop", "length"}:
            raise ValueError(f"invalid finish reason in {run_root}")
        if int(row.get("actual_output_tokens", -1)) != int(
            row.get("observed_stream_tokens", -2)
        ):
            raise ValueError(f"stream token mismatch in {run_root}")

    startup_log = (run_root / "startup.log").read_text(encoding="utf-8", errors="replace")
    found = [marker for marker in FORBIDDEN_LOG_MARKERS if marker in startup_log]
    if re.search(r"(?m)^.*\bERROR\b", startup_log):
        found.append("ERROR log line")
    if found:
        raise ValueError(f"forbidden server log markers in {run_root}: {found}")
    return {
        "valid": True,
        "run_id": run_root.name,
        "policy": expected_run.policy,
        "qps": expected_run.qps,
        "seed": expected_run.seed,
        "request_count": expected_request_count,
        "trace_sha256": resolved.get("trace_sha256"),
        "finish_reason_counts": summary.get("finish_reason_counts"),
    }


def _request_identity(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row.get("request_id"),
            row.get("prompt_id"),
            row.get("planned_arrival_s"),
            row.get("generation_seed"),
            row.get("client_safety_ceiling_tokens"),
            row.get("input_tokens"),
        )
        for row in rows
    ]


def validate_pair(stock_root: Path, dpp_root: Path, *, smoke: bool) -> dict[str, Any]:
    stock = _read_json(stock_root / "run_manifest.json")
    dpp = _read_json(dpp_root / "run_manifest.json")
    if stock.get("scheduler_policy") != "stock" or dpp.get("scheduler_policy") != "dpp":
        raise ValueError("comparison pair policy identity mismatch")
    stock_resolved = stock.get("resolved")
    dpp_resolved = dpp.get("resolved")
    if not isinstance(stock_resolved, dict) or not isinstance(dpp_resolved, dict):
        raise ValueError("comparison pair resolved payload is missing")
    identity_fields = (
        "config_sha256",
        "trace",
        "trace_sha256",
        "trace_manifest",
        "trace_manifest_sha256",
        "request_count",
        "source_request_count",
        "diagnostic_prefix",
        "planned_arrival_span_s",
        "comparison_scope",
        "client_safety_ceiling_tokens",
        "scheduler_receives_safety_ceiling",
        "required_env",
    )
    mismatches = [
        field
        for field in identity_fields
        if stock_resolved.get(field) != dpp_resolved.get(field)
    ]
    if mismatches:
        raise ValueError(f"comparison pair resolved mismatch: {mismatches}")
    if stock.get("git") != dpp.get("git"):
        raise ValueError("comparison pair Git state mismatch")

    stock_command = stock_resolved.get("server_command")
    dpp_command = dpp_resolved.get("server_command")
    if not isinstance(stock_command, list) or not isinstance(dpp_command, list):
        raise ValueError("comparison pair server command is missing")
    expected_suffix = ["--scheduler-cls", MODULAR_DPP_SCHEDULER_CLASS]
    if dpp_command != stock_command + expected_suffix:
        raise ValueError("DPP server command is not Stock plus the Scheduler class")
    if "--scheduler-cls" in stock_command:
        raise ValueError("Stock comparison command overrides the Scheduler class")

    stock_rows = _read_jsonl(stock_root / "per_request.jsonl")
    dpp_rows = _read_jsonl(dpp_root / "per_request.jsonl")
    if _request_identity(stock_rows) != _request_identity(dpp_rows):
        raise ValueError("comparison pair request identity mismatch")
    if bool(stock.get("comparison_eligible")) == smoke:
        raise ValueError("comparison pair smoke/main eligibility mismatch")
    return {
        "valid": True,
        "smoke": smoke,
        "request_count": len(stock_rows),
        "trace_sha256": stock_resolved.get("trace_sha256"),
        "stock_run_id": stock.get("run_id"),
        "dpp_run_id": dpp.get("run_id"),
    }


def _metrics(run_root: Path, *, ttft_slo_ms: float, tbt_slo_ms: float) -> dict[str, Any]:
    manifest = _read_json(run_root / "run_manifest.json")
    summary = manifest["summary"]
    rows = _read_jsonl(run_root / "per_request.jsonl")
    slo_success = 0
    for row in rows:
        ttft = row.get("ttft_ms")
        itls = row.get("itls_ms")
        success = bool(
            row.get("completed")
            and row.get("finish_reason") == "stop"
            and row.get("token_timing_exact")
            and ttft is not None
            and float(ttft) <= ttft_slo_ms
            and isinstance(itls, list)
            and all(float(value) <= tbt_slo_ms for value in itls)
        )
        slo_success += int(success)
    elapsed = float(summary["elapsed_s"])
    return {
        "run_id": manifest["run_id"],
        "policy": manifest["scheduler_policy"],
        "request_count": len(rows),
        "completed": int(summary["completed"]),
        "failed": int(summary["failed"]),
        "finish_reason_counts": summary["finish_reason_counts"],
        "natural_eos_slo_success_requests": slo_success,
        "natural_eos_slo_success_rate": slo_success / len(rows),
        "natural_eos_slo_goodput_rps": slo_success / elapsed if elapsed > 0 else None,
        "completion_throughput_rps": summary["completion_throughput_rps"],
        "elapsed_s": elapsed,
        "ttft_ms": summary["ttft_ms"],
        "tbt_ms_exact_requests_only": summary["tbt_ms_exact_requests_only"],
        "e2e_ms": summary["e2e_ms"],
        "output_tokens": summary["output_tokens"],
    }


def build_report(
    pair_roots: list[tuple[float, Path, Path]],
    *,
    ttft_slo_seconds: float,
    tbt_slo_seconds: float,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for qps, stock_root, dpp_root in pair_roots:
        stock = _metrics(
            stock_root,
            ttft_slo_ms=ttft_slo_seconds * 1000,
            tbt_slo_ms=tbt_slo_seconds * 1000,
        )
        dpp = _metrics(
            dpp_root,
            ttft_slo_ms=ttft_slo_seconds * 1000,
            tbt_slo_ms=tbt_slo_seconds * 1000,
        )
        pairs.append(
            {
                "qps": qps,
                "stock": stock,
                "dpp": dpp,
                "paired_difference_dpp_minus_stock": {
                    "natural_eos_slo_success_requests": (
                        dpp["natural_eos_slo_success_requests"]
                        - stock["natural_eos_slo_success_requests"]
                    ),
                    "natural_eos_slo_goodput_rps": (
                        dpp["natural_eos_slo_goodput_rps"]
                        - stock["natural_eos_slo_goodput_rps"]
                    ),
                    "completion_throughput_rps": (
                        dpp["completion_throughput_rps"]
                        - stock["completion_throughput_rps"]
                    ),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "slo_seconds": {"ttft": ttft_slo_seconds, "tbt": tbt_slo_seconds},
        "aggregation_warning": (
            "single-seed development results have no cross-seed confidence interval"
        ),
        "pairs": pairs,
    }
