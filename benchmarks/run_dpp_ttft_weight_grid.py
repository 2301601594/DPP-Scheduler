#!/usr/bin/env python3
"""Run the fixed nine-run TTFT-drift-weight development campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from benchmarks.qwen3_runtime import (
    ActiveRuntime,
    load_active_runtime,
    load_obligation_settings,
    require_frozen_for_execution,
    resolve_under,
)
from benchmarks.run_stock_natural_eos import _atomic_json, _git_state
from benchmarks.scheduler_comparison import (
    ComparisonRun,
    _metrics,
    validate_pair,
    validate_run_directory,
)


CHECKPOINT_NAME = "campaign_checkpoint.json"
MANIFEST_NAME = "campaign_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


@dataclass(frozen=True)
class GridSettings:
    campaign_id: str
    tmux_session: str
    qps: float
    seed: int
    request_count: int
    weights: tuple[float, ...]
    request_timeout_seconds: float
    run_timeout_seconds: float
    campaign_timeout_seconds: float
    max_attempts: int
    smoke_campaign_id: str
    smoke_tmux_session: str
    smoke_request_count: int
    smoke_request_timeout_seconds: float
    smoke_run_timeout_seconds: float
    smoke_campaign_timeout_seconds: float
    is_smoke: bool = False


@dataclass(frozen=True)
class GridRun:
    key: str
    policy: str
    selection_mode: str = "normal"
    weight: float | None = None


def load_settings(runtime: ActiveRuntime) -> GridSettings:
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    raw = config.get("experiments", {}).get("ttft_weight_grid")
    if not isinstance(raw, dict):
        raise ValueError("experiments.ttft_weight_grid is missing")
    settings = GridSettings(
        campaign_id=str(raw.get("campaign_id", "")),
        tmux_session=str(raw.get("tmux_session", "")),
        qps=float(raw.get("offered_load_requests_per_second", math.nan)),
        seed=int(raw.get("trace_seed", -1)),
        request_count=int(raw.get("request_count_per_run", 0)),
        weights=tuple(float(value) for value in raw.get("ttft_drift_weights", ())),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", math.nan)),
        run_timeout_seconds=float(raw.get("run_timeout_seconds", math.nan)),
        campaign_timeout_seconds=float(raw.get("campaign_timeout_seconds", math.nan)),
        max_attempts=int(raw.get("max_attempts_per_invocation", 0)),
        smoke_campaign_id=str(raw.get("smoke_campaign_id", "")),
        smoke_tmux_session=str(raw.get("smoke_tmux_session", "")),
        smoke_request_count=int(raw.get("smoke_request_count", 0)),
        smoke_request_timeout_seconds=float(
            raw.get("smoke_request_timeout_seconds", math.nan)
        ),
        smoke_run_timeout_seconds=float(
            raw.get("smoke_run_timeout_seconds", math.nan)
        ),
        smoke_campaign_timeout_seconds=float(
            raw.get("smoke_campaign_timeout_seconds", math.nan)
        ),
    )
    if raw.get("status") != "approved_for_nonformal_engineering_comparison":
        raise ValueError("TTFT grid is not approved for development execution")
    if raw.get("formal_benchmark_eligible") is not False:
        raise ValueError("TTFT grid must remain non-formal")
    if raw.get("shared_stock_run") is not True:
        raise ValueError("TTFT grid must use one shared Stock run")
    if (
        settings.qps != 0.25
        or settings.seed != 1001
        or settings.request_count != 150
        or settings.weights != (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    ):
        raise ValueError("TTFT grid QPS/seed/request-count/weights contract mismatch")
    if settings.max_attempts not in {1, 2}:
        raise ValueError("TTFT grid max_attempts must be one or two")
    if (
        not settings.smoke_campaign_id
        or not settings.smoke_tmux_session
        or settings.smoke_request_count != 1
    ):
        raise ValueError("TTFT grid smoke identity/request-count mismatch")
    for timeout in (
        settings.request_timeout_seconds,
        settings.run_timeout_seconds,
        settings.campaign_timeout_seconds,
        settings.smoke_request_timeout_seconds,
        settings.smoke_run_timeout_seconds,
        settings.smoke_campaign_timeout_seconds,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TTFT grid timeouts must be finite and positive")
    return settings


def as_smoke(settings: GridSettings) -> GridSettings:
    """Return the isolated three-path, one-request smoke campaign settings."""
    return replace(
        settings,
        campaign_id=settings.smoke_campaign_id,
        tmux_session=settings.smoke_tmux_session,
        request_count=settings.smoke_request_count,
        weights=(1.0,),
        request_timeout_seconds=settings.smoke_request_timeout_seconds,
        run_timeout_seconds=settings.smoke_run_timeout_seconds,
        campaign_timeout_seconds=settings.smoke_campaign_timeout_seconds,
        is_smoke=True,
    )


def matrix(settings: GridSettings) -> tuple[GridRun, ...]:
    return (
        GridRun("stock", "stock"),
        GridRun(
            "dpp_forced_stock_plan",
            "dpp",
            selection_mode="forced_stock_plan",
        ),
        *(
            GridRun(f"dpp_lambda_{_tag(weight)}", "dpp", weight=weight)
            for weight in settings.weights
        ),
    )


def _root(runtime: ActiveRuntime, settings: GridSettings) -> Path:
    return resolve_under(runtime.raw_results, settings.campaign_id, label="TTFT grid")


def _trace_name(settings: GridSettings) -> str:
    return f"qps_{settings.qps}_seed_{settings.seed}.jsonl"


def preview(runtime: ActiveRuntime, settings: GridSettings) -> dict[str, Any]:
    return {
        "campaign_id": settings.campaign_id,
        "tmux_session": settings.tmux_session,
        "scope": "development_nonformal_single_seed",
        "smoke": settings.is_smoke,
        "formal_benchmark_eligible": False,
        "qps": settings.qps,
        "seed": settings.seed,
        "request_count_per_run": settings.request_count,
        "shared_trace": _trace_name(settings),
        "shared_stock_run": True,
        "run_count": len(matrix(settings)),
        "matrix": [run.__dict__ for run in matrix(settings)],
        "output_dir": str(_root(runtime, settings)),
    }


def _state(settings: GridSettings) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": settings.campaign_id,
        "status": "running",
        "created_at_utc": _utc_now(),
        "trace_status": "pending",
        "trace_directory": None,
        "trace_attempts": [],
        "runs": [
            {
                **run.__dict__,
                "status": "pending",
                "valid_attempt": None,
                "attempts": [],
            }
            for run in matrix(settings)
        ],
    }


def _save(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(root / CHECKPOINT_NAME, state)


def _load(root: Path, settings: GridSettings) -> dict[str, Any]:
    with (root / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("campaign_id") != settings.campaign_id:
        raise ValueError("TTFT grid checkpoint identity mismatch")
    expected = [run.__dict__ for run in matrix(settings)]
    observed = [
        {key: item.get(key) for key in ("key", "policy", "selection_mode", "weight")}
        for item in state.get("runs", [])
    ]
    if observed != expected:
        raise ValueError("TTFT grid checkpoint matrix mismatch")
    return state


def _run_command(command: list[str], *, cwd: Path, log: Path, timeout: float) -> int:
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            raise


def _ensure_trace(
    runtime: ActiveRuntime, settings: GridSettings, root: Path, state: dict[str, Any]
) -> None:
    recorded_directory = state.get("trace_directory")
    if state.get("trace_status") == "valid":
        if not isinstance(recorded_directory, str) or not recorded_directory:
            raise ValueError("recorded TTFT grid trace directory is invalid")
        trace_root = root / recorded_directory
        trace_path = trace_root / _trace_name(settings)
        manifest = trace_root / "manifest.json"
        if not trace_path.is_file() or not manifest.is_file():
            raise FileNotFoundError("recorded TTFT grid trace is missing")
        return

    trace_attempt = len(state.setdefault("trace_attempts", [])) + 1
    trace_directory = f"traces_attempt_{trace_attempt:02d}"
    trace_root = root / trace_directory
    trace_path = trace_root / _trace_name(settings)
    manifest = trace_root / "manifest.json"
    record = {
        "directory": trace_directory,
        "started_at_utc": _utc_now(),
        "status": "running",
    }
    state["trace_attempts"].append(record)
    _save(root, state)
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.generate_qwen3_poisson_traces",
        "--config",
        str(runtime.config_path),
        "--output-dir",
        f"{settings.campaign_id}/{trace_directory}",
        "--num-requests",
        str(settings.request_count),
        "--qps-seed",
        f"{settings.qps}:{settings.seed}",
    ]
    try:
        code = _run_command(
            command,
            cwd=runtime.workspace,
            log=root / f"trace_generation_attempt_{trace_attempt:02d}.log",
            timeout=600,
        )
        record["exit_code"] = code
        if code != 0 or not trace_path.is_file() or not manifest.is_file():
            raise RuntimeError("TTFT grid trace generation failed")
        record["status"] = "valid"
        state["trace_status"] = "valid"
        state["trace_directory"] = trace_directory
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        state["trace_status"] = "failed"
        raise
    finally:
        record["finished_at_utc"] = _utc_now()
        _save(root, state)


def _run_one(
    runtime: ActiveRuntime,
    settings: GridSettings,
    root: Path,
    state: dict[str, Any],
    run: GridRun,
    item: dict[str, Any],
    deadline: float,
) -> bool:
    if item.get("status") == "valid":
        return True
    trace_directory = state.get("trace_directory")
    if not isinstance(trace_directory, str) or not trace_directory:
        raise ValueError("TTFT grid shared trace is not ready")
    for _ in range(settings.max_attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            item["status"] = "failed"
            return False
        attempt = len(item["attempts"]) + 1
        run_id = f"{run.key}_attempt_{attempt:02d}"
        command = [
            str(runtime.python),
            "-m",
            "benchmarks.run_stock_natural_eos",
            "--config",
            str(runtime.config_path),
            "--campaign-id",
            settings.campaign_id,
            "--development-trace-dir",
            f"{settings.campaign_id}/{trace_directory}",
            "--trace",
            _trace_name(settings),
            "--trace-manifest",
            "manifest.json",
            "--run-id",
            run_id,
            "--policy",
            run.policy,
            "--request-timeout",
            str(settings.request_timeout_seconds),
        ]
        if run.policy == "dpp":
            command.extend(["--dpp-selection-mode", run.selection_mode])
            if run.weight is not None:
                command.extend(["--dpp-ttft-drift-weight", str(run.weight)])
        record = {"run_id": run_id, "started_at_utc": _utc_now(), "status": "running"}
        item["attempts"].append(record)
        item["status"] = "running"
        _save(root, state)
        try:
            code = _run_command(
                command,
                cwd=runtime.workspace,
                log=root / "launcher_logs" / f"{run_id}.log",
                timeout=min(settings.run_timeout_seconds, remaining),
            )
            record["exit_code"] = code
            if code != 0:
                raise RuntimeError(f"run exited with {code}")
            run_root = root / "runs" / run_id
            validate_run_directory(
                run_root,
                expected_run=ComparisonRun(run.policy, settings.qps, settings.seed),
                expected_request_count=settings.request_count,
                smoke=False,
                expected_source_request_count=settings.request_count,
            )
            record["status"] = "valid"
            item["status"] = "valid"
            item["valid_attempt"] = run_id
            return True
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            item["status"] = "failed"
        finally:
            record["finished_at_utc"] = _utc_now()
    return False


def _run_root(root: Path, item: dict[str, Any]) -> Path:
    value = item.get("valid_attempt")
    if not value:
        raise ValueError(f"grid run has no valid attempt: {item.get('key')}")
    return root / "runs" / str(value)


def validate_campaign(
    runtime: ActiveRuntime, settings: GridSettings, root: Path
) -> dict[str, Any]:
    state = _load(root, settings)
    if state.get("trace_status") != "valid":
        raise ValueError("TTFT grid trace is not valid")
    by_key = {item["key"]: item for item in state["runs"]}
    stock_root = _run_root(root, by_key["stock"])
    validate_run_directory(
        stock_root,
        expected_run=ComparisonRun("stock", settings.qps, settings.seed),
        expected_request_count=settings.request_count,
        expected_source_request_count=settings.request_count,
        smoke=False,
    )
    comparisons: list[dict[str, Any]] = []
    slo = load_obligation_settings(runtime)
    ttft_slo_ms = slo.ttft_slo_seconds * 1000.0
    tbt_slo_ms = slo.tbt_slo_seconds * 1000.0
    stock_metrics = _metrics(
        stock_root, ttft_slo_ms=ttft_slo_ms, tbt_slo_ms=tbt_slo_ms
    )
    for run in matrix(settings)[1:]:
        item = by_key[run.key]
        run_root = _run_root(root, item)
        validate_run_directory(
            run_root,
            expected_run=ComparisonRun(run.policy, settings.qps, settings.seed),
            expected_request_count=settings.request_count,
            expected_source_request_count=settings.request_count,
            smoke=False,
        )
        validate_pair(stock_root, run_root, smoke=False)
        with (run_root / "run_manifest.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        resolved = manifest.get("resolved", {})
        if resolved.get("dpp_selection_mode") != run.selection_mode:
            raise ValueError(f"selection mode mismatch for {run.key}")
        if run.weight is not None and float(
            resolved.get("dpp_ttft_drift_weight", math.nan)
        ) != run.weight:
            raise ValueError(f"TTFT drift weight mismatch for {run.key}")
        if run.selection_mode == "forced_stock_plan":
            with (run_root / "dpp_diagnostic_aggregate.json").open(
                "r", encoding="utf-8"
            ) as stream:
                aggregate = json.load(stream)
            if aggregate.get("selection_mode") != "forced_stock_plan":
                raise ValueError("forced Stock aggregate mode mismatch")
            histogram = aggregate.get("selection_histogram", {})
            if not isinstance(histogram, dict) or histogram.get("STOCK", 0) <= 0:
                raise ValueError("forced Stock aggregate has no STOCK iterations")
            if sum(value for key, value in histogram.items() if key != "STOCK") != 0:
                raise ValueError("forced Stock run selected a non-STOCK plan")
            pipeline = aggregate.get("pipeline_call_counts")
            if not isinstance(pipeline, dict) or set(pipeline) != {
                "predictor",
                "safe_set",
                "selector",
                "fallback",
            }:
                raise ValueError("forced Stock aggregate pipeline audit is incomplete")
            if any(pipeline.values()):
                raise ValueError("forced Stock run invoked a bypassed pipeline stage")
        metrics = _metrics(
            run_root, ttft_slo_ms=ttft_slo_ms, tbt_slo_ms=tbt_slo_ms
        )
        comparisons.append(
            {
                "run": run.__dict__,
                "stock": stock_metrics,
                "dpp": metrics,
                "dpp_minus_stock": {
                    "natural_eos_slo_success_requests": (
                        metrics["natural_eos_slo_success_requests"]
                        - stock_metrics["natural_eos_slo_success_requests"]
                    ),
                    "natural_eos_slo_goodput_rps": (
                        metrics["natural_eos_slo_goodput_rps"]
                        - stock_metrics["natural_eos_slo_goodput_rps"]
                    ),
                    "completion_throughput_rps": (
                        metrics["completion_throughput_rps"]
                        - stock_metrics["completion_throughput_rps"]
                    ),
                },
            }
        )
    forced = [
        item
        for item in comparisons
        if item["run"]["selection_mode"] == "forced_stock_plan"
    ]
    weighted = [item for item in comparisons if item["run"]["weight"] is not None]
    if len(forced) != 1 or len(weighted) != len(settings.weights):
        raise ValueError("TTFT grid comparison grouping mismatch")
    return {
        "schema_version": 1,
        "valid": True,
        "campaign_id": settings.campaign_id,
        "scope": (
            "development_nonformal_smoke"
            if settings.is_smoke
            else "development_nonformal_single_seed"
        ),
        "smoke": settings.is_smoke,
        "formal_benchmark_eligible": False,
        "run_count": len(matrix(settings)),
        "request_count_per_run": settings.request_count,
        "slo_seconds": {
            "ttft": slo.ttft_slo_seconds,
            "tbt": slo.tbt_slo_seconds,
        },
        "aggregation_warning": "single-seed results have no confidence interval",
        "best_lambda_policy": "report_only_no_automatic_freeze",
        "forced_stock_plan_comparison": forced[0],
        "lambda_comparisons": weighted,
    }


def worker(runtime: ActiveRuntime, settings: GridSettings, *, resume: bool) -> int:
    require_frozen_for_execution(runtime)
    root = _root(runtime, settings)
    if resume:
        state = _load(root, settings)
    else:
        if root.exists():
            raise FileExistsError(f"append-only TTFT grid already exists: {root}")
        root.mkdir(parents=True)
        (root / "launcher_logs").mkdir()
        state = _state(settings)
        _save(root, state)
        _atomic_json(
            root / MANIFEST_NAME,
            {
                **preview(runtime, settings),
                "schema_version": 1,
                "config_sha256": runtime.config_sha256,
                "git": {
                    "root": _git_state(runtime.workspace),
                    "vllm": _git_state(runtime.workspace, "vllm"),
                },
                "created_at_utc": _utc_now(),
            },
        )
    deadline = time.monotonic() + settings.campaign_timeout_seconds
    try:
        _ensure_trace(runtime, settings, root, state)
        for run, item in zip(matrix(settings), state["runs"]):
            if time.monotonic() >= deadline:
                raise TimeoutError("TTFT grid campaign timeout expired")
            if not _run_one(
                runtime, settings, root, state, run, item, deadline
            ):
                _save(root, state)
                raise RuntimeError(f"TTFT grid run failed: {run.key}")
            _save(root, state)
        report = validate_campaign(runtime, settings, root)
        state["status"] = "complete"
        state["finished_at_utc"] = _utc_now()
        _save(root, state)
        _atomic_json(root / "comparison_report.json", report)
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _save(root, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preview",
            "worker",
            "resume",
            "status",
            "validate",
            "smoke-preview",
            "smoke-worker",
            "smoke-resume",
            "smoke-status",
            "smoke-validate",
        ),
    )
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    args = parser.parse_args()
    runtime = load_active_runtime(args.config)
    settings = load_settings(runtime)
    if args.command.startswith("smoke-"):
        settings = as_smoke(settings)
    root = _root(runtime, settings)
    if args.command in {"preview", "smoke-preview"}:
        print(json.dumps(preview(runtime, settings), ensure_ascii=False, indent=2))
        return 0
    if args.command in {"worker", "smoke-worker"}:
        return worker(runtime, settings, resume=False)
    if args.command in {"resume", "smoke-resume"}:
        return worker(runtime, settings, resume=True)
    if args.command in {"status", "smoke-status"}:
        payload = (
            _load(root, settings)
            if root.exists()
            else {"campaign_id": settings.campaign_id, "status": "not_started"}
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    report = validate_campaign(runtime, settings, root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
