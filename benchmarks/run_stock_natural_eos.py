#!/usr/bin/env python3
"""Run one config-locked Stock or modular-DPP natural-output trace.

The finite ``max_tokens`` sent to the API is a client termination guard only.
It is recorded with each request but is never exposed through Scheduler
contracts, features, labels, or decisions.  A request ending with ``length`` is
retained and reported; it does not retroactively turn the guard into a target
output length.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import socket
import statistics
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.qwen3_runtime import (
    ActiveConfigError,
    ActiveRuntime,
    build_dpp_server_command,
    build_stock_server_command,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
    sha256_file,
)


TRACE_FORBIDDEN_FIELDS = frozenset(
    {
        "output_tokens",
        "expected_output_tokens",
        "remaining_output_tokens",
        "eventual_eos_position",
        "target_output_tokens",
    }
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SCHEDULER_POLICIES = ("stock", "dpp")
DPP_DIAGNOSTIC_ITERATION_LOG_ENV = "DPP_DIAGNOSTIC_ITERATION_LOG"
DPP_DIAGNOSTIC_AGGREGATE_PATH_ENV = "DPP_DIAGNOSTIC_AGGREGATE_PATH"
DPP_EXECUTION_SCOPE_ENV = "DPP_EXECUTION_SCOPE"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _git_state(workspace: Path, relative: str = ".") -> dict[str, Any]:
    repository = workspace / relative
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--short"], text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _safety_ceiling(row: dict[str, Any], runtime: ActiveRuntime) -> int:
    canonical = row.get("client_safety_ceiling_tokens")
    if "max_tokens_safety" in row:
        raise ValueError(
            "deprecated max_tokens_safety trace schema is not accepted; "
            "regenerate and review the trace"
        )
    value = canonical
    if value is None:
        value = runtime.client_safety_ceiling_tokens
    ceiling = int(value)
    if ceiling != runtime.client_safety_ceiling_tokens:
        raise ValueError(
            "trace safety ceiling does not match the reviewed active config: "
            f"trace={ceiling}, config={runtime.client_safety_ceiling_tokens}"
        )
    return ceiling


def load_trace(path: Path, runtime: ActiveRuntime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_arrival = -math.inf
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"trace line {line_number} is not an object")
            forbidden = TRACE_FORBIDDEN_FIELDS.intersection(row)
            if forbidden:
                raise ValueError(
                    f"trace line {line_number} exposes predetermined output state: "
                    f"{sorted(forbidden)}"
                )
            for key in (
                "request_id",
                "prompt_id",
                "prompt",
                "input_tokens",
                "arrival_time_s",
                "generation_seed",
            ):
                if key not in row:
                    raise ValueError(f"trace line {line_number} is missing {key}")
            request_id = str(row["request_id"])
            if request_id in seen_ids:
                raise ValueError(f"duplicate request_id: {request_id}")
            seen_ids.add(request_id)

            arrival = float(row["arrival_time_s"])
            if not math.isfinite(arrival) or arrival < 0 or arrival < previous_arrival:
                raise ValueError(
                    f"trace line {line_number} has invalid/non-monotonic arrival"
                )
            previous_arrival = arrival
            input_tokens = int(row["input_tokens"])
            if input_tokens <= 0:
                raise ValueError(f"trace line {line_number} has invalid input_tokens")
            ceiling = _safety_ceiling(row, runtime)
            if input_tokens + ceiling > runtime.max_model_len:
                raise ValueError(
                    f"trace line {line_number} can exceed max_model_len: "
                    f"{input_tokens}+{ceiling}>{runtime.max_model_len}"
                )
            temperature = float(row.get("temperature", runtime.temperature))
            top_p = float(row.get("top_p", runtime.top_p))
            ignore_eos = bool(row.get("ignore_eos", runtime.ignore_eos))
            if (
                temperature != runtime.temperature
                or top_p != runtime.top_p
                or ignore_eos != runtime.ignore_eos
            ):
                raise ValueError(
                    f"trace line {line_number} sampling parameters differ from config"
                )
            normalized = dict(row)
            normalized["client_safety_ceiling_tokens"] = ceiling
            normalized["temperature"] = temperature
            normalized["top_p"] = top_p
            normalized["ignore_eos"] = ignore_eos
            normalized["generation_seed"] = int(row["generation_seed"])
            rows.append(normalized)
    if not rows:
        raise ValueError("trace is empty")
    return rows


def verify_trace_manifest(
    trace_path: Path, manifest_path: Path, runtime: ActiveRuntime
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    required = {
        "model_revision": runtime.model_revision,
        "tokenizer_revision": runtime.tokenizer_revision,
        "client_safety_ceiling_tokens": runtime.client_safety_ceiling_tokens,
        "client_safety_ceiling_role": "termination_guard_only_never_scheduler_input",
        "predetermined_output_length": False,
        "temperature": runtime.temperature,
        "top_p": runtime.top_p,
        "ignore_eos": runtime.ignore_eos,
        "seed_source": runtime.seed_source,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"trace manifest {key} mismatch: expected {expected!r}, "
                f"got {manifest.get(key)!r}"
            )
    # This is immutable provenance for the configuration that generated the
    # trace. Scheduler-only config additions must not invalidate unchanged
    # prompts, arrivals, seeds, or generation settings.
    provenance_hash = manifest.get("config_sha256")
    if not isinstance(provenance_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", provenance_hash
    ):
        raise ValueError("trace manifest config_sha256 provenance is invalid")
    matches = [
        item
        for item in manifest.get("files", [])
        if item.get("file") == trace_path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"trace manifest must contain exactly one entry for {trace_path.name}")
    observed_hash = sha256_file(trace_path)
    if matches[0].get("sha256") != observed_hash:
        raise ValueError("trace SHA256 does not match its manifest")
    return manifest


def build_request_payload(runtime: ActiveRuntime, row: dict[str, Any]) -> dict[str, Any]:
    if TRACE_FORBIDDEN_FIELDS.intersection(row):
        raise ValueError("predetermined output-length field reached request builder")
    return {
        "model": runtime.model_name,
        "prompt": row["prompt"],
        # API termination guard only. It is deliberately absent from every
        # dpp_scheduler public contract and Predictor feature.
        "max_tokens": int(row["client_safety_ceiling_tokens"]),
        "temperature": float(row["temperature"]),
        "top_p": float(row["top_p"]),
        "seed": int(row["generation_seed"]),
        "ignore_eos": bool(row["ignore_eos"]),
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }


def _token_count(choice: dict[str, Any]) -> tuple[int, bool]:
    token_ids = choice.get("token_ids")
    text = choice.get("text") or ""
    if token_ids is None:
        # A terminal SSE chunk commonly carries no token IDs and empty text.
        return (0, not bool(text))
    if not isinstance(token_ids, list):
        return (0, False)
    return (len(token_ids), True)


async def send_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    runtime: ActiveRuntime,
    row: dict[str, Any],
    *,
    benchmark_start: float,
    dispatched_at: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "request_id": row["request_id"],
        "prompt_id": row["prompt_id"],
        "planned_arrival_s": float(row["arrival_time_s"]),
        "actual_dispatch_s": dispatched_at - benchmark_start,
        "dispatch_error_ms": (
            dispatched_at - benchmark_start - float(row["arrival_time_s"])
        )
        * 1000,
        "generation_seed": int(row["generation_seed"]),
        "client_safety_ceiling_tokens": int(
            row["client_safety_ceiling_tokens"]
        ),
        "input_tokens": int(row["input_tokens"]),
        "completed": False,
        "error": None,
        "http_status": None,
        "ttft_ms": None,
        "itls_ms": [],
        "token_chunk_events": [],
        "token_timing_exact": True,
        "multi_token_chunks": 0,
        "e2e_ms": None,
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "observed_stream_tokens": 0,
        "finish_reason": None,
        "safety_ceiling_reached": False,
    }
    request_start = time.perf_counter()
    token_times: list[float] = []
    try:
        payload = build_request_payload(runtime, row)
        async with session.post(endpoint, json=payload, headers={
            "Content-Type": "application/json",
            "x-request-id": str(row["request_id"]),
        }) as response:
            result["http_status"] = response.status
            if response.status != 200:
                result["error"] = await response.text()
                return result
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                message = line.removeprefix("data:").strip()
                if message == "[DONE]":
                    continue
                data = json.loads(message)
                if usage := data.get("usage"):
                    result["actual_input_tokens"] = usage.get("prompt_tokens")
                    result["actual_output_tokens"] = usage.get("completion_tokens")
                choices = data.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    result["finish_reason"] = str(choice["finish_reason"])
                count, exact = _token_count(choice)
                result["token_timing_exact"] &= exact
                if count <= 0:
                    continue
                now = time.perf_counter()
                result["observed_stream_tokens"] += count
                result["token_chunk_events"].append(
                    {
                        "offset_ms": (now - request_start) * 1000,
                        "token_count": count,
                    }
                )
                if count != 1:
                    result["multi_token_chunks"] += 1
                    result["token_timing_exact"] = False
                token_times.append(now)

        if token_times:
            result["ttft_ms"] = (token_times[0] - request_start) * 1000
            if result["token_timing_exact"]:
                result["itls_ms"] = [
                    (current - previous) * 1000
                    for previous, current in zip(token_times, token_times[1:])
                ]
        result["safety_ceiling_reached"] = result["finish_reason"] == "length"
        result["completed"] = bool(
            result["http_status"] == 200
            and result["finish_reason"] is not None
            and result["actual_output_tokens"] is not None
            and result["error"] is None
        )
    except Exception as error:  # noqa: BLE001 - preserve per-request failure
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["e2e_ms"] = (time.perf_counter() - request_start) * 1000
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "mean": statistics.fmean(values) if values else None,
    }


def summarize(results: list[dict[str, Any]], *, elapsed_s: float) -> dict[str, Any]:
    completed = [item for item in results if item["completed"]]
    exact = [item for item in completed if item["token_timing_exact"]]
    ttft = [float(item["ttft_ms"]) for item in completed if item["ttft_ms"] is not None]
    itls = [float(value) for item in exact for value in item["itls_ms"]]
    e2e = [float(item["e2e_ms"]) for item in completed]
    outputs = [int(item["actual_output_tokens"]) for item in completed]
    dispatch_errors = [float(item["dispatch_error_ms"]) for item in results]
    finish_reasons = Counter(
        item["finish_reason"] or ("error" if item["error"] else "missing")
        for item in results
    )
    input_mismatches = sum(
        item["actual_input_tokens"] is not None
        and int(item["actual_input_tokens"]) != int(item["input_tokens"])
        for item in results
    )
    stream_count_mismatches = sum(
        item["actual_output_tokens"] is not None
        and int(item["actual_output_tokens"]) != int(item["observed_stream_tokens"])
        for item in results
    )
    return {
        "num_requests": len(results),
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "safety_ceiling_reached": sum(
            bool(item["safety_ceiling_reached"]) for item in results
        ),
        "input_token_mismatches": input_mismatches,
        "stream_token_count_mismatches": stream_count_mismatches,
        "token_timing_exact_requests": len(exact),
        "multi_token_chunks": sum(int(item["multi_token_chunks"]) for item in results),
        "ttft_ms": _distribution(ttft),
        "tbt_ms_exact_requests_only": _distribution(itls),
        "e2e_ms": _distribution(e2e),
        "output_tokens": {
            "min": min(outputs) if outputs else None,
            "max": max(outputs) if outputs else None,
            "mean": statistics.fmean(outputs) if outputs else None,
        },
        "dispatch_error_ms": _distribution(dispatch_errors),
        "elapsed_s": elapsed_s,
        "completion_throughput_rps": len(completed) / elapsed_s if elapsed_s > 0 else None,
    }


def _wait_for_health(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(3)
    raise RuntimeError(f"vLLM health endpoint did not become ready: {last_error}")


def _require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use")


async def _run_trace(
    runtime: ActiveRuntime,
    rows: list[dict[str, Any]],
    *,
    port: int,
    request_timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        benchmark_start = time.perf_counter()

        async def dispatch(row: dict[str, Any]) -> dict[str, Any]:
            delay = benchmark_start + float(row["arrival_time_s"]) - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            dispatched_at = time.perf_counter()
            return await send_one(
                session,
                f"http://127.0.0.1:{port}/v1/completions",
                runtime,
                row,
                benchmark_start=benchmark_start,
                dispatched_at=dispatched_at,
            )

        results = await asyncio.gather(*(dispatch(row) for row in rows))
        return results, time.perf_counter() - benchmark_start


def _resolved_preview(
    runtime: ActiveRuntime,
    trace_path: Path,
    trace_manifest_path: Path,
    output_dir: Path,
    command: list[str],
    rows: list[dict[str, Any]],
    *,
    scheduler_policy: str,
    source_request_count: int,
    diagnostic_iteration_log: bool,
    campaign_id: str | None,
    comparison_scope: str,
) -> dict[str, Any]:
    return {
        "config": str(runtime.config_path),
        "config_sha256": runtime.config_sha256,
        "config_status": runtime.status,
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "trace_manifest": str(trace_manifest_path),
        "trace_manifest_sha256": sha256_file(trace_manifest_path),
        "request_count": len(rows),
        "source_request_count": source_request_count,
        "diagnostic_prefix": len(rows) != source_request_count,
        "planned_arrival_span_s": float(rows[-1]["arrival_time_s"]),
        "scheduler_policy": scheduler_policy,
        "campaign_id": campaign_id,
        "comparison_scope": comparison_scope,
        "dpp_diagnostic_iteration_log": diagnostic_iteration_log,
        "client_safety_ceiling_tokens": runtime.client_safety_ceiling_tokens,
        "scheduler_receives_safety_ceiling": False,
        "output_dir": str(output_dir),
        "server_command": command,
        "required_env": dict(runtime.required_env),
        "runner_env_overrides": {
            DPP_DIAGNOSTIC_ITERATION_LOG_ENV: (
                "1" if diagnostic_iteration_log else "0"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--trace-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--campaign-id",
        help="place the append-only run under <raw>/<campaign-id>/runs/",
    )
    parser.add_argument(
        "--development-trace-dir",
        help=(
            "development-only trace directory relative to the active raw-results "
            "root; runs using it are never formal-comparison eligible"
        ),
    )
    parser.add_argument("--policy", choices=SCHEDULER_POLICIES, default="stock")
    parser.add_argument(
        "--request-limit",
        type=int,
        help="run only the unchanged trace prefix as a diagnostic smoke",
    )
    parser.add_argument(
        "--dpp-diagnostic-iteration-log",
        action="store_true",
        help="enable per-iteration DPP INFO logs for a diagnostic run only",
    )
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runtime = load_active_runtime(args.config)
    if args.campaign_id is not None and not RUN_ID_PATTERN.fullmatch(args.campaign_id):
        raise ActiveConfigError(f"invalid campaign_id: {args.campaign_id!r}")
    if args.development_trace_dir is None:
        trace_root = runtime.active_traces
        comparison_scope = "active_frozen_trace"
    else:
        trace_root = resolve_under(
            runtime.raw_results,
            args.development_trace_dir,
            label="development trace directory",
        )
        comparison_scope = "development_nonformal"
    trace_path = resolve_under(trace_root, args.trace, label="trace")
    manifest_path = resolve_under(
        trace_root, args.trace_manifest, label="trace manifest"
    )
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ActiveConfigError(f"invalid run_id: {args.run_id!r}")
    if args.campaign_id is None:
        output_dir = resolve_under(
            runtime.raw_results, args.run_id, label="output directory"
        )
    else:
        campaign_root = resolve_under(
            runtime.raw_results, args.campaign_id, label="campaign directory"
        )
        output_dir = resolve_under(
            campaign_root,
            Path("runs") / args.run_id,
            label="campaign run output directory",
        )
    rows = load_trace(trace_path, runtime)
    source_request_count = len(rows)
    verify_trace_manifest(trace_path, manifest_path, runtime)
    if args.request_limit is not None:
        if not 1 <= args.request_limit <= source_request_count:
            raise ActiveConfigError(
                f"request limit must be in [1, {source_request_count}]"
            )
        rows = rows[: args.request_limit]
    if args.dpp_diagnostic_iteration_log and args.policy != "dpp":
        raise ActiveConfigError(
            "DPP diagnostic iteration logging requires --policy dpp"
        )
    command_builder = (
        build_dpp_server_command if args.policy == "dpp" else build_stock_server_command
    )
    command = command_builder(runtime, port=args.port)
    preview = _resolved_preview(
        runtime,
        trace_path,
        manifest_path,
        output_dir,
        command,
        rows,
        scheduler_policy=args.policy,
        source_request_count=source_request_count,
        diagnostic_iteration_log=args.dpp_diagnostic_iteration_log,
        campaign_id=args.campaign_id,
        comparison_scope=comparison_scope,
    )
    execution_scope = (
        "stock"
        if args.policy == "stock"
        else "development_nonformal"
        if args.request_limit is not None
        else "formal"
    )
    preview["runner_env_overrides"][DPP_EXECUTION_SCOPE_ENV] = execution_scope
    if args.policy == "dpp":
        preview["runner_env_overrides"][DPP_DIAGNOSTIC_AGGREGATE_PATH_ENV] = str(
            output_dir / "dpp_diagnostic_aggregate.json"
        )
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    require_frozen_for_execution(runtime)
    if output_dir.exists():
        raise FileExistsError(f"append-only output directory already exists: {output_dir}")
    _require_free_port(args.port)
    output_dir.mkdir(parents=True)

    environment = os.environ.copy()
    environment.update(dict(runtime.required_env))
    environment["PATH"] = f"{runtime.python.parent}:{environment.get('PATH', '')}"
    environment[DPP_DIAGNOSTIC_ITERATION_LOG_ENV] = (
        "1" if args.dpp_diagnostic_iteration_log else "0"
    )
    environment[DPP_EXECUTION_SCOPE_ENV] = execution_scope
    if args.policy == "dpp":
        environment[DPP_DIAGNOSTIC_AGGREGATE_PATH_ENV] = str(
            output_dir / "dpp_diagnostic_aggregate.json"
        )
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "kind": "qwen3_14b_scheduler_natural_output",
        "run_id": args.run_id,
        "scheduler_policy": args.policy,
        "comparison_eligible": args.request_limit is None,
        "formal_comparison_eligible": (
            args.request_limit is None and args.development_trace_dir is None
        ),
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved": preview,
        "git": {
            "root": _git_state(runtime.workspace),
            "vllm": _git_state(runtime.workspace, "vllm"),
        },
    }
    manifest_path_out = output_dir / "run_manifest.json"
    _atomic_json(manifest_path_out, manifest)

    process: subprocess.Popen[str] | None = None
    startup_log = output_dir / "startup.log"
    try:
        with startup_log.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=runtime.workspace,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_for_health(args.port, args.startup_timeout)
            results, elapsed_s = asyncio.run(
                _run_trace(
                    runtime,
                    rows,
                    port=args.port,
                    request_timeout=args.request_timeout,
                )
            )
        with (output_dir / "per_request.jsonl").open("x", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        summary = summarize(results, elapsed_s=elapsed_s)
        _atomic_json(output_dir / "summary.json", summary)
        manifest["status"] = "complete" if summary["failed"] == 0 else "complete_with_failures"
        manifest["summary"] = summary
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=runtime.shutdown_timeout_seconds + 20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if startup_log.exists():
            manifest["startup_log_sha256"] = sha256_file(startup_log)
        _atomic_json(manifest_path_out, manifest)

    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote append-only run: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
