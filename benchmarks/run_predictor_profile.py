#!/usr/bin/env python3
"""Run one config-locked Stock trace with Predictor profiling enabled."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.predictor_profile import (
    PROFILE_SCHEMA_VERSION,
    REQUEST_TIMEOUT_SECONDS,
    RUN_TIMEOUT_SECONDS,
    merge_iteration_profiles,
)
from benchmarks.qwen3_runtime import (
    ActiveConfigError,
    build_stock_profile_server_command,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
    sha256_file,
)
from benchmarks.run_stock_natural_eos import (
    RUN_ID_PATTERN,
    _atomic_json,
    _git_state,
    _require_free_port,
    _resolved_preview,
    _run_trace,
    load_trace,
    summarize,
    verify_trace_manifest,
)
from dpp_scheduler.vllm_adapter import (
    STOCK_PROFILE_PATH_ENV,
    STOCK_PROFILE_RUN_ID_ENV,
)


def _stop_server(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_server(
    process: subprocess.Popen[str], *, port: int, timeout: float
) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"vLLM exited before becoming healthy: {exit_code}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception as error:  # noqa: BLE001 - preserve the last health error
            last_error = error
        time.sleep(3)
    raise RuntimeError(f"vLLM health endpoint did not become ready: {last_error}")


def _verify_trace_identity(
    manifest: dict[str, Any],
    *,
    trace_name: str,
    qps: float,
    seed: int,
    request_count: int,
) -> None:
    matches = [item for item in manifest.get("files", []) if item.get("file") == trace_name]
    if len(matches) != 1:
        raise ValueError(f"trace manifest identity is ambiguous for {trace_name}")
    entry = matches[0]
    if float(entry.get("requested_qps")) != qps:
        raise ValueError("trace QPS does not match requested profiling run")
    if int(entry.get("seed")) != seed:
        raise ValueError("trace seed does not match requested profiling run")
    if int(entry.get("num_requests")) != request_count:
        raise ValueError("trace request count does not match trace contents")


async def _bounded_trace(
    runtime: Any,
    rows: list[dict[str, Any]],
    *,
    port: int,
    request_timeout: float,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    return await asyncio.wait_for(
        _run_trace(
            runtime,
            rows,
            port=port,
            request_timeout=request_timeout,
        ),
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--trace-manifest", default="manifest.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument(
        "--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument("--run-timeout", type=float, default=RUN_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runtime = load_active_runtime(args.config)
    if not RUN_ID_PATTERN.fullmatch(args.campaign_id):
        raise ActiveConfigError(f"invalid campaign_id: {args.campaign_id!r}")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ActiveConfigError(f"invalid run_id: {args.run_id!r}")
    if (
        not math.isfinite(args.qps)
        or args.qps <= 0
        or args.startup_timeout <= 0
        or args.request_timeout <= 0
        or args.run_timeout <= 0
    ):
        raise ActiveConfigError("QPS and timeouts must be positive")

    campaign_root = resolve_under(
        runtime.raw_results, args.campaign_id, label="profiling campaign"
    )
    trace_root = resolve_under(campaign_root, args.trace_dir, label="profile traces")
    trace_path = resolve_under(trace_root, args.trace, label="profile trace")
    trace_manifest_path = resolve_under(
        trace_root, args.trace_manifest, label="profile trace manifest"
    )
    output_dir = resolve_under(
        campaign_root, Path("runs") / args.run_id, label="profile run output"
    )
    rows = load_trace(trace_path, runtime)
    trace_manifest = verify_trace_manifest(trace_path, trace_manifest_path, runtime)
    _verify_trace_identity(
        trace_manifest,
        trace_name=trace_path.name,
        qps=args.qps,
        seed=args.seed,
        request_count=len(rows),
    )
    command = build_stock_profile_server_command(runtime, port=args.port)
    preview = _resolved_preview(
        runtime,
        trace_path,
        trace_manifest_path,
        output_dir,
        command,
        rows,
    )
    preview.update(
        {
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "campaign_id": args.campaign_id,
            "run_id": args.run_id,
            "qps": args.qps,
            "seed": args.seed,
            "run_timeout_seconds": args.run_timeout,
            "request_timeout_seconds": args.request_timeout,
        }
    )
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    require_frozen_for_execution(runtime)
    if output_dir.exists():
        raise FileExistsError(f"append-only output directory exists: {output_dir}")
    _require_free_port(args.port)
    output_dir.mkdir(parents=True)

    scheduled_batches_path = output_dir / "scheduled_batches.jsonl"
    startup_log = output_dir / "startup.log"
    manifest: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "kind": "qwen3_14b_stock_predictor_profile",
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "qps": args.qps,
        "seed": args.seed,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved": preview,
        "git": {
            "root": _git_state(runtime.workspace),
            "vllm": _git_state(runtime.workspace, "vllm"),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    _atomic_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment.update(dict(runtime.required_env))
    environment["PATH"] = f"{runtime.python.parent}:{environment.get('PATH', '')}"
    environment[STOCK_PROFILE_PATH_ENV] = str(scheduled_batches_path)
    environment[STOCK_PROFILE_RUN_ID_ENV] = args.run_id

    process: subprocess.Popen[str] | None = None
    return_code = 1
    try:
        run_deadline = time.monotonic() + args.run_timeout
        with startup_log.open("x", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=runtime.workspace,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                startup_budget = min(
                    args.startup_timeout, max(0.0, run_deadline - time.monotonic())
                )
                if startup_budget <= 0:
                    raise TimeoutError("run timeout expired before server startup")
                _wait_for_server(process, port=args.port, timeout=startup_budget)
                execution_budget = max(0.0, run_deadline - time.monotonic())
                if execution_budget <= 0:
                    raise TimeoutError("run timeout expired before trace execution")
                results, elapsed_s = asyncio.run(
                    _bounded_trace(
                        runtime,
                        rows,
                        port=args.port,
                        request_timeout=args.request_timeout,
                        timeout=execution_budget,
                    )
                )
            finally:
                _stop_server(process)

        with (output_dir / "per_request.jsonl").open("x", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        summary = summarize(results, elapsed_s=elapsed_s)
        _atomic_json(output_dir / "summary.json", summary)
        profile_validation = merge_iteration_profiles(
            scheduled_batches_path=scheduled_batches_path,
            startup_log_path=startup_log,
            output_path=output_dir / "iteration_profile.jsonl",
            expected_run_id=args.run_id,
        )
        _atomic_json(output_dir / "profile_validation.json", profile_validation)
        manifest["summary"] = summary
        manifest["profile_validation"] = profile_validation
        if summary["failed"] == 0:
            manifest["status"] = "complete"
            return_code = 0
        else:
            manifest["status"] = "complete_with_failures"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        validation_path = output_dir / "profile_validation.json"
        if not validation_path.exists():
            _atomic_json(
                validation_path,
                {
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "valid": False,
                    "run_id": args.run_id,
                    "error": manifest["error"],
                },
            )
        raise
    finally:
        _stop_server(process)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if startup_log.exists():
            manifest["startup_log_sha256"] = sha256_file(startup_log)
        _atomic_json(manifest_path, manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
