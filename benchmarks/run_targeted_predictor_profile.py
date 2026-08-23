#!/usr/bin/env python3
"""Run one config-locked exact targeted Predictor profiling trace."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.predictor_profile import (
    CAMPAIGN_ID as STOCK_CAMPAIGN_ID,
    merge_iteration_profiles,
)
from benchmarks.qwen3_runtime import (
    ActiveConfigError,
    build_targeted_profile_server_command,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
    sha256_file,
)
from benchmarks.run_predictor_profile import (
    _bounded_trace,
    _stop_server,
    _wait_for_server,
)
from benchmarks.run_stock_natural_eos import (
    RUN_ID_PATTERN,
    _atomic_json,
    _git_state,
    _require_free_port,
    _resolved_preview,
    load_trace,
    summarize,
    verify_trace_manifest,
)
from benchmarks.targeted_predictor_profile import validate_target_rows
from dpp_scheduler.targeted_profile import (
    TARGET_CAMPAIGN_ID,
    TARGET_REQUEST_COUNT,
    TARGET_REQUEST_TIMEOUT_SECONDS,
    TARGET_RUN_TIMEOUT_SECONDS,
    TARGET_SMOKE_CAMPAIGN_ID,
    build_target_recipes,
)
from dpp_scheduler.vllm_adapter import (
    TARGET_PROFILE_PATH_ENV,
    TARGET_PROFILE_RECIPE_MODE_ENV,
    TARGET_PROFILE_RECIPE_SEED_ENV,
    TARGET_PROFILE_RUN_ID_ENV,
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _implementation_hashes(workspace: Path) -> dict[str, str]:
    relative_paths = (
        "benchmarks/predictor_profile.py",
        "benchmarks/run_targeted_predictor_profile.py",
        "benchmarks/targeted_predictor_profile.py",
        "dpp_scheduler/contracts.py",
        "dpp_scheduler/targeted_profile.py",
        "dpp_scheduler/targeted_profile_scheduler.py",
        "dpp_scheduler/vllm_adapter.py",
    )
    return {
        relative: sha256_file(workspace / relative)
        for relative in relative_paths
    }


def _prepare_controlled_rows(
    rows: list[dict[str, Any]],
    *,
    request_count: int,
    recipe_seed: int,
    dispatch_interval_seconds: float,
) -> list[dict[str, Any]]:
    if request_count <= 0 or request_count > len(rows):
        raise ValueError("target request count is outside the source trace")
    if not math.isfinite(dispatch_interval_seconds) or dispatch_interval_seconds < 0:
        raise ValueError("dispatch interval must be finite and non-negative")
    controlled: list[dict[str, Any]] = []
    for index, source in enumerate(rows[:request_count]):
        row = dict(source)
        row["request_id"] = f"target_s{recipe_seed}_{index:04d}"
        row["arrival_time_s"] = round(index * dispatch_interval_seconds, 6)
        controlled.append(row)
    return controlled


def _verify_source_trace(
    manifest: dict[str, Any],
    *,
    trace_name: str,
    source_qps: float,
    source_seed: int,
) -> None:
    matches = [item for item in manifest.get("files", []) if item.get("file") == trace_name]
    if len(matches) != 1:
        raise ValueError("target source trace identity is ambiguous")
    entry = matches[0]
    if float(entry.get("requested_qps")) != source_qps:
        raise ValueError("target source trace QPS mismatch")
    if int(entry.get("seed")) != source_seed:
        raise ValueError("target source trace seed mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    parser.add_argument("--campaign-id", default=TARGET_CAMPAIGN_ID)
    parser.add_argument("--source-campaign-id", default=STOCK_CAMPAIGN_ID)
    parser.add_argument("--source-trace-dir", required=True)
    parser.add_argument("--source-trace", required=True)
    parser.add_argument("--source-qps", type=float, required=True)
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe-seed", type=int, required=True)
    parser.add_argument("--recipe-mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--request-count", type=int, default=TARGET_REQUEST_COUNT)
    parser.add_argument("--dispatch-interval", type=float, default=0.002)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument(
        "--request-timeout", type=float, default=TARGET_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument("--run-timeout", type=float, default=TARGET_RUN_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runtime = load_active_runtime(args.config)
    for value, label in (
        (args.campaign_id, "campaign_id"),
        (args.source_campaign_id, "source_campaign_id"),
        (args.run_id, "run_id"),
    ):
        if not RUN_ID_PATTERN.fullmatch(value):
            raise ActiveConfigError(f"invalid {label}: {value!r}")
    expected_campaign_id = (
        TARGET_SMOKE_CAMPAIGN_ID
        if args.recipe_mode == "smoke"
        else TARGET_CAMPAIGN_ID
    )
    if args.campaign_id != expected_campaign_id:
        raise ActiveConfigError(
            f"{args.recipe_mode} targeted profiling requires campaign "
            f"{expected_campaign_id}"
        )
    if (
        not math.isfinite(args.source_qps)
        or args.source_qps <= 0
        or args.startup_timeout <= 0
        or args.request_timeout <= 0
        or args.run_timeout <= 0
    ):
        raise ActiveConfigError("QPS and timeouts must be positive")

    source_root = resolve_under(
        runtime.raw_results, args.source_campaign_id, label="source campaign"
    )
    trace_root = resolve_under(
        source_root, args.source_trace_dir, label="source trace directory"
    )
    trace_path = resolve_under(trace_root, args.source_trace, label="source trace")
    trace_manifest_path = resolve_under(
        trace_root, "manifest.json", label="source trace manifest"
    )
    campaign_root = resolve_under(
        runtime.raw_results, args.campaign_id, label="target campaign"
    )
    output_dir = resolve_under(
        campaign_root, Path("runs") / args.run_id, label="target run output"
    )
    source_rows = load_trace(trace_path, runtime)
    source_manifest = verify_trace_manifest(
        trace_path, trace_manifest_path, runtime
    )
    _verify_source_trace(
        source_manifest,
        trace_name=trace_path.name,
        source_qps=args.source_qps,
        source_seed=args.source_seed,
    )
    rows = _prepare_controlled_rows(
        source_rows,
        request_count=args.request_count,
        recipe_seed=args.recipe_seed,
        dispatch_interval_seconds=args.dispatch_interval,
    )
    recipes = build_target_recipes(args.recipe_seed, mode=args.recipe_mode)
    recipe_payload = {
        "schema_version": 1,
        "mode": args.recipe_mode,
        "seed": args.recipe_seed,
        "recipes": [recipe.as_dict() for recipe in recipes],
    }
    command = build_targeted_profile_server_command(runtime, port=args.port)
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
            "campaign_id": args.campaign_id,
            "source_campaign_id": args.source_campaign_id,
            "source_qps": args.source_qps,
            "source_seed": args.source_seed,
            "recipe_seed": args.recipe_seed,
            "recipe_mode": args.recipe_mode,
            "recipe_count": len(recipes),
            "recipe_sha256": _canonical_sha256(recipe_payload),
            "implementation_sha256": _implementation_hashes(runtime.workspace),
            "arrival_mode": "controlled_burst",
            "dispatch_interval_seconds": args.dispatch_interval,
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
    _atomic_json(output_dir / "recipes.json", recipe_payload)

    scheduled_batches_path = output_dir / "scheduled_batches.jsonl"
    startup_log = output_dir / "startup.log"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen3_14b_targeted_predictor_profile",
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "source_qps": args.source_qps,
        "source_seed": args.source_seed,
        "recipe_seed": args.recipe_seed,
        "recipe_mode": args.recipe_mode,
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
    environment[TARGET_PROFILE_PATH_ENV] = str(scheduled_batches_path)
    environment[TARGET_PROFILE_RUN_ID_ENV] = args.run_id
    environment[TARGET_PROFILE_RECIPE_SEED_ENV] = str(args.recipe_seed)
    environment[TARGET_PROFILE_RECIPE_MODE_ENV] = args.recipe_mode

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
                    args.startup_timeout,
                    max(0.0, run_deadline - time.monotonic()),
                )
                if startup_budget <= 0:
                    raise TimeoutError("run timeout expired before server startup")
                _wait_for_server(process, port=args.port, timeout=startup_budget)
                execution_budget = max(0.0, run_deadline - time.monotonic())
                if execution_budget <= 0:
                    raise TimeoutError("run timeout expired before request execution")
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
            allowed_plan_prefixes=("stock-", "target-"),
        )
        _atomic_json(output_dir / "profile_validation.json", profile_validation)
        target_validation = validate_target_rows(
            output_dir / "iteration_profile.jsonl",
            expected_run_id=args.run_id,
            recipe_seed=args.recipe_seed,
            recipe_mode=args.recipe_mode,
        )
        _atomic_json(output_dir / "target_validation.json", target_validation)
        manifest["summary"] = summary
        manifest["profile_validation"] = profile_validation
        manifest["target_validation"] = target_validation
        if summary["failed"] == 0:
            manifest["status"] = "complete"
            return_code = 0
        else:
            manifest["status"] = "complete_with_failures"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _stop_server(process)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if startup_log.exists():
            manifest["startup_log_sha256"] = sha256_file(startup_log)
        _atomic_json(manifest_path, manifest)

    print(json.dumps(manifest["target_validation"], ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
