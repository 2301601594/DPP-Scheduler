#!/usr/bin/env python3
"""Run one config-locked real-vLLM Predictor shadow evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.predictor_online_evaluation import analyze_evaluation
from benchmarks.qwen3_runtime import (
    ActiveConfigError,
    build_predictor_evaluation_server_command,
    candidate_runtime_signature,
    load_active_runtime,
    load_frozen_predictor,
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
from benchmarks.run_targeted_predictor_profile import (
    _prepare_controlled_rows,
    _verify_source_trace,
)
from dpp_scheduler.targeted_profile import build_target_recipes
from dpp_scheduler.vllm_adapter import (
    PREDICTOR_ARTIFACT_PATH_ENV,
    PREDICTOR_EVAL_PATH_ENV,
    PREDICTOR_EVAL_RECIPE_MODE_ENV,
    PREDICTOR_EVAL_RECIPE_SEED_ENV,
    PREDICTOR_EVAL_RUN_ID_ENV,
)


EVALUATION_CAMPAIGN_ID = "predictor_online_timing_aligned_n200_v1"
EVALUATION_SMOKE_CAMPAIGN_ID = "predictor_online_timing_aligned_smoke_v1"
OOD_CALIBRATION_CAMPAIGN_ID = "predictor_ood_calibration_v2"
OOD_VALIDATION_CAMPAIGN_ID = "predictor_ood_validation_v2"


def _implementation_hashes(workspace: Path) -> dict[str, str]:
    files = (
        "benchmarks/predictor_online_evaluation.py",
        "benchmarks/run_predictor_online_evaluation.py",
        "dpp_scheduler/contracts.py",
        "dpp_scheduler/predictor.py",
        "dpp_scheduler/targeted_profile.py",
        "dpp_scheduler/vllm_adapter.py",
    )
    return {name: sha256_file(workspace / name) for name in files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--source-campaign-id", required=True)
    parser.add_argument("--source-trace-dir", required=True)
    parser.add_argument("--source-trace", required=True)
    parser.add_argument("--source-qps", type=float, required=True)
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe-seed", type=int, required=True)
    parser.add_argument(
        "--recipe-mode", choices=("formal", "smoke", "ood"), required=True
    )
    parser.add_argument("--request-count", type=int, required=True)
    parser.add_argument("--dispatch-interval", type=float, default=0.002)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--run-timeout", type=float, default=10800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runtime = load_active_runtime(args.config)
    expected_campaigns = (
        {EVALUATION_SMOKE_CAMPAIGN_ID}
        if args.recipe_mode == "smoke"
        else {EVALUATION_CAMPAIGN_ID}
        if args.recipe_mode == "formal"
        else {OOD_CALIBRATION_CAMPAIGN_ID, OOD_VALIDATION_CAMPAIGN_ID}
    )
    if args.campaign_id not in expected_campaigns:
        raise ActiveConfigError("campaign ID does not match Predictor evaluation mode")
    for value, label in (
        (args.campaign_id, "campaign_id"),
        (args.source_campaign_id, "source_campaign_id"),
        (args.run_id, "run_id"),
    ):
        if not RUN_ID_PATTERN.fullmatch(value):
            raise ActiveConfigError(f"invalid {label}: {value!r}")
    if (
        args.request_count <= 0
        or not math.isfinite(args.source_qps)
        or args.source_qps <= 0
        or args.run_timeout <= 0
        or args.request_timeout <= 0
    ):
        raise ActiveConfigError("Predictor evaluation numeric arguments are invalid")

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
        runtime.raw_results, args.campaign_id, label="evaluation campaign"
    )
    output_dir = resolve_under(
        campaign_root, Path("runs") / args.run_id, label="evaluation run"
    )
    source_rows = load_trace(trace_path, runtime)
    source_manifest = verify_trace_manifest(trace_path, trace_manifest_path, runtime)
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
    predictor = load_frozen_predictor(runtime)
    command = build_predictor_evaluation_server_command(runtime, port=args.port)
    runtime_consistency, runtime_consistency_sha256 = candidate_runtime_signature(
        runtime
    )
    preview = _resolved_preview(
        runtime,
        trace_path,
        trace_manifest_path,
        output_dir,
        command,
        rows,
        scheduler_policy="predictor-shadow-evaluation",
        source_request_count=len(source_rows),
        diagnostic_iteration_log=False,
        campaign_id=args.campaign_id,
        comparison_scope="held_out_ood_predictor_evaluation",
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
            "request_count": args.request_count,
            "arrival_mode": "controlled_burst",
            "dispatch_interval_seconds": args.dispatch_interval,
            "predictor_artifact": str(predictor.artifact_root),
            "predictor_artifact_manifest_sha256": (
                predictor.artifact_manifest_sha256
            ),
            "predictor_version": predictor.predictor_version,
            "implementation_sha256": _implementation_hashes(runtime.workspace),
            "run_timeout_seconds": args.run_timeout,
            "request_timeout_seconds": args.request_timeout,
            "runtime_consistency": runtime_consistency,
            "runtime_consistency_sha256": runtime_consistency_sha256,
        }
    )
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    require_frozen_for_execution(runtime)
    if output_dir.exists():
        raise FileExistsError(f"append-only evaluation output exists: {output_dir}")
    _require_free_port(args.port)
    output_dir.mkdir(parents=True)
    telemetry_path = output_dir / "predictor_evaluation.jsonl"
    startup_log = output_dir / "startup.log"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen3_14b_predictor_online_shadow_evaluation",
        "status": "running",
        "run_id": args.run_id,
        "campaign_id": args.campaign_id,
        "source_seed": args.source_seed,
        "recipe_seed": args.recipe_seed,
        "recipe_mode": args.recipe_mode,
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
    environment[PREDICTOR_EVAL_PATH_ENV] = str(telemetry_path)
    environment[PREDICTOR_EVAL_RUN_ID_ENV] = args.run_id
    environment[PREDICTOR_ARTIFACT_PATH_ENV] = str(predictor.artifact_root)
    environment[PREDICTOR_EVAL_RECIPE_SEED_ENV] = str(args.recipe_seed)
    environment[PREDICTOR_EVAL_RECIPE_MODE_ENV] = args.recipe_mode

    process: subprocess.Popen[str] | None = None
    results: list[dict[str, Any]] = []
    try:
        deadline = time.monotonic() + args.run_timeout
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
                startup_budget = min(args.startup_timeout, deadline - time.monotonic())
                if startup_budget <= 0:
                    raise TimeoutError("run timeout expired before startup")
                _wait_for_server(process, port=args.port, timeout=startup_budget)
                execution_budget = deadline - time.monotonic()
                if execution_budget <= 0:
                    raise TimeoutError("run timeout expired before requests")
                results, elapsed = asyncio.run(
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
        with (output_dir / "per_request.jsonl").open(
            "x", encoding="utf-8"
        ) as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        summary = summarize(results, elapsed_s=elapsed)
        _atomic_json(output_dir / "summary.json", summary)
        evaluation = analyze_evaluation(
            telemetry_path=telemetry_path,
            startup_log_path=startup_log,
            output_path=output_dir / "iteration_evaluation.jsonl",
            expected_run_id=args.run_id,
            recipe_seed=args.recipe_seed,
            recipe_mode=args.recipe_mode,
        )
        _atomic_json(output_dir / "evaluation_summary.json", evaluation)
        manifest["request_summary"] = summary
        manifest["evaluation_summary"] = evaluation
        manifest["status"] = (
            "complete" if summary["failed"] == 0 else "complete_with_failures"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "complete" else 1
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _stop_server(process)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        for name in (
            "startup.log",
            "predictor_evaluation.jsonl",
            "iteration_evaluation.jsonl",
            "summary.json",
            "evaluation_summary.json",
        ):
            path = output_dir / name
            if path.exists():
                manifest.setdefault("file_sha256", {})[name] = sha256_file(path)
        _atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
