#!/usr/bin/env python3
"""Orchestrate the fixed two-run targeted Predictor profiling campaign."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.predictor_profile import CAMPAIGN_ID as STOCK_CAMPAIGN_ID
from benchmarks.qwen3_runtime import (
    ActiveRuntime,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
)
from benchmarks.run_predictor_profile_campaign import validate_campaign as validate_stock_campaign
from benchmarks.run_stock_natural_eos import _atomic_json, _git_state
from benchmarks.targeted_predictor_profile import validate_target_run_directory
from dpp_scheduler.targeted_profile import (
    TARGET_CAMPAIGN_ID,
    TARGET_CAMPAIGN_MATRIX,
    TARGET_CAMPAIGN_TIMEOUT_SECONDS,
    TARGET_MAX_ATTEMPTS,
    TARGET_REQUEST_COUNT,
    TARGET_REQUEST_TIMEOUT_SECONDS,
    TARGET_RUN_TIMEOUT_SECONDS,
    TARGET_SMOKE_CAMPAIGN_ID,
    TargetCampaignRun,
    build_target_recipes,
)


CHECKPOINT_NAME = "campaign_checkpoint.json"
MANIFEST_NAME = "campaign_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_root(runtime: ActiveRuntime) -> Path:
    return resolve_under(runtime.raw_results, TARGET_CAMPAIGN_ID, label="target campaign")


def _source_root(runtime: ActiveRuntime) -> Path:
    return resolve_under(runtime.raw_results, STOCK_CAMPAIGN_ID, label="source campaign")


def _append_log(root: Path, message: str) -> None:
    rendered = f"[{_utc_now()}] {message}"
    print(rendered, flush=True)
    with (root / "campaign.log").open("a", encoding="utf-8") as stream:
        stream.write(rendered + "\n")


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": TARGET_CAMPAIGN_ID,
        "status": "running",
        "request_count_per_run": TARGET_REQUEST_COUNT,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "source_trace_dir": None,
        "runs": [
            {
                "key": item.key,
                "source_qps": item.source_qps,
                "source_seed": item.source_seed,
                "recipe_seed": item.recipe_seed,
                "status": "pending",
                "valid_attempt": None,
                "attempts": [],
            }
            for item in TARGET_CAMPAIGN_MATRIX
        ],
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(root / CHECKPOINT_NAME, state)


def _load_state(root: Path) -> dict[str, Any]:
    with (root / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("campaign_id") != TARGET_CAMPAIGN_ID:
        raise ValueError("target campaign checkpoint identity mismatch")
    expected = [
        (item.key, item.source_qps, item.source_seed, item.recipe_seed)
        for item in TARGET_CAMPAIGN_MATRIX
    ]
    observed = [
        (
            item.get("key"),
            float(item.get("source_qps")),
            int(item.get("source_seed")),
            int(item.get("recipe_seed")),
        )
        for item in state.get("runs", [])
    ]
    if observed != expected:
        raise ValueError("target campaign checkpoint matrix mismatch")
    if int(state.get("request_count_per_run", 0)) != TARGET_REQUEST_COUNT:
        raise ValueError("target campaign request count mismatch")
    return state


def _source_trace_dir(runtime: ActiveRuntime) -> str:
    source = _source_root(runtime)
    validate_stock_campaign(runtime, source)
    with (source / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    trace_dir = checkpoint.get("trace_dir")
    if not trace_dir:
        raise ValueError("source Stock campaign has no valid trace directory")
    return str(trace_dir)


def _run_logged_command(
    command: list[str], *, cwd: Path, log_path: Path, timeout: float
) -> int:
    with log_path.open("x", encoding="utf-8") as stream:
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
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            raise


def _attempt_run(
    runtime: ActiveRuntime,
    root: Path,
    source_trace_dir: str,
    matrix_item: TargetCampaignRun,
    state_item: dict[str, Any],
    *,
    deadline: float,
) -> bool:
    attempt_number = len(state_item["attempts"]) + 1
    run_id = f"{matrix_item.key}_attempt_{attempt_number:02d}"
    output_dir = root / "runs" / run_id
    launcher_logs = root / "launcher_logs"
    launcher_logs.mkdir(exist_ok=True)
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.run_targeted_predictor_profile",
        "--config",
        str(runtime.config_path),
        "--source-trace-dir",
        source_trace_dir,
        "--source-trace",
        matrix_item.source_trace_filename,
        "--source-qps",
        str(matrix_item.source_qps),
        "--source-seed",
        str(matrix_item.source_seed),
        "--run-id",
        run_id,
        "--recipe-seed",
        str(matrix_item.recipe_seed),
        "--request-count",
        str(TARGET_REQUEST_COUNT),
        "--request-timeout",
        str(TARGET_REQUEST_TIMEOUT_SECONDS),
        "--run-timeout",
        str(TARGET_RUN_TIMEOUT_SECONDS),
    ]
    record: dict[str, Any] = {
        "attempt": attempt_number,
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "status": "running",
    }
    state_item["attempts"].append(record)
    state_item["status"] = "running"
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        record["status"] = "failed"
        record["error"] = "target campaign timeout expired"
        return False

    _append_log(root, f"starting {run_id}")
    try:
        exit_code = _run_logged_command(
            command,
            cwd=runtime.workspace,
            log_path=launcher_logs / f"{run_id}.log",
            timeout=min(TARGET_RUN_TIMEOUT_SECONDS + 120.0, remaining),
        )
        record["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"target profile runner exited with {exit_code}")
        validation = validate_target_run_directory(
            output_dir,
            expected_run=matrix_item,
            expected_request_count=TARGET_REQUEST_COUNT,
        )
        record["status"] = "valid"
        record["validation"] = validation
        state_item["status"] = "valid"
        state_item["valid_attempt"] = run_id
        _append_log(root, f"completed {run_id}")
        return True
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        state_item["status"] = "failed"
        _append_log(root, f"failed {run_id}: {record['error']}")
        return False
    finally:
        record["finished_at_utc"] = _utc_now()


def validate_campaign(runtime: ActiveRuntime, root: Path) -> dict[str, Any]:
    state = _load_state(root)
    trace_dir = _source_trace_dir(runtime)
    if state.get("source_trace_dir") != trace_dir:
        raise ValueError("target campaign source trace directory changed")
    validations = []
    for matrix_item, state_item in zip(TARGET_CAMPAIGN_MATRIX, state["runs"]):
        run_id = state_item.get("valid_attempt")
        if not run_id:
            raise ValueError(f"target campaign item has no valid attempt: {matrix_item.key}")
        validations.append(
            validate_target_run_directory(
                root / "runs" / str(run_id),
                expected_run=matrix_item,
                expected_request_count=TARGET_REQUEST_COUNT,
            )
        )
    return {
        "schema_version": 1,
        "valid": True,
        "campaign_id": TARGET_CAMPAIGN_ID,
        "valid_runs": len(validations),
        "request_count_per_run": TARGET_REQUEST_COUNT,
        "total_requests": len(validations) * TARGET_REQUEST_COUNT,
        "target_rows": sum(int(item["target_count"]) for item in validations),
        "target_batch_kind_counts": {
            kind: sum(
                int(item["target_batch_kind_counts"].get(kind, 0))
                for item in validations
            )
            for kind in ("prefill_only", "mixed")
        },
    }


def worker(runtime: ActiveRuntime, *, resume: bool) -> int:
    require_frozen_for_execution(runtime)
    root = _campaign_root(runtime)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"target campaign does not exist: {root}")
        state = _load_state(root)
        if state.get("status") == "complete":
            validate_campaign(runtime, root)
            return 0
    else:
        if root.exists():
            raise FileExistsError(f"append-only target campaign exists: {root}")
        root.mkdir(parents=True)
        state = _initial_state()
        _save_state(root, state)

    source_trace_dir = _source_trace_dir(runtime)
    state["source_trace_dir"] = source_trace_dir
    _save_state(root, state)
    if not (root / MANIFEST_NAME).exists():
        _atomic_json(
            root / MANIFEST_NAME,
            {
                "schema_version": 1,
                "campaign_id": TARGET_CAMPAIGN_ID,
                "source_campaign_id": STOCK_CAMPAIGN_ID,
                "source_trace_dir": source_trace_dir,
                "request_count_per_run": TARGET_REQUEST_COUNT,
                "matrix": [
                    {
                        "key": item.key,
                        "source_qps": item.source_qps,
                        "source_seed": item.source_seed,
                        "recipe_seed": item.recipe_seed,
                        "target_recipe_count": len(
                            build_target_recipes(item.recipe_seed)
                        ),
                    }
                    for item in TARGET_CAMPAIGN_MATRIX
                ],
                "git": {
                    "root": _git_state(runtime.workspace),
                    "vllm": _git_state(runtime.workspace, "vllm"),
                },
                "created_at_utc": _utc_now(),
            },
        )

    state["status"] = "running"
    _save_state(root, state)
    deadline = time.monotonic() + TARGET_CAMPAIGN_TIMEOUT_SECONDS
    _append_log(root, "target campaign worker started" + (" (resume)" if resume else ""))
    for matrix_item, state_item in zip(TARGET_CAMPAIGN_MATRIX, state["runs"]):
        if state_item.get("status") == "valid":
            continue
        succeeded = False
        for _ in range(TARGET_MAX_ATTEMPTS):
            succeeded = _attempt_run(
                runtime,
                root,
                source_trace_dir,
                matrix_item,
                state_item,
                deadline=deadline,
            )
            _save_state(root, state)
            if succeeded or time.monotonic() >= deadline:
                break
        if not succeeded:
            state_item["status"] = "failed"
            _save_state(root, state)
        if time.monotonic() >= deadline:
            break

    try:
        validation = validate_campaign(runtime, root)
    except Exception as error:
        state["status"] = "complete_with_failures"
        state["validation_error"] = f"{type(error).__name__}: {error}"
        _save_state(root, state)
        _append_log(root, f"target campaign complete with failures: {error}")
        return 1
    state["status"] = "complete"
    state["validation"] = validation
    state["finished_at_utc"] = _utc_now()
    _save_state(root, state)
    _atomic_json(root / "campaign_validation.json", validation)
    _append_log(root, "target campaign completed and validated")
    return 0


def _print_preview(runtime: ActiveRuntime) -> None:
    payload = {
        "campaign_id": TARGET_CAMPAIGN_ID,
        "source_campaign_id": STOCK_CAMPAIGN_ID,
        "request_count_per_run": TARGET_REQUEST_COUNT,
        "runs": [
            {
                "key": item.key,
                "source_trace": item.source_trace_filename,
                "source_seed": item.source_seed,
                "recipe_seed": item.recipe_seed,
                "target_recipe_count": len(build_target_recipes(item.recipe_seed)),
            }
            for item in TARGET_CAMPAIGN_MATRIX
        ],
        "output": str(_campaign_root(runtime)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run_smoke(runtime: ActiveRuntime) -> int:
    require_frozen_for_execution(runtime)
    source_trace_dir = _source_trace_dir(runtime)
    smoke_root = resolve_under(
        runtime.raw_results, TARGET_SMOKE_CAMPAIGN_ID, label="target smoke"
    )
    if smoke_root.exists():
        raise FileExistsError(f"append-only target smoke exists: {smoke_root}")
    item = TARGET_CAMPAIGN_MATRIX[0]
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.run_targeted_predictor_profile",
        "--config",
        str(runtime.config_path),
        "--campaign-id",
        TARGET_SMOKE_CAMPAIGN_ID,
        "--source-trace-dir",
        source_trace_dir,
        "--source-trace",
        item.source_trace_filename,
        "--source-qps",
        str(item.source_qps),
        "--source-seed",
        str(item.source_seed),
        "--run-id",
        "targeted_smoke_seed_9001",
        "--recipe-seed",
        "9001",
        "--recipe-mode",
        "smoke",
        "--request-count",
        "10",
        "--request-timeout",
        "1800",
        "--run-timeout",
        "2400",
    ]
    return subprocess.run(command, cwd=runtime.workspace, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preview", "worker", "resume", "status", "validate", "smoke"),
    )
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    args = parser.parse_args()
    runtime = load_active_runtime(args.config)
    root = _campaign_root(runtime)
    if args.command == "preview":
        _print_preview(runtime)
        return 0
    if args.command == "smoke":
        return run_smoke(runtime)
    if args.command == "worker":
        return worker(runtime, resume=False)
    if args.command == "resume":
        return worker(runtime, resume=True)
    if args.command == "status":
        if not root.exists():
            print(json.dumps({"campaign_id": TARGET_CAMPAIGN_ID, "status": "not_started"}))
            return 0
        print(json.dumps(_load_state(root), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    validation = validate_campaign(runtime, root)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
