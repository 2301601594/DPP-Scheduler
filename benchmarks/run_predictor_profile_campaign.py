#!/usr/bin/env python3
"""Orchestrate the fixed 12-run Predictor profiling campaign."""

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

from benchmarks.predictor_profile import (
    CAMPAIGN_ID,
    CAMPAIGN_MATRIX,
    CAMPAIGN_TIMEOUT_SECONDS,
    FORMAL_REQUEST_COUNT,
    MAX_ATTEMPTS_PER_INVOCATION,
    REQUEST_TIMEOUT_SECONDS,
    RUN_TIMEOUT_SECONDS,
    CampaignRun,
    qps_text,
    validate_run_directory,
)
from benchmarks.qwen3_runtime import (
    ActiveRuntime,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
)
from benchmarks.run_stock_natural_eos import (
    _atomic_json,
    _git_state,
    load_trace,
    verify_trace_manifest,
)


CHECKPOINT_NAME = "campaign_checkpoint.json"
MANIFEST_NAME = "campaign_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_root(runtime: ActiveRuntime) -> Path:
    return resolve_under(runtime.raw_results, CAMPAIGN_ID, label="profile campaign")


def _append_log(root: Path, message: str) -> None:
    rendered = f"[{_utc_now()}] {message}"
    print(rendered, flush=True)
    with (root / "campaign.log").open("a", encoding="utf-8") as stream:
        stream.write(rendered + "\n")


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "running",
        "request_count_per_run": FORMAL_REQUEST_COUNT,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "trace_dir": None,
        "trace_attempts": [],
        "runs": [
            {
                "key": item.key,
                "qps": item.qps,
                "seed": item.seed,
                "status": "pending",
                "valid_attempt": None,
                "attempts": [],
            }
            for item in CAMPAIGN_MATRIX
        ],
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(root / CHECKPOINT_NAME, state)


def _load_state(root: Path) -> dict[str, Any]:
    with (root / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("campaign checkpoint identity mismatch")
    expected = [(item.key, item.qps, item.seed) for item in CAMPAIGN_MATRIX]
    observed = [
        (item.get("key"), float(item.get("qps")), int(item.get("seed")))
        for item in state.get("runs", [])
    ]
    if observed != expected:
        raise ValueError("campaign checkpoint matrix mismatch")
    if int(state.get("request_count_per_run", 0)) != FORMAL_REQUEST_COUNT:
        raise ValueError("campaign checkpoint request count mismatch")
    return state


def _validate_trace_directory(
    runtime: ActiveRuntime, trace_dir: Path
) -> dict[str, Any]:
    manifest_path = trace_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("pairing") != "explicit":
        raise ValueError("campaign trace manifest is not explicitly paired")
    if int(manifest.get("num_requests_per_trace", 0)) != FORMAL_REQUEST_COUNT:
        raise ValueError("campaign trace request count mismatch")
    expected = {
        (item.trace_filename, item.qps, item.seed) for item in CAMPAIGN_MATRIX
    }
    observed = {
        (
            entry.get("file"),
            float(entry.get("requested_qps")),
            int(entry.get("seed")),
        )
        for entry in manifest.get("files", [])
    }
    if observed != expected or len(manifest.get("files", [])) != len(expected):
        raise ValueError("campaign trace matrix mismatch")
    for item in CAMPAIGN_MATRIX:
        trace_path = trace_dir / item.trace_filename
        rows = load_trace(trace_path, runtime)
        if len(rows) != FORMAL_REQUEST_COUNT:
            raise ValueError(f"trace row count mismatch: {trace_path}")
        verify_trace_manifest(trace_path, manifest_path, runtime)
    return {
        "valid": True,
        "trace_count": len(CAMPAIGN_MATRIX),
        "request_count_per_trace": FORMAL_REQUEST_COUNT,
    }


def _run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: float,
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


def _ensure_traces(
    runtime: ActiveRuntime,
    root: Path,
    state: dict[str, Any],
    *,
    deadline: float,
) -> str:
    if state.get("trace_dir"):
        trace_name = str(state["trace_dir"])
        _validate_trace_directory(runtime, resolve_under(root, trace_name, label="traces"))
        return trace_name

    attempt_number = len(state["trace_attempts"]) + 1
    trace_name = f"traces_attempt_{attempt_number:02d}"
    output_relative = f"{CAMPAIGN_ID}/{trace_name}"
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.generate_qwen3_poisson_traces",
        "--config",
        str(runtime.config_path),
        "--output-dir",
        output_relative,
        "--num-requests",
        str(FORMAL_REQUEST_COUNT),
    ]
    for item in CAMPAIGN_MATRIX:
        command.extend(["--qps-seed", f"{qps_text(item.qps)}:{item.seed}"])

    record: dict[str, Any] = {
        "attempt": attempt_number,
        "trace_dir": trace_name,
        "started_at_utc": _utc_now(),
        "status": "running",
    }
    state["trace_attempts"].append(record)
    _save_state(root, state)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("campaign timeout expired before trace generation")
    log_path = root / f"trace_generation_attempt_{attempt_number:02d}.log"
    try:
        exit_code = _run_logged_command(
            command,
            cwd=runtime.workspace,
            log_path=log_path,
            timeout=min(600.0, remaining),
        )
        record["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"trace generator exited with {exit_code}")
        validation = _validate_trace_directory(runtime, root / trace_name)
        record["status"] = "valid"
        record["validation"] = validation
        state["trace_dir"] = trace_name
        _save_state(root, state)
        return trace_name
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        _save_state(root, state)
        raise
    finally:
        record["finished_at_utc"] = _utc_now()
        _save_state(root, state)


def _attempt_run(
    runtime: ActiveRuntime,
    root: Path,
    trace_dir: str,
    matrix_item: CampaignRun,
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
        "benchmarks.run_predictor_profile",
        "--config",
        str(runtime.config_path),
        "--campaign-id",
        CAMPAIGN_ID,
        "--trace-dir",
        trace_dir,
        "--trace",
        matrix_item.trace_filename,
        "--run-id",
        run_id,
        "--qps",
        qps_text(matrix_item.qps),
        "--seed",
        str(matrix_item.seed),
        "--request-timeout",
        str(REQUEST_TIMEOUT_SECONDS),
        "--run-timeout",
        str(RUN_TIMEOUT_SECONDS),
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
        record["error"] = "campaign timeout expired"
        return False

    _append_log(root, f"starting {run_id}")
    try:
        exit_code = _run_logged_command(
            command,
            cwd=runtime.workspace,
            log_path=launcher_logs / f"{run_id}.log",
            timeout=min(RUN_TIMEOUT_SECONDS + 120.0, remaining),
        )
        record["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"profile runner exited with {exit_code}")
        validation = validate_run_directory(
            output_dir,
            expected_run=matrix_item,
            expected_request_count=FORMAL_REQUEST_COUNT,
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
    trace_name = state.get("trace_dir")
    if not trace_name:
        raise ValueError("campaign has no valid trace directory")
    trace_validation = _validate_trace_directory(runtime, root / str(trace_name))
    run_validations: list[dict[str, Any]] = []
    for matrix_item, state_item in zip(CAMPAIGN_MATRIX, state["runs"]):
        run_id = state_item.get("valid_attempt")
        if not run_id:
            raise ValueError(f"campaign item has no valid attempt: {matrix_item.key}")
        run_validations.append(
            validate_run_directory(
                root / "runs" / str(run_id),
                expected_run=matrix_item,
                expected_request_count=FORMAL_REQUEST_COUNT,
            )
        )
    return {
        "schema_version": 1,
        "valid": True,
        "campaign_id": CAMPAIGN_ID,
        "trace_validation": trace_validation,
        "valid_runs": len(run_validations),
        "request_count_per_run": FORMAL_REQUEST_COUNT,
        "total_requests": len(run_validations) * FORMAL_REQUEST_COUNT,
    }


def worker(runtime: ActiveRuntime, *, resume: bool) -> int:
    require_frozen_for_execution(runtime)
    root = _campaign_root(runtime)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"campaign does not exist: {root}")
        state = _load_state(root)
        if state.get("status") == "complete":
            validate_campaign(runtime, root)
            return 0
    else:
        if root.exists():
            raise FileExistsError(f"append-only campaign exists: {root}")
        root.mkdir(parents=True)
        state = _initial_state()
        _save_state(root, state)
        _atomic_json(
            root / MANIFEST_NAME,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "request_count_per_run": FORMAL_REQUEST_COUNT,
                "matrix": [
                    {"qps": item.qps, "seed": item.seed, "key": item.key}
                    for item in CAMPAIGN_MATRIX
                ],
                "timeouts_seconds": {
                    "request": REQUEST_TIMEOUT_SECONDS,
                    "run": RUN_TIMEOUT_SECONDS,
                    "campaign": CAMPAIGN_TIMEOUT_SECONDS,
                },
                "git": {
                    "root": _git_state(runtime.workspace),
                    "vllm": _git_state(runtime.workspace, "vllm"),
                },
                "created_at_utc": _utc_now(),
            },
        )

    state["status"] = "running"
    _save_state(root, state)
    deadline = time.monotonic() + CAMPAIGN_TIMEOUT_SECONDS
    _append_log(root, "campaign worker started" + (" (resume)" if resume else ""))
    try:
        trace_dir = _ensure_traces(runtime, root, state, deadline=deadline)
        for matrix_item, state_item in zip(CAMPAIGN_MATRIX, state["runs"]):
            if state_item.get("status") == "valid":
                continue
            succeeded = False
            for _ in range(MAX_ATTEMPTS_PER_INVOCATION):
                succeeded = _attempt_run(
                    runtime,
                    root,
                    trace_dir,
                    matrix_item,
                    state_item,
                    deadline=deadline,
                )
                _save_state(root, state)
                if succeeded:
                    break
                if time.monotonic() >= deadline:
                    break
            if not succeeded:
                state_item["status"] = "failed"
                _save_state(root, state)
            if time.monotonic() >= deadline:
                _append_log(root, "campaign timeout reached; remaining items stay pending")
                break

        try:
            validation = validate_campaign(runtime, root)
        except Exception as error:
            state["status"] = "complete_with_failures"
            state["validation_error"] = f"{type(error).__name__}: {error}"
            _save_state(root, state)
            _append_log(root, f"campaign complete with failures: {state['validation_error']}")
            return 1
        state["status"] = "complete"
        state["validation"] = validation
        state["finished_at_utc"] = _utc_now()
        _save_state(root, state)
        _atomic_json(root / "campaign_validation.json", validation)
        _append_log(root, "campaign complete and valid")
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _save_state(root, state)
        _append_log(root, f"campaign worker failed: {state['error']}")
        return 1


def preview(runtime: ActiveRuntime) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "output_dir": str(_campaign_root(runtime)),
        "execution": "sequential",
        "run_count": len(CAMPAIGN_MATRIX),
        "request_count_per_run": FORMAL_REQUEST_COUNT,
        "total_requests": len(CAMPAIGN_MATRIX) * FORMAL_REQUEST_COUNT,
        "matrix": [{"qps": item.qps, "seed": item.seed} for item in CAMPAIGN_MATRIX],
        "timeouts_seconds": {
            "request": REQUEST_TIMEOUT_SECONDS,
            "run": RUN_TIMEOUT_SECONDS,
            "campaign": CAMPAIGN_TIMEOUT_SECONDS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "worker", "resume", "status", "validate"))
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    args = parser.parse_args()
    runtime = load_active_runtime(args.config)
    root = _campaign_root(runtime)

    if args.command == "preview":
        print(json.dumps(preview(runtime), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        return worker(runtime, resume=False)
    if args.command == "resume":
        return worker(runtime, resume=True)
    if args.command == "status":
        if not root.exists():
            print(json.dumps({"campaign_id": CAMPAIGN_ID, "status": "not_started"}))
            return 0
        print(json.dumps(_load_state(root), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        result = validate_campaign(runtime, root)
    except Exception as error:
        print(
            json.dumps(
                {"valid": False, "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
