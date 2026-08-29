#!/usr/bin/env python3
"""Smoke-gated Stage-1 ZERO-relative delta-N grid campaign.

One Stock run plus five DPP runs over the single staged n=150 development
trace (QPS 0.25, seed 1001). Each DPP run sets DPP_STAGE1_MAX_DELTA_N to one
grid value and enables Selector Diagnosis, whose replay must be zero-mismatch
for the run to validate. The campaign is development_nonformal and is not
formal-benchmark eligible.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.qwen3_runtime import (
    ActiveRuntime,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
    sha256_file,
)
from benchmarks.run_stock_natural_eos import _atomic_json, _git_state

CHECKPOINT_NAME = "campaign_checkpoint.json"
MANIFEST_NAME = "campaign_manifest.json"

CAMPAIGN_ID = "stage1_delta_n_grid_qps0p25_n150_seed1001_v1"
TRACE_CAMPAIGN_DIR = "dpp_ttft_weight_grid_qps0p25_n150_seed1001_v1"
TRACE_DIR = f"{TRACE_CAMPAIGN_DIR}/traces_attempt_01"
TRACE_FILE = "qps_0.25_seed_1001.jsonl"
TRACE_MANIFEST = "manifest.json"
TRACE_SHA256 = "203e7ed43522f71e44b7ee99a5cf3d5593f2e2d31215f010f67afd5ee2819e31"
TRACE_MANIFEST_SHA256 = (
    "c2ca59bd211059a5aab7105a315508d8d1d7df71c88f6eb077b645ceae214a33"
)
DELTA_N_VALUES = (0, 2, 4, 8, 16)
REQUEST_COUNT = 150
SMOKE_REQUEST_COUNT = 20
REQUEST_TIMEOUT_SECONDS = 3600
RUN_TIMEOUT_SECONDS = 5400
CAMPAIGN_TIMEOUT_SECONDS = 43200
MAX_ATTEMPTS_PER_INVOCATION = 2
DPP_STAGE1_MAX_DELTA_N_ENV = "DPP_STAGE1_MAX_DELTA_N"


@dataclass(frozen=True)
class GridRun:
    key: str
    policy: str
    delta_n: int | None


def _matrix() -> tuple[GridRun, ...]:
    return (
        GridRun("stock", "stock", None),
        *(GridRun(f"dpp_n{value}", "dpp", value) for value in DELTA_N_VALUES),
    )


def _smoke_matrix() -> tuple[GridRun, ...]:
    return (GridRun("stock", "stock", None), GridRun("dpp_n0", "dpp", 0))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_root(runtime: ActiveRuntime) -> Path:
    return resolve_under(
        runtime.raw_results, CAMPAIGN_ID, label="delta-N grid campaign"
    )


def _append_log(root: Path, message: str) -> None:
    rendered = f"[{_utc_now()}] {message}"
    print(rendered, flush=True)
    with (root / "campaign.log").open("a", encoding="utf-8") as stream:
        stream.write(rendered + "\n")


def _state_item(item: GridRun) -> dict[str, Any]:
    return {
        "key": item.key,
        "policy": item.policy,
        "delta_n": item.delta_n,
        "status": "pending",
        "valid_attempt": None,
        "attempts": [],
    }


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "status": "running",
        "request_count_per_main_run": REQUEST_COUNT,
        "smoke_request_count": SMOKE_REQUEST_COUNT,
        "delta_n_values": list(DELTA_N_VALUES),
        "trace_dir": TRACE_DIR,
        "trace_file": TRACE_FILE,
        "trace_sha256": TRACE_SHA256,
        "trace_manifest_sha256": TRACE_MANIFEST_SHA256,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "smoke_gate": {
            "status": "pending",
            "passed_at_utc": None,
            "validation": None,
            "runs": [_state_item(item) for item in _smoke_matrix()],
        },
        "runs": [_state_item(item) for item in _matrix()],
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(root / CHECKPOINT_NAME, state)


def _load_state(root: Path) -> dict[str, Any]:
    with (root / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("delta-N grid checkpoint campaign identity mismatch")
    if state.get("scope") != "development_nonformal_single_seed":
        raise ValueError("delta-N grid checkpoint scope mismatch")
    if state.get("formal_benchmark_eligible") is not False:
        raise ValueError("delta-N grid checkpoint became formal eligible")
    if tuple(state.get("delta_n_values", ())) != tuple(DELTA_N_VALUES):
        raise ValueError("delta-N grid checkpoint value set mismatch")
    if state.get("trace_sha256") != TRACE_SHA256:
        raise ValueError("delta-N grid checkpoint trace identity mismatch")
    if state.get("trace_manifest_sha256") != TRACE_MANIFEST_SHA256:
        raise ValueError("delta-N grid checkpoint trace manifest mismatch")
    if int(state.get("request_count_per_main_run", 0)) != REQUEST_COUNT:
        raise ValueError("delta-N grid checkpoint request count mismatch")
    return state


def _verify_staged_trace(runtime: ActiveRuntime, root: Path) -> dict[str, Any]:
    """Validate the fixed staged trace: existence, hashes, runner contract."""
    trace_root = resolve_under(
        runtime.raw_results, TRACE_DIR, label="staged grid trace directory"
    )
    trace_path = trace_root / TRACE_FILE
    manifest_path = trace_root / TRACE_MANIFEST
    if not trace_path.is_file():
        raise FileNotFoundError(f"staged grid trace is missing: {trace_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"staged grid trace manifest is missing: {manifest_path}")
    actual_trace = sha256_file(trace_path)
    actual_manifest = sha256_file(manifest_path)
    if actual_trace != TRACE_SHA256:
        raise ValueError(
            f"staged grid trace SHA-256 mismatch: expected {TRACE_SHA256}, "
            f"got {actual_trace}"
        )
    if actual_manifest != TRACE_MANIFEST_SHA256:
        raise ValueError(
            f"staged grid trace manifest SHA-256 mismatch: expected "
            f"{TRACE_MANIFEST_SHA256}, got {actual_manifest}"
        )
    return {
        "trace_path": str(trace_path),
        "trace_sha256": actual_trace,
        "manifest_sha256": actual_manifest,
    }


def _run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> int:
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
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


def _validate_grid_run_directory(
    run_root: Path,
    *,
    item: GridRun,
    expected_request_count: int,
    smoke: bool,
) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"run {manifest.get('run_id')} status is not complete: "
            f"{manifest.get('status')}"
        )
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"run {manifest.get('run_id')} summary is missing")
    if int(summary.get("failed", -1)) != 0:
        raise ValueError(
            f"run {manifest.get('run_id')} has request failures: "
            f"{summary.get('failed')}"
        )
    resolved = manifest.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError(f"run {manifest.get('run_id')} resolved preview missing")
    if int(resolved.get("request_count", -1)) != expected_request_count:
        raise ValueError(
            f"run {manifest.get('run_id')} request count mismatch"
        )
    if resolved.get("scheduler_policy") != item.policy:
        raise ValueError(f"run {manifest.get('run_id')} policy mismatch")
    if resolved.get("trace_sha256") != TRACE_SHA256:
        raise ValueError(f"run {manifest.get('run_id')} trace identity mismatch")
    overrides = resolved.get("runner_env_overrides")
    if not isinstance(overrides, dict):
        raise ValueError(f"run {manifest.get('run_id')} env overrides missing")
    recorded_n = overrides.get(DPP_STAGE1_MAX_DELTA_N_ENV)
    if item.delta_n is None:
        if recorded_n is not None:
            raise ValueError(
                f"Stock run {manifest.get('run_id')} must not set "
                f"{DPP_STAGE1_MAX_DELTA_N_ENV}"
            )
    else:
        if recorded_n != str(item.delta_n):
            raise ValueError(
                f"run {manifest.get('run_id')} delta-N override mismatch: "
                f"expected {item.delta_n}, got {recorded_n!r}"
            )
    if item.policy == "dpp":
        if manifest.get("selector_diagnosis_valid") is not True:
            raise ValueError(
                f"run {manifest.get('run_id')} selector diagnosis did not validate"
            )
        replay = manifest.get("selector_diagnosis_replay")
        if not isinstance(replay, dict) or any(
            value for key, value in replay.items() if key.endswith("_mismatch")
        ):
            raise ValueError(
                f"run {manifest.get('run_id')} selector diagnosis replay "
                "has mismatches"
            )
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "request_count": resolved.get("request_count"),
        "smoke": smoke,
        "delta_n": item.delta_n,
        "selector_diagnosis_valid": manifest.get("selector_diagnosis_valid"),
    }


def _attempt_run(
    runtime: ActiveRuntime,
    root: Path,
    item: GridRun,
    state_item: dict[str, Any],
    *,
    smoke: bool,
    deadline: float,
) -> bool:
    attempt_number = len(state_item["attempts"]) + 1
    kind = "smoke" if smoke else "main"
    run_id = f"{kind}_{item.key}_attempt_{attempt_number:02d}"
    output_dir = root / "runs" / run_id
    launcher_logs = root / "launcher_logs"
    launcher_logs.mkdir(exist_ok=True)
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.run_stock_natural_eos",
        "--config",
        str(runtime.config_path),
        "--campaign-id",
        CAMPAIGN_ID,
        "--development-trace-dir",
        TRACE_DIR,
        "--trace",
        TRACE_FILE,
        "--trace-manifest",
        TRACE_MANIFEST,
        "--run-id",
        run_id,
        "--policy",
        item.policy,
        "--request-timeout",
        str(REQUEST_TIMEOUT_SECONDS),
    ]
    run_env = dict(os.environ)
    run_env.pop(DPP_STAGE1_MAX_DELTA_N_ENV, None)
    if item.policy == "dpp":
        command.append("--dpp-selector-diagnosis")
        if item.delta_n is not None:
            run_env[DPP_STAGE1_MAX_DELTA_N_ENV] = str(item.delta_n)
    if smoke:
        command.extend(["--request-limit", str(SMOKE_REQUEST_COUNT)])

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
        state_item["status"] = "failed"
        return False

    _append_log(root, f"starting {run_id}")
    try:
        exit_code = _run_logged_command(
            command,
            cwd=runtime.workspace,
            log_path=launcher_logs / f"{run_id}.log",
            timeout=min(RUN_TIMEOUT_SECONDS, remaining),
            env=run_env,
        )
        record["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"natural-EOS runner exited with {exit_code}")
        validation = _validate_grid_run_directory(
            output_dir,
            item=item,
            expected_request_count=SMOKE_REQUEST_COUNT if smoke else REQUEST_COUNT,
            smoke=smoke,
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


def _run_with_attempt_bound(
    runtime: ActiveRuntime,
    root: Path,
    item: GridRun,
    state_item: dict[str, Any],
    *,
    smoke: bool,
    deadline: float,
) -> bool:
    if state_item.get("status") == "valid":
        return True
    for _ in range(MAX_ATTEMPTS_PER_INVOCATION):
        if _attempt_run(
            runtime, root, item, state_item, smoke=smoke, deadline=deadline
        ):
            return True
        if time.monotonic() >= deadline:
            break
    return False


def _valid_run_root(root: Path, state_item: dict[str, Any]) -> Path:
    run_id = state_item.get("valid_attempt")
    if not run_id:
        raise ValueError(f"grid item has no valid attempt: {state_item.get('key')}")
    return root / "runs" / str(run_id)


def validate_campaign(runtime: ActiveRuntime, root: Path) -> dict[str, Any]:
    state = _load_state(root)
    trace_validation = _verify_staged_trace(runtime, root)
    smoke_gate = state["smoke_gate"]
    if smoke_gate.get("status") != "passed":
        raise ValueError("delta-N grid smoke gate did not pass")
    smoke_validations = [
        _validate_grid_run_directory(
            _valid_run_root(root, state_item),
            item=item,
            expected_request_count=SMOKE_REQUEST_COUNT,
            smoke=True,
        )
        for item, state_item in zip(_smoke_matrix(), smoke_gate["runs"])
    ]
    run_validations = []
    for item, state_item in zip(_matrix(), state["runs"]):
        run_validations.append(
            _validate_grid_run_directory(
                _valid_run_root(root, state_item),
                item=item,
                expected_request_count=REQUEST_COUNT,
                smoke=False,
            )
        )
    return {
        "schema_version": 1,
        "valid": True,
        "campaign_id": CAMPAIGN_ID,
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "trace_validation": trace_validation,
        "smoke_validations": smoke_validations,
        "valid_main_runs": len(run_validations),
        "request_count_per_main_run": REQUEST_COUNT,
        "main_request_total": len(run_validations) * REQUEST_COUNT,
        "run_validations": run_validations,
    }


def worker(
    runtime: ActiveRuntime,
    *,
    resume: bool,
    smoke_only: bool = False,
) -> int:
    require_frozen_for_execution(runtime)
    root = _campaign_root(runtime)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"grid campaign does not exist: {root}")
        state = _load_state(root)
        if state.get("status") == "complete":
            validate_campaign(runtime, root)
            return 0
    else:
        if root.exists():
            raise FileExistsError(f"append-only grid campaign exists: {root}")
        root.mkdir(parents=True)
        state = _initial_state()
        _save_state(root, state)
        _atomic_json(
            root / MANIFEST_NAME,
            {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "scope": "development_nonformal_single_seed",
                "formal_benchmark_eligible": False,
                "smoke_gate_required_before_main_runs": True,
                "smoke_matrix": [
                    {"policy": item.policy, "delta_n": item.delta_n}
                    for item in _smoke_matrix()
                ],
                "main_matrix": [
                    {"policy": item.policy, "delta_n": item.delta_n}
                    for item in _matrix()
                ],
                "request_count_per_main_run": REQUEST_COUNT,
                "trace_dir": TRACE_DIR,
                "trace_file": TRACE_FILE,
                "trace_sha256": TRACE_SHA256,
                "trace_manifest_sha256": TRACE_MANIFEST_SHA256,
                "timeouts_seconds": {
                    "request": REQUEST_TIMEOUT_SECONDS,
                    "run": RUN_TIMEOUT_SECONDS,
                    "campaign": CAMPAIGN_TIMEOUT_SECONDS,
                },
                "config_sha256": runtime.config_sha256,
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
    _append_log(root, "grid campaign worker started" + (" (resume)" if resume else ""))
    try:
        trace_validation = _verify_staged_trace(runtime, root)
        _append_log(root, f"staged trace verified: {trace_validation['trace_sha256']}")

        smoke_gate = state["smoke_gate"]
        smoke_gate["status"] = "running"
        _save_state(root, state)
        for item, state_item in zip(_smoke_matrix(), smoke_gate["runs"]):
            if not _run_with_attempt_bound(
                runtime, root, item, state_item, smoke=True, deadline=deadline
            ):
                smoke_gate["status"] = "failed"
                state["status"] = "smoke_failed"
                _save_state(root, state)
                _append_log(root, "smoke gate failed; main matrix was not started")
                return 1
            _save_state(root, state)

        smoke_gate["status"] = "passed"
        smoke_gate["passed_at_utc"] = _utc_now()
        _save_state(root, state)
        if smoke_only:
            state["status"] = "smoke_passed"
            _save_state(root, state)
            _append_log(root, "smoke gate passed; stopped before the main matrix as requested")
            return 0
        _append_log(root, "smoke gate passed; starting the main grid matrix")

        for item, state_item in zip(_matrix(), state["runs"]):
            if not _run_with_attempt_bound(
                runtime, root, item, state_item, smoke=False, deadline=deadline
            ):
                state_item["status"] = "failed"
            _save_state(root, state)
            if time.monotonic() >= deadline:
                _append_log(root, "campaign timeout reached; remaining runs stay pending")
                break

        try:
            validation = validate_campaign(runtime, root)
        except Exception as error:
            state["status"] = "complete_with_failures"
            state["validation_error"] = f"{type(error).__name__}: {error}"
            _save_state(root, state)
            _append_log(root, f"grid campaign complete with failures: {state['validation_error']}")
            return 1
        state["status"] = "complete"
        state["validation"] = {
            "valid": True,
            "valid_main_runs": validation["valid_main_runs"],
        }
        state["finished_at_utc"] = _utc_now()
        _save_state(root, state)
        _atomic_json(root / "campaign_validation.json", validation)
        _append_log(root, "grid campaign complete and valid")
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _save_state(root, state)
        _append_log(root, f"grid campaign worker failed: {state['error']}")
        return 1


def preview(runtime: ActiveRuntime) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "output_dir": str(_campaign_root(runtime)),
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "execution": "sequential_detached_tmux",
        "trace_dir": TRACE_DIR,
        "trace_file": TRACE_FILE,
        "trace_sha256": TRACE_SHA256,
        "trace_manifest_sha256": TRACE_MANIFEST_SHA256,
        "smoke_gate": {
            "required_before_main_runs": True,
            "request_count_per_run": SMOKE_REQUEST_COUNT,
            "matrix": [
                {"policy": item.policy, "delta_n": item.delta_n}
                for item in _smoke_matrix()
            ],
        },
        "main_run_count": len(_matrix()),
        "request_count_per_main_run": REQUEST_COUNT,
        "main_request_total": len(_matrix()) * REQUEST_COUNT,
        "matrix": [
            {"policy": item.policy, "delta_n": item.delta_n}
            for item in _matrix()
        ],
        "dpp_selector_diagnosis": "enabled for all DPP runs",
        "timeouts_seconds": {
            "request": REQUEST_TIMEOUT_SECONDS,
            "run": RUN_TIMEOUT_SECONDS,
            "campaign": CAMPAIGN_TIMEOUT_SECONDS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preview",
            "smoke",
            "resume-smoke",
            "worker",
            "resume",
            "status",
            "validate",
        ),
    )
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    args = parser.parse_args()
    runtime = load_active_runtime(args.config)
    root = _campaign_root(runtime)

    if args.command == "preview":
        print(json.dumps(preview(runtime), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        return worker(runtime, resume=False)
    if args.command == "smoke":
        return worker(runtime, resume=False, smoke_only=True)
    if args.command == "resume-smoke":
        return worker(runtime, resume=True, smoke_only=True)
    if args.command == "resume":
        return worker(runtime, resume=True)
    if args.command == "status":
        if not root.exists():
            print(json.dumps({"campaign_id": CAMPAIGN_ID, "status": "not_started"}))
            return 0
        print(json.dumps(_load_state(root), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    validation = validate_campaign(runtime, root)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
