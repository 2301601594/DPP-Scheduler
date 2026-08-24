#!/usr/bin/env python3
"""Run the smoke-gated six-run Stock-versus-DPP development campaign."""

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

import yaml

from benchmarks.qwen3_runtime import (
    ActiveRuntime,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
)
from benchmarks.run_stock_natural_eos import _atomic_json, _git_state
from benchmarks.scheduler_comparison import (
    ComparisonRun,
    ComparisonSettings,
    build_report,
    load_comparison_settings,
    qps_text,
    validate_pair,
    validate_run_directory,
    validate_trace_directory,
)


CHECKPOINT_NAME = "campaign_checkpoint.json"
MANIFEST_NAME = "campaign_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_root(runtime: ActiveRuntime, settings: ComparisonSettings) -> Path:
    return resolve_under(
        runtime.raw_results, settings.campaign_id, label="scheduler comparison campaign"
    )


def _append_log(root: Path, message: str) -> None:
    rendered = f"[{_utc_now()}] {message}"
    print(rendered, flush=True)
    with (root / "campaign.log").open("a", encoding="utf-8") as stream:
        stream.write(rendered + "\n")


def _state_item(item: ComparisonRun) -> dict[str, Any]:
    return {
        "key": item.key,
        "pair_key": item.pair_key,
        "policy": item.policy,
        "qps": item.qps,
        "seed": item.seed,
        "status": "pending",
        "valid_attempt": None,
        "attempts": [],
    }


def _initial_state(settings: ComparisonSettings) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": settings.campaign_id,
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "status": "running",
        "request_count_per_main_run": settings.request_count,
        "smoke_request_count": settings.smoke_request_count,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "trace_dir": None,
        "trace_attempts": [],
        "smoke_gate": {
            "status": "pending",
            "passed_at_utc": None,
            "validation": None,
            "runs": [_state_item(item) for item in settings.smoke_matrix],
        },
        "runs": [_state_item(item) for item in settings.matrix],
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(root / CHECKPOINT_NAME, state)


def _expected_identity(items: tuple[ComparisonRun, ...]) -> list[tuple[Any, ...]]:
    return [(item.key, item.policy, item.qps, item.seed) for item in items]


def _observed_identity(items: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            item.get("key"),
            item.get("policy"),
            float(item.get("qps")),
            int(item.get("seed")),
        )
        for item in items
    ]


def _load_state(root: Path, settings: ComparisonSettings) -> dict[str, Any]:
    with (root / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("campaign_id") != settings.campaign_id:
        raise ValueError("comparison checkpoint campaign identity mismatch")
    if state.get("scope") != "development_nonformal_single_seed":
        raise ValueError("comparison checkpoint scope mismatch")
    if state.get("formal_benchmark_eligible") is not False:
        raise ValueError("comparison checkpoint became formal eligible")
    if int(state.get("request_count_per_main_run", 0)) != settings.request_count:
        raise ValueError("comparison checkpoint request count mismatch")
    smoke_gate = state.get("smoke_gate")
    if not isinstance(smoke_gate, dict):
        raise ValueError("comparison checkpoint smoke gate is missing")
    if _observed_identity(smoke_gate.get("runs", [])) != _expected_identity(
        settings.smoke_matrix
    ):
        raise ValueError("comparison checkpoint smoke matrix mismatch")
    if _observed_identity(state.get("runs", [])) != _expected_identity(settings.matrix):
        raise ValueError("comparison checkpoint main matrix mismatch")
    return state


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
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            raise


def _ensure_traces(
    runtime: ActiveRuntime,
    settings: ComparisonSettings,
    root: Path,
    state: dict[str, Any],
    *,
    deadline: float,
) -> str:
    if state.get("trace_dir"):
        trace_name = str(state["trace_dir"])
        validate_trace_directory(runtime, root / trace_name, settings)
        return trace_name

    attempt_number = len(state["trace_attempts"]) + 1
    trace_name = f"traces_attempt_{attempt_number:02d}"
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.generate_qwen3_poisson_traces",
        "--config",
        str(runtime.config_path),
        "--output-dir",
        f"{settings.campaign_id}/{trace_name}",
        "--num-requests",
        str(settings.request_count),
    ]
    for qps in settings.qps_values:
        command.extend(["--qps-seed", f"{qps_text(qps)}:{settings.seed}"])

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
    try:
        exit_code = _run_logged_command(
            command,
            cwd=runtime.workspace,
            log_path=root / f"trace_generation_attempt_{attempt_number:02d}.log",
            timeout=min(600.0, remaining),
        )
        record["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"trace generator exited with {exit_code}")
        validation = validate_trace_directory(runtime, root / trace_name, settings)
        record["status"] = "valid"
        record["validation"] = validation
        state["trace_dir"] = trace_name
        return trace_name
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at_utc"] = _utc_now()
        _save_state(root, state)


def _attempt_run(
    runtime: ActiveRuntime,
    settings: ComparisonSettings,
    root: Path,
    trace_dir: str,
    item: ComparisonRun,
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
        settings.campaign_id,
        "--development-trace-dir",
        f"{settings.campaign_id}/{trace_dir}",
        "--trace",
        item.trace_filename,
        "--trace-manifest",
        "manifest.json",
        "--run-id",
        run_id,
        "--policy",
        item.policy,
        "--request-timeout",
        str(settings.request_timeout_seconds),
    ]
    if smoke:
        command.extend(["--request-limit", str(settings.smoke_request_count)])

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
            timeout=min(settings.run_timeout_seconds, remaining),
        )
        record["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"natural-EOS runner exited with {exit_code}")
        validation = validate_run_directory(
            output_dir,
            expected_run=item,
            expected_request_count=(
                settings.smoke_request_count if smoke else settings.request_count
            ),
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
    settings: ComparisonSettings,
    root: Path,
    trace_dir: str,
    item: ComparisonRun,
    state_item: dict[str, Any],
    *,
    smoke: bool,
    deadline: float,
) -> bool:
    if state_item.get("status") == "valid":
        return True
    for _ in range(settings.max_attempts_per_invocation):
        if _attempt_run(
            runtime,
            settings,
            root,
            trace_dir,
            item,
            state_item,
            smoke=smoke,
            deadline=deadline,
        ):
            return True
        if time.monotonic() >= deadline:
            break
    return False


def _valid_run_root(root: Path, state_item: dict[str, Any]) -> Path:
    run_id = state_item.get("valid_attempt")
    if not run_id:
        raise ValueError(f"campaign item has no valid attempt: {state_item.get('key')}")
    return root / "runs" / str(run_id)


def _pair_state_items(items: list[dict[str, Any]], qps: float) -> tuple[dict[str, Any], dict[str, Any]]:
    by_policy = {
        str(item["policy"]): item for item in items if float(item["qps"]) == qps
    }
    if set(by_policy) != {"stock", "dpp"}:
        raise ValueError(f"missing Stock/DPP pair at QPS {qps}")
    return by_policy["stock"], by_policy["dpp"]


def _slo_seconds(runtime: ActiveRuntime) -> tuple[float, float]:
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    slo = config.get("slo") if isinstance(config, dict) else None
    if not isinstance(slo, dict):
        raise ValueError("active config SLO section is missing")
    ttft = float(slo.get("ttft_seconds", 0))
    tbt = float(slo.get("tbt_seconds", 0))
    if ttft <= 0 or tbt <= 0:
        raise ValueError("active config SLO values must be positive")
    return ttft, tbt


def validate_campaign(
    runtime: ActiveRuntime, settings: ComparisonSettings, root: Path
) -> dict[str, Any]:
    state = _load_state(root, settings)
    trace_name = state.get("trace_dir")
    if not trace_name:
        raise ValueError("campaign has no valid trace directory")
    trace_validation = validate_trace_directory(runtime, root / str(trace_name), settings)

    smoke_gate = state["smoke_gate"]
    if smoke_gate.get("status") != "passed":
        raise ValueError("campaign smoke gate did not pass")
    smoke_runs = smoke_gate["runs"]
    smoke_stock, smoke_dpp = _pair_state_items(smoke_runs, settings.smoke_qps)
    smoke_validation = validate_pair(
        _valid_run_root(root, smoke_stock),
        _valid_run_root(root, smoke_dpp),
        smoke=True,
    )

    run_validations: list[dict[str, Any]] = []
    pair_validations: list[dict[str, Any]] = []
    pair_roots: list[tuple[float, Path, Path]] = []
    for item, state_item in zip(settings.matrix, state["runs"]):
        run_root = _valid_run_root(root, state_item)
        run_validations.append(
            validate_run_directory(
                run_root,
                expected_run=item,
                expected_request_count=settings.request_count,
                smoke=False,
            )
        )
    for qps in settings.qps_values:
        stock_item, dpp_item = _pair_state_items(state["runs"], qps)
        stock_root = _valid_run_root(root, stock_item)
        dpp_root = _valid_run_root(root, dpp_item)
        pair_validations.append(validate_pair(stock_root, dpp_root, smoke=False))
        pair_roots.append((qps, stock_root, dpp_root))

    ttft_slo, tbt_slo = _slo_seconds(runtime)
    report = build_report(
        pair_roots,
        ttft_slo_seconds=ttft_slo,
        tbt_slo_seconds=tbt_slo,
    )
    return {
        "schema_version": 1,
        "valid": True,
        "campaign_id": settings.campaign_id,
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "trace_validation": trace_validation,
        "smoke_gate": smoke_validation,
        "valid_main_runs": len(run_validations),
        "valid_pairs": len(pair_validations),
        "request_count_per_main_run": settings.request_count,
        "main_request_total": len(run_validations) * settings.request_count,
        "pair_validations": pair_validations,
        "report": report,
    }


def worker(
    runtime: ActiveRuntime,
    settings: ComparisonSettings,
    *,
    resume: bool,
    smoke_only: bool = False,
) -> int:
    require_frozen_for_execution(runtime)
    root = _campaign_root(runtime, settings)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"campaign does not exist: {root}")
        state = _load_state(root, settings)
        if state.get("status") == "complete":
            validate_campaign(runtime, settings, root)
            return 0
    else:
        if root.exists():
            raise FileExistsError(f"append-only campaign exists: {root}")
        root.mkdir(parents=True)
        state = _initial_state(settings)
        _save_state(root, state)
        _atomic_json(
            root / MANIFEST_NAME,
            {
                "schema_version": 1,
                "campaign_id": settings.campaign_id,
                "scope": "development_nonformal_single_seed",
                "formal_benchmark_eligible": False,
                "smoke_gate_required_before_main_runs": True,
                "smoke_matrix": [
                    {"policy": item.policy, "qps": item.qps, "seed": item.seed}
                    for item in settings.smoke_matrix
                ],
                "main_matrix": [
                    {"policy": item.policy, "qps": item.qps, "seed": item.seed}
                    for item in settings.matrix
                ],
                "request_count_per_main_run": settings.request_count,
                "timeouts_seconds": {
                    "request": settings.request_timeout_seconds,
                    "run": settings.run_timeout_seconds,
                    "campaign": settings.campaign_timeout_seconds,
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
    deadline = time.monotonic() + settings.campaign_timeout_seconds
    _append_log(root, "campaign worker started" + (" (resume)" if resume else ""))
    try:
        trace_dir = _ensure_traces(
            runtime, settings, root, state, deadline=deadline
        )

        smoke_gate = state["smoke_gate"]
        smoke_gate["status"] = "running"
        _save_state(root, state)
        for item, state_item in zip(settings.smoke_matrix, smoke_gate["runs"]):
            if not _run_with_attempt_bound(
                runtime,
                settings,
                root,
                trace_dir,
                item,
                state_item,
                smoke=True,
                deadline=deadline,
            ):
                smoke_gate["status"] = "failed"
                state["status"] = "smoke_failed"
                _save_state(root, state)
                _append_log(root, "smoke gate failed; main six-run matrix was not started")
                return 1
            _save_state(root, state)

        smoke_stock, smoke_dpp = _pair_state_items(
            smoke_gate["runs"], settings.smoke_qps
        )
        smoke_validation = validate_pair(
            _valid_run_root(root, smoke_stock),
            _valid_run_root(root, smoke_dpp),
            smoke=True,
        )
        smoke_gate["status"] = "passed"
        smoke_gate["passed_at_utc"] = _utc_now()
        smoke_gate["validation"] = smoke_validation
        _save_state(root, state)
        if smoke_only:
            state["status"] = "smoke_passed"
            _save_state(root, state)
            _append_log(
                root,
                "smoke gate passed; stopped before the main six-run matrix as requested",
            )
            return 0
        _append_log(root, "smoke gate passed; starting the main six-run matrix")

        for item, state_item in zip(settings.matrix, state["runs"]):
            if not _run_with_attempt_bound(
                runtime,
                settings,
                root,
                trace_dir,
                item,
                state_item,
                smoke=False,
                deadline=deadline,
            ):
                state_item["status"] = "failed"
            _save_state(root, state)
            if time.monotonic() >= deadline:
                _append_log(root, "campaign timeout reached; remaining runs stay pending")
                break

        try:
            validation = validate_campaign(runtime, settings, root)
        except Exception as error:
            state["status"] = "complete_with_failures"
            state["validation_error"] = f"{type(error).__name__}: {error}"
            _save_state(root, state)
            _append_log(root, f"campaign complete with failures: {state['validation_error']}")
            return 1
        state["status"] = "complete"
        state["validation"] = {
            "valid": True,
            "valid_main_runs": validation["valid_main_runs"],
            "valid_pairs": validation["valid_pairs"],
        }
        state["finished_at_utc"] = _utc_now()
        _save_state(root, state)
        _atomic_json(root / "campaign_validation.json", validation)
        _atomic_json(root / "comparison_report.json", validation["report"])
        _append_log(root, "campaign complete and valid")
        return 0
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _save_state(root, state)
        _append_log(root, f"campaign worker failed: {state['error']}")
        return 1


def preview(runtime: ActiveRuntime, settings: ComparisonSettings) -> dict[str, Any]:
    return {
        "campaign_id": settings.campaign_id,
        "tmux_session": settings.tmux_session,
        "output_dir": str(_campaign_root(runtime, settings)),
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "execution": "sequential_detached_tmux",
        "smoke_gate": {
            "required_before_main_runs": True,
            "request_count_per_policy": settings.smoke_request_count,
            "matrix": [
                {"policy": item.policy, "qps": item.qps, "seed": item.seed}
                for item in settings.smoke_matrix
            ],
        },
        "main_run_count": len(settings.matrix),
        "request_count_per_main_run": settings.request_count,
        "main_request_total": len(settings.matrix) * settings.request_count,
        "matrix": [
            {"policy": item.policy, "qps": item.qps, "seed": item.seed}
            for item in settings.matrix
        ],
        "timeouts_seconds": {
            "request": settings.request_timeout_seconds,
            "run": settings.run_timeout_seconds,
            "campaign": settings.campaign_timeout_seconds,
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
    settings = load_comparison_settings(runtime)
    root = _campaign_root(runtime, settings)

    if args.command == "preview":
        print(json.dumps(preview(runtime, settings), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        return worker(runtime, settings, resume=False)
    if args.command == "smoke":
        return worker(runtime, settings, resume=False, smoke_only=True)
    if args.command == "resume-smoke":
        return worker(runtime, settings, resume=True, smoke_only=True)
    if args.command == "resume":
        return worker(runtime, settings, resume=True)
    if args.command == "status":
        if not root.exists():
            print(json.dumps({"campaign_id": settings.campaign_id, "status": "not_started"}))
            return 0
        print(json.dumps(_load_state(root, settings), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    validation = validate_campaign(runtime, settings, root)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
