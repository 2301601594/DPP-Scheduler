#!/usr/bin/env python3
"""Orchestrate the isolated exact-batch Candidate knee campaign."""

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
from benchmarks.isolated_candidate_profile import validate_isolated_run_directory
from benchmarks.qwen3_runtime import (
    ActiveRuntime,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
)
from benchmarks.run_stock_natural_eos import _atomic_json, _git_state
from benchmarks.targeted_predictor_profile import (
    validate_reused_stock_trace,
)
from dpp_scheduler.targeted_profile import (
    ISOLATED_KNEE_CAMPAIGN_ID,
    ISOLATED_KNEE_SMOKE_CAMPAIGN_ID,
    ISOLATED_KNEE_TARGET_COUNT,
    KNEE_CAMPAIGN_MATRIX,
    KNEE_MAX_ATTEMPTS,
    KNEE_REQUEST_COUNT,
    TargetCampaignRun,
    build_target_recipes,
)


CHECKPOINT_NAME = "campaign_checkpoint.json"
MANIFEST_NAME = "campaign_manifest.json"
CAMPAIGN_TIMEOUT_SECONDS = 10 * 60 * 60
RUN_TIMEOUT_SECONDS = 8 * 60 * 60


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_root(runtime: ActiveRuntime) -> Path:
    return resolve_under(
        runtime.raw_results, ISOLATED_KNEE_CAMPAIGN_ID, label="knee campaign"
    )


def _source_root(runtime: ActiveRuntime) -> Path:
    return resolve_under(runtime.raw_results, STOCK_CAMPAIGN_ID, label="source campaign")


def _append_log(root: Path, message: str) -> None:
    rendered = f"[{_utc_now()}] {message}"
    print(rendered, flush=True)
    with (root / "campaign.log").open("a", encoding="utf-8") as stream:
        stream.write(rendered + "\n")


def _initial_state() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign_id": ISOLATED_KNEE_CAMPAIGN_ID,
        "status": "running",
        "target_batch_count_per_run": ISOLATED_KNEE_TARGET_COUNT,
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
            for item in KNEE_CAMPAIGN_MATRIX
        ],
    }


def _save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _atomic_json(root / CHECKPOINT_NAME, state)


def _load_state(root: Path) -> dict[str, Any]:
    with (root / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("campaign_id") != ISOLATED_KNEE_CAMPAIGN_ID:
        raise ValueError("knee campaign checkpoint identity mismatch")
    expected = [
        (item.key, item.source_qps, item.source_seed, item.recipe_seed)
        for item in KNEE_CAMPAIGN_MATRIX
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
        raise ValueError("knee campaign checkpoint matrix mismatch")
    if int(state.get("target_batch_count_per_run", 0)) != ISOLATED_KNEE_TARGET_COUNT:
        raise ValueError("knee campaign target count mismatch")
    return state


def _remaining_attempts(state_item: dict[str, Any]) -> int:
    return max(0, KNEE_MAX_ATTEMPTS - len(state_item.get("attempts", [])))


def _source_trace_info(runtime: ActiveRuntime) -> tuple[str, dict[str, Any]]:
    source = _source_root(runtime)
    with (source / CHECKPOINT_NAME).open("r", encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    if checkpoint.get("campaign_id") != STOCK_CAMPAIGN_ID:
        raise ValueError("source Stock campaign checkpoint identity mismatch")
    trace_dir = checkpoint.get("trace_dir")
    if not trace_dir:
        raise ValueError("source Stock campaign has no valid trace directory")
    item = KNEE_CAMPAIGN_MATRIX[0]
    trace_root = resolve_under(source, str(trace_dir), label="source trace directory")
    validation = validate_reused_stock_trace(
        trace_root / item.source_trace_filename,
        trace_root / "manifest.json",
        runtime,
        source_qps=item.source_qps,
        source_seed=item.source_seed,
        expected_request_count=KNEE_REQUEST_COUNT,
    )
    return str(trace_dir), validation


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
        "benchmarks.run_isolated_candidate_profile",
        "--config",
        str(runtime.config_path),
        "--campaign-id",
        ISOLATED_KNEE_CAMPAIGN_ID,
        "--source-campaign-id",
        STOCK_CAMPAIGN_ID,
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
        "--recipe-mode",
        "isolated_knee",
        "--batch-timeout",
        "600",
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
        record["error"] = "knee campaign timeout expired"
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
            raise RuntimeError(f"knee profile runner exited with {exit_code}")
        validation = validate_isolated_run_directory(
            output_dir,
            expected_run_id=run_id,
            recipe_seed=matrix_item.recipe_seed,
            recipe_mode="isolated_knee",
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
    trace_dir, _ = _source_trace_info(runtime)
    if state.get("source_trace_dir") != trace_dir:
        raise ValueError("knee campaign source trace directory changed")
    validations = []
    for matrix_item, state_item in zip(KNEE_CAMPAIGN_MATRIX, state["runs"]):
        run_id = state_item.get("valid_attempt")
        if not run_id:
            raise ValueError(f"knee campaign item has no valid attempt: {matrix_item.key}")
        validations.append(
            validate_isolated_run_directory(
                root / "runs" / str(run_id),
                expected_run_id=str(run_id),
                recipe_seed=matrix_item.recipe_seed,
                recipe_mode="isolated_knee",
            )
        )
    return {
        "schema_version": 2,
        "valid": True,
        "campaign_id": ISOLATED_KNEE_CAMPAIGN_ID,
        "valid_runs": len(validations),
        "target_batch_count_per_run": ISOLATED_KNEE_TARGET_COUNT,
        "target_rows": sum(int(item["target_count"]) for item in validations),
        "target_batch_kind_counts": {
            "prefill_only": sum(
                int(item["target_batch_kind_counts"].get("prefill_only", 0))
                for item in validations
            ),
            "mixed": sum(
                int(item["target_batch_kind_counts"].get("mixed", 0))
                for item in validations
            ),
        },
    }


def worker(runtime: ActiveRuntime, *, resume: bool) -> int:
    require_frozen_for_execution(runtime)
    root = _campaign_root(runtime)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"knee campaign does not exist: {root}")
        state = _load_state(root)
        if state.get("status") == "complete":
            validate_campaign(runtime, root)
            return 0
    else:
        if root.exists():
            raise FileExistsError(f"append-only knee campaign exists: {root}")
        root.mkdir(parents=True)
        state = _initial_state()
        _save_state(root, state)

    source_trace_dir, source_trace_compatibility = _source_trace_info(runtime)
    state["source_trace_dir"] = source_trace_dir
    _save_state(root, state)
    if not (root / MANIFEST_NAME).exists():
        _atomic_json(
            root / MANIFEST_NAME,
            {
                "schema_version": 2,
                "campaign_id": ISOLATED_KNEE_CAMPAIGN_ID,
                "source_campaign_id": STOCK_CAMPAIGN_ID,
                "source_trace_dir": source_trace_dir,
                "source_trace_compatibility": source_trace_compatibility,
                "target_batch_count_per_run": ISOLATED_KNEE_TARGET_COUNT,
                "isolation_protocol": (
                    "prepare_pause_exact_time_verify_abort_then_require_clean_baseline"
                ),
                "matrix": [
                    {
                        "key": item.key,
                        "source_qps": item.source_qps,
                        "source_seed": item.source_seed,
                        "recipe_seed": item.recipe_seed,
                        "target_recipe_count": len(
                            build_target_recipes(
                                item.recipe_seed, mode="isolated_knee"
                            )
                        ),
                    }
                    for item in KNEE_CAMPAIGN_MATRIX
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
    deadline = time.monotonic() + CAMPAIGN_TIMEOUT_SECONDS
    _append_log(root, "knee campaign worker started" + (" (resume)" if resume else ""))
    for matrix_item, state_item in zip(KNEE_CAMPAIGN_MATRIX, state["runs"]):
        if state_item.get("status") == "valid":
            continue
        succeeded = False
        remaining_attempts = _remaining_attempts(state_item)
        for _ in range(remaining_attempts):
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
        if not succeeded and state_item.get("status") != "valid":
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
        _append_log(root, f"knee campaign complete with failures: {error}")
        return 1
    state["status"] = "complete"
    state["validation"] = validation
    state["finished_at_utc"] = _utc_now()
    _save_state(root, state)
    _atomic_json(root / "campaign_validation.json", validation)
    _append_log(root, "knee campaign completed and validated")
    return 0


def _print_preview(runtime: ActiveRuntime) -> None:
    item = KNEE_CAMPAIGN_MATRIX[0]
    recipes = build_target_recipes(item.recipe_seed, mode="isolated_knee")
    payload = {
        "campaign_id": ISOLATED_KNEE_CAMPAIGN_ID,
        "source_campaign_id": STOCK_CAMPAIGN_ID,
        "target_batch_count_per_run": ISOLATED_KNEE_TARGET_COUNT,
        "constructed_request_count": sum(
            recipe.prefill_request_cap + recipe.decode_request_cap
            for recipe in recipes
        ),
        "prefill_only_target_count": sum(
            recipe.decode_request_cap == 0 for recipe in recipes
        ),
        "mixed_target_count": sum(
            recipe.decode_request_cap > 0 for recipe in recipes
        ),
        "prefill_caps": [256, 384, 512, 768, 1024, 1280, 1536, 2048],
        "decode_counts": [0, 8, 16, 32, 48],
        "prefill_request_counts": [1, 4, 8],
        "states": ["fresh", "partial"],
        "allocations": ["balanced", "skewed"],
        "repeats": 5,
        "sequence_budget_reason_for_48": (
            "64 Decode plus Prefill exceeds frozen C_seq=64"
        ),
        "token_budget_rule": (
            "omit mixed cells where Prefill tokens plus Decode tokens exceed C_tok=2048"
        ),
        "runs": [
            {
                "key": item.key,
                "source_trace": item.source_trace_filename,
                "source_qps": item.source_qps,
                "source_seed": item.source_seed,
                "recipe_seed": item.recipe_seed,
                "target_recipe_count": len(
                    recipes
                ),
            }
        ],
        "output": str(_campaign_root(runtime)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run_smoke(runtime: ActiveRuntime) -> int:
    require_frozen_for_execution(runtime)
    source_trace_dir, _ = _source_trace_info(runtime)
    smoke_root = resolve_under(
        runtime.raw_results, ISOLATED_KNEE_SMOKE_CAMPAIGN_ID, label="knee smoke"
    )
    if smoke_root.exists():
        raise FileExistsError(f"append-only knee smoke exists: {smoke_root}")
    item = KNEE_CAMPAIGN_MATRIX[0]
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.run_isolated_candidate_profile",
        "--config",
        str(runtime.config_path),
        "--campaign-id",
        ISOLATED_KNEE_SMOKE_CAMPAIGN_ID,
        "--source-campaign-id",
        STOCK_CAMPAIGN_ID,
        "--source-trace-dir",
        source_trace_dir,
        "--source-trace",
        item.source_trace_filename,
        "--source-qps",
        str(item.source_qps),
        "--source-seed",
        str(item.source_seed),
        "--run-id",
        "candidate_knee_isolated_smoke_seed_3001",
        "--recipe-seed",
        str(item.recipe_seed),
        "--recipe-mode",
        "isolated_knee_smoke",
        "--batch-timeout",
        "600",
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
            print(json.dumps({"campaign_id": ISOLATED_KNEE_CAMPAIGN_ID, "status": "not_started"}))
            return 0
        print(json.dumps(_load_state(root), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    validation = validate_campaign(runtime, root)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
