#!/usr/bin/env python3
"""Run the fresh-seed n=200 validation for the three-segment Mixed Predictor."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.build_predictor_dataset import extract_features
from benchmarks.predictor_online_evaluation import load_evaluation_rows
from benchmarks.qwen3_runtime import load_active_runtime, load_frozen_predictor, resolve_under
from benchmarks.run_predictor_online_evaluation import (
    SEGMENTED_EVALUATION_CAMPAIGN_ID,
    SEGMENTED_EVALUATION_SMOKE_CAMPAIGN_ID,
)
from benchmarks.run_stock_natural_eos import _atomic_json, _git_state
from dpp_scheduler.predictor import SEGMENTED_ONLINE_PREDICTOR_VERSION


SOURCE_QPS = 0.2
SOURCE_SEED = 3002
RECIPE_SEED = 5001
SMOKE_RECIPE_SEED = 5000
REQUEST_COUNT = 200
TRACE_DIR = "source_traces_seed_3002"
TRACE_FILE = "qps_0.2_seed_3002.jsonl"
RUN_ID = "mixed_decode_three_segment_n200_source_3002_recipe_5001"
SMOKE_RUN_ID = "mixed_decode_three_segment_smoke_source_3002_recipe_5000"
CHECKPOINT_NAME = "campaign_checkpoint.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root(runtime) -> Path:
    return resolve_under(
        runtime.raw_results,
        SEGMENTED_EVALUATION_CAMPAIGN_ID,
        label="segmented Predictor validation campaign",
    )


def _trace_path(runtime) -> Path:
    return _root(runtime) / TRACE_DIR / TRACE_FILE


def prepare(runtime) -> int:
    trace_root = _root(runtime) / TRACE_DIR
    if trace_root.exists():
        manifest = json.loads(
            (trace_root / "manifest.json").read_text(encoding="utf-8")
        )
        entries = manifest.get("files", [])
        if (
            len(entries) != 1
            or entries[0].get("file") != TRACE_FILE
            or int(entries[0].get("seed", -1)) != SOURCE_SEED
            or int(entries[0].get("num_requests", 0)) != REQUEST_COUNT
        ):
            raise ValueError("existing independent-validation trace identity mismatch")
        return 0
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.generate_qwen3_poisson_traces",
        "--config",
        str(runtime.config_path),
        "--output-dir",
        f"{SEGMENTED_EVALUATION_CAMPAIGN_ID}/{TRACE_DIR}",
        "--num-requests",
        str(REQUEST_COUNT),
        "--qps-seed",
        f"{SOURCE_QPS}:{SOURCE_SEED}",
    ]
    return subprocess.run(command, cwd=runtime.workspace, check=False).returncode


def _runner_command(runtime, *, smoke: bool, dry_run: bool = False) -> list[str]:
    command = [
        str(runtime.python),
        "-m",
        "benchmarks.run_predictor_online_evaluation",
        "--config",
        str(runtime.config_path),
        "--campaign-id",
        (
            SEGMENTED_EVALUATION_SMOKE_CAMPAIGN_ID
            if smoke
            else SEGMENTED_EVALUATION_CAMPAIGN_ID
        ),
        "--source-campaign-id",
        SEGMENTED_EVALUATION_CAMPAIGN_ID,
        "--source-trace-dir",
        TRACE_DIR,
        "--source-trace",
        TRACE_FILE,
        "--source-qps",
        str(SOURCE_QPS),
        "--source-seed",
        str(SOURCE_SEED),
        "--run-id",
        SMOKE_RUN_ID if smoke else RUN_ID,
        "--recipe-seed",
        str(SMOKE_RECIPE_SEED if smoke else RECIPE_SEED),
        "--recipe-mode",
        "smoke" if smoke else "formal",
        "--request-count",
        "10" if smoke else str(REQUEST_COUNT),
        "--request-timeout",
        "1800" if smoke else "7200",
        "--run-timeout",
        "2400" if smoke else "10800",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def preview(runtime) -> int:
    predictor = load_frozen_predictor(runtime)
    if predictor.predictor_version != SEGMENTED_ONLINE_PREDICTOR_VERSION:
        raise ValueError("active config does not select the segmented Predictor")
    print(
        json.dumps(
            {
                "campaign_id": SEGMENTED_EVALUATION_CAMPAIGN_ID,
                "output": str(_root(runtime)),
                "source_trace": str(_trace_path(runtime)),
                "source_qps": SOURCE_QPS,
                "source_seed": SOURCE_SEED,
                "recipe_seed": RECIPE_SEED,
                "request_count": REQUEST_COUNT,
                "predictor_version": predictor.predictor_version,
                "predictor_artifact_manifest_sha256": (
                    predictor.artifact_manifest_sha256
                ),
                "validation_role": "fresh_independent_data_not_used_for_design",
                "expected_duration": "about 20-40 minutes including startup",
                "gpu_count": 1,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if _trace_path(runtime).exists():
        return subprocess.run(
            _runner_command(runtime, smoke=False, dry_run=True),
            cwd=runtime.workspace,
            check=False,
        ).returncode
    return 0


def smoke(runtime) -> int:
    if prepare(runtime) != 0:
        return 1
    return subprocess.run(
        _runner_command(runtime, smoke=True), cwd=runtime.workspace, check=False
    ).returncode


def worker(runtime) -> int:
    if prepare(runtime) != 0:
        return 1
    root = _root(runtime)
    checkpoint = root / CHECKPOINT_NAME
    if checkpoint.exists():
        raise FileExistsError(f"append-only validation checkpoint exists: {checkpoint}")
    state = {
        "schema_version": 1,
        "campaign_id": SEGMENTED_EVALUATION_CAMPAIGN_ID,
        "run_id": RUN_ID,
        "status": "running",
        "source_seed": SOURCE_SEED,
        "recipe_seed": RECIPE_SEED,
        "request_count": REQUEST_COUNT,
        "started_at_utc": _now(),
        "git": {
            "root": _git_state(runtime.workspace),
            "vllm": _git_state(runtime.workspace, "vllm"),
        },
    }
    _atomic_json(checkpoint, state)
    result = subprocess.run(
        _runner_command(runtime, smoke=False), cwd=runtime.workspace, check=False
    )
    state["exit_code"] = result.returncode
    state["finished_at_utc"] = _now()
    state["status"] = "complete" if result.returncode == 0 else "failed"
    _atomic_json(checkpoint, state)
    return result.returncode


def validate(runtime) -> int:
    root = _root(runtime)
    checkpoint = json.loads((root / CHECKPOINT_NAME).read_text(encoding="utf-8"))
    if checkpoint.get("status") != "complete":
        raise ValueError("segmented Predictor validation campaign is incomplete")
    run_root = root / "runs" / RUN_ID
    summary = json.loads(
        (run_root / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    if not summary.get("valid") or int(summary.get("iteration_count", 0)) <= 0:
        raise ValueError("segmented Predictor evaluation summary is invalid")
    rows = load_evaluation_rows(
        run_root / "predictor_evaluation.jsonl",
        expected_run_id=RUN_ID,
        recipe_seed=RECIPE_SEED,
        recipe_mode="formal",
    )
    mixed_counts = sorted(
        {
            int(extract_features(row["selected_requests"])["x_4"])
            for row in rows
            if row.get("batch_kind") == "mixed"
        }
    )
    if not mixed_counts:
        raise ValueError("independent validation contains no Mixed rows")
    payload = {
        "valid": True,
        "campaign_id": SEGMENTED_EVALUATION_CAMPAIGN_ID,
        "run_id": RUN_ID,
        "fresh_source_seed": SOURCE_SEED,
        "fresh_recipe_seed": RECIPE_SEED,
        "mixed_decode_counts": mixed_counts,
        "evaluation": summary,
    }
    _atomic_json(root / "campaign_validation.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "preview", "smoke", "worker", "status", "validate")
    )
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    args = parser.parse_args()
    runtime = load_active_runtime(args.config)
    if args.command == "prepare":
        return prepare(runtime)
    if args.command == "preview":
        return preview(runtime)
    if args.command == "smoke":
        return smoke(runtime)
    if args.command == "worker":
        return worker(runtime)
    if args.command == "validate":
        return validate(runtime)
    checkpoint = _root(runtime) / CHECKPOINT_NAME
    print(
        checkpoint.read_text(encoding="utf-8")
        if checkpoint.exists()
        else json.dumps(
            {"campaign_id": SEGMENTED_EVALUATION_CAMPAIGN_ID, "status": "absent"}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
