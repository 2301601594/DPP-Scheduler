#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from benchmarks.dppbench.aggregate import aggregate_g1, aggregate_g2
from benchmarks.dppbench.config import load_config, workspace_path
from benchmarks.dppbench.matrix import (
    g1_initial_specs,
    g1_scenarios,
    g2_condition_scenarios,
    g2_measurement_requests,
    g2_seeds,
    planned_request_count,
    preflight_specs,
)
from benchmarks.dppbench.results import load_processed
from benchmarks.dppbench.runner import server_is_healthy
from benchmarks.dppbench.traces import verify_manifest
from benchmarks.run_matrix import execute_g1, execute_g2, run_specs


@contextmanager
def execution_lock(config: dict[str, Any]) -> Iterator[None]:
    path = Path(config["paths"]["workspace"]) / ".cache/dppbench/g1_g2.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another G1-G2 benchmark holds the execution lock: {path}"
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(Path("/proc/self").resolve().name) + "\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _gpu_compute_processes() -> list[str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]


def _server_port_in_use(config: dict[str, Any]) -> bool:
    model = config["model"]
    try:
        with socket.create_connection(
            (str(model["host"]), int(model["port"])), timeout=2
        ):
            return True
    except OSError:
        return False
    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"nvidia-smi preflight failed: {message}")
    return [
        line.strip()
        for line in process.stdout.splitlines()
        if line.strip() and "no running processes" not in line.lower()
    ]


def execution_preflight(config: dict[str, Any]) -> dict[str, Any]:
    errors = verify_manifest(config)
    if errors:
        raise RuntimeError(f"trace manifest verification failed: {errors}")
    required_paths = {
        "python": Path(config["paths"]["python"]),
        "vllm_cli": Path(config["paths"]["vllm_cli"]),
        "model_snapshot": Path(config["paths"]["model_snapshot"]),
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"preflight paths are missing: {missing}")
    if _server_port_in_use(config) or server_is_healthy(config):
        model = config["model"]
        raise RuntimeError(
            f"refusing to start: {model['host']}:{model['port']} already serves "
            "a health endpoint"
        )
    compute_processes = _gpu_compute_processes()
    if compute_processes:
        raise RuntimeError(
            "refusing to benchmark while GPU compute processes are active: "
            + "; ".join(compute_processes)
        )
    return {
        "trace_manifest_valid": True,
        "server_port_available": True,
        "gpu_compute_processes": [],
    }


def dry_run_summary(config: dict[str, Any]) -> dict[str, Any]:
    seeds = len(config["statistics"]["seeds"])
    capacity_seeds = len(g2_seeds(config))
    scenarios = len(g1_scenarios(config))
    initial = g1_initial_specs(config)
    preflight = preflight_specs(config)
    serial_slo = config["g1"].get("slo_source", "low_load") == "serial"
    low_load_runs = 0 if serial_slo else scenarios * seeds
    low_load_requests = low_load_runs * int(
        config["statistics"]["low_load_measurement_requests"]
    )
    g1_nominal_runs = len(initial) + low_load_runs
    g1_nominal_requests = planned_request_count(config, initial) + low_load_requests
    start_attempts = config["g1"].get("low_load_start_attempt", {})
    adaptive_attempts = (
        0
        if serial_slo
        else sum(
            4 - int(start_attempts.get(scenario, 0))
            for scenario in g1_scenarios(config)
        )
    )
    g1_max_runs = len(initial) + adaptive_attempts * seeds
    g1_max_requests = planned_request_count(config, initial) + (
        adaptive_attempts
        * seeds
        * int(config["statistics"]["low_load_measurement_requests"])
    )

    conditions = len(g2_condition_scenarios(config))
    coarse_runs = conditions * len(config["arrivals"]["coarse_saturation_factors"])
    fine_runs = (
        conditions * int(config["arrivals"]["fine_points"]) * capacity_seeds
    )
    # After at least one valid coarse point, only one side of a bracket can be
    # missing, so each extension round adds at most one run per condition.
    extension_runs = conditions * int(
        config["arrivals"]["max_bracket_extensions"]
    )
    measurement = g2_measurement_requests(config)
    g2_nominal_runs = coarse_runs + fine_runs
    g2_max_runs = g2_nominal_runs + extension_runs

    g1 = load_processed(config, "g1")
    duration_note = (
        "G1 saturation is complete; exact G2 fine-scan duration remains pending "
        "the measured pass/fail brackets."
        if g1 and g1.get("gate_passed")
        else "Wall-clock duration is pending measured G1 saturation and adaptive "
        "G2 brackets; no synthetic timing estimate is reported."
    )
    return {
        "schema_version": 1,
        "pipeline": "g1_g2_stock_auto",
        "result_root": str(workspace_path(config, "raw_results")),
        "processed_root": str(workspace_path(config, "processed_results")),
        "preflight": {
            "runs": len(preflight),
            "measurement_requests": planned_request_count(config, preflight),
        },
        "g1": {
            "scenarios": list(g1_scenarios(config)),
            "slo_source": config["g1"].get("slo_source", "low_load"),
            "low_load_scheduled": not serial_slo,
            "low_load_requests_per_seed": int(
                config["statistics"]["low_load_measurement_requests"]
            ),
            "low_load_start_attempt": {
                scenario: int(start_attempts.get(scenario, 0))
                for scenario in g1_scenarios(config)
            },
            "nominal_runs": g1_nominal_runs,
            "nominal_measurement_requests": g1_nominal_requests,
            "maximum_adaptive_runs": g1_max_runs,
            "maximum_adaptive_measurement_requests": g1_max_requests,
        },
        "g2": {
            "seeds": list(g2_seeds(config)),
            "replication_status": (
                "exploratory_single_seed"
                if capacity_seeds == 1
                else "replicated"
            ),
            "conditions": [
                {
                    "scenario": scenario,
                    "arrival": arrival,
                    "burstiness": burstiness,
                }
                for scenario, arrival, burstiness in g2_condition_scenarios(config)
            ],
            "coarse_order": "descending_early_stop_per_condition",
            "minimum_runs_if_second_coarse_point_passes": conditions
            * (2 + int(config["arrivals"]["fine_points"])),
            "nominal_runs": g2_nominal_runs,
            "nominal_measurement_requests": g2_nominal_runs * measurement,
            "maximum_adaptive_runs": g2_max_runs,
            "maximum_adaptive_measurement_requests": g2_max_runs * measurement,
        },
        "total": {
            "nominal_runs_including_preflight": len(preflight)
            + g1_nominal_runs
            + g2_nominal_runs,
            "nominal_measurement_requests_including_preflight": planned_request_count(
                config, preflight
            )
            + g1_nominal_requests
            + g2_nominal_runs * measurement,
            "maximum_adaptive_runs_excluding_retries": len(preflight)
            + g1_max_runs
            + g2_max_runs,
            "maximum_adaptive_measurement_requests_excluding_retries": planned_request_count(
                config, preflight
            )
            + g1_max_requests
            + g2_max_runs * measurement,
        },
        "estimated_sending_duration_s": None,
        "duration_note": duration_note,
        "hard_run_attempts": int(config["validity"]["max_run_attempts"]),
    }


def rebuild_reports(config: dict[str, Any]) -> dict[str, Any]:
    g1 = aggregate_g1(config, allow_missing_low_load=True)
    if not g1.get("gate_passed"):
        return {"g1": g1, "g2": None, "gate_passed": False}
    g2 = aggregate_g2(config)
    return {"g1": g1, "g2": g2, "gate_passed": bool(g2.get("gate_passed"))}


def execute_pipeline(config: dict[str, Any], resume: bool) -> dict[str, Any]:
    with execution_lock(config):
        preflight = execution_preflight(config)
        run_specs(config, "preflight", preflight_specs(config), resume=True)
        execute_g1(config, resume=resume)
        execute_g2(config, resume=True)
        result = rebuild_reports(config)
        result["execution_preflight"] = preflight
        return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and report the frozen Stock-Auto G1-G2 benchmark"
    )
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume is only valid with --execute")
    config = load_config(args.config)
    if args.dry_run:
        result = dry_run_summary(config)
    elif args.report_only:
        errors = verify_manifest(config)
        if errors:
            raise RuntimeError(f"trace manifest verification failed: {errors}")
        result = rebuild_reports(config)
    else:
        result = execute_pipeline(config, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.dry_run or result.get("gate_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
