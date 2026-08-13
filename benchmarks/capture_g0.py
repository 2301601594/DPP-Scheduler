#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.dppbench.config import config_hash, load_config, workspace_path
from benchmarks.dppbench.io import atomic_write_json, sha256_file
from benchmarks.dppbench.results import iter_run_records
from benchmarks.dppbench.runner import capture_environment, code_snapshot
from benchmarks.dppbench.traces import verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the G0 reproducibility manifest")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    errors = verify_manifest(config)
    if errors:
        raise RuntimeError(f"trace manifest verification failed: {errors}")

    latest: dict[str, dict] = {}
    for record in iter_run_records(config, "smoke"):
        metadata = record["metadata"]
        if metadata.get("status") != "complete" or not metadata.get("validity", {}).get("valid"):
            continue
        spec = metadata["run_spec"]
        key = f"{spec['policy']}:{spec['mode']}"
        latest[key] = {
            "run_id": metadata["run_id"],
            "resolved_max_num_batched_tokens": metadata.get("resolved_max_num_batched_tokens"),
            "kv_cache_capacity_tokens": metadata.get("kv_cache_capacity_tokens"),
            "scheduler_config": metadata.get("scheduler_config"),
        }
    required = {
        "stock_auto:serial",
        "stock_auto:open_loop",
        "fixed_b8192:serial",
        "fixed_b8192:open_loop",
    }
    missing = required - latest.keys()
    if missing:
        raise RuntimeError(f"G0 smoke evidence missing: {sorted(missing)}")

    manifest_path = workspace_path(config, "traces") / "manifest.json"
    payload = {
        "schema_version": 1,
        "stage": "g0",
        "status": "complete",
        "config_sha256": config_hash(config),
        "trace_manifest_sha256": sha256_file(manifest_path),
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "snapshot": config["paths"]["model_snapshot"],
        },
        "environment": capture_environment(config),
        "code_sha256": code_snapshot(config),
        "smoke_evidence": latest,
    }
    destination = Path(config["paths"]["workspace"]) / "configs/environment_manifest.json"
    atomic_write_json(destination, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
