#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from benchmarks.dppbench.config import load_config
from benchmarks.dppbench.matrix import RunSpec, trace_for
from benchmarks.dppbench.runner import execute_run
from benchmarks.dppbench.traces import verify_manifest


def smoke_specs(config: dict) -> list[RunSpec]:
    trace = str(trace_for(config, "balanced"))
    specs = []
    for policy, budget in (("stock_auto", None), ("fixed_b8192", 8192)):
        specs.extend(
            [
                RunSpec(
                    stage="smoke",
                    scenario="balanced",
                    mode="serial",
                    policy=policy,
                    budget=budget,
                    seed=1,
                    trace_path=trace,
                    request_rate_rps=float("inf"),
                    max_concurrency=1,
                    request_limit=10,
                ),
                RunSpec(
                    stage="smoke",
                    scenario="balanced",
                    mode="open_loop",
                    policy=policy,
                    budget=budget,
                    seed=1,
                    trace_path=trace,
                    request_rate_rps=1.0,
                    burstiness=1.0,
                    request_limit=10,
                ),
            ]
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen four-run GPU smoke test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    config = load_config(args.config)
    errors = verify_manifest(config)
    if errors:
        raise RuntimeError(f"trace manifest verification failed: {errors}")
    specs = smoke_specs(config)
    if args.dry_run:
        print(json.dumps({"runs": [asdict(spec) for spec in specs], "requests": 40}, indent=2))
        return 0
    paths = [str(execute_run(config, spec)) for spec in specs]
    print(json.dumps({"runs": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
