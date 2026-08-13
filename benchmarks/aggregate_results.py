#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from benchmarks.dppbench.aggregate import aggregate_stage
from benchmarks.dppbench.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate append-only benchmark results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("g1", "g2", "g3", "all"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "all":
        result = {stage: aggregate_stage(config, stage) for stage in ("g1", "g2", "g3")}
        passed = all(value.get("gate_passed") for value in result.values())
    else:
        result = aggregate_stage(config, args.stage)
        passed = bool(result.get("gate_passed"))
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
