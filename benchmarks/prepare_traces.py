#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from benchmarks.dppbench.config import load_config
from benchmarks.dppbench.traces import (
    materialize_phase_trace,
    prepare_traces,
    verify_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and freeze G0 traces")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--materialize-phase", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.verify:
        errors = verify_manifest(config)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 1 if errors else 0
    if args.materialize_phase:
        result = materialize_phase_trace(config, force=args.force)
    else:
        result = prepare_traces(
            config, download=not args.no_download, force=args.force
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

