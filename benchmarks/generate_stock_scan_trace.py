#!/usr/bin/env python3
"""Generate a small Stock scan trace from the Stage-1 request pool.

The trace contains a fixed number of requests, explicit arrival times, a
safety output cap, and greedy generation. It does not contain a target output
length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-tokens-safety", type=int, default=16384)
    args = parser.parse_args()

    with open(args.request_pool, "r", encoding="utf-8") as stream:
        pool = [json.loads(line) for line in stream if line.strip()]
    print(f"Loaded request pool: {len(pool)}")

    rng = random.Random(args.seed)
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    selected_indices = indices[: args.num_requests]

    # Poisson arrivals (Gamma with shape 1), scaled so total duration is exact.
    delays = [rng.expovariate(args.request_rate) for _ in range(args.num_requests - 1)]
    arrivals = [0.0]
    for delay in delays:
        arrivals.append(arrivals[-1] + delay)
    if len(arrivals) > 1:
        target_duration = (args.num_requests - 1) / args.request_rate
        scale = target_duration / arrivals[-1]
        arrivals = [value * scale for value in arrivals]

    rows = []
    for offset, pool_index in enumerate(selected_indices):
        record = pool[pool_index]
        rows.append(
            {
                "request_id": f"scan-{offset:04d}",
                "prompt_id": record["prompt_id"],
                "prompt": record["prompt"],
                "input_tokens": record["input_tokens"],
                "arrival_time_s": round(arrivals[offset], 6),
                "max_tokens_safety": args.max_tokens_safety,
                "temperature": 0.0,
                "ignore_eos": False,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "num_requests": len(rows),
        "request_rate": args.request_rate,
        "seed": args.seed,
        "max_tokens_safety": args.max_tokens_safety,
        "trace_sha256": sha256_file(out),
        "generated_from_request_pool": str(Path(args.request_pool).resolve()),
    }
    meta_path = out.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Wrote {out} ({len(rows)} rows)")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
